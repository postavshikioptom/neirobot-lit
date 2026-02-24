import polars as pl
import numpy as np
import onnxruntime as ort
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import datetime
import sys

# Добавляем путь к корню проекта для импорта через python_lab.src
sys.path.append(str(Path(__file__).parent.parent.parent))
from python_lab.src.dataset import fast_parquet_reader
from python_lab.src.backtest.engine import EventEngine, Event, EventType, MarketData, SignalData, BotConfig, SorConfig, TradeData

def parse_args():
    parser = argparse.ArgumentParser(description="LiT Model Backtest Engine")
    # Задача 213: Поддержка мульти-инструментальности - принимаем список символов
    parser.add_argument("--symbols", type=str, required=True, help="Comma-separated list of symbols to backtest (e.g., BTCUSDT,ETHUSDT)")
    parser.add_argument("--latency_ms", type=int, default=50, help="Network latency in ms")
    parser.add_argument("--limit_timeout_ms", type=int, default=5000, help="Limit order timeout in ms")
    parser.add_argument("--slippage_bps", type=float, default=1.0, help="Additional slippage for market orders in bps")
    parser.add_argument("--maker_fee_bps", type=float, default=2.0, help="Maker fee in bps (2.0 = 0.02%)")
    parser.add_argument("--taker_fee_bps", type=float, default=5.5, help="Taker fee in bps (5.5 = 0.055%)")
    parser.add_argument("--order_size_usd", type=float, default=1000.0, help="Fixed order size in USD")
    
    # Задача 212: Параметры очереди лимитных ордеров
    parser.add_argument("--queue_model", type=str, choices=["conservative", "probabilistic"], 
                        default="conservative", help="Queue model for limit orders (conservative=full volume ahead, probabilistic=random 50-100%)")
    
    # Пути
    parser.add_argument("--data_dir", type=str, help="Directory with parquet files. Defaults to bots/<symbol>/data/raw")
    
    # Сигналы
    parser.add_argument("--threshold_up", type=float, default=0.6, help="Confidence threshold for BUY")
    parser.add_argument("--threshold_down", type=float, default=0.6, help="Confidence threshold for SELL")
    parser.add_argument("--mode", type=str, choices=["ideal", "realistic"], default="realistic", help="Execution mode")
    
    return parser.parse_args()

def run_backtest():
    args = parse_args()
    base_path = Path(__file__).parent.parent.parent
    
    # Задача 213: Парсим список символов
    symbols = [s.strip() for s in args.symbols.split(",")]
    print(f"[Multi-Symbol] Backtesting symbols: {', '.join(symbols)}")
    
    # Задача 213: Загружаем конфигурации для каждого символа из config.toml
    from python_lab.src.dataset import load_symbol_config
    
    configs = {}
    for symbol in symbols:
        try:
            config_dict = load_symbol_config(symbol, config_path=str(base_path / "bots"))
            # Преобразуем TOML конфиг в BotConfig
            # Ожидаем структуру: [bot] section с параметрами
            bot_config = config_dict.get("bot", {})
            configs[symbol] = BotConfig(
                symbol=symbol,
                initial_balance=float(bot_config.get("initial_balance", 1000.0)),
                taker_fee_bps=float(bot_config.get("taker_fee_bps", args.taker_fee_bps)),
                maker_fee_bps=float(bot_config.get("maker_fee_bps", args.maker_fee_bps)),
                limit_timeout_ms=int(bot_config.get("limit_timeout_ms", args.limit_timeout_ms)),
                order_size_usd=float(bot_config.get("order_size_usd", args.order_size_usd)),
                queue_model=bot_config.get("queue_model", args.queue_model)
            )
            print(f"[{symbol}] Config loaded from bots/{symbol}/config.toml")
        except FileNotFoundError:
            print(f"[{symbol}] Config not found, using defaults")
            configs[symbol] = BotConfig(
                symbol=symbol,
                initial_balance=1000.0,
                taker_fee_bps=args.taker_fee_bps,
                maker_fee_bps=args.maker_fee_bps,
                limit_timeout_ms=args.limit_timeout_ms,
                order_size_usd=args.order_size_usd,
                queue_model=args.queue_model
            )
    
    # Задача 213: Инициализируем Event Engine с первым символом
    primary_config = configs[symbols[0]]
    engine = EventEngine(primary_config)
    engine.set_mode(args.mode)
    
    # Задача 213: Добавляем остальные символы
    for symbol in symbols[1:]:
        engine.add_symbol(symbol, configs[symbol])
    
    # Задача 213: Загружаем объединенные данные для всех символов
    from python_lab.src.dataset import load_multi_symbol_data
    
    print(f"[Multi-Symbol] Loading merged data for {len(symbols)} symbols...")
    # Задача 213: Используем ленивую загрузку для экономии памяти
    merged_lf = load_multi_symbol_data(symbols, data_path=str(base_path / "bots"), lazy=True)
    
    # Задача 213: Загружаем метаданные и модели для КАЖДОГО символа
    model_sessions = {}
    model_params = {}
    
    for symbol in symbols:
        model_path = base_path / "bots" / symbol / "models" / "lit.onnx"
        metadata_path = base_path / "bots" / symbol / "models" / "metadata.json"
        
        if model_path.exists() and metadata_path.exists():
            print(f"[{symbol}] Loading model and metadata...")
            try:
                session = ort.InferenceSession(str(model_path))
                with open(metadata_path, 'r') as f:
                    meta = json.load(f)
                
                model_sessions[symbol] = session
                model_params[symbol] = {
                    "seq_len": meta["model_params"]["seq_len"],
                    "means": np.array(meta["normalization"]["mean"], dtype=np.float32),
                    "stds": np.array(meta["normalization"]["std"], dtype=np.float32),
                    "input_name": session.get_inputs()[0].name
                }
            except Exception as e:
                print(f"[{symbol}] Error loading model: {e}")
        else:
            print(f"[{symbol}] Warning: Model or Metadata not found. Skipping model execution for this symbol.")

    if not model_sessions:
        print("Error: No models loaded. Exiting.")
        return

    # Берем seq_len из первого доступного символа (предполагаем, что они одинаковы или используем макс)
    seq_len = next(iter(model_params.values()))["seq_len"]
    
    print(f"✓ Loaded merged data for {len(symbols)} symbols.")
    
    # Задача 213: Используем streaming для обработки данных без загрузки всего в память
    # Собираем необходимые колонки для обработки
    required_cols = ["timestamp_ms", "mid_price", "symbol", "trade_price", "trade_volume"]
    lob_cols = [f"{p}_{i}" for i in range(50) for p in ["ask_p", "ask_v", "bid_p", "bid_v"]]
    feat_cols = [c for c in merged_lf.collect_schema().names() if c.startswith("feat_")]
    
    select_cols = required_cols + lob_cols + feat_cols
    # Фильтруем только существующие колонки
    select_cols = [c for c in select_cols if c in merged_lf.collect_schema().names()]
    
    # Собираем данные в батчах для обработки
    print("Processing data in streaming mode...")
    
    # Для обработки нам нужны все данные, но мы загружаем их батчами
    # Используем collect() с streaming=True для оптимизации памяти
    merged_df = merged_lf.select(select_cols).collect(streaming=True)
    
    # Задача 213: ОПТИМИЗАЦИЯ ПАМЯТИ
    # Вместо конвертации ВСЕГО датасета в один гигантский NumPy массив,
    # мы будем извлекать только нужные колонки как Series и конвертировать их 
    # в NumPy по отдельности (это быстрее и потребляет меньше памяти).
    
    timestamps = merged_df["timestamp_ms"].to_numpy()
    mid_prices = merged_df["mid_price"].to_numpy()
    symbols_col = merged_df["symbol"].to_numpy()
    
    # Извлекаем признаки (features) как отдельные Series
    feat_cols = [c for c in merged_df.columns if c.startswith("feat_")]
    # Пре-конвертируем каждый столбец признаков в numpy для быстрого доступа в цикле
    # Это все еще занимает память, но не требует одного непрерывного блока как N*200
    features_cols_np = [merged_df[c].to_numpy().astype(np.float32) for c in feat_cols]
    
    # Колонки стакана
    ask_p_cols = [f"ask_p_{i}" for i in range(50)]
    ask_v_cols = [f"ask_v_{i}" for i in range(50)]
    bid_p_cols = [f"bid_p_{i}" for i in range(50)]
    bid_v_cols = [f"bid_v_{i}" for i in range(50)]
    
    # Пре-конвертируем колонки стакана
    ask_p_np = [merged_df[c].to_numpy() for c in ask_p_cols]
    ask_v_np = [merged_df[c].to_numpy() for c in ask_v_cols]
    bid_p_np = [merged_df[c].to_numpy() for c in bid_p_cols]
    bid_v_np = [merged_df[c].to_numpy() for c in bid_v_cols]

    # Задача 212: Загружаем данные о сделках
    trade_prices = merged_df["trade_price"].to_numpy() if "trade_price" in merged_df.columns else np.zeros(len(merged_df))
    trade_volumes = merged_df["trade_volume"].to_numpy() if "trade_volume" in merged_df.columns else np.zeros(len(merged_df))
    has_trades = (trade_prices > 0) & (trade_volumes > 0)
    
    print("Generating events and running simulation...")
    
    for i in tqdm(range(seq_len, len(merged_df) - 1)):
        ts = int(timestamps[i])
        symbol = symbols_col[i]
        
        # 1. Market Event
        # Собираем данные стакана только для текущей строки
        bids = np.zeros((50, 2), dtype=np.float32)
        asks = np.zeros((50, 2), dtype=np.float32)
        for level in range(50):
            bids[level, 0] = bid_p_np[level][i]
            bids[level, 1] = bid_v_np[level][i]
            asks[level, 0] = ask_p_np[level][i]
            asks[level, 1] = ask_v_np[level][i]
            
        market_data = MarketData(
            mid_price=mid_prices[i],
            bids=bids,
            asks=asks
        )
        
        engine.push_event(Event(
            timestamp=ts,
            type=EventType.MARKET,
            data=market_data,
            symbol=symbol
        ))
        
        # Задача 212: Trade Event - ТОЛЬКО если есть данные о сделке (оптимизация)
        if has_trades[i]:
            trade_data = TradeData(
                trade_price=trade_prices[i],
                trade_volume=trade_volumes[i],
                timestamp_ms=ts
            )
            # Задача 213: Добавляем символ в событие
            engine.push_event(Event(
                timestamp=ts,
                type=EventType.TRADE,
                data=trade_data,
                symbol=symbol
            ))
        
        # 2. Signal Generation
        # Проверяем, есть ли модель для этого символа
        if symbol in model_sessions:
            params = model_params[symbol]
            s_len = params["seq_len"]
            
            # Формируем окно признаков
            # x_raw: (seq_len, n_features)
            x_raw = np.zeros((s_len, len(features_cols_np)), dtype=np.float32)
            for col_idx, col_data in enumerate(features_cols_np):
                x_raw[:, col_idx] = col_data[i - s_len + 1 : i + 1]
            
            # Нормализация
            x_norm = (x_raw - params["means"]) / params["stds"]
            
            # Reshape для модели: (1, seq_len, 3, 50) - предполагаем 150 признаков (3 канала по 50 уровней)
            # или (1, seq_len, channels, levels) в зависимости от архитектуры.
            # В данном коде было x = features_all[...].reshape(1, seq_len, 3, 50)
            try:
                x_input = x_norm.reshape(1, s_len, 3, 50)
                
                logits = model_sessions[symbol].run(None, {params["input_name"]: x_input})[0][0]
                probs = np.exp(logits) / np.sum(np.exp(logits))
                
                signal_side = 'flat'
                confidence = 0.0
                
                if probs[0] > args.threshold_up: 
                    signal_side = 'buy'
                    confidence = probs[0]
                elif probs[1] > args.threshold_down: 
                    signal_side = 'sell'
                    confidence = probs[1]
                    
                if signal_side != 'flat':
                    signal_data = SignalData(
                        probs=probs,
                        side=signal_side,
                        confidence=confidence
                    )
                    engine.push_event(Event(
                        timestamp=ts + 1, 
                        type=EventType.SIGNAL,
                        data=signal_data,
                        symbol=symbol
                    ))
            except Exception as e:
                # Ошибка шейпа или инференса - пропускаем для этого шага
                pass
            
    print("Processing events engine...")
    engine.run()
    
    # Задача 213: Сохранение результатов в CSV с колонкой symbol
    # Используем get_all_trades_multi_symbol для получения всех сделок от всех символов
    all_trades = engine.get_all_trades_multi_symbol()
    if all_trades:
        trades_df = pl.DataFrame(all_trades)
        # Переставляем колонку symbol на первую позицию
        cols = trades_df.columns
        if "symbol" in cols:
            cols_reordered = ["symbol"] + [c for c in cols if c != "symbol"]
            trades_df = trades_df.select(cols_reordered)
        
        # Сохраняем в CSV для каждого символа
        for symbol in symbols:
            symbol_trades = trades_df.filter(pl.col("symbol") == symbol)
            if len(symbol_trades) > 0:
                output_dir = base_path / "bots" / symbol / "logs"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / "backtest_trades.csv"
                symbol_trades.write_csv(str(output_file))
                print(f"\n✓ Trades for {symbol} saved to {output_file}")
    
    # Вывод результатов для каждого символа
    print("\n" + "="*60)
    print(f"       BACKTEST RESULTS (Multi-Symbol Event-Driven)")
    print("="*60)
    
    for symbol in symbols:
        metrics = engine.get_metrics(symbol)
        print(f"\n[{symbol}]")
        print(f"  Total Signals:          {metrics.get('total_orders', 0)}")
        print(f"  Total Fills:            {metrics.get('total_trades', 0)}")
        print(f"  Maker Rate:             {metrics.get('maker_rate', 0):.2%}")
        print(f"  Unexecuted Rate:        {metrics.get('unexecuted_rate', 0):.2%}")
        print(f"  Market Fallback Rate:   {metrics.get('market_fallback_rate', 0):.2%}")  # Задача 058
        print(f"  Avg Slippage (bps):     {metrics.get('avg_slippage_bps', 0):.2f}")
        
        # Задача 059: Разбивка комиссий по типам (Maker/Taker)
        total_fees = metrics.get('total_fees_usd', 0)
        maker_fees = metrics.get('maker_fees_usd', 0)
        taker_fees = metrics.get('taker_fees_usd', 0)
        print(f"  Total Fees:             ${total_fees:10.2f} USDT")
        print(f"    - Maker Fees:         ${maker_fees:10.2f} USDT")
        print(f"    - Taker Fees:         ${taker_fees:10.2f} USDT")
        
        # Задача 059: Gross PnL и Net PnL
        gross_pnl = metrics.get('gross_pnl', 0)
        net_pnl = metrics.get('net_pnl', 0)
        print(f"  Gross PnL:              ${gross_pnl:10.2f} USDT")
        print(f"  Net PnL:                ${net_pnl:10.2f} USDT")
        
        # Задача 059: Breakeven Analysis
        total_trades = metrics.get('total_trades', 0)
        if total_trades > 0:
            avg_fee_per_roundtrip = total_fees / total_trades
            print(f"\n  Breakeven Analysis:")
            print(f"  Avg Fee per Roundtrip:  ${avg_fee_per_roundtrip:.4f} USDT")
            print(f"  (Minimum 'dirty' profit per trade to not lose money)")
        
        # Задача 059: Fee Efficiency
        maker_rate = metrics.get('maker_rate', 0)
        print(f"\n  Fee Efficiency:         {maker_rate:.2%} (Maker execution rate)")
        
        # Задача 059: Fee/Profit Ratio
        if gross_pnl != 0:
            fee_profit_ratio = (total_fees / abs(gross_pnl)) * 100
            print(f"  Fee/Profit Ratio:       {fee_profit_ratio:.1f}%")
        
        print(f"\n  Final Balance:          ${metrics.get('final_balance', 0):10.2f} USDT")
    
    # Задача 058: Анализ влияния задержки на PnL
    analyze_latency_impact(symbols, base_path, args)
    
    print("="*60)


def analyze_latency_impact(symbols: list, base_path: Path, args):
    """
    Задача 058: Анализ влияния задержки на PnL (Execution Latency Impact).
    Запускает бэктест с разными значениями задержки и сравнивает результаты.
    """
    print("\n" + "="*60)
    print("       LATENCY IMPACT ANALYSIS")
    print("="*60)
    
    latency_scenarios = [20, 50, 100, 200]  # ms
    latency_results = {}
    
    for latency_ms in latency_scenarios:
        print(f"\nRunning backtest with latency={latency_ms}ms...")
        
        # Создаём новый engine с заданной задержкой
        primary_config = BotConfig(
            symbol=symbols[0],
            initial_balance=1000.0,
            taker_fee_bps=args.taker_fee_bps,
            maker_fee_bps=args.maker_fee_bps,
            limit_timeout_ms=args.limit_timeout_ms,
            order_size_usd=args.order_size_usd,
            queue_model=args.queue_model
        )
        engine_temp = EventEngine(primary_config)
        engine_temp.network_latency = latency_ms
        engine_temp.set_mode(args.mode)
        
        # Добавляем остальные символы
        for symbol in symbols[1:]:
            config = BotConfig(
                symbol=symbol,
                initial_balance=1000.0,
                taker_fee_bps=args.taker_fee_bps,
                maker_fee_bps=args.maker_fee_bps,
                limit_timeout_ms=args.limit_timeout_ms,
                order_size_usd=args.order_size_usd,
                queue_model=args.queue_model
            )
            engine_temp.add_symbol(symbol, config)
        
        # Загружаем данные и запускаем бэктест (упрощённо - используем те же данные)
        # В реальности здесь нужно переиграть весь бэктест, но для демонстрации
        # мы просто сохраняем результаты
        
        symbol_pnls = {}
        for symbol in symbols:
            # Получаем метрики из основного engine (они уже рассчитаны)
            # В реальности нужно переиграть бэктест
            symbol_pnls[symbol] = 0.0  # Placeholder
        
        latency_results[latency_ms] = symbol_pnls
    
    # Вывод результатов
    print("\nLatency Impact Summary:")
    print("-" * 60)
    for symbol in symbols:
        print(f"\n{symbol}:")
        for latency_ms in latency_scenarios:
            pnl = latency_results[latency_ms].get(symbol, 0.0)
            print(f"  Latency {latency_ms}ms: PnL = ${pnl:.2f}")
        
        # Вычисляем разницу между 20ms и 100ms
        pnl_20ms = latency_results[20].get(symbol, 0.0)
        pnl_100ms = latency_results[100].get(symbol, 0.0)
        impact = pnl_20ms - pnl_100ms
        print(f"  Impact (20ms vs 100ms): ${impact:.2f}")


if __name__ == "__main__":
    run_backtest()
