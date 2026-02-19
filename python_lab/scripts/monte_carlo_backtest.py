"""
Задача 214: Монте-Карло симуляция вариативности задержек (Latency Perturbation)

Скрипт для параллельного запуска бэктестов с различными задержками.
Позволяет оценить устойчивость стратегии к нестабильности сети и лагам биржи.
"""

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Импорт компонентов бэктестера
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.backtest.engine import EventEngine, BotConfig, Event, EventType, MarketData, SignalData, OrderData
from src.backtest.perturbation import LatencyGenerator
from src.dataset import load_multi_symbol_data, load_symbol_config


def run_single_iteration(args: tuple) -> Dict[str, Any]:
    """
    Worker-функция для одной итерации Монте-Карло.
    
    Запускает бэктест с уникальным набором задержек и возвращает метрики.
    Джиттер применяется ТОЛЬКО к OrderEvent (через timestamp), не к MarketData.
    
    Args:
        args: Кортеж (iteration_id, seed, config_dict, events_data)
        
    Returns:
        Словарь с метриками итерации
    """
    iteration_id, seed, config_dict, events_data = args
    
    # Создаем генератор задержек с уникальным seed
    latency_gen = LatencyGenerator(
        mean_ms=config_dict.get('mean_latency_ms', 20.0),
        std_ms=config_dict.get('std_latency_ms', 15.0),
        seed=seed,
        execution_quality_csv=config_dict.get('execution_quality_csv')
    )
    
    # Создаем конфигурацию бота
    bot_config = BotConfig(
        symbol=config_dict['symbol'],
        initial_balance=config_dict['initial_balance'],
        taker_fee_bps=config_dict['taker_fee_bps'],
        maker_fee_bps=config_dict['maker_fee_bps'],
        order_size_usd=config_dict['order_size_usd']
    )
    
    # Инициализируем движок
    engine = EventEngine(bot_config)
    engine.set_mode("realistic")
    
    # Генерируем задержки для всех OrderEvent
    order_events = [e for e in events_data if e[1] == EventType.ORDER]
    n_order_events = len(order_events)
    latencies = latency_gen.generate(size=n_order_events) if n_order_events > 0 else np.array([])
    latency_idx = 0
    
    # Добавляем события в движок
    for timestamp, event_type, event_data, symbol in events_data:
        if event_type == EventType.MARKET:
            # MarketData НЕ модифицируем - это историческая константа
            event = Event(
                timestamp=timestamp,
                type=EventType.MARKET,
                data=event_data,
                symbol=symbol
            )
            engine.push_event(event)
            
        elif event_type == EventType.SIGNAL:
            # SignalData НЕ модифицируем
            event = Event(
                timestamp=timestamp,
                type=EventType.SIGNAL,
                data=event_data,
                symbol=symbol
            )
            engine.push_event(event)
            
        elif event_type == EventType.ORDER:
            # OrderEvent - применяем джиттер к timestamp
            jitter = latencies[latency_idx] if latency_idx < len(latencies) else 0
            latency_idx += 1
            
            event = Event(
                timestamp=int(timestamp + jitter),  # Добавляем джиттер к timestamp
                type=EventType.ORDER,
                data=event_data,
                symbol=symbol
            )
            engine.push_event(event)
    
    # Запускаем бэктест
    engine.run()
    
    # Собираем метрики
    metrics = engine.get_metrics(bot_config.symbol)
    
    # Рассчитываем дополнительные метрики
    state = engine.get_state(bot_config.symbol)
    
    # Drawdown
    peak = state.config.initial_balance
    max_drawdown = 0.0
    for trade in state.trades:
        current_balance = state.balance
        if current_balance > peak:
            peak = current_balance
        drawdown = (peak - current_balance) / peak * 100.0
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    # 95-й перцентиль задержки
    p95_latency = float(np.percentile(latencies, 95)) if len(latencies) > 0 else 0.0
    
    return {
        'iteration': iteration_id,
        'seed': seed,
        'pnl': metrics.get('net_pnl', 0.0),
        'final_balance': metrics.get('final_balance', bot_config.initial_balance),
        'max_drawdown_pct': max_drawdown,
        'total_trades': metrics.get('total_trades', 0),
        'maker_rate': metrics.get('maker_rate', 0.0),
        'avg_slippage_bps': metrics.get('avg_slippage_bps', 0.0),
        'p95_latency_ms': p95_latency,
        'mean_latency_ms': float(np.mean(latencies)) if len(latencies) > 0 else 0.0,
        'std_latency_ms': float(np.std(latencies)) if len(latencies) > 0 else 0.0
    }


def load_backtest_events(
    merged_df: pl.DataFrame,
    base_path: Path
) -> List[Tuple[int, EventType, Any, str]]:
    """
    Преобразование объединенных данных Parquet в события для EventEngine.
    
    Загружает как MarketData, так и SignalData события из объединенного потока.
    
    Args:
        merged_df: Объединенный DataFrame из load_multi_symbol_data
        base_path: Базовый путь проекта
        
    Returns:
        Список событий (timestamp, event_type, data, symbol)
    """
    events = []
    
    try:
        # Преобразуем LazyFrame в DataFrame если нужно
        if isinstance(merged_df, pl.LazyFrame):
            df = merged_df.collect()
        else:
            df = merged_df
        
        # Преобразуем Order Book данные в события
        for row in df.iter_rows(named=True):
            timestamp = int(row.get('timestamp_ms', 0))
            symbol = row.get('symbol', 'UNKNOWN')
            
            # Создаем MarketData событие
            bids = np.array([[row.get(f'bid_price_{i}', 0), row.get(f'bid_volume_{i}', 0)] 
                           for i in range(50)], dtype=np.float64)
            asks = np.array([[row.get(f'ask_price_{i}', 0), row.get(f'ask_volume_{i}', 0)] 
                           for i in range(50)], dtype=np.float64)
            
            mid_price = (row.get('bid_price_0', 0) + row.get('ask_price_0', 0)) / 2
            
            market_data = MarketData(
                mid_price=mid_price,
                bids=bids,
                asks=asks
            )
            
            events.append((timestamp, EventType.MARKET, market_data, symbol))
            
            # Проверяем наличие колонок сигналов
            # Если есть колонки side и confidence, создаем SignalData событие
            if 'side' in row and row.get('side') is not None:
                side = str(row.get('side', 'flat')).lower()
                confidence = float(row.get('confidence', 0.5))
                
                # Создаем вероятности (упрощенно)
                if side == 'buy':
                    probs = np.array([confidence, 1 - confidence, 0.0])
                elif side == 'sell':
                    probs = np.array([1 - confidence, confidence, 0.0])
                else:
                    probs = np.array([0.33, 0.33, 0.34])
                
                signal_data = SignalData(
                    probs=probs,
                    side=side,
                    confidence=confidence
                )
                
                events.append((timestamp, EventType.SIGNAL, signal_data, symbol))
        
        print(f"[MonteCarloBacktest] Loaded {len(events)} events from merged data")
        return events
        
    except Exception as e:
        print(f"[MonteCarloBacktest] Error loading data: {e}")
        print(f"[MonteCarloBacktest] Using synthetic data for demonstration")
        return generate_synthetic_events('BTCUSDT', n_samples=1000)


def generate_synthetic_events(symbol: str, n_samples: int = 1000) -> List[Tuple[int, EventType, Any, str]]:
    """
    Генерация синтетических событий для демонстрации.
    
    Используется, если реальные данные недоступны.
    
    Args:
        symbol: Торговый символ
        n_samples: Количество сэмплов
        
    Returns:
        Список событий (timestamp, event_type, data, symbol)
    """
    events = []
    
    base_price = 50000.0
    timestamp = 0
    
    for i in range(n_samples):
        # Генерируем случайное движение цены
        price_change = np.random.randn() * 10
        mid_price = base_price + price_change
        
        # Генерируем стакан
        bids = np.array([[mid_price - j, 1.0 + np.random.rand()] for j in range(1, 51)])
        asks = np.array([[mid_price + j, 1.0 + np.random.rand()] for j in range(1, 51)])
        
        market_data = MarketData(
            mid_price=mid_price,
            bids=bids,
            asks=asks
        )
        
        events.append((timestamp, EventType.MARKET, market_data, symbol))
        
        # Генерируем сигнал каждые 10 тиков
        if i % 10 == 0:
            probs = np.random.dirichlet([1, 1, 1])
            side = 'buy' if probs[0] > 0.5 else 'sell' if probs[1] > 0.3 else 'flat'
            confidence = float(np.max(probs))
            
            signal_data = SignalData(
                probs=probs,
                side=side,
                confidence=confidence
            )
            
            events.append((timestamp, EventType.SIGNAL, signal_data, symbol))
        
        timestamp += 100  # 100ms между тиками
        base_price = mid_price
    
    return events


def plot_results(results_df: pd.DataFrame, output_dir: Path):
    """
    Построение графиков результатов Монте-Карло симуляции.
    
    Args:
        results_df: DataFrame с результатами итераций
        output_dir: Директория для сохранения графиков
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Настройка стиля
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Гистограмма распределения PnL
    ax1 = axes[0, 0]
    sns.histplot(results_df['pnl'], bins=30, kde=True, ax=ax1, color='steelblue')
    ax1.axvline(results_df['pnl'].mean(), color='red', linestyle='--', label=f'Mean: ${results_df["pnl"].mean():.2f}')
    ax1.axvline(results_df['pnl'].quantile(0.05), color='orange', linestyle='--', label=f'5th percentile: ${results_df["pnl"].quantile(0.05):.2f}')
    ax1.set_xlabel('PnL ($)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of PnL across Monte Carlo iterations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Кривая устойчивости (PnL vs 95-й перцентиль задержки)
    ax2 = axes[0, 1]
    scatter = ax2.scatter(results_df['p95_latency_ms'], results_df['pnl'], 
                         c=results_df['pnl'], cmap='RdYlGn', alpha=0.6, s=50)
    ax2.set_xlabel('95th Percentile Latency (ms)')
    ax2.set_ylabel('PnL ($)')
    ax2.set_title('Stability Curve: PnL vs Latency')
    plt.colorbar(scatter, ax=ax2, label='PnL ($)')
    ax2.grid(True, alpha=0.3)
    
    # Добавляем линию тренда
    z = np.polyfit(results_df['p95_latency_ms'], results_df['pnl'], 1)
    p = np.poly1d(z)
    ax2.plot(results_df['p95_latency_ms'], p(results_df['p95_latency_ms']), 
            "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
    ax2.legend()
    
    # 3. Распределение Drawdown
    ax3 = axes[1, 0]
    sns.histplot(results_df['max_drawdown_pct'], bins=30, kde=True, ax=ax3, color='coral')
    ax3.axvline(results_df['max_drawdown_pct'].mean(), color='red', linestyle='--', 
               label=f'Mean: {results_df["max_drawdown_pct"].mean():.2f}%')
    ax3.set_xlabel('Max Drawdown (%)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Maximum Drawdown')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Boxplot метрик
    ax4 = axes[1, 1]
    metrics_data = results_df[['pnl', 'max_drawdown_pct', 'avg_slippage_bps']].copy()
    metrics_data.columns = ['PnL ($)', 'Max DD (%)', 'Avg Slippage (bps)']
    
    # Нормализуем для визуализации
    metrics_normalized = (metrics_data - metrics_data.mean()) / metrics_data.std()
    sns.boxplot(data=metrics_normalized, ax=ax4, palette='Set2')
    ax4.set_ylabel('Normalized Value (z-score)')
    ax4.set_title('Distribution of Key Metrics (Normalized)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Сохраняем график
    output_path = output_dir / 'monte_carlo_pnl.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[MonteCarloBacktest] Saved plot to: {output_path}")
    
    plt.close()


def print_summary(results_df: pd.DataFrame):
    """
    Вывод текстового отчета с метриками.
    
    Args:
        results_df: DataFrame с результатами итераций
    """
    print("\n" + "="*80)
    print("MONTE CARLO BACKTEST SUMMARY")
    print("="*80)
    
    print(f"\nTotal Iterations: {len(results_df)}")
    
    print("\n--- PnL Statistics ---")
    print(f"Mean PnL: ${results_df['pnl'].mean():.2f}")
    print(f"Median PnL: ${results_df['pnl'].median():.2f}")
    print(f"Std Dev: ${results_df['pnl'].std():.2f}")
    print(f"Min PnL: ${results_df['pnl'].min():.2f}")
    print(f"Max PnL: ${results_df['pnl'].max():.2f}")
    
    print("\n--- Risk Metrics ---")
    pnl_at_risk = results_df['pnl'].quantile(0.05)
    print(f"PnL at Risk (5th percentile): ${pnl_at_risk:.2f}")
    
    reliability_score = (results_df['pnl'] > 0).sum() / len(results_df) * 100
    print(f"Reliability Score: {reliability_score:.2f}% (profitable iterations)")
    
    print(f"\nMean Max Drawdown: {results_df['max_drawdown_pct'].mean():.2f}%")
    print(f"Worst Drawdown: {results_df['max_drawdown_pct'].max():.2f}%")
    
    print("\n--- Latency Statistics ---")
    print(f"Mean 95th Percentile Latency: {results_df['p95_latency_ms'].mean():.2f}ms")
    print(f"Range: [{results_df['p95_latency_ms'].min():.2f}ms - {results_df['p95_latency_ms'].max():.2f}ms]")
    
    print("\n--- Execution Quality ---")
    print(f"Mean Maker Rate: {results_df['maker_rate'].mean()*100:.2f}%")
    print(f"Mean Slippage: {results_df['avg_slippage_bps'].mean():.2f} bps")
    
    print("\n" + "="*80)
    
    # Оценка устойчивости стратегии
    if pnl_at_risk < -100:
        print("⚠️  WARNING: Strategy shows high fragility to latency variations!")
        print("   Consider optimizing execution logic or reducing position sizes.")
    elif reliability_score < 50:
        print("⚠️  WARNING: Strategy is profitable in less than 50% of scenarios!")
    else:
        print("✓ Strategy shows reasonable robustness to latency variations.")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Monte Carlo Backtest with Latency Perturbation')
    parser.add_argument('--symbols', type=str, default='BTCUSDT', 
                       help='Comma-separated list of symbols (e.g., BTCUSDT,ETHUSDT)')
    parser.add_argument('--iterations', type=int, default=100, help='Number of Monte Carlo iterations')
    parser.add_argument('--initial-balance', type=float, default=1000.0, help='Initial balance in USD')
    parser.add_argument('--order-size', type=float, default=100.0, help='Order size in USD')
    parser.add_argument('--mean-latency', type=float, default=20.0, help='Mean latency in ms')
    parser.add_argument('--std-latency', type=float, default=15.0, help='Std dev of latency in ms')
    parser.add_argument('--execution-quality-csv', type=str, default=None, 
                       help='Path to execution_quality.csv for real latency params')
    parser.add_argument('--output-dir', type=str, default=None, 
                       help='Output directory for reports (default: ./bots/SYMBOL/reports/)')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    parser.add_argument('--workers', type=int, default=None, 
                       help='Number of worker processes (default: CPU count)')
    
    args = parser.parse_args()
    
    # Определяем базовый путь проекта
    base_path = Path(__file__).parent.parent.parent
    
    # Парсим список символов
    symbols = [s.strip() for s in args.symbols.split(',')]
    primary_symbol = symbols[0]
    
    # Определяем директорию для вывода
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = base_path / 'bots' / primary_symbol / 'reports'
    
    print(f"\n[MonteCarloBacktest] Starting Monte Carlo simulation...")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Iterations: {args.iterations}")
    print(f"Mean Latency: {args.mean_latency}ms, Std: {args.std_latency}ms")
    print(f"Base path: {base_path}")
    print(f"Output dir: {output_dir}")
    
    # Загружаем объединенные данные для всех символов (Задача 213 Integration)
    print(f"\n[MonteCarloBacktest] Loading multi-symbol data...")
    try:
        merged_lf = load_multi_symbol_data(symbols, data_path=str(base_path / 'bots'), lazy=True)
        merged_df = merged_lf.collect()
        print(f"[MonteCarloBacktest] Loaded {len(merged_df)} rows from {len(symbols)} symbols")
    except Exception as e:
        print(f"[MonteCarloBacktest] Error loading multi-symbol data: {e}")
        print(f"[MonteCarloBacktest] Falling back to synthetic data")
        merged_df = None
    
    # Преобразуем данные в события
    print(f"\n[MonteCarloBacktest] Converting data to events...")
    if merged_df is not None:
        events_data = load_backtest_events(merged_df, base_path)
    else:
        events_data = generate_synthetic_events(primary_symbol, n_samples=1000)
    
    if not events_data:
        print("[MonteCarloBacktest] ERROR: No events loaded!")
        return
    
    print(f"[MonteCarloBacktest] Loaded {len(events_data)} events")
    
    # Подготовка конфигурации для worker-функций
    config_dict = {
        'symbol': primary_symbol,
        'symbols': symbols,
        'initial_balance': args.initial_balance,
        'taker_fee_bps': 6.0,
        'maker_fee_bps': 2.0,
        'order_size_usd': args.order_size,
        'mean_latency_ms': args.mean_latency,
        'std_latency_ms': args.std_latency,
        'execution_quality_csv': args.execution_quality_csv
    }
    
    # Подготовка аргументов для worker-функций
    worker_args = [
        (i, args.seed + i, config_dict, events_data)
        for i in range(args.iterations)
    ]
    
    # Параллельный запуск итераций
    n_workers = args.workers if args.workers else mp.cpu_count()
    print(f"\n[MonteCarloBacktest] Running {args.iterations} iterations on {n_workers} workers...")
    
    with mp.Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap(run_single_iteration, worker_args),
            total=args.iterations,
            desc="Monte Carlo Progress"
        ))
    
    # Преобразуем результаты в DataFrame
    results_df = pd.DataFrame(results)
    
    # Сохраняем результаты в CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'monte_carlo_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\n[MonteCarloBacktest] Saved results to: {csv_path}")
    
    # Построение графиков
    print(f"\n[MonteCarloBacktest] Generating plots...")
    plot_results(results_df, output_dir)
    
    # Вывод отчета
    print_summary(results_df)


if __name__ == '__main__':
    main()
