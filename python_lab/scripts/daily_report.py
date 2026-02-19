#!/usr/bin/env python3
"""
Задача 205: Автоматизированные ежедневные PnL-отчеты

Скрипт для расчета ежедневных метрик производительности бота:
- Net PnL: Чистая прибыль
- Sharpe Ratio: Коэффициент Шарпа (риск-скорректированная доходность)
- Max Drawdown: Максимальная просадка
- Calmar Ratio: Отношение доходности к просадке
- Win Rate: % прибыльных сделок
- Slippage Leakage: Потери на проскальзывании

Использование:
    python daily_report.py --symbol BTCUSDT [--date YYYYMMDD]
    python daily_report.py --symbol BTCUSDT --date 20250215
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl


def load_equity_data(bot_path: Path, date: Optional[str] = None) -> Optional[pl.DataFrame]:
    """
    Загружает данные equity из CSV файла.
    
    Args:
        bot_path: Путь к директории бота (bots/SYMBOL)
        date: Дата в формате YYYYMMDD. Если None, используется сегодняшняя дата
    
    Returns:
        DataFrame с данными equity или None если файл не найден
    """
    equity_csv = bot_path / "logs" / "equity.csv"
    
    if not equity_csv.exists():
        print(f"Warning: equity.csv not found at {equity_csv}")
        return None
    
    try:
        df = pl.read_csv(equity_csv)
        
        # Если дата не указана, используем сегодняшнюю
        if date is None:
            date = datetime.utcnow().strftime("%Y%m%d")
        
        # Парсим дату
        target_date = datetime.strptime(date, "%Y%m%d")
        next_date = target_date + timedelta(days=1)
        
        # Фильтруем данные по дате
        df = df.with_columns(
            pl.col("timestamp").cast(pl.UInt64).alias("ts_ms")
        )
        
        # Конвертируем миллисекунды в дату
        df = df.with_columns(
            (pl.col("ts_ms") / 1000).cast(pl.Int64).cast(pl.Datetime("us")).alias("date_time")
        )
        
        # Фильтруем по дате
        df = df.filter(
            (pl.col("date_time") >= target_date) & 
            (pl.col("date_time") < next_date)
        )
        
        if df.height == 0:
            print(f"Warning: No equity data found for date {date}")
            return None
        
        return df
    except Exception as e:
        print(f"Error loading equity data: {e}")
        return None


def load_trades_data(bot_path: Path, date: Optional[str] = None) -> Optional[pl.DataFrame]:
    """
    Загружает данные о сделках из CSV файла (из CsvTradeLogger).
    
    Args:
        bot_path: Путь к директории бота
        date: Дата в формате YYYYMMDD
    
    Returns:
        DataFrame с данными о сделках или None если файл не найден
    """
    trades_csv = bot_path / "logs" / "trades.csv"
    
    if not trades_csv.exists():
        print(f"Warning: trades.csv not found at {trades_csv}")
        return None
    
    try:
        df = pl.read_csv(trades_csv)
        
        if date is None:
            date = datetime.utcnow().strftime("%Y%m%d")
        
        target_date = datetime.strptime(date, "%Y%m%d")
        next_date = target_date + timedelta(days=1)
        
        # Парсим время из RFC3339 формата
        df = df.with_columns(
            pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M:%S%.fZ").alias("date_time")
        )
        
        # Фильтруем по дате
        df = df.filter(
            (pl.col("date_time") >= target_date) & 
            (pl.col("date_time") < next_date)
        )
        
        return df if df.height > 0 else None
    except Exception as e:
        print(f"Error loading trades data: {e}")
        return None


def calculate_net_pnl(equity_df: pl.DataFrame) -> float:
    """
    Рассчитывает чистую прибыль за период.
    
    Args:
        equity_df: DataFrame с данными equity
    
    Returns:
        Net PnL в абсолютном значении
    """
    if equity_df is None or equity_df.height == 0:
        return 0.0
    
    # Берем первое и последнее значение rest_equity
    first_equity = equity_df["rest_equity"].first()
    last_equity = equity_df["rest_equity"].last()
    
    if first_equity is None or last_equity is None:
        return 0.0
    
    return float(last_equity - first_equity)


def calculate_sharpe_ratio(equity_df: pl.DataFrame, risk_free_rate: float = 0.0) -> float:
    """
    Рассчитывает коэффициент Шарпа.
    
    Args:
        equity_df: DataFrame с данными equity
        risk_free_rate: Безрисковая ставка (годовая)
    
    Returns:
        Sharpe Ratio
    """
    if equity_df is None or equity_df.height < 2:
        return 0.0
    
    # Рассчитываем дневные доходности
    equity_values = equity_df["rest_equity"].to_numpy()
    returns = np.diff(equity_values) / equity_values[:-1]
    
    if len(returns) == 0:
        return 0.0
    
    # Рассчитываем Sharpe Ratio (для дневных данных)
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 0.0
    
    # Annualize: 252 торговых дней в году
    sharpe = (mean_return - risk_free_rate / 252) / std_return * np.sqrt(252)
    
    return float(sharpe)


def calculate_max_drawdown(equity_df: pl.DataFrame) -> float:
    """
    Рассчитывает максимальную просадку.
    
    Args:
        equity_df: DataFrame с данными equity
    
    Returns:
        Max Drawdown в процентах
    """
    if equity_df is None or equity_df.height == 0:
        return 0.0
    
    equity_values = equity_df["rest_equity"].to_numpy()
    
    # Рассчитываем running maximum
    running_max = np.maximum.accumulate(equity_values)
    
    # Рассчитываем drawdown
    drawdown = (equity_values - running_max) / running_max
    
    max_dd = np.min(drawdown)
    
    return float(max_dd * 100)  # В процентах


def calculate_calmar_ratio(equity_df: pl.DataFrame) -> float:
    """
    Рассчитывает Calmar Ratio (доходность / максимальная просадка).
    
    Args:
        equity_df: DataFrame с данными equity
    
    Returns:
        Calmar Ratio
    """
    if equity_df is None or equity_df.height == 0:
        return 0.0
    
    net_pnl = calculate_net_pnl(equity_df)
    max_dd = calculate_max_drawdown(equity_df)
    
    if max_dd == 0 or max_dd > 0:  # Если нет просадки или она положительная
        return 0.0
    
    # Calmar = Annual Return / Max Drawdown
    # Для дневных данных: (Net PnL / Initial Equity) * 252 / |Max DD|
    initial_equity = equity_df["rest_equity"].first()
    
    if initial_equity is None or initial_equity == 0:
        return 0.0
    
    annual_return = (net_pnl / initial_equity) * 252
    calmar = annual_return / abs(max_dd / 100)
    
    return float(calmar)


def calculate_win_rate(trades_df: Optional[pl.DataFrame]) -> float:
    """
    Рассчитывает процент прибыльных сделок.
    
    Args:
        trades_df: DataFrame с данными о сделках
    
    Returns:
        Win Rate в процентах
    """
    if trades_df is None or trades_df.height == 0:
        return 0.0
    
    # Фильтруем только сделки с realized_pnl (закрытые позиции)
    trades_with_pnl = trades_df.filter(pl.col("realized_pnl").is_not_null())
    
    if trades_with_pnl.height == 0:
        return 0.0
    
    # Считаем прибыльные сделки (realized_pnl > 0)
    winning_trades = trades_with_pnl.filter(pl.col("realized_pnl") > 0).height
    total_trades = trades_with_pnl.height
    
    if total_trades == 0:
        return 0.0
    
    win_rate = (winning_trades / total_trades) * 100
    
    return float(win_rate)


def load_slippage_data(bot_path: Path, date: Optional[str] = None) -> Optional[pl.DataFrame]:
    """
    Загружает данные о проскальзывании из CSV файла.
    
    Args:
        bot_path: Путь к директории бота
        date: Дата в формате YYYYMMDD
    
    Returns:
        DataFrame с данными о slippage или None если файл не найден
    """
    slippage_csv = bot_path / "logs" / "slippage.csv"
    
    if not slippage_csv.exists():
        print(f"Warning: slippage.csv not found at {slippage_csv}")
        return None
    
    try:
        df = pl.read_csv(slippage_csv)
        
        if date is None:
            date = datetime.utcnow().strftime("%Y%m%d")
        
        target_date = datetime.strptime(date, "%Y%m%d")
        next_date = target_date + timedelta(days=1)
        
        # Фильтруем по дате
        df = df.with_columns(
            (pl.col("timestamp_utc") / 1000).cast(pl.Int64).cast(pl.Datetime("us")).alias("date_time")
        )
        
        df = df.filter(
            (pl.col("date_time") >= target_date) & 
            (pl.col("date_time") < next_date)
        )
        
        return df if df.height > 0 else None
    except Exception as e:
        print(f"Error loading slippage data: {e}")
        return None


def calculate_slippage_leakage(slippage_df: Optional[pl.DataFrame]) -> float:
    """
    Рассчитывает общие потери на проскальзывании.
    
    Args:
        slippage_df: DataFrame с данными о slippage
    
    Returns:
        Slippage Leakage в базисных пунктах
    """
    if slippage_df is None or slippage_df.height == 0:
        return 0.0
    
    # Суммируем все slippage_bps
    total_slippage = slippage_df["slippage_bps"].sum()
    
    return float(total_slippage)


def generate_json_report(
    symbol: str,
    date: str,
    metrics: Dict[str, float],
    bot_path: Path
) -> Path:
    """
    Генерирует JSON отчет.
    
    Args:
        symbol: Символ (например: BTCUSDT)
        date: Дата в формате YYYYMMDD
        metrics: Словарь с метриками
        bot_path: Путь к директории бота
    
    Returns:
        Путь к созданному файлу
    """
    reports_dir = bot_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = reports_dir / f"daily_{date}.json"
    
    report_data = {
        "symbol": symbol,
        "date": date,
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": metrics,
    }
    
    with open(report_file, "w") as f:
        json.dump(report_data, f, indent=2)
    
    return report_file


def generate_markdown_report(
    symbol: str,
    date: str,
    metrics: Dict[str, float],
    bot_path: Path
) -> Path:
    """
    Генерирует Markdown отчет.
    
    Args:
        symbol: Символ
        date: Дата в формате YYYYMMDD
        metrics: Словарь с метриками
        bot_path: Путь к директории бота
    
    Returns:
        Путь к созданному файлу
    """
    reports_dir = bot_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = reports_dir / f"daily_{date}.md"
    
    # Форматируем дату
    date_obj = datetime.strptime(date, "%Y%m%d")
    formatted_date = date_obj.strftime("%Y-%m-%d")
    
    content = f"""# Daily PnL Report - {symbol}

**Date:** {formatted_date}  
**Generated:** {datetime.utcnow().isoformat()}

## Performance Metrics

| Metric | Value |
|--------|-------|
| Net PnL | ${metrics['net_pnl']:.2f} |
| Sharpe Ratio | {metrics['sharpe_ratio']:.4f} |
| Max Drawdown | {metrics['max_drawdown']:.2f}% |
| Calmar Ratio | {metrics['calmar_ratio']:.4f} |
| Win Rate | {metrics['win_rate']:.2f}% |
| Slippage Leakage | {metrics['slippage_leakage']:.2f} bps |

## Summary

- **Total Profit/Loss:** ${metrics['net_pnl']:.2f}
- **Risk-Adjusted Return (Sharpe):** {metrics['sharpe_ratio']:.4f}
- **Maximum Drawdown:** {metrics['max_drawdown']:.2f}%
- **Return/Risk Ratio (Calmar):** {metrics['calmar_ratio']:.4f}
- **Winning Trades:** {metrics['win_rate']:.2f}%
- **Total Slippage Cost:** {metrics['slippage_leakage']:.2f} bps

---
*Report generated by Neirobot LiT Daily Report System*
"""
    
    with open(report_file, "w") as f:
        f.write(content)
    
    return report_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate daily PnL reports for trading bot"
    )
    parser.add_argument(
        "--symbol",
        required=True,
        help="Trading symbol (e.g., BTCUSDT)"
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Report date in YYYYMMDD format (default: today)"
    )
    parser.add_argument(
        "--bot-root",
        default="bots",
        help="Root directory for bot data (default: bots)"
    )
    
    args = parser.parse_args()
    
    # Определяем путь к директории бота
    bot_path = Path(args.bot_root) / args.symbol
    
    if not bot_path.exists():
        print(f"Error: Bot directory not found at {bot_path}")
        sys.exit(1)
    
    # Используем сегодняшнюю дату если не указана
    date = args.date or datetime.utcnow().strftime("%Y%m%d")
    
    print(f"Generating daily report for {args.symbol} on {date}...")
    
    # Загружаем данные
    equity_df = load_equity_data(bot_path, date)
    trades_df = load_trades_data(bot_path, date)
    slippage_df = load_slippage_data(bot_path, date)
    
    if equity_df is None:
        print(f"Error: No equity data found for {date}")
        sys.exit(1)
    
    # Рассчитываем метрики
    metrics = {
        "net_pnl": calculate_net_pnl(equity_df),
        "sharpe_ratio": calculate_sharpe_ratio(equity_df),
        "max_drawdown": calculate_max_drawdown(equity_df),
        "calmar_ratio": calculate_calmar_ratio(equity_df),
        "win_rate": calculate_win_rate(trades_df),
        "slippage_leakage": calculate_slippage_leakage(slippage_df),
    }
    
    # Генерируем отчеты
    json_report = generate_json_report(args.symbol, date, metrics, bot_path)
    md_report = generate_markdown_report(args.symbol, date, metrics, bot_path)
    
    print(f"\n✓ JSON report: {json_report}")
    print(f"✓ Markdown report: {md_report}")
    
    # Выводим метрики в консоль
    print(f"\n{'='*50}")
    print(f"Daily Report Summary - {args.symbol} ({date})")
    print(f"{'='*50}")
    print(f"Net PnL:           ${metrics['net_pnl']:.2f}")
    print(f"Sharpe Ratio:      {metrics['sharpe_ratio']:.4f}")
    print(f"Max Drawdown:      {metrics['max_drawdown']:.2f}%")
    print(f"Calmar Ratio:      {metrics['calmar_ratio']:.4f}")
    print(f"Win Rate:          {metrics['win_rate']:.2f}%")
    print(f"Slippage Leakage:  {metrics['slippage_leakage']:.2f} bps")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
