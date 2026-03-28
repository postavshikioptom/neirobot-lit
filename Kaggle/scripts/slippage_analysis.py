#!/usr/bin/env python3
"""
Задача 201: Анализ реального проскальзывания против ожидаемого (Real vs Expected Slippage)

Скрипт для анализа логов trades.csv и построения графиков корреляции slippage
с волатильностью и спредом.

Использование:
    python slippage_analysis.py --symbol BTCUSDT --days 7 --plot
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_trades_data(symbol: str, days: int) -> pl.DataFrame:
    """
    Загружает данные о сделках из trades.csv
    
    Args:
        symbol: Торговый символ (например, BTCUSDT)
        days: Количество дней для анализа
        
    Returns:
        DataFrame с данными о сделках
    """
    trades_path = Path(f"bots/{symbol}/logs/trades.csv")
    
    if not trades_path.exists():
        raise FileNotFoundError(f"Файл {trades_path} не найден")
    
    # Загружаем данные через polars (быстрее pandas)
    df = pl.read_csv(trades_path)
    
    # Фильтруем по дате если указано
    if days > 0:
        cutoff_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        df = df.filter(pl.col("timestamp_utc") >= cutoff_ts)
    
    return df


def calculate_statistics(df: pl.DataFrame) -> dict:
    """
    Вычисляет статистику по slippage и корреляции
    
    Args:
        df: DataFrame с данными о сделках
        
    Returns:
        Словарь со статистикой
    """
    # Вычисляем волатильность как скользящее стандартное отклонение slippage
    df_pd = df.to_pandas()
    df_pd['volatility'] = df_pd['slippage_bps'].rolling(window=20, min_periods=1).std()
    
    # Вычисляем корреляции
    corr_slippage_volatility = df_pd['slippage_bps'].corr(df_pd['volatility'])
    corr_slippage_spread = df_pd['slippage_bps'].corr(df_pd['spread_bps'])
    corr_slippage_latency = df_pd['slippage_bps'].corr(df_pd['latency_ms'])
    
    stats = {
        "total_trades": len(df),
        "mean_slippage_bps": df["slippage_bps"].mean(),
        "median_slippage_bps": df["slippage_bps"].median(),
        "std_slippage_bps": df["slippage_bps"].std(),
        "min_slippage_bps": df["slippage_bps"].min(),
        "max_slippage_bps": df["slippage_bps"].max(),
        "mean_latency_ms": df["latency_ms"].mean(),
        "median_latency_ms": df["latency_ms"].median(),
        "mean_spread_bps": df["spread_bps"].mean(),
        "corr_slippage_volatility": corr_slippage_volatility,
        "corr_slippage_spread": corr_slippage_spread,
        "corr_slippage_latency": corr_slippage_latency,
    }
    
    return stats, df_pd


def print_statistics(stats: dict):
    """Выводит статистику в консоль"""
    print("\n" + "="*60)
    print("СТАТИСТИКА ПРОСКАЛЬЗЫВАНИЯ")
    print("="*60)
    print(f"Всего сделок: {stats['total_trades']}")
    print(f"\nSlippage (базисные пункты):")
    print(f"  Среднее:  {stats['mean_slippage_bps']:.2f} bps")
    print(f"  Медиана:  {stats['median_slippage_bps']:.2f} bps")
    print(f"  Ст. откл: {stats['std_slippage_bps']:.2f} bps")
    print(f"  Мин:      {stats['min_slippage_bps']:.2f} bps")
    print(f"  Макс:     {stats['max_slippage_bps']:.2f} bps")
    print(f"\nЗадержка (миллисекунды):")
    print(f"  Среднее:  {stats['mean_latency_ms']:.2f} ms")
    print(f"  Медиана:  {stats['median_latency_ms']:.2f} ms")
    print(f"\nСпред (базисные пункты):")
    print(f"  Среднее:  {stats['mean_spread_bps']:.2f} bps")
    print(f"\nКОРРЕЛЯЦИИ:")
    print(f"  Slippage vs Volatility: {stats['corr_slippage_volatility']:.4f}")
    print(f"  Slippage vs Spread:     {stats['corr_slippage_spread']:.4f}")
    print(f"  Slippage vs Latency:    {stats['corr_slippage_latency']:.4f}")
    print("="*60 + "\n")


def plot_slippage_analysis(df: pl.DataFrame, symbol: str):
    """
    Строит графики анализа slippage
    
    Args:
        df: DataFrame с данными о сделках
        symbol: Торговый символ для заголовка
    """
    # Конвертируем в pandas для matplotlib (polars не поддерживает прямую интеграцию)
    df_pd = df.to_pandas()
    
    # Вычисляем волатильность
    df_pd['volatility'] = df_pd['slippage_bps'].rolling(window=20, min_periods=1).std()
    
    # Создаем фигуру с 6 подграфиками
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'Анализ Slippage для {symbol}', fontsize=16, fontweight='bold')
    
    # 1. Распределение slippage
    axes[0, 0].hist(df_pd['slippage_bps'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Slippage (bps)')
    axes[0, 0].set_ylabel('Частота')
    axes[0, 0].set_title('Распределение Slippage')
    axes[0, 0].axvline(df_pd['slippage_bps'].mean(), color='red', linestyle='--', 
                       label=f'Среднее: {df_pd["slippage_bps"].mean():.2f} bps')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Slippage vs Spread (корреляция)
    axes[0, 1].scatter(df_pd['spread_bps'], df_pd['slippage_bps'], alpha=0.5)
    axes[0, 1].set_xlabel('Spread (bps)')
    axes[0, 1].set_ylabel('Slippage (bps)')
    corr_spread = df_pd['slippage_bps'].corr(df_pd['spread_bps'])
    axes[0, 1].set_title(f'Slippage vs Spread (корр: {corr_spread:.4f})')
    
    # Добавляем линию тренда
    z = df_pd[['spread_bps', 'slippage_bps']].dropna().values
    if len(z) > 1:
        from numpy.polynomial import Polynomial
        p = Polynomial.fit(z[:, 0], z[:, 1], 1)
        x_trend = np.linspace(z[:, 0].min(), z[:, 0].max(), 100)
        axes[0, 1].plot(x_trend, p(x_trend), 'r--', alpha=0.8, label='Тренд')
        axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Slippage vs Volatility (корреляция)
    axes[1, 0].scatter(df_pd['volatility'], df_pd['slippage_bps'], alpha=0.5, color='green')
    axes[1, 0].set_xlabel('Volatility (std slippage)')
    axes[1, 0].set_ylabel('Slippage (bps)')
    corr_vol = df_pd['slippage_bps'].corr(df_pd['volatility'])
    axes[1, 0].set_title(f'Slippage vs Volatility (корр: {corr_vol:.4f})')
    
    # Добавляем линию тренда
    z = df_pd[['volatility', 'slippage_bps']].dropna().values
    if len(z) > 1:
        from numpy.polynomial import Polynomial
        p = Polynomial.fit(z[:, 0], z[:, 1], 1)
        x_trend = np.linspace(z[:, 0].min(), z[:, 0].max(), 100)
        axes[1, 0].plot(x_trend, p(x_trend), 'r--', alpha=0.8, label='Тренд')
        axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Slippage vs Latency
    axes[1, 1].scatter(df_pd['latency_ms'], df_pd['slippage_bps'], alpha=0.5, color='orange')
    axes[1, 1].set_xlabel('Latency (ms)')
    axes[1, 1].set_ylabel('Slippage (bps)')
    corr_lat = df_pd['slippage_bps'].corr(df_pd['latency_ms'])
    axes[1, 1].set_title(f'Slippage vs Latency (корр: {corr_lat:.4f})')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. Временной ряд slippage
    df_pd['datetime'] = pl.from_epoch(df_pd['timestamp_utc'], time_unit='ms').to_pandas()
    axes[2, 0].plot(df_pd['datetime'], df_pd['slippage_bps'], alpha=0.6, linewidth=0.8)
    axes[2, 0].set_xlabel('Время')
    axes[2, 0].set_ylabel('Slippage (bps)')
    axes[2, 0].set_title('Slippage во времени')
    axes[2, 0].tick_params(axis='x', rotation=45)
    axes[2, 0].grid(True, alpha=0.3)
    
    # 6. Распределение spread
    axes[2, 1].hist(df_pd['spread_bps'], bins=50, edgecolor='black', alpha=0.7, color='purple')
    axes[2, 1].set_xlabel('Spread (bps)')
    axes[2, 1].set_ylabel('Частота')
    axes[2, 1].set_title('Распределение Spread')
    axes[2, 1].axvline(df_pd['spread_bps'].mean(), color='red', linestyle='--',
                       label=f'Среднее: {df_pd["spread_bps"].mean():.2f} bps')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Сохраняем график
    output_path = Path(f"bots/{symbol}/logs/slippage_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"График сохранен: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Анализ проскальзывания (slippage) для торгового бота'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        required=True,
        help='Торговый символ (например, BTCUSDT)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Количество дней для анализа (0 = все данные)'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Построить графики'
    )
    
    args = parser.parse_args()
    
    try:
        # Загружаем данные
        print(f"Загрузка данных для {args.symbol}...")
        df = load_trades_data(args.symbol, args.days)
        
        if len(df) == 0:
            print("Нет данных для анализа")
            return 1
        
        # Вычисляем и выводим статистику
        stats, df_pd = calculate_statistics(df)
        print_statistics(stats)
        
        # Строим графики если указан флаг --plot
        if args.plot:
            print("Построение графиков...")
            plot_slippage_analysis(df, args.symbol)
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Неожиданная ошибка: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
