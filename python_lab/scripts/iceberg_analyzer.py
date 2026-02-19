#!/usr/bin/env python3
"""
Задача 207: Анализ эффективности Iceberg-ордеров

Этот скрипт сравнивает эффективность Iceberg-ордеров с агрессивным исполнением
и вычисляет метрику "Time to invisibility" - время, через которое стакан начинает "убегать".
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import sys

# Добавляем путь к src для импорта утилит
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_market_impact_data(bot_path: Path) -> pd.DataFrame:
    """Загружает данные о влиянии на рынок из market_impact.csv"""
    impact_file = bot_path / "logs" / "market_impact.csv"
    
    if not impact_file.exists():
        raise FileNotFoundError(f"Market impact file not found: {impact_file}")
    
    df = pd.read_csv(impact_file)
    df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    
    return df


def load_execution_data(bot_path: Path) -> pd.DataFrame:
    """Загружает данные об исполнении из trades.csv"""
    trades_file = bot_path / "logs" / "trades.csv"
    
    if not trades_file.exists():
        raise FileNotFoundError(f"Trades file not found: {trades_file}")
    
    df = pd.read_csv(trades_file)
    df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    
    return df


def identify_iceberg_orders(impact_df: pd.DataFrame) -> pd.DataFrame:
    """
    Идентифицирует Iceberg-ордера по паттерну множественных fills
    с одинаковой ценой и стороной в короткий промежуток времени
    """
    # Группируем по order_id и считаем количество fills
    order_groups = impact_df.groupby('order_id').agg({
        'fill_counter': 'count',
        'exec_qty': 'sum',
        'timestamp': ['min', 'max'],
        'side': 'first',
        'exec_price': 'first'
    }).reset_index()
    
    order_groups.columns = ['order_id', 'fill_count', 'total_qty', 
                            'start_time', 'end_time', 'side', 'price']
    
    # Iceberg-ордера имеют множественные fills
    iceberg_orders = order_groups[order_groups['fill_count'] > 1].copy()
    iceberg_orders['duration_sec'] = (
        iceberg_orders['end_time'] - iceberg_orders['start_time']
    ).dt.total_seconds()
    
    return iceberg_orders


def calculate_time_to_invisibility(
    impact_df: pd.DataFrame,
    iceberg_orders: pd.DataFrame,
    window_sec: int = 60
) -> pd.DataFrame:
    """
    Вычисляет "Time to invisibility" - время до момента, когда цена
    начинает значительно отклоняться от начальной цены Iceberg-ордера
    
    Args:
        impact_df: DataFrame с данными о влиянии на рынок
        iceberg_orders: DataFrame с идентифицированными Iceberg-ордерами
        window_sec: Окно наблюдения в секундах
    """
    results = []
    
    for _, order in iceberg_orders.iterrows():
        order_id = order['order_id']
        start_time = order['start_time']
        initial_price = order['price']
        side = order['side']
        
        # Получаем все fills для этого ордера
        order_fills = impact_df[impact_df['order_id'] == order_id].sort_values('timestamp')
        
        if len(order_fills) < 2:
            continue
        
        # Анализируем изменение mid_price_after после каждого fill
        price_deviations = []
        time_to_escape = None
        
        for idx, fill in order_fills.iterrows():
            if pd.notna(fill.get('mid_price_after')):
                deviation_bps = abs(
                    (fill['mid_price_after'] - initial_price) / initial_price * 10000
                )
                price_deviations.append(deviation_bps)
                
                # Считаем "побег" если отклонение > 10 bps
                if deviation_bps > 10.0 and time_to_escape is None:
                    time_elapsed = (fill['timestamp'] - start_time).total_seconds()
                    time_to_escape = time_elapsed
        
        results.append({
            'order_id': order_id,
            'total_qty': order['total_qty'],
            'fill_count': order['fill_count'],
            'duration_sec': order['duration_sec'],
            'time_to_invisibility': time_to_escape,
            'max_deviation_bps': max(price_deviations) if price_deviations else 0,
            'avg_deviation_bps': np.mean(price_deviations) if price_deviations else 0,
            'side': side
        })
    
    return pd.DataFrame(results)


def compare_with_aggressive(
    iceberg_stats: pd.DataFrame,
    trades_df: pd.DataFrame
) -> dict:
    """
    Сравнивает эффективность Iceberg-ордеров с агрессивным исполнением
    """
    # Фильтруем агрессивные ордера (предполагаем, что они имеют высокий slippage)
    aggressive_orders = trades_df[trades_df['slippage_bps'].abs() > 5.0]
    
    comparison = {
        'iceberg': {
            'count': len(iceberg_stats),
            'avg_duration': iceberg_stats['duration_sec'].mean(),
            'avg_time_to_invisibility': iceberg_stats['time_to_invisibility'].mean(),
            'avg_max_deviation': iceberg_stats['max_deviation_bps'].mean(),
            'avg_fills_per_order': iceberg_stats['fill_count'].mean(),
        },
        'aggressive': {
            'count': len(aggressive_orders),
            'avg_slippage': aggressive_orders['slippage_bps'].abs().mean(),
            'avg_latency': aggressive_orders['latency_ms'].mean(),
        }
    }
    
    return comparison


def plot_iceberg_analysis(
    iceberg_stats: pd.DataFrame,
    output_path: Path
):
    """Создает визуализации анализа Iceberg-ордеров"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Распределение Time to Invisibility
    ax = axes[0, 0]
    iceberg_stats['time_to_invisibility'].dropna().hist(bins=20, ax=ax, edgecolor='black')
    ax.set_xlabel('Time to Invisibility (seconds)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Time to Invisibility')
    ax.grid(True, alpha=0.3)
    
    # 2. Корреляция размера и времени невидимости
    ax = axes[0, 1]
    valid_data = iceberg_stats.dropna(subset=['time_to_invisibility'])
    ax.scatter(valid_data['total_qty'], valid_data['time_to_invisibility'], alpha=0.6)
    ax.set_xlabel('Total Order Size')
    ax.set_ylabel('Time to Invisibility (seconds)')
    ax.set_title('Order Size vs Time to Invisibility')
    ax.grid(True, alpha=0.3)
    
    # 3. Количество fills vs отклонение цены
    ax = axes[1, 0]
    ax.scatter(iceberg_stats['fill_count'], iceberg_stats['max_deviation_bps'], alpha=0.6)
    ax.set_xlabel('Number of Fills')
    ax.set_ylabel('Max Price Deviation (bps)')
    ax.set_title('Fills vs Price Impact')
    ax.grid(True, alpha=0.3)
    
    # 4. Длительность исполнения
    ax = axes[1, 1]
    iceberg_stats['duration_sec'].hist(bins=20, ax=ax, edgecolor='black')
    ax.set_xlabel('Execution Duration (seconds)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Execution Duration')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'iceberg_analysis.png', dpi=150)
    print(f"Visualization saved to {output_path / 'iceberg_analysis.png'}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Iceberg order effectiveness'
    )
    parser.add_argument(
        '--bot-path',
        type=Path,
        required=True,
        help='Path to bot directory (e.g., bots/BTCUSDT)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output directory for results (default: bot_path/analysis)'
    )
    
    args = parser.parse_args()
    
    # Определяем путь для вывода
    output_path = args.output or (args.bot_path / 'analysis')
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {args.bot_path}...")
    
    # Загружаем данные
    try:
        impact_df = load_market_impact_data(args.bot_path)
        trades_df = load_execution_data(args.bot_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    print(f"Loaded {len(impact_df)} market impact records")
    print(f"Loaded {len(trades_df)} trade records")
    
    # Идентифицируем Iceberg-ордера
    print("\nIdentifying Iceberg orders...")
    iceberg_orders = identify_iceberg_orders(impact_df)
    print(f"Found {len(iceberg_orders)} potential Iceberg orders")
    
    if len(iceberg_orders) == 0:
        print("No Iceberg orders found. Exiting.")
        return 0
    
    # Вычисляем Time to Invisibility
    print("\nCalculating Time to Invisibility...")
    iceberg_stats = calculate_time_to_invisibility(impact_df, iceberg_orders)
    
    # Сравниваем с агрессивным исполнением
    print("\nComparing with aggressive execution...")
    comparison = compare_with_aggressive(iceberg_stats, trades_df)
    
    # Выводим результаты
    print("\n" + "="*60)
    print("ICEBERG ORDER ANALYSIS RESULTS")
    print("="*60)
    
    print("\nIceberg Orders:")
    print(f"  Total orders: {comparison['iceberg']['count']}")
    print(f"  Avg duration: {comparison['iceberg']['avg_duration']:.2f} sec")
    print(f"  Avg time to invisibility: {comparison['iceberg']['avg_time_to_invisibility']:.2f} sec")
    print(f"  Avg max deviation: {comparison['iceberg']['avg_max_deviation']:.2f} bps")
    print(f"  Avg fills per order: {comparison['iceberg']['avg_fills_per_order']:.1f}")
    
    print("\nAggressive Orders:")
    print(f"  Total orders: {comparison['aggressive']['count']}")
    print(f"  Avg slippage: {comparison['aggressive']['avg_slippage']:.2f} bps")
    print(f"  Avg latency: {comparison['aggressive']['avg_latency']:.2f} ms")
    
    # Сохраняем детальные результаты
    iceberg_stats.to_csv(output_path / 'iceberg_stats.csv', index=False)
    print(f"\nDetailed statistics saved to {output_path / 'iceberg_stats.csv'}")
    
    # Создаем визуализации
    print("\nGenerating visualizations...")
    plot_iceberg_analysis(iceberg_stats, output_path)
    
    print("\nAnalysis complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
