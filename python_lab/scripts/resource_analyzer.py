#!/usr/bin/env python3
"""
Задача 225: Анализатор корреляций системных ресурсов и задержек исполнения

Скрипт анализирует корреляцию между:
- Системной нагрузкой (CPU, RAM)
- internal_latency_us из логов исполнения

Использование:
    python resource_analyzer.py --log-dir ./bots/BTCUSDT/logs --output report.html
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка стиля графиков
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def parse_log_file(log_path: Path) -> pd.DataFrame:
    """
    Парсит лог-файл и извлекает метрики ресурсов и задержки
    
    Args:
        log_path: Путь к лог-файлу
        
    Returns:
        DataFrame с колонками: timestamp, cpu_usage_pct, memory_rss_kb, internal_latency_us
    """
    records = []
    
    # Паттерны для извлечения данных
    metrics_pattern = re.compile(
        r'System metrics: CPU=([\d.]+)%, MEM=(\d+)KB'
    )
    latency_pattern = re.compile(
        r'internal_latency_us["\s:=]+(\d+)'
    )
    timestamp_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)'
    )
    
    current_metrics = {}
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Извлекаем timestamp
            ts_match = timestamp_pattern.search(line)
            if not ts_match:
                continue
                
            timestamp = pd.to_datetime(ts_match.group(1))
            
            # Извлекаем метрики ресурсов
            metrics_match = metrics_pattern.search(line)
            if metrics_match:
                current_metrics = {
                    'timestamp': timestamp,
                    'cpu_usage_pct': float(metrics_match.group(1)),
                    'memory_rss_kb': int(metrics_match.group(2)),
                }
                continue
            
            # Извлекаем задержки
            latency_match = latency_pattern.search(line)
            if latency_match and current_metrics:
                record = current_metrics.copy()
                record['internal_latency_us'] = int(latency_match.group(1))
                records.append(record)
    
    if not records:
        raise ValueError(f"No data found in {log_path}")
    
    df = pd.DataFrame(records)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df


def calculate_correlations(df: pd.DataFrame) -> Dict[str, float]:
    """
    Вычисляет корреляции между ресурсами и задержками
    
    Args:
        df: DataFrame с метриками
        
    Returns:
        Словарь с коэффициентами корреляции
    """
    correlations = {}
    
    # Корреляция Пирсона (линейная зависимость)
    correlations['cpu_latency_pearson'] = stats.pearsonr(
        df['cpu_usage_pct'], 
        df['internal_latency_us']
    )[0]
    
    correlations['memory_latency_pearson'] = stats.pearsonr(
        df['memory_rss_kb'], 
        df['internal_latency_us']
    )[0]
    
    # Корреляция Спирмена (монотонная зависимость)
    correlations['cpu_latency_spearman'] = stats.spearmanr(
        df['cpu_usage_pct'], 
        df['internal_latency_us']
    )[0]
    
    correlations['memory_latency_spearman'] = stats.spearmanr(
        df['memory_rss_kb'], 
        df['internal_latency_us']
    )[0]
    
    return correlations


def plot_correlations(df: pd.DataFrame, output_path: Path):
    """
    Создает визуализации корреляций
    
    Args:
        df: DataFrame с метриками
        output_path: Путь для сохранения графиков
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. CPU vs Latency scatter
    axes[0, 0].scatter(df['cpu_usage_pct'], df['internal_latency_us'], alpha=0.5)
    axes[0, 0].set_xlabel('CPU Usage (%)')
    axes[0, 0].set_ylabel('Internal Latency (μs)')
    axes[0, 0].set_title('CPU Usage vs Internal Latency')
    
    # Добавляем линию тренда
    z = np.polyfit(df['cpu_usage_pct'], df['internal_latency_us'], 1)
    p = np.poly1d(z)
    axes[0, 0].plot(df['cpu_usage_pct'], p(df['cpu_usage_pct']), "r--", alpha=0.8)
    
    # 2. Memory vs Latency scatter
    axes[0, 1].scatter(df['memory_rss_kb'] / 1024, df['internal_latency_us'], alpha=0.5)
    axes[0, 1].set_xlabel('Memory RSS (MB)')
    axes[0, 1].set_ylabel('Internal Latency (μs)')
    axes[0, 1].set_title('Memory Usage vs Internal Latency')
    
    # 3. Time series - CPU and Latency
    ax1 = axes[1, 0]
    ax2 = ax1.twinx()
    
    ax1.plot(df['timestamp'], df['cpu_usage_pct'], 'b-', label='CPU %')
    ax2.plot(df['timestamp'], df['internal_latency_us'], 'r-', label='Latency μs')
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('CPU Usage (%)', color='b')
    ax2.set_ylabel('Internal Latency (μs)', color='r')
    ax1.set_title('CPU and Latency Over Time')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # 4. Correlation heatmap
    corr_matrix = df[['cpu_usage_pct', 'memory_rss_kb', 'internal_latency_us']].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, ax=axes[1, 1])
    axes[1, 1].set_title('Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plots saved to {output_path}")


def generate_report(df: pd.DataFrame, correlations: Dict[str, float], output_path: Path):
    """
    Генерирует HTML отчет с анализом
    
    Args:
        df: DataFrame с метриками
        correlations: Словарь с корреляциями
        output_path: Путь для сохранения отчета
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Resource Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            .metric {{ font-weight: bold; }}
            .good {{ color: green; }}
            .warning {{ color: orange; }}
            .bad {{ color: red; }}
        </style>
    </head>
    <body>
        <h1>System Resource Analysis Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Summary Statistics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Mean</th>
                <th>Std Dev</th>
                <th>Min</th>
                <th>Max</th>
            </tr>
            <tr>
                <td class="metric">CPU Usage (%)</td>
                <td>{df['cpu_usage_pct'].mean():.2f}</td>
                <td>{df['cpu_usage_pct'].std():.2f}</td>
                <td>{df['cpu_usage_pct'].min():.2f}</td>
                <td>{df['cpu_usage_pct'].max():.2f}</td>
            </tr>
            <tr>
                <td class="metric">Memory RSS (MB)</td>
                <td>{df['memory_rss_kb'].mean() / 1024:.2f}</td>
                <td>{df['memory_rss_kb'].std() / 1024:.2f}</td>
                <td>{df['memory_rss_kb'].min() / 1024:.2f}</td>
                <td>{df['memory_rss_kb'].max() / 1024:.2f}</td>
            </tr>
            <tr>
                <td class="metric">Internal Latency (μs)</td>
                <td>{df['internal_latency_us'].mean():.2f}</td>
                <td>{df['internal_latency_us'].std():.2f}</td>
                <td>{df['internal_latency_us'].min():.2f}</td>
                <td>{df['internal_latency_us'].max():.2f}</td>
            </tr>
        </table>
        
        <h2>Correlation Analysis</h2>
        <table>
            <tr>
                <th>Correlation Type</th>
                <th>Coefficient</th>
                <th>Interpretation</th>
            </tr>
            <tr>
                <td>CPU vs Latency (Pearson)</td>
                <td>{correlations['cpu_latency_pearson']:.4f}</td>
                <td class="{'good' if abs(correlations['cpu_latency_pearson']) < 0.3 else 'warning' if abs(correlations['cpu_latency_pearson']) < 0.7 else 'bad'}">
                    {'Weak' if abs(correlations['cpu_latency_pearson']) < 0.3 else 'Moderate' if abs(correlations['cpu_latency_pearson']) < 0.7 else 'Strong'}
                </td>
            </tr>
            <tr>
                <td>Memory vs Latency (Pearson)</td>
                <td>{correlations['memory_latency_pearson']:.4f}</td>
                <td class="{'good' if abs(correlations['memory_latency_pearson']) < 0.3 else 'warning' if abs(correlations['memory_latency_pearson']) < 0.7 else 'bad'}">
                    {'Weak' if abs(correlations['memory_latency_pearson']) < 0.3 else 'Moderate' if abs(correlations['memory_latency_pearson']) < 0.7 else 'Strong'}
                </td>
            </tr>
            <tr>
                <td>CPU vs Latency (Spearman)</td>
                <td>{correlations['cpu_latency_spearman']:.4f}</td>
                <td class="{'good' if abs(correlations['cpu_latency_spearman']) < 0.3 else 'warning' if abs(correlations['cpu_latency_spearman']) < 0.7 else 'bad'}">
                    {'Weak' if abs(correlations['cpu_latency_spearman']) < 0.3 else 'Moderate' if abs(correlations['cpu_latency_spearman']) < 0.7 else 'Strong'}
                </td>
            </tr>
            <tr>
                <td>Memory vs Latency (Spearman)</td>
                <td>{correlations['memory_latency_spearman']:.4f}</td>
                <td class="{'good' if abs(correlations['memory_latency_spearman']) < 0.3 else 'warning' if abs(correlations['memory_latency_spearman']) < 0.7 else 'bad'}">
                    {'Weak' if abs(correlations['memory_latency_spearman']) < 0.3 else 'Moderate' if abs(correlations['memory_latency_spearman']) < 0.7 else 'Strong'}
                </td>
            </tr>
        </table>
        
        <h2>Recommendations</h2>
        <ul>
    """
    
    # Добавляем рекомендации на основе корреляций
    if abs(correlations['cpu_latency_pearson']) > 0.7:
        html += "<li class='bad'>⚠️ Strong correlation between CPU usage and latency detected. Consider optimizing CPU-intensive operations.</li>"
    
    if abs(correlations['memory_latency_pearson']) > 0.7:
        html += "<li class='bad'>⚠️ Strong correlation between memory usage and latency detected. Check for memory leaks or excessive allocations.</li>"
    
    if df['cpu_usage_pct'].max() > 80:
        html += f"<li class='warning'>⚠️ Peak CPU usage reached {df['cpu_usage_pct'].max():.1f}%. Consider increasing CPU resources.</li>"
    
    if df['internal_latency_us'].mean() > 100000:  # 100ms
        html += f"<li class='warning'>⚠️ Average latency is {df['internal_latency_us'].mean() / 1000:.1f}ms. This may impact trading performance.</li>"
    
    html += """
        </ul>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze correlation between system resources and execution latency'
    )
    parser.add_argument(
        '--log-dir',
        type=Path,
        required=True,
        help='Directory containing log files'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('resource_analysis_report.html'),
        help='Output HTML report path'
    )
    parser.add_argument(
        '--plot',
        type=Path,
        default=Path('resource_analysis_plots.png'),
        help='Output plot image path'
    )
    
    args = parser.parse_args()
    
    # Находим лог-файлы
    log_files = list(args.log_dir.glob('*.log'))
    if not log_files:
        print(f"No log files found in {args.log_dir}")
        return
    
    print(f"Found {len(log_files)} log files")
    
    # Парсим все лог-файлы
    all_data = []
    for log_file in log_files:
        try:
            df = parse_log_file(log_file)
            all_data.append(df)
            print(f"Parsed {log_file.name}: {len(df)} records")
        except Exception as e:
            print(f"Error parsing {log_file.name}: {e}")
    
    if not all_data:
        print("No data could be parsed from log files")
        return
    
    # Объединяем все данные
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"\nTotal records: {len(combined_df)}")
    print(f"Time range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    
    # Вычисляем корреляции
    correlations = calculate_correlations(combined_df)
    
    print("\nCorrelation Analysis:")
    print(f"  CPU vs Latency (Pearson):  {correlations['cpu_latency_pearson']:.4f}")
    print(f"  CPU vs Latency (Spearman): {correlations['cpu_latency_spearman']:.4f}")
    print(f"  Memory vs Latency (Pearson):  {correlations['memory_latency_pearson']:.4f}")
    print(f"  Memory vs Latency (Spearman): {correlations['memory_latency_spearman']:.4f}")
    
    # Создаем визуализации
    plot_correlations(combined_df, args.plot)
    
    # Генерируем отчет
    generate_report(combined_df, correlations, args.output)
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
