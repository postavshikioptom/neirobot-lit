#!/usr/bin/env python3
"""
Задача 204: Анализ влияния сделок на Mid-Price (Trade Impact on Mid-Price)

Скрипт анализирует логи market_impact.csv и order_context.csv для расчета:
1. Weighted Average Impact - средневзвешенное смещение цены на один order_id
2. Нормализацию по объему уровня (level_total_vol)
3. Построение кривой: impact_bps как функция от order_size / level_total_vol

Методология:
- Для каждого order_id рассчитываем: sum(fill_size * (mid_at_fill - mid_before)) / total_size
- Выполняем JOIN с order_context.csv по order_id для получения level_total_vol
- Строим кривую зависимости impact_bps от participation_ratio (order_size / level_total_vol)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_market_impact_logs(bot_path: Path) -> Optional[pd.DataFrame]:
    """
    Загружает логи market_impact.csv
    
    Колонки: timestamp, order_id, fill_id, side, fill_size, mid_before, mid_at_fill
    """
    csv_path = bot_path / "logs" / "market_impact.csv"
    
    if not csv_path.exists():
        logger.warning(f"market_impact.csv не найден: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Загружено {len(df)} записей из market_impact.csv")
        return df
    except Exception as e:
        logger.error(f"Ошибка при загрузке market_impact.csv: {e}")
        return None


def load_order_context_logs(bot_path: Path) -> Optional[pd.DataFrame]:
    """
    Загружает логи order_context.csv (Задача 203)
    
    Колонки: timestamp, order_id, level_total_vol, imbalance_5l, order_size, fill_duration_us
    """
    csv_path = bot_path / "logs" / "order_context.csv"
    
    if not csv_path.exists():
        logger.warning(f"order_context.csv не найден: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Загружено {len(df)} записей из order_context.csv")
        return df
    except Exception as e:
        logger.error(f"Ошибка при загрузке order_context.csv: {e}")
        return None


def calculate_weighted_average_impact(impact_df: pd.DataFrame) -> pd.DataFrame:
    """
    Рассчитывает weighted average impact для каждого order_id
    
    Формула: sum(fill_size * (mid_at_fill - mid_before)) / total_size
    
    Возвращает DataFrame с колонками:
    - order_id
    - total_fill_size
    - weighted_impact_price (абсолютное смещение цены)
    - weighted_impact_bps (смещение в базисных пунктах)
    - num_fills (количество частичных исполнений)
    """
    
    # Рассчитываем impact для каждого fill
    impact_df['impact_price'] = impact_df['fill_size'] * (impact_df['mid_at_fill'] - impact_df['mid_before'])
    
    # Группируем по order_id
    grouped = impact_df.groupby('order_id').agg({
        'fill_size': 'sum',
        'impact_price': 'sum',
        'mid_before': 'first',  # Берем первый mid_before как базовую цену
        'fill_id': 'count',  # Количество fills
    }).reset_index()
    
    grouped.columns = ['order_id', 'total_fill_size', 'total_impact_price', 'mid_before', 'num_fills']
    
    # Рассчитываем weighted average impact
    grouped['weighted_impact_price'] = grouped['total_impact_price'] / grouped['total_fill_size']
    
    # Рассчитываем в базисных пунктах (bps)
    # 1 bps = 0.01% = 0.0001 в абсолютном значении
    grouped['weighted_impact_bps'] = (grouped['weighted_impact_price'] / grouped['mid_before']) * 10000
    
    return grouped[['order_id', 'total_fill_size', 'weighted_impact_price', 'weighted_impact_bps', 'num_fills']]


def normalize_by_level_volume(impact_summary: pd.DataFrame, context_df: pd.DataFrame) -> pd.DataFrame:
    """
    Выполняет JOIN с order_context.csv для получения level_total_vol
    и рассчитывает participation_ratio (order_size / level_total_vol)
    """
    
    # Берем первую запись для каждого order_id из context_df
    context_first = context_df.drop_duplicates(subset=['order_id'], keep='first')
    
    # Выполняем LEFT JOIN
    result = impact_summary.merge(
        context_first[['order_id', 'level_total_vol', 'order_size', 'imbalance_5l']],
        on='order_id',
        how='left'
    )
    
    # Рассчитываем participation_ratio
    result['participation_ratio'] = result['order_size'] / result['level_total_vol']
    
    # Фильтруем некорректные значения
    result = result[result['level_total_vol'] > 0]
    result = result[result['participation_ratio'] > 0]
    
    return result


def analyze_impact_curve(normalized_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Строит кривую зависимости impact_bps от participation_ratio
    
    Возвращает кортеж (participation_ratios, impact_bps_values)
    """
    
    # Сортируем по participation_ratio
    sorted_df = normalized_df.sort_values('participation_ratio')
    
    # Удаляем NaN значения
    sorted_df = sorted_df.dropna(subset=['participation_ratio', 'weighted_impact_bps'])
    
    return sorted_df['participation_ratio'].values, sorted_df['weighted_impact_bps'].values


def plot_impact_curve(participation_ratios: np.ndarray, impact_bps: np.ndarray, output_path: Path):
    """
    Строит график зависимости impact_bps от participation_ratio
    """
    
    plt.figure(figsize=(12, 7))
    
    # Основной график
    plt.scatter(participation_ratios, impact_bps, alpha=0.6, s=50, label='Observed Impact')
    
    # Добавляем линию тренда если достаточно точек
    if len(participation_ratios) > 2:
        z = np.polyfit(participation_ratios, impact_bps, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(participation_ratios.min(), participation_ratios.max(), 100)
        plt.plot(x_smooth, p(x_smooth), "r--", alpha=0.8, label='Trend (2nd order)')
    
    plt.xlabel('Participation Ratio (order_size / level_total_vol)', fontsize=12)
    plt.ylabel('Weighted Average Impact (bps)', fontsize=12)
    plt.title('Trade Impact on Mid-Price (Задача 204)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"График сохранен: {output_path}")
    plt.close()


def generate_report(normalized_df: pd.DataFrame, output_path: Path):
    """
    Генерирует текстовый отчет с статистикой
    """
    
    report = []
    report.append("=" * 80)
    report.append("АНАЛИЗ ВЛИЯНИЯ СДЕЛОК НА MID-PRICE (Задача 204)")
    report.append("=" * 80)
    report.append("")
    
    # Общая статистика
    report.append("ОБЩАЯ СТАТИСТИКА:")
    report.append(f"  Всего ордеров: {len(normalized_df)}")
    report.append(f"  Средний weighted impact: {normalized_df['weighted_impact_bps'].mean():.4f} bps")
    report.append(f"  Медианный weighted impact: {normalized_df['weighted_impact_bps'].median():.4f} bps")
    report.append(f"  Std Dev: {normalized_df['weighted_impact_bps'].std():.4f} bps")
    report.append(f"  Min impact: {normalized_df['weighted_impact_bps'].min():.4f} bps")
    report.append(f"  Max impact: {normalized_df['weighted_impact_bps'].max():.4f} bps")
    report.append("")
    
    # Статистика по participation_ratio
    report.append("СТАТИСТИКА ПО PARTICIPATION RATIO:")
    report.append(f"  Средний participation ratio: {normalized_df['participation_ratio'].mean():.6f}")
    report.append(f"  Медианный participation ratio: {normalized_df['participation_ratio'].median():.6f}")
    report.append(f"  Min ratio: {normalized_df['participation_ratio'].min():.6f}")
    report.append(f"  Max ratio: {normalized_df['participation_ratio'].max():.6f}")
    report.append("")
    
    # Статистика по количеству fills
    report.append("СТАТИСТИКА ПО КОЛИЧЕСТВУ FILLS:")
    report.append(f"  Средний num_fills: {normalized_df['num_fills'].mean():.2f}")
    report.append(f"  Медианный num_fills: {normalized_df['num_fills'].median():.0f}")
    report.append(f"  Max num_fills: {normalized_df['num_fills'].max():.0f}")
    report.append("")
    
    # Анализ по квартилям participation_ratio
    report.append("АНАЛИЗ ПО КВАРТИЛЯМ PARTICIPATION RATIO:")
    quartiles = pd.qcut(normalized_df['participation_ratio'], q=4, duplicates='drop')
    for i, (name, group) in enumerate(normalized_df.groupby(quartiles)):
        report.append(f"  Квартиль {i+1} (ratio: {name.left:.6f} - {name.right:.6f}):")
        report.append(f"    Кол-во ордеров: {len(group)}")
        report.append(f"    Средний impact: {group['weighted_impact_bps'].mean():.4f} bps")
        report.append(f"    Медианный impact: {group['weighted_impact_bps'].median():.4f} bps")
    report.append("")
    
    # Ожидаемый результат
    report.append("ОЖИДАЕМЫЙ РЕЗУЛЬТАТ (из задачи):")
    report.append('  "Сделка размером 5% от объема уровня (Top level) вызывает')
    report.append('   мгновенное смещение цены на 0.8 bps, которое восстанавливается')
    report.append('   на 50% через 100мс"')
    report.append("")
    
    report_text = "\n".join(report)
    
    # Выводим в консоль
    print(report_text)
    
    # Сохраняем в файл
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    logger.info(f"Отчет сохранен: {output_path}")


def main(bot_path: Optional[str] = None):
    """
    Основная функция анализа
    """
    
    if bot_path is None:
        # Используем текущую директорию
        bot_path = Path.cwd()
    else:
        bot_path = Path(bot_path)
    
    logger.info(f"Анализ влияния на цену для: {bot_path}")
    
    # Загружаем логи
    impact_df = load_market_impact_logs(bot_path)
    if impact_df is None or len(impact_df) == 0:
        logger.error("Не удалось загрузить market_impact.csv или он пуст")
        return
    
    context_df = load_order_context_logs(bot_path)
    if context_df is None or len(context_df) == 0:
        logger.error("Не удалось загрузить order_context.csv или он пуст")
        return
    
    # Рассчитываем weighted average impact
    logger.info("Расчет weighted average impact...")
    impact_summary = calculate_weighted_average_impact(impact_df)
    logger.info(f"Рассчитано {len(impact_summary)} ордеров")
    
    # Нормализуем по объему уровня
    logger.info("Нормализация по level_total_vol...")
    normalized_df = normalize_by_level_volume(impact_summary, context_df)
    logger.info(f"После нормализации: {len(normalized_df)} ордеров")
    
    if len(normalized_df) == 0:
        logger.error("Нет данных после нормализации")
        return
    
    # Анализируем кривую
    logger.info("Анализ кривой impact_bps vs participation_ratio...")
    participation_ratios, impact_bps = analyze_impact_curve(normalized_df)
    
    # Сохраняем результаты
    output_dir = bot_path / "analysis"
    output_dir.mkdir(exist_ok=True)
    
    # График
    plot_impact_curve(participation_ratios, impact_bps, output_dir / "market_impact_curve.png")
    
    # Отчет
    generate_report(normalized_df, output_dir / "market_impact_report.txt")
    
    # Сохраняем детальные результаты в CSV
    normalized_df.to_csv(output_dir / "market_impact_analysis.csv", index=False)
    logger.info(f"Детальные результаты сохранены: {output_dir / 'market_impact_analysis.csv'}")
    
    logger.info("Анализ завершен успешно!")


if __name__ == "__main__":
    bot_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(bot_path)
