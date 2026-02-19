#!/usr/bin/env python3
"""
Задача 209: Анализ экономии API лимитов при использовании массовых отмен (Mass Cancellation Optimization)

Скрипт подсчитывает экономию "веса" API запросов при использовании эндпоинта cancel-all
вместо множественных одиночных cancel запросов.

По спецификации Bybit V5 2026:
- Один cancel-all запрос тратит меньше лимитов, чем 5-10 одиночных cancel запросов
- Метрика: RateLimit_Cost_Savings

Использование:
    python api_limit_analysis.py --bot-path /path/to/bot --output-dir /path/to/output
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Optional, Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def load_execution_quality_csv(bot_path: Path) -> Optional[pd.DataFrame]:
    """Загружает execution_quality.csv из папки бота."""
    csv_path = bot_path / "logs" / "execution_quality.csv"
    
    if not csv_path.exists():
        print(f"⚠️  Файл не найден: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Загружено {len(df)} записей из execution_quality.csv")
        return df
    except Exception as e:
        print(f"❌ Ошибка при загрузке execution_quality.csv: {e}")
        return None


def analyze_mass_cancellation_savings(df: pd.DataFrame) -> Dict:
    """
    Анализирует экономию API лимитов при использовании cancel-all.
    
    По спецификации Bybit V5:
    - Один cancel-all запрос: ~1 вес
    - Один cancel запрос: ~1 вес
    - Но cancel-all обрабатывает все ордера за один запрос
    
    Экономия = (количество отмен - 1) * вес_одного_cancel - вес_cancel_all
    """
    
    results = {
        "total_cancellations": 0,
        "mass_cancellations": 0,
        "single_cancellations": 0,
        "estimated_weight_saved": 0.0,
        "cancellations_per_mass_event": [],
        "savings_by_event": [],
    }
    
    # Ищем события отмены в логах
    if "action" not in df.columns:
        print("⚠️  Колонка 'action' не найдена в execution_quality.csv")
        return results
    
    # Фильтруем события отмены
    cancel_events = df[df["action"].str.contains("CANCEL", case=False, na=False)]
    
    if len(cancel_events) == 0:
        print("⚠️  События отмены не найдены в логах")
        return results
    
    results["total_cancellations"] = len(cancel_events)
    
    # Ищем события MASS_CANCEL
    mass_cancel_events = cancel_events[cancel_events["action"].str.contains("MASS_CANCEL", case=False, na=False)]
    results["mass_cancellations"] = len(mass_cancel_events)
    results["single_cancellations"] = len(cancel_events) - len(mass_cancel_events)
    
    # Если есть информация о количестве отмен в каждом событии
    if "details" in df.columns:
        for idx, row in mass_cancel_events.iterrows():
            try:
                details = json.loads(row["details"]) if isinstance(row["details"], str) else row["details"]
                if isinstance(details, dict) and "cancelled_count" in details:
                    count = details["cancelled_count"]
                    results["cancellations_per_mass_event"].append(count)
                    # Экономия: (count - 1) * 1 - 1 = count - 2
                    # Но на практике cancel-all экономит примерно 50% от стоимости count отмен
                    savings = (count - 1) * 0.5
                    results["savings_by_event"].append(savings)
                    results["estimated_weight_saved"] += savings
            except (json.JSONDecodeError, TypeError):
                pass
    
    # Если нет информации о количестве отмен, используем среднее значение
    if not results["cancellations_per_mass_event"] and results["mass_cancellations"] > 0:
        avg_cancellations = results["single_cancellations"] / max(results["mass_cancellations"], 1)
        results["cancellations_per_mass_event"] = [avg_cancellations] * results["mass_cancellations"]
        results["estimated_weight_saved"] = results["mass_cancellations"] * (avg_cancellations - 1) * 0.5
    
    return results


def calculate_rate_limit_cost_savings(analysis: Dict) -> Dict:
    """
    Вычисляет метрику RateLimit_Cost_Savings.
    
    RateLimit_Cost_Savings = (estimated_weight_saved / total_weight_used) * 100%
    """
    
    # Предполагаем, что каждый cancel запрос стоит 1 вес
    total_weight_without_optimization = analysis["total_cancellations"]
    
    # С оптимизацией: single_cancellations + mass_cancellations (каждый mass_cancel = 1 вес)
    total_weight_with_optimization = analysis["single_cancellations"] + analysis["mass_cancellations"]
    
    # Экономия в процентах
    if total_weight_without_optimization > 0:
        savings_pct = ((total_weight_without_optimization - total_weight_with_optimization) / 
                       total_weight_without_optimization) * 100
    else:
        savings_pct = 0.0
    
    return {
        "RateLimit_Cost_Savings_Pct": savings_pct,
        "Total_Weight_Without_Optimization": total_weight_without_optimization,
        "Total_Weight_With_Optimization": total_weight_with_optimization,
        "Estimated_Weight_Saved": analysis["estimated_weight_saved"],
    }


def generate_report(bot_path: Path, analysis: Dict, savings: Dict, output_dir: Path) -> None:
    """Генерирует отчет об экономии API лимитов."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / "api_limit_analysis_report.txt"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("АНАЛИЗ ЭКОНОМИИ API ЛИМИТОВ (Задача 209)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Путь к боту: {bot_path}\n\n")
        
        f.write("СТАТИСТИКА ОТМЕН:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Всего событий отмены: {analysis['total_cancellations']}\n")
        f.write(f"Массовых отмен (cancel-all): {analysis['mass_cancellations']}\n")
        f.write(f"Одиночных отмен (cancel): {analysis['single_cancellations']}\n\n")
        
        if analysis['cancellations_per_mass_event']:
            avg_per_event = np.mean(analysis['cancellations_per_mass_event'])
            max_per_event = np.max(analysis['cancellations_per_mass_event'])
            min_per_event = np.min(analysis['cancellations_per_mass_event'])
            
            f.write("СТАТИСТИКА МАССОВЫХ ОТМЕН:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Среднее ордеров на одну массовую отмену: {avg_per_event:.2f}\n")
            f.write(f"Максимум ордеров в одной отмене: {max_per_event:.0f}\n")
            f.write(f"Минимум ордеров в одной отмене: {min_per_event:.0f}\n\n")
        
        f.write("ЭКОНОМИЯ API ЛИМИТОВ:\n")
        f.write("-" * 80 + "\n")
        f.write(f"RateLimit_Cost_Savings: {savings['RateLimit_Cost_Savings_Pct']:.2f}%\n")
        f.write(f"Вес без оптимизации: {savings['Total_Weight_Without_Optimization']}\n")
        f.write(f"Вес с оптимизацией: {savings['Total_Weight_With_Optimization']}\n")
        f.write(f"Сэкономлено веса: {savings['Estimated_Weight_Saved']:.2f}\n\n")
        
        f.write("ВЫВОДЫ:\n")
        f.write("-" * 80 + "\n")
        if savings['RateLimit_Cost_Savings_Pct'] > 0:
            f.write(f"✓ Оптимизация массовых отмен сэкономила {savings['RateLimit_Cost_Savings_Pct']:.2f}% API лимитов\n")
        else:
            f.write("⚠️  Оптимизация массовых отмен не применялась или не дала результатов\n")
    
    print(f"✓ Отчет сохранен: {report_path}")


def plot_cancellation_analysis(analysis: Dict, output_dir: Path) -> None:
    """Создает графики анализа отмен."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # График 1: Распределение типов отмен
    cancel_types = [analysis['single_cancellations'], analysis['mass_cancellations']]
    labels = ['Одиночные отмены', 'Массовые отмены']
    colors = ['#FF6B6B', '#4ECDC4']
    
    axes[0].pie(cancel_types, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[0].set_title('Распределение типов отмен')
    
    # График 2: Ордеров на одну массовую отмену
    if analysis['cancellations_per_mass_event']:
        axes[1].hist(analysis['cancellations_per_mass_event'], bins=20, color='#4ECDC4', edgecolor='black')
        axes[1].set_xlabel('Количество ордеров')
        axes[1].set_ylabel('Частота')
        axes[1].set_title('Распределение ордеров на одну массовую отмену')
        axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / "cancellation_analysis.png"
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    print(f"✓ График сохранен: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Анализ экономии API лимитов при использовании массовых отмен"
    )
    parser.add_argument("--bot-path", type=Path, required=True, help="Путь к папке бота")
    parser.add_argument("--output-dir", type=Path, default=None, help="Папка для сохранения результатов")
    
    args = parser.parse_args()
    
    bot_path = args.bot_path
    if not bot_path.exists():
        print(f"❌ Папка бота не найдена: {bot_path}")
        return
    
    output_dir = args.output_dir or bot_path / "analysis"
    
    print(f"📊 Анализ экономии API лимитов для бота: {bot_path.name}")
    print()
    
    # Загружаем данные
    df = load_execution_quality_csv(bot_path)
    if df is None:
        print("❌ Не удалось загрузить данные для анализа")
        return
    
    # Анализируем массовые отмены
    analysis = analyze_mass_cancellation_savings(df)
    
    # Вычисляем экономию лимитов
    savings = calculate_rate_limit_cost_savings(analysis)
    
    # Генерируем отчет
    generate_report(bot_path, analysis, savings, output_dir)
    
    # Создаем графики
    if analysis['total_cancellations'] > 0:
        plot_cancellation_analysis(analysis, output_dir)
    
    print()
    print("=" * 80)
    print(f"RateLimit_Cost_Savings: {savings['RateLimit_Cost_Savings_Pct']:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
