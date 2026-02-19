#!/usr/bin/env python3
"""
Задача 202: Анализ качества исполнения в зависимости от задержки (Execution Quality by Latency)

Скрипт анализирует влияние задержки (latency) на вероятность исполнения (Fill Rate)
и определяет "точку деградации" стратегии.

Использование:
    python execution_analysis.py --bot-path /path/to/bot --output-dir /path/to/output
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Optional, Tuple, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns


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


def load_trades_csv(bot_path: Path) -> Optional[pd.DataFrame]:
    """Загружает trades.csv из папки бота."""
    csv_path = bot_path / "logs" / "trades.csv"
    
    if not csv_path.exists():
        print(f"⚠️  Файл не найден: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Загружено {len(df)} записей из trades.csv")
        return df
    except Exception as e:
        print(f"❌ Ошибка при загрузке trades.csv: {e}")
        return None


def merge_execution_and_trades(
    execution_df: pd.DataFrame,
    trades_df: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """
    Выполняет JOIN execution_quality.csv с trades.csv по order_id.
    Если trades.csv не существует, возвращает только execution_quality.
    """
    if trades_df is None or trades_df.empty:
        print("⚠️  trades.csv не найден или пуст, используем только execution_quality.csv")
        return execution_df
    
    try:
        # Выполняем LEFT JOIN по order_id
        merged = execution_df.merge(
            trades_df,
            on="order_id",
            how="left",
            suffixes=("_exec", "_trade")
        )
        print(f"✓ Выполнен JOIN: {len(merged)} записей")
        return merged
    except Exception as e:
        print(f"⚠️  Ошибка при JOIN: {e}, используем только execution_quality.csv")
        return execution_df


def create_latency_buckets(
    df: pd.DataFrame,
    bucket_size_us: int = 5000  # 5мс по умолчанию
) -> pd.DataFrame:
    """
    Группирует данные по бакетам задержки.
    bucket_size_us: размер бакета в микросекундах (по умолчанию 5мс = 5000мкс)
    """
    # Используем сетевую задержку (network_lat_us) как основной показатель
    df["network_lat_ms"] = df["network_lat_us"] / 1000.0
    df["latency_bucket_ms"] = (df["network_lat_us"] // bucket_size_us * bucket_size_us / 1000.0).astype(int)
    
    return df


def calculate_fill_rate_correlation(df: pd.DataFrame) -> Dict[str, any]:
    """
    Рассчитывает Fill Rate Correlation: как вероятность исполнения падает с ростом задержки.
    """
    # Группируем по бакетам задержки
    grouped = df.groupby("latency_bucket_ms").agg({
        "fill_rate": ["mean", "std", "count"],
        "network_lat_us": "mean",
        "internal_lat_us": "mean",
        "is_cancelled": "sum"
    }).round(4)
    
    grouped.columns = ["fill_rate_mean", "fill_rate_std", "order_count", 
                       "avg_network_lat_us", "avg_internal_lat_us", "cancelled_count"]
    grouped = grouped.reset_index()
    
    # Рассчитываем процент отмен
    grouped["cancel_rate"] = (grouped["cancelled_count"] / grouped["order_count"] * 100).round(2)
    
    # Рассчитываем корреляцию между задержкой и Fill Rate
    correlation = df["network_lat_us"].corr(df["fill_rate"])
    
    return {
        "grouped_stats": grouped,
        "correlation": correlation,
        "total_orders": len(df),
        "avg_fill_rate": df["fill_rate"].mean(),
        "avg_network_latency_us": df["network_lat_us"].mean(),
        "avg_internal_latency_us": df["internal_lat_us"].mean(),
    }


def find_degradation_point(stats: Dict[str, any], threshold: float = 0.85) -> Optional[Dict]:
    """
    Определяет "точку деградации" - уровень задержки, при котором Fill Rate падает ниже threshold.
    """
    grouped = stats["grouped_stats"]
    
    # Находим первый бакет, где Fill Rate ниже threshold
    degradation = grouped[grouped["fill_rate_mean"] < threshold]
    
    if degradation.empty:
        return None
    
    first_degradation = degradation.iloc[0]
    
    return {
        "latency_bucket_ms": int(first_degradation["latency_bucket_ms"]),
        "fill_rate": float(first_degradation["fill_rate_mean"]),
        "order_count": int(first_degradation["order_count"]),
        "cancel_rate": float(first_degradation["cancel_rate"]),
    }


def generate_report(
    stats: Dict[str, any],
    degradation_point: Optional[Dict],
    output_dir: Path
) -> str:
    """Генерирует текстовый отчёт анализа."""
    report = []
    report.append("=" * 80)
    report.append("АНАЛИЗ КАЧЕСТВА ИСПОЛНЕНИЯ В ЗАВИСИМОСТИ ОТ ЗАДЕРЖКИ (Задача 202)")
    report.append("=" * 80)
    report.append("")
    
    # Общая статистика
    report.append("📊 ОБЩАЯ СТАТИСТИКА")
    report.append("-" * 80)
    report.append(f"Всего ордеров: {stats['total_orders']}")
    report.append(f"Средний Fill Rate: {stats['avg_fill_rate']:.4f} ({stats['avg_fill_rate']*100:.2f}%)")
    report.append(f"Средняя сетевая задержка: {stats['avg_network_latency_us']:.0f} мкс ({stats['avg_network_latency_us']/1000:.2f} мс)")
    report.append(f"Средняя внутренняя задержка: {stats['avg_internal_latency_us']:.0f} мкс ({stats['avg_internal_latency_us']/1000:.2f} мс)")
    report.append(f"Корреляция (задержка vs Fill Rate): {stats['correlation']:.4f}")
    report.append("")
    
    # Статистика по бакетам
    report.append("📈 СТАТИСТИКА ПО БАКЕТАМ ЗАДЕРЖКИ")
    report.append("-" * 80)
    grouped = stats["grouped_stats"]
    for _, row in grouped.iterrows():
        report.append(
            f"Задержка {int(row['latency_bucket_ms']):4d}мс: "
            f"Fill Rate={row['fill_rate_mean']:.4f}, "
            f"Ордеров={int(row['order_count']):4d}, "
            f"Отмен={row['cancel_rate']:.1f}%"
        )
    report.append("")
    
    # Точка деградации
    if degradation_point:
        report.append("⚠️  ТОЧКА ДЕГРАДАЦИИ (Fill Rate < 85%)")
        report.append("-" * 80)
        report.append(f"Задержка: {degradation_point['latency_bucket_ms']} мс")
        report.append(f"Fill Rate: {degradation_point['fill_rate']:.4f} ({degradation_point['fill_rate']*100:.2f}%)")
        report.append(f"Количество ордеров: {degradation_point['order_count']}")
        report.append(f"Процент отмен: {degradation_point['cancel_rate']:.2f}%")
        report.append("")
        report.append("💡 ВЫВОД: При росте задержки выше этого уровня стратегия становится убыточной.")
    else:
        report.append("✓ ТОЧКА ДЕГРАДАЦИИ НЕ НАЙДЕНА")
        report.append("Fill Rate остаётся выше 85% для всех уровней задержки.")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)


def plot_fill_rate_vs_latency(stats: Dict[str, any], output_dir: Path) -> None:
    """Создаёт график зависимости Fill Rate от задержки."""
    grouped = stats["grouped_stats"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # График 1: Fill Rate vs Latency
    ax1.plot(grouped["latency_bucket_ms"], grouped["fill_rate_mean"], 
             marker="o", linewidth=2, markersize=8, color="blue")
    ax1.fill_between(grouped["latency_bucket_ms"], 
                      grouped["fill_rate_mean"] - grouped["fill_rate_std"],
                      grouped["fill_rate_mean"] + grouped["fill_rate_std"],
                      alpha=0.2, color="blue")
    ax1.axhline(y=0.85, color="red", linestyle="--", label="Порог деградации (85%)")
    ax1.set_xlabel("Задержка (мс)")
    ax1.set_ylabel("Fill Rate")
    ax1.set_title("Зависимость Fill Rate от сетевой задержки")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # График 2: Cancel Rate vs Latency
    ax2.bar(grouped["latency_bucket_ms"], grouped["cancel_rate"], 
            color="orange", alpha=0.7, width=2)
    ax2.set_xlabel("Задержка (мс)")
    ax2.set_ylabel("Процент отмен (%)")
    ax2.set_title("Зависимость процента отмен от задержки")
    ax2.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    output_path = output_dir / "execution_quality_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ График сохранён: {output_path}")
    plt.close()


def save_detailed_csv(stats: Dict[str, any], output_dir: Path) -> None:
    """Сохраняет детальную статистику в CSV."""
    grouped = stats["grouped_stats"]
    output_path = output_dir / "execution_quality_by_latency.csv"
    grouped.to_csv(output_path, index=False)
    print(f"✓ Детальная статистика сохранена: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Анализ качества исполнения в зависимости от задержки (Задача 202)"
    )
    parser.add_argument("--bot-path", type=Path, required=True,
                        help="Путь к папке бота (содержит logs/execution_quality.csv)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Директория для сохранения результатов (по умолчанию: bot_path/analysis)")
    parser.add_argument("--bucket-size-ms", type=int, default=5,
                        help="Размер бакета задержки в миллисекундах (по умолчанию: 5мс)")
    parser.add_argument("--degradation-threshold", type=float, default=0.85,
                        help="Порог Fill Rate для определения деградации (по умолчанию: 0.85)")
    
    args = parser.parse_args()
    
    # Проверяем пути
    bot_path = args.bot_path.resolve()
    if not bot_path.exists():
        print(f"❌ Ошибка: папка бота не найдена: {bot_path}")
        return
    
    output_dir = args.output_dir or bot_path / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Папка бота: {bot_path}")
    print(f"📁 Директория результатов: {output_dir}")
    print()
    
    # Загружаем данные
    execution_df = load_execution_quality_csv(bot_path)
    if execution_df is None or execution_df.empty:
        print("❌ Ошибка: execution_quality.csv пуст или не найден")
        return
    
    trades_df = load_trades_csv(bot_path)
    
    # Объединяем данные
    merged_df = merge_execution_and_trades(execution_df, trades_df)
    
    # Создаём бакеты задержки
    bucket_size_us = args.bucket_size_ms * 1000
    merged_df = create_latency_buckets(merged_df, bucket_size_us)
    
    # Рассчитываем статистику
    stats = calculate_fill_rate_correlation(merged_df)
    
    # Определяем точку деградации
    degradation_point = find_degradation_point(stats, args.degradation_threshold)
    
    # Генерируем отчёт
    report = generate_report(stats, degradation_point, output_dir)
    print(report)
    
    # Сохраняем отчёт
    report_path = output_dir / "execution_quality_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"✓ Отчёт сохранён: {report_path}")
    print()
    
    # Сохраняем детальную статистику
    save_detailed_csv(stats, output_dir)
    
    # Создаём графики
    try:
        plot_fill_rate_vs_latency(stats, output_dir)
    except Exception as e:
        print(f"⚠️  Ошибка при создании графиков: {e}")
    
    # Сохраняем JSON с результатами
    json_results = {
        "total_orders": stats["total_orders"],
        "avg_fill_rate": float(stats["avg_fill_rate"]),
        "avg_network_latency_us": float(stats["avg_network_latency_us"]),
        "avg_internal_latency_us": float(stats["avg_internal_latency_us"]),
        "correlation": float(stats["correlation"]),
        "degradation_point": degradation_point,
    }
    
    json_path = output_dir / "execution_quality_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"✓ Результаты в JSON сохранены: {json_path}")
    print()
    print("✅ Анализ завершён успешно!")


if __name__ == "__main__":
    main()
