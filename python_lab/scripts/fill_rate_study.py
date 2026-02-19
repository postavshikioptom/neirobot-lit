#!/usr/bin/env python3
"""
Задача 203: Исследование распределения Fill Rate (Fill Rate Distribution Study)

Скрипт проводит статистический анализ зависимости исполнения ордеров от состояния стакана (LOB).
Анализирует, как объем уровня и дисбаланс (Imbalance) влияют на вероятность Full Fill 
и время нахождения ордера в стакане.

Использование:
    python fill_rate_study.py --bot-path /path/to/bot --output-dir /path/to/output
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Optional, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve


def load_order_context_csv(bot_path: Path) -> Optional[pd.DataFrame]:
    """Загружает order_context.csv из папки бота (Задача 203)."""
    csv_path = bot_path / "logs" / "order_context.csv"
    
    if not csv_path.exists():
        print(f"⚠️  Файл не найден: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Загружено {len(df)} записей из order_context.csv")
        return df
    except Exception as e:
        print(f"❌ Ошибка при загрузке order_context.csv: {e}")
        return None


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


def merge_all_data(
    order_context_df: pd.DataFrame,
    execution_quality_df: Optional[pd.DataFrame],
    trades_df: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """
    Объединяет данные из трех файлов по order_id.
    order_context.csv - основной источник данных о стакане
    execution_quality.csv - данные о задержке и fill_rate
    trades.csv - данные о сделках
    """
    merged = order_context_df.copy()
    
    # Объединяем с execution_quality.csv
    if execution_quality_df is not None and not execution_quality_df.empty:
        try:
            merged = merged.merge(
                execution_quality_df,
                on="order_id",
                how="left",
                suffixes=("_context", "_exec")
            )
            print(f"✓ Объединено с execution_quality.csv: {len(merged)} записей")
        except Exception as e:
            print(f"⚠️  Ошибка при объединении с execution_quality.csv: {e}")
    
    # Объединяем с trades.csv
    if trades_df is not None and not trades_df.empty:
        try:
            merged = merged.merge(
                trades_df,
                on="order_id",
                how="left",
                suffixes=("", "_trade")
            )
            print(f"✓ Объединено с trades.csv: {len(merged)} записей")
        except Exception as e:
            print(f"⚠️  Ошибка при объединении с trades.csv: {e}")
    
    return merged


def calculate_correlations(df: pd.DataFrame) -> Dict[str, float]:
    """
    Рассчитывает корреляции между параметрами стакана и исполнением.
    """
    correlations = {}
    
    # Корреляция между объемом на уровне и временем исполнения
    if "level_total_vol" in df.columns and "fill_duration_us" in df.columns:
        corr = df["level_total_vol"].corr(df["fill_duration_us"])
        correlations["level_vol_vs_fill_duration"] = float(corr)
        print(f"✓ Корреляция (level_total_vol vs fill_duration_us): {corr:.4f}")
    
    # Корреляция между дисбалансом и временем исполнения
    if "imbalance_5l" in df.columns and "fill_duration_us" in df.columns:
        corr = df["imbalance_5l"].corr(df["fill_duration_us"])
        correlations["imbalance_vs_fill_duration"] = float(corr)
        print(f"✓ Корреляция (imbalance_5l vs fill_duration_us): {corr:.4f}")
    
    # Корреляция между размером ордера и временем исполнения
    if "order_size" in df.columns and "fill_duration_us" in df.columns:
        corr = df["order_size"].corr(df["fill_duration_us"])
        correlations["order_size_vs_fill_duration"] = float(corr)
        print(f"✓ Корреляция (order_size vs fill_duration_us): {corr:.4f}")
    
    # Корреляция между дисбалансом и fill_rate (если есть)
    if "imbalance_5l" in df.columns and "fill_rate" in df.columns:
        corr = df["imbalance_5l"].corr(df["fill_rate"])
        correlations["imbalance_vs_fill_rate"] = float(corr)
        print(f"✓ Корреляция (imbalance_5l vs fill_rate): {corr:.4f}")
    
    return correlations


def build_logistic_regression(df: pd.DataFrame) -> Dict:
    """
    Строит Logistic Regression модель для предсказания вероятности исполнения.
    Входные признаки: order_size, imbalance_5l, level_total_vol
    Целевая переменная: fill_rate > 0.8 (успешное исполнение)
    """
    # Подготавливаем данные
    required_cols = ["order_size", "imbalance_5l", "level_total_vol", "fill_rate"]
    available_cols = [col for col in required_cols if col in df.columns]
    
    if len(available_cols) < 3:
        print(f"⚠️  Недостаточно данных для Logistic Regression. Доступные колонки: {available_cols}")
        return {}
    
    # Удаляем NaN значения
    df_clean = df[available_cols].dropna()
    
    if len(df_clean) < 10:
        print(f"⚠️  Недостаточно данных для обучения модели (< 10 записей)")
        return {}
    
    # Создаем целевую переменную: успешное исполнение (fill_rate > 0.8)
    y = (df_clean["fill_rate"] > 0.8).astype(int)
    
    # Выбираем признаки
    feature_cols = [col for col in ["order_size", "imbalance_5l", "level_total_vol"] if col in available_cols]
    X = df_clean[feature_cols]
    
    # Нормализуем признаки
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Обучаем модель
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y)
    
    # Получаем предсказания
    y_pred = model.predict(X_scaled)
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    
    # Рассчитываем метрики
    results = {
        "model_coefficients": dict(zip(feature_cols, model.coef_[0])),
        "model_intercept": float(model.intercept_[0]),
        "accuracy": float((y_pred == y).mean()),
        "roc_auc": float(roc_auc_score(y, y_pred_proba)),
        "feature_names": feature_cols,
        "training_samples": len(df_clean),
        "success_rate": float(y.mean()),
    }
    
    print(f"✓ Logistic Regression модель обучена на {len(df_clean)} образцах")
    print(f"  Точность: {results['accuracy']:.4f}")
    print(f"  ROC-AUC: {results['roc_auc']:.4f}")
    print(f"  Коэффициенты: {results['model_coefficients']}")
    
    return results


def create_volume_buckets(df: pd.DataFrame, bucket_size: float = 100.0) -> pd.DataFrame:
    """
    Группирует данные по бакетам объема на уровне.
    """
    if "level_total_vol" not in df.columns:
        return df
    
    df["volume_bucket"] = (df["level_total_vol"] // bucket_size * bucket_size).astype(int)
    return df


def analyze_by_volume_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Анализирует Fill Rate в зависимости от объема на уровне.
    """
    if "volume_bucket" not in df.columns:
        df = create_volume_buckets(df)
    
    if "fill_rate" not in df.columns:
        return pd.DataFrame()
    
    grouped = df.groupby("volume_bucket").agg({
        "fill_rate": ["mean", "std", "count"],
        "fill_duration_us": "mean",
        "order_size": "mean",
        "imbalance_5l": "mean",
    }).round(4)
    
    grouped.columns = ["fill_rate_mean", "fill_rate_std", "order_count",
                       "avg_fill_duration_us", "avg_order_size", "avg_imbalance"]
    grouped = grouped.reset_index()
    
    return grouped


def plot_fill_rate_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """Создает графики анализа Fill Rate."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # График 1: Fill Rate vs Level Volume
        if "level_total_vol" in df.columns and "fill_rate" in df.columns:
            df_clean = df[["level_total_vol", "fill_rate"]].dropna()
            if len(df_clean) > 0:
                axes[0, 0].scatter(df_clean["level_total_vol"], df_clean["fill_rate"], alpha=0.5)
                axes[0, 0].set_xlabel("Объем на уровне")
                axes[0, 0].set_ylabel("Fill Rate")
                axes[0, 0].set_title("Fill Rate vs Объем на уровне")
                axes[0, 0].grid(True, alpha=0.3)
        
        # График 2: Fill Rate vs Imbalance
        if "imbalance_5l" in df.columns and "fill_rate" in df.columns:
            df_clean = df[["imbalance_5l", "fill_rate"]].dropna()
            if len(df_clean) > 0:
                axes[0, 1].scatter(df_clean["imbalance_5l"], df_clean["fill_rate"], alpha=0.5, color="orange")
                axes[0, 1].set_xlabel("Дисбаланс (5 уровней)")
                axes[0, 1].set_ylabel("Fill Rate")
                axes[0, 1].set_title("Fill Rate vs Дисбаланс")
                axes[0, 1].grid(True, alpha=0.3)
        
        # График 3: Fill Duration vs Level Volume
        if "level_total_vol" in df.columns and "fill_duration_us" in df.columns:
            df_clean = df[["level_total_vol", "fill_duration_us"]].dropna()
            if len(df_clean) > 0:
                axes[1, 0].scatter(df_clean["level_total_vol"], df_clean["fill_duration_us"], alpha=0.5, color="green")
                axes[1, 0].set_xlabel("Объем на уровне")
                axes[1, 0].set_ylabel("Время исполнения (мкс)")
                axes[1, 0].set_title("Время исполнения vs Объем на уровне")
                axes[1, 0].grid(True, alpha=0.3)
        
        # График 4: Order Size Distribution
        if "order_size" in df.columns:
            df_clean = df["order_size"].dropna()
            if len(df_clean) > 0:
                axes[1, 1].hist(df_clean, bins=30, color="purple", alpha=0.7)
                axes[1, 1].set_xlabel("Размер ордера")
                axes[1, 1].set_ylabel("Количество")
                axes[1, 1].set_title("Распределение размеров ордеров")
                axes[1, 1].grid(True, alpha=0.3, axis="y")
        
        plt.tight_layout()
        output_path = output_dir / "fill_rate_distribution_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✓ График сохранён: {output_path}")
        plt.close()
    except Exception as e:
        print(f"⚠️  Ошибка при создании графиков: {e}")


def generate_report(
    correlations: Dict[str, float],
    regression_results: Dict,
    volume_analysis: pd.DataFrame,
    output_dir: Path
) -> str:
    """Генерирует текстовый отчёт анализа."""
    report = []
    report.append("=" * 80)
    report.append("ИССЛЕДОВАНИЕ РАСПРЕДЕЛЕНИЯ FILL RATE (Задача 203)")
    report.append("=" * 80)
    report.append("")
    
    # Корреляции
    report.append("📊 АНАЛИЗ КОРРЕЛЯЦИЙ")
    report.append("-" * 80)
    for key, value in correlations.items():
        report.append(f"{key}: {value:.4f}")
    report.append("")
    
    # Logistic Regression
    if regression_results:
        report.append("🤖 LOGISTIC REGRESSION МОДЕЛЬ")
        report.append("-" * 80)
        report.append(f"Обучено на {regression_results['training_samples']} образцах")
        report.append(f"Точность: {regression_results['accuracy']:.4f}")
        report.append(f"ROC-AUC: {regression_results['roc_auc']:.4f}")
        report.append(f"Процент успешных исполнений: {regression_results['success_rate']:.4f}")
        report.append("")
        report.append("Коэффициенты модели:")
        for feature, coef in regression_results['model_coefficients'].items():
            report.append(f"  {feature}: {coef:.6f}")
        report.append(f"  Intercept: {regression_results['model_intercept']:.6f}")
        report.append("")
    
    # Анализ по объемам
    if not volume_analysis.empty:
        report.append("📈 АНАЛИЗ ПО БАКЕТАМ ОБЪЕМА")
        report.append("-" * 80)
        for _, row in volume_analysis.iterrows():
            report.append(
                f"Объем {int(row['volume_bucket']):6d}: "
                f"Fill Rate={row['fill_rate_mean']:.4f}, "
                f"Ордеров={int(row['order_count']):4d}, "
                f"Avg Duration={row['avg_fill_duration_us']:.0f}мкс"
            )
        report.append("")
    
    report.append("=" * 80)
    report.append("💡 ВЫВОДЫ:")
    report.append("Модель может использоваться для предсказания вероятности исполнения ордера")
    report.append("на основе состояния стакана. Если вероятность < 20%, бот может игнорировать сигнал.")
    report.append("=" * 80)
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="Исследование распределения Fill Rate (Задача 203)"
    )
    parser.add_argument("--bot-path", type=Path, required=True,
                        help="Путь к папке бота (содержит logs/order_context.csv)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Директория для сохранения результатов (по умолчанию: bot_path/analysis)")
    
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
    order_context_df = load_order_context_csv(bot_path)
    if order_context_df is None or order_context_df.empty:
        print("❌ Ошибка: order_context.csv пуст или не найден")
        return
    
    execution_quality_df = load_execution_quality_csv(bot_path)
    trades_df = load_trades_csv(bot_path)
    
    # Объединяем данные
    merged_df = merge_all_data(order_context_df, execution_quality_df, trades_df)
    print(f"✓ Всего объединено {len(merged_df)} записей")
    print()
    
    # Рассчитываем корреляции
    print("📊 Рассчитываем корреляции...")
    correlations = calculate_correlations(merged_df)
    print()
    
    # Строим Logistic Regression модель
    print("🤖 Строим Logistic Regression модель...")
    regression_results = build_logistic_regression(merged_df)
    print()
    
    # Анализируем по объемам
    print("📈 Анализируем по бакетам объема...")
    volume_analysis = analyze_by_volume_buckets(merged_df)
    print()
    
    # Генерируем отчёт
    report = generate_report(correlations, regression_results, volume_analysis, output_dir)
    print(report)
    
    # Сохраняем отчёт
    report_path = output_dir / "fill_rate_study_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"✓ Отчёт сохранён: {report_path}")
    print()
    
    # Сохраняем анализ по объемам
    if not volume_analysis.empty:
        csv_path = output_dir / "fill_rate_by_volume.csv"
        volume_analysis.to_csv(csv_path, index=False)
        print(f"✓ Анализ по объемам сохранён: {csv_path}")
    
    # Создаём графики
    plot_fill_rate_analysis(merged_df, output_dir)
    
    # Сохраняем JSON с результатами
    json_results = {
        "correlations": correlations,
        "regression_results": regression_results,
        "total_records": len(merged_df),
    }
    
    json_path = output_dir / "fill_rate_study_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"✓ Результаты в JSON сохранены: {json_path}")
    print()
    print("✅ Анализ завершён успешно!")


if __name__ == "__main__":
    main()
