#!/usr/bin/env python3
"""
Задача 224: Анализ распределения уверенности инференса (Inference Confidence Distribution)

Скрипт для анализа "здоровья" ML-модели через:
1. Expected Calibration Error (ECE) - ошибка калибровки
2. KS-Test - сравнение распределений уверенности с baseline
3. Визуализация Reliability Diagrams и гистограмм энтропии

Использование:
    python scripts/model_confidence_analysis.py --model-dir bots/BTCUSDT/model --baseline-csv baseline_confidence.csv

Автор: Задача 224
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def calculate_entropy(probs: np.ndarray) -> np.ndarray:
    """
    Вычисляет энтропию распределения вероятностей.
    
    Формула: H = -Σ p_i * log(p_i)
    
    Args:
        probs: Массив вероятностей формы (N, num_classes)
    
    Returns:
        Массив энтропий формы (N,)
    """
    # Избегаем log(0) заменяя нули на малое значение
    probs_safe = np.where(probs > 0, probs, 1e-10)
    entropy = -np.sum(probs * np.log(probs_safe), axis=1)
    return entropy


def calculate_ece(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 10) -> float:
    """
    Вычисляет Expected Calibration Error (ECE).
    
    ECE измеряет, насколько хорошо предсказанные вероятности модели
    соответствуют истинным частотам событий.
    
    Формула: ECE = Σ (|Bₘ|/n) * |acc(Bₘ) - conf(Bₘ)|
    
    Args:
        confidences: Максимальные вероятности предсказаний (N,)
        accuracies: Булевы значения правильности предсказаний (N,)
        n_bins: Количество bins для разбиения
    
    Returns:
        Значение ECE (0 = идеальная калибровка)
    
    References:
        [Content rephrased for compliance with licensing restrictions]
        Based on: "On Calibration of Modern Neural Networks" (Guo et al., 2017)
        Implementation adapted from: https://towardsdatascience.com/expected-calibration-error-ece-a-step-by-step-visual-explanation-with-python-code-c3e9aa12937d/
    """
    # Равномерное разбиение на bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Определяем, какие сэмплы попадают в текущий bin
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prob_in_bin = in_bin.mean()
        
        if prob_in_bin > 0:
            # Точность в bin: acc(Bₘ)
            accuracy_in_bin = accuracies[in_bin].mean()
            # Средняя уверенность в bin: conf(Bₘ)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # Вклад bin в ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    
    return ece


def plot_reliability_diagram(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[Path] = None
):
    """
    Строит Reliability Diagram (диаграмма надежности).
    
    Показывает, насколько предсказанная уверенность соответствует
    фактической точности предсказаний.
    
    Args:
        confidences: Максимальные вероятности предсказаний (N,)
        accuracies: Булевы значения правильности предсказаний (N,)
        n_bins: Количество bins
        save_path: Путь для сохранения графика
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_centers = (bin_lowers + bin_uppers) / 2
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        if in_bin.sum() > 0:
            bin_accuracies.append(accuracies[in_bin].mean())
            bin_confidences.append(confidences[in_bin].mean())
            bin_counts.append(in_bin.sum())
        else:
            bin_accuracies.append(0)
            bin_confidences.append(0)
            bin_counts.append(0)
    
    bin_accuracies = np.array(bin_accuracies)
    bin_confidences = np.array(bin_confidences)
    bin_counts = np.array(bin_counts)
    
    # Построение графика
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Идеальная калибровка (диагональ)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    
    # Фактическая калибровка
    # Размер точек пропорционален количеству сэмплов в bin
    sizes = (bin_counts / bin_counts.max()) * 500 if bin_counts.max() > 0 else bin_counts
    scatter = ax.scatter(
        bin_confidences, bin_accuracies,
        s=sizes, alpha=0.6, c=bin_centers, cmap='viridis',
        edgecolors='black', linewidth=1.5
    )
    
    # Соединяем точки линией
    mask = bin_counts > 0
    ax.plot(bin_confidences[mask], bin_accuracies[mask], 'r-', alpha=0.5, linewidth=2)
    
    ax.set_xlabel('Predicted Confidence', fontsize=12)
    ax.set_ylabel('Actual Accuracy', fontsize=12)
    ax.set_title('Reliability Diagram', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Colorbar для показа bin centers
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Confidence Bin Center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Reliability diagram saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_entropy_distribution(
    entropy_live: np.ndarray,
    entropy_baseline: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None
):
    """
    Строит гистограммы распределения энтропии.
    
    Args:
        entropy_live: Энтропия из живых данных
        entropy_baseline: Энтропия из baseline (валидационная выборка)
        save_path: Путь для сохранения графика
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Гистограмма живых данных
    ax.hist(entropy_live, bins=50, alpha=0.6, label='Live Data', color='blue', density=True)
    
    # Гистограмма baseline (если есть)
    if entropy_baseline is not None:
        ax.hist(entropy_baseline, bins=50, alpha=0.6, label='Baseline (Validation)', color='green', density=True)
    
    # Статистика
    ax.axvline(entropy_live.mean(), color='blue', linestyle='--', linewidth=2, label=f'Live Mean: {entropy_live.mean():.4f}')
    if entropy_baseline is not None:
        ax.axvline(entropy_baseline.mean(), color='green', linestyle='--', linewidth=2, label=f'Baseline Mean: {entropy_baseline.mean():.4f}')
    
    ax.set_xlabel('Entropy (nats)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Entropy Distribution Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Entropy distribution plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def ks_test_comparison(
    entropy_live: np.ndarray,
    entropy_baseline: np.ndarray
) -> Tuple[float, float]:
    """
    Выполняет Kolmogorov-Smirnov тест для сравнения распределений.
    
    KS-тест проверяет гипотезу о том, что два распределения одинаковы.
    
    Args:
        entropy_live: Энтропия из живых данных
        entropy_baseline: Энтропия из baseline
    
    Returns:
        Кортеж (statistic, p_value)
        - statistic: KS статистика (0-1, чем больше - тем больше различие)
        - p_value: p-значение (< 0.05 означает значимое различие)
    """
    statistic, p_value = stats.ks_2samp(entropy_live, entropy_baseline)
    return statistic, p_value


def load_confidence_samples(csv_path: Path) -> pd.DataFrame:
    """
    Загружает сэмплы уверенности из CSV файла.
    
    Args:
        csv_path: Путь к CSV файлу с колонками:
                  timestamp_ms, prob_flat, prob_up, prob_down, entropy, ema_entropy
    
    Returns:
        DataFrame с загруженными данными
    """
    df = pd.read_csv(csv_path)
    required_cols = ['timestamp_ms', 'prob_flat', 'prob_up', 'prob_down', 'entropy', 'ema_entropy']
    
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    return df


def analyze_model_confidence(
    model_dir: Path,
    baseline_csv: Optional[Path] = None,
    output_dir: Optional[Path] = None
):
    """
    Основная функция анализа уверенности модели.
    
    Args:
        model_dir: Директория модели (содержит confidence_samples.csv и metadata.json)
        baseline_csv: Путь к CSV с baseline данными (опционально)
        output_dir: Директория для сохранения результатов
    """
    # Загружаем живые данные
    live_csv = model_dir / "confidence_samples.csv"
    if not live_csv.exists():
        print(f"Error: confidence_samples.csv not found in {model_dir}")
        print("Make sure the bot has been running with confidence tracking enabled.")
        sys.exit(1)
    
    print(f"Loading live data from: {live_csv}")
    df_live = load_confidence_samples(live_csv)
    print(f"Loaded {len(df_live)} samples")
    
    # Извлекаем вероятности и энтропию
    probs_live = df_live[['prob_flat', 'prob_up', 'prob_down']].values
    entropy_live = df_live['entropy'].values
    
    # Вычисляем максимальные уверенности и предсказания
    confidences_live = np.max(probs_live, axis=1)
    predictions_live = np.argmax(probs_live, axis=1)
    
    # Для расчета ECE нужны истинные метки, но у нас их нет в CSV
    # Поэтому мы можем только показать распределение уверенности
    print("\n=== Live Data Statistics ===")
    print(f"Mean entropy: {entropy_live.mean():.4f} ± {entropy_live.std():.4f}")
    print(f"Mean confidence: {confidences_live.mean():.4f} ± {confidences_live.std():.4f}")
    print(f"Entropy range: [{entropy_live.min():.4f}, {entropy_live.max():.4f}]")
    
    # Загружаем baseline если есть
    entropy_baseline = None
    if baseline_csv and baseline_csv.exists():
        print(f"\nLoading baseline data from: {baseline_csv}")
        df_baseline = load_confidence_samples(baseline_csv)
        entropy_baseline = df_baseline['entropy'].values
        
        print(f"\n=== Baseline Data Statistics ===")
        print(f"Mean entropy: {entropy_baseline.mean():.4f} ± {entropy_baseline.std():.4f}")
        print(f"Entropy range: [{entropy_baseline.min():.4f}, {entropy_baseline.max():.4f}]")
        
        # KS-Test
        ks_stat, ks_pval = ks_test_comparison(entropy_live, entropy_baseline)
        print(f"\n=== Kolmogorov-Smirnov Test ===")
        print(f"KS Statistic: {ks_stat:.4f}")
        print(f"P-value: {ks_pval:.6f}")
        if ks_pval < 0.05:
            print("⚠️  SIGNIFICANT DRIFT DETECTED (p < 0.05)")
            print("The live distribution differs significantly from baseline!")
        else:
            print("✓ No significant drift detected (p >= 0.05)")
    
    # Создаем output директорию
    if output_dir is None:
        output_dir = model_dir / "confidence_analysis"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Визуализация
    print(f"\n=== Generating Visualizations ===")
    
    # Entropy distribution
    plot_entropy_distribution(
        entropy_live,
        entropy_baseline,
        save_path=output_dir / "entropy_distribution.png"
    )
    
    # Примечание: Reliability Diagram требует истинных меток
    # Если у вас есть файл с истинными метками, раскомментируйте:
    # true_labels = ...  # загрузите истинные метки
    # accuracies = (predictions_live == true_labels)
    # ece = calculate_ece(confidences_live, accuracies)
    # print(f"\nExpected Calibration Error (ECE): {ece:.4f}")
    # plot_reliability_diagram(confidences_live, accuracies, save_path=output_dir / "reliability_diagram.png")
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze model confidence distribution and detect drift (Task 224)"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to model directory (contains confidence_samples.csv)"
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        help="Path to baseline confidence CSV (from validation set)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for analysis results (default: model_dir/confidence_analysis)"
    )
    
    args = parser.parse_args()
    
    analyze_model_confidence(
        model_dir=args.model_dir,
        baseline_csv=args.baseline_csv,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
