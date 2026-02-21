#!/usr/bin/env python3
"""
Evaluate model uncertainty and rejection curves using MC Dropout.

Задача 125: Оценка неопределенности через MC Dropout (v2.1)

Этот скрипт анализирует неопределенность модели, полученную через MC Dropout,
и строит графики отказа (Rejection Curves) для оценки качества калибровки.

Использование:
    python evaluate_uncertainty.py --symbol BTCUSDT \\
        --checkpoint /path/to/checkpoint \\
        --uncertainty_data /path/to/mc_dropout_uncertainty.pt
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from sklearn.metrics import accuracy_score


def plot_rejection_curve(labels, probs_mc, metric_values, save_path, title="Rejection Curve"):
    """
    Построение Rejection Curve.
    
    Показывает, насколько вырастет точность (Accuracy), если мы откажемся 
    от самых «мутных» предсказаний.
    
    Args:
        labels: массив истинных меток (N,)
        probs_mc: массив средних вероятностей MC Dropout (N, C)
        metric_values: значения энтропии или MI для каждого сэмпла (N,)
        save_path: путь для сохранения графика
        title: название графика
    
    Returns:
        dict: словарь с результатами (rejection_rates, accuracies, auc_rejection)
    """
    
    # Получаем предсказания из средних вероятностей
    preds = probs_mc.argmax(axis=1)
    
    # Вычисляем корректность предсказаний
    correct = (labels == preds).astype(int)
    
    # Сортируем по убыванию неопределенности (сначала самые плохие)
    indices = np.argsort(metric_values)[::-1]
    sorted_correct = correct[indices]
    
    # Вычисляем метрики по мере отклонения худших примеров
    rejection_rates = np.linspace(0, 0.95, 50)
    accuracies = []
    
    for rate in rejection_rates:
        # Берем только оставшиеся примеры (отклоняем % худших)
        keep_idx = int(len(sorted_correct) * (1 - rate))
        if keep_idx > 0:
            remaining_acc = sorted_correct[keep_idx:].mean()
        else:
            remaining_acc = 1.0
        accuracies.append(remaining_acc)
    
    # Вычисляем AUC для Rejection Curve
    auc_rejection = np.trapz(accuracies, x=rejection_rates * 100) / 95.0
    
    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(rejection_rates * 100, accuracies, linewidth=2, marker='o', markersize=4)
    
    # Добавляем справочные линии
    baseline_acc = correct.mean()
    plt.axhline(y=baseline_acc, color='r', linestyle='--', label=f'Baseline Accuracy: {baseline_acc:.4f}')
    plt.axhline(y=1.0, color='g', linestyle=':', alpha=0.5, label='Perfect Accuracy')
    
    plt.xlabel('Rejection Rate (%)', fontsize=12)
    plt.ylabel('Accuracy on Remaining Samples', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    
    # Добавляем текст с AUC
    plt.text(0.98, 0.02, f'AUC-RC: {auc_rejection:.4f}', 
             transform=plt.gca().transAxes, 
             ha='right', va='bottom',
             fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Rejection curve saved to: {save_path}")
    plt.close()
    
    return {
        'rejection_rates': rejection_rates,
        'accuracies': accuracies,
        'auc_rejection': auc_rejection,
        'baseline_accuracy': baseline_acc
    }


def plot_uncertainty_distributions(entropy, mutual_info, save_dir):
    """
    Построение распределений энтропии и MI.
    
    Args:
        entropy: массив значений энтропии (N,)
        mutual_info: массив значений MI (N,)
        save_dir: директория для сохранения графиков
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Энтропия
    axes[0].hist(entropy, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_xlabel('Entropy', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Predictive Entropy', fontsize=12, fontweight='bold')
    axes[0].axvline(entropy.mean(), color='red', linestyle='--', label=f'Mean: {entropy.mean():.4f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MI
    axes[1].hist(mutual_info, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_xlabel('Mutual Information', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Mutual Information (Epistemic Uncertainty)', fontsize=12, fontweight='bold')
    axes[1].axvline(mutual_info.mean(), color='red', linestyle='--', label=f'Mean: {mutual_info.mean():.4f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / "uncertainty_distributions.png", dpi=300, bbox_inches='tight')
    print(f"Uncertainty distributions saved to: {save_dir / 'uncertainty_distributions.png'}")
    plt.close()


def plot_entropy_vs_accuracy(entropy, mutual_info, labels, mean_probs, save_dir):
    """
    Построение графиков зависимости точности от неопределенности.
    
    Args:
        entropy: массив значений энтропии (N,)
        mutual_info: массив значений MI (N,)
        labels: истинные метки (N,)
        mean_probs: средние вероятности (N, C)
        save_dir: директория для сохранения графиков
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Получаем предсказания
    preds = mean_probs.argmax(axis=1)
    correct = (labels == preds).astype(int)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Энтропия vs точность
    # Разделяем на корректные и неправильные
    correct_entropy = entropy[correct == 1]
    incorrect_entropy = entropy[correct == 0]
    
    axes[0].hist([correct_entropy, incorrect_entropy], bins=30, label=['Correct', 'Incorrect'], 
                 alpha=0.7, color=['blue', 'red'])
    axes[0].set_xlabel('Entropy', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Entropy Distribution: Correct vs Incorrect Predictions', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MI vs точность
    correct_mi = mutual_info[correct == 1]
    incorrect_mi = mutual_info[correct == 0]
    
    axes[1].hist([correct_mi, incorrect_mi], bins=30, label=['Correct', 'Incorrect'], 
                 alpha=0.7, color=['blue', 'red'])
    axes[1].set_xlabel('Mutual Information', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('MI Distribution: Correct vs Incorrect Predictions', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / "entropy_vs_accuracy.png", dpi=300, bbox_inches='tight')
    print(f"Entropy vs accuracy saved to: {save_dir / 'entropy_vs_accuracy.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate model uncertainty using MC Dropout")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--uncertainty_data", type=str, required=True, 
                       help="Path to MC Dropout uncertainty data (mc_dropout_uncertainty.pt)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for plots (default: reports/{symbol})")
    
    args = parser.parse_args()
    
    # Загружаем данные неопределенности
    print(f"Loading uncertainty data from: {args.uncertainty_data}")
    uncertainty_data = torch.load(args.uncertainty_data, map_location='cpu')
    
    # Извлекаем компоненты
    mc_logits = uncertainty_data['mc_logits']
    entropy = uncertainty_data['entropy'].numpy()
    mutual_info = uncertainty_data['mutual_info'].numpy()
    val_labels = uncertainty_data['val_labels'].numpy()
    mean_probs = uncertainty_data['mean_probs'].numpy()
    
    # Определяем выходную директорию
    if args.output_dir is None:
        base_path = Path(__file__).parent.parent
        output_dir = base_path / "reports" / args.symbol
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("MC DROPOUT UNCERTAINTY ANALYSIS")
    print(f"{'='*60}\n")
    
    # Печатаем статистику
    print(f"Number of validation samples: {len(val_labels)}")
    print(f"Number of MC passes: {mc_logits.shape[0]}")
    print(f"\nEntropy Statistics:")
    print(f"  Mean: {entropy.mean():.4f}")
    print(f"  Std: {entropy.std():.4f}")
    print(f"  Min: {entropy.min():.4f}")
    print(f"  Max: {entropy.max():.4f}")
    print(f"\nMutual Information Statistics:")
    print(f"  Mean: {mutual_info.mean():.4f}")
    print(f"  Std: {mutual_info.std():.4f}")
    print(f"  Min: {mutual_info.min():.4f}")
    print(f"  Max: {mutual_info.max():.4f}")
    print()
    
    # Построение Rejection Curves
    print("Building rejection curves...")
    entropy_results = plot_rejection_curve(
        val_labels, mean_probs, entropy,
        save_path=str(output_dir / "rejection_curve_entropy.png"),
        title="Rejection Curve (Entropy)"
    )
    
    mi_results = plot_rejection_curve(
        val_labels, mean_probs, mutual_info,
        save_path=str(output_dir / "rejection_curve_mi.png"),
        title="Rejection Curve (Mutual Information)"
    )
    
    # Построение распределений
    print("Plotting uncertainty distributions...")
    plot_uncertainty_distributions(entropy, mutual_info, output_dir)
    
    # Построение графиков корреляции с точностью
    print("Plotting entropy vs accuracy...")
    plot_entropy_vs_accuracy(entropy, mutual_info, val_labels, mean_probs, output_dir)
    
    # Выводим итоговые результаты
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Entropy AUC-RC: {entropy_results['auc_rejection']:.4f}")
    print(f"  MI AUC-RC: {mi_results['auc_rejection']:.4f}")
    print(f"  Baseline Accuracy: {entropy_results['baseline_accuracy']:.4f}")
    print(f"{'='*60}\n")
    
    print(f"All plots saved to: {output_dir}")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
