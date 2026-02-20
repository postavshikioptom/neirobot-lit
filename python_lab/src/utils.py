import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    matthews_corrcoef,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

def save_confusion_matrices(y_true, y_pred, class_names, output_dir):
    """
    Генерирует и сохраняет две матрицы ошибок: сырую и нормализованную.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Стилизация графиков
    plt.rcParams.update({'font.size': 12})
    
    # 1. Сырая матрица (Raw)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, 
        y_pred, 
        display_labels=class_names,
        cmap='Blues',
        normalize=None,
        ax=ax
    )
    ax.set_title('Confusion Matrix (Raw Counts)')
    plt.tight_layout()
    raw_path = output_dir / "confusion_matrix_raw.png"
    plt.savefig(raw_path, dpi=150)
    plt.close()
    
    # 2. Нормализованная матрица (Recall по строкам)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, 
        y_pred, 
        display_labels=class_names,
        cmap='Blues',
        normalize='true',
        values_format='.2f',
        ax=ax
    )
    ax.set_title('Confusion Matrix (Normalized Recall)')
    plt.tight_layout()
    norm_path = output_dir / "confusion_matrix_normalized.png"
    plt.savefig(norm_path, dpi=150)
    plt.close()
    
    print(f"Confusion matrices saved to: {output_dir}")

class FocalLoss(nn.Module):
    """
    Реализация Focal Loss для борьбы с дисбалансом классов и фокусировки на сложных примерах.
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Базовый CrossEntropy с поддержкой весов и сглаживания меток
        ce_loss = F.cross_entropy(
            inputs, 
            targets, 
            reduction='none', 
            weight=self.alpha, 
            label_smoothing=self.label_smoothing
        )
        
        # Получаем вероятности предсказанного класса
        pt = torch.exp(-ce_loss)
        
        # Вычисляем Focal Loss
        # (1-pt)**gamma фокусирует модель на сложных (низкое pt) примерах
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Применяем reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'none':
            return focal_loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")

class MultiHorizonLoss(nn.Module):
    """
    Multi-Horizon Loss для обучения модели на нескольких временных масштабах (Задача 160).
    
    Особенности:
    - Использует CrossEntropyLoss с ignore_index=-100 для маскирования недоступных горизонтов
    - Поддерживает динамическое взвешивание горизонтов (веса можно оптимизировать через Optuna)
    - Поддерживает временное взвешивание примеров (time_weighting)
    
    Args:
        num_horizons: количество горизонтов предсказания
        horizon_weights: веса для каждого горизонта (по умолчанию равные)
        class_weights: веса классов для каждого горизонта (опционально)
        label_smoothing: сглаживание меток
        reduction: 'mean', 'sum' или 'none'
    """
    def __init__(self, num_horizons=3, horizon_weights=None, class_weights=None, label_smoothing=0.0, reduction='mean'):
        super().__init__()
        self.num_horizons = num_horizons
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        
        # Веса горизонтов (по умолчанию равные)
        if horizon_weights is None:
            self.horizon_weights = torch.ones(num_horizons) / num_horizons
        else:
            self.horizon_weights = torch.tensor(horizon_weights, dtype=torch.float32)
            # Нормализуем веса
            self.horizon_weights = self.horizon_weights / self.horizon_weights.sum()
        
        # Веса классов (опционально)
        self.class_weights = class_weights
        if class_weights is not None and not isinstance(class_weights, torch.Tensor):
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    def forward(self, logits, targets, sample_weights=None):
        """
        Args:
            logits: (batch, num_horizons, 3) - предсказания модели
            targets: (batch, num_horizons) - целевые метки (может содержать -100 для маскирования)
            sample_weights: (batch,) - веса примеров (опционально, для time weighting)
        
        Returns:
            loss: скалярное значение лосса
        """
        batch_size = logits.shape[0]
        device = logits.device
        
        # Переносим веса на нужное устройство
        horizon_weights = self.horizon_weights.to(device)
        class_weights = self.class_weights.to(device) if self.class_weights is not None else None
        
        # Вычисляем лосс для каждого горизонта
        total_loss = 0.0
        
        for h in range(self.num_horizons):
            # Извлекаем логиты и таргеты для текущего горизонта
            logits_h = logits[:, h, :]  # (batch, 3)
            targets_h = targets[:, h]    # (batch,)
            
            # CrossEntropyLoss с ignore_index=-100 (автоматически игнорирует маскированные примеры)
            loss_h = F.cross_entropy(
                logits_h,
                targets_h,
                weight=class_weights,
                ignore_index=-100,
                label_smoothing=self.label_smoothing,
                reduction='none'  # Получаем лосс для каждого примера
            )
            
            # Применяем временное взвешивание (если передано)
            if sample_weights is not None:
                loss_h = loss_h * sample_weights
            
            # Применяем взвешивание по горизонтам
            weighted_loss_h = horizon_weights[h] * loss_h.mean()
            total_loss += weighted_loss_h
        
        # Применяем reduction
        if self.reduction == 'mean':
            return total_loss
        elif self.reduction == 'sum':
            return total_loss * batch_size
        elif self.reduction == 'none':
            # Для 'none' возвращаем лосс по примерам (усредненный по горизонтам)
            return total_loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")

def compute_metrics(y_true, y_pred, class_weights=None):
    """
    Вычисляет комплексные метрики для оценки модели в условиях дисбаланса классов.
    
    y_true: numpy array истинных меток
    y_pred: numpy array предсказанных меток
    class_weights: опционально, веса классов для логирования
    """
    # 1. Основные агрегированные метрики
    mcc = matthews_corrcoef(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    # 2. Поклассовые метрики через classification_report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # Извлекаем F1 для каждого из 3-х классов (0=Flat, 1=Up, 2=Down)
    # Используем строковые ключи, так как sklearn возвращает их такими
    f1_flat = report.get('0', {}).get('f1-score', 0.0)
    f1_up = report.get('1', {}).get('f1-score', 0.0)
    f1_down = report.get('2', {}).get('f1-score', 0.0)
    
    # 3. Формируем плоский словарь для логирования
    metrics = {
        "mcc": float(mcc),
        "balanced_acc": float(balanced_acc),
        "f1_flat": float(f1_flat),
        "f1_up": float(f1_up),
        "f1_down": float(f1_down),
        "f1_macro": float(f1_macro)
    }

    if class_weights is not None:
        metrics["weight_flat"] = float(class_weights[0])
        metrics["weight_up"] = float(class_weights[1])
        metrics["weight_down"] = float(class_weights[2])

    return metrics


class CalibrationMetrics:
    """
    Класс для вычисления метрик калибровки модели: ECE и MCE.
    
    ECE (Expected Calibration Error) - средневзвешенная разность между точностью и уверенностью.
    MCE (Maximum Calibration Error) - максимальная разность между точностью и уверенностью среди всех корзин.
    """
    def __init__(self, n_bins=15):
        self.n_bins = n_bins
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def calculate(self, logits, labels):
        """
        Вычисляет ECE, MCE и данные по корзинам для визуализации.
        
        Args:
            logits: тензор логитов модели (N, C)
            labels: истинные метки (N,)
        
        Returns:
            tuple: (ece, mce, bin_data)
                - ece: Expected Calibration Error
                - mce: Maximum Calibration Error
                - bin_data: список словарей с данными по каждой корзине
        """
        softmaxes = torch.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        mce = torch.zeros(1, device=logits.device)
        
        bin_data = []

        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Маска для текущей корзины
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                abs_diff = torch.abs(accuracy_in_bin - avg_confidence_in_bin)
                
                # ECE: взвешенная сумма разностей
                ece += abs_diff * prop_in_bin
                # MCE: максимальная разность среди всех корзин
                mce = torch.max(mce, abs_diff)
                
                bin_data.append({
                    "acc": accuracy_in_bin.item(),
                    "conf": avg_confidence_in_bin.item(),
                    "count": in_bin.sum().item()
                })
            else:
                bin_data.append({
                    "acc": 0, 
                    "conf": (bin_lower + bin_upper).item() / 2, 
                    "count": 0
                })

        return ece.item(), mce.item(), bin_data


def plot_reliability_diagram(bin_data, ece, mce, save_path):
    """
    Строит диаграмму надежности (Reliability Diagram) с двойной осью.
    
    Основная ось показывает точность (Accuracy), вторичная ось показывает 
    количество примеров в каждой корзине (Count) с логарифмической шкалой.
    
    Args:
        bin_data: список словарей с данными по корзинам (из CalibrationMetrics.calculate)
        ece: Expected Calibration Error
        mce: Maximum Calibration Error
        save_path: путь для сохранения графика (например, reports/SYMBOL/calibration.png)
    """
    accs = [d['acc'] for d in bin_data]
    confs = [d['conf'] for d in bin_data]
    counts = [d['count'] for d in bin_data]
    bins = np.linspace(0, 1, len(bin_data) + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    fig, ax1 = plt.subplots(figsize=(8, 8))
    
    # Основная ось: Точность
    ax1.bar(bin_centers, accs, width=1/len(bin_data), alpha=0.7, 
            edgecolor='black', label='Outputs', color='blue')
    ax1.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
    
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'Reliability Diagram (ECE: {ece:.4f}, MCE: {mce:.4f})')
    ax1.legend(loc='upper left')

    # Вторичная ось: Количество примеров в корзине
    ax2 = ax1.twinx()
    ax2.bar(bin_centers, counts, width=1/len(bin_data), alpha=0.2, 
            color='gray', label='Bin Size')
    ax2.set_ylabel('Count (Log Scale)')
    ax2.set_yscale('log')  # Логарифмическая шкала для лучшей видимости
    ax2.legend(loc='upper right')

    plt.tight_layout()
    
    # Создаем директорию, если не существует
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(save_path)
    plt.close()
    
    print(f"✓ Reliability diagram saved to: {save_path}")


def analyze_labels(df: pl.DataFrame, output_dir=None, save_plots=True):
    """
    Анализирует распределение меток (Up/Down/Flat) и матрицу переходов.
    
    Args:
        df: Polars DataFrame с колонкой 'label'
        output_dir: Путь для сохранения графиков и метаданных (опционально)
        save_plots: Флаг для сохранения визуализаций
    
    Returns:
        dict: Словарь с метриками распределения и матрицей переходов
    """
    if "label" not in df.columns:
        raise ValueError("DataFrame должен содержать колонку 'label'")
    
    # 1. Распределение классов
    counts = df.group_by("label").len().sort("label")
    total = counts["len"].sum()
    
    # Добавляем процентное соотношение
    counts = counts.with_columns(
        (pl.col("len") / total * 100).alias("percentage")
    )
    
    print("\n=== Распределение меток ===")
    print(counts)
    
    # Вычисляем Imbalance Ratio (защита от деления на ноль)
    min_count = counts["len"].min()
    max_count = counts["len"].max()
    
    if min_count == 0:
        print("\n⚠️  WARNING: Один из классов полностью отсутствует в выборке!")
        imbalance_ratio = float('inf')
    else:
        imbalance_ratio = max_count / min_count
    
    print(f"\nImbalance Ratio (Max/Min): {imbalance_ratio:.2f}")
    
    # 2. Матрица переходов (Transition Matrix)
    labels_ser = df["label"].to_pandas()
    
    # Создаем матрицу переходов с нормализацией по строкам
    transition = pd.crosstab(
        labels_ser.shift(1), 
        labels_ser, 
        normalize='index',
        dropna=False
    )
    
    print("\n=== Матрица переходов (вероятности) ===")
    print(transition)
    
    # Проверка: сумма строк должна быть ~1.0
    row_sums = transition.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        print("\n⚠️  WARNING: Сумма строк в матрице переходов не равна 1.0!")
    
    # 3. Формируем результат
    result = {
        "distribution": counts.to_dicts(),
        "imbalance_ratio": float(imbalance_ratio) if imbalance_ratio != float('inf') else None,
        "transition_matrix": transition.to_dict()
    }
    
    # 4. Визуализация (если требуется)
    if save_plots and output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Настройка стиля
        sns.set_theme(style="whitegrid")
        
        # 4.1. Гистограмма частот классов
        fig, ax = plt.subplots(figsize=(10, 6))
        counts_pd = counts.to_pandas()
        
        sns.barplot(
            data=counts_pd,
            x="label",
            y="len",
            hue="label",
            palette="viridis",
            legend=False,
            ax=ax
        )
        
        # Добавляем проценты на столбцы
        for i, row in counts_pd.iterrows():
            ax.text(
                i, 
                row["len"], 
                f'{row["percentage"]:.1f}%',
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold'
            )
        
        ax.set_title('Распределение классов меток', fontsize=14, fontweight='bold')
        ax.set_xlabel('Класс метки', fontsize=12)
        ax.set_ylabel('Количество', fontsize=12)
        plt.tight_layout()
        
        hist_path = output_path / "label_distribution.png"
        plt.savefig(hist_path, dpi=150)
        plt.close()
        print(f"\n✓ Гистограмма сохранена: {hist_path}")
        
        # 4.2. Тепловая карта матрицы переходов
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(
            transition,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Вероятность перехода'},
            linewidths=0.5,
            ax=ax
        )
        
        ax.set_title('Матрица переходов между классами', fontsize=14, fontweight='bold')
        ax.set_xlabel('Следующий класс', fontsize=12)
        ax.set_ylabel('Текущий класс', fontsize=12)
        plt.tight_layout()
        
        heatmap_path = output_path / "transition_matrix.png"
        plt.savefig(heatmap_path, dpi=150)
        plt.close()
        print(f"✓ Тепловая карта сохранена: {heatmap_path}")
    
    return result

def calculate_uncertainty(mc_logits):
    """
    mc_logits: тензор (N_passes, Batch, Classes)
    Возвращает: средние вероятности, энтропию и MI.
    """
    # Переход к вероятностям (N, B, C)
    probs = torch.softmax(mc_logits, dim=-1)
    
    # 1. Predictive Mean (средняя вероятность по прогонам)
    mean_probs = probs.mean(dim=0)
    
    # 2. Predictive Entropy (Общая неопределенность)
    # H = -sum(p * log(p))
    entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-9), dim=-1)
    
    # 3. Mutual Information (Эпистемическая неопределенность)
    # MI = Entropy(Mean_Probs) - Mean(Entropy_per_pass)
    per_pass_entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    expected_entropy = per_pass_entropy.mean(dim=0)
    mutual_info = entropy - expected_entropy
    
    return mean_probs, entropy, mutual_info


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss для передачи знаний от Teacher к Student.
    
    Комбинирует:
    1. Soft Loss (KL Divergence) - учит Student имитировать распределение Teacher
    2. Hard Loss (Cross Entropy) - учит Student на истинных метках
    
    Args:
        alpha: вес soft loss (обычно 0.7-0.9 для LOB данных)
        temperature: температура для размягчения логитов (обычно 2-5)
        reduction: тип редукции ('mean', 'sum', 'none')
    """
    def __init__(self, alpha=0.9, temperature=3.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.reduction = reduction
        self.kl_div = nn.KLDivLoss(reduction=reduction)
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
    
    def forward(self, student_logits, teacher_logits, labels):
        """
        Args:
            student_logits: логиты student модели (B, C)
            teacher_logits: логиты teacher модели (B, C)
            labels: истинные метки (B,)
        
        Returns:
            combined_loss: взвешенная комбинация soft и hard loss
        """
        T = self.temperature
        
        # Soft Loss: KL Divergence между размягченными распределениями
        # KLDivLoss ожидает log_softmax от student и softmax от teacher
        soft_loss = self.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1)
        ) * (T * T)  # Масштабирование T^2 (из оригинальной статьи Hinton)
        
        # Hard Loss: обычный Cross Entropy с истинными метками
        hard_loss = self.ce_loss(student_logits, labels)
        
        # Комбинированный loss
        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return loss


def count_parameters(model):
    """
    Подсчитывает количество обучаемых параметров в модели.
    
    Args:
        model: PyTorch модель
    
    Returns:
        int: количество параметров
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_latency(model, sample_input, device='cuda', warmup_runs=10, test_runs=100):
    """
    Замеряет среднюю латентность инференса модели с использованием CUDA events.
    
    Args:
        model: PyTorch модель для тестирования
        sample_input: тестовый батч (B, S, C, L)
        device: устройство ('cuda' или 'cpu')
        warmup_runs: количество прогревочных прогонов
        test_runs: количество тестовых прогонов для усреднения
    
    Returns:
        float: средняя латентность в миллисекундах
    """
    model.eval()
    model.to(device)
    sample_input = sample_input.to(device)
    
    # Прогрев (warmup) для стабилизации GPU
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(sample_input)
    
    # Синхронизация перед замером
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Замер латентности с CUDA events
    if device == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        timings = []
        with torch.no_grad():
            for _ in range(test_runs):
                start_event.record()
                _ = model(sample_input)
                end_event.record()
                torch.cuda.synchronize()
                timings.append(start_event.elapsed_time(end_event))
        
        mean_latency = np.mean(timings)
    else:
        # Для CPU используем time.perf_counter
        import time
        timings = []
        with torch.no_grad():
            for _ in range(test_runs):
                start = time.perf_counter()
                _ = model(sample_input)
                end = time.perf_counter()
                timings.append((end - start) * 1000)  # Конвертируем в мс
        
        mean_latency = np.mean(timings)
    
    return mean_latency



def plot_feature_importance_bar(importance_dict, top_k=20, save_path=None):
    """
    Строит bar chart для топ-K наиболее важных признаков с усами погрешности.
    
    Args:
        importance_dict: словарь с результатами важности признаков
        top_k: количество топ признаков для отображения
        save_path: путь для сохранения графика
    """
    # Сортируем по убыванию важности
    sorted_features = sorted(
        importance_dict.items(),
        key=lambda x: x[1]['mean_importance'],
        reverse=True
    )[:top_k]
    
    # Извлекаем данные
    feature_names = [f[0] for f in sorted_features]
    mean_importance = [f[1]['mean_importance'] for f in sorted_features]
    std_dev = [f[1]['std_dev'] for f in sorted_features]
    
    # Создаем график
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(feature_names))
    
    # Горизонтальный bar chart с усами погрешности
    ax.barh(y_pos, mean_importance, xerr=std_dev, align='center',
            alpha=0.7, ecolor='black', capsize=5, color='steelblue')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names, fontsize=10)
    ax.invert_yaxis()  # Самый важный признак сверху
    ax.set_xlabel('Mean Importance (MCC Drop)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top-{top_k} Feature Importance (Permutation Method)', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Bar chart сохранен: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_lob_importance_heatmap(importance_dict, n_levels=50, save_path=None):
    """
    Строит heatmap важности признаков для LOB данных.
    Матрица: 50 уровней (Y-ось) × 3 канала (X-ось: Price, Volume, Imbalance).
    
    Args:
        importance_dict: словарь с результатами важности признаков
        n_levels: количество уровней стакана
        save_path: путь для сохранения графика
    """
    # Создаем матрицу важности
    # Строки: уровни (0-49), Столбцы: каналы (price, volume, imbalance)
    channel_names = ['price', 'volume', 'imbalance']
    importance_matrix = np.zeros((n_levels, 3))
    
    # Заполняем матрицу
    for feature_name, data in importance_dict.items():
        channel = data.get('channel', '')
        level = data.get('level', -1)
        
        # Пропускаем past returns и другие каналы
        if channel not in channel_names or level < 0 or level >= n_levels:
            continue
        
        channel_idx = channel_names.index(channel)
        importance_matrix[level, channel_idx] = data['mean_importance']
    
    # Создаем heatmap
    fig, ax = plt.subplots(figsize=(10, 14))
    
    # Используем seaborn для красивой визуализации
    sns.heatmap(
        importance_matrix,
        cmap='YlOrRd',
        cbar_kws={'label': 'Mean Importance (MCC Drop)'},
        linewidths=0.1,
        linecolor='gray',
        ax=ax,
        xticklabels=['Ask Price', 'Ask Volume', 'Bid Price', 'Bid Volume'],
        yticklabels=[f'Level {i}' if i % 5 == 0 else '' for i in range(n_levels)]
    )
    
    ax.set_title('LOB Feature Importance Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Channel', fontsize=12, fontweight='bold')
    ax.set_ylabel('Order Book Level', fontsize=12, fontweight='bold')
    
    # Добавляем горизонтальную линию для выделения первых 10 уровней
    ax.axhline(y=10, color='blue', linestyle='--', linewidth=2, alpha=0.5)
    ax.text(4.2, 10, 'Top 10 Levels', fontsize=10, color='blue', 
            verticalalignment='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ LOB Heatmap сохранен: {save_path}")
    else:
        plt.show()
    
    plt.close()


class PurgedKFold:
    """
    Purged K-Fold Cross-Validation по методу Marcos López de Prado.
    
    Предотвращает утечку данных в временных рядах через:
    1. Purging: удаление сэмплов из train, которые перекрываются с началом validation
    2. Embargo: удаление сэмплов из train сразу после validation блока
    
    Использует количество событий (events count) вместо временных интервалов
    для устойчивости к пропускам в данных и изменениям плотности LOB данных.
    
    Args:
        n_splits: количество фолдов (обычно 5)
        purge_buffer_events: количество событий для удаления перед validation
        embargo_buffer_events: количество событий для удаления после validation
    
    Example:
        >>> cv = PurgedKFold(n_splits=5, purge_buffer_events=100, embargo_buffer_events=50)
        >>> for train_idx, val_idx in cv.split(X, y, timestamps):
        ...     X_train, X_val = X[train_idx], X[val_idx]
        ...     y_train, y_val = y[train_idx], y[val_idx]
        ...     # Обучение модели на train, валидация на val
    
    References:
        Marcos López de Prado, "Advances in Financial Machine Learning" (2018), Chapter 7
    """
    
    def __init__(self, n_splits=5, purge_buffer_events=0, embargo_buffer_events=0):
        if n_splits < 2:
            raise ValueError(f"n_splits должен быть >= 2, получено {n_splits}")
        if purge_buffer_events < 0:
            raise ValueError(f"purge_buffer_events должен быть >= 0, получено {purge_buffer_events}")
        if embargo_buffer_events < 0:
            raise ValueError(f"embargo_buffer_events должен быть >= 0, получено {embargo_buffer_events}")
        
        self.n_splits = n_splits
        self.purge_buffer_events = purge_buffer_events
        self.embargo_buffer_events = embargo_buffer_events
    
    def split(self, X, y=None, timestamps=None):
        """
        Генератор фолдов с purging и embargo.
        
        Args:
            X: features (может быть None, используем только для получения длины)
            y: labels (опционально)
            timestamps: временные метки (опционально, используется для проверки сортировки)
        
        Yields:
            (train_indices, val_indices): кортежи numpy массивов индексов
        
        Notes:
            - Данные должны быть отсортированы по времени!
            - Если timestamps предоставлены, выполняется проверка сортировки
        """
        # Определяем количество сэмплов
        if X is not None:
            n_samples = len(X)
        elif y is not None:
            n_samples = len(y)
        elif timestamps is not None:
            n_samples = len(timestamps)
        else:
            raise ValueError("Необходимо предоставить хотя бы один из: X, y, timestamps")
        
        # Проверка сортировки по времени (если timestamps предоставлены)
        if timestamps is not None:
            if not np.all(timestamps[:-1] <= timestamps[1:]):
                raise ValueError(
                    "Данные должны быть отсортированы по времени! "
                    "Используйте df.sort('timestamp_ms') перед созданием датасета."
                )
        
        indices = np.arange(n_samples)
        
        # Разбиваем на K равных частей
        fold_size = n_samples // self.n_splits
        
        # Статистика для логирования
        total_removed = 0
        
        for fold_idx in range(self.n_splits):
            # Определяем границы validation фолда
            val_start = fold_idx * fold_size
            val_end = (fold_idx + 1) * fold_size if fold_idx < self.n_splits - 1 else n_samples
            
            val_indices = indices[val_start:val_end]
            
            # Начальный train: все кроме validation
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[val_start:val_end] = False
            
            # Purge: удаляем сэмплы перед validation
            purge_start = max(0, val_start - self.purge_buffer_events)
            if self.purge_buffer_events > 0:
                train_mask[purge_start:val_start] = False
            
            # Embargo: удаляем сэмплы после validation
            embargo_end = min(n_samples, val_end + self.embargo_buffer_events)
            if self.embargo_buffer_events > 0:
                train_mask[val_end:embargo_end] = False
            
            train_indices = indices[train_mask]
            
            # Подсчет удаленных данных
            removed = (val_end - val_start) + (val_start - purge_start) + (embargo_end - val_end)
            total_removed += removed
            
            yield train_indices, val_indices
        
        # Логирование процента удаленных данных (leakage prevention overhead)
        overhead_pct = (total_removed / (n_samples * self.n_splits)) * 100
        print(f"\n=== Purged K-Fold Statistics ===")
        print(f"Total samples: {n_samples}")
        print(f"Folds: {self.n_splits}")
        print(f"Purge buffer: {self.purge_buffer_events} events")
        print(f"Embargo buffer: {self.embargo_buffer_events} events")
        print(f"Leakage prevention overhead: {overhead_pct:.2f}%")
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Возвращает количество фолдов (для совместимости с sklearn).
        
        Args:
            X: не используется, для совместимости с sklearn
            y: не используется, для совместимости с sklearn
            groups: не используется, для совместимости с sklearn
        
        Returns:
            int: количество фолдов
        """
        return self.n_splits


def calculate_purged_kfold_stats(cv, X, y=None, timestamps=None):
    """
    Вычисляет статистику для Purged K-Fold кросс-валидации.
    
    Args:
        cv: экземпляр PurgedKFold
        X: features
        y: labels (опционально)
        timestamps: временные метки (опционально)
    
    Returns:
        dict: словарь со статистикой по фолдам
    """
    n_samples = len(X) if X is not None else len(y)
    
    stats = {
        'n_splits': cv.n_splits,
        'total_samples': n_samples,
        'purge_buffer': cv.purge_buffer_events,
        'embargo_buffer': cv.embargo_buffer_events,
        'folds': []
    }
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, timestamps)):
        fold_stats = {
            'fold': fold_idx,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'train_pct': (len(train_idx) / n_samples) * 100,
            'val_pct': (len(val_idx) / n_samples) * 100
        }
        
        # Если есть метки, вычисляем распределение классов
        if y is not None:
            train_labels = y[train_idx]
            val_labels = y[val_idx]
            
            fold_stats['train_class_dist'] = {
                int(cls): int(np.sum(train_labels == cls)) 
                for cls in np.unique(train_labels)
            }
            fold_stats['val_class_dist'] = {
                int(cls): int(np.sum(val_labels == cls)) 
                for cls in np.unique(val_labels)
            }
        
        stats['folds'].append(fold_stats)
    
    return stats


def adaptive_gradient_clipping(model, clip_factor=0.01, eps=1e-6):
    """
    Adaptive Gradient Clipping (AGC) для стабилизации обучения на волатильных данных.
    
    Реализация основана на NFNet (https://arxiv.org/abs/2102.06171).
    AGC масштабирует градиенты пропорционально норме весов, предотвращая взрывные градиенты
    без подавления обучения.
    
    Алгоритм:
    1. Для каждого параметра вычисляется отношение: ||G|| / ||W||*
       где ||W||* = max(||W||, eps) для защиты от деления на ноль
    2. Если отношение > clip_factor, градиент масштабируется:
       G_new = G * (clip_factor * ||W||*) / ||G||
    
    Исключения (критично для стабильности):
    - Одномерные параметры (bias, LayerNorm) НЕ клиппируются
    - Embeddings (включая многомерные: cls_token, pos_emb) НЕ клиппируются
    - Применяется только к многомерным весовым матрицам
    
    Args:
        model: PyTorch модель
        clip_factor: порог отношения ||G|| / ||W|| (обычно 0.01-0.1)
        eps: минимальное значение нормы весов для защиты от деления на ноль
    
    Returns:
        dict: статистика клиппинга для мониторинга
            - clipped_count: количество обрезанных параметров
            - total_count: общее количество параметров
            - max_ratio: максимальное отношение ||G|| / ||W|| среди всех параметров
            - max_ratio_attention: максимальное отношение среди Attention слоев
    
    References:
        Brock et al., "High-Performance Large-Scale Image Recognition Without Normalization" (2021)
        https://arxiv.org/abs/2102.06171
    """
    clipped_count = 0
    total_count = 0
    max_ratio = 0.0
    max_ratio_attention = 0.0
    
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        
        # Исключаем одномерные параметры (bias, LayerNorm)
        if param.ndim <= 1:
            continue
        
        # Исключаем embeddings (включая многомерные: cls_token, pos_emb, level_pos_emb)
        # Согласно практике NFNet/HF, embeddings не должны клиппироваться
        name_lower = name.lower()
        if any(keyword in name_lower for keyword in ['embed', 'pos_emb', 'cls_token', 'position']):
            continue
        
        total_count += 1
        
        # Вычисляем нормы
        param_norm = param.detach().norm(2.0)
        grad_norm = param.grad.detach().norm(2.0)
        
        # Защита от деления на ноль
        param_norm_clamped = torch.clamp(param_norm, min=eps)
        
        # Вычисляем отношение ||G|| / ||W||*
        ratio = grad_norm / param_norm_clamped
        
        # Обновляем максимальное отношение
        max_ratio = max(max_ratio, ratio.item())
        
        # Отслеживаем отношение для Attention слоев
        if 'attention' in name_lower or 'attn' in name_lower:
            max_ratio_attention = max(max_ratio_attention, ratio.item())
        
        # Применяем клиппинг если ratio > clip_factor
        if ratio > clip_factor:
            clipped_count += 1
            # Масштабируем градиент: G_new = G * (clip_factor * ||W||*) / ||G||
            max_norm = clip_factor * param_norm_clamped
            param.grad.detach().mul_(max_norm / grad_norm.clamp(min=1e-6))
    
    # Формируем статистику
    stats = {
        'clipped_count': clipped_count,
        'total_count': total_count,
        'clipped_pct': (clipped_count / total_count * 100) if total_count > 0 else 0.0,
        'max_ratio': max_ratio,
        'max_ratio_attention': max_ratio_attention
    }
    
    return stats


def log_grad_stats(model, clip_stats=None, logger=None, global_step=None):
    """
    Расширенная диагностика градиентов для мониторинга обучения.
    
    Вычисляет и логирует:
    1. Процент параметров, к которым был применен клиппинг
    2. Максимальное значение отношения ||G|| / ||W|| среди всех слоев
    3. Максимальное значение отношения среди слоев Attention (критично для Transformer)
    4. Средние нормы градиентов по типам слоев
    
    Цель: Понять, не "задыхается" ли модель от слишком жесткого ограничения градиентов.
    
    Args:
        model: PyTorch модель
        clip_stats: словарь со статистикой клиппинга (из adaptive_gradient_clipping)
        logger: логгер для записи метрик (опционально)
        global_step: текущий шаг обучения для логирования (опционально)
    
    Returns:
        dict: словарь с метриками градиентов
    """
    grad_norms = {
        'embedding': [],
        'attention': [],
        'feedforward': [],
        'layernorm': [],
        'other': []
    }
    
    total_grad_norm = 0.0
    param_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        
        grad_norm = param.grad.detach().norm(2.0).item()
        total_grad_norm += grad_norm ** 2
        param_count += 1
        
        # Классифицируем параметр по типу слоя
        name_lower = name.lower()
        if 'embed' in name_lower:
            grad_norms['embedding'].append(grad_norm)
        elif 'attention' in name_lower or 'attn' in name_lower:
            grad_norms['attention'].append(grad_norm)
        elif 'mlp' in name_lower or 'fc' in name_lower or 'linear' in name_lower:
            grad_norms['feedforward'].append(grad_norm)
        elif 'norm' in name_lower:
            grad_norms['layernorm'].append(grad_norm)
        else:
            grad_norms['other'].append(grad_norm)
    
    # Вычисляем глобальную норму градиентов
    global_grad_norm = (total_grad_norm ** 0.5) if param_count > 0 else 0.0
    
    # Вычисляем средние нормы по типам слоев
    stats = {
        'global_grad_norm': global_grad_norm,
        'mean_grad_embedding': np.mean(grad_norms['embedding']) if grad_norms['embedding'] else 0.0,
        'mean_grad_attention': np.mean(grad_norms['attention']) if grad_norms['attention'] else 0.0,
        'mean_grad_feedforward': np.mean(grad_norms['feedforward']) if grad_norms['feedforward'] else 0.0,
        'mean_grad_layernorm': np.mean(grad_norms['layernorm']) if grad_norms['layernorm'] else 0.0,
    }
    
    # Добавляем статистику клиппинга если доступна
    if clip_stats is not None:
        stats.update({
            'clip_pct': clip_stats['clipped_pct'],
            'clip_max_ratio': clip_stats['max_ratio'],
            'clip_max_ratio_attention': clip_stats['max_ratio_attention']
        })
    
    # Логируем метрики если доступен логгер
    if logger is not None and global_step is not None:
        for key, value in stats.items():
            logger.log_metrics({f'grad/{key}': value}, step=global_step)
    
    return stats


def print_pruning_stats(study, n_warmup_steps=25):
    """
    Выводит статистику pruning для Optuna study (Задача 156, пункт 5).
    
    Args:
        study: Optuna study объект
        n_warmup_steps: Количество warmup шагов для оценки эффективности
    
    Выводит:
        - Общее количество trials
        - Количество завершенных trials
        - Количество pruned trials и процент
        - Среднюю/минимальную/максимальную эпоху отсечения
        - Оценку эффективности pruning
    """
    import optuna
    import numpy as np
    
    print("\n" + "="*60)
    print("PRUNING STATISTICS")
    print("="*60)
    
    # Получаем все trials
    all_trials = study.get_trials()
    pruned_trials = study.get_trials(states=[optuna.trial.TrialState.PRUNED])
    completed_trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
    
    n_total = len(all_trials)
    n_pruned = len(pruned_trials)
    n_completed = len(completed_trials)
    pruned_pct = (n_pruned / n_total * 100) if n_total > 0 else 0
    
    print(f"Total trials: {n_total}")
    print(f"Completed trials: {n_completed}")
    print(f"Pruned trials: {n_pruned} ({pruned_pct:.1f}%)")
    
    # Анализ средней эпохи отсечения
    if n_pruned > 0:
        # Для каждого pruned trial находим последний step, на котором был report
        pruned_steps = []
        for trial in pruned_trials:
            if trial.intermediate_values:
                last_step = max(trial.intermediate_values.keys())
                pruned_steps.append(last_step)
        
        if pruned_steps:
            avg_pruned_step = np.mean(pruned_steps)
            min_pruned_step = np.min(pruned_steps)
            max_pruned_step = np.max(pruned_steps)
            
            print(f"\nPruning timing:")
            print(f"  - Average pruning step: {avg_pruned_step:.1f}")
            print(f"  - Min pruning step: {min_pruned_step}")
            print(f"  - Max pruning step: {max_pruned_step}")
            
            # Оценка эффективности pruning
            if avg_pruned_step < n_warmup_steps:
                print(f"\n⚠️  WARNING: Average pruning step ({avg_pruned_step:.1f}) is below warmup steps ({n_warmup_steps})")
                print(f"  This suggests pruner may be too aggressive. Consider:")
                print(f"    - Increasing --n_warmup_steps")
                print(f"    - Using --pruner_type patience for noisy metrics")
            else:
                print(f"\n✓ Pruning is working effectively (avg step > warmup)")
    
    print("="*60 + "\n")


# ============================================================================
# TensorBoard Visualization Functions (Задача 158)
# ============================================================================

def setup_activation_hooks(model, writer, epoch, hist_freq=10):
    """
    Настраивает forward hooks для мониторинга активаций слоев Patching и Attention.
    
    Записывает только mean, std, max на каждой эпохе для производительности.
    Полные гистограммы записываются редко (каждые hist_freq эпох).
    
    Args:
        model: LiTModel для мониторинга
        writer: TensorBoard SummaryWriter
        epoch: текущая эпоха
        hist_freq: частота записи полных гистограмм (по умолчанию каждые 10 эпох)
    
    Returns:
        list: список handle объектов для удаления hooks после использования
    """
    handles = []
    
    def make_hook(layer_name):
        """Создает hook функцию для конкретного слоя."""
        def hook(module, input, output):
            # Записываем статистику активаций
            if isinstance(output, torch.Tensor):
                act = output.detach()
                
                # Основные статистики (каждую эпоху)
                writer.add_scalar(f'activations/{layer_name}/mean', act.mean().item(), epoch)
                writer.add_scalar(f'activations/{layer_name}/std', act.std().item(), epoch)
                writer.add_scalar(f'activations/{layer_name}/max', act.abs().max().item(), epoch)
                
                # Полные гистограммы (редко, для экономии ресурсов)
                if epoch % hist_freq == 0:
                    writer.add_histogram(f'activations/{layer_name}/histogram', act, epoch)
        
        return hook
    
    # Регистрируем hooks для ключевых слоев
    # 1. Patching layer (критично для LOB данных)
    if hasattr(model, 'patching'):
        handle = model.patching.register_forward_hook(make_hook('patching'))
        handles.append(handle)
    
    # 2. Transformer encoder layers (особенно attention)
    if hasattr(model, 'transformer'):
        for idx, layer in enumerate(model.transformer.layers):
            # Self-attention
            if hasattr(layer, 'self_attn'):
                handle = layer.self_attn.register_forward_hook(make_hook(f'transformer_layer_{idx}/attention'))
                handles.append(handle)
            
            # Feed-forward network
            if hasattr(layer, 'linear1'):
                handle = layer.linear1.register_forward_hook(make_hook(f'transformer_layer_{idx}/ffn_1'))
                handles.append(handle)
            if hasattr(layer, 'linear2'):
                handle = layer.linear2.register_forward_hook(make_hook(f'transformer_layer_{idx}/ffn_2'))
                handles.append(handle)
    
    return handles


def log_gradient_norms(model, writer, epoch):
    """
    Записывает нормы градиентов для каждого именованного параметра.
    
    Помогает обнаружить vanishing/exploding gradients в глубоких слоях.
    
    Args:
        model: PyTorch модель
        writer: TensorBoard SummaryWriter
        epoch: текущая эпоха
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.detach().norm(2.0).item()
            writer.add_scalar(f'gradients/{name}/norm', grad_norm, epoch)


def plot_confusion_matrix_tensorboard(y_true, y_pred, class_names, writer, epoch, tag='confusion_matrix'):
    """
    Создает Confusion Matrix и отправляет в TensorBoard через add_figure.
    
    Использует matplotlib для создания интерактивного графика.
    
    Args:
        y_true: истинные метки (numpy array)
        y_pred: предсказанные метки (numpy array)
        class_names: список имен классов
        writer: TensorBoard SummaryWriter
        epoch: текущая эпоха
        tag: тег для логирования
    """
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Вычисляем confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Нормализуем по строкам (recall)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Создаем фигуру
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Сырая матрица
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Confusion Matrix (Raw Counts)')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # 2. Нормализованная матрица (Recall)
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized Recall)')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    
    # Отправляем в TensorBoard
    writer.add_figure(tag, fig, epoch)
    plt.close(fig)


def plot_pr_curves_tensorboard(y_true, y_pred_probs, class_names, writer, epoch, tag='pr_curves'):
    """
    Создает PR-кривые для каждого класса и отправляет в TensorBoard через add_figure.
    
    Использует sklearn.metrics.precision_recall_curve для расчета.
    
    Args:
        y_true: истинные метки (numpy array)
        y_pred_probs: вероятности предсказаний (numpy array, shape: (N, num_classes))
        class_names: список имен классов
        writer: TensorBoard SummaryWriter
        epoch: текущая эпоха
        tag: тег для логирования
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize
    import matplotlib.pyplot as plt
    
    num_classes = len(class_names)
    
    # Бинаризуем метки для multi-class PR-кривых
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    
    # Создаем фигуру
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Строим PR-кривую для каждого класса
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_probs[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_pred_probs[:, i])
        
        ax.plot(recall, precision, label=f'{class_name} (AP={ap:.3f})', linewidth=2)
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Отправляем в TensorBoard
    writer.add_figure(tag, fig, epoch)
    plt.close(fig)


def log_embeddings(model, dataloader, writer, epoch, max_samples=1000, tag='embeddings'):
    """
    Извлекает embeddings после слоя патчинга и отправляет в TensorBoard Projector.
    
    Ограничивает количество сэмплов для предотвращения зависания TensorBoard.
    
    Args:
        model: LiTModel
        dataloader: DataLoader для извлечения данных
        writer: TensorBoard SummaryWriter
        epoch: текущая эпоха
        max_samples: максимальное количество сэмплов для визуализации (по умолчанию 1000)
        tag: тег для логирования
    """
    model.eval()
    
    embeddings_list = []
    labels_list = []
    regime_ids_list = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Распаковываем батч
            if len(batch) == 5:
                x, y, _, _, regime_id = batch
            else:
                x, y, _, _ = batch
                regime_id = None
            
            # Получаем embeddings после патчинга
            # Используем forward hook для извлечения
            patching_output = None
            
            def hook(module, input, output):
                nonlocal patching_output
                patching_output = output.detach()
            
            handle = model.patching.register_forward_hook(hook)
            
            # Forward pass
            _ = model(x.to(model.device), regime_id=regime_id.to(model.device) if regime_id is not None else None)
            
            # Удаляем hook
            handle.remove()
            
            # Собираем embeddings (используем mean pooling по sequence dimension)
            if patching_output is not None:
                # patching_output shape: (Batch, num_patches, d_model)
                # Усредняем по патчам для получения одного вектора на сэмпл
                emb = patching_output.mean(dim=1).cpu()  # (Batch, d_model)
                embeddings_list.append(emb)
                labels_list.append(y.cpu())
                
                if regime_id is not None:
                    regime_ids_list.append(regime_id.cpu())
            
            # Ограничиваем количество сэмплов
            if len(embeddings_list) * x.size(0) >= max_samples:
                break
    
    # Объединяем все embeddings
    if embeddings_list:
        embeddings = torch.cat(embeddings_list, dim=0)[:max_samples]
        labels = torch.cat(labels_list, dim=0)[:max_samples]
        
        # Подготавливаем метаданные
        metadata = [f'Label_{label.item()}' for label in labels]
        
        # Добавляем regime_id если доступен
        if regime_ids_list:
            regime_ids = torch.cat(regime_ids_list, dim=0)[:max_samples]
            metadata = [f'Label_{label.item()}_Regime_{regime.item()}' 
                       for label, regime in zip(labels, regime_ids)]
        
        # Отправляем в TensorBoard Projector
        writer.add_embedding(
            embeddings,
            metadata=metadata,
            global_step=epoch,
            tag=tag
        )
        
        print(f"✓ Logged {len(embeddings)} embeddings to TensorBoard Projector")
    
    model.train()


def setup_custom_scalars_layout(writer):
    """
    Настраивает Custom Scalars Layout для структурированного дашборда TensorBoard.
    
    Группирует метрики по категориям:
    - Losses: train_loss, val_loss_cls, val_loss_vol
    - Performance: val_mcc, val_precision_*, val_recall_*
    - Learning: lr, weight_cls, weight_vol
    - Calibration: val_ece, val_mce
    
    Args:
        writer: TensorBoard SummaryWriter
    """
    layout = {
        'Losses': {
            'train_val_loss': ['Multiline', ['train_loss', 'val_loss_cls']],
            'volatility_loss': ['Multiline', ['train_loss_vol', 'val_mse_vol', 'val_mae_vol']],
        },
        'Performance': {
            'main_metrics': ['Multiline', ['val_mcc', 'val_f1_macro', 'val_balanced_acc']],
            'precision': ['Multiline', ['val_prec_flat', 'val_prec_up', 'val_prec_down']],
            'recall': ['Multiline', ['val_rec_flat', 'val_rec_up', 'val_rec_down']],
        },
        'Learning': {
            'learning_rate': ['Multiline', ['lr']],
            'task_weights': ['Multiline', ['weight_cls', 'weight_vol']],
        },
        'Calibration': {
            'calibration_errors': ['Multiline', ['val_ece', 'val_mce']],
        }
    }
    
    writer.add_custom_scalars(layout)
    print("✓ Custom Scalars Layout configured")


def log_hparams(writer, hparams_dict, metrics_dict):
    """
    Логирует гиперпараметры и итоговые метрики для сравнения запусков.
    
    Args:
        writer: TensorBoard SummaryWriter
        hparams_dict: словарь с гиперпараметрами
        metrics_dict: словарь с итоговыми метриками
    """
    writer.add_hparams(
        hparam_dict=hparams_dict,
        metric_dict=metrics_dict
    )
    print("✓ Hyperparameters logged to TensorBoard")



def cleanup_old_tensorboard_logs(log_dir, max_runs=50):
    """
    Автоматическая очистка старых TensorBoard логов для экономии места на диске.
    
    Удаляет самые старые запуски, оставляя только max_runs последних.
    Полезно при тысячах запусков Optuna.
    
    Args:
        log_dir: директория с TensorBoard логами
        max_runs: максимальное количество запусков для сохранения
    """
    from pathlib import Path
    import shutil
    
    log_path = Path(log_dir)
    if not log_path.exists():
        return
    
    # Получаем все поддиректории (каждая - отдельный запуск)
    run_dirs = [d for d in log_path.iterdir() if d.is_dir()]
    
    # Сортируем по времени модификации (старые первыми)
    run_dirs.sort(key=lambda d: d.stat().st_mtime)
    
    # Удаляем старые запуски
    if len(run_dirs) > max_runs:
        to_delete = run_dirs[:len(run_dirs) - max_runs]
        
        print(f"\n[TensorBoard Cleanup] Removing {len(to_delete)} old runs from {log_dir}")
        for run_dir in to_delete:
            try:
                shutil.rmtree(run_dir)
                print(f"  ✓ Removed: {run_dir.name}")
            except Exception as e:
                print(f"  ✗ Failed to remove {run_dir.name}: {e}")
        
        print(f"[TensorBoard Cleanup] Kept {max_runs} most recent runs\n")


# ============================================================================
# Model Pruning Functions (Задача 159)
# ============================================================================

def apply_iterative_pruning(model, current_amount, prune_mode='unstructured'):
    """
    Применяет итеративный magnitude-based pruning к модели.
    
    Использует torch.nn.utils.prune.l1_unstructured для удаления весов с наименьшей L1-нормой.
    Применяется только к Linear слоям, исключая LayerNorm, bias и embeddings.
    
    Args:
        model: PyTorch модель для прунинга
        current_amount: текущая доля весов для удаления (0.0 - 1.0)
        prune_mode: режим прунинга ('unstructured' или 'structured_2_4')
    
    Returns:
        dict: статистика прунинга
            - pruned_layers: количество обработанных слоев
            - total_params: общее количество параметров
            - pruned_params: количество обнуленных параметров
            - sparsity: итоговая разреженность модели
    
    Notes:
        - Исключает LayerNorm, bias, embeddings для сохранения калибровки активаций
        - Применяет прунинг только к весовым матрицам (weight параметрам)
        - Для structured_2_4 используется WeightNormSparsifier
    """
    import torch.nn.utils.prune as prune
    
    pruned_layers = 0
    total_params = 0
    pruned_params = 0
    
    # Список слоев для исключения
    exclude_keywords = ['norm', 'bias', 'embed', 'cls_token', 'pos_emb', 'position']
    
    for name, module in model.named_modules():
        # Применяем только к Linear слоям
        if not isinstance(module, nn.Linear):
            continue
        
        # Исключаем слои по ключевым словам
        name_lower = name.lower()
        if any(keyword in name_lower for keyword in exclude_keywords):
            continue
        
        # Применяем прунинг к весам
        if hasattr(module, 'weight') and module.weight is not None:
            if prune_mode == 'unstructured':
                # L1 unstructured pruning
                prune.l1_unstructured(module, name='weight', amount=current_amount)
                pruned_layers += 1
            # structured_2_4 обрабатывается отдельной функцией
    
    # Подсчитываем статистику
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_params += param.numel()
            pruned_params += (param == 0).sum().item()
    
    sparsity = pruned_params / total_params if total_params > 0 else 0.0
    
    stats = {
        'pruned_layers': pruned_layers,
        'total_params': total_params,
        'pruned_params': pruned_params,
        'sparsity': sparsity
    }
    
    return stats


def apply_structured_pruning_2_4(model):
    """
    Применяет структурированный прунинг 2:4 для ускорения на NVIDIA GPU.
    
    Использует WeightNormSparsifier из torch.ao.pruning для создания паттерна,
    где из каждых 4 подряд идущих значений 2 являются нулями.
    
    Args:
        model: PyTorch модель для прунинга
    
    Returns:
        sparsifier: объект WeightNormSparsifier для дальнейшего использования
    
    Notes:
        - Требует NVIDIA GPU с Compute Capability 8.0+ (Ampere или новее)
        - Дает потенциальное ускорение до 2x на поддерживаемом оборудовании
        - После fine-tuning нужно вызвать sparsifier.squash_mask()
        - Для экспорта в ONNX с реальным ускорением используйте torch.sparse.to_sparse_semi_structured
    
    References:
        https://pytorch.org/tutorials/advanced/semi_structured_sparse.html
        https://pytorch.org/docs/stable/sparse.html#semi-structured-sparsity
    """
    from torch.ao.pruning import WeightNormSparsifier
    
    # Создаем sparsifier с паттерном 2:4
    sparsifier = WeightNormSparsifier(
        sparsity_level=1.0,  # Применяем ко всем блокам
        sparse_block_shape=(1, 4),  # Блоки по 4 элемента
        zeros_per_block=2  # 2 нуля на каждый блок
    )
    
    # Формируем конфигурацию для Linear слоев
    # Исключаем LayerNorm, bias, embeddings и task-specific головы
    exclude_keywords = ['norm', 'bias', 'embed', 'cls_token', 'pos_emb', 'position', 'head', 'classifier']
    
    sparse_config = []
    for fqn, module in model.named_modules():
        if isinstance(module, nn.Linear):
            fqn_lower = fqn.lower()
            # Проверяем исключения
            if not any(keyword in fqn_lower for keyword in exclude_keywords):
                # Проверяем shape constraints для 2:4 sparsity
                # Требуется: weight.shape[-1] % 4 == 0
                if hasattr(module, 'weight') and module.weight.shape[-1] % 4 == 0:
                    sparse_config.append({"tensor_fqn": f"{fqn}.weight"})
    
    if not sparse_config:
        print("⚠️  WARNING: No layers found for 2:4 structured pruning (shape constraints not met)")
        return None
    
    # Prepare: вставляем параметризации для маскирования
    sparsifier.prepare(model, sparse_config)
    
    # Step: применяем маски
    sparsifier.step()
    
    print(f"✓ Applied 2:4 structured pruning to {len(sparse_config)} layers")
    
    return sparsifier


def remove_pruning_reparametrization(model):
    """
    Удаляет параметризации прунинга и делает маски постоянными.
    
    После итеративного прунинга и fine-tuning, эта функция "замораживает"
    разреженную структуру, удаляя параметризации и оставляя только
    обнуленные веса.
    
    Args:
        model: PyTorch модель с примененным прунингом
    
    Notes:
        - Вызывается после завершения всех итераций прунинга
        - Для unstructured pruning использует prune.remove()
        - Для structured_2_4 использует sparsifier.squash_mask()
    """
    import torch.nn.utils.prune as prune
    
    removed_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Проверяем наличие параметризации прунинга
            if hasattr(module, 'weight_orig'):
                prune.remove(module, 'weight')
                removed_count += 1
    
    if removed_count > 0:
        print(f"✓ Removed pruning reparametrization from {removed_count} layers")
    
    return removed_count


def calculate_sparsity(model, detailed=False):
    """
    Вычисляет разреженность (sparsity) модели.
    
    Подсчитывает процент нулевых весов в модели, опционально
    предоставляя детальную статистику по слоям.
    
    Args:
        model: PyTorch модель
        detailed: если True, возвращает статистику по каждому слою
    
    Returns:
        dict: статистика разреженности
            - global_sparsity: общая разреженность модели (0.0 - 1.0)
            - total_params: общее количество параметров
            - zero_params: количество нулевых параметров
            - layer_sparsity: (опционально) словарь {layer_name: sparsity}
    
    Example:
        >>> stats = calculate_sparsity(model, detailed=True)
        >>> print(f"Model sparsity: {stats['global_sparsity']:.2%}")
    """
    total_params = 0
    zero_params = 0
    layer_stats = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            param_count = param.numel()
            zero_count = (param == 0).sum().item()
            
            total_params += param_count
            zero_params += zero_count
            
            if detailed:
                layer_sparsity = zero_count / param_count if param_count > 0 else 0.0
                layer_stats[name] = {
                    'sparsity': layer_sparsity,
                    'total': param_count,
                    'zeros': zero_count
                }
    
    global_sparsity = zero_params / total_params if total_params > 0 else 0.0
    
    stats = {
        'global_sparsity': global_sparsity,
        'total_params': total_params,
        'zero_params': zero_params
    }
    
    if detailed:
        stats['layer_sparsity'] = layer_stats
    
    return stats


def save_pruned_model(model, save_path, sparsity_stats, baseline_mcc=None):
    """
    Сохраняет разреженную модель с метаданными для последующего квантования.
    
    Экспортирует промежуточную разреженную модель перед квантованием (задача 157).
    Сохраняет метаданные о разреженности и падении точности.
    
    Args:
        model: разреженная PyTorch модель
        save_path: путь для сохранения (например, "models/SYMBOL/pruned_model.pt")
        sparsity_stats: словарь со статистикой разреженности (из calculate_sparsity)
        baseline_mcc: MCC базовой модели для сравнения (опционально)
    
    Notes:
        - Сохраняет state_dict модели
        - Добавляет метаданные о прунинге
        - Подготавливает для задачи 157 (квантование)
    """
    from pathlib import Path
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Формируем данные для сохранения
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'sparsity_stats': sparsity_stats,
        'pruning_metadata': {
            'global_sparsity': sparsity_stats['global_sparsity'],
            'total_params': sparsity_stats['total_params'],
            'zero_params': sparsity_stats['zero_params'],
            'baseline_mcc': baseline_mcc,
            'ready_for_quantization': True
        }
    }
    
    # Сохраняем
    torch.save(checkpoint, save_path)
    
    print(f"\n{'='*60}")
    print(f"PRUNED MODEL SAVED")
    print(f"{'='*60}")
    print(f"Path: {save_path}")
    print(f"Global Sparsity: {sparsity_stats['global_sparsity']:.2%}")
    print(f"Total Params: {sparsity_stats['total_params']:,}")
    print(f"Zero Params: {sparsity_stats['zero_params']:,}")
    if baseline_mcc is not None:
        print(f"Baseline MCC: {baseline_mcc:.4f}")
    print(f"Ready for Quantization: Yes")
    print(f"{'='*60}\n")


def log_pruning_progress(iteration, total_iterations, current_amount, target_amount, 
                        sparsity_stats, mcc, baseline_mcc=None):
    """
    Логирует прогресс итеративного прунинга.
    
    Выводит информацию о текущей итерации прунинга, разреженности
    и падении точности относительно baseline.
    
    Args:
        iteration: текущая итерация (1-based)
        total_iterations: общее количество итераций
        current_amount: текущая доля прунинга
        target_amount: целевая доля прунинга
        sparsity_stats: статистика разреженности
        mcc: текущий MCC
        baseline_mcc: MCC базовой модели (опционально)
    """
    print(f"\n{'='*60}")
    print(f"PRUNING ITERATION {iteration}/{total_iterations}")
    print(f"{'='*60}")
    print(f"Current Prune Amount: {current_amount:.2%}")
    print(f"Target Prune Amount: {target_amount:.2%}")
    print(f"Actual Sparsity: {sparsity_stats['global_sparsity']:.2%}")
    print(f"Current MCC: {mcc:.4f}")
    
    if baseline_mcc is not None:
        mcc_drop = baseline_mcc - mcc
        mcc_drop_pct = (mcc_drop / baseline_mcc * 100) if baseline_mcc != 0 else 0
        print(f"MCC Drop: {mcc_drop:.4f} ({mcc_drop_pct:.2f}%)")
    
    print(f"{'='*60}\n")


def print_pruning_warning():
    """
    Выводит предупреждение о том, что unstructured pruning не дает ускорения в ONNX Runtime.
    
    Важное предупреждение для пользователей: unstructured magnitude pruning
    уменьшает только размер файла модели, но не дает автоматического ускорения
    инференса в стандартном ONNX Runtime, так как операции выполняются как плотные (dense).
    """
    print(f"\n{'='*60}")
    print(f"⚠️  IMPORTANT: UNSTRUCTURED PRUNING PERFORMANCE NOTE")
    print(f"{'='*60}")
    print(f"Unstructured magnitude pruning (L1) reduces model file size")
    print(f"but does NOT provide automatic inference speedup in ONNX Runtime.")
    print(f"")
    print(f"Reason: ONNX Runtime executes sparse operations as dense,")
    print(f"so zero weights are still processed.")
    print(f"")
    print(f"Benefits:")
    print(f"  ✓ Reduced model size on disk and in memory")
    print(f"  ✓ Better synergy with quantization (task 157)")
    print(f"  ✓ Easier calibration for INT8 conversion")
    print(f"")
    print(f"For actual speedup, consider:")
    print(f"  - Structured pruning 2:4 (--prune_mode structured_2_4)")
    print(f"  - NVIDIA GPU with Ampere+ architecture")
    print(f"  - Specialized sparse inference engines")
    print(f"{'='*60}\n")


def convert_to_sparse_semi_structured(model):
    """
    Конвертирует 2:4 разреженные веса в SparseSemiStructuredTensor для ускорения.
    
    Эта функция должна вызываться ПОСЛЕ squash_mask() для моделей с 2:4 sparsity.
    Конвертирует веса в специальный формат, который ускоряется на NVIDIA Sparse Tensor Cores.
    
    Args:
        model: PyTorch модель с 2:4 разреженностью
    
    Returns:
        int: количество конвертированных слоев
    
    Notes:
        - Требует PyTorch 2.1+
        - Работает только на CUDA
        - Дает реальное ускорение до 2x на Ampere+ GPU
        - Используйте перед экспортом в ONNX для максимальной производительности
    
    Example:
        >>> # После прунинга и squash_mask
        >>> sparsifier.squash_mask()
        >>> convert_to_sparse_semi_structured(model)
        >>> # Теперь модель готова к ускоренному инференсу
    
    References:
        https://pytorch.org/docs/stable/sparse.html#semi-structured-sparsity
    """
    try:
        from torch.sparse import to_sparse_semi_structured
    except ImportError:
        print("⚠️  torch.sparse.to_sparse_semi_structured not available (requires PyTorch 2.1+)")
        return 0
    
    converted_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Проверяем, что веса имеют 2:4 разреженность
            if hasattr(module, 'weight') and module.weight is not None:
                weight = module.weight
                
                # Проверяем разреженность
                sparsity = (weight == 0).float().mean().item()
                
                # Если разреженность близка к 50% (2:4), конвертируем
                if 0.45 < sparsity < 0.55:
                    try:
                        # Конвертируем в SparseSemiStructuredTensor
                        module.weight = nn.Parameter(to_sparse_semi_structured(weight))
                        converted_count += 1
                    except Exception as e:
                        print(f"⚠️  Failed to convert {name}: {e}")
    
    if converted_count > 0:
        print(f"✓ Converted {converted_count} layers to SparseSemiStructuredTensor")
        print(f"  Model is now ready for accelerated inference on NVIDIA Ampere+ GPU")
    
    return converted_count


def compute_multi_horizon_metrics(y_true, y_pred, num_horizons):
    """
    Вычисляет метрики для каждого горизонта отдельно (Задача 160).
    
    Args:
        y_true: numpy array формы (n_samples, num_horizons) - истинные метки
        y_pred: numpy array формы (n_samples, num_horizons) - предсказанные метки
        num_horizons: количество горизонтов
    
    Returns:
        dict: метрики для каждого горизонта
    """
    metrics = {}
    
    for h in range(num_horizons):
        # Извлекаем метки для текущего горизонта
        y_true_h = y_true[:, h]
        y_pred_h = y_pred[:, h]
        
        # Фильтруем маскированные примеры (-100)
        mask = y_true_h != -100
        y_true_h_filtered = y_true_h[mask]
        y_pred_h_filtered = y_pred_h[mask]
        
        if len(y_true_h_filtered) == 0:
            # Если все примеры маскированы, пропускаем
            continue
        
        # Вычисляем метрики
        mcc_h = matthews_corrcoef(y_true_h_filtered, y_pred_h_filtered)
        f1_h = f1_score(y_true_h_filtered, y_pred_h_filtered, average='macro', zero_division=0)
        balanced_acc_h = balanced_accuracy_score(y_true_h_filtered, y_pred_h_filtered)
        
        # Сохраняем метрики с префиксом горизонта
        metrics[f"mcc_h{h}"] = float(mcc_h)
        metrics[f"f1_h{h}"] = float(f1_h)
        metrics[f"balanced_acc_h{h}"] = float(balanced_acc_h)
        metrics[f"samples_h{h}"] = int(len(y_true_h_filtered))
    
    return metrics
