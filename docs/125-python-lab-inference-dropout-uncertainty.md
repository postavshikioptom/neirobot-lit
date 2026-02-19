# Задача 125: Оценка неопределенности через MC Dropout (v2.1)

## 1. Математика неопределенности в `python_lab/src/utils.py`
Добавь расчет **Mutual Information (MI)**. Это критично для отделения случайного шума данных (алеаторная) от непонимания модели (эпистемическая неопределенность).

```python
import torch
import torch.nn as nn

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
```

## 2. Анализ качества в `python_lab/evaluate_uncertainty.py`
Реализуй **Rejection Curve**. Мы должны видеть, насколько вырастет точность (Accuracy), если мы откажемся от самых «мутных» сделок.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_rejection_curve(labels, probs_mc, metric_values, save_path):
    """
    metric_values: значения энтропии или MI для каждого сэмпла.
    """
    # Сортируем по убыванию неопределенности (сначала самые плохие)
    indices = np.argsort(metric_values)[::-1]
    sorted_labels = labels[indices]
    sorted_preds = probs_mc.mean(0).argmax(1)[indices]
    
    correct = (sorted_labels == sorted_preds)
    rejection_rates = np.linspace(0, 0.95, 50)
    acc_scores = []
    
    for rate in rejection_rates:
        # Отсекаем % худших
        keep_idx = int(len(correct) * rate)
        remaining_acc = correct[keep_idx:].mean()
        acc_scores.append(remaining_acc)
        
    plt.figure(figsize=(8, 5))
    plt.plot(rejection_rates * 100, acc_scores)
    plt.xlabel('Rejection Rate (%)')
    plt.ylabel('Accuracy on Remaining')
    plt.title('Rejection Curve (Uncertainty-based Filter)')
    plt.grid(True)
    plt.savefig(save_path)
```

## 3. Процедура прогрева (Warm-up) в `python_lab/train.py`
Перед замерами неопределенности обязательно сделай **warm-up**, иначе первый прогон даст ложную дисперсию из-за инициализации весов/памяти.

```python
def enable_dropout(m):
    if isinstance(m, nn.Dropout):
        m.train()

# Перевод в режим замера
model.apply(enable_dropout)

# Warm-up для стабилизации CUDA/JIT
with torch.no_grad():
    for _ in range(5):
        _ = model(dummy_input)

# Теперь выполняем N прогонов (например, 20-50)
```

## 4. Корректировка для Production (Rust)
*   **ONNX Export**: Запрещено экспортировать модель с `training=True`. Dropout должен быть выключен для детерминизма и минимальной задержки.
*   **Calibration**: Основной метод в Rust — **Temperature Scaling** (из задачи 121). 
*   **Static Thresholds**: По результатам Rejection Curve в Python, мы находим порог `entropy_limit` или `mi_limit` и жестко прописываем его в `BotConfig` как фильтр (если логиты после калибровки дают слишком высокую энтропию — скип).

## 5. Что изменить в коде (Gemini):
*   **python_lab/src/utils.py**: Добавить функции расчета MI и энтропии.
*   **python_lab/evaluate_uncertainty.py**: Создать скрипт для генерации графиков отказа.
*   **python_lab/train.py**: Добавить логику переключения Dropout в режим инференса и цикл прогрева.

**Статус**: Готово к передаче кодеру. Жду твоего подтверждения.