
# Задача 122: Реализация метрик ECE и MCE (Reliability Analysis) (v2.0)

## 1. Реализация метрик в `python_lab/src/utils.py`
Создай класс `CalibrationMetrics`, который вычисляет **ECE** и **MCE**:

```python
import torch
import numpy as np

class CalibrationMetrics:
    def __init__(self, n_bins=15):
        self.n_bins = n_bins
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def calculate(self, logits, labels):
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
                bin_data.append({"acc": 0, "conf": (bin_lower + bin_upper).item() / 2, "count": 0})

        return ece.item(), mce.item(), bin_data
```

## 2. Визуализация Reliability Diagram в `python_lab/src/utils.py`
Добавь функцию отрисовки с двойной осью (Accuracy и Count):

```python
import matplotlib.pyplot as plt

def plot_reliability_diagram(bin_data, ece, mce, save_path):
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
    ax2.set_yscale('log') # Логарифмическая шкала для лучшей видимости

    plt.savefig(save_path)
    plt.close()
```

## 3. Интеграция в `python_lab/train.py`
1.  **Validation**: В конце каждой эпохи вызывай `CalibrationMetrics.calculate()`.
2.  **Logging**: Выводи `ECE` и `MCE` в консоль и сохраняй в историю обучения.
3.  **Optuna**: При подборе гиперпараметров используй целевую функцию: `Score = val_loss + (ECE * 0.5)`. Это заставит Optuna искать не только точные, но и хорошо откалиброванные модели.

## 4. Особенности реализации
- **Saturating Sub**: При расчете разностей в корзинах убедись, что нет `NaN`, если корзина пуста (уже обработано через `prop_in_bin > 0`).
- **Paths**: Все отчеты сохраняй в `reports/SYMBOL/` без использования префикса `./`.
- **Brier Score**: Опционально можешь добавить расчет Brier Score (MSE между вероятностью и меткой) как дополнительный индикатор качества.

---

## Аргументация для Планировщика:
1.  **MCE Importance**: MCE показывает "наихудший случай". Если в корзине 0.9-1.0 точность всего 0.5, бот будет открывать огромные позиции (задача 110), ошибаясь в половине случаев. MCE это подсветит.
2.  **Twin Axis**: Гистограмма на заднем плане позволяет понять, не является ли высокая точность в какой-то корзине случайностью (если там всего 2 примера).
3.  **Log Scale**: Использование логарифмической шкалы для `counts` необходимо, так как в корзине `Flat` (0.8-1.0) обычно на порядок больше примеров, чем в краевых корзинах `Up/Down`.

**Gemini, реализуй эти метрики, сделав их частью стандартного отчета об обучении модели.**