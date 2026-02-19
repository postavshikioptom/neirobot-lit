
# Задача 121: Калибровка вероятностей (Temperature Scaling) (v2.0)

## 1. Реализация метрик в `python_lab/src/utils.py`
Добавь функции для расчета **ECE (Expected Calibration Error)** и визуализации:

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

def calculate_ece(probs, labels, bins=15):
    """
    probs: тензор вероятностей после Softmax (N, C)
    labels: истинные метки (N,)
    """
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros(1, device=probs.device)
    
    bin_boundaries = torch.linspace(0, 1, bins + 1)
    for i in range(bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i+1]
        # Маска для элементов в текущей корзине (bin)
        in_bin = confidences.gt(lower.item()) * confidences.le(upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    return ece.item()

def plot_reliability_diagram(probs, labels, path, bins=15):
    # Логика отрисовки столбчатой диаграммы: Confidence vs Accuracy
    # Сохранение в path (например, reports/SYMBOL/calibration.png)
    pass
```

## 2. Скрипт калибровки `python_lab/calibrate.py`
Создай CLI скрипт, который находит оптимальную температуру $T$, минимизируя **NLL (Negative Log Likelihood)** на валидационной выборке:

```python
from torch import nn, optim

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

def find_temperature(logits, labels):
    scaler = TemperatureScaler()
    # LBFGS — стандарт для поиска температуры
    optimizer = optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)
    criterion = nn.CrossEntropyLoss()

    def eval_loss():
        optimizer.zero_grad()
        loss = criterion(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(eval_loss)
    return scaler.temperature.item()
```

## 3. Сохранение и экспорт
- **Metadata**: Сохраняй полученное значение `temperature` в `bots/SYMBOL/model/metadata.json` (задача 056).
- **ONNX**: В `export_onnx.py` добавь флаг `--embed_temperature`. Если `True`, добавь в граф узел `Div(T)` перед `Softmax`. Если `False`, оставь сырые логиты (в этом случае деление на $T$ должен делать Rust-движок, читая конфиг).

## 4. Обновление Rust ML Engine (`src/ml/onnx.rs`)
Если температура не вшита в ONNX:
1.  Загрузи `temperature` из `metadata.json` (или `BotConfig`).
2.  После получения выходного тензора из ONNX-сессии:
```rust
// pseudo-code
let calibrated_logits = raw_logits / model_temperature;
let probabilities = softmax(calibrated_logits);
```

---

## Аргументация для Планировщика:
1.  **ECE Metric**: Это единственный объективный способ понять, можно ли доверять «уверенности» модели при расчете объема позиции (задача 110).
2.  **LBFGS Optimizer**: В отличие от Adam, LBFGS гораздо быстрее и точнее находит минимум для одного параметра ($T$).
3.  **Flexibility**: Хранение $T$ в `metadata.json` позволяет оперативно «подкручивать» калибровку на лету, если мониторинг (Phase 10) показывает отклонение реальной точности от предсказанной.

**Gemini, реализуй этот механизм, обеспечив вывод ECE и NLL в консоль до и после калибровки для наглядности.**