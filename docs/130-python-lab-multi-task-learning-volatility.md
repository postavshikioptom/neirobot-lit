# Задача 130: Multi-Task Learning — Сигнал и Волатильность (v2.0)

## 1. Архитектура в `python_lab/src/model.py`
Модифицируем модель для работы с двумя задачами, используя **Global Average Pooling** после трансформера и специфические **Bottleneck** слои для каждой задачи.

```python
import torch
import torch.nn as nn

class MultiTaskTransformer(nn.Module):
    def __init__(self, backbone, hidden_dim, num_classes=3):
        super().__init__()
        self.backbone = backbone # Наш LiT Backbone (025/026)
        
        # Bottleneck слои для изоляции специфических шумов задач
        self.class_bottleneck = nn.Linear(hidden_dim, hidden_dim)
        self.vol_bottleneck = nn.Linear(hidden_dim, hidden_dim)
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.vol_regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (Batch, Seq_Len, Features)
        features = self.backbone(x) # (Batch, Patches, Hidden_Dim)
        
        # Global Average Pooling по патчам
        pooled = features.mean(dim=1)
        
        # Разделение на ветки
        logits = self.classifier(torch.relu(self.class_bottleneck(pooled)))
        vol = self.vol_regressor(torch.relu(self.vol_bottleneck(pooled)))
        
        return logits, vol
```

## 2. Расчет Target Volatility в `python_lab/src/dataset.py`
Волатильность рассчитывается как стандартное отклонение лог-доходностей на окне вперед, переведенное в логарифмическую шкалу для стабильности регрессии.

```python
def compute_target_vol(mid_prices, window=100):
    """
    mid_prices: массив средних цен.
    window: размер окна в тиках для расчета реализованной волатильности.
    """
    log_prices = np.log(mid_prices)
    log_returns = np.diff(log_prices)
    
    # Скользящее стандартное отклонение (std)
    vol = pd.Series(log_returns).rolling(window).std().shift(-window)
    # Target: Log-Vol (добавляем epsilon для защиты от log(0))
    target_vol = np.log(vol + 1e-8)
    return target_vol.fillna(0).values
```

## 3. Экспорт в ONNX для Rust (`python_lab/scripts/export_onnx.py`)
Для продакшена в Rust нам не нужна голова волатильности (она используется только как регуляризатор при обучении). Мы должны экспортировать только классификатор.

```python
# В скрипте экспорта
if args.signal_only:
    # Указываем только 'logits' в output_names
    # PyTorch автоматически удалит неиспользуемые узлы (vol_head) из графа
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path,
        input_names=['input'],
        output_names=['logits'], # Исключаем 'vol'
        opset_version=18
    )
    # Дополнительная очистка через onnx-simplifier (128)
```

## 4. Спорные моменты и корректировки (Grok + Zencoder)

*   **Pooling**: Согласен с Grok. После трансформера мы имеем последовательность патчей. Использование `mean(dim=1)` (Global Average Pooling) эффективнее, чем просто брать первый токен, так как это собирает информацию со всего окна LOB.
*   **Log-Vol**: Это критично. Регрессия на «сырую» волатильность (0.00001) часто приводит к исчезновению градиентов. Логарифмирование выравнивает масштаб с лоссом классификации.
*   **Bottlenecks**: Добавление линейных слоев перед каждой головой (`class_bottleneck`, `vol_bottleneck`) позволяет модели разделить признаки, которые важны для направления, от тех, что важны для риска.
*   **Uncertainty Weighting**: Используем лосс с обучаемыми параметрами (из v1.0), чтобы модель сама балансировала важность Signal vs Volatility.

## 5. Инструкции для Gemini (Coder AI):
1.  **python_lab/src/model.py**: Реализовать `MultiTaskTransformer` с bottleneck слоями и pooling.
2.  **python_lab/src/dataset.py**: Добавить расчет `target_vol` (логарифм реализованной волатильности).
3.  **python_lab/train.py**: Добавить мониторинг метрик `Vol_MSE` и `Vol_MAE`. Обучать через `MultiTaskLoss` с авто-взвешиванием.
4.  **export_onnx.py**: Реализовать частичный экспорт через `output_names=['logits']`.

**Результат**: Мы получаем более «умный» backbone, который понимает структуру риска на рынке, что повышает точность классификации сигналов в условиях высокой волатильности.
