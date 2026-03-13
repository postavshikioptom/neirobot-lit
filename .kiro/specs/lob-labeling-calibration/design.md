# Дизайн: Улучшение качества лабелирования и калибровки LiT модели

## Архитектурный обзор

```
┌─────────────────────────────────────────────────────────────┐
│                    Улучшенный Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Загрузка сырых данных (parquet)                         │
│         ↓                                                    │
│  2. Расчет весов классов (inverse frequency + smoothing)    │
│         ↓                                                    │
│  3. Создание датасета с аугментацией                        │
│         ↓                                                    │
│  4. Обучение модели с Focal Loss + class weights            │
│         ↓                                                    │
│  5. Temperature Scaling на валидации                        │
│         ↓                                                    │
│  6. Сохранение metadata (weights, temperature)              │
│         ↓                                                    │
│  7. Анализ качества (confusion matrix, ECE, MCE)            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Компоненты

### 1. Улучшенное взвешивание классов (dataset.py)

**Изменения:**
- Добавить функцию `compute_class_weights()` с параметрами:
  - `method: str = "inverse_frequency"` (или "balanced", "manual")
  - `smoothing: float = 1.0` (для сглаживания)
  - `manual_weights: dict = None` (для ручного задания)

**Формула:**
```
weight_c = (1 / (frequency_c + smoothing)) / sum(weights)
```

**Интерфейс:**
```python
weights = compute_class_weights(
    labels,
    method="inverse_frequency",
    smoothing=1.0
)
# weights = [w0, w1, w2] для классов Flat, Up, Down
```

### 2. Аугментация LOB данных (dataset.py)

**Изменения:**
- Добавить функции аугментации:
  - `augment_volume_jitter()` - добавление шума к объемам
  - `augment_level_dropout()` - случайное удаление уровней
  - `augment_symmetric_flip()` - симметричное отражение bid/ask

**Параметры:**
```python
augmentation_config = {
    "volume_jitter_std": 0.05,  # 5% шума
    "level_dropout_prob": 0.1,  # 10% вероятность удаления уровня
    "symmetric_flip_prob": 0.2  # 20% вероятность отражения
}
```

### 3. Temperature Scaling (utils.py)

**Новый класс TemperatureScaler:**
```python
class TemperatureScaler:
    def __init__(self, device='cpu'):
        self.temperature = nn.Parameter(torch.ones(1))
    
    def calibrate(self, logits, labels, lr=0.01, epochs=100):
        """Найти оптимальную температуру на валидации"""
        # Оптимизация температуры через NLL loss
        
    def apply(self, logits):
        """Применить температуру к логитам"""
        return logits / self.temperature
```

**Использование:**
```python
scaler = TemperatureScaler()
scaler.calibrate(val_logits, val_labels)
calibrated_logits = scaler.apply(test_logits)
```

### 4. Анализ качества (utils.py)

**Новые функции:**
- `compute_ece()` - Expected Calibration Error
- `compute_mce()` - Maximum Calibration Error
- `plot_reliability_diagram()` - диаграмма надежности
- `analyze_label_distribution()` - анализ распределения меток

### 5. Metadata сохранение (train.py)

**Структура metadata.json:**
```json
{
  "class_weights": {
    "method": "inverse_frequency",
    "smoothing": 1.0,
    "weights": [0.5, 1.5, 1.5]
  },
  "calibration": {
    "temperature": 1.2,
    "ece_before": 0.25,
    "ece_after": 0.12
  },
  "augmentation": {
    "volume_jitter_std": 0.05,
    "level_dropout_prob": 0.1,
    "symmetric_flip_prob": 0.2
  }
}
```

## Изменения в существующих файлах

### labels.py
- Оставить как есть (threshold и horizon уже настроены)

### dataset.py
- Добавить функцию `compute_class_weights()`
- Добавить функции аугментации
- Интегрировать аугментацию в `LOBDataset.__getitem__()`

### utils.py
- Добавить класс `TemperatureScaler`
- Добавить функции для расчета ECE, MCE
- Добавить функции анализа качества

### train.py
- Использовать адаптивный Labeler
- Вычислять и применять class weights
- Применять Temperature Scaling на валидации
- Сохранять все параметры в metadata.json
- Добавить логирование качества лабелирования

## Обратная совместимость

- Все новые параметры имеют значения по умолчанию
- Старые скрипты будут работать без изменений
- Параметр `dynamic_threshold=False` включает старое поведение

## Тестирование

1. Unit-тесты для каждой функции
2. Integration-тесты для полного pipeline
3. Сравнение метрик до/после улучшений
4. Проверка совместимости со старыми моделями

## Риски и смягчение

| Риск | Вероятность | Влияние | Смягчение |
|------|-------------|--------|----------|
| Адаптивный threshold может быть нестабилен | Средняя | Высокое | Добавить min/max bounds |
| Аугментация может ухудшить качество | Низкая | Среднее | Параметризовать, тестировать |
| Temperature Scaling может переобучиться | Низкая | Среднее | Использовать отдельный val set |
| Совместимость со старыми моделями | Низкая | Высокое | Версионирование metadata |
