# Задача 123: Взвешивание примеров по времени (Time-Weighted Loss) (v2.0)

## 1. Изменения в `python_lab/train.py` (CLI аргументы)
Добавь поддержку параметров временного затухания:
```python
parser.add_argument('--use_time_weighting', action='store_true', help='Enable time-decay weighting')
parser.add_argument('--half_life_hours', type=float, default=24.0, help='Weight decay half-life in hours')
parser.add_argument('--min_sample_weight', type=float, default=0.1, help='Minimum weight for old samples')
```

## 2. Реализация в `python_lab/src/dataset.py`
Модифицируй `LOBDataset` для расчета весов. Если переданы `class_weights`, они должны перемножаться с временными:

```python
class LOBDataset(Dataset):
    def __init__(self, data, half_life_hours=24.0, min_weight=0.1, class_weights=None):
        self.features = torch.tensor(data['features'], dtype=torch.float32)
        self.labels = torch.tensor(data['labels'], dtype=torch.long)
        self.timestamps = data['timestamps'] # np.array int64 (ms)
        
        # 1. Расчет временных весов
        max_ts = self.timestamps.max()
        half_life_ms = half_life_hours * 3600 * 1000
        decay_lambda = np.log(2) / half_life_ms
        
        deltas = max_ts - self.timestamps
        time_weights = np.exp(-decay_lambda * deltas)
        time_weights = np.clip(time_weights, min_weight, 1.0)
        
        # 2. Интеграция с весами классов (задача 052)
        if class_weights is not None:
            # class_weights — тензор весов [w_up, w_down, w_flat]
            sample_class_weights = class_weights[self.labels].numpy()
            final_weights = time_weights * sample_class_weights
        else:
            final_weights = time_weights
            
        # 3. Нормализация (среднее = 1.0 для стабильности градиентов)
        self.sample_weights = torch.tensor(final_weights / final_weights.mean(), dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index], self.sample_weights[index]
```

## 3. Обновление цикла обучения в `python_lab/train.py`
Используй `reduction='none'` в функции потерь для применения индивидуальных весов:

```python
# Инициализация Loss
criterion = nn.CrossEntropyLoss(reduction='none')

for batch_idx, (features, labels, weights) in enumerate(train_loader):
    features, labels, weights = features.to(device), labels.to(device), weights.to(device)
    
    optimizer.zero_grad()
    logits = model(features)
    
    # Поточечный лосс * веса примера
    loss_raw = criterion(logits, labels)
    loss_weighted = (loss_raw * weights).mean()
    
    loss_weighted.backward()
    optimizer.step()
```

## 4. Особенности реализации
- **Validation**: На валидационном сете веса **не используются**. В `DataLoader` для валидации должен передаваться тензор единичных весов или `reduction='mean'` в `CrossEntropyLoss`.
- **Normalization**: Использование `weights.mean()` предпочтительнее `sum()`, так как это сохраняет масштаб функции потерь независимо от размера батча.
- **Data Types**: Убедись, что временные метки (`timestamps`) извлекаются из Parquet как `int64`. Если они в формате `datetime`, конвертируй их в Unix MS перед расчетом.

---

## Аргументация для Планировщика:
1.  **Market Drift**: Рыночные условия (волатильность, спред) меняются. Присвоение большего веса последним 24-48 часам данных позволяет модели быстрее «забывать» неактуальные состояния.
2.  **Hybrid Weighting**: Перемножение временных и классовых весов — самый мощный инструмент. Оно гарантирует, что даже старый сигнал `Up` будет важнее для модели, чем свежий, но избыточный сигнал `Flat`.
3.  **Argparse Flexibility**: Передача параметров через CLI позволяет легко запускать эксперименты с разным `half_life` без правки кода.

**Gemini, реализуй эту логику, обеспечив корректное перемножение тензоров в `LOBDataset`.**