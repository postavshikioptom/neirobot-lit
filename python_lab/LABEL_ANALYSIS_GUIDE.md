# Руководство по анализу распределения меток

## Описание

Скрипт `analyze_labels.py` выполняет статистический анализ распределения классов меток (Up/Down/Flat) и матрицы переходов между состояниями. Это критически важно для:

1. **Калибровки весов потерь** - определение `pos_weight` для `CrossEntropyLoss`
2. **Детекции дисбаланса** - выявление доминирующих классов (например, Flat >90%)
3. **Анализа переходов** - проверка, не "залипает" ли модель в одном состоянии

## Использование

### Базовый запуск

```bash
python scripts/analyze_labels.py --data_path bots/CAKEUSDT/data/raw
```

### С дополнительными параметрами

```bash
python scripts/analyze_labels.py \
    --data_path bots/CAKEUSDT/data/raw \
    --output_dir bots/CAKEUSDT/analysis \
    --consistency_threshold 3.0
```

## Параметры

- `--data_path` (обязательный) - путь к директории с `train.parquet` и `val.parquet`
- `--output_dir` (опциональный) - путь для сохранения результатов (по умолчанию: data_path)
- `--consistency_threshold` (опциональный) - порог различия в % для предупреждения (по умолчанию: 5.0)

## Выходные данные

### 1. Метаданные (JSON)

Файл: `{output_dir}/label_analysis_metadata.json`

```json
{
  "train": {
    "distribution": [
      {"label": 0, "len": 1000, "percentage": 10.5},
      {"label": 1, "len": 950, "percentage": 10.0},
      {"label": 2, "len": 7550, "percentage": 79.5}
    ],
    "imbalance_ratio": 7.95
  },
  "val": {
    "distribution": [...],
    "imbalance_ratio": 8.12
  }
}
```

### 2. Визуализации

#### Train выборка:
- `{output_dir}/train/label_distribution.png` - гистограмма распределения классов
- `{output_dir}/train/transition_matrix.png` - тепловая карта матрицы переходов

#### Val выборка:
- `{output_dir}/val/label_distribution.png`
- `{output_dir}/val/transition_matrix.png`

## Интерпретация результатов

### Imbalance Ratio

- **< 2.0** - сбалансированные классы
- **2.0 - 5.0** - умеренный дисбаланс
- **> 5.0** - сильный дисбаланс (требуется взвешивание потерь)

### Матрица переходов

Показывает вероятность перехода из одного состояния в другое:

```
           0      1      2
0      0.150  0.120  0.730
1      0.125  0.145  0.730
2      0.095  0.090  0.815
```

- Строка = текущее состояние
- Столбец = следующее состояние
- Значение = вероятность перехода

**Важно**: Если диагональные элементы (например, 2→2) близки к 1.0, модель может быть слишком консервативной.

## Проверка консистентности

Скрипт автоматически сравнивает распределения train и val:

- ✅ Различие < 5% - нормально
- ⚠️ Различие > 5% - возможны проблемы с разделением данных

## Использование в train.py

Метаданные из JSON можно использовать для настройки весов:

```python
import json

with open("label_analysis_metadata.json") as f:
    metadata = json.load(f)

# Вычисляем веса для CrossEntropyLoss
train_dist = metadata["train"]["distribution"]
total = sum(item["len"] for item in train_dist)
weights = [total / item["len"] for item in train_dist]

# Нормализуем
weights = torch.tensor(weights) / sum(weights) * len(weights)
```

## Требования

Убедитесь, что установлены все зависимости:

```bash
pip install polars pandas seaborn matplotlib
```

Или используйте `requirements.txt`:

```bash
pip install -r requirements.txt
```
