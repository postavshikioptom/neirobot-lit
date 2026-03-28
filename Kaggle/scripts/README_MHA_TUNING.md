# Задача 237: Оптимизация Multi-Head Attention

## Описание

Скрипт `tune_attention.py` реализует байесовскую оптимизацию архитектуры Multi-Head Attention модели LiT с использованием Optuna. Цель - найти оптимальное сочетание `embed_dim` и `num_heads` для максимизации качества предсказаний (MCC) при строгом ограничении задержки инференса (latency < 2.0ms на CPU).

## Основные возможности

1. **Байесовская оптимизация** с Optuna (TPE sampler)
2. **Пространство поиска** (согласно плану задачи 237):
   - `embed_dim`: [32, 64, 128, 256]
   - `num_heads`: [2, 4, 8, 16]
   - Остальные параметры фиксированы: `num_layers=2`, `dropout=0.1`
3. **Целевая функция**: `score = validation_mcc - lambda * inference_latency_ms`
4. **Latency constraint**: < 2.0ms на CPU через ONNX Runtime
5. **Явное использование SDPA**: В методе forward используется `torch.nn.functional.scaled_dot_product_attention` для активации Flash Attention
6. **Реализация GQA**: Опциональная поддержка Grouped Query Attention через параметр `use_gqa`
7. **Результаты**:
   - `best_mha_config.json` - оптимальная конфигурация
   - `reports/mha_pareto_front.png` - график Парето (Accuracy vs Latency)

## Использование

### Базовый запуск

```bash
python -m python_lab.scripts.tune_attention --symbol BTCUSDT --trials 30
```

### Параметры

- `--symbol`: Торговый символ (по умолчанию: BTCUSDT)
- `--trials`: Количество trials для Optuna (по умолчанию: 30)
- `--epochs`: Количество эпох обучения для каждого trial (по умолчанию: 5)
- `--lambda_latency`: Коэффициент штрафа за latency (по умолчанию: 0.1)
- `--max_latency`: Максимальная допустимая latency в ms (по умолчанию: 2.0)

### Примеры

**Быстрый тест (10 trials, 3 epochs)**:
```bash
python -m python_lab.scripts.tune_attention --symbol BTCUSDT --trials 10 --epochs 3
```

**Полная оптимизация (50 trials, 10 epochs)**:
```bash
python -m python_lab.scripts.tune_attention --symbol BTCUSDT --trials 50 --epochs 10
```

**Более строгий latency constraint (1.5ms)**:
```bash
python -m python_lab.scripts.tune_attention --symbol BTCUSDT --trials 30 --max_latency 1.5
```

**Изменение баланса MCC vs Latency**:
```bash
# Больший штраф за latency (lambda=0.2)
python -m python_lab.scripts.tune_attention --symbol BTCUSDT --trials 30 --lambda_latency 0.2

# Меньший штраф за latency (lambda=0.05)
python -m python_lab.scripts.tune_attention --symbol BTCUSDT --trials 30 --lambda_latency 0.05
```

## Результаты

### Файл конфигурации

Оптимальная конфигурация сохраняется в `bots/{SYMBOL}/model/best_mha_config.json`:

```json
{
    "embed_dim": 128,
    "num_heads": 8,
    "num_layers": 2,
    "dropout": 0.15,
    "validation_mcc": 0.4523,
    "inference_latency_ms": 1.234,
    "combined_score": 0.3289,
    "lambda_latency": 0.1,
    "max_latency_constraint": 2.0
}
```

### График Парето

График сохраняется в `reports/mha_pareto_front.png` и показывает:
- **Все trials** (цветные точки, цвет = combined score)
- **Парето-фронт** (красная пунктирная линия и звезды)
- **Лучший trial** по combined score (золотой ромб)

## Интерпретация результатов

### Combined Score

`score = MCC - lambda * latency_ms`

- **Высокий score**: хороший баланс между точностью и скоростью
- **Lambda = 0.1**: 1ms latency эквивалентна 0.1 MCC penalty
- **Lambda = 0.2**: более агрессивная оптимизация скорости

### Парето-фронт

Точки на Парето-фронте представляют оптимальные компромиссы:
- **Левая часть**: быстрые модели с меньшей точностью
- **Правая часть**: точные модели с большей latency
- **Выбор**: зависит от требований к системе

### Latency Constraint

Trials с latency > max_latency автоматически отсекаются (pruned), что гарантирует соответствие требованиям реального времени.

## Технические детали

### Использование Scaled Dot Product Attention (Задача 237, пункт А.2)

В методе `forward` модели LiT явно используется `torch.nn.functional.scaled_dot_product_attention`:

```python
# Шаг 5: Transformer Encoder с явным использованием SDPA
features = x
for layer in self.transformer.layers:
    # Self-attention через SDPA (Flash-ready в PyTorch 2.0+)
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        features, features, features,
        attn_mask=None,
        dropout_p=layer.self_attn.dropout if self.training else 0.0,
        is_causal=False
    )
    # ... residual connections и layer norm
```

Это гарантирует:
- Использование наиболее эффективного кернела на поддерживаемом железе
- Автоматическая активация Flash Attention в PyTorch 2.0+
- Явное соответствие требованиям задачи

### Реализация GQA (Grouped Query Attention)

Параметр `use_gqa` в конструкторе LiTModel включает Grouped Query Attention:

```python
if self.use_gqa:
    # GQA: уменьшаем количество Key/Value голов для эффективности
    self.num_kv_groups = max(1, nhead // 4)
    self.kv_projection = nn.Linear(d_model, d_model)
```

GQA эффективен для длинных последовательностей, но для фиксированных окон LOB стандартный MHA часто показывает лучшую точность.

### Пространство поиска Optuna

Согласно плану задачи 237 (пункт Б.1), пространство поиска ограничено:
- `embed_dim`: [32, 64, 128, 256]
- `num_heads`: [2, 4, 8, 16]

Остальные параметры фиксированы:
- `num_layers`: 2
- `dropout`: 0.1
- `activation`: 'gelu_exact'

## Интеграция с моделью

После оптимизации используйте найденные параметры при создании модели:

```python
from python_lab.src.lit_model import LiTModel
import json

# Загрузка оптимальной конфигурации
with open('bots/BTCUSDT/model/best_mha_config.json', 'r') as f:
    config = json.load(f)

# Создание модели с оптимальными параметрами
model = LiTModel(
    seq_len=100,
    in_channels=6,
    embed_dim=config['embed_dim'],
    num_heads=config['num_heads'],
    num_layers=2,  # Фиксировано в плане
    dropout=0.1,   # Фиксировано в плане
    activation='gelu_exact',
    multi_task=True
)
```

### Использование GQA

Для экспериментов с Grouped Query Attention:

```python
model = LiTModel(
    seq_len=100,
    in_channels=6,
    embed_dim=config['embed_dim'],
    num_heads=config['num_heads'],
    num_layers=2,
    dropout=0.1,
    activation='gelu_exact',
    multi_task=True,
    use_gqa=True  # Включить GQA
)
```

## Рекомендации

1. **Начните с малого**: 10-20 trials для быстрой оценки
2. **Увеличьте epochs**: для финальной оптимизации используйте 10+ epochs
3. **Настройте lambda**: в зависимости от приоритета (точность vs скорость)
4. **Проверьте Парето-фронт**: выберите точку на фронте в зависимости от требований
5. **Валидация**: протестируйте выбранную конфигурацию на отдельном test set

## Связанные задачи

- **Задача 026**: LOB Patching (используется в модели)
- **Задача 151**: Knowledge Distillation (можно применить после оптимизации)
- **Задача 156**: Optuna Pruning (используется в скрипте)
- **Задача 160**: Multi-Horizon Prediction (совместимо с оптимизацией)

## Troubleshooting

### Ошибка "embed_dim must be divisible by num_heads"

Optuna автоматически отсекает такие комбинации через `TrialPruned`.

### Все trials pruned из-за latency

Увеличьте `--max_latency` или уменьшите `--lambda_latency`.

### Низкий MCC

Увеличьте `--epochs` для лучшего обучения каждого trial.

### ONNX export failed

Проверьте совместимость операций модели с ONNX opset 17.

## Зависимости

```bash
pip install torch>=2.0 optuna onnxruntime numpy scikit-learn matplotlib
```

Опционально для ускорения обучения:
```bash
pip install flash-attn
```
