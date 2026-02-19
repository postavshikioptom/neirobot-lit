# Задача 237: Быстрый старт

## Установка зависимостей

```bash
pip install torch>=2.0 optuna onnxruntime numpy scikit-learn matplotlib
```

## Запуск оптимизации

### Быстрый тест (10 trials, ~10 минут)
```bash
python -m python_lab.scripts.tune_attention --symbol BTCUSDT --trials 10 --epochs 3
```

### Полная оптимизация (30 trials, ~1 час)
```bash
python -m python_lab.scripts.tune_attention --symbol BTCUSDT --trials 30 --epochs 5
```

## Результаты

После завершения проверьте:

1. **Конфигурация**: `bots/BTCUSDT/model/best_mha_config.json`
2. **График**: `reports/mha_pareto_front.png`

## Использование результатов

```python
from python_lab.src.lit_model import LiTModel
import json

# Загрузка оптимальной конфигурации
with open('bots/BTCUSDT/model/best_mha_config.json', 'r') as f:
    config = json.load(f)

# Создание модели
model = LiTModel(
    seq_len=100,
    in_channels=6,
    embed_dim=config['embed_dim'],
    num_heads=config['num_heads'],
    num_layers=config['num_layers'],
    dropout=config['dropout']
)
```

## Подробная документация

См. `python_lab/scripts/README_MHA_TUNING.md`
