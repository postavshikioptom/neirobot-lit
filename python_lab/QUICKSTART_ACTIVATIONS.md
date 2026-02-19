# Быстрый старт: Функции активации

## Обучение модели с разными активациями

### 1. GELU (exact) - Рекомендуется
```bash
python -m python_lab.scripts.train \
  --symbol BTCUSDT \
  --activation gelu_exact \
  --epochs 100 \
  --batch_size 128
```

### 2. SiLU (Swish) - Для экспериментов
```bash
python -m python_lab.scripts.train \
  --symbol BTCUSDT \
  --activation silu \
  --epochs 100 \
  --batch_size 128
```

### 3. GELU (tanh) - Для ускорения
```bash
python -m python_lab.scripts.train \
  --symbol BTCUSDT \
  --activation gelu_tanh \
  --epochs 100 \
  --batch_size 128
```

### 4. ReLU - Базовая линия
```bash
python -m python_lab.scripts.train \
  --symbol BTCUSDT \
  --activation relu \
  --epochs 100 \
  --batch_size 128
```

## Автоматический подбор лучшей активации

Optuna автоматически перебирает все активации и выбирает лучшую:

```bash
python -m python_lab.scripts.tune \
  --symbol BTCUSDT \
  --trials 50
```

После завершения вы увидите:
```
Best MCC: 0.4523
Best params: {
  "activation": "silu",
  "d_model": 64,
  "nhead": 4,
  ...
}
```

## Тестирование паритета

Проверьте, что все активации корректно работают:

```bash
python python_lab/scripts/test_activation_parity.py
```

Ожидаемый результат:
```
ALL TESTS PASSED ✓
```

## Экспорт модели

После обучения экспортируйте модель в ONNX:

```bash
python -m python_lab.scripts.export_onnx \
  --input bots/BTCUSDT/models/checkpoints/best.ckpt \
  --output bots/BTCUSDT/models/lit.onnx
```

Метаданные будут содержать информацию об активации:
```json
{
  "activation": "gelu_exact",
  ...
}
```

## Сравнение результатов

Обучите модель с разными активациями и сравните MCC:

```bash
# GELU exact
python -m python_lab.scripts.train --symbol BTCUSDT --activation gelu_exact
# Смотрите val_mcc в логах TensorBoard

# SiLU
python -m python_lab.scripts.train --symbol BTCUSDT --activation silu
# Смотрите val_mcc в логах TensorBoard

# Сравните результаты в TensorBoard
tensorboard --logdir tb_logs
```

## Что выбрать?

- **Не знаете что выбрать?** → Используйте `gelu_exact`
- **Хотите лучший результат?** → Запустите Optuna
- **Нужна скорость?** → Попробуйте `gelu_tanh`
- **Хотите экспериментировать?** → Попробуйте `silu`
