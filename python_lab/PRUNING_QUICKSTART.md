# Model Pruning - Quick Start

Быстрое руководство по прунингу модели LiT (Задача 159).

## Что такое прунинг?

Прунинг удаляет наименее важные веса из модели, создавая разреженную структуру:
- ✅ Уменьшает размер модели на 50%
- ✅ Подготавливает к квантованию
- ⚠️ Не дает ускорения в ONNX Runtime (только размер)

## Быстрый старт

### 1. Базовый прунинг (50% разреженность)

```bash
python -m python_lab.src.train \
    --symbol BTCUSDT \
    --epochs 100 \
    --prune_mode unstructured \
    --prune_amount 0.5 \
    --prune_iterations 3
```

### 2. Консервативный прунинг (30% разреженность)

Для минимального падения точности:

```bash
python -m python_lab.src.train \
    --symbol BTCUSDT \
    --epochs 100 \
    --prune_mode unstructured \
    --prune_amount 0.3 \
    --prune_iterations 5 \
    --prune_finetune_epochs 2
```

### 3. Агрессивный прунинг (60% разреженность)

Для максимального сжатия:

```bash
python -m python_lab.src.train \
    --symbol BTCUSDT \
    --epochs 100 \
    --prune_mode unstructured \
    --prune_amount 0.6 \
    --prune_iterations 4 \
    --prune_finetune_epochs 3
```

### 4. Structured 2:4 (для NVIDIA GPU)

Для потенциального ускорения на Ampere+ GPU:

```bash
python -m python_lab.src.train \
    --symbol BTCUSDT \
    --epochs 100 \
    --prune_mode structured_2_4 \
    --prune_iterations 3
```

## Параметры

| Параметр | Значение | Описание |
|----------|----------|----------|
| `--prune_mode` | `unstructured` / `structured_2_4` | Режим прунинга |
| `--prune_amount` | 0.3 - 0.6 | Целевая разреженность |
| `--prune_iterations` | 3 - 5 | Количество итераций |
| `--prune_finetune_epochs` | 1 - 3 | Эпохи дообучения |

## Workflow

```
1. Обучение → 2. Прунинг → 3. Экспорт → 4. Квантование
   (100 эпох)   (3 итерации)  (ONNX)     (INT8)
```

## Результаты

После прунинга вы получите:

```
✓ Разреженная модель: bots/SYMBOL/models/pruned_unstructured.pt
✓ Размер уменьшен на ~50%
✓ MCC падение < 5%
✓ Готова к квантованию
```

## Важно

⚠️ **Unstructured pruning не дает ускорения в ONNX Runtime!**

Используется для:
- Уменьшения размера файла
- Подготовки к квантованию
- Упрощения калибровки INT8

Для ускорения:
- Используйте `structured_2_4` с NVIDIA GPU (Ampere+)
- Или комбинируйте с квантованием

## Следующие шаги

1. **Экспорт в ONNX:**
```bash
python -m python_lab.src.export_onnx \
    --input bots/BTCUSDT/models/pruned_unstructured.pt \
    --output bots/BTCUSDT/models/lit_pruned.onnx
```

2. **Квантование (задача 157):**
```bash
python -m python_lab.scripts.quantize_onnx \
    --input bots/BTCUSDT/models/lit_pruned.onnx \
    --output bots/BTCUSDT/models/lit_pruned_int8.onnx
```

## Подробная документация

См. [PRUNING_GUIDE.md](./PRUNING_GUIDE.md) для детального руководства.
