# Model Pruning Guide (Задача 159)

Руководство по использованию magnitude-based pruning для уменьшения размера модели LiT.

## Обзор

Model pruning (прунинг модели) - это техника оптимизации, которая удаляет наименее важные веса из нейронной сети, создавая разреженную (sparse) модель. Это позволяет:

- ✅ Уменьшить размер модели на диске и в памяти
- ✅ Подготовить модель к квантованию (задача 157)
- ✅ Упростить калибровку для INT8 конверсии
- ⚠️ **НЕ дает автоматического ускорения** в стандартном ONNX Runtime

## Режимы прунинга

### 1. Unstructured Pruning (по умолчанию)

Удаляет отдельные веса с наименьшей L1-нормой.

**Преимущества:**
- Лучшая точность модели
- Гибкость в выборе уровня разреженности
- Простота реализации

**Недостатки:**
- Не дает ускорения в ONNX Runtime
- Только уменьшение размера файла

**Использование:**
```bash
python -m python_lab.src.train \
    --symbol BTCUSDT \
    --prune_mode unstructured \
    --prune_amount 0.5 \
    --prune_iterations 3 \
    --prune_finetune_epochs 2
```

### 2. Structured 2:4 Pruning

Создает паттерн, где из каждых 4 подряд идущих весов 2 являются нулями.

**Преимущества:**
- Потенциальное ускорение до 2x на NVIDIA GPU (Ampere+)
- Аппаратная поддержка через Sparse Tensor Cores
- Автоматическая конвертация в SparseSemiStructuredTensor
- Фиксированная разреженность 50%

**Недостатки:**
- Требует NVIDIA GPU с Compute Capability 8.0+
- Менее гибкий (фиксированная разреженность)
- Может дать большее падение точности

**Использование:**
```bash
python -m python_lab.src.train \
    --symbol BTCUSDT \
    --prune_mode structured_2_4 \
    --prune_iterations 3 \
    --prune_finetune_epochs 2
```

**Примечание:** После прунинга модель автоматически конвертируется в `SparseSemiStructuredTensor` для максимального ускорения на NVIDIA GPU.

## Параметры

### `--prune_mode`
- **Значения:** `none`, `unstructured`, `structured_2_4`
- **По умолчанию:** `none`
- **Описание:** Режим прунинга

### `--prune_amount`
- **Диапазон:** 0.0 - 0.6
- **По умолчанию:** 0.5
- **Описание:** Целевая доля нулевых весов
- **Рекомендация:** 0.3-0.6 для LOB данных

⚠️ **ВАЖНО:** Значения > 0.6 могут необратимо повредить способность модели различать микро-тренды в LOB данных.

### `--prune_iterations`
- **Диапазон:** >= 1
- **По умолчанию:** 3
- **Описание:** Количество циклов "прунинг + fine-tuning"
- **Рекомендация:** 3-5 итераций для постепенного удаления весов

### `--prune_finetune_epochs`
- **Диапазон:** >= 1
- **По умолчанию:** 2
- **Описание:** Эпохи дообучения после каждой итерации прунинга
- **Рекомендация:** 1-2 эпохи для восстановления MCC

## Workflow

### Шаг 1: Обучение базовой модели

Сначала обучите модель без прунинга:

```bash
python -m python_lab.src.train \
    --symbol BTCUSDT \
    --epochs 100 \
    --batch_size 128
```

### Шаг 2: Прунинг модели

Примените прунинг к обученной модели:

```bash
python -m python_lab.src.train \
    --symbol BTCUSDT \
    --epochs 100 \
    --batch_size 128 \
    --prune_mode unstructured \
    --prune_amount 0.5 \
    --prune_iterations 3 \
    --prune_finetune_epochs 2
```

**Что происходит:**
1. Модель обучается обычным образом (100 эпох)
2. Загружается лучший checkpoint
3. Выполняется 3 итерации прунинга:
   - Итерация 1: удаляется 16.7% весов (0.5 / 3)
   - Итерация 2: удаляется 33.3% весов
   - Итерация 3: удаляется 50% весов
4. После каждой итерации: 2 эпохи fine-tuning
5. Сохраняется разреженная модель

### Шаг 3: Экспорт в ONNX

Экспортируйте разреженную модель:

```bash
python -m python_lab.src.export_onnx \
    --input bots/BTCUSDT/models/pruned_unstructured.pt \
    --output bots/BTCUSDT/models/lit_pruned.onnx \
    --fp16
```

### Шаг 4: Квантование (опционально)

После прунинга модель готова к квантованию (задача 157):

```bash
python -m python_lab.scripts.quantize_onnx \
    --input bots/BTCUSDT/models/lit_pruned.onnx \
    --output bots/BTCUSDT/models/lit_pruned_int8.onnx
```

## Мониторинг прогресса

Во время прунинга выводится детальная статистика:

```
============================================================
PRUNING ITERATION 1/3
============================================================
Current Prune Amount: 16.67%
Target Prune Amount: 50.00%
Actual Sparsity: 17.23%
Current MCC: 0.7234
MCC Drop: 0.0123 (1.67%)
============================================================
```

## Результаты

После завершения прунинга:

```
============================================================
PRUNING COMPLETED
============================================================
Final Sparsity: 50.12%
Final MCC: 0.7156
MCC Drop: 0.0201 (2.73%)
============================================================

PRUNED MODEL SAVED
============================================================
Path: bots/BTCUSDT/models/pruned_unstructured.pt
Global Sparsity: 50.12%
Total Params: 1,234,567
Zero Params: 618,901
Baseline MCC: 0.7357
Ready for Quantization: Yes
============================================================
```

## Рекомендации

### Для максимальной точности
```bash
--prune_mode unstructured \
--prune_amount 0.3 \
--prune_iterations 5 \
--prune_finetune_epochs 2
```

### Для максимального сжатия
```bash
--prune_mode unstructured \
--prune_amount 0.6 \
--prune_iterations 4 \
--prune_finetune_epochs 3
```

### Для ускорения на NVIDIA GPU
```bash
--prune_mode structured_2_4 \
--prune_iterations 3 \
--prune_finetune_epochs 2
```

## Важные замечания

### ⚠️ Производительность

**Unstructured pruning НЕ дает ускорения в ONNX Runtime!**

Причина: ONNX Runtime выполняет разреженные операции как плотные (dense), поэтому нулевые веса все равно обрабатываются.

**Преимущества:**
- Уменьшение размера файла модели
- Лучшая синергия с квантованием
- Упрощенная калибровка для INT8

**Для реального ускорения:**
- Используйте `--prune_mode structured_2_4` с NVIDIA GPU (Ampere+)
- Или специализированные sparse inference engines
- Или комбинируйте с квантованием

### 🎯 Целевые метрики

Для LOB данных рекомендуется:
- **Sparsity:** 30-60%
- **MCC Drop:** < 5%
- **Iterations:** 3-5

### 🔄 Итеративность обязательна

Одномоментное удаление 50% весов на зашумленных LOB данных необратимо портит MCC. Используйте итеративный подход с постепенным увеличением разреженности.

### 🚫 Исключения из прунинга

Следующие слои НЕ подвергаются прунингу:
- LayerNorm (разрушает калибровку активаций)
- Bias параметры
- Embeddings (cls_token, pos_emb)
- Task-specific головы (для structured_2_4)

## Troubleshooting

### Проблема: MCC падает слишком сильно

**Решение:**
- Уменьшите `--prune_amount` (попробуйте 0.3-0.4)
- Увеличьте `--prune_iterations` (5-7)
- Увеличьте `--prune_finetune_epochs` (3-4)

### Проблема: Structured 2:4 не применяется

**Ошибка:** `No layers found for 2:4 structured pruning`

**Причина:** Shape constraints не выполнены (weight.shape[-1] % 4 != 0)

**Решение:** Используйте `--prune_mode unstructured`

### Проблема: Прунинг не поддерживается в CV режиме

**Ошибка:** `Pruning не поддерживается в режиме cross-validation`

**Решение:** Используйте режим `train` или `distill`:
```bash
--mode train --prune_mode unstructured
```

## Связь с другими задачами

- **Задача 157 (Квантование):** Прунинг выполняется ДО квантования для лучшей калибровки
- **Задача 151 (Distillation):** Можно применять прунинг к student модели
- **Задача 036 (ONNX Runtime):** Разреженные модели работают в существующем инференсе

## Примеры команд

### Базовый прунинг
```bash
python -m python_lab.src.train \
    --symbol BTCUSDT \
    --prune_mode unstructured \
    --prune_amount 0.5
```

### Агрессивный прунинг
```bash
python -m python_lab.src.train \
    --symbol FARTCOINUSDT \
    --prune_mode unstructured \
    --prune_amount 0.6 \
    --prune_iterations 5 \
    --prune_finetune_epochs 3
```

### Прунинг student модели
```bash
python -m python_lab.src.train \
    --symbol BTCUSDT \
    --mode distill \
    --teacher_path bots/BTCUSDT/models/teacher.ckpt \
    --prune_mode unstructured \
    --prune_amount 0.5
```

### Structured 2:4 для NVIDIA
```bash
python -m python_lab.src.train \
    --symbol BTCUSDT \
    --prune_mode structured_2_4 \
    --prune_iterations 3
```

## Ссылки

- [PyTorch Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
- [Semi-Structured Sparsity (2:4)](https://pytorch.org/tutorials/advanced/semi_structured_sparse.html)
- [NVIDIA Sparse Tensor Cores](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/)
- Задача 157: Квантование модели
- Задача 036: ONNX Runtime инференс
