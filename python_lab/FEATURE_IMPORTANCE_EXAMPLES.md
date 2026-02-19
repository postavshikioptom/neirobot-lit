# Примеры использования Feature Importance Analysis

## Сценарий 1: Полный анализ (индивидуальные признаки + группы)

```bash
python python_lab/scripts/feature_importance.py \
  --symbol CAKEUSDT \
  --model_path bots/CAKEUSDT/model/best_model.ckpt \
  --n_repeats 5 \
  --batch_size 256 \
  --device cuda
```

**Время выполнения**: ~1-2 часа на GPU (200 признаков × 5 повторений)

**Результат**:
- `feature_importance.json` с полными результатами
- `feature_importance_bar.png` с топ-20 признаками
- `lob_importance_heatmap.png` с матрицей важности

---

## Сценарий 2: Только групповой анализ (быстрый)

```bash
python python_lab/scripts/feature_importance.py \
  --symbol FARTCOINUSDT \
  --model_path bots/FARTCOINUSDT/model/best_model.ckpt \
  --skip_individual \
  --n_repeats 10 \
  --device cuda
```

**Время выполнения**: ~5-10 минут на GPU (4 группы × 10 повторений)

**Результат**:
- `feature_importance.json` только с групповыми результатами
- Быстрая оценка важности Price vs Volume vs Spread

---

## Сценарий 3: Только индивидуальные признаки (без групп)

```bash
python python_lab/scripts/feature_importance.py \
  --symbol CAKEUSDT \
  --model_path bots/CAKEUSDT/model/best_model.ckpt \
  --skip_groups \
  --n_repeats 3 \
  --device cuda
```

**Время выполнения**: ~30-60 минут на GPU (200 признаков × 3 повторения)

**Результат**:
- Детальная карта важности каждого уровня стакана
- Heatmap для визуализации "горизонта событий"

---

## Сценарий 4: Быстрая оценка (малое количество повторений)

```bash
python python_lab/scripts/feature_importance.py \
  --symbol CAKEUSDT \
  --model_path bots/CAKEUSDT/model/best_model.ckpt \
  --n_repeats 2 \
  --batch_size 512 \
  --device cuda
```

**Время выполнения**: ~20-30 минут на GPU

**Результат**:
- Быстрая оценка важности (менее точная из-за малого n_repeats)
- Подходит для первичного анализа

---

## Сценарий 5: CPU режим (без GPU)

```bash
python python_lab/scripts/feature_importance.py \
  --symbol CAKEUSDT \
  --model_path bots/CAKEUSDT/model/best_model.ckpt \
  --skip_individual \
  --n_repeats 5 \
  --batch_size 64 \
  --device cpu
```

**Время выполнения**: ~30-60 минут на CPU (только группы)

**Результат**:
- Групповой анализ на CPU
- Уменьшенный batch_size для экономии памяти

---

## Сценарий 6: Воспроизводимые результаты

```bash
python python_lab/scripts/feature_importance.py \
  --symbol CAKEUSDT \
  --model_path bots/CAKEUSDT/model/best_model.ckpt \
  --n_repeats 5 \
  --seed 12345 \
  --device cuda
```

**Результат**:
- Фиксированный seed для воспроизводимости
- Одинаковые результаты при повторных запусках

---

## Сценарий 7: Кастомные пути

```bash
python python_lab/scripts/feature_importance.py \
  --symbol BTCUSDT \
  --model_path /path/to/custom/model.ckpt \
  --data_path /path/to/custom/data \
  --output_dir /path/to/custom/output \
  --n_repeats 5 \
  --device cuda
```

**Результат**:
- Использование кастомных путей для данных и вывода
- Полезно для экспериментов с разными моделями

---

## Интерпретация результатов

### Пример 1: Оптимизация количества уровней

Если heatmap показывает:
- Уровни 0-15: высокая важность (яркие цвета)
- Уровни 16-49: низкая важность (темные цвета)

**Действие**: Уменьшить n_levels с 50 до 15 в конфигурации

**Результат**: 
- Уменьшение размера тензора в 3.3 раза
- Ускорение инференса в ~2-3 раза
- Сохранение качества модели

### Пример 2: Удаление неинформативных каналов

Если групповой анализ показывает:
- Price Levels: mean_importance = 0.15
- Volume Levels: mean_importance = 0.02
- Spread/Imbalance: mean_importance = 0.08

**Действие**: Удалить Volume Levels (ask_v, bid_v)

**Результат**:
- Уменьшение количества каналов с 4 до 2
- Уменьшение размера тензора в 2 раза
- Минимальная потеря качества (Volume неинформативен)

### Пример 3: Фокус на важных признаках

Если топ-20 признаков составляют 80% важности:

**Действие**: Использовать только топ-20 признаков

**Результат**:
- Уменьшение размера тензора в 10 раз
- Значительное ускорение инференса
- Возможная потеря качества (требует тестирования)

---

## Troubleshooting

### Ошибка: CUDA out of memory

```bash
# Решение 1: Уменьшить batch_size
python python_lab/scripts/feature_importance.py \
  --symbol CAKEUSDT \
  --model_path bots/CAKEUSDT/model/best_model.ckpt \
  --batch_size 128 \
  --device cuda

# Решение 2: Использовать только групповой анализ
python python_lab/scripts/feature_importance.py \
  --symbol CAKEUSDT \
  --model_path bots/CAKEUSDT/model/best_model.ckpt \
  --skip_individual \
  --device cuda

# Решение 3: Переключиться на CPU
python python_lab/scripts/feature_importance.py \
  --symbol CAKEUSDT \
  --model_path bots/CAKEUSDT/model/best_model.ckpt \
  --batch_size 64 \
  --device cpu
```

### Ошибка: No parquet files found

```bash
# Проверьте путь к данным
ls bots/CAKEUSDT/data/raw/*.parquet

# Укажите правильный путь
python python_lab/scripts/feature_importance.py \
  --symbol CAKEUSDT \
  --model_path bots/CAKEUSDT/model/best_model.ckpt \
  --data_path /correct/path/to/data
```

### Ошибка: Model checkpoint not found

```bash
# Проверьте путь к модели
ls bots/CAKEUSDT/model/*.ckpt

# Укажите правильный путь
python python_lab/scripts/feature_importance.py \
  --symbol CAKEUSDT \
  --model_path /correct/path/to/model.ckpt
```

---

## Рекомендации

1. **Первый запуск**: Используйте Сценарий 2 (только группы) для быстрой оценки
2. **Детальный анализ**: Используйте Сценарий 1 (полный анализ) для финальной оптимизации
3. **Эксперименты**: Используйте Сценарий 4 (малое n_repeats) для быстрых итераций
4. **Воспроизводимость**: Всегда используйте фиксированный seed для научных экспериментов
5. **Валидация**: После оптимизации переобучите модель и проверьте качество на тестовой выборке
