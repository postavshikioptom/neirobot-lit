# Руководство по Optuna Pruning (Задача 156)

## Обзор

Автоматическое отсечение (pruning) в Optuna позволяет останавливать бесперспективные trials на ранних этапах обучения, экономя время и ресурсы при подборе гиперпараметров.

Реализация включает:
- **CLI аргументы в train.py** для настройки pruning параметров
- **Функцию print_pruning_stats в utils.py** для анализа статистики
- **Интеграцию в tune.py** для автоматического подбора гиперпараметров

## Типы Pruner'ов

### 1. MedianPruner (базовый)
Отсекает trial, если его промежуточный результат хуже медианы результатов предыдущих trials на том же шаге.

**Когда использовать:** Для стабильных метрик с низким уровнем шума.

```bash
python -m python_lab.src.tune --symbol BTCUSDT --trials 50 \
    --pruner_type median \
    --n_startup_trials 20 \
    --n_warmup_steps 25
```

### 2. HyperbandPruner (бюджетирование ресурсов)
Использует алгоритм Hyperband для эффективного распределения ресурсов между trials.

**Когда использовать:** Когда нужно быстро протестировать много конфигураций с разным количеством эпох.

```bash
python -m python_lab.src.tune --symbol BTCUSDT --trials 50 \
    --pruner_type hyperband \
    --min_resource 1 \
    --max_resource 20
```

### 3. PatientPruner (устойчивость к шуму) ⭐ РЕКОМЕНДУЕТСЯ ДЛЯ LOB
Оборачивает другой pruner и добавляет "терпение" - не отсекает trial при случайных просадках метрики.

**Когда использовать:** Для шумных метрик (MCC на LOB данных), когда метрика может временно ухудшаться.

```bash
python -m python_lab.src.tune --symbol BTCUSDT --trials 50 \
    --pruner_type patience \
    --n_startup_trials 20 \
    --n_warmup_steps 25 \
    --patience 3
```

## Параметры Pruning

### Обязательные параметры

- `--pruner_type`: Тип pruner (median, hyperband, patience)

### Параметры для MedianPruner и PatientPruner

- `--n_startup_trials`: Количество полных trials перед началом pruning
  - **Минимум для LOB данных: 20**
  - По умолчанию: 20
  
- `--n_warmup_steps`: Количество эпох до первой проверки на pruning
  - **Минимум для трансформеров: 25**
  - По умолчанию: 25

### Параметры для HyperbandPruner

- `--min_resource`: Минимальное количество эпох (по умолчанию: 1)
- `--max_resource`: Максимальное количество эпох (по умолчанию: 20)

### Параметры для PatientPruner

- `--patience`: Количество шагов без улучшения перед отсечением (по умолчанию: 3)

## Примеры использования

### Базовый tuning с pruning
```bash
python -m python_lab.src.tune --symbol BTCUSDT --trials 50 \
    --pruner_type patience \
    --n_startup_trials 20 \
    --n_warmup_steps 25 \
    --patience 3
```

### Distillation режим с pruning
```bash
python -m python_lab.src.tune --symbol BTCUSDT --mode distill \
    --teacher_path bots/BTCUSDT/models/teacher_lit.pt \
    --pruner_type patience \
    --trials 30 \
    --n_startup_trials 15 \
    --n_warmup_steps 20
```

### Cross-validation режим с pruning
```bash
python -m python_lab.src.tune --symbol BTCUSDT --mode cv \
    --n_splits 3 \
    --pruner_type median \
    --trials 30 \
    --n_startup_trials 10 \
    --n_warmup_steps 1
```

### Быстрый поиск с HyperbandPruner
```bash
python -m python_lab.src.tune --symbol BTCUSDT --trials 100 \
    --pruner_type hyperband \
    --min_resource 1 \
    --max_resource 15
```

## Рекомендации для LOB данных

### Почему PatientPruner критичен для LOB?

MCC (Matthews Correlation Coefficient) на данных стакана - это **шумная метрика**:
- Высокая волатильность между эпохами
- Случайные просадки не означают плохую конфигурацию
- Трансформеры начинают сходиться только после 10-20 эпох

### Оптимальные настройки для LOB

```bash
python -m python_lab.src.tune --symbol BTCUSDT --trials 50 \
    --pruner_type patience \
    --n_startup_trials 20 \
    --n_warmup_steps 25 \
    --patience 3
```

**Объяснение:**
- `n_startup_trials=20`: Даем 20 trials завершиться полностью для стабилизации статистики
- `n_warmup_steps=25`: Даем трансформеру "прогреться" 25 эпох перед первой проверкой
- `patience=3`: Не отсекаем при 3 последовательных шагах без улучшения (терпимость к шуму)

## Интерпретация статистики pruning

После завершения tuning выводится статистика:

```
PRUNING STATISTICS
============================================================
Total trials: 50
Completed trials: 35
Pruned trials: 15 (30.0%)

Pruning timing:
  - Average pruning step: 12.3
  - Min pruning step: 5
  - Max pruning step: 18

✓ Pruning is working effectively (avg step > warmup)
============================================================
```

### Что означают метрики?

- **Pruned trials %**: Процент отсеченных trials
  - 20-40% - оптимально (экономия времени без потери качества)
  - <10% - pruner слишком консервативный
  - >60% - pruner слишком агрессивный

- **Average pruning step**: Средняя эпоха отсечения
  - Должна быть > n_warmup_steps
  - Если меньше - увеличьте n_warmup_steps или используйте patience

### Предупреждения

```
⚠️  WARNING: Average pruning step (15.2) is below warmup steps (25)
  This suggests pruner may be too aggressive. Consider:
    - Increasing --n_warmup_steps
    - Using --pruner_type patience for noisy metrics
```

Если видите это предупреждение:
1. Увеличьте `--n_warmup_steps`
2. Переключитесь на `--pruner_type patience`
3. Увеличьте `--patience`

## Целевая метрика

Optuna оптимизирует **MCC (Matthews Correlation Coefficient)** как основную метрику качества:
- Direction: maximize
- Диапазон: [-1, 1]
- 1 = идеальная классификация
- 0 = случайная классификация
- -1 = полностью неправильная классификация

## Дополнительные метрики

В каждом trial логируются дополнительные метрики:
- `val_loss`: Validation loss
- `ece`: Expected Calibration Error
- `mce`: Maximum Calibration Error

Эти метрики доступны через `study.best_trial.user_attrs`.

## Troubleshooting

### Pruner отсекает все trials
**Проблема:** Почти все trials pruned (>80%)

**Решение:**
```bash
# Увеличьте n_startup_trials и n_warmup_steps
--n_startup_trials 30 --n_warmup_steps 35

# Или используйте patience
--pruner_type patience --patience 5
```

### Pruner не отсекает ничего
**Проблема:** Почти нет pruned trials (<5%)

**Решение:**
```bash
# Уменьшите n_warmup_steps
--n_warmup_steps 15

# Или используйте hyperband
--pruner_type hyperband --min_resource 1 --max_resource 15
```

### Tuning слишком медленный
**Проблема:** Каждый trial занимает много времени

**Решение:**
```bash
# Используйте hyperband для быстрого поиска
--pruner_type hyperband --min_resource 1 --max_resource 10

# Или уменьшите max_epochs в objective (требует изменения кода)
```

## Ссылки

- [Optuna Pruning Documentation](https://optuna.readthedocs.io/en/stable/tutorial/pruning.html)
- [MedianPruner API](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html)
- [HyperbandPruner API](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html)
- [PatientPruner API](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.PatientPruner.html)
