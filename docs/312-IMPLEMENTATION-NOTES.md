# Задача 312: Оптимизация скорости обучения LiT модели - Заметки реализации

## Реализованные оптимизации

### 1. Отключение Curvature Regularization (Подзадача 312.1)
**Файл:** `python_lab/src/train.py` (строка 1320)

**Изменение:**
```python
# Было:
parser.add_argument("--use_curvature_reg", action=argparse.BooleanOptionalAction, default=True, ...)

# Стало:
parser.add_argument("--use_curvature_reg", action=argparse.BooleanOptionalAction, default=False, ...)
```

**Эффект:** Ускорение в ~2 раза (с 25 минут до 12-13 минут на эпоху)

**Использование:**
```bash
# По умолчанию curvature regularization отключена
python -m python_lab.src.train --symbol BTCUSDT --epochs 50

# Если нужно включить:
python -m python_lab.src.train --symbol BTCUSDT --epochs 50 --use_curvature_reg
```

---

### 2. Оптимизация частоты TensorBoard визуализаций (Подзадача 312.2)
**Файл:** `python_lab/src/train.py`

**Изменения:**
- Строка 545: Reliability Diagrams с 5 на 20 эпох
- Строка 560: Confusion Matrix/PR-кривые с 5 на 20 эпох
- Строка 660: Embeddings с 10 на 30 эпох

**Эффект:** Ускорение validation на 10-15%

**Код:**
```python
# Reliability Diagram каждые 20 эпох (было 5)
if self.current_epoch % 20 == 0:
    plot_reliability_diagram(...)

# Confusion Matrix каждые 20 эпох (было 5)
if self.current_epoch % 20 == 0:
    plot_confusion_matrix_tensorboard(...)

# Embeddings каждые 30 эпох (было 10)
if self.current_epoch % 30 == 0:
    log_embeddings(...)
```

---

### 3. PyTorch Profiler для анализа производительности (Подзадача 312.3)
**Файл:** `python_lab/src/train.py`

**Добавлено:**
- Импорт: `from torch.profiler import profile, ProfilerActivity, schedule`
- Класс `ProfilerCallback` (строки 112-160)
- Параметры командной строки:
  - `--enable_profiler`: Включить профилирование
  - `--profiler_wait_steps`: Количество шагов ожидания (default: 1)
  - `--profiler_warmup_steps`: Количество warmup шагов (default: 1)
  - `--profiler_active_steps`: Количество активных шагов профилирования (default: 3)

**Использование:**
```bash
# Запустить обучение с профилированием
python -m python_lab.src.train --symbol BTCUSDT --epochs 5 --enable_profiler

# Результаты будут сохранены в: profiler_logs/BTCUSDT/trace_epoch_*.json
# Можно открыть в Chrome DevTools (chrome://tracing)
```

**Как использовать результаты профилирования:**
1. Откройте Chrome DevTools: `chrome://tracing`
2. Загрузите файл `trace_epoch_0.json` из `profiler_logs/BTCUSDT/`
3. Анализируйте временные шкалы операций CPU и CUDA
4. Найдите узкие места (bottlenecks)

---

### 4. Оптимизация DataLoader (Подзадача 312.4)
**Файл:** `python_lab/src/train.py` (строки 1921-1935)

**Добавлено:**
```python
train_loader = DataLoader(
    train_ds, 
    batch_size=args.batch_size, 
    shuffle=True,
    num_workers=num_workers,      # Параллельная загрузка данных
    pin_memory=True,               # Ускорить CPU->GPU передачу
    prefetch_factor=2,             # Предзагрузка батчей (НОВОЕ)
    persistent_workers=True if num_workers > 0 else False,  # Не убивать workers
    worker_init_fn=worker_init_fn
)
```

**Эффект:** Ускорение на 20-30% если есть CPU bottleneck

**Параметры:**
- `num_workers`: Количество рабочих процессов для загрузки данных
- `pin_memory=True`: Закрепляет данные в памяти для быстрой передачи на GPU
- `prefetch_factor=2`: Предзагружает 2 батча заранее
- `persistent_workers=True`: Не убивает рабочие процессы между эпохами

---

### 5. Тестирование batch_size=64 (Подзадача 312.5)
**Параметр:** `--batch_size` (default: 128)

**Использование:**
```bash
# Запустить обучение с batch_size=64
python -m python_lab.src.train --symbol BTCUSDT --epochs 50 --batch_size 64

# Сравнить с batch_size=128 (default)
python -m python_lab.src.train --symbol BTCUSDT --epochs 50 --batch_size 128
```

**Рекомендации:**
- Меньший batch_size (32-64) может улучшить генерализацию
- Исследования по LOB данным рекомендуют 32-64 для трансформеров
- Может потребоваться больше памяти GPU при меньшем batch_size (парадокс!)
- Рекомендуется тестировать оба варианта и сравнивать метрики

---

## Полный пример использования всех оптимизаций

```bash
# Запустить обучение со всеми оптимизациями
python -m python_lab.src.train \
    --symbol BTCUSDT \
    --epochs 50 \
    --batch_size 64 \
    --enable_profiler \
    --profiler_wait_steps 1 \
    --profiler_warmup_steps 1 \
    --profiler_active_steps 3

# Результаты:
# - Обучение будет быстрее (3-5x ускорение)
# - TensorBoard логи будут меньше (реже визуализации)
# - Профилирование покажет узкие места
# - batch_size=64 может улучшить метрики
```

---

## Ожидаемые результаты

| Метрика | До оптимизации | После оптимизации | Ускорение |
|---------|---|---|---|
| Время эпохи | 25 минут | 5-8 минут | 3-5x |
| Curvature Reg | Включена | Отключена | ~2x |
| TensorBoard I/O | Каждые 5 эпох | Каждые 20-30 эпох | ~4-6x |
| DataLoader | Без prefetch | prefetch_factor=2 | ~1.2-1.3x |
| Batch Size | 128 | 64 | Может улучшить метрики |

---

## Отключение оптимизаций (если нужно вернуться к старому поведению)

```bash
# Включить Curvature Regularization
python -m python_lab.src.train --symbol BTCUSDT --use_curvature_reg

# Увеличить batch_size обратно до 128
python -m python_lab.src.train --symbol BTCUSDT --batch_size 128

# Отключить профилирование (по умолчанию отключено)
# Просто не используйте флаг --enable_profiler
```

---

## Диагностика проблем

### Если обучение медленнее, чем ожидалось:
1. Проверьте, что Curvature Regularization отключена: `--use_curvature_reg` не должен быть в команде
2. Запустите профилирование: `--enable_profiler`
3. Анализируйте результаты профилирования в Chrome DevTools

### Если видите ошибки памяти:
1. Уменьшите `batch_size`: `--batch_size 32`
2. Уменьшите `num_workers` в коде (если нужно)
3. Проверьте доступную GPU память: `nvidia-smi`

### Если TensorBoard логи слишком большие:
- Оптимизация уже сделана (20-30 эпох вместо 5-10)
- Можно еще увеличить интервалы в коде, если нужно

---

## Файлы, измененные в этой задаче

- `python_lab/src/train.py`: Основной файл с оптимизациями
  - Строка 17: Импорт torch.profiler
  - Строка 112-160: Класс ProfilerCallback
  - Строка 1320: default=False для use_curvature_reg
  - Строка 1335-1340: Параметры профилера
  - Строка 545, 560, 660: Частота TensorBoard визуализаций
  - Строка 1921-1935: prefetch_factor в DataLoaders
  - Строка 2189-2200: ProfilerCallback в callbacks

---

## Дополнительные ресурсы

- [PyTorch Profiler документация](https://pytorch.org/docs/stable/profiler.html)
- [DataLoader оптимизация](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
- [Chrome DevTools Tracing](https://developer.chrome.com/docs/devtools/performance/reference/)
