# Руководство по режимам загрузки данных (Data Modes)

## Обзор

Python Lab поддерживает три режима загрузки данных для обучения моделей на датасетах различного размера:

1. **Memory** - Стандартная загрузка всех данных в RAM
2. **Streaming** - Потоковая загрузка через Polars Lazy API
3. **Memmap** - Memory-mapped binary файлы для быстрого случайного доступа

## Режимы загрузки

### 1. Memory Mode (по умолчанию)

**Когда использовать:**
- Датасет полностью помещается в RAM
- Нужна максимальная скорость обучения
- Доступно достаточно памяти (датасет < 70% от доступной RAM)

**Преимущества:**
- Самый быстрый доступ к данным
- Простая реализация
- Нет накладных расходов на I/O

**Недостатки:**
- Требует загрузки всего датасета в память
- Не подходит для датасетов > RAM

**Использование:**
```bash
python -m python_lab.scripts.train --symbol BTCUSDT --data_mode memory
```

### 2. Streaming Mode

**Когда использовать:**
- Датасет не помещается в RAM (100GB+)
- Нужен баланс между скоростью и использованием памяти
- Данные хранятся в Parquet формате

**Преимущества:**
- Работает с датасетами любого размера
- Использует Parquet native compression (Snappy/Zstd)
- Zero-copy доступ через PyArrow
- Не требует дополнительного места на диске

**Недостатки:**
- Медленнее на 20-30% по сравнению с memory mode
- Требует оптимизации размера батчей для кэширования

**Технические детали:**
- Использует `polars.scan_parquet()` с `low_memory=True`
- Данные читаются батчами через `collect(engine="streaming")`
- Внутренний кэш батчей для оптимизации последовательного доступа
- Thread-safe для multi-processing DataLoader (используйте `num_workers=2`)

**Использование:**
```bash
python -m python_lab.scripts.train --symbol BTCUSDT --data_mode streaming
```

### 3. Memmap Mode

**Когда использовать:**
- Нужен быстрый случайный доступ к большим датасетам
- Доступно достаточно места на диске (3-5x размер Parquet)
- Датасет используется многократно

**Преимущества:**
- Быстрый случайный доступ (почти как memory mode)
- Работает с датасетами > RAM
- Кэш переиспользуется между запусками

**Недостатки:**
- Требует 3-5x больше места на диске (uncompressed binary)
- Первый запуск медленный (создание кэша)
- Кэш нужно пересоздавать при изменении данных

**Технические детали:**
- Конвертирует данные в uncompressed numpy binary format
- Использует `np.memmap` для memory-mapped доступа
- Кэш хранится в `cache_dir` (по умолчанию: `bots/{symbol}/models/cache`)
- Метаданные сохраняются в JSON для валидации

**Использование:**
```bash
# Первый запуск (создание кэша)
python -m python_lab.scripts.train --symbol BTCUSDT --data_mode memmap --cache_dir ./cache

# Последующие запуски (использование кэша)
python -m python_lab.scripts.train --symbol BTCUSDT --data_mode memmap --cache_dir ./cache
```

## Сравнение производительности

| Режим      | Скорость доступа | Использование RAM | Использование диска | Подходит для |
|------------|------------------|-------------------|---------------------|--------------|
| Memory     | ⚡⚡⚡ Fastest    | 🔴 High           | ✅ None            | < 70% RAM    |
| Streaming  | ⚡⚡ Fast        | ✅ Low            | ✅ None            | > RAM        |
| Memmap     | ⚡⚡⚡ Fast      | ✅ Low            | 🔴 High (3-5x)     | > RAM, reuse |

## Автоматическая валидация ресурсов

При использовании `memory` mode скрипт автоматически проверяет доступную RAM:

```
⚠️  WARNING: Dataset size (~45.2 GB) may exceed available RAM (32.0 GB)
   Consider using --data_mode streaming or --data_mode memmap for large datasets
   Continuing with 'memory' mode as requested...
```

## Тестирование режимов

Для проверки корректности и производительности всех режимов используйте тестовый скрипт:

```bash
python -m python_lab.scripts.test_data_modes
```

Скрипт выполняет:
1. Загрузку данных во всех трех режимах
2. Parity check - проверку идентичности результатов
3. Измерение производительности и использования памяти

## Рекомендации

### Для малых датасетов (< 10 GB)
```bash
python -m python_lab.scripts.train --data_mode memory
```

### Для средних датасетов (10-50 GB)
```bash
# Если RAM достаточно
python -m python_lab.scripts.train --data_mode memory

# Если RAM недостаточно
python -m python_lab.scripts.train --data_mode streaming
```

### Для больших датасетов (> 50 GB)
```bash
# Первый запуск - создание кэша
python -m python_lab.scripts.train --data_mode memmap --cache_dir ./cache

# Или streaming без кэша
python -m python_lab.scripts.train --data_mode streaming
```

## Thread Safety

**Memory mode:**
- Thread-safe, можно использовать `num_workers=4` или больше

**Streaming mode:**
- Требует осторожности с multi-processing
- Рекомендуется `num_workers=2` для избежания блокировок файловых дескрипторов
- Каждый воркер должен иметь свой файловый дескриптор

**Memmap mode:**
- Thread-safe для чтения
- Можно использовать `num_workers=4` или больше

## Troubleshooting

### OOM (Out of Memory) в memory mode
```bash
# Переключитесь на streaming
python -m python_lab.scripts.train --data_mode streaming
```

### Медленная загрузка в streaming mode
```bash
# Размер батча для кэширования уже увеличен до 50000 в dataset.py
# Если все еще медленно, можно увеличить еще больше (редактируя код):
# self._batch_size = 100000  # Для очень больших датасетов
```

### Нехватка места на диске в memmap mode
```bash
# Используйте streaming вместо memmap
python -m python_lab.scripts.train --data_mode streaming
```

### Несоответствие результатов между режимами
```bash
# Запустите parity check
python -m python_lab.scripts.test_data_modes
```

## Технические детали реализации

### Parquet Native Memmap (Streaming)
- Polars использует PyArrow для чтения Parquet
- `memory_map=True` включен по умолчанию в `scan_parquet`
- Колоночный формат позволяет читать только нужные колонки
- Zero-copy доступ минимизирует накладные расходы

### Binary Memmap
- Данные конвертируются в `float32` numpy arrays
- Метки сохраняются как `int64`
- Метаданные (shape, seq_len, n_past_returns) в JSON
- Файлы: `features.npy`, `labels.npy`, `metadata.json`

### Индексная карта (Streaming)
- Для streaming режима создается индексная карта
- Позволяет мгновенно находить нужную строку без перебора
- Кэширование батчей оптимизирует последовательный доступ

## Ссылки

- [Polars Streaming Documentation](https://docs.pola.rs/user-guide/lazy/streaming/)
- [NumPy Memmap Documentation](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)
- [PyArrow Memory Mapping](https://arrow.apache.org/docs/python/memory.html)
