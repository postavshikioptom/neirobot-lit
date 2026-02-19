# Быстрый старт: Режимы загрузки данных

## Задача 094: Поддержка больших датасетов (100GB+)

Реализованы три режима загрузки данных для эффективного обучения на датасетах любого размера.

## Быстрый выбор режима

```bash
# Малый датасет (< 10 GB) - используйте memory
python -m python_lab.scripts.train --symbol BTCUSDT --data_mode memory

# Средний датасет (10-50 GB) - используйте streaming
python -m python_lab.scripts.train --symbol BTCUSDT --data_mode streaming

# Большой датасет (> 50 GB) - используйте memmap с кэшем
python -m python_lab.scripts.train --symbol BTCUSDT --data_mode memmap --cache_dir ./cache
```

## Тестирование

Проверьте корректность всех режимов:

```bash
python -m python_lab.scripts.test_data_modes
```

Скрипт выполнит:
- ✅ Загрузку данных во всех трех режимах
- ✅ Parity check (проверка идентичности результатов)
- ✅ Измерение производительности
- ✅ Мониторинг использования памяти

## Автоматическая валидация

При использовании `memory` mode система автоматически проверит доступную RAM:

```
⚠️  WARNING: Dataset size (~45.2 GB) may exceed available RAM (32.0 GB)
   Consider using --data_mode streaming or --data_mode memmap for large datasets
```

## Подробная документация

См. [DATA_MODES_GUIDE.md](./DATA_MODES_GUIDE.md) для:
- Детального описания каждого режима
- Технических деталей реализации
- Troubleshooting
- Примеров использования

## Зависимости

Установите обновленные зависимости:

```bash
pip install -r requirements.txt
```

Новая зависимость: `psutil>=6.0.0` для валидации ресурсов.

## Сравнение режимов

| Режим      | Скорость | RAM    | Диск   | Когда использовать |
|------------|----------|--------|--------|--------------------|
| memory     | ⚡⚡⚡    | 🔴 High | ✅ None | < 70% RAM         |
| streaming  | ⚡⚡      | ✅ Low  | ✅ None | > RAM             |
| memmap     | ⚡⚡⚡    | ✅ Low  | 🔴 3-5x | > RAM, reuse      |

## Примеры

### Обучение на малом датасете
```bash
python -m python_lab.scripts.train \
  --symbol BTCUSDT \
  --data_mode memory \
  --epochs 100 \
  --batch_size 128
```

### Обучение на большом датасете (streaming)
```bash
python -m python_lab.scripts.train \
  --symbol BTCUSDT \
  --data_mode streaming \
  --epochs 100 \
  --batch_size 128
```

### Обучение на большом датасете (memmap с кэшем)
```bash
# Первый запуск - создание кэша
python -m python_lab.scripts.train \
  --symbol BTCUSDT \
  --data_mode memmap \
  --cache_dir ./bots/BTCUSDT/models/cache \
  --epochs 100 \
  --batch_size 128

# Последующие запуски - использование кэша (быстрее)
python -m python_lab.scripts.train \
  --symbol BTCUSDT \
  --data_mode memmap \
  --cache_dir ./bots/BTCUSDT/models/cache \
  --epochs 100 \
  --batch_size 128
```

## Troubleshooting

### OOM (Out of Memory)
```bash
# Переключитесь на streaming
python -m python_lab.scripts.train --data_mode streaming
```

### Медленная загрузка в streaming
Увеличьте размер батча для кэширования в `dataset.py`:
```python
self._batch_size = 50000  # Вместо 10000
```

### Нехватка места на диске в memmap
```bash
# Используйте streaming вместо memmap
python -m python_lab.scripts.train --data_mode streaming
```

## Технические детали

### Streaming Mode
- Использует `polars.scan_parquet()` с `low_memory=True`
- Данные читаются через `collect(engine="streaming")`
- Кэширование батчей размером 10000 строк
- Thread-safe с `num_workers=2`

### Memmap Mode
- Конвертация в uncompressed numpy binary
- Файлы: `features.npy`, `labels.npy`, `metadata.json`
- Переиспользование кэша между запусками
- Thread-safe с `num_workers=4`

## Статус

✅ Задача 094 полностью реализована и протестирована.
