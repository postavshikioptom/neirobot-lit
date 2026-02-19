# Задача 150: Stress Testing High Frequency Pipeline

## Описание

Комплексное стресс-тестирование торгового конвейера под нагрузкой 20,000+ msg/sec с использованием Criterion.rs для выявления микро-задержек.

## Режимы работы

### Mock Mode (по умолчанию)
Использует легковесный мок ML модели для быстрых тестов. Подходит для CI/CD и разработки.

### Real Model Mode
Использует реальную ONNX модель для production validation. Требует наличия модели в `bots/BTCUSDT/model/model.onnx`.

## Бенчмарки

### 1. `bench_full_hot_path` - Основной End-to-End тест
- **Цель**: Измерение латентности полного цикла обработки
- **Нагрузка**: 20,000 msg/sec
- **Длительность**: 10 секунд
- **Критерии**: Среднее время < 500 мкс
- **Watchdog**: Проверяется на каждой итерации

### 2. `bench_burst_load` - Тест всплеска нагрузки
- **Цель**: Проверка устойчивости к резким скачкам нагрузки
- **Нагрузка**: 50,000 msg/sec
- **Длительность**: 10 секунд
- **Критерии**: Отсутствие паники или потери дескрипторов
- **Watchdog**: Проверяется каждые 1000 сообщений

### 3. `bench_sustained_load_with_memory` - Тест на утечки памяти
- **Цель**: Проверка роста памяти при длительной работе
- **Нагрузка**: 20,000 msg/sec
- **Длительность**: 60 секунд
- **Критерии**: Рост памяти < 100 MB
- **Watchdog**: Проверяется каждые 10 секунд

### 4. `bench_backlog_detection` - Тест заполненности каналов
- **Цель**: Проверка, что каналы не переполняются
- **Нагрузка**: 20,000 msg/sec
- **Длительность**: 5 секунд
- **Критерии**: `tx.capacity() > 0` всегда

### 5. `bench_latency_distribution` - Анализ распределения задержек
- **Цель**: Проверка P99.9 латентности
- **Нагрузка**: 20,000 msg/sec
- **Длительность**: 30 секунд
- **Критерии**: P99.9 < 2 мс
- **Watchdog**: Финальная проверка после теста

### 6. `bench_real_model_inference` - Реальная ONNX модель (NEW!)
- **Цель**: Измерение влияния высокой частоты на session.run()
- **Нагрузка**: 20,000 msg/sec
- **Длительность**: 30 секунд
- **Требования**: Реальная модель в `bots/BTCUSDT/model/model.onnx`
- **Watchdog**: Проверяется каждые 1000 сообщений
- **Метрики**: Mean и P99 inference time

## Интеграция с Watchdog

Все бенчмарки интегрированы с системой мониторинга Watchdog (задача 146):
- Метрика `bot_watchdog_last_check_timestamp` обновляется при каждой обработке
- Если watchdog не обновлялся более 5 секунд - тест провален
- Проверка выполняется автоматически через `check_watchdog_health()`

## Запуск

### Запуск всех бенчмарков
```bash
cargo bench --bench hot_path
```

### Запуск конкретного бенчмарка
```bash
cargo bench --bench hot_path -- pipeline_tick_20k_tps
cargo bench --bench hot_path -- burst_50k_tps_10sec
cargo bench --bench hot_path -- sustained_20k_tps_60sec
cargo bench --bench hot_path -- channel_capacity_check
cargo bench --bench hot_path -- p999_latency_check
cargo bench --bench hot_path -- real_onnx_20k_tps  # Требует реальную модель
```

### Запуск с реальной моделью
Для тестирования с реальной ONNX моделью:
1. Убедитесь, что модель существует: `bots/BTCUSDT/model/model.onnx`
2. Запустите бенчмарк: `cargo bench --bench hot_path -- real_onnx_20k_tps`

Бенчмарк автоматически пропустится, если модель не найдена.

### Запуск с профилированием
```bash
cargo bench --bench hot_path -- --profile-time=5
```

## Отчеты

После запуска Criterion генерирует HTML отчеты в директории `target/criterion/`:
- `target/criterion/hot_path/pipeline_tick_20k_tps/report/index.html`
- `target/criterion/burst_load/burst_50k_tps_10sec/report/index.html`
- И т.д.

Отчеты содержат:
- Графики распределения задержек
- Статистику (mean, median, P99, P99.9)
- Сравнение с предыдущими запусками
- Violin plots для визуализации распределения

## Критерии приемки

- [x] Среднее время прохождения всего цикла (End-to-End) < 500 микросекунд при 20k TPS
- [x] P99.9 задержка не превышает 2мс (отсутствие тяжелых хвостов распределения)
- [x] Отсутствие роста памяти (RSS) в течение 60-секундного теста
- [x] `criterion` генерирует отчет (HTML) с графиками распределения задержек
- [x] Система выдерживает "всплеск" до 50,000 msg/sec в течение 10 секунд без паники

## Архитектура

### MlEngineMode (NEW!)
Enum для выбора между Mock и Real ONNX движком:
- `Mock(MockOnnxEngine)` - быстрый мок для CI/CD
- `Real(OnnxEngine)` - реальная модель для production validation

### MockOnnxEngine
Легковесный мок ML модели для быстрого инференса (~10-50 мкс) без загрузки реальной ONNX модели.

### OnnxEngine (Real)
Реальный ONNX движок из `src/ml/onnx.rs`:
- Загружает модель из файла
- Использует CPU execution provider для бенчмарков
- Измеряет реальное влияние высокой частоты на `session.run()`

### LobUpdateGenerator
Генератор реалистичных LOB Delta обновлений с:
- Случайными отклонениями цены (±0.1%)
- Реалистичным спредом (0.01% - 0.05%)
- 1-3 уровнями обновлений

### TradingPipeline
Полный конвейер обработки:
1. **OrderBook** - обновление стакана
2. **Feature Extraction** - извлечение фич из стакана
3. **ML Inference** - предсказание модели (Mock или Real)
4. **Watchdog Update** - обновление метрики `bot_watchdog_last_check_timestamp`

### Watchdog Integration
Интеграция с системой мониторинга (задача 146):
- `check_watchdog_health()` - проверяет, что метрика обновлялась недавно
- Panic если watchdog не обновлялся более 5 секунд
- Автоматическая проверка во всех бенчмарках

## Примечания

- Бенчмарки используют `tokio::runtime` с multi-thread executor
- Rate limiting реализован через `tokio::time::sleep`
- Память измеряется через `/proc/self/status` (Linux) или `sysinfo` (Windows)
- Все бенчмарки используют `black_box` для предотвращения оптимизаций компилятора
