
# Задача 203: Исследование распределения Fill Rate (Fill Rate Distribution Study)

## 1. Цель задачи
Провести статистический анализ зависимости исполнения ордеров от состояния стакана (LOB). Понять, как объем уровня и дисбаланс (Imbalance) влияют на вероятность `Full Fill` и время нахождения ордера в стакане.

## 2. Инструкции для Gemini по реализации

### А. Изменения в Rust-ядре (Context & Config)

1.  **Конфигурация ([./src/config/types.rs](./src/config/types.rs))**:
    *   Добавить в `BotConfig` флаг: `enable_fill_rate_logging: bool` (по умолчанию `false`). Это предотвратит лишнюю нагрузку на диск в продакшене.

2.  **Аппроксимация очереди ([./src/data/orderbook.rs](./src/data/orderbook.rs))**:
    *   **ВАЖНО**: API Bybit не отдает наше точное место в очереди (Queue Position).
    *   Реализовать метод `get_volume_at_price(price: f64) -> f64`.
    *   В логах использовать `total_volume_at_level` как оценку «соперничающего» объема. В аналитике будем считать это «худшим сценарием» очереди.

3.  **Логирование контекста ([./src/utils/logger.rs](./src/utils/logger.rs))**:
    *   Если флаг включен, писать в **`order_context.csv`**:
        *   `order_id`: Для связи с `trades.csv` и `execution_quality.csv` (задача 202).
        *   `level_total_vol`: Общий объем в стакане по цене нашего ордера в момент выставления.
        *   `imbalance_5l`: Дисбаланс на 5 уровнях (Bids Vol / Asks Vol).
        *   `order_size`: Размер нашего ордера.
        *   `fill_duration_us`: Время (u64) от `New` до `Filled` (или `Cancelled`).

### Б. Изменения в Python Lab (Analytics)

1.  **Скрипт анализа**:
    *   Путь: **[./python_lab/scripts/fill_rate_study.py](./python_lab/scripts/fill_rate_study.py)**.
    *   **Методология**:
        *   Объединить данные по `order_id` из трех файлов: `trades`, `execution_quality` (для reversion/latency) и `order_context`.
        *   Рассчитать корреляцию между `level_total_vol` и `fill_duration_us`.
        *   Построить **Logistic Regression**: вероятность исполнения в зависимости от `order_size` и `imbalance`.

## 3. Критика и аргументация (По правкам Grok)
*   **Спор об очереди**: Согласен с Grok. Точная очередь невозможна без Private WebSocket Feed с данными о `matching engine`, чего Bybit не дает. Используем `total_volume` как прокси.
*   **Спор о токсичности**: Мы не делаем новый Snapshot после сделки. Вместо этого мы используем колонку `reversion` из файла задачи 202 (`execution_quality.csv`), объединяя их в Python. Это избавляет Rust от сложной логики «таймеров после сделки».
*   **Спор о флаге**: Флаг `enable_fill_rate_logging` обязателен, так как запись контекста стакана на каждый ордер — это «тяжелая» операция для NVMe при высокой частоте сделок.

## 4. Ожидаемый результат
Модель в Python, которая выдает коэффициент «вероятности успеха» для каждого нового сигнала. Если вероятность < 20% (слишком большая «стена» перед нами), бот может игнорировать сигнал, экономя на комиссиях за отмену.

---
**Gemini, при реализации в `orderbook.rs` убедись, что поиск `get_volume_at_price` не делает полный перебор стакана, а использует `BTreeMap::get()`.**