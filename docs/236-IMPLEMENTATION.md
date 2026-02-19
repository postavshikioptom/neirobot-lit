# Задача 236: Реализация дисбаланса микро-трейдов

## Обзор
Реализован новый аналитический признак на основе потока публичных сделок (Public Trades), который позволяет модели детектировать агрессивное рыночное давление через маркет-ордера.

## Реализованные компоненты

### 1. Rust-часть (Сбор данных)

#### src/data/dump.rs
- Добавлена функция `start_trades_writer()` для асинхронной записи публичных сделок в Parquet
- Функция `flush_trades_to_parquet()` для батчевой записи trades в файлы `trades_{timestamp}.parquet`
- Схема данных: `timestamp`, `price`, `size`, `side`

#### src/bin/run-bot.rs
- Создан канал `(trades_tx, trades_rx)` для передачи сделок в writer
- Запущен фоновый воркер `start_trades_writer` в background runtime
- В обработке `WsData::Trades` добавлена отправка сделок в канал для записи

#### Существующая инфраструктура (уже была реализована)
- `src/data/websocket.rs`: Подписка на канал `publicTrade.{symbol}` уже присутствует
- `src/data/parser.rs`: Парсинг сообщений `parse_public_trade_msg()` уже реализован
- `src/data/types.rs`: Структура `PublicTradeOwned` с полями `price`, `amount`, `side`, `timestamp`

### 2. Python-часть (Расчет признаков)

#### python_lab/src/dataset.py

##### Функция `compute_trade_imbalance()`
Вычисляет дисбаланс публичных сделок для заданных временных окон.

**Параметры:**
- `df_snapshots`: DataFrame со снапшотами стакана (timestamp_ms, ...)
- `df_trades`: DataFrame с публичными сделками (timestamp, price, size, side)
- `windows`: список временных окон (например, ["1s", "5s", "15s", "60s"])
- `agg_type`: тип агрегации - 'vol' (объем) или 'count' (количество сделок)
- `noise_filter_pct`: процент для фильтрации шума (по умолчанию 0.05)

**Алгоритм:**
1. Фильтрация шума: отсекаются сделки меньше `noise_filter_pct` от медианного размера
2. Подготовка подписанных значений:
   - Для `agg_type='vol'`: `signed_val = size` (если Buy) или `-size` (если Sell)
   - Для `agg_type='count'`: `signed_val = 1` (если Buy) или `-1` (если Sell)
3. Агрегация по временным окнам через `group_by_dynamic`
4. Расчет imbalance: `sum(signed) / (sum(abs) + 1e-6)`
5. Сопоставление со snapshots через `join_asof` (backward strategy)

**Возвращает:**
DataFrame с добавленными колонками `imb_{agg_type}_{window}` для каждого окна.

##### Метод `LOBDataLoader.load_trades()`
Загружает все Parquet файлы с публичными сделками из директории `raw/`.

**Параметры:**
- `lazy`: если True, возвращает LazyFrame для отложенного выполнения

**Возвращает:**
DataFrame или LazyFrame с данными trades (timestamp, price, size, side)

#### python_lab/src/train.py
- Добавлена загрузка trades через `loader.load_trades()`
- Вызов `compute_trade_imbalance()` после загрузки данных, но перед FeatureEngineer
- Параметры по умолчанию:
  - `trade_imb_windows = ["1s", "5s", "15s", "60s"]`
  - `trade_imb_agg = "vol"`
  - `trade_noise_filter_pct = 0.05`

### 3. Тестирование

#### python_lab/tests/test_trade_imbalance.py
Тестовый файл с тремя тестами:
1. `test_compute_trade_imbalance()` - базовая функциональность
2. `test_empty_trades()` - обработка пустого DataFrame trades
3. `test_count_aggregation()` - агрегация по количеству сделок

**Запуск тестов:**
```bash
cd python_lab
python tests/test_trade_imbalance.py
```

## Конфигурация

### Параметры (можно добавить в config.toml позже)
- `trade_imb_windows`: список временных окон для агрегации (по умолчанию ["1s", "5s", "15s", "60s"])
- `trade_imb_agg`: тип агрегации - "vol" или "count" (по умолчанию "vol")
- `trade_noise_filter_pct`: процент для фильтрации шума (по умолчанию 0.05)

## Ожидаемый результат

1. **Файлы данных**: В папке `bots/{SYMBOL}/data/raw/` появляются файлы `trades_{timestamp}.parquet`
2. **Признаки**: В итоговый вектор фичей попадают колонки:
   - `imb_vol_1s`, `imb_vol_5s`, `imb_vol_15s`, `imb_vol_60s` (для agg_type='vol')
   - `imb_count_1s`, `imb_count_5s`, `imb_count_15s`, `imb_count_60s` (для agg_type='count')
3. **Сигнал**: Модель получает сигнал об агрессии рыночных покупателей/продавцов до того, как спред начнет смещаться

## Интерпретация признаков

- **Положительный imbalance** (> 0): Преобладают агрессивные покупки (Buy orders)
- **Отрицательный imbalance** (< 0): Преобладают агрессивные продажи (Sell orders)
- **Близкий к нулю** (≈ 0): Баланс между покупками и продажами

Чем больше абсолютное значение imbalance, тем сильнее рыночное давление в соответствующем направлении.

## Зависимости

### Rust
- `serde_json` - парсинг JSON сообщений
- `parquet` - запись в Parquet формат
- `polars` - работа с DataFrame

### Python
- `polars` - обработка данных
- `numpy` - численные операции

## Примечания

1. Функция `compute_trade_imbalance` использует `join_asof` с `strategy="backward"` для предотвращения заглядывания в будущее
2. Фильтрация шума критична для исключения микро-тиков от спам-ботов
3. Временные окна можно настраивать в зависимости от характеристик рынка
4. Для высокочастотных данных рекомендуется использовать меньшие окна (1s, 5s)
5. Для менее ликвидных инструментов можно использовать большие окна (60s, 300s)

## Дальнейшие улучшения

1. Добавить параметры в `config.toml` для гибкой настройки
2. Реализовать адаптивную фильтрацию шума на основе волатильности
3. Добавить экспоненциальное взвешивание для более свежих сделок
4. Реализовать multi-level imbalance (по разным ценовым уровням)
5. Добавить метрики качества признаков (feature importance)
