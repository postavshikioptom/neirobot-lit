# Задача 236: Дисбаланс микро-трейдов (Micro-Trades Imbalance)

Реализация нового аналитического признака, основанного на потоке рыночных сделок (Public Trades). Признак позволяет модели детектировать агрессивное рыночное давление, которое реализуется через маркет-ордера и часто предшествует изменениям в стакане.

## 1. Цель задачи
Организовать сбор данных о сделках (trades) в реальном времени, их сохранение в Parquet и расчет дисбаланса объемов/количества сделок за фиксированные окна в Python-лаборатории.

## 2. Инструкции по реализации для Gemini

### А. Сбор данных (Rust): Расширение [./src/data/websocket.rs](./src/data/websocket.rs) и [./src/data/dump.rs](./src/data/dump.rs)
1.  **Подписка**: Добавить в WebSocket-клиент подписку на канал `publicTrade.SYMBOL`.
2.  **Парсинг**: Реализовать структуру `TradeUpdate` (`price`, `size`, `side`, `timestamp`).
3.  **Дамп**: Расширить логику задачи 017 для записи сделок в отдельный файл `bots/SYMBOL/data/raw/trades.parquet`.

### Б. Расчет признака (Python): [./python_lab/src/dataset.py](./python_lab/src/dataset.py)
1.  **Фильтрация шума**: Добавить порог `min_size_threshold` (например, `size > 0.001 * avg_daily_vol`), чтобы исключить микро-тики, создаваемые ботами для спама.
2.  **Алгоритм агрегации (Polars)**:
    Реализовать функцию `compute_trade_imbalance`, использующую `join_asof` для сопоставления трейдов со снапшотами стакана без «заглядывания в будущее»:
    ```python
    def compute_trade_imbalance(df_snapshots: pl.DataFrame, df_trades: pl.DataFrame, windows: list[str], agg_type: str = 'vol') -> pl.DataFrame:
        # 1. Фильтрация шума
        df_trades = df_trades.filter(pl.col('size') > threshold)
        
        # 2. Подготовка подписанного объема/количества
        if agg_type == 'vol':
            df_trades = df_trades.with_columns(
                (pl.when(pl.col('side') == 'Buy').then(pl.col('size')).otherwise(-pl.col('size'))).alias('signed_val')
            )
        else: # count
            df_trades = df_trades.with_columns(
                (pl.when(pl.col('side') == 'Buy').then(1).otherwise(-1)).alias('signed_val')
            )

        # 3. Rolling Imbalance через join_asof и group_by_dynamic
        for w in windows:
            # Расчет суммы signed_val и абсолютной суммы за окно 'w'
            # Формула: sum(signed) / (sum(abs) + 1e-6)
            # ... (реализация через polars rolling_sum)
    ```

## 3. Конфигурация
Добавить в настройки датасета:
-   **trade_imb_windows**: `["1s", "5s", "15s", "60s"]`.
-   **trade_imb_agg**: `"vol"` или `"count"`.
-   **trade_noise_filter_pct**: `0.05` (отсекать сделки меньше 5% от медианного размера).

## 4. Ожидаемый результат
1.  В папке `raw/` появляются файлы `trades.parquet`.
2.  В итоговый вектор фичей попадают колонки `imb_vol_1s`, `imb_count_1s` и т.д.
3.  Модель получает сигнал об агрессии рыночных покупателей/продавцов до того, как спред начнет смещаться.

## 5. Необходимые зависимости
-   **Rust**: `serde_json`, `parquet`.
-   **Python**: `polars`, `numpy`.