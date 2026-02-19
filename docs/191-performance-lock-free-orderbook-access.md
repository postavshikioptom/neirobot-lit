# Задача №191: Оптимизация производительности: Lock-Free доступ к стакану (OrderBook)

[./docs/191-performance-lock-free-orderbook-access.md](./docs/191-performance-lock-free-orderbook-access.md)

## План реализации

1.  **Изменения в зависимостях**:
    - Добавить `arc-swap = "1.7"` в `Cargo.toml`.

2.  **Рефакторинг [./src/data/orderbook.rs](./src/data/orderbook.rs)**:
    - Создать структуру `OrderBookSnapshot`, включающую `bids`, `asks`, `last_update_id`, `timestamp_ms` и `checksum`.
    - В основной структуре `OrderBook` реализовать поле `current_snapshot: ArcSwap<OrderBookSnapshot>`.
    - В методе `apply_update` после успешного применения инкрементов и проверки контрольной суммы (задача 180) обновлять снапшот: `self.current_snapshot.store(Arc::new(new_snapshot))`.
    - Использовать `Arc::make_mut` для оптимизации аллокаций при подготовке нового снапшота.

3.  **Интеграция в [./src/bin/run-bot.rs](./src/bin/run-bot.rs)**:
    - Обновить основной торговый цикл: заменить блокировку `Mutex<OrderBook>` на получение снапшота через `ob.current_snapshot.load()`.
    - Передавать полученный `Guard<Arc<OrderBookSnapshot>>` в функции подготовки данных в [./src/ml/tensor_builder.rs](./src/ml/tensor_builder.rs) и логику исполнения в [./src/trading/execution.rs](./src/trading/execution.rs).
    - Это гарантирует, что на протяжении всего прохода (Inference -> Execution) стратегия работает с одним и тем же неизменным срезом данных, даже если WebSocket получает новые пакеты.

4.  **Валидация (Unit-тесты)**:
    - В [./src/data/orderbook.rs](./src/data/orderbook.rs) добавить тест на многопоточное чтение/запись, проверяющий отсутствие блокировок и консистентность данных (чтение не должно возвращать "битый" или частично обновленный стакан).

## Ожидаемый результат
- Основной торговый цикл полностью освобожден от блокировок `Mutex` при доступе к стакану.
- Время доступа к данным стакана для инференса становится константным и минимальным (load атомарного указателя).
- Устранена конкуренция (contention) между потоком WebSocket-клиента и логикой стратегии.