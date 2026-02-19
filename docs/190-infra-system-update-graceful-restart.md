# Задача №190: [./docs/190-infra-system-update-graceful-restart.md](./docs/190-infra-system-update-graceful-restart.md)

## План реализации

1.  **Изменения в зависимостях**:
    - Убедиться в наличии `serde_json = "1"` в `Cargo.toml`.

2.  **Изменения в [./src/config/types.rs](./src/config/types.rs)**:
    - Добавить в структуру `BotConfig` поле `max_state_age_ms: u64` (по умолчанию 60000) для контроля актуальности сохраненного состояния.

3.  **Создание модуля [./src/utils/persistence.rs](./src/utils/persistence.rs)**:
    - Реализовать структуру `BotState`:
        - `position: Option<Position>`: текущая открытая позиция.
        - `active_orders: Vec<OrderIntent>`: список активных намерений/ордеров из `OrderManager`.
        - `timestamp_ms: u64`: время сохранения (UNIX ms).
    - Реализовать функции `save_state(path, state)` и `load_state(path) -> Result<BotState>`.

4.  **Изменения в [./src/bin/run-bot.rs](./src/bin/run-bot.rs)**:
    - **Graceful Shutdown**: Настроить перехват сигналов `SIGINT/SIGTERM` через `tokio::signal::unix`.
    - При получении сигнала вызывать остановку торгового цикла и сохранение `BotState` через `utils::persistence`.
    - **Recovery**: При старте проверять наличие файла `bots/SYMBOL/state.json`.
    - Если `now - state.timestamp_ms < config.max_state_age_ms`, загружать данные в `PositionManager` и `OrderManager`.

5.  **Синхронизация и валидация**:
    - В [./src/trading/order_manager.rs](./src/trading/order_manager.rs) добавить логику сверки восстановленных ордеров с биржей через `GET /v5/order/realtime`.
    - Если состояние устарело или данные биржи расходятся — игнорировать файл и выполнять полную инициализацию.

## Ожидаемый результат
- Бот корректно сохраняет позицию и активные ордера при получении сигнала остановки.
- При перезапуске (например, после обновления бинарника) бот восстанавливает контекст без необходимости экстренного закрытия позиций.
- Логи содержат записи `[Persistence] State saved` и `[Persistence] State restored (age: X ms)`.