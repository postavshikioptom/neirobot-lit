# Задача №189: [./docs/189-monitoring-prometheus-metrics-exporter.md](./docs/189-monitoring-prometheus-metrics-exporter.md)

## План реализации

1.  **Обновление зависимостей**:
    - Добавить в `Cargo.toml` крейты `metrics = "0.23.0"` и `metrics-exporter-prometheus = "0.15.1"`.

2.  **Создание модуля мониторинга**:
    - Создать файл [./src/monitoring/metrics.rs](./src/monitoring/metrics.rs).
    - Реализовать функцию `init_metrics_exporter(port: u16)`, инициализирующую `PrometheusBuilder` с HTTP-слушателем на `0.0.0.0:port`.
    - Зарегистрировать описания метрик через `describe_counter!`, `describe_gauge!` и `describe_histogram!` для улучшения читаемости в Prometheus/Grafana.

3.  **Интеграция в компоненты**:
    - **[./src/data/websocket.rs](./src/data/websocket.rs)**: инкремент `bot_ws_messages_total` при получении каждого сообщения от Bybit.
    - **[./src/ml/onnx.rs](./src/ml/onnx.rs)**: запись длительности инференса в `bot_inference_duration_us`.
    - **[./src/trading/position_manager.rs](./src/trading/position_manager.rs)**: обновление `bot_realized_pnl_bps` и `bot_unrealized_pnl_bps`.
    - **[./src/risk/health_monitor.rs](./src/risk/health_monitor.rs)**: установка `bot_health_status` (1 для OK, 0 для критических ошибок/блокировки).
    - **[./src/trading/execution.rs](./src/trading/execution.rs)**: счетчики `bot_orders_placed_total` и `bot_order_rejections_total`.

4.  **Точка входа**:
    - В [./src/bin/run-bot.rs](./src/bin/run-bot.rs) вызвать инициализацию экспортера сразу после загрузки `BotConfig`.
    - Порт для метрик должен считываться из секции `[monitoring]` конфигурационного файла.

5.  **Тестирование**:
    - Создать тест в `tests/monitoring_test.rs`, проверяющий доступность эндпоинта `/metrics` и наличие ключевых метрик в выводе.

## Ожидаемый результат
- Бот экспортирует метрики в формате Prometheus на порту 9090 (по умолчанию).
- Все критические показатели (PnL, задержки, ошибки) доступны для сбора внешней системой мониторинга.
- Реализация не вносит значимых задержек в основной торговый цикл (используется атомарное обновление метрик в памяти).