# Документ требований: Консолидация системы мониторинга

## Введение

Проект neirobot_lit содержит два отдельных модуля мониторинга Prometheus, которые были созданы в разное время и используют разные подходы к регистрации метрик. Это создает архитектурную избыточность, усложняет поддержку и может привести к конфликтам при инициализации.

Модуль `src/monitoring/prometheus.rs` (задача 143) использует крейт `prometheus` с ручной регистрацией метрик через Registry и глобальные статические переменные OnceLock. Модуль `src/monitoring/metrics.rs` (задача 189) использует современный подход с крейтами `metrics` + `metrics-exporter-prometheus`, где метрики регистрируются автоматически через макросы.

Цель данного рефакторинга - объединить оба модуля в единую систему мониторинга, используя современный подход с автоматической регистрацией метрик, сохранив при этом всю существующую функциональность и обеспечив обратную совместимость.

## Глоссарий

- **Monitoring_System**: Система мониторинга Prometheus в проекте neirobot_lit
- **Legacy_Module**: Старый модуль `src/monitoring/prometheus.rs` с ручной регистрацией метрик
- **Modern_Module**: Новый модуль `src/monitoring/metrics.rs` с автоматической регистрацией
- **Metric**: Измеряемая величина (counter, gauge, histogram) в системе Prometheus
- **Counter**: Тип метрики, который только увеличивается (например, количество тиков)
- **Gauge**: Тип метрики, который может увеличиваться и уменьшаться (например, использование памяти)
- **Histogram**: Тип метрики для измерения распределения значений (например, длительность инференса)
- **Symbol**: Торговый символ (например, "BTCUSDT"), используемый как label в метриках
- **System_Metrics**: Метрики системных ресурсов (CPU, память)
- **Exporter**: HTTP-сервер, который отдает метрики в формате Prometheus

## Требования

### Требование 1: Миграция на современный стек метрик

**User Story:** Как разработчик, я хочу использовать единый современный подход к метрикам, чтобы упростить поддержку и избежать конфликтов при инициализации.

#### Критерии приемки

1. THE Monitoring_System SHALL использовать крейты `metrics` и `metrics-exporter-prometheus` для всех метрик
2. THE Monitoring_System SHALL удалить зависимость от крейта `prometheus` после завершения миграции
3. WHEN система инициализируется, THE Monitoring_System SHALL регистрировать все метрики через единый exporter
4. THE Monitoring_System SHALL использовать макросы `metrics::counter!`, `metrics::gauge!`, `metrics::histogram!` вместо глобальных статических переменных

### Требование 2: Сохранение всех существующих метрик

**User Story:** Как оператор системы, я хочу, чтобы все существующие метрики продолжали работать после рефакторинга, чтобы не потерять возможность мониторинга.

#### Критерии приемки

1. THE Monitoring_System SHALL сохранить метрику `bot_ticks_total` с label `symbol`
2. THE Monitoring_System SHALL сохранить метрику `bot_signal_oscillations_handled_total` с label `symbol`
3. THE Monitoring_System SHALL сохранить метрику `bot_memory_usage_bytes` с label `symbol`
4. THE Monitoring_System SHALL сохранить метрику `bot_cpu_usage_percent` с label `symbol`
5. THE Monitoring_System SHALL сохранить метрику `bot_watchdog_stall_seconds` с label `symbol`
6. THE Monitoring_System SHALL сохранить метрику `bot_watchdog_last_check_timestamp` с label `symbol`
7. THE Monitoring_System SHALL сохранить метрику `bot_time_decay_exits_total` с label `symbol`
8. THE Monitoring_System SHALL сохранить метрику `bot_maker_fills_total` с label `symbol`
9. THE Monitoring_System SHALL сохранить метрику `bot_taker_fills_total` с label `symbol`
10. THE Monitoring_System SHALL сохранить метрику `bot_ws_messages_total`
11. THE Monitoring_System SHALL сохранить метрику `bot_inference_duration_us`
12. THE Monitoring_System SHALL сохранить метрику `bot_realized_pnl_bps`
13. THE Monitoring_System SHALL сохранить метрику `bot_unrealized_pnl_bps`
14. THE Monitoring_System SHALL сохранить метрику `bot_health_status`
15. THE Monitoring_System SHALL сохранить метрику `bot_orders_placed_total`
16. THE Monitoring_System SHALL сохранить метрику `bot_order_rejections_total`

### Требование 3: Обновление системных метрик

**User Story:** Как оператор системы, я хочу видеть актуальные значения CPU и памяти в метриках, чтобы отслеживать состояние системы.

#### Критерии приемки

1. WHEN метрики запрашиваются через HTTP endpoint, THE Monitoring_System SHALL обновить значения `bot_memory_usage_bytes` из `METRICS.rss_bytes`
2. WHEN метрики запрашиваются через HTTP endpoint, THE Monitoring_System SHALL обновить значения `bot_cpu_usage_percent` из `METRICS.cpu_usage`
3. THE Monitoring_System SHALL читать системные метрики из `crate::utils::sys::METRICS` с использованием `Ordering::Relaxed`
4. THE Monitoring_System SHALL корректно обрабатывать случай, когда symbol не инициализирован

### Требование 4: Обратная совместимость вызовов метрик

**User Story:** Как разработчик, я хочу, чтобы все существующие вызовы метрик продолжали работать без изменений, чтобы минимизировать объем изменений в коде.

#### Критерии приемки

1. WHEN код вызывает `metrics::counter!("bot_ticks_total", "symbol" => symbol).increment(1)`, THE Monitoring_System SHALL корректно инкрементировать метрику
2. WHEN код вызывает `metrics::counter!("bot_signal_oscillations_handled_total", "symbol" => symbol).increment(1)`, THE Monitoring_System SHALL корректно инкрементировать метрику
3. WHEN код вызывает `metrics::counter!("bot_time_decay_exits_total", "symbol" => symbol).increment(1)`, THE Monitoring_System SHALL корректно инкрементировать метрику
4. WHEN код вызывает `metrics::counter!("bot_maker_fills_total", "symbol" => symbol).increment(1)`, THE Monitoring_System SHALL корректно инкрементировать метрику
5. WHEN код вызывает `metrics::counter!("bot_taker_fills_total", "symbol" => symbol).increment(1)`, THE Monitoring_System SHALL корректно инкрементировать метрику
6. WHEN код вызывает `metrics::gauge!("bot_watchdog_stall_seconds", "symbol" => symbol).set(value)`, THE Monitoring_System SHALL корректно установить значение метрики
7. WHEN код вызывает `metrics::gauge!("bot_watchdog_last_check_timestamp", "symbol" => symbol).set(value)`, THE Monitoring_System SHALL корректно установить значение метрики

### Требование 5: Единая точка инициализации

**User Story:** Как разработчик, я хочу иметь единую функцию инициализации метрик, чтобы упростить настройку системы мониторинга.

#### Критерии приемки

1. THE Monitoring_System SHALL предоставить функцию `init_metrics_exporter(port: u16, symbol: &str)` для инициализации
2. WHEN функция инициализации вызывается, THE Monitoring_System SHALL запустить HTTP exporter на указанном порту
3. WHEN функция инициализации вызывается, THE Monitoring_System SHALL зарегистрировать описания всех метрик
4. WHEN функция инициализации вызывается, THE Monitoring_System SHALL сохранить symbol для использования в системных метриках
5. THE Monitoring_System SHALL логировать успешную инициализацию с указанием порта

### Требование 6: Удаление дублирующего кода

**User Story:** Как разработчик, я хочу удалить старый модуль после миграции, чтобы избежать путаницы и упростить кодовую базу.

#### Критерии приемки

1. WHEN миграция завершена, THE Monitoring_System SHALL удалить файл `src/monitoring/prometheus.rs`
2. WHEN миграция завершена, THE Monitoring_System SHALL удалить все импорты из `crate::monitoring::prometheus`
3. WHEN миграция завершена, THE Monitoring_System SHALL удалить зависимость `prometheus` из `Cargo.toml`
4. WHEN миграция завершена, THE Monitoring_System SHALL обновить вызов инициализации в `src/bin/run-bot.rs` для использования единой функции

### Требование 7: Обновление тестов

**User Story:** Как разработчик, я хочу иметь тесты, которые проверяют все метрики, чтобы гарантировать корректность работы системы мониторинга.

#### Критерии приемки

1. THE Monitoring_System SHALL иметь тест, проверяющий наличие всех 16 метрик в HTTP endpoint
2. THE Monitoring_System SHALL иметь тест, проверяющий корректность инкремента counter-метрик
3. THE Monitoring_System SHALL иметь тест, проверяющий корректность установки gauge-метрик
4. THE Monitoring_System SHALL иметь тест, проверяющий корректность записи histogram-метрик
5. THE Monitoring_System SHALL иметь тест, проверяющий обновление системных метрик (CPU, память)
6. THE Monitoring_System SHALL иметь тест, проверяющий работу метрик с labels (symbol)

### Требование 8: Документирование метрик

**User Story:** Как оператор системы, я хочу видеть описания метрик в Prometheus/Grafana, чтобы понимать их назначение.

#### Критерии приемки

1. THE Monitoring_System SHALL зарегистрировать описание для метрики `bot_ticks_total`
2. THE Monitoring_System SHALL зарегистрировать описание для метрики `bot_signal_oscillations_handled_total`
3. THE Monitoring_System SHALL зарегистрировать описание для метрики `bot_memory_usage_bytes`
4. THE Monitoring_System SHALL зарегистрировать описание для метрики `bot_cpu_usage_percent`
5. THE Monitoring_System SHALL зарегистрировать описание для метрики `bot_watchdog_stall_seconds`
6. THE Monitoring_System SHALL зарегистрировать описание для метрики `bot_watchdog_last_check_timestamp`
7. THE Monitoring_System SHALL зарегистрировать описание для метрики `bot_time_decay_exits_total`
8. THE Monitoring_System SHALL зарегистрировать описание для метрики `bot_maker_fills_total`
9. THE Monitoring_System SHALL зарегистрировать описание для метрики `bot_taker_fills_total`
10. THE Monitoring_System SHALL использовать `metrics::describe_counter!`, `metrics::describe_gauge!`, `metrics::describe_histogram!` для регистрации описаний
