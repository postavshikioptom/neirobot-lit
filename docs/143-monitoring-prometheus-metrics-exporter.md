# Задача 143: Экспорт метрик в формате Prometheus (v2.0)

## 1. Реестр и метрики в [./src/monitoring/prometheus.rs](./src/monitoring/prometheus.rs)
Используем `OnceLock` для безопасной инициализации глобального реестра и векторные метрики для поддержки меток по символам.

```rust
use prometheus::{Encoder, IntCounterVec, GaugeVec, Registry, TextEncoder};
use std::sync::OnceLock;
use axum::{http::StatusCode, response::IntoResponse};

static REGISTRY: OnceLock<Registry> = OnceLock::new();
static TICK_COUNTER: OnceLock<IntCounterVec> = OnceLock::new();
static MEMORY_GAUGE: OnceLock<GaugeVec> = OnceLock::new();

pub fn registry() -> &'static Registry {
    REGISTRY.get_or_init(Registry::new)
}

pub fn init_metrics(symbol: &str) {
    let r = registry();
    
    let ticks = IntCounterVec::new(
        prometheus::opts!("bot_ticks_total", "Total incoming ticks"),
        &["symbol"]
    ).unwrap();
    r.register(Box::new(ticks.clone())).ok();
    ticks.with_label_values(&[symbol]).inc_by(0); // Force init
    TICK_COUNTER.set(ticks).ok();

    let memory = GaugeVec::new(
        prometheus::opts!("bot_memory_usage_bytes", "Memory usage in bytes"),
        &["symbol"]
    ).unwrap();
    r.register(Box::new(memory.clone())).ok();
    MEMORY_GAUGE.set(memory).ok();
}
```

## 2. Хендлер для Axum
Возвращаем `Vec<u8>` с правильным `Content-Type`, чтобы Prometheus корректно парсил данные.

```rust
pub async fn metrics_handler() -> impl IntoResponse {
    let mut buffer = Vec::new();
    let encoder = TextEncoder::new();
    
    // Сбор системных метрик перед отдачей
    update_system_metrics();

    let metric_families = registry().gather();
    encoder.encode(&metric_families, &mut buffer).unwrap();

    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4")],
        buffer
    )
}

fn update_system_metrics() {
    // В будущем: использование sysinfo для записи в MEMORY_GAUGE
}
```

## 3. Спорные моменты и корректировки (Grok + Zencoder)

- **OnceLock**: Согласен с Grok. Это стандарт Rust 1.70+, исключающий зависимости от `lazy_static`. Он потокобезопасен и более производителен.
- **Labels (Vec)**: Обязательно добавляем лейбл `symbol`. Без него в Grafana невозможно будет отличить PnL бота на BTC от бота на ETH, если они работают в одном кластере.
- **IntoResponse**: Возврат `Vec<u8>` вместо `String` — это правильный путь в Axum. Это исключает лишнюю проверку на валидность UTF-8 при отдаче бинарного буфера и позволяет избежать потенциальных паник.
- **System Metrics**: Добавляем `sysinfo` (опционально) для мониторинга потребления RAM. Это поможет вовремя заметить утечки памяти в ONNX-рантайме или буферах данных.
- **Tick Counter**: Инкремент счетчика тиков должен происходить в [./src/data/websocket.rs](./src/data/websocket.rs) при каждом входящем сообщении. Это позволит в Grafana рассчитать TPS (Ticks Per Second) через функцию `rate()`.

## 4. Инструкции для Gemini (Coder AI):
1. **Cargo.toml**: Добавить `prometheus = "0.14"` и `sysinfo = "0.38.1"`.
2. **[./src/monitoring/prometheus.rs](./src/monitoring/prometheus.rs)**: Реализовать реестр, векторные метрики и `metrics_handler`.
3. **[./src/monitoring/health.rs](./src/monitoring/health.rs)**: Добавить роут `.route("/metrics", get(metrics_handler))` в сервер Axum.
4. **[./src/bin/run-bot.rs](./src/bin/run-bot.rs)**: Вызвать `init_metrics(&config.symbol)` перед запуском основного цикла.

**Результат**: Промышленный уровень наблюдаемости. Полная совместимость с Prometheus/Grafana стеком, позволяющая отслеживать всё: от микро-задержек до потребления ресурсов сервером.
