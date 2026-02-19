# Задача 135: Health-Check эндпоинт на базе Axum (v2.0)

## 1. Общее состояние в `src/monitoring/health.rs`
Создай структуру `SharedState`, использующую атомарные типы для потокобезопасного обновления метрик из основного цикла без блокировок (Lock-free).

```rust
// В [./src/monitoring/health.rs](./src/monitoring/health.rs)
use axum::{extract::State, http::StatusCode, routing::get, Json, Router};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tokio::time::Instant;

pub struct SharedState {
    pub last_update: AtomicU64,      // Unix ms
    pub ws_connected: AtomicBool,
    pub emergency_mode: AtomicBool,
    pub start_time: Instant,
}

#[derive(serde::Serialize)]
pub struct HealthStatus {
    pub status: String,
    pub uptime_sec: u64,
    pub last_update_ms_ago: u64,
    pub ws_connected: bool,
}
```

## 2. Реализация сервера и хендлера
Используем `Router::with_state` для проброса состояния и возвращаем `503`, если бот не в норме.

```rust
pub async fn start_health_server(config: HealthConfig, state: Arc<SharedState>) {
    let app = Router::new()
        .route("/health", get(health_handler))
        .with_state(state);

    let addr: std::net::SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .expect("Invalid health check address");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn health_handler(
    State(state): State<Arc<SharedState>>
) -> (StatusCode, Json<HealthStatus>) {
    let now = crate::utils::helpers::unix_ms();
    let last = state.last_update.load(Ordering::Relaxed);
    let diff = now.saturating_sub(last);
    
    let is_connected = state.ws_connected.load(Ordering::Relaxed);
    let is_emergency = state.emergency_mode.load(Ordering::Relaxed);
    
    // Условие падения: нет данных > 30с или потерян WS
    let is_down = diff > 30_000 || !is_connected || is_emergency;
    
    let status = HealthStatus {
        status: if is_down { "down".into() } else { "up".into() },
        uptime_sec: state.start_time.elapsed().as_secs(),
        last_update_ms_ago: diff,
        ws_connected: is_connected,
    };

    let code = if is_down { StatusCode::SERVICE_UNAVAILABLE } else { StatusCode::OK };
    (code, Json(status))
}
```

## 3. Спорные моменты и корректировки (Grok + Zencoder)

*   **Axum State**: Согласен с Grok. В Axum 0.7 правильный способ передачи `Arc` — через метод `.with_state()` и экстрактор `State`. Это обеспечивает типобезопасность и чистоту кода.
*   **Liveness Logic**: Бот считается живым, только если `last_update` (время последнего тика из стакана) актуально. Это предотвращает ситуации «зомби-бота», когда процесс висит, но данных нет.
*   **Networking**: Биндимся на `SocketAddr`, полученный из конфига. По умолчанию используем `127.0.0.1`, чтобы не открывать диагностику всему интернету.
*   **Lock-free**: Мы специально используем `AtomicBool` и `AtomicU64`, чтобы HTTP-запросы от внешних систем мониторинга **никак не влияли на задержки (latency)** основного торгового потока.

## 4. Интеграция в `run-bot.rs`
```rust
// В [./src/bin/run-bot.rs](./src/bin/run-bot.rs)
let shared_state = Arc::new(SharedState {
    last_update: AtomicU64::new(helpers::unix_ms()),
    ws_connected: AtomicBool::new(true),
    emergency_mode: AtomicBool::new(false),
    start_time: Instant::now(),
});

// Запуск в отдельной задаче
let state_for_server = shared_state.clone();
tokio::spawn(async move {
    start_health_server(config.monitoring.clone(), state_for_server).await;
});

// В основном цикле обновления OrderBook:
shared_state.last_update.store(helpers::unix_ms(), Ordering::Relaxed);
```

## 5. Инструкции для Gemini (Coder AI):
1.  **Cargo.toml**: Добавить `axum = { version = "0.7", features = ["json"] }`.
2.  **[./src/monitoring/health.rs](./src/monitoring/health.rs)**: Реализовать `SharedState`, `health_handler` и `start_health_server`.
3.  **[./src/bin/run-bot.rs](./src/bin/run-bot.rs)**: Инициализировать состояние и обновлять `last_update` при каждом `apply_update` стакана.
4.  **[./src/config/types.rs](./src/config/types.rs)**: Добавить `HealthConfig` (host, port, timeout_sec).

**Результат**: Надежный диагностический интерфейс, совместимый с современными системами оркестрации (Kubernetes/Docker) и мониторинга.
