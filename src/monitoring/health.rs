use axum::{extract::State, http::StatusCode, routing::get, Json, Router};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tokio::time::Instant;

// Импортируем глобальный счетчик отброшенных логов
use crate::utils::logger::DROPPED_LOGS;

// Импортируем хендлер метрик Prometheus
use crate::monitoring::prometheus::metrics_handler;

/// Структура для потокобезопасного хранения состояния бота (Lock-free)
pub struct SharedState {
    pub last_update: AtomicU64,      // Unix ms
    pub last_heartbeat: AtomicU64,   // Unix ms (Hot Path)
    pub ws_connected: AtomicBool,
    pub emergency_mode: AtomicBool,
    pub start_time: Instant,
}

/// Структура для JSON-ответа health-check эндпоинта
#[derive(serde::Serialize)]
pub struct HealthStatus {
    pub status: String,
    pub uptime_sec: u64,
    pub last_update_ms_ago: u64,
    pub ws_connected: bool,
    pub dropped_logs: u64,
    pub rss_mb: u64,
    pub cpu_usage: f32,
}

/// Запускает HTTP-сервер для health-check эндпоинта
pub async fn start_health_server(config: HealthConfig, state: Arc<SharedState>, max_memory_mb: u64) {
    let app = Router::new()
        .route("/health", get(move |State(s): State<Arc<SharedState>>| health_handler(s, max_memory_mb)))
        .route("/metrics", get(metrics_handler))
        .with_state(state);

    let addr: std::net::SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .expect("Invalid health check address");

    tracing::info!("Health check server starting on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

/// Хендлер для /health эндпоинта
async fn health_handler(
    state: Arc<SharedState>,
    max_memory_mb: u64,
) -> (StatusCode, Json<HealthStatus>) {
    let now = crate::utils::helpers::unix_ms();
    let last = state.last_update.load(Ordering::Relaxed);
    let diff = now.saturating_sub(last);
    
    let is_connected = state.ws_connected.load(Ordering::Relaxed);
    let is_emergency = state.emergency_mode.load(Ordering::Relaxed);
    
    // Читаем количество отброшенных логов
    let dropped = DROPPED_LOGS.load(Ordering::Relaxed);
    
    // Читаем метрики ресурсов
    use crate::utils::sys::METRICS;
    let rss_bytes = METRICS.rss_bytes.load(Ordering::Relaxed);
    let cpu_usage = f32::from_bits(METRICS.cpu_usage.load(Ordering::Relaxed));
    let rss_mb = rss_bytes / 1024 / 1024;
    
    // Условие падения: нет данных > 30с или потерян WS
    let is_down = diff > 30_000 || !is_connected || is_emergency;
    
    // Условие деградации: превышение лимита памяти
    let is_degraded = rss_mb > max_memory_mb;
    
    let status_str = if is_down { 
        "down".to_string() 
    } else if is_degraded {
        "degraded".to_string()
    } else { 
        "up".to_string() 
    };

    let status = HealthStatus {
        status: status_str,
        uptime_sec: state.start_time.elapsed().as_secs(),
        last_update_ms_ago: diff,
        ws_connected: is_connected,
        dropped_logs: dropped,
        rss_mb,
        cpu_usage,
    };

    let code = if is_down { 
        StatusCode::SERVICE_UNAVAILABLE 
    } else if is_degraded {
        StatusCode::OK 
    } else { 
        StatusCode::OK 
    };
    (code, Json(status))
}

/// Конфигурация для health-check сервера
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct HealthConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default = "default_timeout_sec")]
    pub timeout_sec: u64,
    #[serde(default = "default_max_memory_mb")]
    pub max_memory_mb: u64,
    #[serde(default = "default_watchdog")]
    pub watchdog: WatchdogConfig,
    #[serde(default = "default_metrics_port")]
    pub metrics_port: u16,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct WatchdogConfig {
    #[serde(default = "default_stall_timeout_ms")]
    pub stall_timeout_ms: u64,
    #[serde(default = "default_check_interval_ms")]
    pub check_interval_ms: u64,
    #[serde(default = "default_suspend_grace_ms")]
    pub suspend_grace_ms: u64,
}

fn default_host() -> String {
    "127.0.0.1".to_string()
}

fn default_port() -> u16 {
    8080
}

fn default_timeout_sec() -> u64 {
    30
}

fn default_max_memory_mb() -> u64 {
    2048
}

fn default_metrics_port() -> u16 {
    9090
}

fn default_stall_timeout_ms() -> u64 {
    5000
}

fn default_check_interval_ms() -> u64 {
    2000
}

fn default_suspend_grace_ms() -> u64 {
    60000
}

fn default_watchdog() -> WatchdogConfig {
    WatchdogConfig::default()
}

impl Default for WatchdogConfig {
    fn default() -> Self {
        Self {
            stall_timeout_ms: default_stall_timeout_ms(),
            check_interval_ms: default_check_interval_ms(),
            suspend_grace_ms: default_suspend_grace_ms(),
        }
    }
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            timeout_sec: default_timeout_sec(),
            max_memory_mb: default_max_memory_mb(),
            watchdog: default_watchdog(),
            metrics_port: default_metrics_port(),
        }
    }
}
