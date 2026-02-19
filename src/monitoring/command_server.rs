//! Командный сервер для удаленного управления ботом
//! 
//! Предоставляет HTTP API для контроля жизненного цикла бота:
//! - GET /status - получение текущего статуса бота
//! - POST /panic - экстренная остановка торговли
//! - POST /reload - перезагрузка конфигурации

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{info, error};

use crate::config::types::BotConfig;

/// Команды управления ботом
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Command {
    /// Экстренная остановка торговли
    Panic,
    /// Приостановка торговли (без закрытия позиций)
    Pause,
    /// Перезагрузка конфигурации
    Reload,
    /// Запрос статуса (для внутреннего использования)
    GetStatus,
}

/// Ответ на запрос статуса
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusResponse {
    pub symbol: String,
    pub status: String,
    pub pnl: Option<f64>,
    pub position: Option<f64>,
    pub latency_ms: Option<f64>,
    pub uptime_secs: u64,
}

/// Общий ответ для команд
#[derive(Debug, Serialize, Deserialize)]
pub struct CommandResponse {
    pub success: bool,
    pub message: String,
}

/// Shared state для handlers
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<BotConfig>,
    pub command_tx: mpsc::Sender<Command>,
    pub status: Arc<parking_lot::RwLock<StatusResponse>>,
}

/// Handler для GET /status
async fn status_handler(State(state): State<AppState>) -> impl IntoResponse {
    let status = state.status.read().clone();
    Json(status)
}

/// Handler для POST /panic
async fn panic_handler(State(state): State<AppState>) -> impl IntoResponse {
    info!("[CommandServer] Received PANIC command");
    
    match state.command_tx.send(Command::Panic).await {
        Ok(_) => {
            let response = CommandResponse {
                success: true,
                message: "Panic command sent successfully".to_string(),
            };
            (StatusCode::OK, Json(response))
        }
        Err(e) => {
            error!("[CommandServer] Failed to send panic command: {}", e);
            let response = CommandResponse {
                success: false,
                message: format!("Failed to send panic command: {}", e),
            };
            (StatusCode::INTERNAL_SERVER_ERROR, Json(response))
        }
    }
}

/// Handler для POST /reload
async fn reload_handler(State(state): State<AppState>) -> impl IntoResponse {
    info!("[CommandServer] Received RELOAD command");
    
    match state.command_tx.send(Command::Reload).await {
        Ok(_) => {
            let response = CommandResponse {
                success: true,
                message: "Reload command sent successfully".to_string(),
            };
            (StatusCode::OK, Json(response))
        }
        Err(e) => {
            error!("[CommandServer] Failed to send reload command: {}", e);
            let response = CommandResponse {
                success: false,
                message: format!("Failed to send reload command: {}", e),
            };
            (StatusCode::INTERNAL_SERVER_ERROR, Json(response))
        }
    }
}

/// Handler для POST /pause
async fn pause_handler(State(state): State<AppState>) -> impl IntoResponse {
    info!("[CommandServer] Received PAUSE command");
    
    match state.command_tx.send(Command::Pause).await {
        Ok(_) => {
            let response = CommandResponse {
                success: true,
                message: "Pause command sent successfully".to_string(),
            };
            (StatusCode::OK, Json(response))
        }
        Err(e) => {
            error!("[CommandServer] Failed to send pause command: {}", e);
            let response = CommandResponse {
                success: false,
                message: format!("Failed to send pause command: {}", e),
            };
            (StatusCode::INTERNAL_SERVER_ERROR, Json(response))
        }
    }
}

/// Запуск командного сервера
/// 
/// # Arguments
/// * `config` - Конфигурация бота
/// * `command_tx` - Канал для отправки команд в основной цикл бота
/// * `status` - Shared state со статусом бота
/// * `port` - Порт для HTTP сервера (по умолчанию 9001)
pub async fn start_command_server(
    config: Arc<BotConfig>,
    command_tx: mpsc::Sender<Command>,
    status: Arc<parking_lot::RwLock<StatusResponse>>,
    port: u16,
) -> Result<(), Box<dyn Error>> {
    let state = AppState {
        config: config.clone(),
        command_tx,
        status,
    };

    let app = Router::new()
        .route("/status", get(status_handler))
        .route("/panic", post(panic_handler))
        .route("/pause", post(pause_handler))
        .route("/reload", post(reload_handler))
        .with_state(state);

    // Bind только к localhost для безопасности
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    
    info!("[CommandServer] Starting on {}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await?;
    
    axum::serve(listener, app)
        .await
        .map_err(|e| Box::new(e) as Box<dyn Error>)?;

    Ok(())
}
