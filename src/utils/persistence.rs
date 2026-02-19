// Задача 190: Graceful Restart - сохранение и восстановление состояния бота
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;
use tracing::{debug, info, warn};

use crate::trading::position_manager::Position;
use crate::trading::types::OrderSide;
use rust_decimal::Decimal;

/// Намерение/ордер для сохранения состояния
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderIntent {
    pub side: OrderSide,
    pub price: f64,
    pub qty: f64,
    pub timestamp: u64,
    pub filled_qty: f64,
}

/// Состояние бота для сохранения при graceful shutdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BotState {
    /// Текущая открытая позиция (если есть)
    pub position: Option<PositionSnapshot>,
    /// Список активных намерений/ордеров
    pub active_orders: Vec<(String, OrderIntent)>, // (link_id, intent)
    /// Время сохранения (UNIX timestamp в миллисекундах)
    pub timestamp_ms: u64,
}

/// Снимок позиции для сериализации
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionSnapshot {
    pub symbol: String,
    pub qty: Decimal,
    pub avg_price: Decimal,
    pub realized_pnl: Decimal,
    pub unrealized_pnl: Decimal,
    pub unrealized_pnl_pct: Decimal,
    pub mark_pnl: Decimal,
    pub leverage: Decimal,
    pub updated_at: u64,
    pub opened_at: Option<u64>,
    pub completed_tp_stages: HashSet<usize>,
    pub initial_size: f64,
    pub side: OrderSide,
    pub extreme_water_mark: f64,
    pub current_stop_loss: f64,
    pub tsl_active: bool,
    pub accumulated_funding: Decimal,
}

impl From<&Position> for PositionSnapshot {
    fn from(pos: &Position) -> Self {
        Self {
            symbol: pos.symbol.clone(),
            qty: pos.qty,
            avg_price: pos.avg_price,
            realized_pnl: pos.realized_pnl,
            unrealized_pnl: pos.unrealized_pnl,
            unrealized_pnl_pct: pos.unrealized_pnl_pct,
            mark_pnl: pos.mark_pnl,
            leverage: pos.leverage,
            updated_at: pos.updated_at,
            opened_at: pos.opened_at,
            completed_tp_stages: pos.completed_tp_stages.clone(),
            initial_size: pos.initial_size,
            side: pos.side,
            extreme_water_mark: pos.extreme_water_mark,
            current_stop_loss: pos.current_stop_loss,
            tsl_active: pos.tsl_active,
            accumulated_funding: pos.accumulated_funding,
        }
    }
}

/// Сохранить состояние бота в файл
pub fn save_state<P: AsRef<Path>>(path: P, state: &BotState) -> Result<()> {
    let path = path.as_ref();
    
    // Создать директорию если не существует
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory: {:?}", parent))?;
    }
    
    // Сериализовать в JSON
    let json = serde_json::to_string_pretty(state)
        .context("Failed to serialize BotState")?;
    
    // Записать в файл
    std::fs::write(path, json)
        .with_context(|| format!("Failed to write state to {:?}", path))?;
    
    info!("[Persistence] State saved to {:?}", path);
    debug!("[Persistence] Saved state: position={}, orders={}", 
        state.position.is_some(), 
        state.active_orders.len()
    );
    
    Ok(())
}

/// Загрузить состояние бота из файла
pub fn load_state<P: AsRef<Path>>(path: P) -> Result<BotState> {
    let path = path.as_ref();
    
    // Проверить существование файла
    if !path.exists() {
        anyhow::bail!("State file does not exist: {:?}", path);
    }
    
    // Прочитать файл
    let json = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read state from {:?}", path))?;
    
    // Десериализовать из JSON
    let state: BotState = serde_json::from_str(&json)
        .context("Failed to deserialize BotState")?;
    
    let age_ms = crate::utils::time::timestamp_ms() - state.timestamp_ms;
    info!("[Persistence] State restored (age: {} ms)", age_ms);
    debug!("[Persistence] Loaded state: position={}, orders={}", 
        state.position.is_some(), 
        state.active_orders.len()
    );
    
    Ok(state)
}

/// Удалить файл состояния
pub fn delete_state<P: AsRef<Path>>(path: P) -> Result<()> {
    let path = path.as_ref();
    
    if path.exists() {
        std::fs::remove_file(path)
            .with_context(|| format!("Failed to delete state file: {:?}", path))?;
        debug!("[Persistence] State file deleted: {:?}", path);
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_save_and_load_state() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("state.json");
        
        let state = BotState {
            position: None,
            active_orders: vec![],
            timestamp_ms: 1234567890,
        };
        
        // Сохранить
        save_state(&path, &state).unwrap();
        assert!(path.exists());
        
        // Загрузить
        let loaded = load_state(&path).unwrap();
        assert_eq!(loaded.timestamp_ms, state.timestamp_ms);
        assert_eq!(loaded.active_orders.len(), 0);
        assert!(loaded.position.is_none());
    }
    
    #[test]
    fn test_load_nonexistent_file() {
        let result = load_state("/nonexistent/path/state.json");
        assert!(result.is_err());
    }
}
