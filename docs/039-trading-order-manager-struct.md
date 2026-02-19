# 039 - Trading Order Manager Struct
Цель задачи: Реализовать структуру OrderManager для управления жизненным циклом ордеров. Менеджер должен отслеживать активные заявки, сопоставлять внутренние client_oid с биржевыми order_id, обновлять статусы и вести историю завершенных операций.

Файлы:

src/trading/mod.rs (создать)
src/trading/types.rs (создать)
src/trading/order_manager.rs (создать)
Инструкции для Gemini:

src/trading/types.rs:
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderSide { Buy, Sell }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderStatus {
    New,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    PendingCancel,
}

impl OrderStatus {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Filled | Self::Cancelled | Self::Rejected)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub client_oid: String,
    pub order_id: Option<String>, // ID от биржи
    pub symbol: String,
    pub side: OrderSide,
    pub price: f64,
    pub qty: f64,
    pub status: OrderStatus,
    pub filled_qty: f64,
    pub created_at: u64,
    pub updated_at: u64,
}
src/trading/order_manager.rs:
use std::collections::HashMap;
use anyhow::{Result, bail};
use crate::trading::types::{Order, OrderStatus};
use crate::utils::timestamp_ms;

pub struct OrderManager {
    active_orders: HashMap<String, Order>,
    exchange_map: HashMap<String, String>, // order_id -> client_oid
    history: Vec<Order>,
}

impl OrderManager {
    pub fn new() -> Self {
        Self {
            active_orders: HashMap::new(),
            exchange_map: HashMap::new(),
            history: Vec::new(),
        }
    }

    pub fn add_order(&mut self, order: Order) -> Result<()> {
        if self.active_orders.contains_key(&order.client_oid) {
            bail!("Duplicate client_oid: {}", order.client_oid);
        }
        self.active_orders.insert(order.client_oid.clone(), order);
        Ok(())
    }

    pub fn update_order(&mut self, client_oid: &str, order_id: Option<String>, status: OrderStatus, filled: f64) {
        if let Some(mut order) = self.active_orders.remove(client_oid) {
            order.status = status;
            order.filled_qty = filled;
            order.updated_at = timestamp_ms();
            
            if let Some(id) = order_id {
                self.exchange_map.insert(id.clone(), client_oid.to_string());
                order.order_id = Some(id);
            }

            if order.status.is_terminal() {
                if let Some(ref id) = order.order_id {
                    self.exchange_map.remove(id);
                }
                self.history.push(order);
            } else {
                self.active_orders.insert(client_oid.to_string(), order);
            }
        }
    }

    pub fn get_by_client_id(&self, client_oid: &str) -> Option<&Order> {
        self.active_orders.get(client_oid)
    }

    pub fn get_by_exchange_id(&self, order_id: &str) -> Option<&Order> {
        let client_oid = self.exchange_map.get(order_id)?;
        self.active_orders.get(client_oid)
    }
}
src/trading/mod.rs:
pub mod types;
pub mod order_manager;
pub use types::*;
pub use order_manager::*;
Технические требования:

Поля времени: Добавить updated_at и обновлять его через crate::utils::timestamp_ms() при каждом изменении.
Маппинг: Использовать exchange_map для быстрого поиска по биржевому order_id.
Изоляция: При переходе в терминальный статус (is_terminal) ордер обязательно удаляется из active_orders и exchange_map, и перемещается в history.
Безопасность: add_order должен возвращать ошибку при попытке добавить ордер с существующим client_oid.