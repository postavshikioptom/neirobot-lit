use serde::{Deserialize, Serialize};
use rust_decimal::Decimal;
use std::str::FromStr;
use anyhow::{anyhow, Result};
use std::collections::HashMap;

// ============================================================================
// State Machine Types (Задача 136)
// ============================================================================

/// Состояния жизненного цикла ордера (State Machine v2.0)
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum OrderState {
    Created,        // Локально создан
    PendingNew,     // Отправлен запрос на создание
    Active,         // Подтвержден биржей
    PartiallyFilled, // Есть частичные исполнения
    Filled,         // Полностью исполнен
    PendingCancel,  // Отправлен запрос на отмену
    Cancelled,      // Отменен
    Rejected(String), // Отклонен (причина)
    Expired,        // Просрочен (Time-in-force)
}

/// События жизненного цикла ордера
#[derive(Debug)]
pub enum OrderEvent {
    Accepted { order_id: String },
    Trade { exec_qty: f64, price: f64 },
    CancelAck,
    Rejected { reason: String },
    Expired,
}

// ============================================================================
// Lot Filter Types (Задача 137)
// ============================================================================

/// Фильтры лота для валидации объемов ордеров
/// 
/// Используется для проверки и округления объемов согласно правилам биржи Bybit.
/// Все значения в f64 для совместимости с API биржи.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LotFilter {
    pub min_qty: f64,   // Минимальный объем ордера
    pub max_qty: f64,   // Максимальный объем ордера
    pub qty_step: f64,  // Шаг объема (например, 0.01)
}

impl LotFilter {
    /// Конвертация из MarketInfo (Decimal) в LotFilter (f64)
    pub fn from_market_info(info: &MarketInfo) -> Self {
        Self {
            min_qty: info.min_order_qty.to_f64().unwrap_or(0.01),
            max_qty: info.max_order_qty.to_f64().unwrap_or(1000000.0),
            qty_step: info.qty_step.to_f64().unwrap_or(0.01),
        }
    }
}

// ============================================================================
// Symbol Info Types (Задача 138)
// ============================================================================

/// Фильтр цены для валидации и форматирования цен ордеров
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PriceFilter {
    pub tick_size: f64,         // Шаг цены (например, 0.1)
    pub price_precision: usize, // Количество знаков после запятой для форматирования
    pub min_price: f64,         // Минимальная цена
    pub max_price: f64,         // Максимальная цена
}

/// Полная информация о символе для динамической загрузки (Задача 138)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SymbolInfo {
    pub lot_filter: LotFilter,
    pub price_filter: PriceFilter,
    pub max_leverage: f64,
}

// ============================================================================
// Legacy Types (для обратной совместимости)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderSide { Buy, Sell }

impl FromStr for OrderSide {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "buy" => Ok(OrderSide::Buy),
            "sell" => Ok(OrderSide::Sell),
            _ => Err(anyhow!("Invalid side: {}", s)),
        }
    }
}

impl std::fmt::Display for OrderSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrderSide::Buy => write!(f, "Buy"),
            OrderSide::Sell => write!(f, "Sell"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderStatus {
    Created,          // Локальный статус: подготовка к отправке
    New,              // Принят биржей, активен в стакане
    PartiallyFilled,  // Частично исполнен (не финальный)
    Filled,           // Полностью исполнен (финальный)
    Cancelled,        // Отменен (финальный)
    Rejected,         // Отклонен при создании (финальный)
    Expired,          // Истек (например, PostOnly-отклонение или IOC) (финальный)
    PostOnlyRejected, // Специфический статус для PostOnly отклонения
    Untracked,        // Ордер, статус которого временно неизвестен (задача 068)
}

impl OrderStatus {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Filled | Self::Cancelled | Self::Rejected | Self::Expired | Self::PostOnlyRejected)
    }

    pub fn from_bybit_status(status: &str) -> Self {
        match status {
            "New" => Self::New,
            "PartiallyFilled" => Self::PartiallyFilled,
            "Filled" => Self::Filled,
            "Cancelled" | "PartiallyFilledCanceled" => Self::Cancelled,
            "Rejected" => Self::Rejected,
            "Expired" | "Deactivated" => Self::Expired,
            _ => Self::New, // Fallback
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MarketInfo {
    pub qty_step: Decimal,       // Шаг объема (например, 0.01)
    pub min_order_qty: Decimal,  // Минимальный лот
    pub max_order_qty: Decimal,  // Максимальный лот
    pub tick_size: Decimal,      // Шаг цены (для будущего)
}

// ============================================================================
// Legacy Types (для обратной совместимости - будет удалено)
// ============================================================================

/// Состояние для управления рисками (используется в RiskManager)
/// Отдельно от BotState в utils::persistence (задача 190)
#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct RiskState {
    pub symbol: String,
    pub position_size: Decimal,
    pub avg_price: Decimal,
    pub cumulative_pnl: Decimal,
    pub active_orders: std::collections::HashMap<String, RiskOrderInfo>, // link_id -> info
    pub day_start_pnl: Decimal,         // Накопленный PnL на начало текущих суток
    pub last_pnl_reset_ts: i64,         // Таймстемп последнего сброса (ms)
    pub recent_trade_timestamps: Vec<i64>, // Для сохранения на диск (Задача 112)
    pub last_update_ts: i64,
    pub loss_streak: usize,             // Текущая серия убыточных сделок подряд (Задача 115)
    pub last_loss_timestamp_ms: i64,    // Unix MS последнего убытка (Задача 118)
    // Задача 149: Поля для механизма нарезки крупных ордеров
    pub pending_slice_qty: Option<Decimal>,     // Оставшийся объем для нарезки
    pub pending_slice_side: Option<OrderSide>,  // Сторона нарезки
    pub pending_slice_signal: Option<crate::ml::types::Signal>,   // Сигнал для нарезки (сохраняется только Signal, не Instant)
    pub pending_slice_probs: Option<[f32; 3]>,  // Вероятности для нарезки
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RiskOrderInfo {
    pub side: OrderSide,
    pub price: Decimal,
    pub qty: Decimal,
    pub state: OrderState,
    pub link_id: Option<String>,
}

/// Информация об ордере для REST API (упрощенная версия)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderInfo {
    pub side: OrderSide,
    pub price: Decimal,
    pub qty: Decimal,
    pub status: OrderStatus,
    pub chase_count: usize,
    pub last_chase_ts: i64,
    pub link_id: Option<String>,
}

/// Type alias для обратной совместимости
pub type BotState = RiskState;

/// @deprecated Используйте новую Order из src/trading/order.rs
/// Эта структура переименована в LegacyOrder и будет удалена после полной миграции.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyOrder {
    pub client_oid: String,
    pub order_id: Option<String>,
    pub symbol: String,
    pub side: OrderSide,
    pub price: Decimal,
    pub qty: Decimal,
    pub status: OrderStatus,
    pub cum_exec_qty: Decimal,
    pub post_only_retry_count: u32,
    pub chase_count: usize,
    pub last_chase_ts: i64,
    pub created_at: u64,
    pub updated_at: u64,
}

/// Type alias для обратной совместимости
/// @deprecated Используйте crate::trading::order::Order
pub type Order = LegacyOrder;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderUpdate {
    pub order_link_id: String,
    pub order_id: String,
    pub status: OrderStatus,
    pub cum_exec_qty: Decimal,
    pub exec_price: Option<Decimal>,
    pub exec_fee: Option<Decimal>,
    pub is_maker: Option<bool>,
    pub reason: Option<String>,
    pub timestamp: u64,
    pub new_price: Option<Decimal>,  // Новая цена после amendment
    pub new_qty: Option<Decimal>,    // Новый объем после amendment
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateOrderRequest {
    pub category: String,       // "linear" для USDT Perpetuals
    pub symbol: String,         // Например, "BTCUSDT"
    pub side: String,           // "Buy" или "Sell"
    pub order_type: String,     // "Limit"
    pub qty: String,            // Decimal as string
    pub price: Option<String>,  // Decimal as string (Option for Market orders)
    pub time_in_force: String,  // "GTC" или "PostOnly"
    pub order_link_id: String,  // Наш уникальный ID
    pub position_idx: i32,      // 0 для One-Way mode, 1 для Long (Hedge), 2 для Short (Hedge)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reduce_only: Option<bool>,
    // Задача 167: Параметры для динамического скользящего стоп-лосса (Exchange-side)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trailing_stop: Option<String>,  // Дистанция трейлинга в базисных пунктах
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_price: Option<String>,   // Активная цена для начала трейлинга
    // Задача 232: Self-Match Prevention (SMP)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub smp_type: Option<String>,       // "None", "CancelMaker", "CancelTaker", "CancelBoth"
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BybitOrderResult {
    pub order_id: String,
    pub order_link_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillEvent {
    pub symbol: String,
    pub side: OrderSide,
    pub exec_qty: Decimal,
    pub exec_price: Decimal,
    pub exec_fee: Decimal,
    pub is_maker: bool,
    pub exec_id: String,
    pub order_id: String,
    pub order_link_id: Option<String>,
    pub timestamp: u64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CancelOrderRequest {
    pub category: String,
    pub symbol: String,
    pub order_link_id: Option<String>, // Отмена по нашему ID
    pub order_id: Option<String>,      // ИЛИ по ID биржи
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CancelAllOrdersRequest {
    pub category: String,
    pub symbol: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AmendOrderRequest {
    pub category: String,
    pub symbol: String,
    pub order_link_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub price: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub qty: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trigger_price: Option<String>,
}

/// Type alias для совместимости с документацией задачи
pub type AmendParams = AmendOrderRequest;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AmendOrderResult {
    pub order_id: String,
    pub order_link_id: String,
}

/// Type alias для совместимости с документацией задачи
pub type AmendedResponse = AmendOrderResult;



// ============================================================================
// Функции save_state/load_state перенесены в state.rs (Задача 107)
// ============================================================================
