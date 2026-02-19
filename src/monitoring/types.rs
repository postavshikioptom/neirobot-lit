use serde::{Deserialize, Serialize};

/// Обновление финансового состояния бота для потоковой трансляции
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquityUpdate {
    /// Временная метка обновления (Unix timestamp в миллисекундах)
    pub timestamp: i64,
    
    /// Общий капитал (баланс + нереализованный PnL)
    pub total_equity: f64,
    
    /// Нереализованная прибыль/убыток (с учетом Taker fee на закрытие)
    pub unrealized_pnl: f64,
    
    /// Реализованная прибыль/убыток за день
    pub realized_pnl_day: f64,
    
    /// Размер текущей позиции
    pub position_size: f64,
}

impl EquityUpdate {
    /// Создает новое обновление equity
    pub fn new(
        timestamp: i64,
        total_equity: f64,
        unrealized_pnl: f64,
        realized_pnl_day: f64,
        position_size: f64,
    ) -> Self {
        Self {
            timestamp,
            total_equity,
            unrealized_pnl,
            realized_pnl_day,
            position_size,
        }
    }
}
