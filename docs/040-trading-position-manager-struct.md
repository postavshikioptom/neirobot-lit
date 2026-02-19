# 040 - Trading Position Manager Struct
Цель задачи: Реализовать структуру PositionManager для отслеживания текущей позиции по инструменту. Менеджер должен рассчитывать среднюю цену входа (WAP) при увеличении позиции и фиксировать реализованный PnL при частичном или полном закрытии, учитывая направление сделки (Side).

Файлы:

src/trading/position_manager.rs (создать)
src/trading/mod.rs (обновить)
Инструкции для Gemini:

src/trading/position_manager.rs: Реализовать учет позиции с поддержкой переворотов (reverse) и расчетом средней цены.
use crate::trading::types::OrderSide;
use crate::utils::timestamp_ms;
use tracing::info;

#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    pub qty: f64,          // Положительная для Long, отрицательная для Short
    pub entry_price: f64,  // Средневзвешенная цена входа (WAP)
    pub realized_pnl: f64,
    pub updated_at: u64,
}

pub struct PositionManager {
    position: Position,
}

impl PositionManager {
    pub fn new(symbol: String) -> Self {
        Self {
            position: Position {
                symbol,
                qty: 0.0,
                entry_price: 0.0,
                realized_pnl: 0.0,
                updated_at: timestamp_ms(),
            },
        }
    }

    /// Обновление позиции на основе исполненной сделки
    pub fn update_from_fill(&mut self, side: OrderSide, fill_qty: f64, fill_price: f64) {
        let signed_fill = if side == OrderSide::Buy { fill_qty } else { -fill_qty };
        let old_qty = self.position.qty;
        let new_qty = old_qty + signed_fill;

        if old_qty == 0.0 {
            // Открытие позиции с нуля
            self.position.entry_price = fill_price;
        } else if old_qty.signum() == signed_fill.signum() {
            // Увеличение существующей позиции (пирамидинг)
            let total_qty_abs = new_qty.abs();
            self.position.entry_price = (old_qty.abs() * self.position.entry_price + fill_qty * fill_price) / total_qty_abs;
        } else {
            // Уменьшение позиции или переворот (side противоположный текущему)
            let closed_qty_abs = old_qty.abs().min(fill_qty);
            let pnl = closed_qty_abs * (fill_price - self.position.entry_price) * old_qty.signum();
            self.position.realized_pnl += pnl;

            if new_qty == 0.0 {
                // Полное закрытие
                self.position.entry_price = 0.0;
            } else if old_qty.signum() != new_qty.signum() {
                // Полный переворот (reverse): остаток сделки открывает новую позицию
                self.position.entry_price = fill_price;
            }
            // При частичном закрытии в том же направлении entry_price не меняется
        }

        self.position.qty = new_qty;
        self.position.updated_at = timestamp_ms();
        
        info!(
            "[{}] Position updated: qty={}, entry_price={:.4}, realized_pnl={:.4}", 
            self.position.symbol, self.position.qty, self.position.entry_price, self.position.realized_pnl
        );
    }

    pub fn calculate_unrealized_pnl(&self, current_price: f64) -> f64 {
        if self.position.qty == 0.0 { return 0.0; }
        self.position.qty * (current_price - self.position.entry_price)
    }

    pub fn get_position(&self) -> &Position {
        &self.position
    }
}
src/trading/mod.rs:
pub mod types;
pub mod order_manager;
pub mod position_manager;

pub use types::*;
pub use order_manager::*;
pub use position_manager::*;
Технические требования:

Аргументы: update_from_fill должен принимать OrderSide и fill_qty (положительное число).
Логика WAP: При old_qty == 0 цена входа устанавливается равной цене сделки.
PNL: Расчет реализованного профита только для той части объема, которая закрывает позицию.
Reverse: При переходе из Long в Short (или наоборот) entry_price оставшейся части позиции должна стать ценой последней сделки.
Защита: Всегда обновлять updated_at.