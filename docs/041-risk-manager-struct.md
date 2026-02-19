# 041 - Risk Manager Struct
Цель задачи: Реализовать структуру RiskManager, выполняющую роль финального фильтра (gatekeeper) перед отправкой торговых команд. Менеджер должен проверять лимиты на размер позиции, количество активных ордеров и текущую просадку (Drawdown) от пикового эквити, используя параметры из RiskConfig.

Файлы:

src/trading/risk_manager.rs (создать)
src/trading/mod.rs (обновить)
Инструкции для Gemini:

src/trading/risk_manager.rs: Реализовать валидацию ордеров и мониторинг просадки.
use crate::config::types::RiskConfig;
use crate::trading::types::OrderSide;
use crate::trading::position_manager::Position;
use anyhow::{Result, bail};

pub struct RiskManager {
    config: RiskConfig,
    peak_equity: f64,          // Пиковое значение эквити для расчета Drawdown
    initial_equity: f64,       // Начальный капитал (задается при старте)
}

impl RiskManager {
    pub fn new(config: RiskConfig, initial_equity: f64) -> Self {
        Self { 
            config, 
            peak_equity: initial_equity,
            initial_equity,
        }
    }

    /// Проверка параметров ордера перед отправкой в API
    pub fn validate_order(
        &self, 
        side: OrderSide, 
        qty: f64, 
        current_pos: &Position,
        active_orders_count: usize
    ) -> Result<()> {
        // 1. Лимит активных ордеров (из конфига)
        if let Some(max_orders) = self.config.max_open_orders {
            if active_orders_count >= max_orders {
                bail!("Risk: Max open orders limit reached ({})", max_orders);
            }
        }

        // 2. Лимит размера позиции (Max Position Size)
        if let Some(max_size) = self.config.max_position_size {
            let signed_qty = if side == OrderSide::Buy { qty } else { -qty };
            let projected_qty = current_pos.qty + signed_qty;
            
            if projected_qty.abs() > max_size {
                bail!(
                    "Risk: Order violates max_position_size. Projected: {}, Limit: {}", 
                    projected_qty.abs(), max_size
                );
            }
        }

        // 3. Заглушки для будущих проверок (задачи 071-075)
        // TODO: Spread filter (073)
        // TODO: Price deviation check (075)

        Ok(())
    }

    /// Обновление и проверка просадки (Drawdown)
    /// Должен вызываться при обновлении рыночной цены или баланса
    pub fn update_and_check_risk(&mut self, current_unrealized_pnl: f64, current_realized_pnl: f64) -> Result<()> {
        let current_equity = self.initial_equity + current_realized_pnl + current_unrealized_pnl;
        
        // Обновляем пиковое значение
        if current_equity > self.peak_equity {
            self.peak_equity = current_equity;
        }

        // Проверка Max Drawdown % (если задан в конфиге как доля, например 0.05 для 5%)
        if let Some(max_dd_pct) = self.config.max_drawdown_pct {
            if self.peak_equity > 0.0 {
                let current_dd_pct = (self.peak_equity - current_equity) / self.peak_equity;
                if current_dd_pct > max_dd_pct {
                    bail!(
                        "Risk: Max drawdown exceeded. Current: {:.2}%, Limit: {:.2}%", 
                        current_dd_pct * 100.0, max_dd_pct * 100.0
                    );
                }
            }
        }

        Ok(())
    }
}
src/trading/mod.rs:
pub mod types;
pub mod order_manager;
pub mod position_manager;
pub mod risk_manager;

pub use types::*;
pub use order_manager::*;
pub use position_manager::*;
pub use risk_manager::*;
Технические требования:

Drawdown: Расчет должен идти от peak_equity. Эквити = initial_equity + realized_pnl + unrealized_pnl.
Единицы: max_drawdown_pct в конфиге обрабатывается как десятичная дробь (0.05 = 5%).
Конфигурация: Значение max_open_orders брать из RiskConfig (если отсутствует — не ограничивать).
Связь: Метод validate_order принимает &Position для доступа к актуальному объему без лишнего копирования.