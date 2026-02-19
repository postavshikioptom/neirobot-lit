# 043 - Trading Execution Skeleton
Цель задачи: Создать «мозг» торгового модуля — структуру ExecutionEngine. Этот компонент объединяет OrderManager, PositionManager и RiskManager, превращая высокоуровневые сигналы от нейросети (InferenceOutput) в конкретные торговые действия (открытие, закрытие или переворот позиции) с учетом всех фильтров риска.

Файлы:

src/trading/execution.rs (создать)
src/trading/mod.rs (обновить)
Инструкции для Gemini:

src/trading/execution.rs: Реализовать скелет исполнителя, управляющий логикой входа/выхода на основе InferenceOutput.
use crate::ml::types::{Signal, InferenceOutput};
use crate::trading::order_manager::OrderManager;
use crate::trading::position_manager::PositionManager;
use crate::trading::risk_manager::RiskManager;
use crate::trading::types::{OrderSide, Order};
use anyhow::{Result, info};

pub struct ExecutionEngine {
    pub order_manager: OrderManager,
    pub position_manager: PositionManager,
    pub risk_manager: RiskManager,
    pub symbol: String,
    pub close_on_flat: bool, // Флаг из конфига: закрывать ли позицию при нейтральном сигнале
}

impl ExecutionEngine {
    pub fn new(symbol: String, risk_manager: RiskManager, close_on_flat: bool) -> Self {
        Self {
            order_manager: OrderManager::new(),
            position_manager: PositionManager::new(symbol.clone()),
            risk_manager,
            symbol,
            close_on_flat,
        }
    }

    /// Обработка результата инференса
    pub fn handle_inference(&mut self, output: InferenceOutput, current_price: f64) -> Result<()> {
        let position = self.position_manager.get_position();
        let unrealized = self.position_manager.calculate_unrealized_pnl(current_price);
        let current_pnl = position.realized_pnl + unrealized;

        // 1. Проверка глобального риска (Drawdown) перед любыми действиями
        self.risk_manager.check_global_risk(current_pnl)?;

        // 2. Логика исполнения на основе сигнала
        match output.signal {
            Signal::Up => {
                if position.qty <= 0.0 {
                    // Если мы в кэше или в шорте — покупаем (переворот или вход)
                    self.execute_trade(OrderSide::Buy, current_price)?;
                }
            }
            Signal::Down => {
                if position.qty >= 0.0 {
                    // Если мы в кэше или в лонге — продаем (переворот или вход)
                    self.execute_trade(OrderSide::Sell, current_price)?;
                }
            }
            Signal::Flat => {
                if self.close_on_flat && position.qty != 0.0 {
                    // Закрытие текущей позиции при сигнале Flat
                    let side = if position.qty > 0.0 { OrderSide::Sell } else { OrderSide::Buy };
                    info!("Execution: Signal is Flat, closing position for {}", self.symbol);
                    self.execute_trade(side, current_price)?;
                }
            }
        }

        Ok(())
    }

    /// Вспомогательный метод для валидации риска и подготовки ордера
    fn execute_trade(&mut self, side: OrderSide, price: f64) -> Result<()> {
        let position = self.position_manager.get_position();
        let unrealized = self.position_manager.calculate_unrealized_pnl(price);
        let current_pnl = position.realized_pnl + unrealized;
        
        let qty = 0.01; // Заглушка: расчет объема будет в задаче 044
        let active_orders = self.order_manager.get_active_count();

        // Проверка через Risk Gate (из задачи 042)
        self.risk_manager.check_order_gate(
            side,
            qty,
            price,
            position,
            active_orders,
            current_pnl,
        )?;

        info!("Execution: Placing {:?} order for {} at price {}", side, self.symbol, price);
        
        // TODO: Генерация client_oid и отправка в REST API (Phase 6)
        
        Ok(())
    }
}
src/trading/mod.rs:
pub mod types;
pub mod order_manager;
pub mod position_manager;
pub mod risk_manager;
pub mod execution;

pub use execution::*;
Технические требования:

Интерфейс: Метод handle_inference принимает InferenceOutput (целиком, для будущего использования вероятностей/confidence).
Flat Logic: Добавить поддержку close_on_flat. Если true, бот должен закрывать любую открытую позицию при получении сигнала Flat.
Координация: ExecutionEngine вызывает check_global_risk в начале цикла и check_order_gate непосредственно перед «отправкой» ордера.
Логирование: Использовать tracing::info! для логирования торговых решений.