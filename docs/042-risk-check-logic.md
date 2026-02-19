# 042 - Risk Check Logic
Цель задачи: Углубить логику RiskManager, добавив отслеживание эквити (Equity), поддержку дневных лимитов и комплексную проверку ордеров. Менеджер должен не просто проверять статические лимиты, а динамически оценивать состояние бота (Drawdown) и блокировать торговлю при критических отклонениях.

Файлы:

src/trading/risk_manager.rs (обновить)
Инструкции для Gemini:

Обновить структуру RiskManager: Добавить поля для отслеживания пикового и дневного эквити.
pub struct RiskManager {
    config: RiskConfig,
    initial_equity: f64,
    peak_equity: f64,
    daily_start_equity: f64, // Для контроля дневного убытка (задача 072)
}
Реализовать методы обновления и глобальной проверки:
impl RiskManager {
    /// Обновление текущего эквити и пиковых значений
    pub fn update_equity(&mut self, current_pnl: f64) {
        let current_equity = self.initial_equity + current_pnl;
        if current_equity > self.peak_equity {
            self.peak_equity = current_equity;
        }
    }

    /// Сброс дневных показателей (вызывать раз в сутки)
    pub fn reset_daily(&mut self, current_pnl: f64) {
        self.daily_start_equity = self.initial_equity + current_pnl;
    }

    /// Глобальная проверка: можно ли боту открывать новые позиции?
    pub fn check_global_risk(&self, current_pnl: f64) -> Result<()> {
        let current_equity = self.initial_equity + current_pnl;

        // 1. Проверка Max Drawdown от пика (Cumulative)
        if let Some(max_dd_pct) = self.config.max_drawdown_pct {
            let dd_pct = (self.peak_equity - current_equity) / self.peak_equity;
            if dd_pct > max_dd_pct {
                bail!("Risk: Cumulative drawdown exceeded: {:.2}% > {:.2}%", dd_pct * 100.0, max_dd_pct * 100.0);
            }
        }

        // 2. Проверка дневного лимита потерь (задача 072)
        // TODO: Использовать max_daily_loss_pct из расширенного конфига
        
        Ok(())
    }

    /// Финальный фильтр перед отправкой ордера
    pub fn check_order_gate(
        &self,
        side: OrderSide,
        qty: f64,
        price: f64,
        current_pos: &Position,
        active_orders_count: usize,
        current_pnl: f64,
    ) -> Result<()> {
        // 1. Проверка глобальных рисков (DD)
        self.check_global_risk(current_pnl)?;

        // 2. Проверка минимальной стоимости ордера (Notional)
        // Используем min_notional из конфига (задача 003/006) или константу
        let min_notional = 5.0; // Bybit default для многих пар. В будущем из конфига.
        if qty * price < min_notional {
            bail!("Risk: Order value too small: {:.2} < {}", qty * price, min_notional);
        }

        // 3. Валидация параметров ордера (из задачи 041)
        self.validate_order(side, qty, current_pos, active_orders_count)?;

        Ok(())
    }
}
Технические требования:

Result: Метод check_global_risk должен возвращать Result<()>, чтобы стратегия получала детальную причину блокировки (например, "Drawdown exceeded").
Equity: Эквити рассчитывается как initial_equity + current_pnl (где pnl = realized + unrealized).
Notional: Добавить проверку минимальной стоимости ордера (по умолчанию 5.0 USDT) для исключения "dust" ордеров.
Интеграция: Метод update_equity должен вызываться при каждом обновлении цены или получении Fill.