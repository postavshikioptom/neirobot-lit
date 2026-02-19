# 044 - Trading Execution Signals
Цель задачи: Реализовать фильтрацию сигналов нейросети на основе порогов уверенности (thresholds). ExecutionEngine должен преобразовывать вероятности из InferenceOutput в торговые действия, используя индивидуальные настройки для покупки и продажи из конфигурации бота.

Файлы:

src/trading/execution.rs (обновить)
Инструкции для Gemini:

Обновить структуру ExecutionEngine: Добавить поля для порогов вероятности и флага закрытия позиции.
pub struct ExecutionEngine {
    pub order_manager: OrderManager,
    pub position_manager: PositionManager,
    pub risk_manager: RiskManager,
    pub symbol: String,
    pub thresh_buy: f32,
    pub thresh_sell: f32,
    pub close_on_flat: bool,
}
Реализовать логику фильтрации и обработки выхода модели:
impl ExecutionEngine {
    /// Точка входа для новых предсказаний модели
    pub fn on_inference_output(&mut self, output: InferenceOutput, current_price: f64) -> Result<()> {
        let position = self.position_manager.get_position();
        let unrealized = self.position_manager.calculate_unrealized_pnl(current_price);
        let current_pnl = position.realized_pnl + unrealized;

        // 1. Проверка глобального лимита просадки
        self.risk_manager.check_global_risk(current_pnl)?;

        // 2. Фильтрация сигнала по порогам вероятности
        let effective_signal = self.filter_signal(&output);

        // 3. Исполнение на основе отфильтрованного сигнала
        match effective_signal {
            Signal::Up => {
                if position.qty <= 0.0 {
                    self.execute_trade(OrderSide::Buy, current_price)?;
                }
            }
            Signal::Down => {
                if position.qty >= 0.0 {
                    self.execute_trade(OrderSide::Sell, current_price)?;
                }
            }
            Signal::Flat => {
                if self.close_on_flat && position.qty != 0.0 {
                    let side = if position.qty > 0.0 { OrderSide::Sell } else { OrderSide::Buy };
                    tracing::info!("[{}] Closing position due to Flat signal", self.symbol);
                    self.execute_trade(side, current_price)?;
                }
            }
        }

        Ok(())
    }

    /// Превращает вероятности в сигнал на основе порогов (thresh_buy/sell)
    fn filter_signal(&self, output: &InferenceOutput) -> Signal {
        let prob_up = output.probabilities[1];
        let prob_down = output.probabilities[2];

        if prob_up >= self.thresh_buy {
            Signal::Up
        } else if prob_down >= self.thresh_sell {
            Signal::Down
        } else {
            // Если уверенность ниже порогов, считаем сигнал нейтральным (Flat)
            if (prob_up > 0.4 || prob_down > 0.4) {
                tracing::debug!(
                    "[{}] Signal suppressed by thresholds: Up={:.2} (>{:.2}), Down={:.2} (>{:.2})",
                    self.symbol, prob_up, self.thresh_buy, prob_down, self.thresh_sell
                );
            }
            Signal::Flat
        }
    }
}
Технические требования:

Параметры: Поля thresh_buy, thresh_sell и close_on_flat должны инициализироваться в new из соответствующих полей BotConfig (задача 004/006).
Логика: Если ни одна вероятность (Up/Down) не превышает порог, метод filter_signal должен возвращать Signal::Flat.
Именование: Использовать метод on_inference_output для обработки результатов инференса.
Логирование: Использовать tracing::debug! для фиксации сигналов, которые были отсеяны порогами уверенности.
Безопасность: Сохранить вызов check_global_risk для предотвращения торговли при превышении просадки.