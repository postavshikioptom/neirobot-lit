
# Задача 115: Динамические пороги входа на основе серии убытков (v2.0)

## 1. Изменения в конфигурации `src/config/types.rs`
Добавь параметры чувствительности к убыткам в `BotConfig`:
```rust
pub struct BotConfig {
    // ...
    pub threshold_base: f64,             // Базовый порог (например, 0.55)
    pub threshold_loss_mult: f64,        // На сколько повышать порог за каждый убыток (например, 0.05)
    pub threshold_max: f64,              // Максимальный порог (например, 0.85)
    pub threshold_min: f64,              // Минимальный порог (например, 0.51)
    pub threshold_max_streak: usize,     // Кап на серию убытков для расчета (например, 5)
}
```

## 2. Расширение стейта в `src/trading/types.rs`
Для персистентности между перезапусками добавим счетчик в `BotState`:
```rust
pub struct BotState {
    // ...
    pub loss_streak: usize,              // Текущая серия убыточных сделок подряд
}
```

## 3. Обновление стейка в `src/trading/position_manager.rs`
Реализуй логику обновления серии при закрытии сделки:
```rust
pub fn update_streak(&mut self, trade_pnl: Decimal) {
    if trade_pnl < Decimal::ZERO {
        self.state.loss_streak += 1;
        tracing::warn!("Loss streak increased to: {}", self.state.loss_streak);
    } else if trade_pnl > Decimal::ZERO {
        self.state.loss_streak = 0;
        tracing::info!("Loss streak reset to 0");
    }
    // Сохранение стейта произойдет в execution.rs после вызова этого метода
}
```

## 4. Реализация в `src/risk/risk_manager.rs`
Метод расчета эффективного порога:
```rust
impl RiskManager {
    pub fn get_effective_threshold(&self, current_streak: usize) -> f64 {
        // Ограничиваем влияние серии лимитом из конфига
        let effective_streak = current_streak.min(self.config.threshold_max_streak) as f64;
        
        let dynamic_part = self.config.threshold_loss_mult * effective_streak;
        
        (self.config.threshold_base + dynamic_part)
            .clamp(self.config.threshold_min, self.config.threshold_max)
    }
}
```

## 5. Интеграция в `src/trading/execution.rs`
В методе `on_signal` перед принятием решения:

```rust
// 1. Получаем текущую серию убытков
let streak = self.state.loss_streak;

// 2. Рассчитываем динамический порог
let required_confidence = self.risk_manager.get_effective_threshold(streak);

// 3. Получаем уверенность модели для текущего сигнала (Up или Down)
let model_confidence = inference.confidence as f64;

if model_confidence < required_confidence {
    tracing::info!(
        "Signal rejected: Confidence {:.2} < Dynamic Threshold {:.2} (Streak: {})", 
        model_confidence, 
        required_confidence,
        streak
    );
    return Ok(());
}
```

---

## Аргументация для Планировщика:
1.  **Persistence**: Мы сохраняем `loss_streak` в `state.json`. Если бот упал после 3-го убытка, после перезапуска он «вспомнит» об этом и сохранит высокий порог входа.
2.  **Conservative Shift**: Этот механизм заставляет бота становиться более «придирчивым» к сигналам именно тогда, когда рынок ведет себя непредсказуемо для модели.
3.  **Soft Reset**: Любая прибыльная сделка (PnL > 0) полностью сбрасывает серию, возвращая бота к базовой агрессивности.

**Gemini, реализуй эту логику, обеспечив корректное обновление `loss_streak` на основе реальных данных об исполнении (Fills) и PnL закрытых позиций.**