# Задача 118: Период блокировки после убытков (Lockout Period) (v2.0)

## 1. Изменения в конфигурации `src/config/types.rs`
Добавь параметры блокировки в структуру `BotConfig`:
```rust
pub struct BotConfig {
    // ...
    pub lockout_period_sec: u64,          // Длительность блокировки (например, 3600)
    pub lockout_streak_threshold: usize,  // Порог серии убытков (например, 2)
}
```

## 2. Изменения в стейте `src/trading/types.rs`
Убедись, что поля присутствуют и сохраняются (на базе задачи 115):
```rust
pub struct BotState {
    // ...
    pub loss_streak: usize,              // Текущая серия убытков
    pub last_loss_timestamp_ms: i64,      // Unix MS последнего убытка
}
```

## 3. Реализация в `src/risk/risk_manager.rs`
Реализуй метод проверки с защитой от отрицательных интервалов времени:

```rust
impl RiskManager {
    pub fn is_in_lockout(&self) -> bool {
        let period = self.config.lockout_period_sec;
        let streak = self.state.loss_streak;
        let threshold = self.config.lockout_streak_threshold;

        // Если лимит не настроен или серия убытков не достигла порога
        if period == 0 || streak < threshold {
            return false;
        }

        let now_ms = Utc::now().timestamp_millis();
        // Используем saturating_sub для защиты от скачков времени
        let elapsed_ms = now_ms.saturating_sub(self.state.last_loss_timestamp_ms);
        let elapsed_sec = (elapsed_ms / 1000) as u64;

        if elapsed_sec < period {
            let remaining = period - elapsed_sec;
            tracing::warn!(
                "LOCKOUT ACTIVE: Streak {} >= {}. Cooling down for another {}s", 
                streak, threshold, remaining
            );
            return true;
        }

        false
    }
}
```

## 4. Обновление логики в `src/trading/execution.rs` (или `PositionManager`)
Обнови метод регистрации результата сделки, чтобы он управлял и серией, и таймером:

```rust
pub fn register_trade_result(&mut self, pnl: Decimal) {
    if pnl < Decimal::ZERO {
        self.state.loss_streak += 1;
        self.state.last_loss_timestamp_ms = Utc::now().timestamp_millis();
        tracing::warn!("Loss registered. Streak: {}", self.state.loss_streak);
    } else if pnl > Decimal::ZERO {
        // Сброс серии И выход из блокировки при профите
        self.state.loss_streak = 0;
        self.state.last_loss_timestamp_ms = 0;
        tracing::info!("Profit registered. Loss streak and lockout reset.");
    }
    
    // Сохраняем стейт (вызов save_state из задачи 107)
    if let Err(e) = self.save_current_state() {
        tracing::error!("Failed to save state after trade result: {}", e);
    }
}
```

## 5. Интеграция в основной цикл
В методе `on_signal` перед любой попыткой расчета объема или отправки ордера:
```rust
if self.risk_manager.is_in_lockout() {
    // Просто выходим, не генерируя лишних логов, если уже предупреждали
    return Ok(()); 
}
```

---

## Аргументация для Планировщика:
1.  **Saturating Sub**: Это стандарт безопасности в Rust. Если `last_loss_timestamp_ms` по какой-то причине окажется в будущем (глюк NTP), бот не запаникует, а просто посчитает `elapsed = 0`.
2.  **Profit Reset**: Если бот закрыл позицию в плюс (например, по тейк-профиту), это означает, что модель снова «попала» в рынок. Продолжать блокировку в этом случае нецелесообразно.
3.  **IO Efficiency**: Мы вызываем `save_state` только в момент изменения серии (событие Fill), а не при каждой проверке `is_in_lockout` на тике стакана.

**Gemini, реализуй эту логику, обеспечив корректное обновление стейта в `src/trading/state.rs`.**