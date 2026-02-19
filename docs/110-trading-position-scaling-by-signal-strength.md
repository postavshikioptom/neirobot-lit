
# Задача 110: Скейлинг объема по силе сигнала (v2.0)

## 1. Изменения в конфигурации `src/config/types.rs`
Добавь параметры напрямую в структуру `BotConfig`:
```rust
pub struct BotConfig {
    // ... существующие поля
    pub signal_min_confidence: f64,    // Минимальный порог входа (например, 0.45)
    pub signal_full_confidence: f64,   // Порог для максимального скейлинга (например, 0.85)
    pub signal_size_mult_min: f64,     // Множитель при min_confidence (например, 0.5)
    pub signal_size_mult_max: f64,     // Множитель при full_confidence (например, 2.0)
    pub total_size_mult_max: f64,      // Глобальный лимит (vol_mult * signal_mult)
}
```

## 2. Обновление типов инференса в `src/ml/types.rs`
Убедись, что выход модели содержит вероятности всех классов:
```rust
pub struct InferenceOutput {
    pub signal: Signal,             // Flat,Up, Down
    pub probabilities: Vec<f32>,    // [prob_flat, prob_up, prob_down]
}
```

## 3. Реализация логики скейлинга в `src/trading/execution.rs`

### Метод расчета множителя силы сигнала:
```rust
fn get_signal_multiplier(&self, output: &InferenceOutput) -> f64 {
    // Если модель предсказывает Flat — не торгуем
    if output.signal == Signal::Flat {
        return 0.0;
    }

    // Находим уверенность (максимальная вероятность среди Up/Down)
    let confidence = output.probabilities.iter()
        .copied()
        .fold(0.0f32, f32::max) as f64;

    if confidence < self.config.signal_min_confidence {
        return 0.0;
    }

    // Линейная интерполяция
    let t = ((confidence - self.config.signal_min_confidence) / 
            (self.config.signal_full_confidence - self.config.signal_min_confidence))
            .clamp(0.0, 1.0);
            
    self.config.signal_size_mult_min + t * (self.config.signal_size_mult_max - self.config.signal_size_mult_min)
}
```

### Полный метод `calculate_order_size` (Интеграция с задачей 105):
```rust
pub fn calculate_order_size(&self, inference: &InferenceOutput, base_size: Decimal) -> Decimal {
    // 1. Получаем множитель волатильности (из задачи 105)
    let vol_mult = self.get_volatility_multiplier(); 
    
    // 2. Получаем множитель силы сигнала
    let signal_mult = self.get_signal_multiplier(inference);

    if signal_mult <= 0.0 {
        return Decimal::ZERO;
    }

    // 3. Комбинируем и ограничиваем общий множитель
    let total_mult = (vol_mult * signal_mult).min(self.config.total_size_mult_max);
    
    // 4. Расчет финального объема в Decimal
    let multiplier_dec = Decimal::from_f64(total_mult).unwrap_or(Decimal::ONE);
    let scaled_size = base_size * multiplier_dec;
    
    // 5. Округление согласно шагу лота биржи
    let rounded_size = round_to_lot_step(scaled_size, self.exchange_config.min_lot_step);

    // 6. Финальная проверка через RiskManager
    if self.risk_manager.can_open_position(rounded_size) {
        rounded_size
    } else {
        tracing::warn!("Order size {} rejected by RiskManager", rounded_size);
        Decimal::ZERO
    }
}
```

## 4. Заметки по реализации
- **Confidence Calculation**: Используй `fold` или `max_by` для нахождения максимальной вероятности в векторе `probabilities`.
- **Flat Logic**: Даже если вероятность `Flat` составляет 0.99, бот должен вернуть объем 0, так как сигнал не является направленным.
- **Consistency**: Все расчеты скейлинга ведутся в `f64`, но финальный объем **обязательно** конвертируется в `Decimal` перед передачей в торговый модуль.

---

## Аргументация для Планировщика:
1.  **Linear Scaling**: Самый предсказуемый метод. Если `min=0.5` и `max=2.0`, то при средней уверенности мы получим объем, близкий к базовому, что логично.
2.  **Safety First**: Глобальный лимит `total_size_mult_max` защищает от ситуации, когда и волатильность низкая (высокий `vol_mult`), и сигнал сильный (высокий `signal_mult`), что могло бы привести к чрезмерному плечу.

**Gemini, реализуй эту логику в `execution.rs`, интегрировав её в существующий пайплайн обработки сигналов.**