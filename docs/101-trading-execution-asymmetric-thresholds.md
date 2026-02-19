
# Задача 101: Trading Execution Asymmetric Thresholds

## 1. Цель
Реализовать поддержку **асимметричных порогов уверенности** в [src/trading/execution.rs](./src/trading/execution.rs). Вместо единого порога для биполярного сигнала, бот должен использовать индивидуальные пороги для вероятностей классов `Up` и `Down`, полученных после `softmax` (задача 037). Это позволяет гибко настраивать чувствительность входа для лонга и шорта отдельно.

## 2. Изменения

### Файл: [src/config/types.rs](./src/config/types.rs)
- Добавить поля напрямую в `BotConfig` (избегая лишней вложенности):
```rust
pub struct BotConfig {
    // ... существующие поля
    pub long_threshold: Decimal,      // Порог для вероятности Up (например, 0.65)
    pub short_threshold: Decimal,     // Порог для вероятности Down (например, 0.60)
    pub exit_threshold: Option<Decimal>, // Порог для выхода (если уверенность падает ниже X)
}
```
- В `config/loader.rs` или модуле валидации (задача 007) добавить проверку: `long_threshold` и `short_threshold` должны быть `> 0` и `< 1`.

### Файл: [src/trading/execution.rs](./src/trading/execution.rs)
- **Логика на основе вероятностей**:
    - Использовать структуру `InferenceOutput` (из задачи 033/037), содержащую `probs: [f32; 3]` (где индексы: 0 - Flat, 1 - Up, 2 - Down).
    - Конвертировать `Decimal` из конфига в `f32` для сопоставления с выходом модели:
```rust
let long_th = config.long_threshold.to_f32().unwrap_or(0.6);
let short_th = config.short_threshold.to_f32().unwrap_or(0.6);

let prob_up = inference.probs[1];
let prob_down = inference.probs[2];

if prob_up > long_th {
    self.execute_trade(Side::Buy).await?;
} else if prob_down > short_th {
    self.execute_trade(Side::Sell).await?;
} else if let Some(exit_th) = config.exit_threshold {
    let exit_th_f = exit_th.to_f32().unwrap_or(0.4);
    // Логика выхода: если мы в позиции, но уверенность в направлении упала ниже порога
    if (self.current_side == Side::Buy && prob_up < exit_th_f) || 
       (self.current_side == Side::Sell && prob_down < exit_th_f) {
        self.close_position().await?;
    }
}
```

## 3. Критические требования
- **Безопасность типов**: Использовать `to_f32()` для `rust_decimal::Decimal`. Обрабатывать возможные ошибки конвертации через `unwrap_or` с безопасными дефолтами.
- **Приоритет входа**: Если обе вероятности (Up и Down) выше порогов (редкий случай при `softmax`), бот должен либо игнорировать сигнал, либо выбирать максимальный. Рекомендуется **игнорировать** как противоречивый сигнал.
- **Логирование**: В `tracing::info!` выводить все три вероятности `[F, D, U]` при принятии решения о входе/выходе.

## 4. Зависимости
- `src/ml/types.rs` (структура `InferenceOutput`).
- `rust_decimal` (конвертация в `f32`).
