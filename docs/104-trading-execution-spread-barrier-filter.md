# Задача 104: Trading Execution Spread Barrier Filter (Dynamic)

## 1. Цель
Реализовать защитный фильтр спреда в [src/trading/execution.rs](./src/trading/execution.rs). Чтобы не повторять ошибок прошлого бота, фильтр будет использовать **двойную проверку**: жесткий "сумасшедший" порог (Static Cap) и динамический порог относительно исторической нормы (Dynamic Multiplier). Это позволит торговать на волатильных мем-коинах, отсекая только реальные разрывы ликвидности.

## 2. Логика и Пороги
1.  **Static Cap (Жесткий порог)**: Устанавливаем **200 bps (2.0%)**. Это абсолютный предел. Если спред шире 2%, вход блокируется всегда (защита от "дырявого" стакана).
2.  **Dynamic Multiplier (Динамический предел)**: Блокировка, если текущий спред превышает историческую норму (EMA) в **5 раз** (как ты просил).
    *   *Пример (CAKEUSDT)*: Норма 0.02% $\times$ 5 = 0.1% (фильтр сработает при спреде > 0.1%).
    *   *Пример (FARTCOINUSDT)*: Норма 0.4% $\times$ 5 = 2.0% (фильтр сработает при спреде > 2.0%).

## 3. Изменения

### Файл: [src/config/types.rs](./src/config/types.rs)
```rust
pub struct BotConfig {
    /// Абсолютный максимум спреда (базисные пункты). По умолчанию: 200 (2%)
    pub max_spread_static_bps: Option<u32>,
    /// Во сколько раз текущий спред может превышать средний. По умолчанию: 5.0
    pub spread_multiplier: Option<f32>,
}
```

### Файл: [src/trading/execution.rs](./src/trading/execution.rs)
- **Хранение состояния**: Добавить поле `spread_ema: Decimal` в структуру (обновлять при каждом сигнале или тике).
- **Метод check_spread_barrier**:
```rust
fn check_spread_barrier(&mut self) -> bool {
    let (bid, ask) = self.orderbook.get_best_bid_ask();
    if bid.is_zero() || ask.is_zero() { return false; }

    let mid = (bid + ask) / Decimal::from(2);
    let current_spread_pct = (ask - bid) / mid;
    let current_bps = (current_spread_pct * Decimal::from(10000)).to_u32().unwrap_or(200);

    // 1. Проверка жесткого лимита (Static Cap)
    let static_limit = self.config.max_spread_static_bps.unwrap_or(200);
    if current_bps > static_limit {
        tracing::warn!("Spread blocked: {} bps > {} bps (Static Cap)", current_bps, static_limit);
        return false;
    }

    // 2. Проверка динамического лимита (5x от нормы)
    let multiplier = Decimal::from_f32_retain(self.config.spread_multiplier.unwrap_or(5.0)).unwrap();
    let dynamic_limit_pct = self.spread_ema * multiplier;
    
    if current_spread_pct > dynamic_limit_pct {
        tracing::warn!("Spread blocked: {:.4}% > {:.4}% (5x Norm)", current_spread_pct, dynamic_limit_pct);
        return false;
    }

    // Обновляем EMA (коэффициент 0.01 для плавности)
    self.spread_ema = self.spread_ema * Decimal::from_str("0.99").unwrap() + current_spread_pct * Decimal::from_str("0.01").unwrap();
    true
}
```

## 4. Критические требования
- **Инициализация EMA**: При старте бота `spread_ema` должна инициализироваться первым полученным значением спреда (чтобы не блокировать первые сделки из-за "нуля").
- **Интеграция**: Вызывать `if !self.check_spread_barrier() { return Ok(()); }` в начале функции обработки сигнала.
- **Отказоустойчивость**: Использовать `clamp` или `unwrap_or` при конвертации типов, чтобы избежать паники при расчетах на экстремальных свечах.

## 5. Зависимости
- `rust_decimal` для точных расчетов процентов и EMA.
