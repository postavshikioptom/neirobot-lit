# Задача 114: Лимит номинального риска на символ (Notional/Delta Limit) (v2.0)

## 1. Изменения в конфигурации `src/config/types.rs`
Добавь параметр в структуру `BotConfig`:
```rust
pub struct BotConfig {
    // ...
    pub max_notional_usd: Decimal,      // Максимальный риск в USD (0 = отключено)
}
```

## 2. Обновление `OrderManager` в `src/trading/order_manager.rs`
Добавь метод для подсчета суммарного объема активных ордеров по конкретной стороне:
```rust
impl OrderManager {
    pub fn get_pending_size(&self, side: Side) -> Decimal {
        self.active_orders.values()
            .filter(|o| o.side == side)
            .map(|o| o.qty)
            .sum()
    }
}
```

## 3. Реализация в `src/risk/risk_manager.rs`
Реализуй проверку, учитывающую текущую позицию, ожидаемые ордера и новую заявку:

```rust
pub fn check_notional_limit(
    &self, 
    current_size: Decimal, 
    pending_size: Decimal, // Объем активных ордеров на той же стороне
    order_size: Decimal, 
    order_side: Side, 
    mid_price: Decimal
) -> bool {
    // 1. Если лимит не установлен — пропускаем
    if self.config.max_notional_usd.is_zero() {
        return true;
    }

    // 2. Рассчитываем целевой размер позиции (интегральный риск)
    // Важно: учитываем знак позиции (Long +, Short -)
    let new_size = match order_side {
        Side::Buy => current_size + pending_size + order_size,
        Side::Sell => current_size - pending_size - order_size,
    };

    let new_notional = new_size.abs() * mid_price;

    // 3. Логика Reduce-Only: если новый риск меньше текущего — разрешаем всегда
    let current_notional = (current_size.abs() + pending_size) * mid_price;
    if new_notional <= current_notional {
        return true; 
    }

    // 4. Проверка лимита на увеличение позиции
    if new_notional > self.config.max_notional_usd {
        tracing::warn!(
            "NOTIONAL BLOCKED: Future exposure ${:.2} > Limit ${:.2}", 
            new_notional, 
            self.config.max_notional_usd
        );
        return false;
    }

    true
}
```

## 4. Интеграция в `src/trading/execution.rs`
В методе обработки сигнала (`on_signal`):

```rust
// 1. Собираем данные
let current_pos = self.position_manager.get_size();
let pending_same_side = self.order_manager.get_pending_size(signal_side);
let mid = self.orderbook.get_mid_price();

// 2. Проверяем лимит перед расчетом или отправкой
if !self.risk_manager.check_notional_limit(
    current_pos, 
    pending_same_side, 
    scaled_size, 
    signal_side, 
    mid
) {
    // Если лимит превышен, мы можем либо обнулить объем, 
    // либо урезать его до максимально допустимого остатка (optional)
    return Ok(()); 
}
```

## 5. Особенности реализации
- **Absolute Value**: Номинальный риск всегда считается через `.abs()`, так как для системы риск в $10,000 в шорте так же критичен, как и в лонге.
- **Pending Orders**: Мы суммируем только ордера на той же стороне (`order_side`), так как ордера на противоположной стороне технически уменьшают риск (являются `ReduceOnly` или закрывающими).
- **Rounding**: Проверка лимита должна происходить **до** финального округления объема по `lot_step`, либо с небольшим запасом, чтобы ошибки округления в несколько центов не блокировали торговлю.

---

## Аргументация для Планировщика:
1.  **Safety First**: Учет `pending_size` гарантирует, что если бот выставил 5 лимитов по 1 BTC, он не выставит шестой, если лимит — 5 BTC. Без этого учета возникла бы "очередь" из ордеров, которая при исполнении взорвала бы риск.
2.  **Directional Neutrality**: Лимит работает одинаково для обеих сторон, защищая от перекоса дельты.
3.  **No Lock-out**: Разрешение любых операций, снижающих `abs(new_size)`, гарантирует, что бот всегда сможет выйти из позиции, даже если она случайно (например, из-за изменения цены) превысила лимит.

**Gemini, реализуй этот гейт, убедившись, что `OrderManager` корректно отдает `pending_size` в реальном времени.**