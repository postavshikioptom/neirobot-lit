# Order State Machine (Задача 136)

## Быстрый старт

Новая архитектура State Machine для управления жизненным циклом ордеров.

**ВАЖНО**: Новая Order использует `f64` вместо `Decimal` для price, qty и executed_qty.
Согласно решению в задаче 136: "Поскольку Bybit API и наш стакан (013) используют f64,
использование Decimal создаст лишние касты и оверхед в 20%."

## Основные компоненты

### 1. OrderState (в `types.rs`)

```rust
pub enum OrderState {
    Created,         // Локально создан
    PendingNew,      // Запрос отправлен
    Active,          // Подтвержден биржей
    PartiallyFilled, // Частично исполнен
    Filled,          // Полностью исполнен
    PendingCancel,   // Запрос на отмену
    Cancelled,       // Отменен
    Rejected(String),// Отклонен
    Expired,         // Истек
}
```

### 2. OrderEvent (в `types.rs`)

```rust
pub enum OrderEvent {
    Accepted { order_id: String },
    Trade { exec_qty: Decimal, price: Decimal },
    CancelAck,
    Rejected { reason: String },
    Expired,
}
```

### 3. Order (в `order.rs`)

```rust
pub struct Order {
    pub link_id: String,
    pub order_id: Option<String>,
    pub state: OrderState,
    pub price: f64,              // f64 вместо Decimal
    pub qty: f64,                // f64 вместо Decimal
    pub executed_qty: f64,       // f64 вместо Decimal
    pub side: OrderSide,
    pub symbol: String,
    pub created_at: u64,
    pub updated_at: u64,
}
```

## Основные методы

### `transition(&mut self, event: OrderEvent) -> Result<(), String>`

Обрабатывает событие и переводит ордер в новое состояние.

```rust
order.transition(OrderEvent::Accepted {
    order_id: "EX123".to_string(),
})?;
```

### `mark_pending_new(&mut self)`

Переводит ордер в состояние PendingNew перед отправкой на биржу.

```rust
order.mark_pending_new();
// Отправка REST запроса
```

### `mark_pending_cancel(&mut self)`

Переводит ордер в состояние PendingCancel перед отправкой запроса на отмену.

```rust
order.mark_pending_cancel();
// Отправка запроса на отмену
```

### `remaining_qty(&self) -> Decimal`

Возвращает остаточный объем ордера.

```rust
let remaining = order.remaining_qty();
```

### `is_terminal(&self) -> bool`

Проверяет, находится ли ордер в терминальном состоянии.

```rust
if order.is_terminal() {
    // Удалить из active_orders
}
```

## Диаграмма переходов

```
Created
  ↓
PendingNew ──→ Rejected
  ↓
Active ──────→ PendingCancel ──→ Cancelled
  ↓                 ↓
PartiallyFilled    Expired
  ↓
Filled
```

## Примеры использования

### Создание и размещение ордера

```rust
use crate::trading::order::Order;
use crate::trading::types::{OrderSide, OrderEvent};

// Создание (используем f64)
let mut order = Order::new(
    "LIT_BTCUSDT_123".to_string(),
    "BTCUSDT".to_string(),
    OrderSide::Buy,
    50000.0,  // f64
    1.0,      // f64
    timestamp_ms(),
);

// Переход в PendingNew
order.mark_pending_new();

// Отправка на биржу
let result = rest_client.post("/v5/order/create", &request).await?;

// Обработка подтверждения
order.transition(OrderEvent::Accepted {
    order_id: result.order_id,
})?;
```

### Обработка исполнения

```rust
// Частичное исполнение (f64)
order.transition(OrderEvent::Trade {
    exec_qty: 0.5,
    price: 50000.0,
})?;

println!("Executed: {}/{}", order.executed_qty, order.qty);
println!("Remaining: {}", order.remaining_qty());
```

### Отмена ордера

```rust
// Переход в PendingCancel
order.mark_pending_cancel();

// Отправка запроса
rest_client.post("/v5/order/cancel", &request).await?;

// Обработка подтверждения
order.transition(OrderEvent::CancelAck)?;

// Проверка
assert!(order.is_terminal());
```

## Обработка ошибок

```rust
match order.transition(event) {
    Ok(()) => {
        info!("Order {} transitioned to {:?}", order.link_id, order.state);
    }
    Err(e) => {
        warn!("Invalid transition for order {}: {}", order.link_id, e);
        // Логируем, но не паникуем
    }
}
```

## Интеграция с OrderManager

```rust
// В order_manager.rs
pub fn process_order_event(&mut self, order_link_id: &str, event: OrderEvent) -> Result<()> {
    let order = self.active_orders.get_mut(order_link_id)?;
    
    match event {
        OrderEvent::Accepted { order_id } => {
            self.update_order(order_link_id, Some(order_id), OrderStatus::New, None);
        }
        OrderEvent::Trade { exec_qty, price } => {
            // Обновление executed_qty и статуса
        }
        // ... другие события
    }
    
    Ok(())
}
```

## Тестирование

```bash
# Запуск тестов State Machine
cargo test --lib trading::order

# Запуск конкретного теста
cargo test test_order_lifecycle_full_fill
```

## Важные принципы

1. **Всегда вызывайте `mark_pending_new()` перед отправкой ордера**
2. **Всегда вызывайте `mark_pending_cancel()` перед отменой**
3. **Не удаляйте ордер до `is_terminal() == true`**
4. **Обрабатывайте ошибки `transition()` gracefully**
5. **Используйте `remaining_qty()` для расчета остатка**

## Дополнительная документация

- Полное руководство: `docs/136-state-machine-migration-guide.md`
- Исходный код: `src/trading/order.rs`
- Тесты: `src/trading/order.rs` (mod tests)
- Задача: `docs/136-trading-order-state-machine-refactor.md`
