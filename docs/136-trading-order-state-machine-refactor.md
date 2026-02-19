# Задача 136: Рефакторинг жизненного цикла ордера через State Machine (v2.0)

## 1. Определение состояний и событий в `src/trading/types.rs`
Определи чистые перечисления без данных внутри вариантов (кроме `Rejected`), чтобы упростить `match`-блоки.

```rust
// В [./src/trading/types.rs](./src/trading/types.rs)
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum OrderState {
    Created,        // Локально создан
    PendingNew,     // Отправлен запрос на создание
    Active,         // Подтвержден биржей
    PartiallyFilled, // Есть частичные исполнения
    Filled,         // Полностью исполнен
    PendingCancel,  // Отправлен запрос на отмену
    Cancelled,      // Отменен
    Rejected(String), // Отклонен (причина)
    Expired,        // Просрочен (Time-in-force)
}

#[derive(Debug)]
pub enum OrderEvent {
    Accepted { order_id: String },
    Trade { exec_qty: f64, price: f64 },
    CancelAck,
    Rejected { reason: String },
    Expired,
}
```

## 2. Логика ордера в `src/trading/order.rs` (Новый файл)
Создай структуру `Order` с поддержкой расчета остаточного объема и логикой переходов.

```rust
// В [./src/trading/order.rs](./src/trading/order.rs)
use crate::trading::types::{OrderState, OrderEvent, Side};

pub struct Order {
    pub link_id: String,
    pub order_id: Option<String>,
    pub state: OrderState,
    pub price: f64,
    pub qty: f64,
    pub executed_qty: f64,
    pub side: Side,
}

impl Order {
    pub fn remaining_qty(&self) -> f64 {
        self.qty - self.executed_qty
    }

    pub fn transition(&mut self, event: OrderEvent) -> Result<(), String> {
        let next_state = match (&self.state, event) {
            (OrderState::PendingNew, OrderEvent::Accepted { order_id }) => {
                self.order_id = Some(order_id);
                OrderState::Active
            }
            (OrderState::Active | OrderState::PartiallyFilled, OrderEvent::Trade { exec_qty, .. }) => {
                self.executed_qty += exec_qty;
                if self.executed_qty >= self.qty {
                    OrderState::Filled
                } else {
                    OrderState::PartiallyFilled
                }
            }
            (OrderState::PendingCancel, OrderEvent::CancelAck) => OrderState::Cancelled,
            (_, OrderEvent::Rejected { reason }) => OrderState::Rejected(reason),
            (_, OrderEvent::Expired) => OrderState::Expired,
            (s, e) => return Err(format!("Invalid transition from {:?} via {:?}", s, e)),
        };
        
        self.state = next_state;
        Ok(())
    }
}
```

## 3. Спорные моменты и корректировки (Grok + Zencoder)

- **f64 vs Decimal**: Согласен с Grok. Поскольку Bybit API и наш стакан (013) используют `f64`, использование `Decimal` создаст лишние касты и оверхед в 20%. Мы принимаем риск потери точности на 10-й значащей цифре ради скорости.
- **Race Conditions**: `transition` должен возвращать `Result`. Если `Accepted` пришел для ордера, который уже `Filled` (из-за лага REST), мы просто логируем это как `warn`, не ломая логику.
- **PendingCancel**: Ордер НЕ удаляется из `OrderManager` до получения `OrderEvent::CancelAck` или `OrderEvent::Trade` (если отмена не успела). Это предотвращает десинхронизацию позиции.
- **Executed Qty**: Поле `executed_qty` обновляется инкрементально. В `OrderManager` мы суммируем `executed_qty` всех активных ордеров для корректного расчета `Position.size`.

## 4. Инструкции для Gemini (Coder AI):
1. **[./src/trading/types.rs](./src/trading/types.rs)**: Добавить `OrderState` (без данных в вариантах) и `OrderEvent`.
2. **[./src/trading/order.rs](./src/trading/order.rs)**: Создать файл и реализовать структуру `Order` с методом `transition`.
3. **[./src/trading/order_manager.rs](./src/trading/order_manager.rs)**: Обновить хранилище ордеров (`HashMap<String, Order>`) и логику обработки `ExecutionReport` из WebSocket.
4. **[./src/trading/execution.rs](./src/trading/execution.rs)**: Рефакторить вызовы `place_order` и `cancel_order`, чтобы они корректно переводили ордера в `PendingNew` и `PendingCancel`.

**Результат**: Прозрачный и надежный жизненный цикл ордера, исключающий «зависшие» состояния и ошибки в расчете текущей позиции.
