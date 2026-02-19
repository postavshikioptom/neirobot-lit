# Задача 108: Логика «погони» (Order Chasing / Re-pegging) (v2.0)

## 1. Изменения в конфигурации `src/config/types.rs`
Добавь параметры в `BotConfig` и новый Enum для режимов погони:

```rust
#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum ChaseMode {
    ToBest,          // Ставить на Best Bid/Ask
    InsideSpread,    // Ставить внутрь спреда на distance_bps
    ToVWAP,          // Ставить по цене текущего VWAP (если возможно)
}

pub struct BotConfig {
    // ...
    pub chase_mode: ChaseMode,
    pub chase_threshold_bps: Decimal,  // Порог срабатывания (например, 2.0)
    pub chase_distance_bps: Decimal,   // Оффсет при переставлении (например, 0.5)
    pub chase_max_attempts: usize,
    pub chase_interval_ms: u64,
}
```

## 2. Обновление `OrderInfo` в `src/trading/types.rs`
Добавь поля для трекинга погони:
```rust
pub struct OrderInfo {
    // ...
    pub chase_count: usize,
    pub last_chase_ts: i64,
}
```

## 3. Реализация в `src/trading/execution.rs`
Внедрить метод `check_and_chase`, который вызывается в основном цикле при получении обновлений стакана (если прошло больше `chase_interval_ms` с последней проверки).

### Основная логика погони:
```rust
async fn check_and_chase(&mut self) {
    let now = Utc::now().timestamp_millis();
    let mid = self.orderbook.get_mid_price();
    let best_bid = self.orderbook.get_best_bid();
    let best_ask = self.orderbook.get_best_ask();

    for order in self.active_orders.values_mut() {
        if order.chase_count >= self.config.chase_max_attempts || 
           now - order.last_chase_ts < self.config.chase_interval_ms as i64 {
            continue;
        }

        let th_amount = mid * self.config.chase_threshold_bps / Decimal::from(10000);
        let mut needs_chase = false;

        if order.side == Side::Buy && order.price < (best_bid - th_amount) {
            needs_chase = true;
        } else if order.side == Side::Sell && order.price > (best_ask + th_amount) {
            needs_chase = true;
        }

        if needs_chase && self.is_signal_still_valid(order.side) {
            let new_price = self.calc_new_peg(order.side, mid, best_bid, best_ask);
            
            // Опциональный фильтр по VWAP (не догонять выше VWAP для Buy)
            if self.config.use_vwap_filter && !self.is_price_fair(new_price, order.side) {
                continue;
            }

            match self.rest_client.amend_order(&order.link_id, new_price).await {
                Ok(_) => {
                    order.price = new_price;
                    order.chase_count += 1;
                    order.last_chase_ts = now;
                    self.save_state(); // Атомарное сохранение (107)
                }
                Err(e) if e.is_not_found() => {
                    // Ордер уже исполнился или отменен биржей
                    self.handle_order_missing(&order.link_id);
                }
                Err(_) => {
                    // Если amend не поддерживается или ошибка — Cancel + Replace
                    self.cancel_and_replace(order, new_price).await;
                }
            }
        }
    }
}
```

### Вспомогательная функция расчета цены:
```rust
fn calc_new_peg(&self, side: Side, mid: Decimal, best_bid: Decimal, best_ask: Decimal) -> Decimal {
    let offset = mid * self.config.chase_distance_bps / Decimal::from(10000);
    match self.config.chase_mode {
        ChaseMode::ToBest => if side == Side::Buy { best_bid } else { best_ask },
        ChaseMode::InsideSpread => {
            if side == Side::Buy { best_bid + offset } else { best_ask - offset }
        },
        ChaseMode::ToVWAP => self.stats.get_vwap(None),
    }
}
```

## 4. Интеграция с циклом (Loop)
В `src/bin/run-bot.rs` (задача 047) добавь вызов `execution.check_and_chase().await` в конце обработки каждого сообщения из WebSocket, но **обязательно** с троттлингом (не чаще раза в 100-500мс), чтобы не перегружать CPU и API.

---

## Аргументация для Планировщика:
1.  **Chase Mode**: Гибкость в выборе цены (Best vs Inside) позволяет подстраиваться под волатильность разных токенов.
2.  **Signal Validity**: Мы не догоняем цену, если нейросеть больше не выдает сигнал `Up/Down`. Это предотвращает вход в сделку на «излете» движения.
3.  **Amend vs Replace**: Приоритет `amendOrder` критичен для сохранения позиции в очереди (Order Priority) на некоторых биржах и экономии Rate Limits.

**Gemini, реализуй эту логику, убедившись, что `calc_new_peg` учитывает минимальный шаг цены (tick_size) биржи.**