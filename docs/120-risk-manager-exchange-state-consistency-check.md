# Задача 120: Детальная сверка состояния с биржей (Reconciliation) (v2.0)

## 1. Изменения в конфигурации `src/config/types.rs`
Добавь параметры в структуру `BotConfig`:
```rust
pub struct BotConfig {
    // ...
    pub reconciliation_interval_sec: u64, // Интервал сверки (например, 60)
    pub sync_on_desync: bool,            // Пытаться ли синхронизировать (default: true)
    pub price_desync_threshold: Decimal, // Допустимая разница в цене (например, tick_size)
}
```

## 2. Реализация в `src/risk/risk_manager.rs`
Метод детальной верификации (сравнение всех параметров):

```rust
impl RiskManager {
    pub fn verify_consistency(
        &self, 
        local_pos: &Position, 
        ex_pos: &Position, 
        local_orders: &HashMap<String, OrderInfo>,
        ex_orders: &Vec<OrderInfo>
    ) -> bool {
        // 1. Сверка позиции: объем и средняя цена
        let size_diff = (local_pos.size - ex_pos.size).abs();
        let price_diff = (local_pos.avg_price - ex_pos.avg_price).abs();

        if size_diff > Decimal::ZERO || price_diff > self.config.price_desync_threshold {
            tracing::error!("POS DESYNC: Local {}@{} vs Ex {}@{}", 
                local_pos.size, local_pos.avg_price, ex_pos.size, ex_pos.avg_price);
            return false;
        }

        // 2. Сверка ордеров: маппинг по link_id
        if local_orders.len() != ex_orders.len() {
            tracing::error!("ORDER COUNT DESYNC: Local {} vs Ex {}", local_orders.len(), ex_orders.len());
            return false;
        }

        for ex_order in ex_orders {
            match local_orders.get(&ex_order.link_id) {
                Some(local_order) => {
                    if local_order.price != ex_order.price || local_order.qty != ex_order.qty {
                        tracing::error!("ORDER DATA DESYNC for {}: Local {}@{} vs Ex {}@{}", 
                            ex_order.link_id, local_order.qty, local_order.price, ex_order.qty, ex_order.price);
                        return false;
                    }
                },
                None => {
                    tracing::error!("ORPHAN ORDER: {} found on exchange but not locally", ex_order.link_id);
                    return false;
                }
            }
        }

        true
    }
}
```

## 3. Интеграция и защита от гонок в `src/trading/execution.rs`
Используй `tokio::sync::Mutex` для защиты торгового стейта:

```rust
pub async fn perform_reconciliation(&mut self) -> anyhow::Result<()> {
    // Блокируем стейт, чтобы WS-апдейты не мешали сверке
    let mut state_guard = self.state.lock().await;

    // 1. Fetch fresh data from REST
    let ex_position = self.rest_client.get_position(&self.config.symbol).await?;
    let ex_orders = self.rest_client.get_open_orders(&self.config.symbol).await?;

    // 2. Проверка
    let is_consistent = self.risk_manager.verify_consistency(
        &state_guard.position,
        &ex_position,
        &state_guard.active_orders,
        &ex_orders
    );

    if !is_consistent {
        if self.config.sync_on_desync {
            tracing::info!("Desync detected. Force syncing local state...");
            
            // Синхронизация позиции
            state_guard.position.size = ex_position.size;
            state_guard.position.avg_price = ex_position.avg_price;

            // Синхронизация ордеров: отменяем локальные "призраки" и принимаем биржевые
            state_guard.active_orders.clear();
            for order in ex_orders {
                state_guard.active_orders.insert(order.link_id.clone(), order);
            }
            
            save_state(&state_guard, &self.state_path).ok();
        } else {
            state_guard.emergency_mode = true;
            return Err(anyhow!("Critical desync! Manual intervention required."));
        }
    }

    Ok(())
}
```

## 4. Особенности реализации
- **Orphan Orders**: Если на бирже обнаружен ордер, которого нет в локальной памяти, бот либо «удочеряет» его (добавляет в `active_orders`), либо (более безопасно) — отменяет его через REST API.
- **Race Condition**: В `src/bin/run-bot.rs` обработка сообщений из WebSocket и запуск сверки должны использовать один и тот же `Mutex` на `Execution` или `BotState`. Это гарантирует, что мы не сравниваем "старый" стейт с "новыми" данными API.
- **Rounding**: При сравнении цен используй `price_desync_threshold` (обычно 1 `tick_size`), так как биржа может округлять `avg_price` иначе, чем бот.

---

## Аргументация для Планировщика:
1.  **Atomic Sync**: Мы полностью очищаем и перезаписываем список ордеров при десинке. Это самый надежный способ «сбросить» ошибки логики.
2.  **Detailed Match**: Сверка по `link_id` — единственный способ гарантировать, что бот контролирует именно свои ордера.
3.  **Locking**: Без блокировки стейта во время `await` запроса к API велика вероятность получить `false positive` десинк, если в этот момент исполнился ордер и пришел WS-пакет.

**Gemini, реализуй эту логику, обеспечив подробное логгирование каждого несовпадающего параметра при обнаружении десинка.**