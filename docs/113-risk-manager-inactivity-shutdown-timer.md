# Задача 113: Таймер неактивности котировок (Dead Man's Switch) (v2.0)

## 1. Изменения в конфигурации `src/config/types.rs`
Добавь параметры в `BotConfig`:
```rust
pub struct BotConfig {
    // ...
    pub max_inactivity_ms: u64,              // Порог неактивности стакана (например, 5000)
    pub close_position_on_inactivity: bool, // Закрывать ли позу при потере данных (default: false)
}
```

## 2. Изменения в структуре `Execution` в `src/trading/execution.rs`
Добавь отслеживание состояния и времени последнего апдейта **стакана**:
```rust
use std::time::Instant;

pub struct Execution {
    // ...
    pub last_book_update: Instant,
    pub waiting_mode: bool,        // Флаг остановки торговли из-за отсутствия данных
}

impl Execution {
    // Вызывается ТОЛЬКО при получении OrderBookUpdate (019)
    pub fn poke_book_activity(&mut self) {
        self.last_book_update = Instant::now();
        if self.waiting_mode {
            tracing::info!("Data stream resumed. Resuming trading mode.");
            self.waiting_mode = false;
        }
    }
}
```

## 3. Реализация проверки в `src/risk/risk_manager.rs`
Метод проверки с предупреждением при достижении 50% порога:
```rust
pub fn check_inactivity(&self, last_update: Instant) -> bool {
    if self.config.max_inactivity_ms == 0 { return true; }

    let elapsed = last_update.elapsed().as_millis() as u64;
    
    if elapsed > self.config.max_inactivity_ms / 2 && elapsed < self.config.max_inactivity_ms {
        tracing::warn!("Inactivity warning: No book updates for {}ms", elapsed);
    }

    elapsed < self.config.max_inactivity_ms
}
```

## 4. Логика обработки в `src/trading/execution.rs`
Реализуй метод перехода в безопасный режим:
```rust
pub async fn handle_inactivity_trigger(&mut self) {
    if self.waiting_mode { return; } // Уже в режиме ожидания
    
    tracing::error!("INACTIVITY TRIGGERED: Quotes frozen. Entering safety mode.");
    self.waiting_mode = true;

    // 1. Отмена всех ордеров (ретраи как в 109)
    self.rest_client.cancel_all_orders(&self.config.symbol).await.ok();

    // 2. Опциональное закрытие позиции
    if self.config.close_position_on_inactivity {
        tracing::warn!("Closing position due to inactivity config.");
        self.emergency_market_close_only_pos().await.ok();
    }
}
```

## 5. Интеграция в основной цикл `src/bin/run-bot.rs`
В основном цикле `tokio::select!` объедини проверку и реконнект:

```rust
let mut heartbeat_tick = tokio::time::interval(Duration::from_millis(1000));

loop {
    tokio::select! {
        _ = heartbeat_tick.tick() => {
            if !execution.risk_manager.check_inactivity(execution.last_book_update) {
                execution.handle_inactivity_trigger().await;
                // Триггер реконнекта для WebSocket (задача 048)
                ws_reconnect_tx.send(ReconnectSignal::Immediate).ok();
            }
        }
        
        msg = ws_rx.recv() => {
            if let Some(update) = msg {
                if update.is_orderbook_update() {
                    execution.poke_book_activity();
                }
                if !execution.waiting_mode {
                    execution.on_update(update).await?;
                }
            }
        }
    }
}
```

---

## Аргументация для Планировщика:
1.  **Book vs Trades**: Мы смотрим именно на апдейты стакана. Если лента сделок идет, а стакан «замерз» — бот не может корректно считать `mid_price` и выставлять лимиты. Это критическая ошибка.
2.  **Waiting Mode**: Флаг `waiting_mode` позволяет не выходить из процесса, сохраняя загруженную модель ONNX и стейт, что ускоряет возобновление торговли после реконнекта.
3.  **Start-up**: Таймер `last_book_update` должен инициализироваться **только** после получения первого валидного снапшота (задача 020), чтобы избежать ложных срабатываний при долгой загрузке стакана.

**Gemini, реализуй эту логику, интегрировав её с системой переподключения WebSocket из задачи 048.**