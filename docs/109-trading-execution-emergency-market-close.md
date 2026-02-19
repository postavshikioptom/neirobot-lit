# Задача 109: Экстренное закрытие позиций (Panic Exit) (v2.0)

## 1. Изменения в конфигурации `src/config/types.rs`
Добавь в `BotConfig`:
```rust
pub struct BotConfig {
    // ...
    pub close_on_exit: bool,           // Закрывать ли по рынку при выходе (default: true)
    pub emergency_timeout_ms: u64,     // Таймаут на всю процедуру (например, 5000)
}
```

## 2. Реализация в `src/trading/execution.rs`
Реализуй метод `emergency_market_close` с использованием таймаута и ретраев:

```rust
pub async fn emergency_market_close(&mut self) -> anyhow::Result<()> {
    // Оборачиваем всю операцию в глобальный таймаут
    let timeout_ms = self.config.emergency_timeout_ms;
    tokio::time::timeout(Duration::from_millis(timeout_ms), self.perform_panic_exit()).await
        .map_err(|_| anyhow!("Emergency close timed out after {}ms!", timeout_ms))?
}

async fn perform_panic_exit(&mut self) -> anyhow::Result<()> {
    tracing::warn!("PANIC EXIT STARTED for {}", self.config.symbol);
    self.emergency_mode = true; // Блокируем новые сигналы

    // 1. Отмена ордеров с ретраями (max 3 попытки)
    let mut attempts = 0;
    while attempts < 3 {
        match self.rest_client.cancel_all_orders(&self.config.symbol).await {
            Ok(_) => {
                tracing::info!("All orders cancelled");
                break;
            }
            Err(e) if attempts < 2 => {
                attempts += 1;
                let delay = 100 * (2_u64.pow(attempts as u32)); // 200ms, 400ms
                tracing::warn!("Cancel orders failed, retrying in {}ms: {}", delay, e);
                tokio::time::sleep(Duration::from_millis(delay)).await;
            }
            Err(e) => {
                tracing::error!("Failed to cancel orders after 3 attempts: {}. Continuing to market close.", e);
                break; 
            }
        }
    }

    // 2. Закрытие позиции
    let position = self.rest_client.get_position(&self.config.symbol).await?;
    if position.size != Decimal::ZERO {
        let side = if position.size > Decimal::ZERO { Side::Sell } else { Side::Buy };
        let close_order = OrderRequest {
            symbol: self.config.symbol.clone(),
            side,
            order_type: OrderType::Market,
            qty: position.size.abs(),
            reduce_only: true, // ОБЯЗАТЕЛЬНО
            ..Default::default()
        };

        if let Err(e) = self.rest_client.place_order(close_order).await {
            tracing::error!("CRITICAL: Failed to place market close order: {}", e);
            return Err(anyhow!("Market close failed: {}", e));
        }
        tracing::info!("Emergency market order placed");
    }

    // 3. Финализация стейта
    self.state.position_size = Decimal::ZERO;
    self.state.active_orders.clear();
    save_state(&self.state, &self.state_path).ok();
    
    Ok(())
}
```

## 3. Интеграция в `src/bin/run-bot.rs`
Обнови основной цикл для обработки сигналов завершения:

```rust
// В main()
let mut sigint = tokio::signal::ctrl_c()?;

loop {
    tokio::select! {
        _ = sigint.recv() => {
            tracing::info!("SIGINT received, shutting down...");
            if config.close_on_exit {
                execution.emergency_market_close().await.ok();
            }
            break;
        }
        msg = ws_rx.recv() => {
            if let Some(update) = msg {
                if !execution.emergency_mode {
                    execution.on_update(update).await?;
                }
            }
        }
    }
}
```

## 4. Связь с Risk Manager `src/risk/risk_manager.rs`
В методе `check_risk` (задача 072), если зафиксирован критический просадок (`drawdown_stop_gate`):
- Вызывай `execution.emergency_market_close().await`.
- Логгируй событие как `CRITICAL_RISK_STOP`.

---

## Аргументация для Планировщика:
1.  **Reduce-Only**: Это «страховка» от двойного открытия позиции при гонке сигналов.
2.  **Backoff**: Ретраи с задержкой помогают пробить временные сетевые лаги API Bybit.
3.  **Timeout**: Мы не можем позволить боту «висеть» вечно при выключении, поэтому `tokio::time::timeout` — это жесткий предел жизни процесса.

**Gemini, твоя задача**: реализовать эту логику максимально надежно, гарантируя, что даже при частичных ошибках API бот попытается выполнить все шаги до конца.