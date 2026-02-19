# Задача 235: Фоновая процедура очистки «зависших» ордеров (Stale Order Cleanup)

Реализация фонового процесса («подметания»), предназначенного для выявления и устранения расхождений между локальным состоянием бота и реальными данными на бирже Bybit. Это критический механизм защиты от «фантомных» ордеров.

## 1. Цель задачи
Гарантировать, что на бирже нет открытых ордеров, о которых не знает `OrderManager`. Процедура должна автоматически отменять неучтенные (untracked) ордера и опционально обновлять слишком старые (stale) заявки.

## 2. Инструкции по реализации для Gemini

### А. Rust: Логика сверки ([./src/trading/order_manager.rs](./src/trading/order_manager.rs))
1.  **Синхронизация через Mutex/RwLock**:
    *   Процедура очистки должна работать с блокировкой `active_orders`, чтобы избежать гонок с основным торговым циклом (использовать `tokio::sync::Mutex` или `RwLock`).
2.  **Асинхронный метод `run_cleanup_routine`**:
    ```rust
    pub async fn run_cleanup_routine(&self) -> Result<(), BotError> {
        // 1. Получение всех открытых ордеров с биржи
        let resp = self.rest_client.get("/v5/order/realtime", json!({
            "category": "linear",
            "symbol": self.symbol,
            "settleCoin": "USDT"
        })).await?;
        
        let exchange_orders: Vec<BybitOrder> = serde_json::from_value(resp["result"]["list"].clone())?;
        let mut orders_to_cancel = Vec::new();

        // 2. Логика фильтрации под блокировкой
        {
            let active = self.active_orders.read().await;
            for ord in exchange_orders {
                let is_untracked = !active.contains_key(&ord.order_id);
                let is_too_old = (Utc::now() - ord.created_time).num_minutes() > self.config.max_stale_age_min;
                
                if is_untracked || (self.config.auto_cancel_stale && is_too_old) {
                    orders_to_cancel.push(ord.order_id);
                }
            }
        }

        // 3. Массовая отмена
        for id in orders_to_cancel {
            tracing::warn!("Cleanup: Cancelling stale/untracked order {}", id);
            let _ = self.cancel_order(&id).await;
        }
        Ok(())
    }
    ```

### Б. Запуск и интеграция ([./src/bin/run-bot.rs](./src/bin/run-bot.rs))
1.  В функции `main` создать долгоживущую задачу:
    ```rust
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(config.cleanup_interval_min * 60));
        loop {
            interval.tick().await;
            if let Err(e) = order_manager.run_cleanup_routine().await {
                tracing::error!("Cleanup routine failed: {:?}", e);
            }
        }
    });
    ```

## 3. Настройки в BotConfig ([./src/config/types.rs](./src/config/types.rs))
Добавить поля:
-   **cleanup_interval_min**: `u64` (рекомендуется `60`).
-   **max_stale_age_min**: `u64` (рекомендуется `120`).
-   **auto_cancel_stale**: `bool` (разрешить ли удаление старых, но известных ордеров).

## 4. Ожидаемый результат
1.  Раз в час бот гарантированно очищает аккаунт от любых «мусорных» ордеров, оставшихся после сбоев.
2.  Исключен риск исполнения ордера, который бот считает удаленным (Ghost Order).
3.  Все действия фиксируются в стандартном логе `bot.log` с тегом `WARN`.

## 5. Необходимые зависимости
-   **Rust**: `chrono`, `serde_json`, `tokio`, `tracing`.
-   **Crate**: `rand` (для добавления небольшого джиттера к интервалу очистки, чтобы не спамить в ровные минуты).