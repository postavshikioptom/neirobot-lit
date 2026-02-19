# Задача 233: Обработка ошибок Price Band и стабилизация цен

Реализация защиты от «ценовых шоков» (Bybit Error `110010`) с обязательной проверкой стабилизации спреда перед возобновлением торгов. Бот должен переходить в режим ожидания не только по таймеру, но и до момента нормализации рыночных условий.

## 1. Цель задачи
Обеспечить сохранность капитала при Flash Crash или резких пампах. Бот должен прекратить торговлю при выходе цены за лимиты биржи и вернуться в работу только когда спред и отклонение от маркированной цены (Mark Price) придут в норму.

## 2. Инструкции по реализации для Gemini

### А. Данные: Подписка на Mark Price ([./src/data/websocket.rs](./src/data/websocket.rs))
1.  **Новый канал**: Добавить подписку на публичный канал `markPrice.SYMBOL` (или `tickers.SYMBOL`), чтобы получать актуальную маркированную цену в реальном времени.
2.  **Хранение**: Сохранять `mark_price` в структуру `OrderBook` или `MarketState` для быстрого доступа из риск-менеджера.

### Б. Rust: Обработка ошибки и режим шока ([./src/trading/order_manager.rs](./src/trading/order_manager.rs))
1.  **Метод перехвата**:
    ```rust
    pub async fn handle_price_band_violation(&mut self) -> Result<(), BotError> {
        self.is_price_shock = true;
        tracing::error!("Price band violation detected for {}. Suspending trading...", self.symbol);
        
        // Ожидание базового периода охлаждения
        tokio::time::sleep(Duration::from_secs(self.config.price_band_cooldown_sec)).await;
        
        // Цикл проверки стабилизации (Spread + Mark Deviation)
        loop {
            let spread = self.market_data.get_spread_bps();
            let mark_dev = (self.market_data.get_mid_price() - self.market_data.get_mark_price()).abs() / self.market_data.get_mark_price();
            
            if spread < self.config.max_spread_bps && mark_dev < self.config.max_mark_deviation {
                tracing::info!("Market stabilized for {}. Resuming...", self.symbol);
                self.is_price_shock = false;
                break;
            }
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
        Ok(())
    }
    ```

### В. Интеграция в Risk Manager ([./src/risk/risk_manager.rs](./src/risk/risk_manager.rs))
1.  **Eligibility Gate**: В методе `is_eligible_to_trade` добавить проверку: `if self.state.is_price_shock { return false; }`.
2.  **Превентивный фильтр**: Перед отправкой ордера сравнивать `order_price` с `mark_price`. Если отклонение превышает лимит биржи (Price Band), блокировать отправку, не дожидаясь ошибки от API.

## 3. Конфигурация ([./src/config/types.rs](./src/config/types.rs))
Добавить параметры:
-   **price_band_cooldown_sec**: `u64` (минимум 60с).
-   **max_mark_deviation**: `f64` (макс. отклонение мида от марки, например, `0.02`).
-   **max_spread_bps**: `f64` (порог спреда для выхода из шока, например, `15.0`).

## 4. Ожидаемый результат
1.  При получении ошибки `110010` бот мгновенно «замирает».
2.  Возобновление торгов происходит плавно: только когда спред сузится и цена «мида» приблизится к цене «индекса».
3.  Исключены лишние запросы к API в моменты хаоса благодаря локальной проверке отклонения от Mark Price.

## 5. Необходимые зависимости
-   **Rust**: `tracing`, `tokio`, `serde_json`.