# Задача 232: Предотвращение самосделок (Self-Match Prevention — SMP)

Реализация гибридного механизма предотвращения самосделок: сочетание серверного `smpType` от Bybit и локальной проверки активных ордеров в логике исполнения. Это необходимо для защиты от **Wash Trading** и минимизации комиссионных потерь.

## 1. Цель задачи
Интегрировать поддержку `smpType` в API-запросы и реализовать в `execution.rs` логику «отмены противоположного», которая срабатывает перед выставлением нового ордера.

## 2. Инструкции по реализации для Gemini

### А. Конфигурация ([./src/config/types.rs](./src/config/types.rs))
Добавить параметры SMP в `BotConfig`:
```rust
pub struct TradingConfig {
    pub smp_type: String, // "None", "CancelMaker", "CancelTaker", "CancelBoth"
    pub local_smp_enabled: bool, // Локальная проверка перед отправкой
}
```
*   **Рекомендация**: Установить `None` по умолчанию для гибкости, но в примерах использовать `CancelMaker`.

### Б. Локальная проверка ([./src/trading/execution.rs](./src/trading/execution.rs))
Перед вызовом `place_order` при получении нового сигнала:
1.  Проверить наличие активных ордеров противоположной стороны (например, если сигнал **Buy**, проверить наличие лимитных **Sell**).
2.  Если `local_smp_enabled == true` и ордер найден — инициировать его отмену:
    ```rust
    if signal.side == Side::Buy && self.order_manager.has_active_sell_orders() {
        tracing::info!("Local SMP: Cancelling opposite Sell orders before Buying");
        self.order_manager.cancel_all_orders().await?;
    }
    ```

### В. Обработка на стороне API ([./src/trading/order_manager.rs](./src/trading/order_manager.rs))
1.  **Отправка ордера**: В методе `place_order` добавить поле в JSON-тело:
    ```rust
    let body = json!({
        "symbol": self.symbol,
        "side": side,
        "orderType": "Limit",
        "qty": qty,
        "price": price,
        "smpType": self.config.trading.smp_type,
        "category": "linear",
    });
    ```
2.  **Обработка ошибки `110037`**:
    ```rust
    pub async fn handle_smp_event(&mut self) -> Result<(), BotError> {
        tracing::warn!("Bybit SMP Triggered for {}. Cooling down...", self.symbol);
        // Небольшая задержка, чтобы стакан очистился от отмененных ордеров
        tokio::time::sleep(Duration::from_millis(200)).await;
        Ok(())
    }
    ```

## 3. Специфика реализации
-   **Hybrid Model**: Локальная проверка (`execution.rs`) страхует от лишних запросов к API, а серверный `smpType` (`order_manager.rs`) является последним рубежом защиты на стороне биржи.
-   **Race Condition**: Использование `CancelMaker` на бирже предпочтительнее для скальпинга, так как позволяет новому (Taker) ордеру исполниться, удалив старый мешающий лимит.

## 4. Ожидаемый результат
1.  Бот никогда не торгует сам с собой (исключен риск «зацикливания»).
2.  При смене направления сигнала (с Buy на Sell) старые ордера противоположной стороны отменяются превентивно.
3.  События SMP логируются как `WARN` для последующего анализа эффективности стратегии.

## 5. Необходимые зависимости
-   **Rust**: `serde_json`, `tracing`, `tokio`.