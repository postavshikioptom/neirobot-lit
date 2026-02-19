Задача 048-data-websocket-reconnect.md
Цель: Реализовать отказоустойчивый цикл WebSocket-соединения с бесконечным (или настраиваемым) реконнектом, экспоненциальной задержкой и защитой от "шторма" подключений (Jitter).

Инструкции для Gemini:
1. Зависимости (Cargo.toml)
Добавить rand = "0.8" для реализации Jitter.
Убедиться в наличии tokio с фичами time и sync.
2. Изменения в ./src/config/types.rs
Добавить в ExchangeConfig структуру для управления WS:
pub struct WebSocketConfig {
    pub base_delay_ms: u64,
    pub max_delay_ms: u64,
    pub max_attempts: Option<u32>, // None = бесконечно
}
3. Изменения в ./src/data/websocket.rs
Main Loop: Вся логика connect + subscribe + read_messages должна находиться внутри loop.
Exponential Backoff + Jitter:
При ошибке подключения рассчитывать задержку: min(base_delay * 2^attempt, max_delay).
Критически важно: Добавить Jitter: delay += rand::thread_rng().gen_range(0..500)ms.
Snapshot Logic (по совету Grok):
НЕ очищать OrderBook при дисконнекте.
В методе обработки сообщений: при типе snapshot вызывать orderbook.reset_with_snapshot(), который полностью очищает старые уровни и записывает новые.
Trigger Reconnect: Предусмотреть возможность прерывания цикла чтения (например, через CancellationToken или возврат специфической ошибки Error::HeartbeatTimeout), чтобы запустить цикл реконнекта принудительно.
4. Изменения в ./src/data/orderbook.rs
Добавить метод reset_with_snapshot(update: OrderBookUpdate), который гарантирует атомарную замену всех данных стакана на новые из пришедшего снимка.
5. Логирование
Логировать каждую попытку реконнекта: warn!("WS Disconnected. Reconnecting in {}ms (Attempt {})", delay, attempt).
При успешном subscribe сбрасывать счетчик attempt = 0 и логировать info!("WS Connected and Subscribed successfully").
Аргументация: Этот план обеспечивает 24/7 выживаемость (через бесконечный реконнект), корректность данных (через сброс по Snapshot) и стабильность сети (через Jitter). Группировка в ExchangeConfig логична, так как параметры реконнекта обычно специфичны для биржи.