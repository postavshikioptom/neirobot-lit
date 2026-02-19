# Задача 234: Адаптивный Rate Limit и динамический Backoff

Реализация интеллектуальной системы управления лимитами запросов (Rate Limiting) с разделением на категории (Orders, Market, Account) и механизмом превентивного замедления на основе анализа HTTP-заголовков Bybit V5.

## 1. Цель задачи
Обеспечить бесперебойную работу бота без получения временных банов (429/10006). Реализовать адаптивное управление частотой запросов, учитывающее различные квоты для торговых операций и получения рыночных данных.

## 2. Инструкции по реализации для Gemini

### А. Rust: Категоризация и состояние ([./src/utils/rate_limiter.rs](./src/utils/rate_limiter.rs))
1.  **Категории лимитов**: Создать перечисление `LimitCategory`:
    *   `Orders` (самые строгие лимиты на создание/отмену).
    *   `Market` (запросы глубины стакана, истории).
    *   `Account` (балансы, плечи, позиции).
2.  **Структура состояния**: Реализовать `RateLimitTracker`, который хранит `remaining`, `limit` и `reset_ts` отдельно для каждой категории в `Arc<RwLock<HashMap<LimitCategory, State>>>`.

### Б. Rust: Middleware-логика в REST-клиенте ([./src/trading/rest_client.rs](./src/trading/rest_client.rs))
1.  **Превентивный Throttle**: В методе `send_request` перед выполнением вызова:
    *   Определить категорию запроса.
    *   Если `remaining` в данной категории < 15%: добавить принудительную задержку `tokio::time::sleep(Duration::from_millis(200))`.
2.  **Парсинг заголовков**: После получения ответа обновить состояние трекера, используя `X-Bapi-Limit-Status` и `X-Bapi-Limit-Reset-Timestamp`.
    *   **Fallback**: Если заголовки отсутствуют, использовать стандартный `Retry-After`.

### В. Rust: Backoff с джиттером ([./src/utils/helpers.rs](./src/utils/helpers.rs))
1.  **Метод `handle_rate_limit_error`**:
    ```rust
    pub async fn apply_backoff(attempt: u32, base_ms: u64) {
        let jitter = rand::thread_rng().gen_range(0..100);
        let wait_ms = (base_ms * 2u64.pow(attempt)) + jitter;
        let final_wait = std::cmp::min(wait_ms, 60_000); // max 60s
        tokio::time::sleep(Duration::from_millis(final_wait)).await;
    }
    ```
    *   Использование **Jitter** обязательно для предотвращения эффекта "грохочущего стада" (thundering herd), когда все боты одновременно возобновляют запросы после паузы.

## 3. Конфигурация ([./src/config/types.rs](./src/config/types.rs))
Добавить в `BotConfig`:
-   **rate_limit_threshold_pct**: `f64` (порог включения замедления, например, `0.15`).
-   **backoff_base_ms**: `u64` (базовая задержка, например, `250`).

## 4. Ожидаемый результат
1.  Бот автоматически снижает активность при приближении к лимитам Bybit, отдавая приоритет торговым ордерам (`Orders`).
2.  При получении ошибки `10006` (Too Many Requests) бот уходит в экспоненциальное ожидание с рандомизацией (Jitter).
3.  Система защиты работает на уровне всего REST-клиента, охватывая и торговые операции, и синхронизацию балансов/позиций.

## 5. Необходимые зависимости
-   **Rust**: `reqwest`, `tokio`, `rand = "0.8"`, `tracing`.