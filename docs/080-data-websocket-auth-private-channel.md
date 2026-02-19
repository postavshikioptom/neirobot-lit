Задача 080: Аутентификация и приватный WebSocket-канал
Цель: Реализовать подключение к приватному WebSocket-каналу Bybit V5 для получения событий по ордерам, исполнениям (fills) и позициям в реальном времени.

1. Изменения в ./src/config/types.rs
ExchangeConfig: Добавить эндпоинт для приватного вещания.
pub struct ExchangeConfig {
    // ...
    pub private_ws_url: String, // e.g., "wss://stream.bybit.com/v5/private"
}
2. Изменения в ./src/data/websocket.rs
Метод generate_auth_message(): Реализовать формирование подписи строго по докам Bybit V5:
Expires: chrono::Utc::now().timestamp_millis() + 10000 (запас 10 секунд на сетевые лаги).
Signature String: Конкатенация "GET/realtime" и значения expires (без пробелов и /auth).
let prehash = format!("GET/realtime{}", expires);
HMAC-SHA256: Подпись prehash с использованием api_secret.
JSON Payload:
{
  "op": "auth",
  "args": [ "API_KEY", 1234567890, "SIGNATURE" ]
}
Логика PrivateWebSocketClient:
Подключение: Открыть сокет по private_ws_url.
Авторизация: Первым сообщением отправить auth пакет.
Ожидание подтверждения: Читать поток, пока не придет {"op": "auth", "success": true}. Если success: false — anyhow::bail!("WS Auth failed").
Подписка (Subscription): Сразу после успеха отправить запрос на темы:
{"op": "subscribe", "args": ["order", "execution", "position", "wallet"]}.
Heartbeat: Использовать логику из задач 076-077 (пинги обязательны и для приватного канала).
3. Критические требования
Signature Accuracy: Строка для подписи обязательно начинается с GET/realtime. Использование GET/auth или других префиксов приведет к ошибке 3303001 (Invalid authentication).
Separation: Приватный клиент должен быть отдельным инстансом WebSocketClient с собственной очередью сообщений.
Secrets Handling: Ключи брать из .env через те же механизмы, что в REST-клиенте (задача 061). Не допускать утечки api_secret в логи.
Dependencies: Убедиться в наличии hmac, sha2, hex, chrono в Cargo.toml.
4. Почему этот план лучше (Аргументы Grok):
GET/realtime: Это единственный корректный префикс для V5 Unified Private WS.
10s Buffer: Увеличенный expires предотвращает отклонение запроса при резких всплесках сетевой задержки.
Auto-subscribe: Подписка на execution и position позволяет PositionManager (задача 064) обновлять состояние мгновенно, не дожидаясь REST-опроса.
Consistency: Единый подход к подписи для REST и WS (HMAC-SHA256) упрощает кодовую базу.
5. Тестирование
Unit test: Проверить корректность формирования строки prehash.
Integration test: Попытка подключения с валидными ключами и проверка получения op: subscribe подтверждения от Bybit.
Error test: Убедиться, что при неверном секрете клиент корректно завершает работу с ошибкой "Auth failed", а не уходит в бесконечный реконнект.