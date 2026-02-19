Задача 077: Обработка WebSocket Pong и мониторинг задержки (RTT)
Цель: Реализовать детальный парсинг ответных сообщений pong от Bybit V5, измерение сетевой задержки (RTT) и интеграцию с механизмом реконнекта при таймауте.

1. Изменения в ./src/data/websocket.rs
Модели данных (внутри файла или в data/types.rs): Точное соответствие структуре Bybit V5 (Unified API).
#[derive(Debug, Deserialize)]
pub struct BybitPong {
    pub op: String,                // Всегда "pong"
    pub req_id: Option<String>,    // Если передавали в ping
    pub conn_id: String,           // ID соединения на стороне Bybit
    pub ts: u64,                   // Timestamp сервера в мс (как число или строка — проверить по факту)
    #[serde(default)]
    pub ret_msg: Option<String>,   // Может содержать "pong" или "OK"
}
Обновление состояния WebSocketClient:
last_ping_sent_at: Arc<Mutex<Option<Instant>>> — для точного расчета RTT.
last_activity: Arc<AtomicU64> — (уже реализовано в 076) для отслеживания таймаута 30с.
2. Логика обработки (Read Loop)
В основном цикле чтения сообщений (while let Some(msg) = ws_stream.next()):

При получении Message::Text:
Попытаться распарсить как BybitPong (или сначала проверить op: "pong" через serde_json::Value).
Если это pong:
Обновить last_activity (текущий timestamp).
Вычислить RTT:
Захватить замок last_ping_sent_at.
Если Some(sent_at), то rtt = Instant::now() - sent_at.
Сбросить замок в None.
Логирование:
debug!("WS Pong [{}]: RTT = {:?}, ServerTS = {}", conn_id, rtt, ts).
Если rtt > 500ms (или config.warn_rtt_ms) -> warn!("High RTT: {:?}", rtt).
3. Интеграция с Heartbeat (Задача 076)
Отправка: Перед ws_sink.send(ping), установить last_ping_sent_at = Some(Instant::now()).
Таймаут (30с): Если last_activity не обновлялась более 30 секунд (независимо от того, были это данные LOB или pong), фоновая задача должна инициировать trigger_reconnect().
4. Почему этот план лучше (Аргументы Grok):
Single File Parsing: Размещение структур в websocket.rs упрощает поддержку, так как формат сообщений жестко связан с логикой клиента.
Unified Timeout: Мы не ждем именно pong, мы ждем любую активность. Если данные идут, но pong нет — это странно, но не критично. Если нет ничего 30с — соединение "зомби".
RTT Precision: Использование Mutex<Option<Instant>> гарантирует, что мы считаем RTT только для последнего отправленного пинга, избегая дубликатов или накопленных задержек.
V5 Exact: Добавление conn_id и req_id позволяет различать ответы, если в будущем мы будем использовать несколько подписок в одном сокете.
5. Критические требования
Non-breaking Parse: Ошибка парсинга pong не должна останавливать цикл (использовать Result::ok()).
Configurable: Порог warn_rtt желательно вынести в ExchangeConfig.
Atomic vs Mutex: last_activity — атомарный (частое обновление), last_ping_sent_at — Mutex (редкое обновление, раз в 20с).
6. Тестирование
Integration: Имитировать задержку сети на мок-сервере (delay 600ms) и проверить появление warn!.
Reliability: Убедиться, что при обрыве интернета (нет pong и нет данных) реконнект срабатывает ровно через 30с.