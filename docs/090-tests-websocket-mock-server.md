Задача 090: Реализация Mock WebSocket-сервера для тестов
Цель: Создать инфраструктуру для детерминированного тестирования WebSocket-клиента. Симуляция сценариев Bybit V5 (публичных и приватных) в изолированном окружении.

1. Изменения в Cargo.toml
Добавить в [dev-dependencies]:
tokio-tungstenite = "0.21"
futures-util = "0.3"
2. Создание ./tests/common/ws_mock.rs
Структура WsMockServer:
pub struct WsMockServer {
    pub port: u16,
    pub tx: mpsc::Sender<tungstenite::Message>, // Канал управления из теста
    handle: tokio::task::JoinHandle<()>,
}
Реализация:
Dynamic Port: Привязка к 127.0.0.1:0, получение порта через listener.local_addr().port().
Server Task: tokio::spawn цикла accept().
Connection Handling:
Ping/Pong: Автоматический ответ pong на ping (можно отключить для теста таймаута).
Auth (Optional): Если пришло сообщение op: auth, проверить формат (задача 080) и отправить success: true/false.
Subscribe: При получении op: subscribe подтвердить подписку.
Scripted Injection: Чтение сообщений из mpsc::Receiver и отправка их подключенному клиенту (позволяет тесту точно контролировать время отправки Snapshot/Delta).
Chaos: Возможность отправить некорректный JSON или "битый" CRC32 для проверки устойчивости (задача 049).
3. Тестовые сценарии в ./tests/websocket_integration_tests.rs
Public Orderbook Flow:
Тест запускает WsMockServer.
Клиент подключается и подписывается на orderbook.50.
Тест через tx отправляет сначала 5 дельт, затем 1 снимок.
Assert: Проверка буферизации и корректной сборки стакана (задача 079).
Private Auth Flow:
Клиент отправляет auth с подписью.
Сервер имитирует ошибку авторизации.
Assert: Клиент должен вернуть AuthFailed и не переподключаться бесконечно.
Heartbeat Timeout:
Сервер игнорирует входящие ping.
Assert: Клиент разрывает соединение через 30 секунд (задача 077).
4. Почему этот план лучше (Аргументы Grok):
Control via MPSC: Тест сам решает, когда отправить "битую" дельту или задержать снимок, что делает тесты 100% воспроизводимыми.
Optional Auth: Мок пригоден и для простых публичных данных (LOB), и для сложных приватных (Fills/Orders).
Dynamic Port: Позволяет запускать тесты параллельно (cargo test) без конфликтов за TCP-порты.
Abortable: Использование JoinHandle позволяет корректно завершать сервер после каждого теста.
5. Критические требования
URL: Клиент должен подключаться к ws://127.0.0.1:{port}.
Async: Использование tokio_tungstenite для неблокирующей обработки.
Reliability: Сервер должен успевать обрабатывать ping, даже если тест "заснул"