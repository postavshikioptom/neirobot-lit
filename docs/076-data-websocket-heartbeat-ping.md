Задача 076: Реализация WebSocket Heartbeat (Ping/Pong Watchdog)
Цель: Реализовать активный мониторинг состояния WebSocket-соединения через фоновую задачу (heartbeat), обеспечивающую отправку ping и автоматический реконнект при отсутствии активности более 30 секунд (требование Bybit V5).

1. Изменения в ./src/config/types.rs
ExchangeConfig: Добавить параметры Heartbeat в общую структуру настроек биржи.
pub struct ExchangeConfig {
    // ... эндпоинты
    pub ws_ping_interval_secs: u64, // Default: 20
    pub ws_pong_timeout_secs: u64,  // Default: 30 (Bybit рекомендует < 60)
}
2. Изменения в ./src/data/websocket.rs
Состояние WebSocketClient:
last_activity: Arc<AtomicU64> — время (timestamp) последнего любого сообщения от биржи (включая pong и данные LOB). Использование Arc<AtomicU64> позволяет безопасно проверять время из фонового потока.
Фоновая задача Heartbeat (Spawn Task):
При установке соединения (метод connect) запускать tokio::spawn с циклом:
let last_activity = self.last_activity.clone();
let ping_interval = Duration::from_secs(config.ws_ping_interval_secs);
let timeout = Duration::from_secs(config.ws_pong_timeout_secs);

tokio::spawn(async move {
    let mut interval = tokio::time::interval(ping_interval);
    loop {
        interval.tick().await;
        
        // 1. Проверка таймаута (Zombie Connection)
        let elapsed = get_elapsed_since(last_activity.load(Ordering::Relaxed));
        if elapsed > timeout {
            warn!("WS Heartbeat timeout: {}s. Triggering reconnect...", elapsed.as_secs());
            trigger_reconnect_signal().await; // Сигнал в основной цикл
            break;
        }

        // 2. Отправка Ping
        if let Err(e) = ws_sink.send(Message::Text(r#"{"op":"ping"}"#.to_string())).await {
            error!("Failed to send WS ping: {}", e);
            break;
        }
    }
});
Основной цикл (Read Loop):
При получении любого сообщения от биржи: last_activity.store(current_timestamp, Ordering::Relaxed).
Если сообщение — {"op":"pong"}, просто обновлять метку времени (специальная обработка данных не требуется, так как pong уже считается "активностью").
3. Почему этот план лучше (Аргументы против упрощений):
Separate Task vs Select!: Использование tokio::spawn разделяет логику чтения данных и логику мониторинга. Это предотвращает "зависание" пинга, если основной цикл занят тяжелым парсингом глубокого стакана (LOB).
Any Message Activity: Bybit рекомендует реконнект, если нет никаких сообщений (даже данных) более 30с. На активном рынке данные идут постоянно, и пинг лишь страхует периоды затишья.
30s Timeout: Согласно документации Bybit V5, 30 секунд — это безопасный порог для обнаружения разрыва на уровне приложения, прежде чем TCP-стек осознает проблему.
Atomic Timestamp: Использование атомарных типов позволяет избежать лишних Mutex/Lock в высокопроизводительном цикле чтения данных.
4. Критические требования
JSON Format: Строго {"op":"ping"} (без пробелов, если возможно, для минимизации трафика).
Error Propagation: Ошибка отправки пинга должна приводить к завершению задачи и сигналу на реконнект в основной цикл (задача 048).
Dependencies: Требует tokio::time и std::sync::atomic.
5. Тестирование
Unit test: Проверить расчет elapsed через мок-атомик.
Integration test: Запустить клиента против мок-сервера, который перестает отвечать, и убедиться, что реконнект инициируется ровно через 30 секунд.