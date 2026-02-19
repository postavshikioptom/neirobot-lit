Задача 084: Реализация Graceful Shutdown (Signal Handling)
Цель: Обеспечить корректное и безопасное завершение работы бота при получении сигналов SIGINT (Ctrl+C) или SIGTERM, включая уведомление всех асинхронных задач, отмену ордеров и закрытие сокетов.

1. Изменения в ./src/bin/run-bot.rs
Механизм уведомления: Использовать tokio_util::sync::CancellationToken для каскадного уведомления всех потоков (WebSocket, Inference, Logger).
Кроссплатформенный слушатель:
async fn wait_for_shutdown() {
    let ctrl_c = tokio::signal::ctrl_c();
    #[cfg(unix)]
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).expect("failed to install signal handler");
    
    tokio::select! {
        _ = ctrl_c => info!("SIGINT (Ctrl+C) received"),
        _ = terminate.recv() => info!("SIGTERM received"),
    }
}
2. Реализация цикла завершения (Shutdown Sequence)
В main() обернуть процедуру завершения в tokio::time::timeout (10 секунд):

Signal Notification: Вызвать token.cancel(). Все циклы (while !token.is_cancelled()) должны завершить текущую итерацию.
Cancel Orders: order_manager.cancel_all_orders().await (задача 063).
WS Graceful Close:
Отправить Message::Close с кодом 1000 (Normal) во все открытые WebSocket-стримы.
Дождаться закрытия соединений.
Logger Flush:
Закрыть канал TradeLogger (задача 083).
Дождаться, пока фоновая задача записи дочитает канал и вызовет flush().
Timeout Handling: Если через 10 секунд бот все еще работает — std::process::exit(1).
Double Ctrl+C: Если сигнал получен повторно во время процесса завершения — немедленный force exit.
3. Изменения в компонентах
WebSocketClient: В цикле while let Some(msg) = stream.next() добавить проверку token.is_cancelled().
Execution Loop: В основном цикле run_loop проверять состояние токена перед каждым инференсом.
Config: Добавить в BotConfig параметр emergency_close_on_exit: bool. Если true, вызвать position_manager.emergency_market_close().await перед отменой ордеров.
4. Почему этот план лучше (Аргументы Grok):
CancellationToken: Это стандарт Tokio для Graceful Shutdown. Он позволяет элегантно прервать select! и другие ожидающие задачи без Mutex<AtomicBool>.
Cross-platform: Использование cfg(unix) гарантирует, что бот будет корректно обрабатывать SIGTERM в Docker/Kubernetes, при этом оставаясь работоспособным на Windows (Ctrl+C).
Double Signal Safety: Защищает от "зависания" процесса завершения (например, при проблемах с сетью во время отмены ордеров).
Tungstenite Close: Отправка явного CloseFrame — это "хороший тон" протокола, позволяющий бирже немедленно освободить ресурсы сессии.
5. Критические требования
Dependencies: Добавить tokio-util = { version = "0.7", features = ["sync"] } в Cargo.toml.
Logging: Каждая стадия завершения должна сопровождаться логом (например, info!("Cancelling all orders...")).
Order of Operations: Сначала уведомление (прекращение новой торговли), затем очистка биржи, затем закрытие логов.
6. Тестирование
Unit test: Проверить, что CancellationToken корректно прерывает тестовый цикл.
Integration test: Запуск cargo run --bin run-bot и отправка kill -15 <pid>. Убедиться, что в CSV логе появилась финальная запись и все ордера на тестовой бирже исчезли.