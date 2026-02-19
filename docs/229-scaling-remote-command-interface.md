# Задача 229: Интерфейс удаленного управления (Remote Command Interface)

Переработка задачи **229** с учетом архитектурных правок **Grok**: перенос CLI-скриптов в `python_lab/scripts/`, детализация Rust-сигнатур и выбор `Textual` для интерактивного дашборда.

## 1. Цель задачи
Реализовать систему RPC-управления (Remote Procedure Call) для контроля жизненного цикла ботов (Panic, Pause, Reload) и централизованный TUI-интерфейс для оперативного мониторинга всей фермы.

## 2. Инструкции по реализации для Gemini

### А. Rust: Командный сервер ([./src/monitoring/command_server.rs](./src/monitoring/command_server.rs))
1.  **Реализация API на Axum**:
    *   Создать функцию `pub async fn start_command_server(config: &BotConfig, tx: mpsc::Sender<Command>) -> Result<(), Box<dyn Error>>`.
    *   Использовать `axum` для маршрутизации:
        ```rust
        let app = Router::new()
            .route("/status", get(status_handler))
            .route("/panic", post(panic_handler))
            .route("/reload", post(reload_handler));
        ```
2.  **Интеграция в Main ([./src/bin/run-bot.rs](./src/bin/run-bot.rs))**:
    *   Запускать сервер в отдельной задаче: `tokio::spawn(start_command_server(&config, cmd_tx))`.
    *   `status_handler` должен запрашивать данные через атомарные ссылки (`Arc`) или каналы у `PositionManager` и `OnnxEngine`.
3.  **Безопасность**: Привязка (bind) строго к `127.0.0.1:PORT`.

### Б. Python: Фермерский контроллер и TUI ([./python_lab/scripts/farm_ctl.py](./python_lab/scripts/farm_ctl.py))
1.  **Textual Dashboard**:
    *   Реализовать класс `FarmApp(App)`, использующий `DataTable` для отображения списка ботов.
    *   Метод `compose()` должен определять структуру: Header, Footer, и центральную таблицу со статусами (Symbol, PnL, Pos, Latency).
    *   Метод `on_mount()` запускает `set_interval(1.0, update_status)` для поллинга `/status` всех ботов.
2.  **CLI Commands**:
    *   Реализовать через `click` или `argparse` быстрые команды: `panic --symbol BTC`, `reload --all`.

## 3. Спорные моменты и аргументация

-   **Путь к скриптам**: Полностью согласен с Grok. Все инструменты управления (train, export, farm_ctl) должны лежать в `python_lab/scripts/`, следуя логике изоляции лаборатории от инфраструктуры.
-   **Выбор Textual**: Textual — это стандарт для TUI-приложений в 2025 году. В отличие от `rich.live`, он событийный (event-driven), что упрощает обработку кликов по кнопкам (например, кнопка «PANIC» напротив каждого бота в таблице).
-   **Сигналы vs HTTP**: Я настаиваю на HTTP (Axum) на localhost. Сигналы (SIGUSR1) ограничены и не позволяют передавать сложные JSON-ответы с метаданными состояния, что критично для дашборда.
-   **Разделение команд**: Команда `panic` должна обрабатываться в Rust немедленно через `CancellationToken` или высокоприоритетный канал, чтобы обойти очередь рыночных сигналов.

## 4. Ожидаемый результат
1.  Запущенный бот открывает управляющий порт (например, `9001`).
2.  Оператор видит всю ферму в интерактивном терминальном окне (`python_lab/scripts/farm_ctl.py ui`).
3.  Возможность мгновенно остановить торговлю по конкретному тикеру одной кнопкой в TUI.

## 5. Необходимые зависимости
-   **Rust**: `axum = "0.7"`, `serde_json`, `tokio`.
-   **Python**: `textual = "0.80"`, `httpx`, `click`.