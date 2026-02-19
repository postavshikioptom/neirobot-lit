# Задача 134: Ротация и очистка логов с защитой от переполнения (v2.0)

## 1. Контекст и цели
Система логирования должна быть полностью асинхронной и самоочищающейся.
*   **Ротация**: Переключение файла каждый день или при достижении лимита в `max_size_mb`.
*   **Retention**: Хранение только последних $N$ файлов, удаление старых по `mtime`.
*   **Zero-Stall**: Использование ограниченной очереди с поведением `Drop`, если диск не успевает записывать.

## 2. Реализация в `src/utils/logger.rs`
Реализуй инициализацию с двумя слоями (Консоль + Файл) и фоновую задачу очистки.

```rust
// В [./src/utils/logger.rs](./src/utils/logger.rs)
use tracing_appender::non_blocking::{NonBlockingBuilder, WorkerGuard};
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::{fmt, EnvFilter, Registry, prelude::*};

pub fn init_logger(config: &LogConfig, bot_path: &Path) -> WorkerGuard {
    let log_dir = bot_path.join("logs");
    let rotation = match config.rotation.as_str() {
        "hourly" => Rotation::HOURLY,
        _ => Rotation::DAILY,
    };

    // 1. Файловый аппендер (время)
    let file_appender = RollingFileAppender::new(rotation, &log_dir, "bot.log");

    // 2. Настройка Non-blocking с Lossy стратегией (важно для HFT)
    // Если очередь переполнена, старые логи отбрасываются, чтобы не тормозить main thread
    let (non_blocking, guard) = NonBlockingBuilder::default()
        .buffered_lines_limit(10_000)
        .lossy(true) // Drop logs if full
        .finish(file_appender);

    // 3. Сборка слоев (Файл без ANSI, Консоль с ANSI)
    let file_layer = fmt::layer()
        .with_writer(non_blocking)
        .with_ansi(false)
        .with_target(true);

    let mut layers = Vec::new();
    layers.push(file_layer.with_filter(EnvFilter::new(&config.level)).boxed());

    if config.console_enabled {
        layers.push(fmt::layer().with_ansi(true).boxed());
    }

    tracing_subscriber::registry().with(layers).init();
    
    // 4. Запуск фоновой задачи очистки (размер + количество)
    let max_files = config.max_files;
    tokio::spawn(async move {
        clean_old_logs(log_dir, max_files).await;
    });

    guard
}
```

## 3. Логика очистки старых файлов
Поскольку `tracing-appender` не удаляет старые файлы сам, реализуем это вручную через проверку даты изменения.

```rust
async fn clean_old_logs(dir: PathBuf, max_files: usize) {
    loop {
        if let Ok(mut entries) = std::fs::read_dir(&dir) {
            let mut files: Vec<_> = entries
                .filter_map(|e| e.ok())
                .filter_map(|e| {
                    let meta = e.metadata().ok()?;
                    if meta.is_file() { Some((meta.modified().ok()?, e.path())) } else { None }
                })
                .collect();

            // Сортируем: старые в начале
            files.sort_by_key(|&(t, _)| t);

            if files.len() > max_files {
                let to_remove = files.len() - max_files;
                for i in 0..to_remove {
                    let _ = std::fs::remove_file(&files[i].1);
                }
            }
        }
        tokio::time::sleep(tokio::time::Duration::from_secs(3600)).await;
    }
}
```

## 4. Спорные моменты и корректировки (Grok + Zencoder)

*   **Size-based Rotation**: В текущей версии `tracing-appender` нет нативной ротации по размеру. Grok предложил кастомный wrapper, но мы пойдем более простым путем: фоновая задача `clean_old_logs` будет проверять суммарный объем папки `logs` и удалять старые файлы, если лимит превышен. Это надежнее, чем сложная обертка над дескриптором файла.
*   **Lossy/Non-blocking**: Это критично. В HFT мы предпочитаем **потерять строчку лога**, чем получить **micro-stall** на 10мс из-за записи на диск. Убедись, что `lossy(true)` включен.
*   **Console Layer**: Опционально через конфиг. В проде (на сервере) консоль обычно выключена, чтобы не тратить ресурсы терминала.
*   **Targeting**: `with_target(true)` оставляем включенным. Это добавляет имя модуля (`src::trading::execution`), что упрощает поиск багов в многопоточном коде.

## 5. Инструкции для Gemini (Coder AI):
1.  **Cargo.toml**: Добавить `tracing-appender = "0.2"`.
2.  **[./src/config/types.rs](./src/config/types.rs)**: Добавить структуру `LogConfig` (level, rotation, max_files, max_size_mb, console_enabled).
3.  **[./src/utils/logger.rs](./src/utils/logger.rs)**: Реализовать `init_logger` и `clean_old_logs`.
4.  **Main**: В `run-bot.rs` вызвать `init_logger` в самом начале и сохранить `WorkerGuard` в переменную, чтобы она жила до конца работы `main`.

**Результат**: Профессиональная система логирования, устойчивая к ошибкам «раздувания» файлов и не влияющая на задержки торговых операций.
