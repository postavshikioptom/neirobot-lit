# Задача 139: Неблокирующее логирование и мониторинг потерь (v2.0)

## 1. Контекст и цели
Логгер в HFT должен быть полностью отделен от основного потока. Мы используем ограниченную (bounded) очередь: если диск не успевает, логгер отбрасывает сообщения (lossy), сохраняя стабильность `tick-to-trade`.
*   **Zero-Stall**: Ни при каких условиях `tracing::info!` не должен блокировать вызывающий поток.
*   **Dropped Metrics**: Учет количества потерянных сообщений.
*   **Panic Safety**: Непосредственный вывод в `stderr` при крахе системы.

## 2. Реализация в `src/utils/logger.rs`
Используем `non_blocking_with_capacity` для создания ограниченной очереди и атомарный счетчик для метрик.

```rust
// В [./src/utils/logger.rs](./src/utils/logger.rs)
use std::sync::atomic::{AtomicU64, Ordering};
use tracing_appender::non_blocking::{NonBlocking, WorkerGuard};

// Глобальный счетчик отброшенных логов (если библиотека поддерживает кастомный callback или через обертку)
pub static DROPPED_LOGS: AtomicU64 = AtomicU64::new(0);

pub fn init_logger(config: &LogConfig, bot_path: &Path) -> WorkerGuard {
    let log_dir = bot_path.join("logs");
    let file_appender = RollingFileAppender::new(Rotation::DAILY, log_dir, "bot.log");

    // Используем non_blocking с фиксированной емкостью очереди
    // При переполнении (queue full) старые сообщения будут отбрасываться
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
    
    // ВАЖНО: В v0.2 capacity настраивается через NonBlockingBuilder, 
    // но мы должны убедиться в использовании ограниченной очереди
    let (non_blocking, guard) = tracing_appender::non_blocking_with_capacity(
        file_appender, 
        config.logger_queue_size // из BotConfig, напр. 10000
    );

    // Установка Panic Hook: только eprintln! для надежности
    std::panic::set_hook(Box::new(|panic_info| {
        eprintln!("!!! FATAL PANIC !!!");
        eprintln!("{}", panic_info);
        // Мы не используем sleep(100ms), полагаясь на то, что WorkerGuard
        // сделает flush при завершении потока, если это возможно.
    }));

    // Инициализация сабскрайбера (см. задачу 134)
    // ...
    
    guard
}
```

## 3. Спорные моменты и корректировки (Grok + Zencoder)

*   **Deprecated Builder**: Согласен с Grok. Переходим на `non_blocking_with_capacity`. Это более прямой и современный путь настройки буфера в `tracing-appender`.
*   **No Sleep in Hook**: Согласен. В условиях паники (panic unwind) любое ожидание может привести к зависанию процесса (deadlock) или некорректному завершению со стороны ОС. `eprintln!` — самый надежный способ выдать критическую информацию.
*   **Dropped Logs**: К сожалению, стандартный `tracing-appender` не предоставляет прямой счетчик `dropped`. Gemini должен реализовать простую фоновую задачу (задача 135/133), которая периодически проверяет состояние системы. Если же требуется точный счетчик, придется написать тонкую обертку над `Writer`.
*   **Queue Size**: В [./src/config/types.rs](./src/config/types.rs) добавляем `logger_queue_size: usize`. Для HFT на ликвидных парах (BTC/ETH) рекомендуем 10 000–50 000, чтобы сглаживать пики во время резких движений рынка.

## 4. Мониторинг потерь (в `health.rs`)
В эндпоинт `/health` (задача 135) стоит добавить информацию о том, насколько стабильно работает система логирования.

```rust
// В [./src/monitoring/health.rs](./src/monitoring/health.rs)
// В будущем: если реализована обертка, отдаем state.dropped_logs.load(Ordering::Relaxed)
```

## 5. Инструкции для Gemini (Coder AI):
1.  **Cargo.toml**: Проверить версию `tracing-appender = "0.2.3"`.
2.  **[./src/utils/logger.rs](./src/utils/logger.rs)**: Реализовать `init_logger` с использованием `non_blocking_with_capacity`.
3.  **Panic Hook**: Установить обработчик, выводящий данные в `stderr` через `eprintln!`.
4.  **Config**: Пробросить `logger_queue_size` через `LogConfig`.

**Результат**: Система логирования с гарантированным отсутствием блокировок основного потока, защищенная от «раздувания» памяти и обеспечивающая мгновенный вывод данных при критических сбоях.
