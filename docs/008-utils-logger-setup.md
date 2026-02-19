# 008 - Utils Logger Setup

Цель задачи: Настроить систему логирования на базе tracing с поддержкой ротации файлов, неблокирующей записи и гибкого форматирования. Логи должны быть изолированы для каждого бота и храниться в bots/SYMBOL/logs/.

Файлы для изменения/создания:

src/utils/mod.rs
src/utils/logger.rs (создать)
Инструкции для Gemini:

src/utils/logger.rs: Реализовать инициализацию логгера, учитывая параметры LoggingConfig. Важно: функция должна возвращать WorkerGuard, который нужно удерживать в main.rs, чтобы логи успевали записываться в файл.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
use crate::config::types::LoggingConfig;

pub fn init_logger(symbol: &str, config: &LoggingConfig) -> Result<WorkerGuard> {
    let log_dir = PathBuf::from("bots").join(symbol).join("logs");
    std::fs::create_dir_all(&log_dir).context("Failed to create log directory")?;

    // Настройка ротации (по времени: каждый день)
    let file_appender = tracing_appender::rolling::daily(&log_dir, &config.file_name);
    let (non_blocking_file, guard) = tracing_appender::non_blocking(file_appender);

    // UTC Таймер с миллисекундами
    let timer = fmt::time::UtcTime::rfc_3339();

    // Слой для файла (всегда без цветов)
    let file_layer = fmt::layer()
        .with_writer(non_blocking_file)
        .with_ansi(false)
        .with_timer(timer.clone())
        .with_target(true);

    // Слой для консоли
    let console_layer = fmt::layer()
        .with_timer(timer)
        .with_target(false);

    // Выбор формата (json, compact, pretty)
    let registry = tracing_subscriber::registry()
        .with(EnvFilter::new(&config.level))
        .with(file_layer);

    match config.format.as_str() {
        "json" => registry.with(console_layer.json()).init(),
        "compact" => registry.with(console_layer.compact()).init(),
        _ => registry.with(console_layer.pretty()).init(),
    };

    Ok(guard)
}
src/utils/mod.rs:

pub mod logger;
Технические требования:

Изоляция: Путь к логам строго bots/{SYMBOL}/logs/.
Производительность: Использовать non_blocking для файлового вывода.
Безопасность: WorkerGuard должен возвращаться в вызывающий код, иначе запись в файл прервется.
Формат: Поддержка JSON для консоли (удобно для лог-менеджеров) и человекочитаемого формата для файла.
Время: Строго UTC с миллисекундами (ISO 8601).
Почему это важно: Ротация предотвращает переполнение диска. Использование non_blocking гарантирует, что медленные операции записи на диск не затормозят критический путь исполнения торговой логики.