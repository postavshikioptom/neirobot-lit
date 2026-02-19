# 016 - Bin Dump CLI

Цель задачи: Создать исполняемый файл src/bin/dump.rs, который обеспечит CLI-интерфейс для запуска процесса выгрузки данных. Бинарник должен принимать символ как позиционный аргумент, загружать соответствующий конфиг, инициализировать логгер и подготавливать пути для сохранения Parquet-файлов.

Файлы для изменения/создания:

src/bin/dump.rs (создать)
Инструкции для Gemini:

src/bin/dump.rs: Реализовать парсинг аргументов с использованием clap и инициализацию окружения.

use clap::Parser;
use std::path::{Path, PathBuf};
use tracing::info;
use anyhow::Result;

// Импорты из нашего проекта
use crate::config::loader::load_full_config;
use crate::utils::logger::init_logger;

#[derive(Parser, Debug)]
#[command(author, version, about = "Neirobot LiT - Data Dumper CLI", long_about = None)]
struct Args {
    /// Символ для выгрузки (позиционный аргумент, например: BTCUSDT)
    symbol: String,

    /// Продолжительность работы в днях
    #[arg(short, long, default_value_t = 1)]
    days: u32,

    /// Путь к конфигу бота (по умолчанию: bots/SYMBOL/config.toml)
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Директория для сохранения Parquet (по умолчанию: bots/SYMBOL/data/raw/)
    #[arg(short, long)]
    output_dir: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // 1. Формируем пути по умолчанию
    let config_path = args.config.unwrap_or_else(|| {
        PathBuf::from("bots").join(&args.symbol).join("config.toml")
    });
    
    let output_dir = args.output_dir.unwrap_or_else(|| {
        PathBuf::from("bots").join(&args.symbol).join("data").join("raw")
    });

    // 2. Загружаем полную конфигурацию (используем "." как корень проекта)
    let full_config = load_full_config(Path::new("."), &config_path)?;

    // 3. Инициализируем изолированный логгер для этого символа
    // Сохраняем guard, чтобы логи не пропали (см. задачу 008)
    let _log_guard = init_logger(&args.symbol, &full_config.logging)?;

    info!(
        "Starting dumper for {}. Duration: {} days. Output: {:?}", 
        args.symbol, args.days, output_dir
    );

    // TODO: В задачах 017-020 здесь появится инициализация WebSocket 
    // и цикл записи данных в Parquet.

    Ok(())
}
Технические требования:

Аргументы: symbol должен быть позиционным (cargo run --bin dump -- BTCUSDT).
Конфигурация: Обязательная загрузка через load_full_config. Если файл не найден — программа должна завершиться с ошибкой.
Логирование: Использовать init_logger из src/utils/logger.rs. Обязательно удерживать _log_guard в main.
Пути: Автоматически создавать output_dir (через fs::create_dir_all в будущих задачах или здесь).
Почему это важно: Это "входная дверь" в процесс сбора данных. Мы объединяем здесь все предыдущие наработки: парсинг путей, типизированный конфиг и систему логирования. Позиционный аргумент symbol делает запуск бота максимально простым и удобным для автоматизации.