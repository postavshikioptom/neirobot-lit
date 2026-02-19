use clap::Parser;
use std::path::{Path, PathBuf};
use tracing::info;
use anyhow::Result;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

// Импорты из библиотеки проекта
use neirobot_lit::config::loader::load_full_config;
use neirobot_lit::utils::logger::init_logger;

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
    // Загрузка переменных окружения из .env
    dotenvy::dotenv().ok();
    
    // Парсинг аргументов командной строки
    let args = Args::parse();

    // 1. Формируем пути по умолчанию, если они не заданы явно
    let config_path = args.config.unwrap_or_else(|| {
        PathBuf::from("bots").join(&args.symbol).join("config.toml")
    });
    
    let output_dir = args.output_dir.unwrap_or_else(|| {
        PathBuf::from("bots").join(&args.symbol).join("data").join("raw")
    });

    // 2. Загружаем полную конфигурацию (мердж global + exchange + bot)
    let full_config = load_full_config(Path::new("."), &config_path)?;

    // Формируем путь к папке бота
    let bot_path = PathBuf::from("bots").join(&args.symbol);

    // Загружаем секреты для маскирования (если доступны)
    let secrets = vec![
        std::env::var("BYBIT_API_KEY").unwrap_or_default(),
        std::env::var("BYBIT_API_SECRET").unwrap_or_default(),
    ];

    // 3. Инициализируем изолированный логгер для конкретного символа с маскированием секретов
    let _log_guard = init_logger(&full_config.logging, &bot_path, secrets)?;

    info!(
        "Starting dumper for {}. Duration: {} days. Output: {:?}", 
        args.symbol, args.days, output_dir
    );

    // TODO: В задачах 017-020 здесь появится инициализация WebSocket 
    // и цикл записи данных в Parquet.

    Ok(())
}
