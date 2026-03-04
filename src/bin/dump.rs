use clap::Parser;
use std::path::{Path, PathBuf};
use tracing::{info, error};
use anyhow::{Result, Context, bail};
use std::fs;
use std::io::{BufRead, BufReader};

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

// Импорты из библиотеки проекта
use neirobot_lit::config::loader::load_full_config_with_validation;
use neirobot_lit::utils::logger::init_logger;

#[derive(Parser, Debug)]
#[command(author, version, about = "Neirobot LiT - Data Dumper CLI", long_about = None)]
struct Args {
    /// Символ для выгрузки (опционально, например: BTCUSDT). 
    /// Если не указан, используются символы из bots/symbols.txt
    symbol: Option<String>,

    /// Продолжительность работы в часах (поддерживается дробная часть, например: 0.5)
    #[arg(long, default_value_t = 1.0)]
    hours: f32,

    /// Путь к конфигу бота (по умолчанию: bots/SYMBOL/config.toml)
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Директория для сохранения Parquet (по умолчанию: bots/SYMBOL/data/raw/)
    #[arg(short, long)]
    output_dir: Option<PathBuf>,
}

/// Читает список символов из файла bots/symbols.txt
fn read_symbols_from_file(path: &Path) -> Result<Vec<String>> {
    if !path.exists() {
        bail!("Symbols file not found at: {}", path.display());
    }

    let file = fs::File::open(path).context("Failed to open symbols.txt")?;
    let reader = BufReader::new(file);
    let mut symbols = Vec::new();

    for line in reader.lines() {
        let line = line?.trim().to_string();
        if !line.is_empty() && !line.starts_with('#') {
            symbols.push(line);
        }
    }

    Ok(symbols)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Загрузка переменных окружения из .env
    dotenvy::dotenv().ok();
    
    // Парсинг аргументов командной строки
    let args = Args::parse();

    // 1. Определяем список символов для выгрузки
    let symbols = if let Some(s) = args.symbol {
        vec![s]
    } else {
        read_symbols_from_file(Path::new("bots/symbols.txt"))?
    };

    if symbols.is_empty() {
        bail!("No symbols provided and bots/symbols.txt is empty");
    }

    // 2. Инициализируем логгер (используем конфиг первого символа или дефолтный)
    // Для простоты берем конфиг первого символа для настройки уровня логирования
    let first_symbol = &symbols[0];
    let first_config_path = args.config.clone().unwrap_or_else(|| {
        PathBuf::from("bots").join(first_symbol).join("config.toml")
    });
    // Загружаем конфиг для первой монеты (чтобы прочитать интервалы и т.д.)
    // Выключаем торговые проверки (модель, пороги), так как для дампа они не нужна
    let first_full_config = load_full_config_with_validation(Path::new("."), &first_config_path, false)?;

    // Путь для логов: если один символ - в его папку, если много - в общую logs/dump
    let log_path = if symbols.len() == 1 {
        PathBuf::from("bots").join(first_symbol)
    } else {
        PathBuf::from("logs/dump")
    };

    let secrets = vec![
        std::env::var("BYBIT_API_KEY").unwrap_or_default(),
        std::env::var("BYBIT_API_SECRET").unwrap_or_default(),
    ];

    let _log_guards = init_logger(&first_full_config.logging, &log_path, secrets)?;

    info!(
        "Starting dumper for {} symbols. Duration: {} hours.", 
        symbols.len(), args.hours
    );

    // 3. Запуск пайплайнов параллельно
    let mut handles = Vec::new();
    let hours = args.hours;

    for symbol in symbols {
        let config_path = args.config.clone().unwrap_or_else(|| {
            PathBuf::from("bots").join(&symbol).join("config.toml")
        });

        // Загружаем конфиг для каждой монеты (могут быть разные интервалы и т.д.)
        // Выключаем торговые проверки (модель, пороги), так как для дампа они не нужны
        let full_config = match load_full_config_with_validation(Path::new("."), &config_path, false) {
            Ok(cfg) => cfg,
            Err(e) => {
                error!("Failed to load config for {}: {}", symbol, e);
                continue;
            }
        };

        info!("Spawning dump task for {}", symbol);

        let handle = tokio::spawn(async move {
            let duration = std::time::Duration::from_secs_f32(hours * 3600.0);
            match tokio::time::timeout(duration, neirobot_lit::data::snapshot::run_snapshot_pipeline(full_config)).await {
                Ok(res) => res.with_context(|| format!("Error in pipeline for {}", symbol)),
                Err(_) => {
                    info!("Duration reached for {}. Stopping.", symbol);
                    Ok(())
                }
            }
        });
        handles.push(handle);
    }

    // Ожидаем завершения всех задач
    for handle in handles {
        if let Err(e) = handle.await? {
            error!("Dumper task error: {:?}", e);
        }
    }

    info!("All dump tasks completed.");
    Ok(())
}
