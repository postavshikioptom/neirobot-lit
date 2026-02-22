use anyhow::{Context, Result};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{atomic::AtomicU64, Arc};
use tokio::sync::mpsc;
use tracing_appender::non_blocking::{NonBlockingBuilder, WorkerGuard};
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};
use crate::config::types::LoggingConfig;
use crate::utils::telegram::TelegramWorker;

/// Структура для передачи данных логирования метрик исполнения через канал (Задача 202)
#[derive(Debug, Clone)]
pub struct ExecutionQualityLog {
    pub timestamp_ms: u64,
    pub order_id: String,
    pub internal_lat_us: u64,
    pub network_lat_us: u64,
    pub fill_rate: f64,
    pub is_cancelled: bool,
    pub bot_path: PathBuf,
}

/// Структура для логирования контекста стакана при выставлении ордера (Задача 203)
#[derive(Debug, Clone)]
pub struct FillRateContextLog {
    pub timestamp_ms: u64,
    pub order_id: String,
    pub level_total_vol: f64,
    pub imbalance_5l: f64,
    pub order_size: f64,
    pub fill_duration_us: u64,
    pub bot_path: PathBuf,
}

/// Структура для логирования влияния сделок на Mid-Price (Задача 204)
#[derive(Debug, Clone)]
pub struct MarketImpactLog {
    pub timestamp_ms: u64,
    pub order_id: String,
    pub fill_id: u32,
    pub side: String,
    pub fill_size: f64,
    pub mid_before: f64,
    pub mid_at_fill: f64,
    pub bot_path: PathBuf,
}

/// Глобальный счетчик отброшенных логов
/// Примечание: tracing-appender не предоставляет прямой API для отслеживания потерь,
/// поэтому этот счетчик служит для будущей интеграции с кастомной оберткой
pub static DROPPED_LOGS: AtomicU64 = AtomicU64::new(0);

/// Writer который маскирует секреты в логах
/// 
/// Оборачивает любой Writer и заменяет все вхождения секретов на [MASKED]
/// перед записью в underlying writer
struct MaskingWriter<W> {
    inner: W,
    secrets: Arc<Vec<String>>,
}

impl<W: Write> Write for MaskingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        // Конвертируем буфер в строку
        let text = String::from_utf8_lossy(buf);
        let mut masked = text.to_string();
        
        // Заменяем все секреты на [MASKED]
        for secret in self.secrets.iter() {
            if !secret.is_empty() {
                masked = masked.replace(secret, "[MASKED]");
            }
        }
        
        // Записываем замаскированный текст
        self.inner.write_all(masked.as_bytes())?;
        Ok(buf.len()) // Возвращаем оригинальный размер для корректной работы tracing
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// MakeWriter который создает MaskingWriter для каждого лог-события
/// 
/// Используется с tracing-subscriber для автоматического маскирования секретов
struct MaskingMakeWriter<M> {
    inner: M,
    secrets: Arc<Vec<String>>,
}

impl<M> MaskingMakeWriter<M> {
    fn new(inner: M, secrets: Vec<String>) -> Self {
        Self {
            inner,
            secrets: Arc::new(secrets),
        }
    }
}

impl<'a, M> tracing_subscriber::fmt::MakeWriter<'a> for MaskingMakeWriter<M>
where
    M: tracing_subscriber::fmt::MakeWriter<'a>,
{
    type Writer = MaskingWriter<M::Writer>;

    fn make_writer(&'a self) -> Self::Writer {
        MaskingWriter {
            inner: self.inner.make_writer(),
            secrets: self.secrets.clone(),
        }
    }
}

/// Кастомный Layer для отправки критических событий в Telegram
/// 
/// Использует неблокирующую отправку через bounded канал для Zero-Stall гарантии
pub struct TelegramLayer {
    tx: mpsc::Sender<String>,
    level: tracing::Level,
}

impl TelegramLayer {
    /// Создает новый TelegramLayer
    /// 
    /// # Параметры
    /// - `tx`: Sender для отправки сообщений в Telegram воркер
    /// - `level`: Минимальный уровень логирования для отправки в Telegram
    pub fn new(tx: mpsc::Sender<String>, level: tracing::Level) -> Self {
        Self { tx, level }
    }
}

impl<S> Layer<S> for TelegramLayer
where
    S: tracing::Subscriber,
{
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        // Проверяем уровень события
        if event.metadata().level() <= &self.level {
            // Формируем сообщение с эмодзи для визуального выделения
            let emoji = match *event.metadata().level() {
                tracing::Level::ERROR => "🚨",
                tracing::Level::WARN => "⚠️",
                _ => "ℹ️",
            };
            
            let msg = format!(
                "{} *{}*: Critical event detected",
                emoji,
                event.metadata().level()
            );
            
            // Неблокирующая отправка: если канал полон, алерт отбрасывается
            // Это критично для Zero-Stall гарантии в HFT системе
            let _ = self.tx.try_send(msg);
        }
    }
}

/// Инициализация логгера с асинхронной записью и фоновой очисткой
/// 
/// # Параметры
/// - `config`: Конфигурация логирования
/// - `bot_path`: Путь к директории бота
/// - `secrets`: Список секретов для маскирования в логах (API ключи и т.д.)
pub fn init_logger(config: &LoggingConfig, bot_path: &Path, secrets: Vec<String>) -> Result<WorkerGuard> {
    let log_dir = bot_path.join("logs");
    std::fs::create_dir_all(&log_dir).context("Failed to create log directory")?;

    // 1. Определение типа ротации (hourly или daily)
    let rotation = match config.rotation.as_str() {
        "hourly" => Rotation::HOURLY,
        _ => Rotation::DAILY,
    };

    // 2. Файловый аппендер с ротацией по времени
    let file_appender = RollingFileAppender::new(rotation, &log_dir, &config.file_name);

    // 3. Настройка Non-blocking с Lossy стратегией (критично для HFT)
    // Используем logger_queue_size из конфига вместо хардкода
    // Если очередь переполнена, старые логи отбрасываются, чтобы не тормозить main thread
    let (non_blocking, guard) = NonBlockingBuilder::default()
        .buffered_lines_limit(config.logger_queue_size)
        .lossy(true) // Drop logs if full - критично для HFT!
        .finish(file_appender);

    // 4. Установка Panic Hook для надежного вывода критических ошибок
    // Используем только eprintln! для гарантированного вывода в stderr
    std::panic::set_hook(Box::new(|panic_info| {
        eprintln!("!!! FATAL PANIC !!!");
        eprintln!("{}", panic_info);
        // Не используем sleep, полагаясь на WorkerGuard для flush при завершении
    }));

    // 5. UTC Таймер с миллисекундами
    let timer = fmt::time::UtcTime::rfc_3339();

    // 6. Слой для файла с маскированием секретов (без ANSI, с target для отладки многопоточного кода)
    let masking_writer = MaskingMakeWriter::new(non_blocking, secrets.clone());
    let file_layer = fmt::layer()
        .with_writer(masking_writer)
        .with_ansi(false)
        .with_timer(timer.clone())
        .with_thread_names(true)
        .with_target(true)
        .with_filter(EnvFilter::new(&config.level));

    // 7. Инициализация Telegram Layer (опционально)
    let registry = tracing_subscriber::registry().with(file_layer);
    
    // Попытка загрузить Telegram credentials из переменных окружения
    let telegram_layer = match (
        std::env::var("TELEGRAM_TOKEN").ok(),
        std::env::var("TELEGRAM_CHAT_ID").ok(),
    ) {
        (Some(token), Some(chat_id)) => {
            // Создаем bounded канал для Telegram сообщений (размер 100)
            let (tx, rx) = mpsc::channel::<String>(100);
            
            // Запускаем Telegram воркер в фоновом режиме
            let worker = TelegramWorker::new(token, chat_id);
            tokio::spawn(async move {
                worker.run(rx, 1000).await; // 1 сообщение в секунду
            });
            
            // Создаем Layer для ERROR уровня
            Some(TelegramLayer::new(tx, tracing::Level::ERROR))
        }
        _ => {
            tracing::info!("Telegram notifications disabled: TELEGRAM_TOKEN or TELEGRAM_CHAT_ID not set");
            None
        }
    };

    // 8. Сборка слоев с опциональным Telegram Layer
    let registry = if let Some(tg_layer) = telegram_layer {
        registry.with(tg_layer)
    } else {
        registry
    };

    // 9. Опциональный консольный слой с маскированием секретов и выбором формата
    if config.console_enabled {
        let console_masking_writer = MaskingMakeWriter::new(std::io::stdout, secrets);
        let console_layer = fmt::layer()
            .with_writer(console_masking_writer)
            .with_ansi(true)
            .with_timer(timer)
            .with_thread_names(true)
            .with_target(true);
        match config.format.as_str() {
            "json" => registry.with(console_layer.json().with_filter(EnvFilter::new(&config.level))).init(),
            "compact" => registry.with(console_layer.compact().with_filter(EnvFilter::new(&config.level))).init(),
            _ => registry.with(console_layer.pretty().with_filter(EnvFilter::new(&config.level))).init(),
        };
    } else {
        registry.init();
    }

    let max_files = config.max_files;
    let log_dir_clone = log_dir.clone();
    tokio::spawn(async move {
        clean_old_logs(log_dir_clone, max_files).await;
    });

    Ok(guard)
}

async fn clean_old_logs(dir: PathBuf, max_files: usize) {
    loop {
        if let Ok(entries) = std::fs::read_dir(&dir) {
            let mut files: Vec<_> = entries
                .filter_map(|e| e.ok())
                .filter_map(|e| {
                    let meta = e.metadata().ok()?;
                    if meta.is_file() { 
                        Some((meta.modified().ok()?, e.path())) 
                    } else { 
                        None 
                    }
                })
                .collect();

            files.sort_by_key(|&(t, _)| t);

            if files.len() > max_files {
                let to_remove = files.len() - max_files;
                for i in 0..to_remove {
                    if let Err(e) = std::fs::remove_file(&files[i].1) {
                        tracing::warn!("Failed to remove old log file {:?}: {}", files[i].1, e);
                    } else {
                        tracing::info!("Removed old log file: {:?}", files[i].1);
                    }
                }
            }
        }
        tokio::time::sleep(tokio::time::Duration::from_secs(3600)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_masking_writer_masks_secrets() {
        let secrets = vec![
            "secret_api_key_12345".to_string(),
            "secret_token_67890".to_string(),
        ];
        
        let mut buffer = Vec::new();
        let mut writer = MaskingWriter {
            inner: &mut buffer,
            secrets: Arc::new(secrets),
        };

        // Пишем текст содержащий секреты
        let test_text = "API Key: secret_api_key_12345, Token: secret_token_67890";
        writer.write_all(test_text.as_bytes()).unwrap();

        let result = String::from_utf8(buffer).unwrap();
        
        // Проверяем что секреты замаскированы
        assert_eq!(result, "API Key: [MASKED], Token: [MASKED]");
        assert!(!result.contains("secret_api_key_12345"));
        assert!(!result.contains("secret_token_67890"));
    }

    #[test]
    fn test_masking_writer_handles_empty_secrets() {
        let secrets = vec!["".to_string()];
        
        let mut buffer = Vec::new();
        let mut writer = MaskingWriter {
            inner: &mut buffer,
            secrets: Arc::new(secrets),
        };

        let test_text = "No secrets here";
        writer.write_all(test_text.as_bytes()).unwrap();

        let result = String::from_utf8(buffer).unwrap();
        assert_eq!(result, "No secrets here");
    }

    #[test]
    fn test_masking_writer_handles_no_secrets_in_text() {
        let secrets = vec!["secret123".to_string()];
        
        let mut buffer = Vec::new();
        let mut writer = MaskingWriter {
            inner: &mut buffer,
            secrets: Arc::new(secrets),
        };

        let test_text = "This text has no secrets";
        writer.write_all(test_text.as_bytes()).unwrap();

        let result = String::from_utf8(buffer).unwrap();
        assert_eq!(result, "This text has no secrets");
    }
}

/// Логирование исполнения сделки для анализа slippage (Задача 201)
/// 
/// Записывает данные о сделке в bots/SYMBOL/logs/slippage.csv
/// 
/// # Параметры
/// - `bot_path`: Путь к директории бота (bots/SYMBOL)
/// - `timestamp_utc`: Временная метка в UTC (миллисекунды)
/// - `signal_price`: Mid price в момент генерации сигнала
/// - `fill_price`: Средневзвешенная цена исполнения
/// - `slippage_bps`: Проскальзывание в базисных пунктах
/// - `latency_ms`: Задержка от генерации сигнала до исполнения (миллисекунды)
/// - `spread_bps`: Спред в момент исполнения (bid-ask spread в базисных пунктах)
pub fn log_trade_execution(
    bot_path: &Path,
    timestamp_utc: u64,
    signal_price: f64,
    fill_price: f64,
    slippage_bps: f64,
    latency_ms: u64,
    spread_bps: f64,
) -> Result<()> {
    let log_dir = bot_path.join("logs");
    std::fs::create_dir_all(&log_dir).context("Failed to create log directory")?;
    
    let csv_path = log_dir.join("slippage.csv");
    let file_exists = csv_path.exists();
    
    // Открываем файл в режиме append
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .append(true)
        .open(&csv_path)
        .context("Failed to open slippage.csv")?;
    
    let mut wtr = csv::WriterBuilder::new()
        .has_headers(!file_exists) // Записываем заголовки только если файл новый
        .from_writer(file);
    
    // Если файл новый, записываем заголовки
    if !file_exists {
        wtr.write_record(&["timestamp_utc", "signal_price", "fill_price", "slippage_bps", "latency_ms", "spread_bps"])
            .context("Failed to write CSV headers")?;
    }
    
    // Записываем данные сделки
    wtr.write_record(&[
        timestamp_utc.to_string(),
        signal_price.to_string(),
        fill_price.to_string(),
        slippage_bps.to_string(),
        latency_ms.to_string(),
        spread_bps.to_string(),
    ])
    .context("Failed to write trade record")?;
    
    wtr.flush().context("Failed to flush CSV writer")?;
    
    Ok(())
}

/// Логирование метрик качества исполнения (Задача 202)
/// Записывает данные в execution_quality.csv для анализа влияния задержки на Fill Rate
/// Эта функция предназначена для вызова из фонового worker потока
pub fn log_execution_quality(log: &ExecutionQualityLog) -> Result<()> {
    let log_dir = log.bot_path.join("logs");
    std::fs::create_dir_all(&log_dir).context("Failed to create log directory")?;
    
    let csv_path = log_dir.join("execution_quality.csv");
    let file_exists = csv_path.exists();
    
    // Открываем файл в режиме append
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .append(true)
        .open(&csv_path)
        .context("Failed to open execution_quality.csv")?;
    
    let mut wtr = csv::WriterBuilder::new()
        .has_headers(!file_exists)
        .from_writer(file);
    
    // Если файл новый, записываем заголовки
    if !file_exists {
        wtr.write_record(&[
            "timestamp",
            "order_id",
            "internal_lat_us",
            "network_lat_us",
            "fill_rate",
            "is_cancelled",
        ])
        .context("Failed to write CSV headers")?;
    }
    
    // Записываем данные метрик исполнения
    wtr.write_record(&[
        log.timestamp_ms.to_string(),
        log.order_id.clone(),
        log.internal_lat_us.to_string(),
        log.network_lat_us.to_string(),
        log.fill_rate.to_string(),
        log.is_cancelled.to_string(),
    ])
    .context("Failed to write execution quality record")?;
    
    wtr.flush().context("Failed to flush CSV writer")?;
    
    Ok(())
}


/// Запускает фоновый worker для обработки логов метрик исполнения (Задача 202)
/// Возвращает sender для отправки логов в worker
pub fn spawn_execution_quality_logger() -> mpsc::Sender<ExecutionQualityLog> {
    let (tx, mut rx) = mpsc::channel::<ExecutionQualityLog>(1000);
    
    tokio::spawn(async move {
        while let Some(log) = rx.recv().await {
            // Логируем метрики в фоновом потоке, чтобы не блокировать основной цикл
            if let Err(e) = log_execution_quality(&log) {
                tracing::warn!("Failed to log execution quality: {}", e);
            }
        }
    });
    
    tx
}

/// Логирование контекста стакана для анализа Fill Rate (Задача 203)
/// Записывает данные в order_context.csv для анализа влияния состояния LOB на исполнение
pub fn log_fill_rate_context(log: &FillRateContextLog) -> Result<()> {
    let log_dir = log.bot_path.join("logs");
    std::fs::create_dir_all(&log_dir).context("Failed to create log directory")?;
    
    let csv_path = log_dir.join("order_context.csv");
    let file_exists = csv_path.exists();
    
    // Открываем файл в режиме append
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .append(true)
        .open(&csv_path)
        .context("Failed to open order_context.csv")?;
    
    let mut wtr = csv::WriterBuilder::new()
        .has_headers(!file_exists)
        .from_writer(file);
    
    // Если файл новый, записываем заголовки
    if !file_exists {
        wtr.write_record(&[
            "timestamp",
            "order_id",
            "level_total_vol",
            "imbalance_5l",
            "order_size",
            "fill_duration_us",
        ])
        .context("Failed to write CSV headers")?;
    }
    
    // Записываем данные контекста стакана
    wtr.write_record(&[
        log.timestamp_ms.to_string(),
        log.order_id.clone(),
        log.level_total_vol.to_string(),
        log.imbalance_5l.to_string(),
        log.order_size.to_string(),
        log.fill_duration_us.to_string(),
    ])
    .context("Failed to write fill rate context record")?;
    
    wtr.flush().context("Failed to flush CSV writer")?;
    
    Ok(())
}

/// Запускает фоновый worker для обработки логов контекста Fill Rate (Задача 203)
/// Возвращает sender для отправки логов в worker
pub fn spawn_fill_rate_context_logger() -> mpsc::Sender<FillRateContextLog> {
    let (tx, mut rx) = mpsc::channel::<FillRateContextLog>(1000);
    
    tokio::spawn(async move {
        while let Some(log) = rx.recv().await {
            // Логируем контекст в фоновом потоке, чтобы не блокировать основной цикл
            if let Err(e) = log_fill_rate_context(&log) {
                tracing::warn!("Failed to log fill rate context: {}", e);
            }
        }
    });
    
    tx
}

/// Логирование влияния сделок на Mid-Price (Задача 204)
/// Записывает данные в market_impact.csv для анализа влияния наших ордеров на цену
pub fn log_market_impact(log: &MarketImpactLog) -> Result<()> {
    let log_dir = log.bot_path.join("logs");
    std::fs::create_dir_all(&log_dir).context("Failed to create log directory")?;
    
    let csv_path = log_dir.join("market_impact.csv");
    let file_exists = csv_path.exists();
    
    // Открываем файл в режиме append
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .append(true)
        .open(&csv_path)
        .context("Failed to open market_impact.csv")?;
    
    let mut wtr = csv::WriterBuilder::new()
        .has_headers(!file_exists)
        .from_writer(file);
    
    // Если файл новый, записываем заголовки
    if !file_exists {
        wtr.write_record(&[
            "timestamp",
            "order_id",
            "fill_id",
            "side",
            "fill_size",
            "mid_before",
            "mid_at_fill",
        ])
        .context("Failed to write CSV headers")?;
    }
    
    // Записываем данные влияния на цену
    wtr.write_record(&[
        log.timestamp_ms.to_string(),
        log.order_id.clone(),
        log.fill_id.to_string(),
        log.side.clone(),
        log.fill_size.to_string(),
        log.mid_before.to_string(),
        log.mid_at_fill.to_string(),
    ])
    .context("Failed to write market impact record")?;
    
    wtr.flush().context("Failed to flush CSV writer")?;
    
    Ok(())
}

/// Запускает фоновый worker для обработки логов влияния на цену (Задача 204)
/// Возвращает sender для отправки логов в worker
pub fn spawn_market_impact_logger() -> mpsc::Sender<MarketImpactLog> {
    let (tx, mut rx) = mpsc::channel::<MarketImpactLog>(1000);
    
    tokio::spawn(async move {
        while let Some(log) = rx.recv().await {
            // Логируем влияние на цену в фоновом потоке, чтобы не блокировать основной цикл
            if let Err(e) = log_market_impact(&log) {
                tracing::warn!("Failed to log market impact: {}", e);
            }
        }
    });
    
    tx
}

/// Структура для логирования данных equity (Задача 205)
#[derive(Debug, Clone)]
pub struct EquityLog {
    pub timestamp_ms: u64,
    pub rest_equity: f64,
    pub available_balance: f64,
    pub rest_margin: f64,
    pub local_unrealized_pnl: f64,
    pub total_pnl_delta: f64,
    pub bot_path: PathBuf,
}

/// Логирование данных equity для ежедневных PnL-отчетов (Задача 205)
/// Записывает данные в equity.csv для анализа производительности бота
pub fn log_equity(log: &EquityLog) -> Result<()> {
    let log_dir = log.bot_path.join("logs");
    std::fs::create_dir_all(&log_dir).context("Failed to create log directory")?;
    
    let csv_path = log_dir.join("equity.csv");
    let file_exists = csv_path.exists();
    
    // Открываем файл в режиме append
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .append(true)
        .open(&csv_path)
        .context("Failed to open equity.csv")?;
    
    let mut wtr = csv::WriterBuilder::new()
        .has_headers(!file_exists)
        .from_writer(file);
    
    // Если файл новый, записываем заголовки
    if !file_exists {
        wtr.write_record(&[
            "timestamp",
            "rest_equity",
            "available_balance",
            "rest_margin",
            "local_unrealized_pnl",
            "total_pnl_delta",
        ])
        .context("Failed to write CSV headers")?;
    }
    
    // Записываем данные equity
    wtr.write_record(&[
        log.timestamp_ms.to_string(),
        log.rest_equity.to_string(),
        log.available_balance.to_string(),
        log.rest_margin.to_string(),
        log.local_unrealized_pnl.to_string(),
        log.total_pnl_delta.to_string(),
    ])
    .context("Failed to write equity record")?;
    
    wtr.flush().context("Failed to flush CSV writer")?;
    
    Ok(())
}

/// Запускает фоновый worker для обработки логов equity (Задача 205)
/// Возвращает sender для отправки логов в worker
pub fn spawn_equity_logger() -> mpsc::Sender<EquityLog> {
    let (tx, mut rx) = mpsc::channel::<EquityLog>(1000);
    
    tokio::spawn(async move {
        while let Some(log) = rx.recv().await {
            // Логируем equity в фоновом потоке, чтобы не блокировать основной цикл
            if let Err(e) = log_equity(&log) {
                tracing::warn!("Failed to log equity: {}", e);
            }
        }
    });
    
    tx
}
