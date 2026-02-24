use clap::Parser;
use std::path::{Path, PathBuf};
use std::fs::{self, File};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{info, warn, error, debug};
use anyhow::{Result, Context, bail};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use polars::prelude::*;
use std::panic;
use std::process;

#[cfg(all(windows, not(feature = "jemalloc")))]
use mimalloc::MiMalloc;

#[cfg(all(windows, not(feature = "jemalloc")))]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

// Задача 230: Опциональный jemalloc аллокатор для лучшего контроля фрагментации
#[cfg(feature = "jemalloc")]
use tikv_jemallocator::Jemalloc;

#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

// Импорты из нашего проекта
use neirobot_lit::config::loader::load_full_config;
use neirobot_lit::config::types::FullConfig;
use neirobot_lit::data::websocket::{BybitWsClient, BybitPrivateWsClient, ReconnectSignal};
use neirobot_lit::data::types::WsData;
use neirobot_lit::ml::{OnnxEngine, TensorBuilder, Normalizer};
use neirobot_lit::trading::{ExecutionEngine, RiskManager, BybitRestClient};
use neirobot_lit::trading::emergency;
use neirobot_lit::data::orderbook::OrderBook;
use neirobot_lit::utils::logger::init_logger;
use neirobot_lit::utils::trade_logger::CsvTradeLogger;
use neirobot_lit::monitoring::resource_profiler::SystemMetricsUpdate;
use neirobot_lit::monitoring::command_server::{start_command_server, StatusResponse, Command};
use rust_decimal::Decimal;
use rust_decimal::prelude::FromPrimitive;

#[derive(Parser)]
#[command(author, version, about = "Neirobot LiT - Live Bot Runner", long_about = None)]
struct Args {
    /// Символ для запуска (например: BTCUSDT)
    symbol: String,

    /// Путь к конфигу бота (по умолчанию: bots/SYMBOL/config.toml)
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Путь к Parquet файлу для воспроизведения данных (Replay Mode)
    #[arg(long)]
    replay: Option<PathBuf>,

    /// Тестовое сообщение для проверки AlertManager (задача 222)
    #[arg(long)]
    test_alert: Option<String>,

    /// Порт для WebSocket мониторинга (задача 226: Farm Manager)
    /// Может быть передан через CLI или переменную окружения MONITORING_PORT
    #[arg(long, env = "MONITORING_PORT")]
    monitoring_port: Option<u16>,
}

use neirobot_lit::trading::types::MarketInfo;

fn setup_panic_handler(key: Arc<String>, secret: Arc<String>, symbol: String) {
    let default_hook = panic::take_hook();

    panic::set_hook(Box::new(move |panic_info| {
        eprintln!("\nFATAL ERROR: {}", panic_info);
        
        emergency::cancel_all_sync(&key, &secret, &symbol);

        default_hook(panic_info);
        
        process::exit(1);
    }));
}

fn main() -> Result<()> {
    // 1. Загрузка окружения для получения лимитов потоков
    dotenvy::dotenv().ok();
    let args = Args::parse();

    // Настройка Runtime согласно задаче №194
    let total_cpus = num_cpus::get();
    let hot_workers = (total_cpus.saturating_sub(2)).max(1);
    
    // Получаем доступные ядра для привязки (CPU Affinity)
    let core_ids = core_affinity::get_core_ids().unwrap_or_default();
    let core_counter = Arc::new(AtomicUsize::new(0));
    let core_ids_len = core_ids.len();

    // 1. Hot Path Runtime (Trading, Inference, WS)
    let hot_core_counter = core_counter.clone();
    let hot_core_ids = core_ids.clone();
    
    let hot_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(hot_workers)
        .thread_name("hot-worker")
        .thread_keep_alive(Duration::from_secs(60))
        .max_blocking_threads(64) // Ограничиваем блокирующие потоки
        .global_queue_interval(64) // Оптимизируем опрос глобальной очереди
        .on_thread_start(move || {
            let idx = hot_core_counter.fetch_add(1, Ordering::SeqCst);
            if core_ids_len > 0 {
                let core_id = hot_core_ids[idx % core_ids_len];
                neirobot_lit::utils::sys::set_hot_thread_config(Some(core_id.id));
                info!("[Runtime] Worker thread {} pinned to core {}", idx, core_id.id);
            } else {
                neirobot_lit::utils::sys::set_hot_thread_config(None);
            }
        })
        .enable_all()
        .build()
        .unwrap();

    // 2. Background Runtime (Logging, Metrics, Health)
    let bg_core_counter = core_counter.clone();
    let bg_core_ids = core_ids;
    
    let bg_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("bg-worker")
        .thread_keep_alive(Duration::from_secs(60))
        .max_blocking_threads(32) // Ограничиваем блокирующие потоки для фоновых задач
        .global_queue_interval(64) // Оптимизируем опрос глобальной очереди
        .on_thread_start(move || {
            // Также привязываем фоновые потоки, чтобы не прыгали по ядрам
            let idx = bg_core_counter.fetch_add(1, Ordering::SeqCst);
            if core_ids_len > 0 {
                let core_id = bg_core_ids[idx % core_ids_len];
                core_affinity::set_for_current(core_id);
                info!("[Runtime] Background worker thread {} pinned to core {}", idx, core_id.id);
            }
            
            #[cfg(target_os = "linux")]
            unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, 10); } // Низкий приоритет
        })
        .enable_all()
        .build()
        .unwrap();

    let bg_handle = bg_rt.handle().clone();

    // Запуск основного цикла в hot_rt
    hot_rt.block_on(async move {
        async_main(args, bg_handle).await
    })
}

async fn async_main(args: Args, bg_handle: tokio::runtime::Handle) -> Result<()> {
    // 1. Загрузка конфигурации
    let shutdown_token = CancellationToken::new();
    let token_clone = shutdown_token.clone();

    bg_handle.spawn(async move {
        wait_for_shutdown().await;
        token_clone.cancel();
    });
    
    let config_path = args.config.unwrap_or_else(|| {
        PathBuf::from("bots").join(&args.symbol).join("config.toml")
    });

    // Загружаем полную конфигурацию (используем "." как корень проекта)
    let mut full_config = load_full_config(Path::new("."), &config_path)
        .context("Failed to load full configuration")?;

    // Инициализация AuditLogger (Задача 217) - в начале для аудита всех действий
    let master_key = std::env::var("NEIRO_MASTER_KEY")
        .unwrap_or_else(|_| "default_master_key".to_string());
    let audit_logger = neirobot_lit::utils::audit::AuditLogger::init(&args.symbol, &master_key)
        .context("Failed to initialize audit logger")?;
    
    // Задача 003: Загрузка API-ключей из api_key_path для изоляции по ботам
    let api_key_path = &full_config.exchange.bybit.api_key_path;
    if std::path::Path::new(api_key_path).exists() {
        dotenvy::from_path(api_key_path).ok();
        info!("Loaded API credentials from: {}", api_key_path);
    } else {
        warn!("API key file not found at: {}. Falling back to environment variables.", api_key_path);
    }
    
    // Задача 226: Переопределение monitoring_port из CLI (для Farm Manager)
    if let Some(port) = args.monitoring_port {
        info!("Overriding monitoring_port from CLI: {}", port);
        full_config.bot.monitoring_port = port;
    }
    
    // Задача 222: Обработка --test-alert
    if let Some(test_message) = args.test_alert {
        info!("Test alert mode activated");
        
        // Получаем токен и chat_id из конфигурации
        let telegram_token = full_config.global.as_ref()
            .and_then(|g| g.telegram_token.clone())
            .or_else(|| std::env::var("TELEGRAM_TOKEN").ok())
            .context("TELEGRAM_TOKEN must be set in config or .env for test-alert mode")?;
        
        let chat_id = full_config.bot.override_chat_id.clone()
            .or_else(|| full_config.global.as_ref()
                .and_then(|g| g.default_chat_id.clone()))
            .or_else(|| std::env::var("TELEGRAM_CHAT_ID").ok())
            .context("TELEGRAM_CHAT_ID must be set in config or .env for test-alert mode")?;
        
        // Расшифровываем если нужно (Задача 217)
        let telegram_token = neirobot_lit::config::loader::decrypt_if_needed(&telegram_token, Some(&audit_logger))?;
        let chat_id = neirobot_lit::config::loader::decrypt_if_needed(&chat_id, Some(&audit_logger))?;
        
        let master_password = std::env::var("MASTER_PASSWORD").ok();
        
        // Создаем AlertManager
        let alert_manager = neirobot_lit::monitoring::alert_manager::AlertManager::new(
            telegram_token,
            chat_id,
            full_config.bot.alert_dedup_ttl_secs,
            master_password.as_deref(),
            Some(audit_logger.clone()),
        ).context("Failed to create AlertManager")?;
        
        // Отправляем тестовый алерт
        let alert = neirobot_lit::monitoring::alert_manager::Alert::new(
            neirobot_lit::monitoring::alert_manager::AlertLevel::Critical,
            test_message,
            format!("TestBot-{}", args.symbol),
        );
        
        alert_manager.send_alert(alert);
        
        info!("Test alert sent. Waiting for delivery...");
        tokio::time::sleep(Duration::from_secs(5)).await;
        
        info!("Test alert completed successfully");
        return Ok(());
    }
    
    // Вычисляем и логируем хэш конфигурации при старте
    let initial_config_hash = neirobot_lit::config::loader::compute_config_hash_from_file(&config_path)
        .context("Failed to compute initial config hash")?;
    info!("[Audit] Config SHA-256: {}", initial_config_hash);
    
    // Извлечение секретов для торговли через loader (с аудитом расшифровки)
    let (api_key, api_secret) = neirobot_lit::config::loader::load_secrets(Some(&audit_logger))
        .context("Failed to load API secrets")?;
    
    // Формируем путь к папке бота
    let bot_path = PathBuf::from("bots").join(&args.symbol);
    
    // Собираем список секретов для маскирования в логах
    let secrets = vec![api_key.clone(), api_secret.clone()];
    
    // Инициализируем логгер с маскированием секретов и сохраняем guard
    let _log_guard = init_logger(&full_config.logging, &bot_path, secrets)
        .context("Failed to initialize logger")?;

    // Задача 219: Инициализация Liveness Heartbeat
    neirobot_lit::utils::liveness::initialize_heartbeat(&bot_path, &args.symbol)
        .context("Failed to initialize liveness heartbeat")?;

    info!("Starting Neirobot LiT for {}", args.symbol);

    // Задача 140: Установка panic handler для экстренной отмены ордеров
    let api_key_arc = Arc::new(api_key.clone());
    let api_secret_arc = Arc::new(api_secret.clone());
    setup_panic_handler(api_key_arc, api_secret_arc, args.symbol.clone());
    info!("Panic handler initialized for emergency order cancellation");

    // Задача 184: Обработка SIGHUP для перезагрузки конфигурации
    let config_path_clone = config_path.clone();
    let symbol_clone = args.symbol.clone();
    
    // Создаем канал для передачи обновленной конфигурации в run_bot_loop
    let (config_tx, config_rx) = mpsc::channel::<neirobot_lit::config::types::FullConfig>(1);
    
    tokio::spawn(async move {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{signal, SignalKind};
            use std::sync::Mutex;
            use std::sync::Arc;
            
            let mut sighup = signal(SignalKind::hangup())
                .expect("Failed to install SIGHUP handler");
            
            // Сохраняем текущее содержимое файла для сравнения при перезагрузке
            let current_content = Arc::new(Mutex::new(
                std::fs::read_to_string(&config_path_clone).unwrap_or_default()
            ));
            
            loop {
                sighup.recv().await;
                info!("[Audit] SIGHUP received, reloading configuration for {}", symbol_clone);
                
                // Читаем новую конфигурацию
                match neirobot_lit::config::loader::load_full_config(std::path::Path::new("."), &config_path_clone) {
                    Ok(new_full_config) => {
                        // Читаем новое содержимое из файла
                        let new_content = std::fs::read_to_string(&config_path_clone).unwrap_or_default();
                        
                        // Получаем старое содержимое из памяти
                        let old_content = current_content.lock().unwrap().clone();
                        
                        // Вычисляем хэши для сравнения
                        let old_hash = neirobot_lit::config::loader::compute_config_hash(&old_content);
                        let new_hash = neirobot_lit::config::loader::compute_config_hash(&new_content);
                        
                        // Логируем изменения (сравниваем с предыдущим, не с начальным)
                        if old_hash != new_hash {
                            info!("[Audit] Config SHA-256 changed: {} -> {}", old_hash, new_hash);
                            
                            // Генерируем diff используя СТАРОЕ содержимое из памяти
                            let diff = neirobot_lit::config::loader::generate_config_diff(&old_content, &new_content);
                            if !diff.is_empty() {
                                info!("[Audit] Config diff:\n{}", diff);
                            }
                        } else {
                            info!("[Audit] Config hash unchanged: {}", new_hash);
                        }
                        
                        // Отправляем новую конфигурацию в run_bot_loop
                        if let Err(e) = config_tx.send(new_full_config).await {
                            warn!("[Audit] Failed to send config update to run_bot_loop: {}", e);
                        }
                        
                        // Обновляем текущее содержимое после успешной перезагрузки
                        *current_content.lock().unwrap() = new_content;
                    }
                    Err(e) => {
                        warn!("[Audit] Failed to reload config on SIGHUP: {}", e);
                    }
                }
            }
        }
        
        #[cfg(not(unix))]
        {
            // На Windows SIGHUP не поддерживается
            warn!("SIGHUP handler not available on this platform");
        }
    });

    // Задача 184: Проверка и создание config_history директории
    let config_history_dir = bot_path.join("config_history");
    if let Err(e) = fs::create_dir_all(&config_history_dir) {
        warn!("Failed to create config_history directory: {}", e);
    }

    // Задача 184: Проверка и создание резервной копии конфигурации при старте
    if let Err(e) = neirobot_lit::config::loader::backup_config(&config_path) {
        warn!("Failed to backup config on startup: {}", e);
    }

    // Задача 169: Проверка синхронизации времени с биржей
    info!("Checking clock synchronization with Bybit...");
    match neirobot_lit::utils::helpers::check_clock_skew(
        &full_config.exchange.rest_api_url,
        full_config.bot.max_clock_skew_ms,
    ).await {
        Ok(delta) => {
            if delta.abs() > full_config.bot.max_clock_skew_ms {
                error!(
                    "CRITICAL: Clock skew {}ms exceeds limit {}ms. Bot will not start.",
                    delta, full_config.bot.max_clock_skew_ms
                );
                return Err(anyhow::anyhow!("Clock skew check failed"));
            }
            info!("Clock synchronization OK: delta = {}ms", delta);
        },
        Err(e) => {
            warn!("Clock skew check failed (non-critical): {}", e);
        }
    }

    // Задача 230: Инициализация CPU Affinity для процесса
    if let Some(core_id) = full_config.bot.system.cpu_core {
        neirobot_lit::utils::sys::set_process_affinity(core_id);
    }

    // Задача 230: Запуск мониторинга памяти процесса
    let max_mem_mb = full_config.bot.system.max_memory_mb;
    let (system_event_tx, mut system_event_rx) = mpsc::channel::<neirobot_lit::utils::SystemEvent>(10);
    
    bg_handle.spawn(async move {
        neirobot_lit::utils::sys::monitor_resources(max_mem_mb, system_event_tx).await;
    });
    
    // Задача 230: Обработка событий системного мониторинга
    let shutdown_token_clone = shutdown_token.clone();
    tokio::spawn(async move {
        while let Some(event) = system_event_rx.recv().await {
            match event {
                neirobot_lit::utils::SystemEvent::SoftLimitReached { current_mb, limit_mb } => {
                    error!(
                        "CRITICAL: Memory soft limit reached! current={}MB, limit={}MB. Initiating graceful shutdown.",
                        current_mb, limit_mb
                    );
                    // TODO: Интеграция с Graceful Degradation (задача 220)
                    // Пока просто инициируем shutdown
                    shutdown_token_clone.cancel();
                }
            }
        }
    });

    // Инициализация SharedState для health-check (задача 135)
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64};
    use neirobot_lit::monitoring::health::{SharedState, start_health_server};
    use neirobot_lit::utils::helpers;
    
    let shared_state = Arc::new(SharedState {
        last_update: AtomicU64::new(helpers::unix_ms()),
        last_heartbeat: AtomicU64::new(helpers::unix_ms()),
        ws_connected: AtomicBool::new(true),
        emergency_mode: AtomicBool::new(false),
        start_time: tokio::time::Instant::now(),
        config: full_config.monitoring.clone().unwrap_or_default(),
    });

    // Запуск задачи мониторинга ресурсов в фоновом рантайме (каждые 5 секунд)
    bg_handle.spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(5));
        loop {
            interval.tick().await;
            neirobot_lit::utils::sys::update_resource_metrics();
        }
    });

    // Запуск health-check сервера в отдельной задаче в фоновом рантайме
    let state_for_server = shared_state.clone();
    let health_config = full_config.monitoring.clone();
    let max_memory_mb = full_config.bot.system.max_memory_mb;
    bg_handle.spawn(async move {
        start_health_server(health_config, state_for_server, max_memory_mb).await;
    });

    // Запуск Watchdog для Hot Path (Задача 146)
    let state_for_watchdog = shared_state.clone();
    let watchdog_config = full_config.monitoring.watchdog.clone();
    let watchdog_symbol = args.symbol.clone();
    
    std::thread::spawn(move || {
        use std::sync::atomic::Ordering;
        use neirobot_lit::monitoring::prometheus::{WATCHDOG_STALL_GAUGE, WATCHDOG_CHECK_GAUGE};
        
        info!("Watchdog thread started for hot path monitoring");
        let mut consecutive_misses = 0;
        
        loop {
            let now = helpers::unix_ms();
            let last = state_for_watchdog.last_heartbeat.load(Ordering::Relaxed);
            let delta = now.saturating_sub(last);
            
            // Обновляем метрику последней проверки
            if let Some(gauge) = WATCHDOG_CHECK_GAUGE.get() {
                gauge.with_label_values(&[&watchdog_symbol]).set(now as f64 / 1000.0);
            }
            
            if delta > watchdog_config.suspend_grace_ms {
                warn!("Possible suspend detected (delta {}ms), resetting watchdog", delta);
                state_for_watchdog.last_heartbeat.store(now, Ordering::Relaxed);
                consecutive_misses = 0;
            } else if delta > watchdog_config.stall_timeout_ms {
                consecutive_misses += 1;
                error!("Watchdog: stall detected! Miss {}/3 (delta {}ms)", consecutive_misses, delta);
                
                if let Some(gauge) = WATCHDOG_STALL_GAUGE.get() {
                    gauge.with_label_values(&[&watchdog_symbol]).set(delta as f64 / 1000.0);
                }
                
                if consecutive_misses >= 3 {
                    panic!("CRITICAL: HOT PATH STALLED for {}ms", delta);
                }
            } else {
                consecutive_misses = 0;
                if let Some(gauge) = WATCHDOG_STALL_GAUGE.get() {
                    gauge.with_label_values(&[&watchdog_symbol]).set(0.0);
                }
            }
            
            std::thread::sleep(std::time::Duration::from_millis(watchdog_config.check_interval_ms));
        }
    });

    // Инициализация Prometheus metrics exporter (задача 189)
    // Консолидированная система мониторинга с использованием metrics крейта
    let metrics_port = full_config.monitoring.metrics_port;
    neirobot_lit::monitoring::metrics::init_metrics_exporter(metrics_port)
        .context("Failed to initialize Prometheus metrics exporter")?;
    info!("Prometheus metrics exporter initialized on port {}", metrics_port);

    // 2. Инициализация ML компонентов
    let model_dir = full_config.bot.model_path.parent()
        .unwrap_or_else(|| Path::new("models"));
    
    // Путь к metadata.json (задача 097)
    let metadata_path = model_dir.join("metadata.json");
    
    let tensor_builder = TensorBuilder::from_metadata(
        metadata_path.to_str().context("Invalid metadata path")?
    ).context("Failed to load TensorBuilder from metadata")?;
    
    let mut engine = OnnxEngine::load(
        &full_config.bot.model_path,
        full_config.bot.seq_len,
        full_config.bot.features_dim,
        &full_config.bot.onnx,
        &args.symbol,
        Some(&full_config.bot)
    ).context("Failed to load ONNX model")?;
    
    info!(
        "ML Engine loaded: seq_len={}, features_dim={}", 
        full_config.bot.seq_len, full_config.bot.features_dim
    );
    
    // Прогрев модели
    engine.warmup().context("Failed to warmup ONNX engine")?;

    // 3. Инициализация Торговых компонентов
    let rest_client = BybitRestClient::with_rate_limit_config(
        &full_config.exchange,
        full_config.bot.rate_limit_threshold_pct,
        full_config.bot.backoff_base_ms,
    )
        .context("Failed to initialize REST client")?;
    
    // Загрузка информации о символе с кэшированием (Задача 138)
    let cache_path = bot_path.join("cache").join("symbol_info.json");
    fs::create_dir_all(cache_path.parent().unwrap())?;

    let symbol_info = match rest_client.fetch_symbol_info(&args.symbol).await {
        Ok(info) => {
            // Сохраняем в кэш
            let f = File::create(&cache_path)?;
            serde_json::to_writer_pretty(f, &info)?;
            info!("Symbol info loaded from API and cached for {}", args.symbol);
            info!(
                "Contract params: tick_size={}, price_precision={}, min_price={}, max_price={}",
                info.price_filter.tick_size, 
                info.price_filter.price_precision,
                info.price_filter.min_price,
                info.price_filter.max_price
            );
            info!(
                "Lot params: min_qty={}, max_qty={}, qty_step={}",
                info.lot_filter.min_qty,
                info.lot_filter.max_qty,
                info.lot_filter.qty_step
            );
            info!("Max leverage for {}: {}", args.symbol, info.max_leverage);
            info
        }
        Err(e) => {
            warn!("Failed to fetch symbol info: {}. Trying cache...", e);
            let f = File::open(&cache_path)
                .context("No cache available and API fetch failed")?;
            let info: neirobot_lit::trading::types::SymbolInfo = serde_json::from_reader(f)
                .context("Failed to parse cached symbol info")?;
            info!("Symbol info loaded from cache for {}", args.symbol);
            info!(
                "Contract params (cached): tick_size={}, price_precision={}, min_price={}, max_price={}",
                info.price_filter.tick_size, 
                info.price_filter.price_precision,
                info.price_filter.min_price,
                info.price_filter.max_price
            );
            info!(
                "Lot params (cached): min_qty={}, max_qty={}, qty_step={}",
                info.lot_filter.min_qty,
                info.lot_filter.max_qty,
                info.lot_filter.qty_step
            );
            info!("Max leverage (cached) for {}: {}", args.symbol, info.max_leverage);
            info
        }
    };
    
    // Конвертируем SymbolInfo в MarketInfo для обратной совместимости
    let market_info = MarketInfo {
        qty_step: Decimal::from_f64(symbol_info.lot_filter.qty_step).unwrap_or_default(),
        min_order_qty: Decimal::from_f64(symbol_info.lot_filter.min_qty).unwrap_or_default(),
        max_order_qty: Decimal::from_f64(symbol_info.lot_filter.max_qty).unwrap_or_default(),
        tick_size: Decimal::from_f64(symbol_info.price_filter.tick_size).unwrap_or_default(),
    };

    let risk_manager = RiskManager::new(
        full_config.risk.clone(), 
        full_config.bot.initial_balance
    );
    
    // Устанавливаем AuditLogger в RiskManager
    let mut risk_manager = risk_manager;
    risk_manager.set_audit_logger(audit_logger.clone());
    
    // Задача 222: Инициализация AlertManager
    let alert_manager = if let Some(ref telegram_token) = full_config.global.as_ref()
        .and_then(|g| g.telegram_token.clone())
        .or_else(|| std::env::var("TELEGRAM_TOKEN").ok()) {
        
        let chat_id = full_config.bot.override_chat_id.clone()
            .or_else(|| full_config.global.as_ref()
                .and_then(|g| g.default_chat_id.clone()))
            .or_else(|| std::env::var("TELEGRAM_CHAT_ID").ok())
            .context("TELEGRAM_CHAT_ID must be set in config or .env")?;
        
        // Расшифровываем если нужно (Задача 217)
        let telegram_token = neirobot_lit::config::loader::decrypt_if_needed(&telegram_token, Some(&audit_logger))?;
        let chat_id = neirobot_lit::config::loader::decrypt_if_needed(&chat_id, Some(&audit_logger))?;
        
        let master_password = std::env::var("MASTER_PASSWORD").ok();
        
        match neirobot_lit::monitoring::alert_manager::AlertManager::new(
            telegram_token,
            chat_id,
            full_config.bot.alert_dedup_ttl_secs,
            master_password.as_deref(),
            Some(audit_logger.clone()),
        ) {
            Ok(manager) => {
                info!("AlertManager initialized for {}", args.symbol);
                Some(manager)
            }
            Err(e) => {
                warn!("Failed to initialize AlertManager: {}. Continuing without alerts.", e);
                None
            }
        }
    } else {
        info!("Telegram token not configured. AlertManager disabled.");
        None
    };
    
    info!(
        "Trading strategy params: threshold_buy={:.2}, threshold_sell={:.2}, close_on_flat={}",
        full_config.bot.threshold_buy, full_config.bot.threshold_sell, full_config.bot.close_on_flat
    );

    // Инициализация Trade Logger в фоновом рантайме
    let trade_log_path = PathBuf::from("bots").join(&args.symbol).join("logs").join("trades.csv");
    let (trade_logger, trade_logger_handle) = {
        let _guard = bg_handle.enter();
        CsvTradeLogger::init(trade_log_path)
            .context("Failed to initialize trade logger")?
    };

    let state_path = PathBuf::from("bots").join(&args.symbol).join("state.json");

    let mut risk_manager = risk_manager;
    if let Some(am) = alert_manager.clone() {
        risk_manager.set_alert_manager(am);
    }

    let mut execution = ExecutionEngine::new(
        args.symbol.clone(),
        risk_manager,
        full_config.bot.clone(),
        market_info,
        trade_logger.get_sender(),
        state_path,
    );
    
    // Задача 204: Инициализация логирования влияния на цену если включено
    if full_config.bot.enable_impact_logging {
        let market_impact_tx = neirobot_lit::utils::logger::spawn_market_impact_logger();
        execution.set_market_impact_logger(market_impact_tx);
        info!("Market impact logging enabled for {}", args.symbol);
    }
    
    // Задача 182: Установка директории логов для архивации
    let log_dir = bot_path.join("logs");
    execution.health_monitor.set_log_dir(log_dir);
    
    // Задача 161: Инициализация детектора режимов рынка
    let regime_config_path = PathBuf::from("bots").join(&args.symbol).join("model").join("regime_config.json");
    if regime_config_path.exists() {
        match neirobot_lit::trading::regime_detector::RegimeDetector::new(&regime_config_path, 10) {
            Ok(detector) => {
                info!("Regime detector initialized for {}", args.symbol);
                execution.set_regime_detector(detector);
            }
            Err(e) => {
                warn!("Failed to initialize regime detector for {}: {}. Continuing without regime-based thresholds.", args.symbol, e);
            }
        }
    } else {
        info!("Regime config not found at {:?}. Continuing without regime-based thresholds.", regime_config_path);
    }

    // 4. Первичная синхронизация состояния и позиции
    info!("Performing initial state sync and reconciliation for {}...", args.symbol);
    
    // Задача 174: Проверка прав API-ключа перед запуском
    info!("Checking API key permissions and expiry...");
    match rest_client.get_api_key_info().await {
        Ok(info) => {
            if let Err(e) = execution.risk_manager.health_monitor.validate_api_permissions(&info, &full_config.exchange) {
                error!("CRITICAL: API permission check failed: {}", e);
                return Err(e);
            }
            info!("API key validation successful.");
        }
        Err(e) => {
            error!("Failed to query API key information: {}", e);
            return Err(e.context("API key info query failed"));
        }
    }

    // Задача 218: Восстановление состояния при перезапуске (Reliability & Safety)
    info!("[Persistence] Initializing state recovery sequence...");
    if let Err(e) = execution.load_state_on_startup(&rest_client, &full_config.exchange).await {
        error!("[Persistence] Fatal error during state recovery: {}. Starting in safe mode.", e);
        execution.emergency_mode = true;
    }

    // Синхронизация состояния с биржей (проверка расхождений)
    execution.sync_state(&rest_client, &full_config.exchange).await
        .context("Failed to perform initial state sync")?;

    if execution.emergency_mode {
        error!("Bot started in EMERGENCY MODE due to state desync. Fix manually and restart.");
    }

    // Задача 221: Инициализация Real-time Equity Streamer
    info!("Initializing Real-time Equity Streamer on port {}...", full_config.bot.monitoring_port);
    
    // Создаем broadcast канал для трансляции equity обновлений
    let (equity_broadcast_tx, _equity_broadcast_rx) = tokio::sync::broadcast::channel::<neirobot_lit::monitoring::types::EquityUpdate>(100);
    
    // Устанавливаем канал в PositionManager
    execution.position_manager.set_equity_channel(
        equity_broadcast_tx.clone(),
        full_config.bot.initial_balance,
        full_config.bot.taker_fee_bps,
        full_config.bot.min_update_ms,
    );
    
    // Запускаем WebSocket сервер для мониторинга в отдельной задаче
    let monitoring_port = full_config.bot.monitoring_port;
    let equity_tx_for_ws = equity_broadcast_tx.clone();
    let monitoring_symbol = args.symbol.clone();
    let monitoring_shutdown = shutdown_token.clone();
    let audit_logger_for_ws = audit_logger.clone();
    
    tokio::spawn(async move {
        use axum::{
            extract::ws::{WebSocket, WebSocketUpgrade},
            response::IntoResponse,
            routing::get,
            Router,
        };
        use futures_util::{SinkExt, StreamExt};
        
        async fn ws_handler(
            ws: WebSocketUpgrade,
            axum::extract::State(state): axum::extract::State<(tokio::sync::broadcast::Sender<neirobot_lit::monitoring::types::EquityUpdate>, String, neirobot_lit::utils::audit::AuditLogger)>,
        ) -> impl IntoResponse {
            ws.on_upgrade(move |socket| handle_socket(socket, state.0, state.1, state.2))
        }
        
        async fn handle_socket(
            socket: WebSocket,
            equity_tx: tokio::sync::broadcast::Sender<neirobot_lit::monitoring::types::EquityUpdate>,
            symbol: String,
            audit_logger: neirobot_lit::utils::audit::AuditLogger,
        ) {
            let (mut sender, mut receiver) = socket.split();
            let mut equity_rx = equity_tx.subscribe();
            
            tracing::info!("[Equity Streamer] Client connected for {}", symbol);
            
            // Логируем подключение в security_audit.csv (Задача 217)
            let _ = audit_logger.log_event(
                "MONITORING_CONNECT",
                &format!("WebSocket client connected to equity streamer"),
                "SUCCESS",
                "",
                "",
            );
            
            let mut send_task = tokio::spawn(async move {
                while let Ok(update) = equity_rx.recv().await {
                    let json = match serde_json::to_string(&update) {
                        Ok(j) => j,
                        Err(e) => {
                            tracing::error!("[Equity Streamer] Failed to serialize update: {}", e);
                            continue;
                        }
                    };
                    
                    if sender.send(axum::extract::ws::Message::Text(json)).await.is_err() {
                        break;
                    }
                }
            });
            
            let mut recv_task = tokio::spawn(async move {
                while let Some(Ok(_msg)) = receiver.next().await {
                    // Игнорируем входящие сообщения от клиента
                }
            });
            
            tokio::select! {
                _ = (&mut send_task) => recv_task.abort(),
                _ = (&mut recv_task) => send_task.abort(),
            }
            
            tracing::info!("[Equity Streamer] Client disconnected for {}", symbol);
        }
        
        let app = Router::new()
            .route("/ws", get(ws_handler))
            .with_state((equity_tx_for_ws, monitoring_symbol, audit_logger_for_ws));
        
        let addr = format!("127.0.0.1:{}", monitoring_port);
        let listener = match tokio::net::TcpListener::bind(&addr).await {
            Ok(l) => l,
            Err(e) => {
                tracing::error!("[Equity Streamer] Failed to bind to {}: {}", addr, e);
                return;
            }
        };
        
        tracing::info!("[Equity Streamer] WebSocket server listening on {}", addr);
        
        let server = axum::serve(listener, app);
        
        tokio::select! {
            result = server => {
                if let Err(e) = result {
                    tracing::error!("[Equity Streamer] Server error: {}", e);
                }
            }
            _ = monitoring_shutdown.cancelled() => {
                tracing::info!("[Equity Streamer] Shutting down WebSocket server");
            }
        }
    });
    
    // Задача 221: Запуск фоновой задачи синхронизации баланса с биржей
    let balance_sync_interval = full_config.bot.balance_sync_interval;
    let balance_rest_client = rest_client.clone();
    let balance_shutdown = shutdown_token.clone();
    let balance_symbol = args.symbol.clone();
    
    // Создаем канал для передачи обновлений баланса в основной цикл
    let (balance_sync_tx, mut balance_sync_rx) = mpsc::channel::<rust_decimal::Decimal>(1);
    
    bg_handle.spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(balance_sync_interval));
        
        loop {
            tokio::select! {
                _ = interval.tick() => {
                    match balance_rest_client.get_wallet_balance().await {
                        Ok(wallet_response) => {
                            let mut total_equity = rust_decimal::Decimal::ZERO;
                            for wallet in &wallet_response.list {
                                total_equity += wallet.total_equity;
                            }
                            
                            tracing::debug!("[Equity Streamer] Balance sync: {}", total_equity);
                            
                            if let Err(e) = balance_sync_tx.send(total_equity).await {
                                tracing::warn!("[Equity Streamer] Failed to send balance sync: {}", e);
                            }
                        }
                        Err(e) => {
                            tracing::warn!("[Equity Streamer] Failed to fetch balance for sync: {}", e);
                        }
                    }
                }
                _ = balance_shutdown.cancelled() => {
                    tracing::info!("[Equity Streamer] Balance sync task shutting down for {}", balance_symbol);
                    break;
                }
            }
        }
    });
    
    // Сохраняем receiver для использования в run_bot_loop
    let balance_sync_rx_for_loop = balance_sync_rx;

    // Задача 235: Запуск фоновой задачи очистки зависших ордеров
    let cleanup_interval_min = full_config.bot.cleanup_interval_min;
    let cleanup_shutdown = shutdown_token.clone();
    let cleanup_symbol = args.symbol.clone();
    
    // Создаем канал для триггера cleanup routine
    let (cleanup_trigger_tx, mut cleanup_trigger_rx) = mpsc::channel::<()>(1);
    
    bg_handle.spawn(async move {
        // Добавляем небольшой джиттер к интервалу для избежания спама в ровные минуты
        use rand::Rng;
        let jitter_secs = rand::thread_rng().gen_range(0..60);
        let base_interval_secs = cleanup_interval_min * 60;
        let interval_with_jitter = base_interval_secs + jitter_secs;
        
        let mut interval = tokio::time::interval(Duration::from_secs(interval_with_jitter));
        
        loop {
            tokio::select! {
                _ = interval.tick() => {
                    tracing::info!("[Cleanup] Triggering stale order cleanup routine for {}", cleanup_symbol);
                    
                    if let Err(e) = cleanup_trigger_tx.send(()).await {
                        tracing::warn!("[Cleanup] Failed to send cleanup trigger: {}", e);
                    }
                }
                _ = cleanup_shutdown.cancelled() => {
                    tracing::info!("[Cleanup] Cleanup trigger task shutting down for {}", cleanup_symbol);
                    break;
                }
            }
        }
    });
    
    // Сохраняем receiver для использования в run_bot_loop
    let cleanup_trigger_rx_for_loop = cleanup_trigger_rx;

    // 5. Запуск WebSocket клиентов
    let ob = OrderBook::new(&args.symbol);

    if let Some(replay_path) = args.replay {
        return run_replay_loop(
            replay_path,
            ob,
            tensor_builder,
            &mut engine,
            &mut execution,
            &args.symbol,
            &rest_client,
            &full_config,
        ).await;
    }

    let (tx, rx) = mpsc::channel(1024);
    let ws_client = BybitWsClient::new(full_config.exchange.clone(), args.symbol.clone());
    let (ws_reconnect_tx, ws_reconnect_rx) = mpsc::channel(1);
    
    let (private_tx, private_rx) = mpsc::channel(1024);
    let private_ws = BybitPrivateWsClient::new(full_config.exchange.clone());
    let (priv_reconnect_tx, priv_reconnect_rx) = mpsc::channel(1);
    
    let token_ws = shutdown_token.clone();
    tokio::spawn(async move {
        if let Err(e) = ws_client.run(tx, ws_reconnect_rx, token_ws).await {
            error!("Public WS Client fatal error for {}: {}", args.symbol, e);
        }
    });

    let token_priv = shutdown_token.clone();
    tokio::spawn(async move {
        if let Err(e) = private_ws.run(private_tx, priv_reconnect_rx, token_priv).await {
            error!("Private WS Client fatal error: {}", e);
        }
    });

    // 6. Основной цикл обработки
    let mut ob = OrderBook::new(&args.symbol);
    info!("Bot is ready and waiting for market data for {}...", args.symbol);

    // Задача 174: Канал для периодической проверки прав API
    let (api_check_tx, mut api_check_rx) = mpsc::channel(1);
    let api_rest_client = rest_client.clone();
    let api_full_config = full_config.clone();
    let api_shutdown_token = shutdown_token.clone();

    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(24 * 3600));
        // Пропускаем первый тик, так как первичная проверка уже была при запуске
        interval.tick().await; 
        
        loop {
            tokio::select! {
                _ = interval.tick() => {
                    info!("Performing periodic API key permission check...");
                    match api_rest_client.get_api_key_info().await {
                        Ok(info) => {
                            if let Err(e) = api_check_tx.send(info).await {
                                error!("Failed to send API check result: {}", e);
                                break;
                            }
                        }
                        Err(e) => {
                            error!("Periodic API key check failed: {}", e);
                        }
                    }
                }
                _ = api_shutdown_token.cancelled() => {
                    break;
                }
            }
        }
    });

    // Задача 205: Инициализация синхронизации баланса для ежедневных PnL-отчетов
    let equity_tx = neirobot_lit::utils::logger::spawn_equity_logger();
    let equity_rest_client = rest_client.clone();
    let equity_bot_path = PathBuf::from("bots").join(&args.symbol);
    let equity_shutdown_token = shutdown_token.clone();
    
    bg_handle.spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(180)); // 3 минуты
        
        loop {
            tokio::select! {
                _ = interval.tick() => {
                    // Запрашиваем баланс с биржи
                    match equity_rest_client.get_wallet_balance().await {
                        Ok(wallet_response) => {
                            // Рассчитываем total equity и margin из всех кошельков
                            let mut total_equity = 0.0;
                            let mut total_available_balance = 0.0;
                            let mut total_margin = 0.0;
                            
                            for wallet in &wallet_response.list {
                                total_equity += wallet.total_equity.to_f64().unwrap_or(0.0);
                                total_available_balance += wallet.total_available_balance.to_f64().unwrap_or(0.0);
                                total_margin += wallet.total_margin_balance.to_f64().unwrap_or(0.0);
                            }
                            
                            // Логируем данные equity
                            let equity_log = neirobot_lit::utils::logger::EquityLog {
                                timestamp_ms: neirobot_lit::utils::time::timestamp_ms(),
                                rest_equity: total_equity,
                                available_balance: total_available_balance,
                                rest_margin: total_margin,
                                local_unrealized_pnl: 0.0, // Будет заполнено из стакана в основном цикле
                                total_pnl_delta: 0.0,
                                bot_path: equity_bot_path.clone(),
                            };
                            
                            if let Err(e) = equity_tx.send(equity_log).await {
                                tracing::warn!("Failed to send equity log: {}", e);
                            }
                        }
                        Err(e) => {
                            tracing::warn!("Failed to fetch wallet balance for equity logging: {}", e);
                        }
                    }
                }
                _ = equity_shutdown_token.cancelled() => {
                    break;
                }
            }
        }
    });

    // Создаем канал для дампа снимков стакана (задача 132)
    let (snapshot_tx, snapshot_rx) = mpsc::channel(100);
    let bot_path_clone = PathBuf::from("bots").join(&args.symbol);
    
    // Запускаем фоновый воркер для записи снимков в фоновом рантайме
    bg_handle.spawn(async move {
        neirobot_lit::data::dump::start_snapshot_writer(snapshot_rx, bot_path_clone).await;
    });

    // Создаем канал для дампа публичных сделок (задача 236)
    let (trades_tx, trades_rx) = mpsc::channel(1000);
    let bot_path_trades = PathBuf::from("bots").join(&args.symbol);
    
    // Запускаем фоновый воркер для записи сделок в фоновом рантайме
    bg_handle.spawn(async move {
        neirobot_lit::data::dump::start_trades_writer(trades_rx, bot_path_trades).await;
    });

    let rx_stream = ReceiverStream::new(rx);
    
    // Вычисляем путь к папке бота для проверки файла остановки
    let bot_path = PathBuf::from("bots").join(&args.symbol);
    
    // Задача 225: Инициализация ResourceProfiler для мониторинга системных ресурсов
    let (mut resource_profiler, mut metrics_rx) = {
        let _guard = bg_handle.enter();
        neirobot_lit::monitoring::resource_profiler::ResourceProfiler::new(
            full_config.bot.resource_thresholds.clone()
        ).context("Failed to initialize ResourceProfiler")?
    };
    
    // Задача 230: Устанавливаем лимит памяти для мониторинга
    resource_profiler.max_memory_kb = full_config.bot.system.max_memory_mb * 1024;
    
    // Запускаем профилировщик в фоновом рантайме
    let _profiler_handle = {
        let _guard = bg_handle.enter();
        resource_profiler.spawn()
    };
    
    // Задача 229: Инициализация Command Server для удаленного управления
    let (command_tx, command_rx) = mpsc::channel(32);
    let command_port = full_config.bot.monitoring_port; // Используем порт из конфигурации
    
    // Создаем shared state для статуса бота
    let status_state = Arc::new(parking_lot::RwLock::new(StatusResponse {
        symbol: args.symbol.clone(),
        status: "running".to_string(),
        pnl: None,
        position: None,
        latency_ms: None,
        uptime_secs: 0,
    }));
    
    // Запускаем command server в фоновом рантайме
    let command_config = Arc::new(full_config.bot.clone());
    let command_status = status_state.clone();
    let command_tx_clone = command_tx.clone();
    
    bg_handle.spawn(async move {
        if let Err(e) = start_command_server(
            command_config,
            command_tx_clone,
            command_status,
            command_port,
        ).await {
            error!("[CommandServer] Failed to start: {}", e);
        }
    });
    
    info!("[CommandServer] Started on port {}", command_port);
    
    // Задача 066: Initial Sync перед запуском основного цикла
    info!("Performing initial position sync for {}...", args.symbol);
    if let Ok((remote_qty, remote_avg_price, remote_leverage, remote_pnl)) = 
        rest_client.get_position_signed(&full_config.exchange.bybit.category, &args.symbol, full_config.bot.position_idx).await {
        execution.position_manager.sync_from_remote(
            remote_qty, 
            remote_avg_price, 
            remote_leverage, 
            remote_pnl,
            &ob.market_info
        );
        info!("Initial position sync completed. Local qty: {}", execution.position_manager.get_position().qty);
    } else {
        warn!("Initial position sync failed, continuing with local state");
    }
    
    // Задача 184: Создаем Arc для хранения содержимого конфига
    let config_content = Arc::new(std::sync::Mutex::new(
        std::fs::read_to_string(&config_path).unwrap_or_default()
    ));
    
    // Задача 184: Клонируем config_tx для передачи в run_bot_loop
    let config_tx_for_loop = config_tx.clone();
    
    // Задача 218: Периодическое сохранение состояния
    let persistence_interval = full_config.bot.persistence_interval_sec;
    let persistence_shutdown = shutdown_token.clone();
    let persistence_symbol = args.symbol.clone();
    let (persistence_tx, persistence_rx) = mpsc::channel(1);

    bg_handle.spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(persistence_interval));
        // Пропускаем первый тик, чтобы не сохранять сразу при запуске
        interval.tick().await; 
        
        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if let Err(e) = persistence_tx.send(()).await {
                        tracing::warn!("[Persistence] Failed to send persistence trigger for {}: {}", persistence_symbol, e);
                        break;
                    }
                }
                _ = persistence_shutdown.cancelled() => {
                    tracing::info!("[Persistence] Periodic state saver shutting down for {}", persistence_symbol);
                    break;
                }
            }
        }
    });

    let loop_result = run_bot_loop(
        rx_stream,
        private_rx,
        ob,
        tensor_builder,
        &mut engine,
        &mut execution,
        &args.symbol,
        &rest_client,
        &full_config,
        ws_reconnect_tx,
        shutdown_token.clone(),
        &bot_path,
        snapshot_tx,
        shared_state.clone(),
        api_check_rx,
        config_rx,
        balance_sync_rx_for_loop,
        cleanup_trigger_rx_for_loop, // Задача 235: Триггер для cleanup routine
        metrics_rx,
        command_rx,
        status_state.clone(),
        config_content, // Задача 184: Передаем содержимое конфига
        config_tx_for_loop, // Задача 184: Передаем sender для отправки обновлений
        persistence_rx, // Задача 218: Канал для периодического сохранения состояния
    ).await;

    if let Err(ref e) = loop_result {
        error!("Bot loop terminated with error: {:?}", e);
    }

    // --- Graceful Shutdown Sequence ---
    info!("Starting graceful shutdown sequence (timeout 10s)...");
    
    // Повторный сигнал во время завершения — немедленный выход
    bg_handle.spawn(async move {
        wait_for_shutdown().await;
        error!("Second shutdown signal received! Force exiting.");
        std::process::exit(1);
    });

    let shutdown_future = async {
        // Задача 190: Сохранение состояния перед shutdown
        info!("[Persistence] Saving bot state before shutdown...");
        let state_save_path = PathBuf::from("bots").join(&args.symbol).join("state.json");
        
        // Собираем состояние
        let position_snapshot = if !execution.position_manager.get_position().qty.is_zero() {
            Some(neirobot_lit::utils::persistence::PositionSnapshot::from(
                execution.position_manager.get_position()
            ))
        } else {
            None
        };
        
        // Собираем активные ордера из RiskManager
        let active_orders: Vec<(String, neirobot_lit::utils::persistence::OrderIntent)> = 
            execution.risk_manager.active_intents.iter()
                .map(|(link_id, intent)| {
                    let persist_intent = neirobot_lit::utils::persistence::OrderIntent {
                        side: intent.side,
                        price: intent.price,
                        qty: intent.qty,
                        timestamp: intent.timestamp,
                        filled_qty: intent.filled_qty,
                    };
                    (link_id.clone(), persist_intent)
                })
                .collect();
        
        let bot_state = neirobot_lit::utils::persistence::BotState {
            position: position_snapshot,
            active_orders,
            timestamp_ms: neirobot_lit::utils::time::timestamp_ms(),
        };
        
        if let Err(e) = neirobot_lit::utils::persistence::save_state(&state_save_path, &bot_state) {
            error!("[Persistence] Failed to save state: {}", e);
        }
        
        // 1. Panic Exit (Emergency Close) if configured
        if full_config.bot.close_on_exit {
            info!("Emergency market close enabled. Executing Panic Exit...");
            if let Err(e) = execution.emergency_market_close(&rest_client, &full_config.exchange).await {
                error!("Panic Exit failed: {}", e);
            }
        } else {
            // 2. Fallback: just cancel all orders if close_on_exit is false
            info!("Cancelling all open orders...");
            if let Err(e) = execution.order_manager.cancel_all_orders(&rest_client, &mut execution.risk_manager, &full_config.bot, &full_config.exchange).await {
                error!("Failed to cancel all orders: {}", e);
            }
        }

        // 3. Drop execution and logger to close trade logger senders
        drop(execution);
        drop(trade_logger);

        // 4. Wait for trade logger to flush
        info!("Waiting for trade logger to flush...");
        let _ = trade_logger_handle.await;
    };

    if let Err(_) = tokio::time::timeout(Duration::from_secs(10), shutdown_future).await {
        error!("Shutdown sequence timed out! Force exiting.");
        std::process::exit(1);
    }

    info!("Bot for {} stopped.", args.symbol);
    loop_result
}

async fn wait_for_shutdown() {
    let ctrl_c = tokio::signal::ctrl_c();
    
    #[cfg(unix)]
    let terminate = async {
        let mut sig = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler");
        sig.recv().await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => info!("SIGINT (Ctrl+C) received"),
        _ = terminate => info!("SIGTERM received"),
    }
}

/// Универсальный цикл обработки рыночных данных
pub async fn run_bot_loop<S>(
    mut stream: S,
    mut private_rx: mpsc::Receiver<serde_json::Value>,
    mut ob: OrderBook,
    mut tensor_builder: TensorBuilder,
    engine: &mut OnnxEngine,
    execution: &mut ExecutionEngine,
    symbol: &str,
    rest_client: &BybitRestClient,
    config: &FullConfig,
    ws_reconnect_tx: mpsc::Sender<ReconnectSignal>,
    token: CancellationToken,
    bot_path: &Path,
    snapshot_tx: mpsc::Sender<neirobot_lit::data::orderbook::OrderBookSnapshot>,
    shared_state: std::sync::Arc<neirobot_lit::monitoring::health::SharedState>,
    mut api_check_rx: mpsc::Receiver<neirobot_lit::trading::rest_client::ApiKeyInfoResponse>,
    mut config_rx: mpsc::Receiver<neirobot_lit::config::types::FullConfig>,
    mut balance_sync_rx: mpsc::Receiver<rust_decimal::Decimal>,
    mut cleanup_trigger_rx: mpsc::Receiver<()>, // Задача 235: Триггер для cleanup routine
    mut metrics_rx: tokio::sync::broadcast::Receiver<SystemMetricsUpdate>,
    mut command_rx: mpsc::Receiver<Command>,
    status_state: Arc<parking_lot::RwLock<StatusResponse>>,
    config_content: Arc<std::sync::Mutex<String>>, // Задача 184: Хранение содержимого конфига
    config_tx_clone: mpsc::Sender<neirobot_lit::config::types::FullConfig>, // Задача 184: Отправка обновлений конфига
    mut persistence_rx: mpsc::Receiver<()>, // Задача 218: Канал для периодического сохранения состояния
) -> Result<()> 
where S: tokio_stream::Stream<Item = WsData> + Unpin
{
    let mut reconciliation_interval = tokio::time::interval(Duration::from_secs(config.bot.reconciliation_interval_sec));
    let mut latency_interval = tokio::time::interval(Duration::from_secs(config.bot.latency_report_interval_sec));
    let mut heartbeat_tick = tokio::time::interval(Duration::from_millis(1000));
    let mut stop_check_interval = tokio::time::interval(Duration::from_millis(config.bot.stop_check_interval_ms));
    let mut snapshot_interval = tokio::time::interval(Duration::from_millis(config.bot.snapshot_interval_ms));
    let mut stale_order_check_interval = tokio::time::interval(Duration::from_millis(config.bot.stale_check_interval_ms)); // Задача 179
    let mut log_archival_interval = tokio::time::interval(Duration::from_secs(3600)); // Задача 182: Архивация логов раз в час
    let mut data_cleanup_interval = tokio::time::interval(Duration::from_secs(config.bot.risk.cleanup_interval_hours as u64 * 3600)); // Задача 187: Очистка данных
    let mut clock_drift_check_interval = tokio::time::interval(Duration::from_secs(config.bot.risk.clock_sync_interval_s)); // Задача 172: Проверка дрифта часов
    let mut position_sync_interval = tokio::time::interval(Duration::from_secs(config.bot.position_sync_interval_secs)); // Задача 066: Синхронизация позиции
    let mut last_chase_time = 0u64;
    let chase_interval_ms = config.bot.chase_interval_ms.max(200);
    let start_time = neirobot_lit::utils::helpers::unix_ms();
    
    loop {
        // Обновляем Heartbeat для Watchdog (задача 146)
        shared_state.last_heartbeat.store(
            neirobot_lit::utils::helpers::unix_ms(), 
            std::sync::atomic::Ordering::Relaxed
        );
        
        // Задача 219: Обновляем файл heartbeat для внешнего watchdog
        if let Err(e) = neirobot_lit::utils::liveness::write_heartbeat(bot_path, symbol) {
            warn!("Failed to update liveness heartbeat: {}", e);
        }

        // Синхронизация emergency_mode для health-check (задача 135)
        shared_state.emergency_mode.store(
            execution.emergency_mode, 
            std::sync::atomic::Ordering::Relaxed
        );
        
        tokio::select! {
            _ = token.cancelled() => {
                info!("Bot loop stopping due to cancellation...");
                break;
            }
            _ = stop_check_interval.tick() => {
                if let Some(stop_path) = execution.risk_manager.check_manual_stop(bot_path, &config.bot) {
                    error!("MANUAL STOP DETECTED: {}. Shutting down...", stop_path.display());
                    
                    // 1. Экстренное закрытие (задача 109)
                    if let Err(e) = execution.emergency_market_close(rest_client, &config.exchange).await {
                        error!("Emergency market close failed during manual stop: {}", e);
                    }
                    
                    // 2. Попытка переименования файла для подтверждения (ACK)
                    let ack_path = stop_path.with_extension(&config.bot.ack_extension);
                    if let Err(e) = fs::rename(&stop_path, &ack_path) {
                        error!("Failed to rename stop file to {}: {}", ack_path.display(), e);
                        // Даже если переименовать не удалось (права доступа), мы все равно выходим
                    } else {
                        info!("Stop file acknowledged: {}", ack_path.display());
                    }
                    
                    return Ok(()); // Выход из процесса
                }
            }
            _ = latency_interval.tick() => {
                use neirobot_lit::monitoring::latency::{E2E_LATENCY, PROC_LATENCY, HOT_PATH_STATS, LATENCY_MONITOR};

                // Старая статистика (для обратной совместимости)
                let e2e_avg = E2E_LATENCY.get_avg();
                let e2e_max = E2E_LATENCY.get_max();
                let proc_avg = PROC_LATENCY.get_avg();
                let proc_max = PROC_LATENCY.get_max();
                
                let json_avg = HOT_PATH_STATS.get_avg_json_parsing();
                let lob_avg = HOT_PATH_STATS.get_avg_lob();
                let feature_avg = HOT_PATH_STATS.get_avg_feature();
                let inference_avg = HOT_PATH_STATS.get_avg_inference();
                let inference_max = HOT_PATH_STATS.get_max_inference();
                let samples = HOT_PATH_STATS.get_count();

                info!(
                    "[Latency 60s] E2E: avg={:.1}ms, max={}ms | Proc: avg={:.1}us, max={}us",
                    e2e_avg, e2e_max, proc_avg, proc_max
                );
                info!(
                    "[HotPath Stats] JSON: {}us, LOB: {}us, Feat: {}us, Infer: {}us (max {}us), Samples: {}",
                    json_avg, lob_avg, feature_avg, inference_avg, inference_max, samples
                );

                E2E_LATENCY.reset();
                PROC_LATENCY.reset();
                HOT_PATH_STATS.reset();

                // Новая прецизионная статистика через HdrHistogram
                LATENCY_MONITOR.print_report();
                LATENCY_MONITOR.reset();
            }
            _ = heartbeat_tick.tick() => {
                if !execution.risk_manager.check_inactivity(execution.last_book_update) {
                    execution.handle_inactivity_trigger(rest_client, &config.exchange).await?;
                    // Триггер реконнекта для WebSocket (задача 048)
                    ws_reconnect_tx.send(ReconnectSignal::Immediate).await.ok();
                }
                
                // Задача 163: Проверка Time Decay на таймере (каждую секунду)
                // Это обеспечивает выход даже если нет рыночных обновлений
                let position = execution.position_manager.get_position();
                if execution.risk_manager.check_time_stop(position, &config.bot) {
                    let side = if position.qty.is_sign_positive() { "Long" } else { "Short" };
                    let opened_at = position.opened_at.unwrap_or(0);
                    let now = neirobot_lit::utils::helpers::get_unix_ms();
                    let age_ms = now.saturating_sub(opened_at);
                    let limit_ms = if position.qty.is_sign_positive() {
                        config.bot.time_decay.max_age_long_ms
                    } else {
                        config.bot.time_decay.max_age_short_ms
                    };
                    
                    error!(
                        "[Risk] TIME STOP TRIGGERED: Side: {}, Age: {}ms, Limit: {}ms. Executing emergency market close.",
                        side, age_ms, limit_ms
                    );
                    
                    // Инкрементируем метрику
                    if let Some(counter) = neirobot_lit::monitoring::prometheus::TIME_DECAY_EXIT_COUNTER.get() {
                        counter.with_label_values(&[execution.symbol.as_str()]).inc();
                    }
                    
                    if let Err(e) = execution.emergency_market_close(rest_client, &config.exchange).await {
                        error!("Emergency market close failed on time stop: {}", e);
                    }
                }
            }
            _ = reconciliation_interval.tick() => {
                debug!("Periodic reconciliation triggered for {}", symbol);
                if let Err(e) = execution.perform_reconciliation(rest_client, &config.exchange).await {
                    error!("Reconciliation failed for {}: {}", symbol, e);
                }
            }
            _ = position_sync_interval.tick() => {
                // Задача 066: Периодическая синхронизация позиции с биржей
                debug!("Periodic position sync triggered for {}", symbol);
                if let Ok((remote_qty, remote_avg_price, remote_leverage, remote_pnl)) = 
                    rest_client.get_position_signed(&config.exchange.bybit.category, symbol, config.bot.position_idx).await {
                    execution.position_manager.sync_from_remote(
                        remote_qty, 
                        remote_avg_price, 
                        remote_leverage, 
                        remote_pnl,
                        &ob.market_info
                    );
                } else {
                    debug!("Position sync failed for {}, will retry on next interval", symbol);
                }
            }
            _ = snapshot_interval.tick() => {
                // Периодический дамп снимков стакана (задача 132)
                if ob.is_dirty() {
                    let snap = ob.take_snapshot();
                    // Отправка в фоновый поток без блокировки
                    if let Err(e) = snapshot_tx.try_send(snap) {
                        warn!("Failed to send snapshot to writer: {}", e);
                    }
                    ob.reset_dirty();
                }
            }
            api_info = api_check_rx.recv() => {
                if let Some(info) = api_info {
                    info!("Received periodic API key info update.");
                    if let Err(e) = execution.risk_manager.health_monitor.validate_api_permissions(&info, &config.exchange) {
                        error!("PERIODIC API CHECK FAILED: {}. Blocking trading.", e);
                        // Блокировка произойдет автоматически через health_monitor.is_sane() в следующем цикле
                    } else {
                        info!("Periodic API key validation successful.");
                    }
                }
            }
            priv_msg = private_rx.recv() => {
                if let Some(val) = priv_msg {
                    use neirobot_lit::data::parser::parse_private_msg;
                    let json_str = val.to_string();
                    match parse_private_msg(&json_str) {
                        Ok(updates) => {
                            // Задача 191: Используем lock-free снапшот для чтения цен
                            let snap = ob.current_snapshot.load();
                            let (best_bid, best_ask) = snap.get_best_bid_ask();
                            for update in updates {
                                if let Err(e) = execution.handle_order_update(
                                    update, 
                                    rest_client, 
                                    &config.exchange,
                                    Decimal::from_f64(best_bid).unwrap_or_default(),
                                    Decimal::from_f64(best_ask).unwrap_or_default(),
                                    &ob  // Задача 191: Передаем живой стакан для консистентности
                                ).await {
                                    error!("Failed to handle private order update: {}", e);
                                }
                            }
                        }
                        Err(e) => warn!("Failed to parse private WS message: {}", e),
                    }
                }
            }
            msg = stream.next() => {
                let ws_data = match msg {
                    Some(data) => {
                        // WebSocket подключен и работает (задача 135)
                        shared_state.ws_connected.store(true, std::sync::atomic::Ordering::Relaxed);
                        data
                    },
                    None => {
                        error!("Market data stream closed for {}. Attempting to reconnect logic...", symbol);
                        // Обновляем статус подключения (задача 135)
                        shared_state.ws_connected.store(false, std::sync::atomic::Ordering::Relaxed);
                        // В реальном боте здесь может быть логика ожидания реконнекта
                        break; 
                    }
                };

                // Обработка разных типов данных из WebSocket
                match ws_data {
                    WsData::OrderBook(update) => {
                        if execution.emergency_mode {
                            continue;
                        }
                        execution.poke_book_activity();
                        
                        // Обновляем last_update для health-check (задача 135)
                        shared_state.last_update.store(
                            neirobot_lit::utils::helpers::unix_ms(), 
                            std::sync::atomic::Ordering::Relaxed
                        );
                        
                        if let Err(e) = handle_market_update(update, &mut ob, &mut tensor_builder, engine, execution, rest_client, &config.exchange, ws_reconnect_tx.clone()).await {
                            if e.to_string().contains("Checksum mismatch") {
                                // При ошибке checksum — выходим из цикла, чтобы сработал реконнект
                                error!("[{}] Checksum error, restarting loop...", symbol);
                                break;
                            }
                            if execution.risk_manager.is_blocked {
                                error!("[{}] HARD STOP: RiskManager blocked. Triggering emergency market close...", symbol);
                                if let Err(he) = execution.emergency_market_close(rest_client, &config.exchange).await {
                                    error!("[{}] Emergency procedures failed: {}", symbol, he);
                                }
                                break; // Останавливаем торговлю
                            }
                            error!("[{}] Error processing market update: {:?}", symbol, e);
                        }
                    }
                    WsData::Trades(trades) => {
                        // Обновление статистики VWAP/TWAP (Задача 106)
                        for trade in &trades {
                            execution.on_public_trade(trade.clone());
                        }
                        
                        // Задача 236: Отправка сделок в канал для записи в Parquet
                        for trade in trades {
                            if let Err(e) = trades_tx.try_send(trade) {
                                debug!("[{}] Failed to send trade to dump channel: {}", symbol, e);
                            }
                        }
                    }
                    WsData::Ticker(ticker) => {
                        // Обновление информации о фандинге (Задача 170)
                        // Получаем mark_price из orderbook для расчета фандинга
                        let mark_price = Decimal::from_f64(ob.current_snapshot.load().mark_price).unwrap_or_default();
                        execution.update_funding_info(ticker.funding_rate, ticker.next_funding_time, mark_price);
                    }
                    WsData::MarkPrice(symbol, mark_price) => {
                        // Задача 233: Обновление маркированной цены
                        if symbol.as_ref() == ob.symbol {
                            ob.set_mark_price(mark_price);
                            debug!("[{}] Updated mark price: {}", symbol, mark_price);
                        }
                    }
                }

                // Логика «погони» (Order Chasing) после обработки рыночных данных с троттлингом
                let now = neirobot_lit::utils::timestamp_ms();
                if now - last_chase_time >= chase_interval_ms {
                    // Задача 191: Используем lock-free снапшот для чтения цен
                    let snap = ob.current_snapshot.load();
                    let (best_bid, best_ask) = snap.get_best_bid_ask();
                    if best_bid > 0.0 && best_ask > 0.0 {
                        // Задача 210: Передаем orderbook для адаптивных порогов
                        if let Err(e) = execution.check_and_chase(
                            rest_client, 
                            &config.exchange,
                            Decimal::from_f64(best_bid).unwrap_or_default(),
                            Decimal::from_f64(best_ask).unwrap_or_default(),
                            &ob  // Задача 191: Передаем живой стакан для адаптивных порогов
                        ).await {
                            error!("Failed to check and chase orders: {}", e);
                        }
                    }
                    last_chase_time = now;
                }
            }
            _ = stale_order_check_interval.tick() => {
                // Задача 179: Периодическая проверка "зависших" ордеров
                if let Err(e) = execution.health_monitor.check_stale_orders(
                    &rest_client,
                    &config.bot,
                    &config.exchange,
                    &mut execution.risk_manager.active_intents,
                    &mut execution.order_manager,
                    &mut execution.risk_manager,
                ).await {
                    error!("Failed to check stale orders: {}", e);
                }
            }
            _ = log_archival_interval.tick() => {
                // Задача 182: Периодическая архивация логов (раз в час)
                if let Err(e) = execution.health_monitor.run_log_archival_task(
                    config.logging.log_retention_days
                ).await {
                    error!("Failed to run log archival task: {}", e);
                }
            }
            _ = clock_drift_check_interval.tick() => {
                // Задача 172: Периодическая проверка дрифта часов
                if let Err(e) = execution.health_monitor.check_clock_drift(
                    &config.exchange.base_url
                ).await {
                    error!("Clock drift check failed: {}", e);
                }
            }
            _ = data_cleanup_interval.tick() => {
                // Задача 187: Периодическая очистка данных
                let data_dir = bot_path.join("data").join("raw");
                let config_clone = config.bot.risk.clone();
                
                tokio::task::spawn_blocking(move || {
                    match neirobot_lit::risk::health_monitor::HealthMonitor::perform_data_cleanup(&data_dir, &config_clone) {
                        Ok((freed_bytes, deleted_count)) => {
                            let freed_gb = freed_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                            info!(
                                "[Health] Data cleanup finished. Freed: {:.2} GB, Deleted files: {}",
                                freed_gb, deleted_count
                            );
                        }
                        Err(e) => {
                            warn!("[Health] Data cleanup failed: {}", e);
                        }
                    }
                });
            }
            Some(new_config) = config_rx.recv() => {
                // Задача 184: Получение обновленной конфигурации через SIGHUP
                info!("[Audit] Applying config update in run_bot_loop");
                
                // Обновляем конфигурацию в execution engine (каскадно обновляет все компоненты)
                if let Err(e) = execution.update_config(&new_config.bot) {
                    error!("[Audit] Failed to apply config update: {}", e);
                } else {
                    info!("[Audit] Config update applied successfully");
                }
            }
            Some(remote_balance) = balance_sync_rx.recv() => {
                // Задача 221: Синхронизация баланса с биржей для устранения дрейфа
                debug!("[Equity Streamer] Received balance sync: {}", remote_balance);
                execution.position_manager.sync_balance(remote_balance);
            }
            Some(_) = cleanup_trigger_rx.recv() => {
                // Задача 235: Выполнение процедуры очистки зависших ордеров
                info!("[Cleanup] Running stale order cleanup routine for {}", symbol);
                if let Err(e) = execution.order_manager.run_cleanup_routine(
                    rest_client,
                    &mut execution.risk_manager,
                    &config.bot,
                    &config.exchange,
                ).await {
                    error!("[Cleanup] Cleanup routine failed: {:?}", e);
                } else {
                    info!("[Cleanup] Cleanup routine completed successfully");
                }
            }
            Some(_) = persistence_rx.recv() => {
                // Задача 218: Периодическое сохранение состояния
                tracing::debug!("[Persistence] Periodic state save for {}", symbol);
                if let Err(e) = execution.save_current_state().await {
                    tracing::error!("[Persistence] Failed to save state: {}", e);
                }
            }
            Some(metrics) = metrics_rx.recv() => {
                // Задача 225: Обработка системных метрик от ResourceProfiler
                execution.on_system_metrics(metrics);
            }
            Some(cmd) = command_rx.recv() => {
                // Задача 229: Обработка команд удаленного управления
                match cmd {
                    Command::Panic => {
                        error!("[Command] PANIC command received! Executing emergency market close...");
                        
                        // Экстренное закрытие позиции
                        if let Err(e) = execution.emergency_market_close(rest_client, &config.exchange).await {
                            error!("[Command] Emergency market close failed: {}", e);
                        }
                        
                        // Обновляем статус
                        {
                            let mut status = status_state.write();
                            status.status = "panic_stopped".to_string();
                        }
                        
                        info!("[Command] PANIC command completed. Shutting down...");
                        return Ok(());
                    }
                    Command::Pause => {
                        info!("[Command] PAUSE command received. Pausing trading...");
                        
                        // Блокируем торговлю через RiskManager
                        execution.risk_manager.is_blocked = true;
                        execution.emergency_mode = true;
                        
                        // Обновляем статус
                        {
                            let mut status = status_state.write();
                            status.status = "paused".to_string();
                        }
                        
                        info!("[Command] Trading paused. Bot will not open new positions.");
                    }
                    Command::Reload => {
                        info!("[Command] RELOAD command received. Reloading configuration...");
                        
                        // Перезагружаем конфигурацию (аналогично SIGHUP)
                        let config_path = PathBuf::from("bots").join(symbol).join("config.toml");
                        match load_full_config(Path::new("."), &config_path) {
                            Ok(new_config) => {
                                // Читаем новое содержимое из файла
                                let new_content = std::fs::read_to_string(&config_path).unwrap_or_default();
                                
                                // Получаем старое содержимое из памяти
                                let old_content = config_content.lock().unwrap().clone();
                                
                                // Вычисляем хэши для сравнения
                                let old_hash = neirobot_lit::config::loader::compute_config_hash(&old_content);
                                let new_hash = neirobot_lit::config::loader::compute_config_hash(&new_content);
                                
                                // Логируем изменения
                                if old_hash != new_hash {
                                    info!("[Audit] Config SHA-256 changed: {} -> {}", old_hash, new_hash);
                                    
                                    // Генерируем diff используя СТАРОЕ содержимое из памяти
                                    let diff = neirobot_lit::config::loader::generate_config_diff(&old_content, &new_content);
                                    if !diff.is_empty() {
                                        info!("[Audit] Config diff:\n{}", diff);
                                    }
                                } else {
                                    info!("[Audit] Config hash unchanged: {}", new_hash);
                                }
                                
                                // Отправляем новую конфигурацию через config_tx
                                if let Err(e) = config_tx_clone.send(new_config).await {
                                    error!("[Command] Failed to send config update: {}", e);
                                } else {
                                    info!("[Command] Configuration reloaded successfully");
                                    
                                    // Обновляем текущее содержимое после успешной отправки
                                    *config_content.lock().unwrap() = new_content;
                                }
                            }
                            Err(e) => {
                                error!("[Command] Failed to reload configuration: {}", e);
                            }
                        }
                    }
                    Command::GetStatus => {
                        // Эта команда используется только внутри для обновления статуса
                        // Обновление статуса происходит автоматически в цикле
                        debug!("[Command] Status update requested");
                    }
                }
            }
        }
        
        // Задача 229: Периодическое обновление статуса для command server
        {
            let position = execution.position_manager.get_position();
            let mid_price = ob.current_snapshot.load().get_mid_price();
            let mid_dec = Decimal::from_f64(mid_price).unwrap_or_default();
            let pnl = execution.position_manager.get_total_pnl(mid_dec);
            let now = neirobot_lit::utils::helpers::unix_ms();
            
            let mut status = status_state.write();
            status.uptime_secs = (now - start_time) / 1000;
            status.pnl = Some(pnl.to_f64().unwrap_or(0.0));
            status.position = Some(position.qty.to_f64().unwrap_or(0.0));
            
            // Получаем latency из мониторинга
            use neirobot_lit::monitoring::latency::LATENCY_MONITOR;
            status.latency_ms = Some(LATENCY_MONITOR.get_last_e2e_ms());

            status.status = if execution.emergency_mode {
                "emergency".to_string()
            } else if execution.risk_manager.is_blocked {
                "blocked".to_string()
            } else {
                "running".to_string()
            };
        }
        }
    }
    
    Ok(())
}

/// Координирует поток данных: OrderBook -> Tensor -> Inference -> Execution
async fn handle_market_update(
    update: neirobot_lit::data::types::OrderBookUpdateArc,
    ob: &mut OrderBook,
    tensor_builder: &mut TensorBuilder,
    engine: &mut OnnxEngine,
    execution: &mut ExecutionEngine,
    rest_client: &neirobot_lit::trading::BybitRestClient,
    exchange_config: &neirobot_lit::config::types::ExchangeConfig,
    ws_reconnect_tx: mpsc::Sender<crate::data::websocket::ReconnectSignal>,
) -> Result<()> {
    use chrono::Utc;
    
    // Замер времени начала обработки
    let start_ts = std::time::Instant::now();
    
    // 1. Замер сетевой задержки (из WS сообщения) с защитой от clock skew
    let now_ms = Utc::now().timestamp_millis();
    let exchange_ts = update.timestamp_ms as i64;
    let network_micros = if now_ms >= exchange_ts {
        ((now_ms - exchange_ts) * 1000) as u64
    } else {
        tracing::warn!("Clock skew detected: local < exchange_ts");
        0
    };

    // 2. Применяем обновление в стакан
    let start_lob = std::time::Instant::now();
    ob.apply_update(&update);
    use neirobot_lit::monitoring::latency::HOT_PATH_STATS;
    HOT_PATH_STATS.record_lob(start_lob.elapsed().as_micros() as u64);

    // 2.5. Сброс состояния при получении полного снапшота через WebSocket (Задача 180)
    if update.is_snapshot {
        tracing::info!(
            "[{}] Full snapshot received via WebSocket (u={}). Resetting corruption state.",
            execution.symbol,
            update.last_update_id
        );
        execution.risk_manager.health_monitor.reset_corruption();
    }

    // 3. Валидация Checksum ПОСЛЕ применения обновления (Задача 180)
    // Bybit требует проверять контрольную сумму после применения обновления к стакану
    if execution.risk_manager.config.checksum_validation_enabled {
        if let Some(expected_cs) = update.checksum {
            if !ob.verify_checksum(expected_cs) {
                // Вызываем health_monitor для накопления ошибок
                let limit_reached = execution.risk_manager.health_monitor.checksum_mismatch();
                
                if limit_reached {
                    // Лимит превышен - выполняем полное восстановление
                    tracing::error!(
                        "[{}] Max checksum mismatches reached! Clearing OrderBook and triggering reconnect...", 
                        execution.symbol
                    );
                    
                    // Очищаем стакан перед реконнектом
                    ob.clear();
                    
                    // Отправляем сигнал на немедленный реконнект в WebSocket
                    let _ = ws_reconnect_tx.send(crate::data::websocket::ReconnectSignal::Immediate).await;
                    
                    return Err(anyhow::anyhow!("Checksum mismatch limit exceeded"));
                }
                // Если лимит не достигнут, продолжаем работу (мягкая обработка ошибки)
            }
        }
    }
    
    // Задача 191: Получаем lock-free снапшот для чтения данных
    // Это гарантирует, что весь цикл Inference -> Execution работает с одним неизменным срезом данных
    let snapshot = ob.current_snapshot.load();
    
    // 3.2. Проверка последовательности обновлений (Задача 171)
    if execution.risk_manager.health_monitor.check_u(update.last_update_id) {
        tracing::warn!("[{}] OrderBook sequence gap detected. Initiating Resync...", execution.symbol);
        
        // 1. Запрос нового снимка через REST
        match rest_client.fetch_orderbook(
            &exchange_config.bybit.category,
            &execution.symbol,
            50
        ).await {
            Ok(snapshot) => {
                // 2. Сброс стакана и применение снимка
                ob.apply_update(&snapshot);
                
                // 3. Сброс флага коррупции в HealthMonitor и разблокировка RiskManager
                execution.risk_manager.health_monitor.reset_corruption();
                execution.risk_manager.is_blocked = false;
                
                tracing::info!(
                    "[{}] Resync completed successfully. New last_update_id: {}", 
                    execution.symbol, snapshot.last_update_id
                );
                return Ok(());
            },
            Err(e) => {
                tracing::error!("[{}] Resync failed: {}. Critical shutdown.", execution.symbol, e);
                return Err(e);
            }
        }
    }
    
    let price = snapshot.get_mid_price();
    if price == 0.0 {
        return Ok(()); // Ждем заполнения стакана
    }
    
    // 3.2. Глобальная проверка рисков и здоровья (Задача 171)
    let mid_dec = Decimal::from_f64(price).unwrap_or_default();
    let current_pnl = execution.position_manager.get_total_pnl(mid_dec);
    execution.risk_manager.check_risk_gates(current_pnl)?;

    // 3.5. Проверка Time Decay Stop (задача 163)
    // Проверяем возраст позиции и принудительно закрываем если время вышло
    let position = execution.position_manager.get_position();
    if execution.risk_manager.check_time_stop(position, &execution.bot_config) {
        // Получаем детали для логирования
        let side = if position.qty.is_sign_positive() { "Long" } else { "Short" };
        let opened_at = position.opened_at.unwrap_or(0);
        let now = neirobot_lit::utils::timestamp_ms();
        let age_ms = now.saturating_sub(opened_at);
        let limit_ms = if position.qty.is_sign_positive() {
            execution.bot_config.time_decay.max_age_long_ms
        } else {
            execution.bot_config.time_decay.max_age_short_ms
        };

        tracing::error!(
            "[Risk] TIME STOP TRIGGERED: Side: {}, Age: {}ms, Limit: {}ms. Executing emergency market close.",
            side, age_ms, limit_ms
        );

        // Инкрементируем метрику для Prometheus (задача 163)
        if let Some(counter) = neirobot_lit::monitoring::prometheus::TIME_DECAY_EXIT_COUNTER.get() {
            counter.with_label_values(&[symbol]).inc();
        }

        // Экстренное закрытие позиции
        execution.emergency_market_close(rest_client, exchange_config).await
            .context("Emergency market close failed after time stop")?;
        
        return Ok(());
    }

    // 4. Задача 161: Обновление и детекция режима рынка (ПЕРЕД инференсом)
    let current_regime = if let Some(ref mut detector) = execution.regime_detector {
        detector.update(&ob, now_ms as u64);
        detector.detect()
    } else {
        neirobot_lit::config::types::RegimeId::Unknown
    };
    
    // Преобразуем RegimeId в usize для передачи в инференс
    let regime_id = match current_regime {
        neirobot_lit::config::types::RegimeId::Quiet => 0,
        neirobot_lit::config::types::RegimeId::Trend => 1,
        neirobot_lit::config::types::RegimeId::Volatile => 2,
        neirobot_lit::config::types::RegimeId::Unknown => 255,
    };

    // 4. ML Pipeline (Задача №197: Использование пре-аллоцированных буферов и Zero-copy)
    
    // Заполняем буфер признаками из стакана и нормализуем их на месте (Zero-copy версия, Задача 078.3)
    let start_feature = std::time::Instant::now();
    let mut buffer = vec![0.0f32; 150];
    let feature_res = tensor_builder.process_snapshot_to_buffer(&snapshot, &mut buffer);
    HOT_PATH_STATS.record_feature(start_feature.elapsed().as_micros() as u64);
    
    if let Some(tensor_data) = feature_res.context("Tensor building failed")? {
        // Копируем данные в input_view для инференса
        let mut input_view = engine.get_input_view_mut();
        let flat_data = input_view.as_slice_mut().unwrap();
        flat_data.copy_from_slice(&tensor_data);
        
        // Замер инференса
        let start_inf = std::time::Instant::now();
        
        // Выполняем инференс напрямую из заполненного буфера (Zero-copy)
        let mut inference = engine.predict_with_buffer(Some(regime_id)).context("Inference prediction failed")?;
        let inference_micros = start_inf.elapsed().as_micros() as u64;
        
        // Задача 169: Устанавливаем timestamp источника сигнала (receive_ts из snapshot)
        // Это время получения исходного снепшота стакана для проверки свежести сигнала
        inference.source_timestamp_ms = snapshot.timestamp_ms as u64;

        // Логирование высокой задержки (задача 047, задача 082)
        if inference_micros > 50_000 {
            tracing::warn!("High inference latency: {}μs (threshold: 50ms) for {}", inference_micros, execution.symbol);
        }

        // 5. Проверка через RiskManager (Задача 116)
        if !execution.risk_manager.check_latency(network_micros, inference_micros, &execution.bot_config) {
            // Если слишком много отказов подряд — уходим в Waiting Mode
            if execution.risk_manager.consecutive_latency_rejects >= execution.bot_config.max_latency_rejects_limit {
                execution.handle_inactivity_trigger(rest_client, exchange_config).await
                    .context("Failed to trigger inactivity mode after latency rejects")?;
                tracing::error!("FATAL LATENCY: Too many rejects. Trading suspended.");
            }
            return Ok(());
        }

        // Извлекаем лучшие цены для Execution из снапшота (Задача 191)
        let (best_bid, bid_vol, best_ask, ask_vol) = snapshot.get_best_bid_ask_with_vol();

        // 5.5. Проверка условий частичной фиксации прибыли при обновлении orderbook (Задача 166)
        let mid_price = (Decimal::from_f64(best_bid).unwrap_or_default() + Decimal::from_f64(best_ask).unwrap_or_default()) / Decimal::from(2);
        execution.on_ob_update(
            mid_price,
            Decimal::from_f64(best_bid).unwrap_or_default(),
            Decimal::from_f64(best_ask).unwrap_or_default(),
            rest_client,
            exchange_config
        ).await.context("OrderBook update processing failed")?;

        // 6. Execution logic - Задача 161: передаем current_regime для динамических порогов
        execution.on_inference_output(
            inference, 
            price, 
            Decimal::from_f64(best_bid).unwrap_or_default(), 
            Decimal::from_f64(bid_vol).unwrap_or_default(),
            Decimal::from_f64(best_ask).unwrap_or_default(),
            Decimal::from_f64(ask_vol).unwrap_or_default(),
            &ob,  // Задача 191: Передаем живой стакан вместо снапшота для консистентности
            &update,
            rest_client,
            exchange_config,
            current_regime  // Задача 161: текущий режим рынка для динамических порогов
        ).await.context("Trading execution failed")?;
    }

    // 7. Запись статистики задержек в HdrHistogram
    let proc_micros = start_ts.elapsed().as_micros() as u64;
    let total_micros = network_micros + proc_micros;
    
    use neirobot_lit::monitoring::latency::LATENCY_MONITOR;
    LATENCY_MONITOR.record_network(network_micros);
    LATENCY_MONITOR.record_processing(proc_micros);
    LATENCY_MONITOR.record_total(total_micros);

    // 7.1. Обновление задержки в HealthMonitor (Задача 171)
    execution.risk_manager.health_monitor.update_latency(total_micros as f64 / 1000.0);

    Ok(())
}

/// Настройка глобального обработчика паник для экстренной отмены ордеров
/// Вызывается в любом потоке при критическом сбое программы


/// Задача 199: Replay Mode Loop
async fn run_replay_loop(
    path: PathBuf,
    mut ob: OrderBook,
    mut tensor_builder: TensorBuilder,
    engine: &mut OnnxEngine,
    execution: &mut ExecutionEngine,
    symbol: &str,
    rest_client: &BybitRestClient,
    config: &FullConfig,
) -> Result<()> {
    use neirobot_lit::data::parquet::FastParquetScanner;
    use neirobot_lit::data::types::{OrderBookUpdateArc, PriceLevel};
    use smallvec::SmallVec;
    use std::sync::Arc;

    info!("Starting Replay Mode from {:?}", path);
    // Используем батч 1000 для баланса скорости и памяти
    let scanner = FastParquetScanner::new(path.to_str().unwrap(), 1000);
    let batches = scanner.get_batches().context("Failed to initialize parquet scanner")?;
    
    // Канал для реконнекта (не используется в replay, но нужен для handle_market_update)
    let (ws_reconnect_tx, _) = mpsc::channel(1);
    
    let mut processed_count = 0;
    
    for df in batches {
        let height = df.height();
        
        // Получаем колонки как Series (быстрый доступ через ChunkedArray)
        let ts_col = df.column("timestamp_ms")?.i64()?;
        let id_col = df.column("last_update_id")?.i64()?;
        
        // Подготавливаем колонки цен и объемов (50 уровней)
        let mut ask_p_series = Vec::with_capacity(50);
        let mut ask_v_series = Vec::with_capacity(50);
        let mut bid_p_series = Vec::with_capacity(50);
        let mut bid_v_series = Vec::with_capacity(50);
        
        for i in 0..50 {
            ask_p_series.push(df.column(&format!("ask_p_{}", i))?.f64()?);
            ask_v_series.push(df.column(&format!("ask_v_{}", i))?.f64()?);
            bid_p_series.push(df.column(&format!("bid_p_{}", i))?.f64()?);
            bid_v_series.push(df.column(&format!("bid_v_{}", i))?.f64()?);
        }
        
        for row_idx in 0..height {
            let timestamp_ms = ts_col.get(row_idx).unwrap_or(0) as u64;
            let last_update_id = id_col.get(row_idx).unwrap_or(0) as u64;
            
            let mut asks = SmallVec::with_capacity(50);
            let mut bids = SmallVec::with_capacity(50);
            
            for i in 0..50 {
                let ap = ask_p_series[i].get(row_idx).unwrap_or(0.0);
                let av = ask_v_series[i].get(row_idx).unwrap_or(0.0);
                let bp = bid_p_series[i].get(row_idx).unwrap_or(0.0);
                let bv = bid_v_series[i].get(row_idx).unwrap_or(0.0);
                
                // Добавляем уровень, только если цена > 0
                if ap > 0.0 {
                    asks.push(PriceLevel { price: ap, size: av });
                }
                if bp > 0.0 {
                    bids.push(PriceLevel { price: bp, size: bv });
                }
            }
            
            let update = OrderBookUpdateArc {
                symbol: Arc::from(symbol),
                timestamp_ms,
                last_update_id,
                is_snapshot: true, 
                bids,
                asks,
                checksum: None,
            };
            
            // Вызываем основной обработчик рынка для каждого ряда
            handle_market_update(
                update,
                &mut ob,
                &mut tensor_builder,
                engine,
                execution,
                rest_client,
                &config.exchange,
                ws_reconnect_tx.clone(),
            ).await?;
            
            processed_count += 1;
        }
        
        if processed_count % 10000 == 0 {
            info!("Replay progress: {} rows processed...", processed_count);
        }
    }
    
    info!("Replay finished successfully. Processed {} rows.", processed_count);
    Ok(())
}
