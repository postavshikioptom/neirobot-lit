# 046 - Run Bot Init Flow
Цель задачи: Реализовать надежный и упорядоченный процесс инициализации в src/bin/run-bot.rs. Бот должен следовать строгому алгоритму: загрузка окружения -> инициализация локальных компонентов -> проверка ML-модели -> установка соединения. При любом сбое на этапе инициализации бот должен завершаться с ошибкой (fail-fast).

Файлы: src/bin/run-bot.rs (обновить)

Инструкции для Gemini:

Последовательность в main:

Env: Загрузить .env через dotenvy::dotenv().ok().
Config: Получить путь из Args и загрузить load_full_config.
Logger: Инициализировать init_logger сразу после загрузки конфига.
Secrets: Получить BYBIT_API_KEY и BYBIT_API_SECRET через std::env::var, используя .context() для информативности.
LOB: Создать экземпляр OrderBook::new(&args.symbol).
ML: Загрузить Normalizer, OnnxEngine и TensorBuilder. Вывести в лог seq_len и features_dim из конфига.
Trading: Создать RiskManager (с initial_balance из config.bot) и ExecutionEngine.
WebSocket: Создать BybitWsClient и запустить его.
Проверки:

Все этапы обернуть в anyhow::Context.
Логировать параметры стратегии: threshold_buy, threshold_sell, close_on_flat.
// Пример логики инициализации:
dotenvy::dotenv().ok();
let args = Args::parse();
let config = load_full_config(&config_path).context("Failed to load config")?;
init_logger(&args.symbol, &config.logging)?;

let api_key = std::env::var("BYBIT_API_KEY").context("BYBIT_API_KEY must be set in .env")?;
let api_secret = std::env::var("BYBIT_API_SECRET").context("BYBIT_API_SECRET must be set in .env")?;

let mut ob = OrderBook::new(&args.symbol);
info!("LOB initialized for {}", args.symbol);

let engine = OnnxEngine::load(&model_path, config.bot.seq_len, config.bot.features_dim)
    .context("Failed to load ONNX model")?;
info!("ML Engine loaded: seq_len={}, features={}", config.bot.seq_len, config.bot.features_dim);

let risk_manager = RiskManager::new(config.risk.clone(), config.bot.initial_balance);
let mut execution = ExecutionEngine::new(
    args.symbol.clone(),
    risk_manager,
    config.bot.close_on_flat,
    config.bot.threshold_buy,
    config.bot.threshold_sell,
);

// Spawn WS и переход в основной цикл (select!)
Технические требования:

Fail-fast: Использовать оператор ? с контекстом на каждом шаге.
Изоляция: OrderBook создается до входа в асинхронный цикл.
Конфиг: Все лимиты и параметры (включая initial_balance) — только из FullConfig.
Логирование: Скрывать значения API-ключей в логах.