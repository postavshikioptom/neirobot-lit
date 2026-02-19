# 045 - Bin Run Bot Main
Цель задачи: Реализовать основной исполняемый файл бота src/bin/run-bot.rs. Бинарный файл объединяет загрузку конфигурации, секретов .env, инициализацию ML-компонентов и асинхронный цикл обработки данных из WebSocket для принятия торговых решений.

Файлы: src/bin/run-bot.rs (создать)

Инструкции для Gemini:

Инициализация окружения: Загрузить .env через dotenvy, инициализировать логгер и загрузить FullConfig.
Компоненты ML: Создать Normalizer, OnnxEngine (с features_dim из конфига) и TensorBuilder.
Компоненты Trading: Создать RiskManager (с initial_balance из конфига) и ExecutionEngine (с порогами и флагом закрытия).
Event Loop:
Запустить BybitWsClient в tokio::spawn с каналом mpsc (емкость 1024).
Обрабатывать сообщения: parse -> apply_update -> checksum (placeholder) -> build_tensor -> predict -> execution.
Реализовать Graceful Shutdown через tokio::signal::ctrl_c().
use clap::Parser;
use std::path::{Path, PathBuf};
use tokio::sync::mpsc;
use tracing::{info, warn, error};
use crate::config::loader::load_full_config;
use crate::data::websocket::BybitWsClient;
use crate::data::parser::parse_orderbook_update;
use crate::ml::{OnnxEngine, TensorBuilder, Normalizer};
use crate::trading::{ExecutionEngine, RiskManager};

#[derive(Parser)]
struct Args {
    symbol: String,
    #[arg(short, long)]
    config: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Load Env & Config
    dotenvy::dotenv().ok();
    let args = Args::parse();
    let config_path = args.config.unwrap_or_else(|| PathBuf::from("bots").join(&args.symbol).join("config.toml"));
    let full_config = load_full_config(&config_path)?;
    crate::utils::init_logger(&args.symbol, &full_config.logging)?;

    // 2. Init ML (using features_dim from config)
    let model_dir = &full_config.bot.model_path;
    let normalizer = Normalizer::load(Path::new("docs/data_schema.json"), &model_dir.join("norm.json"))?;
    let engine = OnnxEngine::load(
        &model_dir.join("lit.onnx"), 
        full_config.bot.seq_len, 
        full_config.bot.features_dim
    )?;
    let mut tensor_builder = TensorBuilder::new(normalizer, full_config.bot.seq_len);

    // 3. Init Trading (using initial_balance and thresholds from config)
    let risk_manager = RiskManager::new(full_config.risk.clone(), full_config.bot.initial_balance);
    let mut execution = ExecutionEngine::new(
        args.symbol.clone(),
        risk_manager,
        full_config.bot.close_on_flat,
        full_config.bot.threshold_buy,
        full_config.bot.threshold_sell,
    );

    // 4. Background WS Client (Capacity 1024)
    let (tx, mut rx) = mpsc::channel(1024);
    let ws_client = BybitWsClient::new(full_config.exchange.clone(), args.symbol.clone());
    tokio::spawn(async move {
        if let Err(e) = ws_client.run(tx).await {
            error!("WS Client fatal error: {}", e);
        }
    });

    // 5. Main Loop with Signal Handling
    let mut ob = crate::data::OrderBook::new(&args.symbol);
    info!("Bot started for {}. Waiting for data...", args.symbol);

    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                info!("Shutdown signal received. Closing...");
                // TODO: execution.close_all_positions() (Phase 11)
                break;
            }
            msg = rx.recv() => {
                let text = match msg {
                    Some(t) => t,
                    None => { warn!("Channel closed, exiting..."); break; }
                };

                if let Some(update) = parse_orderbook_update(&text, &args.symbol)? {
                    // TODO: Checksum validation (Task 049)
                    ob.apply_update(update);
                    
                    let price = ob.get_mid_price();
                    if price == 0.0 { continue; }

                    if let Some(tensor) = tensor_builder.process_snapshot(&ob)? {
                        let inference = engine.predict(&tensor)?;
                        execution.on_inference_output(inference, price)?;
                    }
                }
            }
        }
    }

    Ok(())
}
Технические требования:

Динамика: Все параметры (features_dim, initial_balance, thresholds) берутся из full_config.
Имена методов: tensor_builder.process_snapshot и execution.on_inference_output согласно задачам 035 и 044.
Безопасность: Загрузка секретов через dotenvy перед стартом клиентов.
Отказоустойчивость: tokio::select! для одновременной обработки рыночных данных и сигнала завершения