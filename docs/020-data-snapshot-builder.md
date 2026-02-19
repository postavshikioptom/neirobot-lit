# 020 - Data Snapshot Builder

Цель задачи: Реализовать «дирижёр» модуля выгрузки данных в src/data/snapshot.rs. Этот компонент объединяет WebSocket-клиент, парсер, хранилище OrderBook и ParquetDumper. Билдер должен обеспечивать равномерную выгрузку снимков стакана с фиксированным шагом времени (Time Step), что критично для обучения рекуррентных и трансформерных моделей (LiT).

Файлы для изменения/создания:

src/data/snapshot.rs (создать)
src/data/mod.rs (обновить)
Инструкции для Gemini:

src/data/snapshot.rs: Реализовать функцию run_snapshot_pipeline с использованием tokio::select! и поддержкой плавного завершения (Graceful Shutdown).

use tokio::sync::mpsc;
use tokio::time::{interval, Duration};
use tokio::signal;
use crate::data::orderbook::OrderBook;
use crate::data::dump::ParquetDumper;
use crate::data::parser::parse_orderbook_msg;
use crate::data::websocket::BybitWsClient;
use crate::config::types::FullConfig;
use crate::utils::time::timestamp_ms;
use anyhow::Result;
use tracing::{info, warn, error};

pub async fn run_snapshot_pipeline(config: FullConfig) -> Result<()> {
    let symbol = config.bot.symbol.clone();
    let interval_ms = config.trading.snapshot_interval_ms; // Берём из конфига (задача 002)
    
    let mut ob = OrderBook::new(&symbol);
    let output_dir = std::path::PathBuf::from("bots").join(&symbol).join("data").join("raw");
    let mut dumper = ParquetDumper::new(&symbol, &output_dir, 5000)?;

    let (tx, mut rx) = mpsc::channel::<String>(1000);
    let ws_client = BybitWsClient::new(config.exchange.clone(), symbol.clone());

    // 1. Запуск WebSocket в фоне
    tokio::spawn(async move {
        if let Err(e) = ws_client.run(tx).await {
            error!("[{}] WebSocket runner exited with error: {}", symbol, e);
        }
    });

    // 2. Настройка таймера (фиксированный шаг для обучения модели)
    let mut timer = interval(Duration::from_millis(interval_ms));
    info!("Starting snapshot pipeline for {}. Interval: {}ms", symbol, interval_ms);

    loop {
        tokio::select! {
            // Получаем рыночные данные
            Some(raw_msg) = rx.recv() => {
                match parse_orderbook_msg(&raw_msg) {
                    Ok(Some(update)) => ob.apply_update(&update),
                    Ok(None) => {}, // Сервисные сообщения
                    Err(e) => warn!("[{}] Parse error: {}", symbol, e),
                }
            }
            // Тик таймера: всегда записываем состояние (даже если цена не менялась)
            _ = timer.tick() => {
                if ob.last_update_id != 0 {
                    let flat_lob = ob.get_flat_snapshot();
                    // Записываем exchange timestamp и id для синхронизации
                    if let Err(e) = dumper.push_snapshot(ob.timestamp_ms, ob.last_update_id, flat_lob) {
                        error!("[{}] Dumper error: {}", symbol, e);
                    }
                }
            }
            // Обработка Ctrl+C для сохранения данных
            _ = signal::ctrl_c() => {
                info!("[{}] Shutdown signal received. Flushing data...", symbol);
                dumper.flush()?;
                break;
            }
        }
    }

    Ok(())
}
Технические требования:

Конфигурируемость: Интервал выгрузки должен браться из config.trading.snapshot_interval_ms.
Равномерность: Записывать снимок на каждом тике таймера (если last_update_id > 0). Это создает идеальный Time Series для ML (модель видит и активные фазы, и фазы затишья).
Graceful Shutdown: При получении сигнала прерывания вызвать dumper.flush(), чтобы не потерять последний чанк данных в буфере.
Отказоустойчивость: Ошибки парсинга или записи логгировать через warn!/error!, но не прерывать основной цикл.
Почему это важно: Этот модуль — "сердце" системы сбора данных. Он превращает хаотичные и асинхронные WebSocket-сообщения в упорядоченную структуру. Фиксированный интервал времени (например, 100 мс) позволяет нам не передавать dt (изменение времени) в нейросеть, так как шаг всегда константен.
