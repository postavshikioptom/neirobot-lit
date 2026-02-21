use tokio::sync::mpsc;
use tokio::time::{interval, Duration};
use tokio::signal;
use crate::data::orderbook::OrderBook;
use crate::data::dump::ParquetDumper;
use crate::data::websocket::BybitWsClient;
use crate::data::types::WsData;
use crate::config::types::FullConfig;
use anyhow::Result;
use tracing::{info, error};

/// Основной пайплайн для сбора данных в реальном времени и записи снапшотов стакана.
pub async fn run_snapshot_pipeline(config: FullConfig) -> Result<()> {
    let symbol = config.bot.symbol.clone();
    let interval_ms = config.trading.snapshot_interval_ms;
    
    // Инициализация компонентов
    let mut ob = OrderBook::new(&symbol);
    let output_dir = std::path::PathBuf::from("bots").join(&symbol).join("data").join("raw");
    
    // Создаем dumper с буфером на 5000 записей (примерно 8 минут при 100мс)
    let mut dumper = ParquetDumper::new(&symbol, &output_dir, 5000)?;

    // Создаем канал для получения обновлений стакана от WebSocket клиента
    // Используем типизированный WsData, так как парсинг уже настроен в WS клиенте
    let (tx, mut rx) = mpsc::channel::<WsData>(1000);
    let (_reconnect_tx, reconnect_rx) = mpsc::channel(1);
    let token = tokio_util::sync::CancellationToken::new();
    let ws_client = BybitWsClient::new(config.exchange.clone(), symbol.clone());

    // 1. Запуск WebSocket клиента в отдельной задаче
    let symbol_clone = symbol.clone();
    let token_clone = token.clone();
    tokio::spawn(async move {
        if let Err(e) = ws_client.run(tx, reconnect_rx, token_clone).await {
            error!("[{}] WebSocket runner exited with error: {}", symbol_clone, e);
        }
    });

    // 2. Настройка таймера для фиксации снапшотов с равным шагом
    let mut timer = interval(Duration::from_millis(interval_ms));
    info!("Starting snapshot pipeline for {}. Interval: {}ms", symbol, interval_ms);

    loop {
        tokio::select! {
            // Получаем обновления стакана
            Some(ws_data) = rx.recv() => {
                match ws_data {
                    WsData::OrderBook(update) => {
                        ob.apply_update(&update);
                    }
                    WsData::MarkPrice(sym, mp) => {
                        if sym == ob.symbol {
                            ob.set_mark_price(mp);
                        }
                    }
                    _ => {} // Игнорируем остальные типы данных (Trades, Ticker)
                }
            }
            
            // Тик таймера: фиксируем текущее состояние стакана
            _ = timer.tick() => {
                // Записываем снапшот только если мы получили хотя бы одно обновление (last_update_id > 0)
                if ob.last_update_id != 0 {
                    let flat_lob = ob.get_flat_snapshot();
                    
                    // Сохраняем exchange timestamp и last_update_id для точности
                    if let Err(e) = dumper.push_snapshot(ob.timestamp_ms, ob.last_update_id, flat_lob) {
                        error!("[{}] Dumper error: {}", symbol, e);
                    }
                }
            }
            
            // Обработка сигнала завершения (Graceful Shutdown)
            _ = signal::ctrl_c() => {
                info!("[{}] Shutdown signal received. Flushing data and exiting...", symbol);
                // Отменяем token для остановки WebSocket клиента
                token.cancel();
                // Принудительно сбрасываем буфер дампера на диск
                dumper.flush()?;
                break;
            }
        }
    }

    Ok(())
}
