use serde::Serialize;
use rust_decimal::Decimal;
use tokio::sync::mpsc;
use std::path::PathBuf;
use std::fs::OpenOptions;
use tracing::{info, error};
use anyhow::Result;

#[derive(Debug, Serialize, Clone)]
pub struct TradeRecord {
    pub time: String,            // RFC3339 (chrono::Utc::now().to_rfc3339())
    pub symbol: String,
    pub side: String,            // "Buy" / "Sell"
    pub price: Decimal,
    pub qty: Decimal,
    pub order_type: String,      // "Limit" / "Market"
    pub is_maker: bool,
    pub signal_up: f32,          // Вероятность Up
    pub signal_down: f32,        // Вероятность Down
    pub realized_pnl: Option<Decimal>, // Только при закрытии/сокращении
    pub fee: Decimal,
}

pub struct CsvTradeLogger {
    tx: mpsc::Sender<TradeRecord>,
}

impl CsvTradeLogger {
    pub fn init(log_path: PathBuf) -> Result<(Self, tokio::task::JoinHandle<()>)> {
        let (tx, mut rx) = mpsc::channel::<TradeRecord>(1000);

        // Создаем директорию, если ее нет
        if let Some(parent) = log_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Запуск фоновой задачи для записи в файл
        let handle = tokio::spawn(async move {
            let file_exists = log_path.exists();
            
            let file = match OpenOptions::new()
                .create(true)
                .append(true)
                .open(&log_path) {
                    Ok(f) => f,
                    Err(e) => {
                        error!("Failed to open trade log file {:?}: {}", log_path, e);
                        return;
                    }
                };

            let mut wtr = csv::WriterBuilder::new()
                .has_headers(!file_exists)
                .from_writer(file);

            while let Some(record) = rx.recv().await {
                if let Err(e) = wtr.serialize(record) {
                    error!("Failed to serialize trade record: {}", e);
                    continue;
                }
                if let Err(e) = wtr.flush() {
                    error!("Failed to flush trade log: {}", e);
                }
            }
            
            info!("Trade logger background task shutting down");
        });

        Ok((Self { tx }, handle))
    }

    pub fn get_sender(&self) -> mpsc::Sender<TradeRecord> {
        self.tx.clone()
    }
}
