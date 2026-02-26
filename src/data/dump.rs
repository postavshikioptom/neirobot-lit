use anyhow::{Context, Result};
use polars::prelude::*;
use std::fs::{File, create_dir_all};
use std::path::{Path, PathBuf};
use tokio::sync::mpsc;
use tracing::{info, error};
use crate::data::orderbook::{OrderBookSnapshot, LOB_DEPTH};

/// Асинхронный воркер для периодического дампа снимков стакана (задача 132)
/// Принимает снимки через канал и сохраняет их в JSON (для восстановления) и Parquet (для истории)
pub async fn start_snapshot_writer(
    mut rx: mpsc::Receiver<OrderBookSnapshot>,
    bot_path: PathBuf,
) {
    let mut buffer: Vec<OrderBookSnapshot> = Vec::with_capacity(100);
    let data_dir = bot_path.join("data");
    
    // Создаем директорию для данных
    if let Err(e) = create_dir_all(&data_dir) {
        error!("Failed to create data directory: {}", e);
        return;
    }

    info!("Snapshot writer started for bot at: {}", bot_path.display());

    while let Some(snap) = rx.recv().await {
        // 1. Быстрое сохранение JSON для восстановления (всегда актуальный стейт)
        let json_path = data_dir.join("last_snapshot.json");
        if let Ok(f) = File::create(&json_path) {
            if let Err(e) = serde_json::to_writer(f, &snap) {
                error!("Failed to write JSON snapshot: {}", e);
            }
        } else {
            error!("Failed to create JSON snapshot file: {}", json_path.display());
        }

        // 2. Накопление батча для Parquet
        buffer.push(snap);
        if buffer.len() >= 100 {
            if let Err(e) = flush_to_parquet(&mut buffer, &data_dir).await {
                error!("Failed to flush Parquet batch: {}", e);
            }
        }
    }

    // Финальный flush при закрытии канала
    if !buffer.is_empty() {
        if let Err(e) = flush_to_parquet(&mut buffer, &data_dir).await {
            error!("Failed to flush final Parquet batch: {}", e);
        }
    }

    info!("Snapshot writer stopped");
}

/// Записывает накопленный буфер снимков в Parquet файл
/// Соответствует глобальной схеме данных (задача 012): 50 уровней, timestamp_ms, last_update_id
async fn flush_to_parquet(buffer: &mut Vec<OrderBookSnapshot>, data_dir: &Path) -> Result<()> {
    if buffer.is_empty() {
        return Ok(());
    }

    let raw_dir = data_dir.join("raw");
    create_dir_all(&raw_dir).context("Failed to create raw data directory")?;

    // Подготовка данных для DataFrame
    let mut timestamps: Vec<i64> = Vec::with_capacity(buffer.len());
    let mut update_ids: Vec<u64> = Vec::with_capacity(buffer.len());
    let mut symbols: Vec<String> = Vec::with_capacity(buffer.len());
    
    // Для каждого уровня bid/ask создаем отдельные векторы (50 уровней согласно схеме 012)
    let mut bid_prices: Vec<Vec<f64>> = vec![Vec::with_capacity(buffer.len()); LOB_DEPTH];
    let mut bid_volumes: Vec<Vec<f64>> = vec![Vec::with_capacity(buffer.len()); LOB_DEPTH];
    let mut ask_prices: Vec<Vec<f64>> = vec![Vec::with_capacity(buffer.len()); LOB_DEPTH];
    let mut ask_volumes: Vec<Vec<f64>> = vec![Vec::with_capacity(buffer.len()); LOB_DEPTH];

    for snap in buffer.iter() {
        timestamps.push(snap.timestamp_ms);
        update_ids.push(snap.last_update_id);
        symbols.push(snap.symbol.clone());

        // Заполняем bid уровни (до 50)
        for i in 0..LOB_DEPTH {
            if let Some((p, v)) = snap.bids.get(i) {
                bid_prices[i].push(*p);
                bid_volumes[i].push(*v);
            } else {
                bid_prices[i].push(0.0);
                bid_volumes[i].push(0.0);
            }
        }

        // Заполняем ask уровни (до 50)
        for i in 0..LOB_DEPTH {
            if let Some((p, v)) = snap.asks.get(i) {
                ask_prices[i].push(*p);
                ask_volumes[i].push(*v);
            } else {
                ask_prices[i].push(0.0);
                ask_volumes[i].push(0.0);
            }
        }
    }

    // Создаем DataFrame согласно схеме 012: timestamp_ms, last_update_id, затем ask/bid уровни
    let mut columns = Vec::with_capacity(203); // 3 meta + 50*4 = 203
    columns.push(Column::new("timestamp_ms".into(), &timestamps));
    columns.push(Column::new("last_update_id".into(), &update_ids));
    columns.push(Column::new("symbol".into(), &symbols));

    // Добавляем колонки для ask уровней (interleaved price/volume)
    for i in 0..LOB_DEPTH {
        columns.push(Column::new(format!("ask_p_{}", i).into(), &ask_prices[i]));
        columns.push(Column::new(format!("ask_v_{}", i).into(), &ask_volumes[i]));
    }

    // Добавляем колонки для bid уровней (interleaved price/volume)
    for i in 0..LOB_DEPTH {
        columns.push(Column::new(format!("bid_p_{}", i).into(), &bid_prices[i]));
        columns.push(Column::new(format!("bid_v_{}", i).into(), &bid_volumes[i]));
    }

    let mut df = DataFrame::new(buffer.len(), columns).context("Failed to create DataFrame")?;

    // Имя файла: SYMBOL_FIRSTTIMESTAMP.parquet
    let first_ts = timestamps[0];
    let symbol = &buffer[0].symbol;
    let file_path = raw_dir.join(format!("{}_{}.parquet", symbol, first_ts));

    let file = File::create(&file_path).context("Failed to create Parquet file")?;
    ParquetWriter::new(file)
        .with_compression(ParquetCompression::Zstd(None))
        .finish(&mut df)
        .context("Failed to write Parquet data")?;

    info!("Flushed {} snapshots to {}", buffer.len(), file_path.display());
    buffer.clear();

    Ok(())
}

// Старый код ParquetDumper оставляем для совместимости с другими задачами

pub struct ParquetDumper {
    symbol: String,
    output_dir: PathBuf,
    buffer_limit: usize,
    timestamps: Vec<u64>,
    update_ids: Vec<u64>,
    // 200 векторов (по одному на каждую колонку стакана p/v: 50 asks * 2 + 50 bids * 2)
    lob_columns: Vec<Vec<f32>>,
}

impl ParquetDumper {
    pub fn new(symbol: &str, output_dir: &Path, buffer_limit: usize) -> Result<Self> {
        create_dir_all(output_dir).context("Failed to create output directory for Parquet")?;
        
        // Предварительная аллокация 200 векторов для колонок LOB (ask_p, ask_v, bid_p, bid_v для каждого уровня)
        let lob_columns = vec![Vec::with_capacity(buffer_limit); LOB_DEPTH * 4];

        Ok(Self {
            symbol: symbol.to_string(),
            output_dir: output_dir.to_path_buf(),
            buffer_limit,
            timestamps: Vec::with_capacity(buffer_limit),
            update_ids: Vec::with_capacity(buffer_limit),
            lob_columns,
        })
    }

    /// Добавляет снапшот в буфер. Если буфер полон — сбрасывает на диск.
    pub fn push_snapshot(&mut self, ts: u64, id: u64, flat_lob: Vec<f32>) -> Result<()> {
        if flat_lob.len() != LOB_DEPTH * 4 {
            anyhow::bail!("Invalid flat_lob length: expected {}, got {}", LOB_DEPTH * 4, flat_lob.len());
        }

        self.timestamps.push(ts);
        self.update_ids.push(id);
        
        for (i, &val) in flat_lob.iter().enumerate() {
            self.lob_columns[i].push(val);
        }

        if self.timestamps.len() >= self.buffer_limit {
            self.flush()?;
        }
        Ok(())
    }

    /// Записывает накопленные данные в Parquet файл и очищает буфер.
    pub fn flush(&mut self) -> Result<()> {
        if self.timestamps.is_empty() { return Ok(()); }

        let mut columns = Vec::with_capacity(202);
        columns.push(Column::new("timestamp_ms".into(), &self.timestamps));
        columns.push(Column::new("last_update_id".into(), &self.update_ids));

        // Генерация имен колонок согласно схеме (ask_p_0, ask_v_0... ask_p_49, ask_v_49, bid_p_0, bid_v_0... bid_p_49, bid_v_49)
        let mut col_idx = 0;
        
        // Asks (interleaved price/volume)
        for i in 0..LOB_DEPTH {
            columns.push(Column::new(format!("ask_p_{i}").into(), &self.lob_columns[col_idx]));
            columns.push(Column::new(format!("ask_v_{i}").into(), &self.lob_columns[col_idx + 1]));
            col_idx += 2;
        }
        
        // Bids (interleaved price/volume)
        for i in 0..LOB_DEPTH {
            columns.push(Column::new(format!("bid_p_{i}").into(), &self.lob_columns[col_idx]));
            columns.push(Column::new(format!("bid_v_{i}").into(), &self.lob_columns[col_idx + 1]));
            col_idx += 2;
        }

        let mut df = DataFrame::new(self.timestamps.len(), columns).context("Failed to create DataFrame")?;
        
        // Имя файла: SYMBOL_FIRSTTIMESTAMP.parquet
        let file_path = self.output_dir.join(format!("{}_{}.parquet", self.symbol, self.timestamps[0]));
        
        let file = File::create(&file_path).context("Failed to create Parquet file")?;
        ParquetWriter::new(file)
            .with_compression(ParquetCompression::Zstd(None))
            .finish(&mut df)
            .context("Failed to write Parquet data")?;

        // Очистка всех буферов с сохранением аллоцированной памяти
        self.timestamps.clear();
        self.update_ids.clear();
        for col in &mut self.lob_columns { 
            col.clear(); 
        }

        Ok(())
    }
}

impl Drop for ParquetDumper {
    fn drop(&mut self) {
        // Пытаемся сбросить остатки данных при выходе. 
        // Ошибки здесь игнорируются, так как в drop нельзя возвращать Result.
        let _ = self.flush();
    }
}

/// Асинхронный воркер для периодического дампа публичных сделок (задача 236)
/// Принимает сделки через канал и сохраняет их в Parquet
pub async fn start_trades_writer(
    mut rx: mpsc::Receiver<crate::data::types::PublicTradeOwned>,
    bot_path: PathBuf,
) {
    let mut buffer: Vec<crate::data::types::PublicTradeOwned> = Vec::with_capacity(1000);
    let data_dir = bot_path.join("data");
    
    // Создаем директорию для данных
    if let Err(e) = create_dir_all(&data_dir) {
        error!("Failed to create data directory for trades: {}", e);
        return;
    }

    info!("Trades writer started for bot at: {}", bot_path.display());

    while let Some(trade) = rx.recv().await {
        buffer.push(trade);
        
        // Сбрасываем батч когда накопится 1000 сделок
        if buffer.len() >= 1000 {
            if let Err(e) = flush_trades_to_parquet(&mut buffer, &data_dir).await {
                error!("Failed to flush trades Parquet batch: {}", e);
            }
        }
    }

    // Финальный flush при закрытии канала
    if !buffer.is_empty() {
        if let Err(e) = flush_trades_to_parquet(&mut buffer, &data_dir).await {
            error!("Failed to flush final trades Parquet batch: {}", e);
        }
    }

    info!("Trades writer stopped");
}

/// Записывает накопленный буфер сделок в Parquet файл
async fn flush_trades_to_parquet(buffer: &mut Vec<crate::data::types::PublicTradeOwned>, data_dir: &Path) -> Result<()> {
    if buffer.is_empty() {
        return Ok(());
    }

    let raw_dir = data_dir.join("raw");
    create_dir_all(&raw_dir).context("Failed to create raw data directory for trades")?;

    // Подготовка данных для DataFrame
    let mut timestamps: Vec<i64> = Vec::with_capacity(buffer.len());
    let mut prices: Vec<f64> = Vec::with_capacity(buffer.len());
    let mut sizes: Vec<f64> = Vec::with_capacity(buffer.len());
    let mut sides: Vec<String> = Vec::with_capacity(buffer.len());

    for trade in buffer.iter() {
        timestamps.push(trade.timestamp);
        prices.push(trade.price.to_string().parse::<f64>().unwrap_or(0.0));
        sizes.push(trade.size.to_string().parse::<f64>().unwrap_or(0.0));
        sides.push(match trade.side {
            crate::data::types::Side::Buy => "Buy".to_string(),
            crate::data::types::Side::Sell => "Sell".to_string(),
        });
    }

    // Создаем DataFrame: timestamp, price, size, side
    let columns = vec![
        Column::new("timestamp".into(), &timestamps),
        Column::new("price".into(), &prices),
        Column::new("size".into(), &sizes),
        Column::new("side".into(), &sides),
    ];

    let mut df = DataFrame::new(buffer.len(), columns).context("Failed to create trades DataFrame")?;

    // Имя файла: trades_FIRSTTIMESTAMP.parquet
    let first_ts = timestamps[0];
    let file_path = raw_dir.join(format!("trades_{}.parquet", first_ts));

    let file = File::create(&file_path).context("Failed to create trades Parquet file")?;
    ParquetWriter::new(file)
        .with_compression(ParquetCompression::Zstd(None))
        .finish(&mut df)
        .context("Failed to write trades Parquet data")?;

    info!("Flushed {} trades to {}", buffer.len(), file_path.display());
    buffer.clear();

    Ok(())
}
