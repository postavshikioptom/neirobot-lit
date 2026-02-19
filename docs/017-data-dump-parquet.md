# 017 - Data Dump Parquet

Цель задачи: Реализовать модуль src/data/dump.rs для сохранения снимков стакана в формате Parquet с использованием библиотеки Polars. Запись должна быть чанковой (накопление в буфере) для минимизации I/O и строго следовать схеме из docs/data_schema.json (202 колонки).

Файлы для изменения/создания:

Cargo.toml (добавить зависимости)
src/data/dump.rs (создать)
src/data/mod.rs (обновить)
Инструкции для Gemini:

Добавить зависимости в Cargo.toml:

polars = { version = "0.42", features = ["parquet", "lazy", "dtype-u64"] }
src/data/dump.rs: Реализовать ParquetDumper с эффективным хранением колонок и автоматическим сбросом при закрытии.

use anyhow::{Context, Result};
use polars::prelude::*;
use std::fs::{File, create_dir_all};
use std::path::{Path, PathBuf};
use crate::data::orderbook::LOB_DEPTH;

pub struct ParquetDumper {
    symbol: String,
    output_dir: PathBuf,
    buffer_limit: usize,
    timestamps: Vec<u64>,
    update_ids: Vec<u64>,
    // 200 векторов (по одному на каждую колонку стакана p/v)
    lob_columns: Vec<Vec<f32>>,
}

impl ParquetDumper {
    pub fn new(symbol: &str, output_dir: &Path, buffer_limit: usize) -> Result<Self> {
        create_dir_all(output_dir)?;
        
        // Предварительная аллокация 200 векторов для колонок LOB
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

    pub fn flush(&mut self) -> Result<()> {
        if self.timestamps.is_empty() { return Ok(()); }

        let mut columns = Vec::with_capacity(202);
        columns.push(Series::new("timestamp_ms", &self.timestamps));
        columns.push(Series::new("last_update_id", &self.update_ids));

        // Генерация имен колонок согласно схеме (ask_p_0, ask_v_0...)
        let mut col_idx = 0;
        for i in 0..LOB_DEPTH {
            columns.push(Series::new(&format!("ask_p_{i}"), &self.lob_columns[col_idx]));
            columns.push(Series::new(&format!("ask_v_{i}"), &self.lob_columns[col_idx + 1]));
            col_idx += 2;
        }
        for i in 0..LOB_DEPTH {
            columns.push(Series::new(&format!("bid_p_{i}"), &self.lob_columns[col_idx]));
            columns.push(Series::new(&format!("bid_v_{i}"), &self.lob_columns[col_idx + 1]));
            col_idx += 2;
        }

        let mut df = DataFrame::new(columns)?;
        let file_path = self.output_dir.join(format!("{}_{}.parquet", self.symbol, self.timestamps[0]));
        
        let file = File::create(&file_path)?;
        ParquetWriter::new(file)
            .with_compression(ParquetCompression::Zstd(None))
            .finish(&mut df)?;

        // Очистка всех буферов
        self.timestamps.clear();
        self.update_ids.clear();
        for col in &mut self.lob_columns { col.clear(); }

        Ok(())
    }
}

impl Drop for ParquetDumper {
    fn drop(&mut self) {
        let _ = self.flush(); // Сбрасываем остатки при завершении
    }
}
Технические требования:

Сжатие: Использовать Zstd (оптимально для повторяющихся цен в LOB).
Структура: Хранить данные в транспонированном виде (по колонкам), чтобы Polars мог мгновенно создать Series без перепаковки данных в памяти при каждом flush.
Безопасность: Реализовать Drop для записи остатков данных при выключении бота.
Производительность: Использовать with_capacity для всех векторов, чтобы избежать реаллокаций при заполнении чанка.
Почему это важно: Формат Parquet с колончатым сжатием Zstd позволяет сократить объем данных LOB в 5-10 раз по сравнению с JSON/CSV. Использование Polars напрямую в Rust обеспечивает максимальную скорость записи, которая необходима при одновременной выгрузке 50+ инструментов.