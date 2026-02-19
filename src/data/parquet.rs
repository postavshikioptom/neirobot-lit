use polars::prelude::*;
use std::fs::File;

pub struct FastParquetScanner {
    file_path: String,
    batch_size: usize,
}

impl FastParquetScanner {
    pub fn new(path: &str, batch_size: usize) -> Self {
        Self { file_path: path.to_string(), batch_size }
    }

    pub fn get_batches(&self) -> Result<impl Iterator<Item = DataFrame>, PolarsError> {
        let file = File::open(&self.file_path).map_err(|e| PolarsError::ComputeError(e.into()))?;
        
        let mut reader = ParquetReader::new(file);
        
        // 1. Validation (Task 012 & Audit)
        let schema = reader.schema()?;
        
        // Check timestamp_ms
        match schema.get("timestamp_ms") {
            Some(dtype) if matches!(dtype, DataType::Int64) => {},
            Some(_) => return Err(PolarsError::ComputeError("Column 'timestamp_ms' must be Int64".into())),
            None => return Err(PolarsError::ComputeError("Column 'timestamp_ms' missing in schema".into())),
        }
        
        // 2. Build list of all 202 columns (2 meta + 50 levels * 4 fields = 202)
        let mut columns = Vec::with_capacity(202);
        columns.push("timestamp_ms".to_string());
        columns.push("last_update_id".to_string());
        
        for i in 0..50 {
            for prefix in &["ask_p_", "ask_v_", "bid_p_", "bid_v_"] {
                let col_name = format!("{}{}", prefix, i);
                
                // Validate existence and type (Float32 in Parquet, may be read as Float64)
                match schema.get(&col_name) {
                    Some(dtype) if matches!(dtype, DataType::Float32 | DataType::Float64) => {},
                    Some(_) => return Err(PolarsError::ComputeError(format!("Column '{}' must be Float32/64", col_name).into())),
                    None => return Err(PolarsError::ComputeError(format!("Column '{}' missing in schema", col_name).into())),
                }
                columns.push(col_name);
            }
        }

        // 3. Batched reader for low-memory processing
        let batched_reader = reader
            .with_columns(Some(columns))
            .batched(self.batch_size)?;

        Ok(batched_reader.into_iter().filter_map(|b| b.ok()))
    }
}
