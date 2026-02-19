use neirobot_lit::data::parquet::FastParquetScanner;
use polars::prelude::*;
use std::time::Instant;
use tempfile::NamedTempFile;

#[test]
fn test_parquet_reader_benchmark() -> anyhow::Result<()> {
    // 1. Создаем временный файл с данными (100k строк для теста)
    let n_rows = 100_000;
    
    // Создаем колонки для 50 уровней, чтобы пройти валидацию
    let mut df_builder = df!(
        "timestamp_ms" => (0..n_rows).collect::<Vec<i64>>()
    )?;

    for i in 0..50 {
        df_builder.with_column(Series::new(&format!("ask_p_{}", i), vec![100.0f64; n_rows as usize]))?;
        df_builder.with_column(Series::new(&format!("ask_v_{}", i), vec![1.0f64; n_rows as usize]))?;
        df_builder.with_column(Series::new(&format!("bid_p_{}", i), vec![99.9f64; n_rows as usize]))?;
        df_builder.with_column(Series::new(&format!("bid_v_{}", i), vec![2.0f64; n_rows as usize]))?;
    }
    
    let mut df = df_builder;


    let tmp_file = NamedTempFile::new()?;
    let path = tmp_file.path().to_str().unwrap();
    
    let file = std::fs::File::create(path)?;
    ParquetWriter::new(file).finish(&mut df)?;

    println!("✓ Dummy parquet created with {} rows", n_rows);

    // 2. Бенчмарк чтения
    let scanner = FastParquetScanner::new(path, 10_000);
    
    let start = Instant::now();
    let mut total_rows = 0;
    
    for batch in scanner.get_batches()? {
        total_rows += batch.height();
    }
    
    let duration = start.elapsed();
    
    println!("✓ Benchmark completed:");
    println!("  Rows read: {}", total_rows);
    println!("  Time taken: {:?}", duration);
    println!("  Speed: {:.2} million rows/sec", (total_rows as f64 / 1_000_000.0) / duration.as_secs_f64());

    assert_eq!(total_rows, n_rows as usize);
    
    Ok(())
}
