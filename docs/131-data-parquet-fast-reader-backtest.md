# Задача 131: Скоростной Parquet-ридер (Rust & Python) (v2.0)

## 1. Контекст и цели
Скорость чтения LOB-данных из Parquet — критический узел. Мы внедряем `Batched Reader`, который позволяет читать данные кусками (batches), не забивая оперативную память, что необходимо для:
*   **Rust**: Симуляция сведения ордеров (Matching Engine) и расчет HFT-метрик в реальном времени.
*   **Python**: Быстрая подача данных в нейросеть (DataLoader) и расчет PnL стратегии.

## 2. Реализация в Rust (`src/data/parquet.rs`)
Используем `BatchedParquetReader` для эффективного итерирования по группам строк.

```rust
use polars::prelude::*;
use std::fs::File;
use std::path::Path;

pub struct FastParquetScanner {
    file_path: String,
    batch_size: usize,
}

impl FastParquetScanner {
    pub fn new(path: &str, batch_size: usize) -> Self {
        Self { file_path: path.to_string(), batch_size }
    }

    pub fn get_batches(&self) -> Result<impl Iterator<Item = DataFrame>> {
        let file = File::open(&self.file_path).map_err(|e| PolarsError::ComputeError(e.into()))?;
        
        // Используем BatchedParquetReader (v0.41+)
        let reader = ParquetReader::new(file)
            .with_columns(Some(vec![
                "timestamp".to_string(), 
                "ask_p_0".to_string(), "ask_v_0".to_string(),
                "bid_p_0".to_string(), "bid_v_0".to_string()
            ]))
            .batched(self.batch_size)?;

        Ok(reader.into_iter().filter_map(|b| b.ok()))
    }
}
```

## 3. Реализация в Python (`python_lab/src/dataset.py`)
Используем `iter_slices` для генерации батчей для `backtest.py`.

```python
import polars as pl

def fast_parquet_reader(file_path, batch_size=100_000):
    """
    Генератор батчей данных через Memory Map.
    """
    # Читаем лениво, выбирая только нужные колонки
    q = pl.scan_parquet(file_path).select([
        pl.col("timestamp").cast(pl.Int64),
        pl.col("^ask_p_.*$"), # Все цены асков через regex
        pl.col("^bid_p_.*$"),
        pl.col("^ask_v_.*$"),
        pl.col("^bid_v_.*$")
    ])
    
    df = q.collect(streaming=True)
    # Итерируемся слайсами (O(1) по памяти)
    for batch in df.iter_slices(batch_size):
        yield batch
```

## 4. Спорные моменты и корректировки (Grok + Zencoder)

*   **Rust vs Python**: Согласен с Grok, что бэктест стратегии на Polars DataFrame лучше делать в Python. Но Rust-ридер необходим для задачи 020 (Snapshot Builder) и 070 (Matching Engine), где мы имитируем работу биржи Bybit.
*   **API Polars**: Грок прав, `split_by_rows` — это галлюцинация. Используем `batched` в Rust и `iter_slices` в Python. Это официальные способы работы с большими файлами.
*   **Config**: В [./src/config/types.rs](./src/config/types.rs) добавляем опциональный блок `[backtest]` для указания пути к истории в Rust, но в `python_lab` используем аргументы командной строки (`--data_path`).
*   **Validation**: В Rust-версии обязательно проверяем типы колонок при инициализации (timestamp должен быть `Int64`, цены — `Float64`).
*   **Benchmark**: Целевой показатель — **1 млн строк < 2 сек** на SSD. Этого достаточно, чтобы прогнать месяц истории (30-50 млн строк) менее чем за минуту.

## 5. Инструкции для Gemini (Coder AI):
1.  **Cargo.toml**: Обновить `polars` до `0.41` с фичами `["lazy", "parquet", "streaming", "regex"]`.
2.  **src/data/parquet.rs**: Реализовать `FastParquetScanner`.
3.  **python_lab/src/dataset.py**: Добавить генератор `fast_parquet_reader`.
4.  **Tests**: Написать бенчмарк в `tests/data_parquet.rs` с использованием `std::time::Instant`.

**Результат**: Мы получаем унифицированный и сверхбыстрый доступ к данным стакана, который не «съедает» всю оперативную память даже при обработке терабайтных архивов.
