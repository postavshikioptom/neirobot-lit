# 021 - Python Lab Dataset Loading

Цель задачи: Реализовать модуль python_lab/src/dataset.py для загрузки и первичной валидации датасета. Мы должны обеспечить максимально быструю загрузку Parquet-файлов через Polars, поддержку работы с данными, превышающими объем RAM (через LazyFrame), и строгую проверку соответствия колонок нашей схеме из docs/data_schema.json.

Файлы для изменения/создания:

python_lab/src/dataset.py (создать)
Инструкции для Gemini:

python_lab/src/dataset.py: Создать класс LOBDataset с поддержкой ленивых вычислений и валидацией схемы.

import polars as pl
import json
from pathlib import Path
from typing import Union

class LOBDataset:
    def __init__(self, data_path: str, symbol: str, schema_path: str = "../../docs/data_schema.json"):
        self.data_path = Path(data_path)
        self.symbol = symbol
        self.schema_path = Path(schema_path)
        self.expected_columns = self._load_schema()

    def _load_schema(self):
        with open(self.schema_path, 'r') as f:
            schema = json.load(f)
        return schema["columns"]

    def load_data(self, lazy: bool = False) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        Загружает данные из Parquet.
        lazy=True: возвращает LazyFrame для работы с данными на диске.
        lazy=False: загружает всё в RAM (collect).
        """
        # Фильтруем файлы по символу и расширению
        pattern = f"{self.symbol}_*.parquet"
        
        # Используем scan_parquet для ленивой загрузки и параллелизма
        lf = pl.scan_parquet(self.data_path / pattern)
        
        # 1. Валидация схемы (сравнение названий колонок)
        actual_columns = lf.collect_schema().names()
        if actual_columns != self.expected_columns:
            missing = set(self.expected_columns) - set(actual_columns)
            extra = set(actual_columns) - set(self.expected_columns)
            raise ValueError(f"Schema mismatch for {self.symbol}! Missing: {missing}, Extra: {extra}")

        # 2. Сортировка по времени
        lf = lf.sort("timestamp_ms")

        if lazy:
            return lf
        
        df = lf.collect()
        print(f"[{self.symbol}] Loaded {len(df)} rows. Memory: {df.estimated_size('mb'):.2f} MB")
        return df

if __name__ == "__main__":
    # Тестовый запуск
    try:
        loader = LOBDataset("../../bots/BTCUSDT/data/raw", "BTCUSDT")
        df = loader.load_data(lazy=False)
        print(df.select(["timestamp_ms", "ask_p_0", "bid_p_0"]).head())
    except Exception as e:
        print(f"Error loading dataset: {e}")
Технические требования:

Lazy Loading: Использовать pl.scan_parquet для эффективного сканирования файлов без полной загрузки в память.
Валидация: Загружать docs/data_schema.json и сравнивать список колонок. Если Rust-дампер запишет колонки не в том порядке или с другими именами — Python должен выдать ошибку немедленно.
Типизация: Добавить type hints для аргументов и возвращаемых значений.