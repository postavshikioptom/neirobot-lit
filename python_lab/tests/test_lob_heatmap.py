#!/usr/bin/env python3
"""
Тесты для lob_heatmap.py (Задача 223)
Проверяют корректность работы с реальной схемой данных из dump.rs и trade_logger.rs
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import polars as pl
import pytest

# Добавляем путь к scripts
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from lob_heatmap import (
    compute_signed_bps_vectorized,
    create_heatmap_matrix,
    export_matrix_to_csv
)


def test_imports():
    """Проверка что все необходимые библиотеки импортируются."""
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    
    assert go is not None
    assert plt is not None
    assert LogNorm is not None


def create_test_orderbook_data():
    """Создает тестовые данные в формате dump.rs (schema 012)."""
    now = datetime.now()
    timestamp_ms = int(now.timestamp() * 1000)
    
    # Создаем данные для 2 снимков стакана
    data = {
        "timestamp_ms": [timestamp_ms, timestamp_ms + 1000],
        "last_update_id": [1, 2],
        "symbol": ["BTCUSDT", "BTCUSDT"],
    }
    
    # Добавляем ask уровни (50 уровней)
    for i in range(50):
        # Первый снимок: цены от 100.01 до 102.50
        ask_price_1 = 100.01 + i * 0.05
        ask_vol_1 = 10.0 - i * 0.1 if i < 10 else 0.0
        
        # Второй снимок: цены от 100.02 до 102.51
        ask_price_2 = 100.02 + i * 0.05
        ask_vol_2 = 9.0 - i * 0.1 if i < 10 else 0.0
        
        data[f"ask_p_{i}"] = [ask_price_1, ask_price_2]
        data[f"ask_v_{i}"] = [ask_vol_1, ask_vol_2]
    
    # Добавляем bid уровни (50 уровней)
    for i in range(50):
        # Первый снимок: цены от 99.99 до 97.50
        bid_price_1 = 99.99 - i * 0.05
        bid_vol_1 = 10.0 - i * 0.1 if i < 10 else 0.0
        
        # Второй снимок: цены от 99.98 до 97.49
        bid_price_2 = 99.98 - i * 0.05
        bid_vol_2 = 9.0 - i * 0.1 if i < 10 else 0.0
        
        data[f"bid_p_{i}"] = [bid_price_1, bid_price_2]
        data[f"bid_v_{i}"] = [bid_vol_1, bid_vol_2]
    
    return pl.DataFrame(data)


def test_compute_signed_bps_vectorized():
    """Тест преобразования в Signed BPS координаты с реальной схемой."""
    df = create_test_orderbook_data()
    
    # Вычисляем mid_price
    df = df.with_columns([
        ((pl.col("ask_p_0") + pl.col("bid_p_0")) / 2).alias("mid_price"),
        (pl.col("timestamp_ms") / 1000.0).alias("timestamp")
    ])
    
    timestamps, bps_values, volumes = compute_signed_bps_vectorized(df, depth_bps=200.0)
    
    # Проверки
    assert len(timestamps) > 0, "Должны быть данные после преобразования"
    assert len(timestamps) == len(bps_values) == len(volumes), "Длины массивов должны совпадать"
    
    # Проверка что Asks положительные
    ask_mask = bps_values > 0
    assert np.any(ask_mask), "Должны быть положительные BPS (Asks)"
    
    # Проверка что Bids отрицательные
    bid_mask = bps_values < 0
    assert np.any(bid_mask), "Должны быть отрицательные BPS (Bids)"
    
    # Проверка что все объемы положительные
    assert np.all(volumes > 0), "Все объемы должны быть положительными"


def test_create_heatmap_matrix():
    """Тест создания 2D гистограммы."""
    start_time = datetime(2024, 2, 16, 10, 0, 0)
    end_time = datetime(2024, 2, 16, 10, 0, 10)
    
    # Генерируем случайные данные
    np.random.seed(42)
    n_points = 1000
    
    timestamps = np.random.uniform(
        start_time.timestamp(),
        end_time.timestamp(),
        n_points
    )
    bps_values = np.random.uniform(-50, 50, n_points)
    volumes = np.random.uniform(1, 100, n_points)
    
    matrix, time_edges, bps_edges = create_heatmap_matrix(
        timestamps,
        bps_values,
        volumes,
        start_time,
        end_time,
        time_bin_seconds=1.0,
        bps_bin_size=5.0,
        depth_bps=100.0
    )
    
    # Проверки
    assert matrix.shape[0] > 0, "Матрица должна иметь строки"
    assert matrix.shape[1] > 0, "Матрица должна иметь столбцы"
    assert len(time_edges) == matrix.shape[1] + 1, "Количество time_edges должно быть на 1 больше"
    assert len(bps_edges) == matrix.shape[0] + 1, "Количество bps_edges должно быть на 1 больше"
    
    # Проверка что матрица содержит неотрицательные значения (log1p)
    assert np.all(matrix >= 0), "Матрица должна содержать неотрицательные значения"


def test_export_matrix_to_csv(tmp_path):
    """Тест экспорта матрицы в CSV."""
    matrix = np.random.rand(10, 20)
    
    start_time = datetime(2024, 2, 16, 10, 0, 0)
    time_edges = np.array([start_time.timestamp() + i for i in range(21)])
    bps_edges = np.linspace(-50, 50, 11)
    
    output_path = tmp_path / "test_matrix.csv"
    
    export_matrix_to_csv(matrix, time_edges, bps_edges, output_path)
    
    # Проверка что файл создан
    assert output_path.exists(), "CSV файл должен быть создан"
    
    # Проверка что файл можно прочитать
    df = pl.read_csv(output_path)
    assert len(df) == matrix.shape[1], "Количество строк должно совпадать"
    assert "timestamp" in df.columns, "Должна быть колонка timestamp"


def test_signed_bps_formula():
    """Тест правильности формулы Signed BPS."""
    mid_price = 100.0
    
    # Ask выше mid_price -> положительный BPS
    ask_price = 100.1
    expected_bps = (ask_price - mid_price) / mid_price * 10000
    assert expected_bps == 100.0, "Ask BPS должен быть 100"
    
    # Bid ниже mid_price -> отрицательный BPS
    bid_price = 99.9
    expected_bps = (bid_price - mid_price) / mid_price * 10000
    assert expected_bps == -100.0, "Bid BPS должен быть -100"
    
    # На mid_price -> 0 BPS
    mid_price_order = 100.0
    expected_bps = (mid_price_order - mid_price) / mid_price * 10000
    assert expected_bps == 0.0, "Mid price BPS должен быть 0"


def test_depth_filtering():
    """Тест фильтрации по глубине depth_bps."""
    df = create_test_orderbook_data()
    
    # Вычисляем mid_price
    df = df.with_columns([
        ((pl.col("ask_p_0") + pl.col("bid_p_0")) / 2).alias("mid_price"),
        (pl.col("timestamp_ms") / 1000.0).alias("timestamp")
    ])
    
    # Фильтрация с depth_bps=100
    timestamps, bps_values, volumes = compute_signed_bps_vectorized(df, depth_bps=100.0)
    
    # Должны остаться только точки с |BPS| <= 100
    assert np.all(np.abs(bps_values) <= 100.0), "Все BPS должны быть в пределах depth_bps"


def test_mid_price_calculation():
    """Тест правильности вычисления mid_price из ask_p_0 и bid_p_0."""
    df = create_test_orderbook_data()
    
    # Вычисляем mid_price
    df = df.with_columns([
        ((pl.col("ask_p_0") + pl.col("bid_p_0")) / 2).alias("mid_price")
    ])
    
    # Проверяем что mid_price находится между bid и ask
    for row in df.iter_rows(named=True):
        assert row["bid_p_0"] <= row["mid_price"] <= row["ask_p_0"], \
            "mid_price должен быть между bid и ask"


def test_trades_csv_format():
    """Тест парсинга CSV с форматом trade_logger.rs."""
    # Создаем тестовый CSV в формате trade_logger.rs
    trades_data = {
        "time": [
            "2024-02-16T10:00:00Z",
            "2024-02-16T10:00:01Z"
        ],
        "symbol": ["BTCUSDT", "BTCUSDT"],
        "side": ["Buy", "Sell"],  # Не BUY/SELL, а Buy/Sell
        "price": [100.5, 100.3],
        "qty": [1.0, 2.0],  # Не quantity, а qty
        "order_type": ["Limit", "Market"],
        "is_maker": [True, False],
        "signal_up": [0.6, 0.4],
        "signal_down": [0.4, 0.6],
        "fee": [0.001, 0.002]
    }
    
    df = pl.DataFrame(trades_data)
    
    # Парсим время в RFC3339 формате
    df = df.with_columns([
        pl.col("time").str.to_datetime()
    ])
    
    # Проверяем что парсинг работает
    assert df["time"].dtype == pl.Datetime, "Время должно быть распарсено как Datetime"
    assert df["side"][0] == "Buy", "Side должен быть 'Buy'"
    assert df["qty"][0] == 1.0, "Должна быть колонка qty"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
