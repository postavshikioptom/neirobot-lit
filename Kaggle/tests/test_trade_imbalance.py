"""
Тест для функции compute_trade_imbalance (Задача 236)
"""
import polars as pl
import numpy as np
from pathlib import Path
import sys

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset import compute_trade_imbalance


def test_compute_trade_imbalance():
    """Тест базовой функциональности compute_trade_imbalance"""
    
    # Создаем тестовые данные snapshots
    timestamps_snap = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    df_snapshots = pl.DataFrame({
        "timestamp_ms": timestamps_snap,
        "mid_price": [100.0] * len(timestamps_snap),
    })
    
    # Создаем тестовые данные trades
    # Несколько покупок и продаж с разными размерами
    df_trades = pl.DataFrame({
        "timestamp": [1500, 2500, 3500, 4500, 5500, 6500, 7500, 8500],
        "price": [100.1, 100.2, 99.9, 100.3, 99.8, 100.4, 99.7, 100.5],
        "size": [1.0, 2.0, 1.5, 3.0, 2.5, 1.0, 2.0, 1.5],
        "side": ["Buy", "Buy", "Sell", "Buy", "Sell", "Buy", "Sell", "Buy"],
    })
    
    print("Тестовые данные snapshots:")
    print(df_snapshots)
    print("\nТестовые данные trades:")
    print(df_trades)
    
    # Вызываем функцию
    result = compute_trade_imbalance(
        df_snapshots, 
        df_trades, 
        windows=["1s", "2s"],
        agg_type='vol',
        noise_filter_pct=0.01
    )
    
    print("\nРезультат с добавленными колонками imbalance:")
    print(result)
    
    # Проверяем, что колонки добавлены
    assert "imb_vol_1s" in result.columns, "Колонка imb_vol_1s не найдена"
    assert "imb_vol_2s" in result.columns, "Колонка imb_vol_2s не найдена"
    
    # Проверяем, что значения не NaN
    assert not result["imb_vol_1s"].is_null().any(), "Есть NaN значения в imb_vol_1s"
    assert not result["imb_vol_2s"].is_null().any(), "Есть NaN значения в imb_vol_2s"
    
    print("\n✓ Тест пройден успешно!")
    print(f"✓ Добавлено {len([c for c in result.columns if c.startswith('imb_')])} колонок imbalance")
    
    # Выводим статистику по imbalance
    print("\nСтатистика imb_vol_1s:")
    print(result["imb_vol_1s"].describe())
    print("\nСтатистика imb_vol_2s:")
    print(result["imb_vol_2s"].describe())


def test_empty_trades():
    """Тест с пустым DataFrame trades"""
    
    df_snapshots = pl.DataFrame({
        "timestamp_ms": [1000, 2000, 3000],
        "mid_price": [100.0, 100.1, 100.2],
    })
    
    df_trades = pl.DataFrame(schema={
        "timestamp": pl.Int64,
        "price": pl.Float64,
        "size": pl.Float64,
        "side": pl.Utf8
    })
    
    result = compute_trade_imbalance(
        df_snapshots, 
        df_trades, 
        windows=["1s"],
        agg_type='vol'
    )
    
    # Проверяем, что колонка добавлена и заполнена нулями
    assert "imb_vol_1s" in result.columns
    assert (result["imb_vol_1s"] == 0.0).all()
    
    print("✓ Тест с пустыми trades пройден успешно!")


def test_count_aggregation():
    """Тест агрегации по количеству сделок"""
    
    df_snapshots = pl.DataFrame({
        "timestamp_ms": [1000, 2000, 3000, 4000, 5000],
        "mid_price": [100.0] * 5,
    })
    
    df_trades = pl.DataFrame({
        "timestamp": [1500, 2500, 3500, 4500],
        "price": [100.1, 100.2, 99.9, 100.3],
        "size": [1.0, 2.0, 1.5, 3.0],
        "side": ["Buy", "Buy", "Sell", "Buy"],
    })
    
    result = compute_trade_imbalance(
        df_snapshots, 
        df_trades, 
        windows=["2s"],
        agg_type='count',
        noise_filter_pct=0.01
    )
    
    assert "imb_count_2s" in result.columns
    print("✓ Тест агрегации по количеству пройден успешно!")
    print(result)


if __name__ == "__main__":
    print("=" * 60)
    print("Тестирование функции compute_trade_imbalance (Задача 236)")
    print("=" * 60)
    
    print("\n1. Тест базовой функциональности:")
    print("-" * 60)
    test_compute_trade_imbalance()
    
    print("\n2. Тест с пустыми trades:")
    print("-" * 60)
    test_empty_trades()
    
    print("\n3. Тест агрегации по количеству:")
    print("-" * 60)
    test_count_aggregation()
    
    print("\n" + "=" * 60)
    print("Все тесты пройдены успешно! ✓")
    print("=" * 60)
