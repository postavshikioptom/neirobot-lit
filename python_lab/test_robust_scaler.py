"""
Тест для проверки функций Robust Scaler и Winsorization (Задача 240)
"""
import polars as pl
import numpy as np
from src.dataset import apply_winsorization, fit_robust_params, apply_robust_scaling


def test_winsorization():
    """Тест функции винзоризации"""
    print("=== Тест винзоризации ===")
    
    # Создаем DataFrame с выбросами
    df = pl.DataFrame({
        "a": [1, 2, 3, 4, 5, 100],  # 100 - выброс
        "b": [-50, 2, 3, 4, 5, 6],  # -50 - выброс
    })
    
    print("Исходные данные:")
    print(df)
    
    # Применяем винзоризацию (1% и 99% перцентили)
    df_clipped = apply_winsorization(df, limits=(0.01, 0.99))
    
    print("\nПосле винзоризации:")
    print(df_clipped)
    
    # Проверяем, что выбросы ограничены
    assert df_clipped["a"].max() < 100, "Выброс в колонке 'a' не ограничен"
    assert df_clipped["b"].min() > -50, "Выброс в колонке 'b' не ограничен"
    
    print("✓ Тест винзоризации пройден\n")


def test_robust_scaling():
    """Тест функций Robust Scaling"""
    print("=== Тест Robust Scaling ===")
    
    # Создаем тренировочные данные
    df_train = pl.DataFrame({
        "feat1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "feat2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    })
    
    print("Тренировочные данные:")
    print(df_train)
    
    # Вычисляем параметры
    params = fit_robust_params(df_train)
    
    print("\nПараметры Robust Scaling:")
    print(f"Type: {params['type']}")
    print(f"Median: {params['median']}")
    print(f"IQR: {params['iqr']}")
    
    # Проверяем корректность параметров
    assert params['type'] == 'robust', "Неверный тип параметров"
    assert 'median' in params, "Отсутствует median"
    assert 'iqr' in params, "Отсутствует iqr"
    
    # Медиана для feat1 должна быть 5.5, для feat2 - 55
    assert abs(params['median']['feat1'] - 5.5) < 0.1, "Неверная медиана для feat1"
    assert abs(params['median']['feat2'] - 55.0) < 0.1, "Неверная медиана для feat2"
    
    # IQR для feat1 = Q3(7.75) - Q1(3.25) = 4.5
    # IQR для feat2 = Q3(77.5) - Q1(32.5) = 45
    assert abs(params['iqr']['feat1'] - 4.5) < 0.1, "Неверный IQR для feat1"
    assert abs(params['iqr']['feat2'] - 45.0) < 0.1, "Неверный IQR для feat2"
    
    # Применяем масштабирование
    df_scaled = apply_robust_scaling(df_train, params)
    
    print("\nМасштабированные данные:")
    print(df_scaled)
    
    # Проверяем, что масштабирование применено
    # Медиана должна быть близка к 0
    median_scaled = df_scaled.median().to_dicts()[0]
    print(f"\nМедиана после масштабирования: {median_scaled}")
    
    assert abs(median_scaled['feat1']) < 0.2, "Медиана feat1 не близка к 0"
    assert abs(median_scaled['feat2']) < 0.2, "Медиана feat2 не близка к 0"
    
    print("✓ Тест Robust Scaling пройден\n")


def test_combined_workflow():
    """Тест комбинированного workflow: винзоризация + robust scaling"""
    print("=== Тест комбинированного workflow ===")
    
    # Создаем данные с выбросами
    np.random.seed(42)
    data = np.random.randn(100)
    data = np.append(data, [100, -100])  # Добавляем выбросы
    
    df = pl.DataFrame({"feature": data})
    
    print(f"Исходные данные: min={df['feature'].min():.2f}, max={df['feature'].max():.2f}")
    
    # 1. Винзоризация
    df_winsor = apply_winsorization(df, limits=(0.05, 0.95))
    print(f"После винзоризации: min={df_winsor['feature'].min():.2f}, max={df_winsor['feature'].max():.2f}")
    
    # 2. Robust Scaling
    params = fit_robust_params(df_winsor)
    df_scaled = apply_robust_scaling(df_winsor, params)
    
    print(f"После robust scaling: median={df_scaled['feature'].median():.4f}")
    
    # Проверяем, что выбросы ограничены и данные масштабированы
    assert df_winsor['feature'].max() < 50, "Выбросы не ограничены"
    assert df_winsor['feature'].min() > -50, "Выбросы не ограничены"
    assert abs(df_scaled['feature'].median()) < 0.2, "Медиана не близка к 0"
    
    print("✓ Тест комбинированного workflow пройден\n")


if __name__ == "__main__":
    test_winsorization()
    test_robust_scaling()
    test_combined_workflow()
    
    print("=" * 50)
    print("Все тесты пройдены успешно! ✓")
    print("=" * 50)
