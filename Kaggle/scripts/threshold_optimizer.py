#!/usr/bin/env python3
"""
Задача 210: Оптимизация параметров адаптивных порогов отмены ордеров

Этот скрипт находит оптимальные значения vol_multiplier и spread_multiplier,
при которых количество отмен коррелирует с защитой от убыточных сделок,
но не блокирует нормальную торговлю.

Использует Optuna для поиска оптимальных параметров.
"""

import argparse
import polars as pl
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


def calculate_adaptive_threshold(
    base_threshold: float,
    volatility_bps: float,
    spread_bps: float,
    vol_multiplier: float,
    spread_multiplier: float,
    min_threshold: float = 10.0,
    max_threshold: float = 200.0
) -> float:
    """
    Расчет адаптивного порога отмены по формуле из задачи 210.
    
    Формула: adaptive_threshold = base_threshold / (1.0 + (volatility * vol_multiplier) + (spread * spread_multiplier))
    """
    denominator = 1.0 + (volatility_bps * vol_multiplier) + (spread_bps * spread_multiplier)
    adaptive_threshold = base_threshold / denominator if denominator > 0 else base_threshold
    
    # Применяем границы
    return np.clip(adaptive_threshold, min_threshold, max_threshold)


def simulate_trading(
    df: pl.DataFrame,
    base_threshold: float,
    vol_multiplier: float,
    spread_multiplier: float
) -> Dict[str, float]:
    """
    Симуляция торговли с заданными параметрами адаптивных порогов.
    
    Возвращает метрики:
    - cancellations: количество отмен
    - bad_fills: количество убыточных исполнений
    - total_pnl: общая прибыль/убыток
    - cancel_rate: процент отмен
    """
    # Вычисляем адаптивные пороги для каждой строки
    adaptive_thresholds = [
        calculate_adaptive_threshold(
            base_threshold,
            row['volatility_bps'],
            row['spread_bps'],
            vol_multiplier,
            spread_multiplier
        )
        for row in df.iter_rows(named=True)
    ]
    
    df = df.with_columns([
        pl.Series('adaptive_threshold', adaptive_thresholds)
    ])
    
    # Симулируем отмены: если price_deviation > adaptive_threshold, отменяем
    df = df.with_columns([
        (pl.col('price_deviation_bps') > pl.col('adaptive_threshold')).alias('should_cancel')
    ])
    
    cancellations = df['should_cancel'].sum()
    total_trades = len(df)
    
    # Симулируем убыточные исполнения: если не отменили, но deviation большое
    # Считаем "плохим" исполнением, если deviation > base_threshold * 1.5
    bad_fills = df.filter(
        (~pl.col('should_cancel')) & (pl.col('price_deviation_bps') > base_threshold * 1.5)
    ).height
    
    # Симулируем PnL: отмены стоят комиссию, плохие исполнения стоят дороже
    cancel_cost = cancellations * 0.0006  # Комиссия за отмену
    bad_fill_cost = bad_fills * 0.01  # Потери от плохого исполнения
    
    # Хорошие исполнения приносят прибыль
    good_fills = total_trades - cancellations - bad_fills
    good_fill_profit = good_fills * 0.0002  # Небольшая прибыль от хороших сделок
    
    total_pnl = good_fill_profit - cancel_cost - bad_fill_cost
    cancel_rate = cancellations / total_trades if total_trades > 0 else 0
    
    return {
        'cancellations': cancellations,
        'bad_fills': bad_fills,
        'total_pnl': total_pnl,
        'cancel_rate': cancel_rate,
        'good_fills': good_fills
    }


def objective(trial: optuna.Trial, df: pl.DataFrame, base_threshold: float) -> float:
    """
    Целевая функция для Optuna.
    Максимизируем PnL при минимизации bad_fills.
    
    Параметры установлены мягко для совместимости со всеми парами.
    """
    # Параметры для оптимизации (мягкие диапазоны)
    vol_multiplier = trial.suggest_float('vol_multiplier', 0.001, 0.1)  # Очень маленькие значения
    spread_multiplier = trial.suggest_float('spread_multiplier', 0.0001, 0.01)  # Экстремально маленькие
    
    # Симулируем торговлю
    metrics = simulate_trading(df, base_threshold, vol_multiplier, spread_multiplier)
    
    # Целевая функция: максимизируем PnL, штрафуем за bad_fills
    score = metrics['total_pnl'] - metrics['bad_fills'] * 0.01
    
    # Логируем метрики
    trial.set_user_attr('cancellations', metrics['cancellations'])
    trial.set_user_attr('bad_fills', metrics['bad_fills'])
    trial.set_user_attr('cancel_rate', metrics['cancel_rate'])
    trial.set_user_attr('good_fills', metrics['good_fills'])
    
    return score


def load_historical_data(data_path: Path) -> pl.DataFrame:
    """
    Загрузка исторических данных из Parquet файлов.
    Ожидаемые колонки: timestamp, mid_price, spread_bps, volatility_bps, price_deviation_bps
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")
    
    # Загружаем все Parquet файлы из директории
    parquet_files = list(data_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_path}")
    
    print(f"Loading {len(parquet_files)} parquet files...")
    df = pl.concat([pl.read_parquet(f) for f in parquet_files])
    
    # Проверяем наличие необходимых колонок
    required_cols = ['mid_price', 'spread_bps', 'volatility_bps', 'price_deviation_bps']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}. Creating synthetic data...")
        df = create_synthetic_data()
    
    return df


def create_synthetic_data(n_samples: int = 10000) -> pl.DataFrame:
    """
    Создание синтетических данных для тестирования.
    """
    np.random.seed(42)
    
    # Генерируем синтетические данные
    timestamps = np.arange(n_samples) * 1000
    mid_prices = 100.0 + np.cumsum(np.random.randn(n_samples) * 0.1)
    
    # Волатильность: случайная с периодами высокой волатильности
    volatility_bps = np.abs(np.random.randn(n_samples) * 20 + 30)
    volatility_bps[5000:5500] *= 3  # Период высокой волатильности
    
    # Спред: коррелирует с волатильностью
    spread_bps = volatility_bps * 0.3 + np.random.randn(n_samples) * 5 + 10
    
    # Отклонение цены: больше при высокой волатильности
    price_deviation_bps = np.abs(np.random.randn(n_samples) * volatility_bps * 0.5)
    
    return pl.DataFrame({
        'timestamp': timestamps,
        'mid_price': mid_prices,
        'volatility_bps': volatility_bps,
        'spread_bps': spread_bps,
        'price_deviation_bps': price_deviation_bps
    })


def main():
    parser = argparse.ArgumentParser(description='Optimize adaptive threshold parameters')
    parser.add_argument('--data-path', type=str, default='bots/*/data/raw',
                        help='Path to historical data (parquet files)')
    parser.add_argument('--base-threshold', type=float, default=500.0,
                        help='Base threshold in bps (default: 500.0 - мягкий параметр)')
    parser.add_argument('--n-trials', type=int, default=100,
                        help='Number of optimization trials (default: 100)')
    parser.add_argument('--use-synthetic', action='store_true',
                        help='Use synthetic data for testing')
    
    args = parser.parse_args()
    
    # Загружаем данные
    if args.use_synthetic:
        print("Using synthetic data...")
        df = create_synthetic_data()
    else:
        data_path = Path(args.data_path)
        try:
            df = load_historical_data(data_path)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Falling back to synthetic data...")
            df = create_synthetic_data()
    
    print(f"Loaded {len(df)} samples")
    print(f"Volatility range: {df['volatility_bps'].min():.2f} - {df['volatility_bps'].max():.2f} bps")
    print(f"Spread range: {df['spread_bps'].min():.2f} - {df['spread_bps'].max():.2f} bps")
    
    # Создаем Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    )
    
    # Запускаем оптимизацию
    print(f"\nStarting optimization with {args.n_trials} trials...")
    study.optimize(
        lambda trial: objective(trial, df, args.base_threshold),
        n_trials=args.n_trials,
        show_progress_bar=True
    )
    
    # Выводим результаты
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    
    best_trial = study.best_trial
    print(f"\nBest score: {best_trial.value:.6f}")
    print(f"\nOptimal parameters:")
    print(f"  vol_multiplier: {best_trial.params['vol_multiplier']:.4f}")
    print(f"  spread_multiplier: {best_trial.params['spread_multiplier']:.4f}")
    
    print(f"\nMetrics with optimal parameters:")
    print(f"  Cancellations: {best_trial.user_attrs['cancellations']}")
    print(f"  Bad fills: {best_trial.user_attrs['bad_fills']}")
    print(f"  Good fills: {best_trial.user_attrs['good_fills']}")
    print(f"  Cancel rate: {best_trial.user_attrs['cancel_rate']:.2%}")
    
    # Сохраняем результаты в TOML формате
    output_path = Path('optimal_thresholds.toml')
    with open(output_path, 'w') as f:
        f.write("# Задача 210: Оптимальные параметры адаптивных порогов\n")
        f.write("# Сгенерировано threshold_optimizer.py\n")
        f.write("# ПАРАМЕТРЫ УСТАНОВЛЕНЫ МЯГКО для совместимости со всеми парами\n\n")
        f.write("[adaptive_thresholds]\n")
        f.write("enabled = true\n")
        f.write(f"base_threshold_bps = {args.base_threshold}\n")
        f.write(f"vol_multiplier = {best_trial.params['vol_multiplier']:.6f}\n")
        f.write(f"spread_multiplier = {best_trial.params['spread_multiplier']:.6f}\n")
        f.write("min_threshold_bps = 100.0\n")
        f.write("max_threshold_bps = 1000.0\n")
        f.write("\n# Примечание: Параметры установлены мягко, чтобы не мешать торговле\n")
        f.write("# на разных монетах, включая низколиквидные альткойны\n")
    
    print(f"\nOptimal parameters saved to: {output_path}")


if __name__ == '__main__':
    main()
