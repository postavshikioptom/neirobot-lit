#!/usr/bin/env python3
"""
Задача 206: Оптимизатор параметров Smart Order Routing (SOR)

Анализирует исторические логи из задач 201-204 для подбора оптимальных значений:
- critical_signal: Порог для перехода в Aggressive (Taker) режим
- max_size_ratio: % от объема уровня, выше которого включается Slicing

Выход: Рекомендованные параметры для config.toml конкретного бота.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SorOptimizationResult:
    """Результат оптимизации параметров SOR"""
    critical_signal: float
    max_size_ratio: float
    default_urgency: float
    slice_interval_ms: int
    
    # Метрики качества
    avg_slippage_bps: float
    avg_fill_rate: float
    total_trades: int
    aggressive_trades: int
    passive_trades: int
    twap_trades: int


class ExecutionOptimizer:
    """Оптимизатор параметров Smart Order Routing"""
    
    def __init__(self, bot_symbol: str, logs_dir: Path):
        """
        Инициализация оптимизатора
        
        Args:
            bot_symbol: Символ бота (например, BTCUSDT)
            logs_dir: Путь к папке логов бота
        """
        self.bot_symbol = bot_symbol
        self.logs_dir = Path(logs_dir)
        self.trades_df: Optional[pd.DataFrame] = None
        self.execution_quality_df: Optional[pd.DataFrame] = None
        self.order_context_df: Optional[pd.DataFrame] = None
    
    def load_logs(self) -> bool:
        """
        Загрузка логов из CSV файлов
        
        Returns:
            True если логи успешно загружены, False иначе
        """
        trades_path = self.logs_dir / "trades.csv"
        execution_quality_path = self.logs_dir / "execution_quality.csv"
        order_context_path = self.logs_dir / "order_context.csv"
        
        try:
            if trades_path.exists():
                self.trades_df = pd.read_csv(trades_path)
                logger.info(f"Загружено {len(self.trades_df)} записей из trades.csv")
            else:
                logger.warning(f"Файл {trades_path} не найден")
                return False
            
            if execution_quality_path.exists():
                self.execution_quality_df = pd.read_csv(execution_quality_path)
                logger.info(f"Загружено {len(self.execution_quality_df)} записей из execution_quality.csv")
            
            if order_context_path.exists():
                self.order_context_df = pd.read_csv(order_context_path)
                logger.info(f"Загружено {len(self.order_context_df)} записей из order_context.csv")
            
            return True
        except Exception as e:
            logger.error(f"Ошибка при загрузке логов: {e}")
            return False
    
    def merge_logs(self) -> pd.DataFrame:
        """
        Объединение логов для анализа
        
        Returns:
            Объединённый DataFrame
        """
        if self.trades_df is None:
            raise ValueError("trades.csv не загружен")
        
        merged_df = self.trades_df.copy()
        
        # Мердж с execution_quality.csv по order_id
        if self.execution_quality_df is not None and 'order_id' in merged_df.columns:
            merged_df = merged_df.merge(
                self.execution_quality_df,
                on='order_id',
                how='left',
                suffixes=('', '_eq')
            )
        
        # Мердж с order_context.csv по order_id
        if self.order_context_df is not None and 'order_id' in merged_df.columns:
            merged_df = merged_df.merge(
                self.order_context_df,
                on='order_id',
                how='left',
                suffixes=('', '_ctx')
            )
        
        return merged_df
    
    def analyze_signal_strength_impact(self, merged_df: pd.DataFrame) -> Tuple[float, float]:
        """
        Анализ влияния силы сигнала на качество исполнения
        
        Args:
            merged_df: Объединённый DataFrame
        
        Returns:
            Кортеж (optimal_critical_signal, avg_slippage_at_optimal)
        """
        if 'signal_strength' not in merged_df.columns or 'slippage_bps' not in merged_df.columns:
            logger.warning("Колонки signal_strength или slippage_bps не найдены")
            return 0.75, 0.0
        
        # Группируем по диапазонам силы сигнала
        signal_bins = np.arange(0.0, 1.05, 0.05)
        merged_df['signal_bin'] = pd.cut(merged_df['signal_strength'], bins=signal_bins)
        
        # Рассчитываем среднее скольжение для каждого диапазона
        slippage_by_signal = merged_df.groupby('signal_bin')['slippage_bps'].agg(['mean', 'count'])
        
        logger.info("\nАнализ влияния силы сигнала на скольжение:")
        logger.info(slippage_by_signal)
        
        # Находим оптимальный порог (где скольжение минимально)
        valid_bins = slippage_by_signal[slippage_by_signal['count'] >= 5]
        if len(valid_bins) > 0:
            optimal_bin = valid_bins['mean'].idxmin()
            optimal_signal = optimal_bin.mid
            min_slippage = valid_bins['mean'].min()
        else:
            optimal_signal = 0.75
            min_slippage = 0.0
        
        logger.info(f"Оптимальный critical_signal: {optimal_signal:.2f} (скольжение: {min_slippage:.2f} bps)")
        
        return optimal_signal, min_slippage
    
    def analyze_size_ratio_impact(self, merged_df: pd.DataFrame) -> Tuple[float, float]:
        """
        Анализ влияния соотношения размера ордера к объему уровня на качество исполнения
        
        Args:
            merged_df: Объединённый DataFrame
        
        Returns:
            Кортеж (optimal_max_size_ratio, avg_fill_rate_at_optimal)
        """
        if 'order_size' not in merged_df.columns or 'level_total_vol' not in merged_df.columns:
            logger.warning("Колонки order_size или level_total_vol не найдены")
            return 0.3, 0.0
        
        # Рассчитываем соотношение размера ордера к объему уровня
        merged_df['size_ratio'] = merged_df['order_size'] / (merged_df['level_total_vol'] + 1e-8)
        
        # Группируем по диапазонам соотношения
        ratio_bins = np.arange(0.0, 1.05, 0.1)
        merged_df['ratio_bin'] = pd.cut(merged_df['size_ratio'], bins=ratio_bins)
        
        # Рассчитываем среднюю fill_rate для каждого диапазона
        if 'fill_rate' in merged_df.columns:
            fill_rate_by_ratio = merged_df.groupby('ratio_bin')['fill_rate'].agg(['mean', 'count'])
            
            logger.info("\nАнализ влияния соотношения размера на fill_rate:")
            logger.info(fill_rate_by_ratio)
            
            # Находим оптимальное соотношение (где fill_rate максимальна)
            valid_bins = fill_rate_by_ratio[fill_rate_by_ratio['count'] >= 5]
            if len(valid_bins) > 0:
                optimal_bin = valid_bins['mean'].idxmax()
                optimal_ratio = optimal_bin.mid
                max_fill_rate = valid_bins['mean'].max()
            else:
                optimal_ratio = 0.3
                max_fill_rate = 0.0
        else:
            logger.warning("Колонка fill_rate не найдена")
            optimal_ratio = 0.3
            max_fill_rate = 0.0
        
        logger.info(f"Оптимальный max_size_ratio: {optimal_ratio:.2f} (fill_rate: {max_fill_rate:.2%})")
        
        return optimal_ratio, max_fill_rate
    
    def calculate_strategy_distribution(self, merged_df: pd.DataFrame) -> Dict[str, int]:
        """
        Расчёт распределения стратегий исполнения
        
        Args:
            merged_df: Объединённый DataFrame
        
        Returns:
            Словарь с количеством сделок по каждой стратегии
        """
        distribution = {
            'passive': 0,
            'aggressive': 0,
            'twap_slice': 0,
        }
        
        if 'strategy' in merged_df.columns:
            strategy_counts = merged_df['strategy'].value_counts()
            for strategy, count in strategy_counts.items():
                if strategy.lower() in distribution:
                    distribution[strategy.lower()] = int(count)
        
        logger.info("\nРаспределение стратегий исполнения:")
        for strategy, count in distribution.items():
            logger.info(f"  {strategy}: {count} сделок")
        
        return distribution
    
    def optimize(self) -> SorOptimizationResult:
        """
        Выполнение оптимизации параметров SOR
        
        Returns:
            Результат оптимизации
        """
        if not self.load_logs():
            logger.error("Не удалось загрузить логи")
            # Возвращаем значения по умолчанию
            return SorOptimizationResult(
                critical_signal=0.75,
                max_size_ratio=0.3,
                default_urgency=0.5,
                slice_interval_ms=100,
                avg_slippage_bps=0.0,
                avg_fill_rate=0.0,
                total_trades=0,
                aggressive_trades=0,
                passive_trades=0,
                twap_trades=0,
            )
        
        # Объединяем логи
        merged_df = self.merge_logs()
        
        # Анализируем влияние параметров
        optimal_critical_signal, avg_slippage = self.analyze_signal_strength_impact(merged_df)
        optimal_max_size_ratio, avg_fill_rate = self.analyze_size_ratio_impact(merged_df)
        
        # Рассчитываем распределение стратегий
        strategy_dist = self.calculate_strategy_distribution(merged_df)
        
        result = SorOptimizationResult(
            critical_signal=optimal_critical_signal,
            max_size_ratio=optimal_max_size_ratio,
            default_urgency=0.5,
            slice_interval_ms=100,
            avg_slippage_bps=avg_slippage,
            avg_fill_rate=avg_fill_rate,
            total_trades=len(merged_df),
            aggressive_trades=strategy_dist.get('aggressive', 0),
            passive_trades=strategy_dist.get('passive', 0),
            twap_trades=strategy_dist.get('twap_slice', 0),
        )
        
        return result
    
    def save_recommendations(self, result: SorOptimizationResult, output_path: Path) -> None:
        """
        Сохранение рекомендованных параметров в JSON
        
        Args:
            result: Результат оптимизации
            output_path: Путь для сохранения
        """
        recommendations = {
            'symbol': self.bot_symbol,
            'sor_config': {
                'critical_signal': round(result.critical_signal, 3),
                'max_size_ratio': round(result.max_size_ratio, 3),
                'default_urgency': round(result.default_urgency, 3),
                'slice_interval_ms': result.slice_interval_ms,
            },
            'metrics': {
                'avg_slippage_bps': round(result.avg_slippage_bps, 2),
                'avg_fill_rate': round(result.avg_fill_rate, 4),
                'total_trades': result.total_trades,
                'strategy_distribution': {
                    'aggressive': result.aggressive_trades,
                    'passive': result.passive_trades,
                    'twap_slice': result.twap_trades,
                }
            }
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        logger.info(f"\nРекомендации сохранены в {output_path}")
        logger.info(json.dumps(recommendations, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description='Оптимизатор параметров Smart Order Routing (SOR)'
    )
    parser.add_argument(
        '--symbol',
        required=True,
        help='Символ бота (например, BTCUSDT)'
    )
    parser.add_argument(
        '--logs-dir',
        required=True,
        help='Путь к папке логов бота'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Путь для сохранения рекомендаций (по умолчанию: logs_dir/sor_recommendations.json)'
    )
    
    args = parser.parse_args()
    
    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        logger.error(f"Папка логов не найдена: {logs_dir}")
        return 1
    
    output_path = Path(args.output) if args.output else logs_dir / 'sor_recommendations.json'
    
    optimizer = ExecutionOptimizer(args.symbol, logs_dir)
    result = optimizer.optimize()
    optimizer.save_recommendations(result, output_path)
    
    return 0


if __name__ == '__main__':
    exit(main())
