"""
Задача 214: Монте-Карло симуляция вариативности задержек (Latency Perturbation)

Модуль для генерации случайных задержек с логнормальным распределением.
Используется для стресс-тестирования стратегии путем внесения джиттера в задержки исполнения.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from scipy.stats import lognorm
import pandas as pd


class LatencyGenerator:
    """
    Генератор задержек с логнормальным распределением.
    
    Логнормальное распределение - стандарт для моделирования сетевых задержек,
    так как задержки всегда положительны и имеют длинный правый хвост (редкие большие лаги).
    """
    
    def __init__(
        self,
        mean_ms: float = 20.0,
        std_ms: float = 15.0,
        seed: Optional[int] = None,
        execution_quality_csv: Optional[str] = None
    ):
        """
        Инициализация генератора задержек.
        
        Args:
            mean_ms: Среднее значение задержки в миллисекундах (по умолчанию 20ms)
            std_ms: Стандартное отклонение задержки в миллисекундах (по умолчанию 15ms)
            seed: Seed для генератора случайных чисел (для воспроизводимости)
            execution_quality_csv: Путь к файлу execution_quality.csv для загрузки реальных параметров
        """
        self.rng = np.random.default_rng(seed)
        
        # Попытка загрузить параметры из CSV
        if execution_quality_csv and Path(execution_quality_csv).exists():
            mean_ms, std_ms = self._load_params_from_csv(execution_quality_csv)
            print(f"[LatencyGenerator] Loaded params from CSV: mean={mean_ms:.2f}ms, std={std_ms:.2f}ms")
        else:
            print(f"[LatencyGenerator] Using default params: mean={mean_ms:.2f}ms, std={std_ms:.2f}ms")
        
        self.mean_ms = mean_ms
        self.std_ms = std_ms
        
        # Преобразование параметров в параметры логнормального распределения
        # Для lognorm в scipy: s=sigma, scale=exp(mu)
        # где mu и sigma - параметры нормального распределения логарифма
        self.mu, self.sigma = self._convert_to_lognorm_params(mean_ms, std_ms)
        
        # Создание распределения
        self.distribution = lognorm(s=self.sigma, scale=np.exp(self.mu))
    
    @staticmethod
    def _convert_to_lognorm_params(mean: float, std: float) -> Tuple[float, float]:
        """
        Преобразование среднего и стандартного отклонения в параметры логнормального распределения.
        
        Формулы:
        - mu = ln(mean^2 / sqrt(mean^2 + std^2))
        - sigma = sqrt(ln(1 + (std/mean)^2))
        
        Args:
            mean: Среднее значение
            std: Стандартное отклонение
            
        Returns:
            Кортеж (mu, sigma) - параметры логнормального распределения
        """
        variance = std ** 2
        mean_squared = mean ** 2
        
        # Формулы для преобразования
        mu = np.log(mean_squared / np.sqrt(mean_squared + variance))
        sigma = np.sqrt(np.log(1 + variance / mean_squared))
        
        return mu, sigma
    
    @staticmethod
    def _load_params_from_csv(csv_path: str) -> Tuple[float, float]:
        """
        Загрузка параметров задержек из execution_quality.csv.
        
        Ожидается, что CSV содержит колонку с задержками (например, 'latency_ms').
        Рассчитываются mean и std по этим данным.
        
        Args:
            csv_path: Путь к CSV файлу
            
        Returns:
            Кортеж (mean, std) в миллисекундах
        """
        try:
            df = pd.read_csv(csv_path)
            
            # Ищем колонку с задержками (возможные варианты названий)
            latency_col = None
            for col in ['latency_ms', 'latency', 'execution_latency_ms', 'network_latency_ms']:
                if col in df.columns:
                    latency_col = col
                    break
            
            if latency_col is None:
                raise ValueError("No latency column found in CSV")
            
            latencies = df[latency_col].dropna()
            mean_ms = float(latencies.mean())
            std_ms = float(latencies.std())
            
            return mean_ms, std_ms
            
        except Exception as e:
            print(f"[LatencyGenerator] Error loading CSV: {e}, using defaults")
            return 20.0, 15.0
    
    def generate(self, size: int = 1) -> np.ndarray:
        """
        Генерация случайных задержек.
        
        Args:
            size: Количество задержек для генерации
            
        Returns:
            Массив задержек в миллисекундах
        """
        # Генерируем из логнормального распределения
        latencies = self.distribution.rvs(size=size, random_state=self.rng)
        
        # Убеждаемся, что все значения положительны
        latencies = np.maximum(latencies, 0.1)  # Минимум 0.1ms
        
        return latencies
    
    def generate_single(self) -> float:
        """
        Генерация одной случайной задержки.
        
        Returns:
            Задержка в миллисекундах
        """
        return float(self.generate(size=1)[0])
    
    def get_percentile(self, percentile: float, n_samples: int = 10000) -> float:
        """
        Расчет перцентиля распределения задержек.
        
        Args:
            percentile: Перцентиль (0-100)
            n_samples: Количество сэмплов для расчета
            
        Returns:
            Значение перцентиля в миллисекундах
        """
        samples = self.generate(size=n_samples)
        return float(np.percentile(samples, percentile))
    
    def get_stats(self, n_samples: int = 10000) -> dict:
        """
        Получение статистики распределения задержек.
        
        Args:
            n_samples: Количество сэмплов для расчета
            
        Returns:
            Словарь со статистикой
        """
        samples = self.generate(size=n_samples)
        
        return {
            'mean_ms': float(np.mean(samples)),
            'std_ms': float(np.std(samples)),
            'median_ms': float(np.median(samples)),
            'p50_ms': float(np.percentile(samples, 50)),
            'p95_ms': float(np.percentile(samples, 95)),
            'p99_ms': float(np.percentile(samples, 99)),
            'min_ms': float(np.min(samples)),
            'max_ms': float(np.max(samples))
        }
