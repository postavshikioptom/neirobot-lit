"""
Задача 215: Симуляция отказов и ошибок ордеров (Order Rejection and Error Simulation)

Реализует механизм симуляции негативных ответов от биржи Bybit V5.
Критично для тестирования устойчивости OrderManager и логики риск-менеджмента.
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import os


class ExchangeErrorType(Enum):
    """Типы ошибок биржи (Bybit V5 API)"""
    RATE_LIMIT = "rate_limit"  # Код 10006 / HTTP 429
    ORDER_REJECTION = "order_rejection"  # Коды 10001, 20001
    TIMEOUT = "timeout"  # Сетевой таймаут, статус неясен
    NONE = "none"  # Нет ошибки


@dataclass
class ExchangeErrorData:
    """Данные об ошибке биржи"""
    error_type: ExchangeErrorType
    error_code: int
    error_message: str
    timestamp_ms: int
    order_id: Optional[str] = None


class ExchangeErrorSimulator:
    """
    Симулятор ошибок биржи с детерминированной генерацией.
    
    Использует numpy.random.default_rng(seed) для воспроизводимости.
    Поддерживает Exponential Backoff с Jitter при получении RateLimitError.
    
    Jitter реализован согласно AWS рекомендациям (Full Jitter):
    sleep = random(0, min(cap, base * 2^attempt))
    """
    
    # Константы для Exponential Backoff (согласно задачам 085 и 234)
    INITIAL_BACKOFF_MS = 1000  # Начальная задержка 1 секунда
    MAX_BACKOFF_MS = 60000  # Максимальная задержка 60 секунд
    BACKOFF_MULTIPLIER = 2.0  # Экспоненциальный множитель
    JITTER_FACTOR = 1.0  # Full Jitter: случайное значение от 0 до backoff
    
    # Коды ошибок Bybit V5 API
    ERROR_CODE_RATE_LIMIT = 10006
    ERROR_CODE_ORDER_REJECTION_PARAM = 10001
    ERROR_CODE_ORDER_REJECTION_OTHER = 20001
    ERROR_CODE_TIMEOUT = -1  # Внутренний код для таймаута
    
    def __init__(
        self,
        seed: Optional[int] = None,
        rate_limit_prob: float = 0.0001,
        rejection_prob: float = 0.0001,
        timeout_prob: float = 0.00005,
        execution_quality_csv: Optional[str] = None
    ):
        """
        Инициализация симулятора ошибок.
        
        Args:
            seed: Seed для генератора случайных чисел (для воспроизводимости)
            rate_limit_prob: Вероятность RateLimitError (по умолчанию 0.01%)
            rejection_prob: Вероятность OrderRejection (по умолчанию 0.01%)
            timeout_prob: Вероятность Timeout (по умолчанию 0.005%)
            execution_quality_csv: Путь к CSV с данными для калибровки (опционально)
        """
        self.rng = np.random.default_rng(seed)
        
        # Калибровка вероятностей из CSV (если доступен)
        if execution_quality_csv and os.path.exists(execution_quality_csv):
            self._calibrate_from_csv(execution_quality_csv)
        else:
            self.rate_limit_prob = rate_limit_prob
            self.rejection_prob = rejection_prob
            self.timeout_prob = timeout_prob
        
        # Состояние Exponential Backoff
        self.backoff_until_ms = 0  # Timestamp до которого нужно ждать
        self.current_backoff_ms = self.INITIAL_BACKOFF_MS
        self.consecutive_rate_limits = 0
        
        # Метрики
        self.total_errors = 0
        self.rate_limit_errors = 0
        self.rejection_errors = 0
        self.timeout_errors = 0
        self.total_requests = 0
        
        # Для расчета Error_Recovery_Time
        self.last_error_time_ms = 0
        self.recovery_times = []
    
    def _calibrate_from_csv(self, csv_path: str):
        """
        Калибровка вероятностей ошибок из execution_quality.csv.
        
        Опциональная функция. Если CSV недоступен, используются дефолтные значения.
        """
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            
            # Предполагаем, что CSV содержит колонки: error_type, count
            if 'error_type' in df.columns and 'count' in df.columns:
                total_requests = df['count'].sum()
                
                rate_limit_count = df[df['error_type'] == 'rate_limit']['count'].sum()
                rejection_count = df[df['error_type'] == 'rejection']['count'].sum()
                timeout_count = df[df['error_type'] == 'timeout']['count'].sum()
                
                self.rate_limit_prob = rate_limit_count / total_requests if total_requests > 0 else 0.0001
                self.rejection_prob = rejection_count / total_requests if total_requests > 0 else 0.0001
                self.timeout_prob = timeout_count / total_requests if total_requests > 0 else 0.00005
            else:
                # Если формат CSV не соответствует, используем дефолтные значения
                self.rate_limit_prob = 0.0001
                self.rejection_prob = 0.0001
                self.timeout_prob = 0.00005
        except Exception:
            # Если не удалось прочитать CSV, используем дефолтные значения
            self.rate_limit_prob = 0.0001
            self.rejection_prob = 0.0001
            self.timeout_prob = 0.00005
    
    def should_fail(
        self,
        current_time_ms: int,
        order_id: Optional[str] = None
    ) -> Tuple[ExchangeErrorType, Optional[ExchangeErrorData]]:
        """
        Проверить, должна ли операция завершиться ошибкой.
        
        Вызывается перед каждой операцией с API в бэктестере.
        
        Args:
            current_time_ms: Текущее время в миллисекундах
            order_id: ID ордера (опционально)
            
        Returns:
            Кортеж (error_type, error_data)
            - Если ошибки нет: (ExchangeErrorType.NONE, None)
            - Если ошибка: (error_type, ExchangeErrorData)
        """
        self.total_requests += 1
        
        # Проверяем, находимся ли мы в состоянии Backoff
        if current_time_ms < self.backoff_until_ms:
            # Все еще в Backoff - возвращаем RateLimitError
            self.rate_limit_errors += 1
            self.total_errors += 1
            self.consecutive_rate_limits += 1
            
            error_data = ExchangeErrorData(
                error_type=ExchangeErrorType.RATE_LIMIT,
                error_code=self.ERROR_CODE_RATE_LIMIT,
                error_message="Too many visits! (Backoff active)",
                timestamp_ms=current_time_ms,
                order_id=order_id
            )
            
            return ExchangeErrorType.RATE_LIMIT, error_data
        
        # Генерируем случайную ошибку на основе вероятностей
        rand = self.rng.random()
        
        # Проверяем RateLimitError
        if rand < self.rate_limit_prob:
            self.rate_limit_errors += 1
            self.total_errors += 1
            self.consecutive_rate_limits += 1
            self.last_error_time_ms = current_time_ms
            
            # Применяем Exponential Backoff
            self._apply_backoff(current_time_ms)
            
            error_data = ExchangeErrorData(
                error_type=ExchangeErrorType.RATE_LIMIT,
                error_code=self.ERROR_CODE_RATE_LIMIT,
                error_message="Too many visits! Exceeded the API Rate Limit.",
                timestamp_ms=current_time_ms,
                order_id=order_id
            )
            
            return ExchangeErrorType.RATE_LIMIT, error_data
        
        # Проверяем OrderRejection
        rand -= self.rate_limit_prob
        if rand < self.rejection_prob:
            self.rejection_errors += 1
            self.total_errors += 1
            self.last_error_time_ms = current_time_ms
            
            # Случайно выбираем код ошибки (10001 или 20001)
            error_code = self.rng.choice([
                self.ERROR_CODE_ORDER_REJECTION_PARAM,
                self.ERROR_CODE_ORDER_REJECTION_OTHER
            ])
            
            error_message = "Request parameter error" if error_code == 10001 else "Order rejection"
            
            error_data = ExchangeErrorData(
                error_type=ExchangeErrorType.ORDER_REJECTION,
                error_code=error_code,
                error_message=error_message,
                timestamp_ms=current_time_ms,
                order_id=order_id
            )
            
            return ExchangeErrorType.ORDER_REJECTION, error_data
        
        # Проверяем Timeout
        rand -= self.rejection_prob
        if rand < self.timeout_prob:
            self.timeout_errors += 1
            self.total_errors += 1
            self.last_error_time_ms = current_time_ms
            
            error_data = ExchangeErrorData(
                error_type=ExchangeErrorType.TIMEOUT,
                error_code=self.ERROR_CODE_TIMEOUT,
                error_message="Network timeout - order status unclear",
                timestamp_ms=current_time_ms,
                order_id=order_id
            )
            
            return ExchangeErrorType.TIMEOUT, error_data
        
        # Нет ошибки - операция успешна
        # Сбрасываем счетчик consecutive_rate_limits
        if self.consecutive_rate_limits > 0:
            # Записываем время восстановления
            recovery_time = current_time_ms - self.last_error_time_ms
            self.recovery_times.append(recovery_time)
            self.consecutive_rate_limits = 0
            self.current_backoff_ms = self.INITIAL_BACKOFF_MS
        
        return ExchangeErrorType.NONE, None
    
    def _apply_backoff(self, current_time_ms: int):
        """
        Применить Exponential Backoff с Jitter при получении RateLimitError.
        
        Согласно задачам 085 и 234 и AWS рекомендациям:
        Используется Full Jitter алгоритм для распределения retry попыток.
        
        Formula: sleep = random(0, min(cap, base * 2^attempt))
        """
        # Вычисляем максимальную задержку для этого attempt
        max_backoff = min(
            int(self.current_backoff_ms * self.BACKOFF_MULTIPLIER),
            self.MAX_BACKOFF_MS
        )
        
        # Full Jitter: случайное значение от 0 до max_backoff
        # Это распределяет retry попытки и предотвращает "thundering herd"
        jittered_backoff = self.rng.integers(0, max_backoff + 1)
        
        self.backoff_until_ms = current_time_ms + jittered_backoff
        
        # Увеличиваем backoff для следующего раза
        self.current_backoff_ms = max_backoff
    
    def reset_backoff(self):
        """Сбросить состояние Backoff (для тестирования)"""
        self.backoff_until_ms = 0
        self.current_backoff_ms = self.INITIAL_BACKOFF_MS
        self.consecutive_rate_limits = 0
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Получить метрики симулятора ошибок.
        
        Returns:
            Словарь с метриками:
            - total_requests: Общее количество запросов
            - total_errors: Общее количество ошибок
            - rate_limit_errors: Количество RateLimitError
            - rejection_errors: Количество OrderRejection
            - timeout_errors: Количество Timeout
            - error_rate: Процент ошибок
            - avg_recovery_time_ms: Среднее время восстановления
        """
        error_rate = (self.total_errors / self.total_requests * 100) if self.total_requests > 0 else 0.0
        avg_recovery_time = np.mean(self.recovery_times) if self.recovery_times else 0.0
        
        return {
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "rate_limit_errors": self.rate_limit_errors,
            "rejection_errors": self.rejection_errors,
            "timeout_errors": self.timeout_errors,
            "error_rate_pct": error_rate,
            "avg_recovery_time_ms": avg_recovery_time,
            "max_recovery_time_ms": max(self.recovery_times) if self.recovery_times else 0.0,
            "recovery_count": len(self.recovery_times)
        }
    
    def get_resilience_score(self) -> float:
        """
        Рассчитать Resilience Score (оценка живучести).
        
        Оценка от 0 до 100, где:
        - 100: Идеальная устойчивость (нет ошибок или быстрое восстановление)
        - 0: Полная неустойчивость (много ошибок, долгое восстановление)
        
        Returns:
            Resilience Score (0-100)
        """
        if self.total_requests == 0:
            return 100.0
        
        # Компоненты оценки:
        # 1. Error Rate (чем меньше ошибок, тем лучше)
        error_rate = self.total_errors / self.total_requests
        error_score = max(0, 100 - error_rate * 10000)  # Штраф за ошибки
        
        # 2. Recovery Time (чем быстрее восстановление, тем лучше)
        if self.recovery_times:
            avg_recovery = np.mean(self.recovery_times)
            # Нормализуем: 1 секунда = 100, 60 секунд = 0
            recovery_score = max(0, 100 - (avg_recovery / 1000) * (100 / 60))
        else:
            recovery_score = 100.0
        
        # Итоговая оценка (средневзвешенная)
        resilience_score = (error_score * 0.6 + recovery_score * 0.4)
        
        return max(0.0, min(100.0, resilience_score))
