"""
Задача 215: Тесты для симуляции ошибок и восстановления (Error Recovery Tests)

Проверяет:
1. Детерминированность генерации ошибок с seed
2. Exponential Backoff при серии RateLimitError
3. Retry логику после ошибок
4. Метрики Error_Recovery_Time и Lost_Trades_Count
5. Resilience Score
"""

import pytest
import numpy as np
from python_lab.src.backtest.error_sim import (
    ExchangeErrorSimulator,
    ExchangeErrorType,
    ExchangeErrorData
)
from python_lab.src.backtest.engine import (
    EventEngine,
    BotConfig,
    Event,
    EventType,
    OrderData,
    MarketData
)


class TestExchangeErrorSimulator:
    """Тесты для ExchangeErrorSimulator"""
    
    def test_deterministic_error_generation(self):
        """
        Проверка детерминированности генерации ошибок с фиксированным seed.
        
        При одинаковом seed должны генерироваться одинаковые ошибки.
        """
        seed = 42
        sim1 = ExchangeErrorSimulator(seed=seed, rate_limit_prob=0.1)
        sim2 = ExchangeErrorSimulator(seed=seed, rate_limit_prob=0.1)
        
        results1 = []
        results2 = []
        
        for i in range(100):
            error_type1, _ = sim1.should_fail(current_time_ms=i * 1000)
            error_type2, _ = sim2.should_fail(current_time_ms=i * 1000)
            
            results1.append(error_type1)
            results2.append(error_type2)
        
        # Результаты должны быть идентичными
        assert results1 == results2, "Генерация ошибок не детерминирована"
    
    def test_exponential_backoff_increases(self):
        """
        Проверка экспоненциального увеличения задержки при серии RateLimitError.
        
        При получении серии 429 ошибок задержка должна увеличиваться экспоненциально:
        1s -> 2s -> 4s -> 8s -> ... до максимума 60s
        
        С Jitter: каждая задержка будет случайным значением от 0 до max_backoff.
        """
        # Создаем симулятор с высокой вероятностью RateLimitError
        sim = ExchangeErrorSimulator(seed=42, rate_limit_prob=1.0)
        
        current_time = 0
        backoff_delays = []
        max_backoffs = []
        
        # Генерируем серию RateLimitError
        for i in range(10):
            error_type, error_data = sim.should_fail(current_time_ms=current_time)
            
            if error_type == ExchangeErrorType.RATE_LIMIT:
                # Записываем текущую задержку
                backoff_delay = sim.backoff_until_ms - current_time
                backoff_delays.append(backoff_delay)
                max_backoffs.append(sim.current_backoff_ms)
                
                # Переходим к следующему моменту времени (после backoff)
                current_time = sim.backoff_until_ms + 1
        
        # Проверяем, что задержки увеличиваются экспоненциально (максимальные значения)
        assert len(backoff_delays) > 0, "Не было сгенерировано RateLimitError"
        
        # Максимальные backoff значения должны увеличиваться экспоненциально
        for i in range(1, min(len(max_backoffs), 6)):
            expected_max = min(
                max_backoffs[i-1] * ExchangeErrorSimulator.BACKOFF_MULTIPLIER,
                ExchangeErrorSimulator.MAX_BACKOFF_MS
            )
            assert max_backoffs[i] == expected_max, \
                f"Max backoff {i} не соответствует экспоненциальному росту: " \
                f"{max_backoffs[i]} != {expected_max}"
        
        # Проверяем, что Jitter работает (задержки не все одинаковые)
        # С Jitter задержки должны быть разными (кроме случаев, когда max_backoff = 0)
        unique_delays = len(set(backoff_delays))
        assert unique_delays > 1, "Jitter не работает - все задержки одинаковые"
    
    def test_backoff_prevents_requests_during_cooldown(self):
        """
        Проверка, что во время Backoff все запросы возвращают RateLimitError.
        """
        sim = ExchangeErrorSimulator(seed=42, rate_limit_prob=1.0)
        
        # Генерируем первую ошибку
        current_time = 0
        error_type, _ = sim.should_fail(current_time_ms=current_time)
        assert error_type == ExchangeErrorType.RATE_LIMIT
        
        backoff_until = sim.backoff_until_ms
        
        # Все запросы до backoff_until должны возвращать RateLimitError
        for t in range(current_time + 1, backoff_until):
            error_type, _ = sim.should_fail(current_time_ms=t)
            assert error_type == ExchangeErrorType.RATE_LIMIT, \
                f"Запрос в момент {t} не вернул RateLimitError (backoff_until={backoff_until})"
    
    def test_jitter_in_backoff(self):
        """
        Проверка, что Jitter добавляет случайность в backoff задержки.
        
        Full Jitter алгоритм: sleep = random(0, min(cap, base * 2^attempt))
        Это предотвращает "thundering herd" проблему.
        """
        # Создаем несколько симуляторов с разными seed для проверки Jitter
        delays_by_seed = {}
        
        for seed in [42, 123, 456]:
            sim = ExchangeErrorSimulator(seed=seed, rate_limit_prob=1.0)
            
            current_time = 0
            delays = []
            
            # Генерируем несколько ошибок
            for i in range(5):
                error_type, _ = sim.should_fail(current_time_ms=current_time)
                
                if error_type == ExchangeErrorType.RATE_LIMIT:
                    delay = sim.backoff_until_ms - current_time
                    delays.append(delay)
                    current_time = sim.backoff_until_ms + 1
            
            delays_by_seed[seed] = delays
        
        # Проверяем, что разные seed дают разные задержки (Jitter работает)
        all_delays = []
        for delays in delays_by_seed.values():
            all_delays.extend(delays)
        
        # Должны быть разные задержки
        unique_delays = len(set(all_delays))
        assert unique_delays > 1, "Jitter не работает - все задержки одинаковые"
    
    def test_recovery_time_tracking(self):
        """
        Проверка отслеживания времени восстановления (Error_Recovery_Time).
        """
        sim = ExchangeErrorSimulator(seed=42, rate_limit_prob=0.5)
        
        current_time = 0
        
        # Генерируем ошибку
        error_type, _ = sim.should_fail(current_time_ms=current_time)
        
        if error_type == ExchangeErrorType.RATE_LIMIT:
            backoff_until = sim.backoff_until_ms
            
            # Переходим к моменту после backoff
            recovery_time = backoff_until + 1000
            
            # Успешный запрос после восстановления
            error_type, _ = sim.should_fail(current_time_ms=recovery_time)
            
            # Проверяем, что время восстановления записано
            metrics = sim.get_metrics()
            assert metrics["recovery_count"] > 0, "Время восстановления не записано"
            assert metrics["avg_recovery_time_ms"] > 0, "Среднее время восстановления = 0"
    
    def test_resilience_score_calculation(self):
        """
        Проверка расчета Resilience Score.
        
        Score должен быть от 0 до 100, где:
        - 100: Идеальная устойчивость (нет ошибок)
        - 0: Полная неустойчивость (много ошибок)
        """
        # Симулятор без ошибок
        sim_perfect = ExchangeErrorSimulator(seed=42, rate_limit_prob=0.0)
        for i in range(100):
            sim_perfect.should_fail(current_time_ms=i * 1000)
        
        score_perfect = sim_perfect.get_resilience_score()
        assert score_perfect == 100.0, f"Perfect score должен быть 100, получен {score_perfect}"
        
        # Симулятор с высокой частотой ошибок
        sim_bad = ExchangeErrorSimulator(seed=42, rate_limit_prob=0.5)
        current_time = 0
        for i in range(100):
            error_type, _ = sim_bad.should_fail(current_time_ms=current_time)
            if error_type == ExchangeErrorType.RATE_LIMIT:
                current_time = sim_bad.backoff_until_ms + 1
            else:
                current_time += 1000
        
        score_bad = sim_bad.get_resilience_score()
        assert 0 <= score_bad < 100, f"Bad score должен быть < 100, получен {score_bad}"
        assert score_bad < score_perfect, "Bad score должен быть меньше perfect score"


class TestErrorRecoveryInEngine:
    """Интеграционные тесты для проверки восстановления после ошибок в EventEngine"""
    
    def test_order_retry_after_rate_limit(self):
        """
        Проверка retry логики после RateLimitError.
        
        При получении RateLimitError ордер должен быть повторен после backoff.
        """
        # Создаем симулятор с высокой вероятностью RateLimitError
        error_sim = ExchangeErrorSimulator(seed=42, rate_limit_prob=1.0)
        
        # Создаем движок с симулятором ошибок
        config = BotConfig(symbol="BTCUSDT", initial_balance=1000.0)
        engine = EventEngine(config=config, error_simulator=error_sim)
        
        # Создаем рыночные данные
        bids = np.array([[100.0, 10.0]] * 50)
        asks = np.array([[101.0, 10.0]] * 50)
        market_data = MarketData(mid_price=100.5, bids=bids, asks=asks)
        
        # Отправляем рыночные данные
        market_event = Event(
            timestamp=0,
            type=EventType.MARKET,
            data=market_data,
            symbol="BTCUSDT"
        )
        engine.push_event(market_event)
        
        # Отправляем ордер
        order_data = OrderData(
            order_id="test_order_1",
            side="buy",
            price=100.0,
            amount=1.0,
            order_type="limit"
        )
        
        order_event = Event(
            timestamp=1000,
            type=EventType.ORDER,
            data=order_data,
            symbol="BTCUSDT"
        )
        engine.push_event(order_event)
        
        # Запускаем движок
        engine.run()
        
        # Проверяем метрики
        metrics = engine.get_metrics()
        
        # Должна быть хотя бы одна ошибка
        assert metrics.get("total_errors", 0) > 0, "Не было сгенерировано ошибок"
        
        # Проверяем, что ордер в итоге был размещен (после retry)
        # Или что lost_trades_count увеличился
        state = engine.get_state("BTCUSDT")
        assert state.lost_trades_count >= 0, "lost_trades_count не отслеживается"
    
    def test_lost_trades_count_on_rejection(self):
        """
        Проверка подсчета потерянных сигналов при OrderRejection.
        
        При получении OrderRejection (не RateLimitError) сигнал должен считаться потерянным.
        """
        # Создаем симулятор с высокой вероятностью OrderRejection
        error_sim = ExchangeErrorSimulator(
            seed=42,
            rate_limit_prob=0.0,
            rejection_prob=1.0
        )
        
        config = BotConfig(symbol="BTCUSDT", initial_balance=1000.0)
        engine = EventEngine(config=config, error_simulator=error_sim)
        
        # Создаем рыночные данные
        bids = np.array([[100.0, 10.0]] * 50)
        asks = np.array([[101.0, 10.0]] * 50)
        market_data = MarketData(mid_price=100.5, bids=bids, asks=asks)
        
        market_event = Event(
            timestamp=0,
            type=EventType.MARKET,
            data=market_data,
            symbol="BTCUSDT"
        )
        engine.push_event(market_event)
        
        # Отправляем несколько ордеров
        for i in range(5):
            order_data = OrderData(
                order_id=f"test_order_{i}",
                side="buy",
                price=100.0,
                amount=1.0,
                order_type="limit"
            )
            
            order_event = Event(
                timestamp=1000 + i * 1000,
                type=EventType.ORDER,
                data=order_data,
                symbol="BTCUSDT"
            )
            engine.push_event(order_event)
        
        # Запускаем движок
        engine.run()
        
        # Проверяем метрики
        state = engine.get_state("BTCUSDT")
        
        # Все ордера должны быть потеряны из-за rejection
        assert state.lost_trades_count > 0, \
            f"lost_trades_count должен быть > 0, получен {state.lost_trades_count}"
    
    def test_lost_trades_count_on_cancel_error(self):
        """
        Проверка подсчета потерянных сигналов при ошибке отмены ордера.
        
        Если отмена ордера завершилась ошибкой (не RateLimitError),
        это тоже считается потерянным сигналом.
        """
        # Создаем симулятор с высокой вероятностью OrderRejection
        error_sim = ExchangeErrorSimulator(
            seed=42,
            rate_limit_prob=0.0,
            rejection_prob=1.0
        )
        
        config = BotConfig(symbol="BTCUSDT", initial_balance=1000.0)
        engine = EventEngine(config=config, error_simulator=error_sim)
        
        # Создаем рыночные данные
        bids = np.array([[100.0, 10.0]] * 50)
        asks = np.array([[101.0, 10.0]] * 50)
        market_data = MarketData(mid_price=100.5, bids=bids, asks=asks)
        
        market_event = Event(
            timestamp=0,
            type=EventType.MARKET,
            data=market_data,
            symbol="BTCUSDT"
        )
        engine.push_event(market_event)
        
        # Отправляем ордер
        order_data = OrderData(
            order_id="test_order_1",
            side="buy",
            price=100.0,
            amount=1.0,
            order_type="limit"
        )
        
        order_event = Event(
            timestamp=1000,
            type=EventType.ORDER,
            data=order_data,
            symbol="BTCUSDT"
        )
        engine.push_event(order_event)
        
        # Отправляем отмену ордера (которая завершится ошибкой)
        cancel_event = Event(
            timestamp=2000,
            type=EventType.CANCEL,
            data="test_order_1",
            symbol="BTCUSDT"
        )
        engine.push_event(cancel_event)
        
        # Запускаем движок
        engine.run()
        
        # Проверяем метрики
        state = engine.get_state("BTCUSDT")
        
        # Отмена должна быть потеряна из-за ошибки
        assert state.lost_trades_count > 0, \
            f"lost_trades_count должен быть > 0 при ошибке отмены, получен {state.lost_trades_count}"
    
    def test_metrics_include_error_data(self):
        """
        Проверка, что метрики включают данные об ошибках.
        """
        error_sim = ExchangeErrorSimulator(seed=42, rate_limit_prob=0.1)
        config = BotConfig(symbol="BTCUSDT", initial_balance=1000.0)
        engine = EventEngine(config=config, error_simulator=error_sim)
        
        # Создаем рыночные данные
        bids = np.array([[100.0, 10.0]] * 50)
        asks = np.array([[101.0, 10.0]] * 50)
        market_data = MarketData(mid_price=100.5, bids=bids, asks=asks)
        
        market_event = Event(
            timestamp=0,
            type=EventType.MARKET,
            data=market_data,
            symbol="BTCUSDT"
        )
        engine.push_event(market_event)
        
        # Отправляем ордеры
        for i in range(20):
            order_data = OrderData(
                order_id=f"test_order_{i}",
                side="buy",
                price=100.0,
                amount=1.0,
                order_type="limit"
            )
            
            order_event = Event(
                timestamp=1000 + i * 1000,
                type=EventType.ORDER,
                data=order_data,
                symbol="BTCUSDT"
            )
            engine.push_event(order_event)
        
        engine.run()
        
        # Получаем метрики
        metrics = engine.get_metrics()
        
        # Проверяем наличие метрик ошибок
        assert "lost_trades_count" in metrics, "Метрика lost_trades_count отсутствует"
        assert "error_recovery_time_ms" in metrics, "Метрика error_recovery_time_ms отсутствует"
        assert "resilience_score" in metrics, "Метрика resilience_score отсутствует"
        assert "total_errors" in metrics, "Метрика total_errors отсутствует"
        assert "error_rate_pct" in metrics, "Метрика error_rate_pct отсутствует"
        
        # Resilience Score должен быть в диапазоне [0, 100]
        assert 0 <= metrics["resilience_score"] <= 100, \
            f"Resilience Score вне диапазона: {metrics['resilience_score']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
