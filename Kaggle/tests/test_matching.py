"""
Задача 212: Тесты для симуляции очереди лимитных ордеров

Проверяет корректность логики FIFO очереди для лимитных ордеров.
"""

import pytest
import numpy as np
from python_lab.src.backtest.matching import (
    QueueTracker,
    OrderQueueManager,
    OrderSide,
    QueueModel
)


class TestQueueTracker:
    """Тесты для класса QueueTracker"""
    
    def test_conservative_model_initialization(self):
        """Тест инициализации консервативной модели"""
        tracker = QueueTracker(
            order_id="order_1",
            price=100.0,
            volume_at_level=10.0,
            side=OrderSide.BUY,
            queue_model=QueueModel.CONSERVATIVE
        )
        
        assert tracker.order_id == "order_1"
        assert tracker.price == 100.0
        assert tracker.side == OrderSide.BUY
        assert tracker.v_ahead == 10.0  # Весь объем перед нами
        assert tracker.is_filled is False
    
    def test_probabilistic_model_initialization(self):
        """Тест инициализации вероятностной модели"""
        np.random.seed(42)
        tracker = QueueTracker(
            order_id="order_2",
            price=100.0,
            volume_at_level=10.0,
            side=OrderSide.SELL,
            queue_model=QueueModel.PROBABILISTIC
        )
        
        assert tracker.order_id == "order_2"
        assert tracker.price == 100.0
        assert tracker.side == OrderSide.SELL
        # v_ahead должен быть между 5.0 и 10.0 (50-100% от 10.0)
        assert 5.0 <= tracker.v_ahead <= 10.0
        assert tracker.is_filled is False
    
    def test_partial_fill(self):
        """Тест 1: Частичное исполнение уровня (ордер не заполнен)"""
        tracker = QueueTracker(
            order_id="order_3",
            price=100.0,
            volume_at_level=10.0,
            side=OrderSide.BUY,
            queue_model=QueueModel.CONSERVATIVE
        )
        
        # Первая сделка: 3 контракта на цене 100.0
        result = tracker.update(trade_price=100.0, trade_volume=3.0)
        assert result is False  # Ордер не исполнен
        assert tracker.v_ahead == 7.0  # 10.0 - 3.0
        assert tracker.is_filled is False
        
        # Вторая сделка: 2 контракта на цене 100.0
        result = tracker.update(trade_price=100.0, trade_volume=2.0)
        assert result is False  # Ордер не исполнен
        assert tracker.v_ahead == 5.0  # 7.0 - 2.0
        assert tracker.is_filled is False
    
    def test_full_fill(self):
        """Тест 2: Полное проедание уровня (ордер заполнен)"""
        tracker = QueueTracker(
            order_id="order_4",
            price=100.0,
            volume_at_level=10.0,
            side=OrderSide.BUY,
            queue_model=QueueModel.CONSERVATIVE
        )
        
        # Первая сделка: 6 контрактов
        result = tracker.update(trade_price=100.0, trade_volume=6.0)
        assert result is False
        assert tracker.v_ahead == 4.0
        
        # Вторая сделка: 5 контрактов (больше чем осталось)
        result = tracker.update(trade_price=100.0, trade_volume=5.0)
        assert result is True  # Ордер исполнен
        assert tracker.v_ahead == -1.0  # 4.0 - 5.0
        assert tracker.is_filled is True
    
    def test_exact_fill(self):
        """Тест: Точное исполнение (v_ahead == 0)"""
        tracker = QueueTracker(
            order_id="order_5",
            price=100.0,
            volume_at_level=10.0,
            side=OrderSide.SELL,
            queue_model=QueueModel.CONSERVATIVE
        )
        
        # Сделка ровно на объем перед нами
        result = tracker.update(trade_price=100.0, trade_volume=10.0)
        assert result is True  # Ордер исполнен
        assert tracker.v_ahead == 0.0
        assert tracker.is_filled is True
    
    def test_different_price_no_update(self):
        """Тест: Сделка на другой цене не обновляет ордер"""
        tracker = QueueTracker(
            order_id="order_6",
            price=100.0,
            volume_at_level=10.0,
            side=OrderSide.BUY,
            queue_model=QueueModel.CONSERVATIVE
        )
        
        # Сделка на другой цене
        result = tracker.update(trade_price=101.0, trade_volume=10.0)
        assert result is False
        assert tracker.v_ahead == 10.0  # Не изменилось
        assert tracker.is_filled is False
    
    def test_cancel_order(self):
        """Тест 3: Отмена ордера"""
        tracker = QueueTracker(
            order_id="order_7",
            price=100.0,
            volume_at_level=10.0,
            side=OrderSide.BUY,
            queue_model=QueueModel.CONSERVATIVE
        )
        
        # Отменяем ордер
        tracker.cancel()
        assert tracker.is_filled is False
        assert tracker.v_ahead == 0.0
    
    def test_both_sides_bid_ask(self):
        """Тест 4: Проверка обеих сторон стакана (Bids/Asks)"""
        # BUY ордер (на Bid стороне)
        buy_tracker = QueueTracker(
            order_id="buy_order",
            price=99.5,
            volume_at_level=5.0,
            side=OrderSide.BUY,
            queue_model=QueueModel.CONSERVATIVE
        )
        
        # SELL ордер (на Ask стороне)
        sell_tracker = QueueTracker(
            order_id="sell_order",
            price=100.5,
            volume_at_level=8.0,
            side=OrderSide.SELL,
            queue_model=QueueModel.CONSERVATIVE
        )
        
        # Сделка на Bid цене
        result_buy = buy_tracker.update(trade_price=99.5, trade_volume=5.0)
        assert result_buy is True
        assert buy_tracker.is_filled is True
        
        # Сделка на Ask цене
        result_sell = sell_tracker.update(trade_price=100.5, trade_volume=8.0)
        assert result_sell is True
        assert sell_tracker.is_filled is True


class TestOrderQueueManager:
    """Тесты для класса OrderQueueManager"""
    
    def test_manager_initialization(self):
        """Тест инициализации менеджера"""
        manager = OrderQueueManager(queue_model=QueueModel.CONSERVATIVE)
        assert manager.get_active_orders_count() == 0
    
    def test_place_order(self):
        """Тест размещения ордера"""
        manager = OrderQueueManager(queue_model=QueueModel.CONSERVATIVE)
        
        tracker = manager.place_order(
            order_id="order_1",
            price=100.0,
            volume_at_level=10.0,
            side=OrderSide.BUY
        )
        
        assert manager.get_active_orders_count() == 1
        assert tracker.order_id == "order_1"
        assert tracker.v_ahead == 10.0
    
    def test_update_on_trade_single_order(self):
        """Тест обновления при сделке (один ордер)"""
        manager = OrderQueueManager(queue_model=QueueModel.CONSERVATIVE)
        
        manager.place_order(
            order_id="order_1",
            price=100.0,
            volume_at_level=10.0,
            side=OrderSide.BUY
        )
        
        # Сделка не исполняет ордер
        filled = manager.update_on_trade(trade_price=100.0, trade_volume=5.0)
        assert len(filled) == 0
        assert manager.get_active_orders_count() == 1
        
        # Сделка исполняет ордер
        filled = manager.update_on_trade(trade_price=100.0, trade_volume=6.0)
        assert len(filled) == 1
        assert filled[0] == "order_1"
        assert manager.get_active_orders_count() == 0
    
    def test_update_on_trade_multiple_orders(self):
        """Тест обновления при сделке (несколько ордеров)"""
        manager = OrderQueueManager(queue_model=QueueModel.CONSERVATIVE)
        
        # Размещаем несколько ордеров на одной цене
        manager.place_order("order_1", 100.0, 5.0, OrderSide.BUY)
        manager.place_order("order_2", 100.0, 8.0, OrderSide.BUY)
        manager.place_order("order_3", 101.0, 10.0, OrderSide.BUY)
        
        assert manager.get_active_orders_count() == 3
        
        # Сделка на цене 100.0 исполняет первый ордер
        filled = manager.update_on_trade(trade_price=100.0, trade_volume=5.0)
        assert len(filled) == 1
        assert "order_1" in filled
        assert manager.get_active_orders_count() == 2
        
        # Сделка на цене 100.0 исполняет второй ордер
        filled = manager.update_on_trade(trade_price=100.0, trade_volume=8.0)
        assert len(filled) == 1
        assert "order_2" in filled
        assert manager.get_active_orders_count() == 1
        
        # Сделка на цене 101.0 исполняет третий ордер
        filled = manager.update_on_trade(trade_price=101.0, trade_volume=10.0)
        assert len(filled) == 1
        assert "order_3" in filled
        assert manager.get_active_orders_count() == 0
    
    def test_cancel_order(self):
        """Тест отмены ордера"""
        manager = OrderQueueManager(queue_model=QueueModel.CONSERVATIVE)
        
        manager.place_order("order_1", 100.0, 10.0, OrderSide.BUY)
        assert manager.get_active_orders_count() == 1
        
        # Отменяем ордер
        result = manager.cancel_order("order_1")
        assert result is True
        assert manager.get_active_orders_count() == 0
        
        # Попытка отменить несуществующий ордер
        result = manager.cancel_order("order_999")
        assert result is False
    
    def test_get_order_status(self):
        """Тест получения статуса ордера"""
        manager = OrderQueueManager(queue_model=QueueModel.CONSERVATIVE)
        
        manager.place_order("order_1", 100.0, 10.0, OrderSide.BUY)
        
        status = manager.get_order_status("order_1")
        assert status is not None
        assert status['order_id'] == "order_1"
        assert status['price'] == 100.0
        assert status['side'] == "BUY"
        assert status['v_ahead'] == 10.0
        assert status['is_filled'] is False
        
        # Статус несуществующего ордера
        status = manager.get_order_status("order_999")
        assert status is None
    
    def test_get_all_orders_status(self):
        """Тест получения статуса всех ордеров"""
        manager = OrderQueueManager(queue_model=QueueModel.CONSERVATIVE)
        
        manager.place_order("order_1", 100.0, 10.0, OrderSide.BUY)
        manager.place_order("order_2", 101.0, 5.0, OrderSide.SELL)
        
        statuses = manager.get_all_orders_status()
        assert len(statuses) == 2
        assert statuses[0]['order_id'] == "order_1"
        assert statuses[1]['order_id'] == "order_2"


class TestQueueModels:
    """Тесты для различных моделей очереди"""
    
    def test_conservative_vs_probabilistic(self):
        """Тест сравнения консервативной и вероятностной моделей"""
        np.random.seed(42)
        
        # Консервативная модель
        conservative = QueueTracker(
            order_id="conservative",
            price=100.0,
            volume_at_level=100.0,
            side=OrderSide.BUY,
            queue_model=QueueModel.CONSERVATIVE
        )
        
        # Вероятностная модель
        probabilistic = QueueTracker(
            order_id="probabilistic",
            price=100.0,
            volume_at_level=100.0,
            side=OrderSide.BUY,
            queue_model=QueueModel.PROBABILISTIC
        )
        
        # Консервативная модель должна иметь весь объем перед нами
        assert conservative.v_ahead == 100.0
        
        # Вероятностная модель должна иметь меньше
        assert probabilistic.v_ahead < 100.0
        assert probabilistic.v_ahead >= 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
