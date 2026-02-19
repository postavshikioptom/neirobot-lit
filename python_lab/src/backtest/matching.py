"""
Задача 212: Симуляция очереди лимитных ордеров (Limit Order Queue Simulation)

Реализует логику FIFO очереди для лимитных ордеров в бэктестере.
Критично для устранения Selection Bias: ордер исполняется только после того,
как через уровень пройдет достаточный объем сделок других участников.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from enum import Enum
from numba import njit


class OrderSide(Enum):
    """Сторона ордера"""
    BUY = "BUY"
    SELL = "SELL"


class QueueModel(Enum):
    """Модель очереди"""
    CONSERVATIVE = "conservative"  # Весь объем уровня перед нами
    PROBABILISTIC = "probabilistic"  # Случайная позиция 50-100%


@njit
def update_v_ahead_numba(v_ahead: float, trade_price: float, order_price: float, 
                         trade_volume: float, price_tolerance: float = 1e-8) -> Tuple[float, bool]:
    """
    Numba-оптимизированная функция для обновления v_ahead при получении сделки.
    
    Вызывается миллионы раз в цикле бэктеста, поэтому критична производительность.
    
    Args:
        v_ahead: Текущий объем перед нами
        trade_price: Цена сделки
        order_price: Цена нашего ордера
        trade_volume: Объем сделки
        price_tolerance: Допуск для сравнения цен (float epsilon)
    
    Returns:
        Кортеж (новый v_ahead, is_filled)
    """
    # Проверяем, что сделка на нашем уровне цены
    if abs(trade_price - order_price) < price_tolerance:
        # Уменьшаем объем перед нами
        new_v_ahead = v_ahead - trade_volume
        
        # Проверяем, исполнен ли ордер
        if new_v_ahead <= 0:
            return 0.0, True
        else:
            return new_v_ahead, False
    
    # Сделка на другой цене - не обновляем
    return v_ahead, False


class QueueTracker:
    """
    Отслеживает позицию лимитного ордера в очереди стакана.
    
    Использует модель FIFO для реалистичной симуляции исполнения.
    Поддерживает две модели:
    - conservative: весь объем уровня считается перед нами
    - probabilistic: случайная позиция в диапазоне 50-100% объема
    """
    
    def __init__(
        self,
        order_id: str,
        price: float,
        volume_at_level: float,
        side: OrderSide,
        queue_model: QueueModel = QueueModel.CONSERVATIVE
    ):
        """
        Инициализация трекера очереди.
        
        Args:
            order_id: Уникальный идентификатор ордера
            price: Цена ордера
            volume_at_level: Объем на уровне цены в момент постановки
            side: Сторона ордера (BUY/SELL)
            queue_model: Модель очереди (conservative/probabilistic)
        """
        self.order_id = order_id
        self.price = price
        self.side = side
        self.queue_model = queue_model
        self.is_filled = False
        
        # Задача 212: Установка v_ahead на основе queue_model
        if queue_model == QueueModel.CONSERVATIVE:
            # Консервативная модель: весь объем уровня перед нами
            self.v_ahead = volume_at_level
        elif queue_model == QueueModel.PROBABILISTIC:
            # Вероятностная модель: случайная позиция 50-100%
            random_position = np.random.uniform(0.5, 1.0)
            self.v_ahead = volume_at_level * random_position
        else:
            raise ValueError(f"Unknown queue model: {queue_model}")
    
    def update(self, trade_price: float, trade_volume: float) -> bool:
        """
        Обновить состояние ордера при получении сделки.
        
        Использует Numba-оптимизированную функцию для максимальной производительности.
        
        Args:
            trade_price: Цена сделки
            trade_volume: Объем сделки
            
        Returns:
            True если ордер был исполнен, False иначе
        """
        # Используем Numba-функцию для быстрого обновления
        self.v_ahead, is_filled = update_v_ahead_numba(
            self.v_ahead, trade_price, self.price, trade_volume
        )
        
        if is_filled:
            self.is_filled = True
        
        return is_filled
    
    def cancel(self) -> None:
        """Отменить ордер"""
        self.is_filled = False
        self.v_ahead = 0.0
    
    def get_status(self) -> Dict:
        """Получить статус ордера"""
        return {
            'order_id': self.order_id,
            'price': self.price,
            'side': self.side.value,
            'v_ahead': self.v_ahead,
            'is_filled': self.is_filled,
            'queue_model': self.queue_model.value
        }


class OrderQueueManager:
    """
    Менеджер для управления очередями лимитных ордеров.
    
    Отслеживает все активные ордера и обновляет их состояние
    при получении событий сделок.
    """
    
    def __init__(self, queue_model: QueueModel = QueueModel.CONSERVATIVE):
        """
        Инициализация менеджера очереди.
        
        Args:
            queue_model: Модель очереди (conservative/probabilistic)
        """
        self.queue_model = queue_model
        self.active_orders: Dict[str, QueueTracker] = {}
    
    def place_order(
        self,
        order_id: str,
        price: float,
        volume_at_level: float,
        side: OrderSide
    ) -> QueueTracker:
        """
        Разместить новый ордер в очереди.
        
        Args:
            order_id: Уникальный идентификатор ордера
            price: Цена ордера
            volume_at_level: Объем на уровне цены
            side: Сторона ордера (BUY/SELL)
            
        Returns:
            QueueTracker для отслеживания ордера
        """
        tracker = QueueTracker(
            order_id=order_id,
            price=price,
            volume_at_level=volume_at_level,
            side=side,
            queue_model=self.queue_model
        )
        self.active_orders[order_id] = tracker
        return tracker
    
    def update_on_trade(self, trade_price: float, trade_volume: float) -> list:
        """
        Обновить все активные ордера при получении сделки.
        
        Args:
            trade_price: Цена сделки
            trade_volume: Объем сделки
            
        Returns:
            Список ID ордеров, которые были исполнены
        """
        filled_orders = []
        
        for order_id, tracker in list(self.active_orders.items()):
            if tracker.update(trade_price, trade_volume):
                filled_orders.append(order_id)
        
        # Удаляем исполненные ордера
        for order_id in filled_orders:
            del self.active_orders[order_id]
        
        return filled_orders
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Отменить ордер.
        
        Args:
            order_id: ID ордера для отмены
            
        Returns:
            True если ордер был отменен, False если не найден
        """
        if order_id in self.active_orders:
            self.active_orders[order_id].cancel()
            del self.active_orders[order_id]
            return True
        return False
    
    def get_active_orders_count(self) -> int:
        """Получить количество активных ордеров"""
        return len(self.active_orders)
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Получить статус ордера"""
        if order_id in self.active_orders:
            return self.active_orders[order_id].get_status()
        return None
    
    def get_all_orders_status(self) -> list:
        """Получить статус всех активных ордеров"""
        return [tracker.get_status() for tracker in self.active_orders.values()]
    
    def check_price_crossing(self, best_bid: float, best_ask: float) -> list:
        """
        Задача 212: Проверить, какие ордера исполнены при пересечении цены.
        
        Если цена рынка ушла далеко за наш лимит, ордер должен исполниться мгновенно.
        Например, если мы выставили BUY @ 100, а Best Bid стал 99, мы "глубоко в деньгах".
        
        Args:
            best_bid: Лучшая цена покупки на рынке
            best_ask: Лучшая цена продажи на рынке
            
        Returns:
            Список ID ордеров, которые были исполнены при пересечении цены
        """
        filled_orders = []
        
        for order_id, tracker in list(self.active_orders.items()):
            should_fill = False
            
            # Для BUY ордера: если Best Bid < наша цена, мы исполнены
            if tracker.side == OrderSide.BUY and best_bid < tracker.price:
                should_fill = True
            # Для SELL ордера: если Best Ask > наша цена, мы исполнены
            elif tracker.side == OrderSide.SELL and best_ask > tracker.price:
                should_fill = True
            
            if should_fill:
                tracker.is_filled = True
                filled_orders.append(order_id)
                del self.active_orders[order_id]
        
        return filled_orders
