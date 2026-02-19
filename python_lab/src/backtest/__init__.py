"""
Задача 212: Модуль бэктестирования с симуляцией очереди лимитных ордеров
Задача 215: Симуляция отказов и ошибок ордеров
"""

from .matching import QueueTracker, OrderQueueManager, OrderSide, QueueModel
from .engine import EventEngine, Event, EventType, BotConfig
from .error_sim import ExchangeErrorSimulator, ExchangeErrorType, ExchangeErrorData

__all__ = [
    'QueueTracker',
    'OrderQueueManager',
    'OrderSide',
    'QueueModel',
    'EventEngine',
    'Event',
    'EventType',
    'BotConfig',
    'ExchangeErrorSimulator',
    'ExchangeErrorType',
    'ExchangeErrorData',
]
