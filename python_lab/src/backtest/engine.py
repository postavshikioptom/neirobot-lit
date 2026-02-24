import heapq
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from numba import jit
from python_lab.src.backtest.matching import OrderQueueManager, QueueModel, OrderSide
from python_lab.src.backtest.error_sim import ExchangeErrorSimulator, ExchangeErrorType, ExchangeErrorData

class EventType(Enum):
    MARKET = auto()
    SIGNAL = auto()
    ORDER = auto()
    FILL = auto()
    CANCEL = auto()
    TRADE = auto()  # Задача 212: События о сделках для обновления очереди
    EXCHANGE_ERROR = auto()  # Задача 215: События об ошибках биржи

@dataclass(order=True)
class Event:
    timestamp: int
    type: EventType = field(compare=False)
    data: Any = field(compare=False)
    symbol: str = field(default="", compare=False)  # Задача 213: Поддержка мульти-инструментальности

@dataclass
class MarketData:
    mid_price: float
    bids: np.ndarray  # (50, 2) [price, volume]
    asks: np.ndarray  # (50, 2) [price, volume]
    
@dataclass
class SignalData:
    probs: np.ndarray  # [up, down, flat]
    side: str  # 'buy', 'sell', 'flat'
    confidence: float

@dataclass
class OrderData:
    order_id: str
    side: str
    price: float
    amount: float
    order_type: str  # 'limit', 'market'
    post_only: bool = False
    created_at_ms: int = 0  # Время создания ордера (для проверки таймаутов)
    chase_count: int = 0  # Задача 211: Счетчик попыток перестановки (Re-pegging)
    is_iceberg: bool = False  # Задача 211: Флаг айсберг-ордера
    iceberg_parent_id: str = ""  # ID родительского ордера для айсберга

@dataclass
class FillData:
    order_id: str
    side: str
    price: float
    amount: float
    fee_usd: float
    fill_type: str  # 'maker', 'taker'
    is_iceberg: bool = False  # Задача 211
    iceberg_parent_id: str = ""  # Задача 211

@dataclass
class TradeData:
    """Задача 212: Данные о сделке для обновления очереди лимитных ордеров"""
    trade_price: float
    trade_volume: float
    timestamp_ms: int

@jit(nopython=True)
def match_limit_order(side: str, limit_price: float, amount: float, 
                      bids: np.ndarray, asks: np.ndarray) -> Tuple[bool, float]:
    """
    Упрощенный матчинг лимитного ордера.
    Для BUY: исполняется, если лучшая цена продажи (ask[0]) <= лимитной цене.
    Для SELL: исполняется, если лучшая цена покупки (bid[0]) >= лимитной цене.
    """
    if side == "buy":
        if asks[0, 0] <= limit_price:
            return True, asks[0, 0]
    else:
        if bids[0, 0] >= limit_price:
            return True, bids[0, 0]
    return False, 0.0

@jit(nopython=True)
def calculate_pnl_numba(entry_price: float, exit_price: float, amount: float, side: str) -> float:
    if side == "buy":
        return (exit_price - entry_price) * amount
    else:
        return (entry_price - exit_price) * amount


@dataclass
class SorConfig:
    """Smart Order Routing configuration"""
    critical_signal: float = 0.75
    max_size_ratio: float = 0.3
    default_urgency: float = 0.5
    slice_interval_ms: int = 100
    iceberg_enabled: bool = False  # Задача 211: Включение режима айсберг-ордеров
    iceberg_randomize: float = 0.2
    iceberg_price_dev_bps: int = 10
    switch_base_timeout_ms: int = 500
    switch_base_distance_bps: int = 5
    max_switches_per_signal: int = 1

@dataclass
class BotConfig:
    symbol: str = "UNKNOWN"
    initial_balance: float = 1000.0
    taker_fee_bps: float = 6.0
    maker_fee_bps: float = 2.0
    limit_timeout_ms: int = 10000
    chase_mode: str = "ToBest" # "ToBest", "InsideSpread", "ToVWAP", "None"
    chase_threshold_bps: float = 2.0
    chase_distance_bps: float = 0.5
    chase_max_attempts: int = 3
    chase_interval_ms: int = 1000
    sor: SorConfig = field(default_factory=SorConfig)
    order_size_usd: float = 1000.0
    
    # Задача 212: Параметры очереди лимитных ордеров
    queue_model: str = "conservative"  # "conservative" или "probabilistic"
    
    # Задача 213: Параметры риск-менеджмента (индивидуально для каждого символа)
    max_position: float = 10.0  # Максимальный размер позиции в базовой валюте
    max_drawdown_pct: float = 10.0  # Максимальная просадка в процентах



@jit(nopython=True)
def calculate_book_price_numba(side: str, amount_usd: float, 
                              bids: np.ndarray, asks: np.ndarray) -> float:
    """
    Расчет средневзвешенной цены исполнения с учетом Book Walking.
    """
    remaining_usd = amount_usd
    total_volume_base = 0.0
    weighted_price_sum = 0.0
    
    levels = asks if side == "buy" else bids
    
    for i in range(len(levels)):
        p = levels[i, 0]
        v = levels[i, 1]
        
        level_capacity_usd = p * v
        
        if remaining_usd <= level_capacity_usd:
            fill_volume_base = remaining_usd / p
            weighted_price_sum += fill_volume_base * p
            total_volume_base += fill_volume_base
            remaining_usd = 0
            break
        else:
            weighted_price_sum += v * p
            total_volume_base += v
            remaining_usd -= level_capacity_usd
            
    if remaining_usd > 0:
        last_price = levels[-1, 0]
        fill_volume_base = remaining_usd / last_price
        weighted_price_sum += fill_volume_base * last_price * 1.01 # +1% штраф
        total_volume_base += fill_volume_base
        
    return weighted_price_sum / total_volume_base

# ============================================================================
# Задача 213: Поддержка мульти-инструментальности
# ============================================================================

@dataclass
class SymbolState:
    """
    Состояние для одного торгового символа.
    Обеспечивает полную изоляцию логики и настроек для каждого символа.
    """
    config: BotConfig
    position: float = 0.0
    balance: float = 0.0
    orders: Dict[str, OrderData] = field(default_factory=dict)
    trades: List[TradeData] = field(default_factory=list)
    entry_prices: Dict[str, float] = field(default_factory=dict)
    gross_pnl: float = 0.0
    signal_mids: Dict[str, float] = field(default_factory=dict)
    slippages: List[float] = field(default_factory=list)
    queue_manager: Optional[OrderQueueManager] = None
    last_market_data: Optional[MarketData] = None
    total_orders_placed: int = 0
    total_orders_cancelled: int = 0
    
    # Задача 211: Состояние для айсберг-ордеров
    iceberg_remaining: Dict[str, float] = field(default_factory=dict)  # parent_id -> remaining_amount
    iceberg_sides: Dict[str, str] = field(default_factory=dict) # parent_id -> side
    
    # Задача 213: Параметры риск-менеджмента
    peak_balance: float = 0.0  # Пиковый баланс для расчета просадки
    
    # Задача 215: Метрики ошибок
    lost_trades_count: int = 0  # Количество пропущенных сигналов из-за ошибок
    
    # Задача 058: Метрики Limit-then-Market
    timeout_orders: int = 0  # Счётчик ордеров, конвертированных в Market по таймауту
    
    def __post_init__(self):
        """Инициализация состояния после создания"""
        self.balance = self.config.initial_balance
        self.peak_balance = self.config.initial_balance
        queue_model = QueueModel.CONSERVATIVE if self.config.queue_model == "conservative" else QueueModel.PROBABILISTIC
        self.queue_manager = OrderQueueManager(queue_model=queue_model)
    
    def get_current_drawdown_pct(self) -> float:
        """Расчет текущей просадки в процентах"""
        if self.peak_balance <= 0:
            return 0.0
        drawdown = (self.peak_balance - self.balance) / self.peak_balance * 100.0
        return max(0.0, drawdown)
    
    def update_peak_balance(self):
        """Обновление пикового баланса"""
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
    
    def is_position_limit_exceeded(self, additional_size: float) -> bool:
        """Проверка превышения лимита позиции"""
        return abs(self.position + additional_size) > self.config.max_position
    
    def is_drawdown_limit_exceeded(self) -> bool:
        """Проверка превышения лимита просадки"""
        return self.get_current_drawdown_pct() > self.config.max_drawdown_pct


class EventEngine:
    def __init__(self, config: BotConfig, error_simulator: Optional[ExchangeErrorSimulator] = None):
        """
        Инициализация движка событий.
        
        Поддерживает как одиночный символ (для обратной совместимости),
        так и мульти-инструментальность (Задача 213).
        
        Args:
            config: Конфигурация бота
            error_simulator: Симулятор ошибок биржи (Задача 215, опционально)
        """
        self.config = config
        self.events: List[Tuple[int, Event]] = []
        self.current_time = 0
        
        # Параметры задержек (ms)
        self.internal_latency = 1  
        self.network_latency = 20
        
        # Режим исполнения
        self.execution_mode = "realistic"
        
        # Задача 215: Симулятор ошибок биржи
        self.error_simulator = error_simulator
        
        # Задача 213: Поддержка мульти-инструментальности
        # Хранение состояний для каждого символа
        self.states: Dict[str, SymbolState] = {}
        
        # Инициализация состояния для основного символа
        self._init_symbol_state(config.symbol, config)
        
        # Для обратной совместимости - прямой доступ к состоянию основного символа
        self._primary_symbol = config.symbol
    
    def _init_symbol_state(self, symbol: str, config: BotConfig):
        """Инициализация состояния для символа"""
        if symbol not in self.states:
            self.states[symbol] = SymbolState(config=config)
    
    def add_symbol(self, symbol: str, config: BotConfig):
        """Добавление нового символа для обработки"""
        self._init_symbol_state(symbol, config)
    
    def get_state(self, symbol: str = "") -> SymbolState:
        """Получение состояния для символа (по умолчанию - основной символ)"""
        if not symbol:
            symbol = self._primary_symbol
        if symbol not in self.states:
            raise ValueError(f"Символ {symbol} не инициализирован")
        return self.states[symbol]



    def set_mode(self, mode: str):
        if mode not in ["realistic", "ideal"]:
            raise ValueError(f"Unknown mode: {mode}")
        self.execution_mode = mode

    def push_event(self, event: Event):
        heapq.heappush(self.events, (event.timestamp, event))

    def run(self):
        while self.events:
            ts, event = heapq.heappop(self.events)
            self.current_time = ts
            self.process_event(event)

    def process_event(self, event: Event):
        """Обработка события с поддержкой мульти-инструментальности"""
        # Задача 213: Используем символ из события
        symbol = event.symbol if event.symbol else self._primary_symbol
        
        if event.type == EventType.MARKET:
            self._on_market(event.data, symbol)
        elif event.type == EventType.SIGNAL:
            self._on_signal(event.data, symbol)
        elif event.type == EventType.ORDER:
            self._on_order(event.data, symbol)
        elif event.type == EventType.FILL:
            self._on_fill(event.data, symbol)
        elif event.type == EventType.CANCEL:
            self._on_cancel(event.data, symbol)
        elif event.type == EventType.TRADE:
            self._on_trade(event.data, symbol)
        elif event.type == EventType.EXCHANGE_ERROR:
            self._on_exchange_error(event.data, symbol)

    def _on_market(self, data: MarketData, symbol: str = ""):
        """Обработка рыночных данных с поддержкой мульти-инструментальности"""
        state = self.get_state(symbol)
        state.last_market_data = data
        
        # Задача 058: Проверка таймаутов лимитных ордеров (Limit-then-Market)
        self._check_limit_timeouts(symbol)
        
        # 1. Матчинг существующих лимитных ордеров
        if self.execution_mode == "ideal":
            # Идеальное исполнение: если mid_price коснулся или пересек цену
            to_fill = []
            for oid, order in state.orders.items():
                 if order.order_type == "limit":
                    mid = data.mid_price
                    if (order.side == "buy" and mid <= order.price) or \
                       (order.side == "sell" and mid >= order.price):
                        to_fill.append((oid, order.price))
                        
            for oid, price in to_fill:
                 self._fill_order(oid, price, "maker", symbol)
                 
        elif self.execution_mode == "realistic":
            to_fill = []
            
            # Задача 212: Проверяем пересечение цены (Price Crossing)
            # Если цена рынка ушла далеко за наш лимит, ордер исполняется мгновенно
            crossed_orders = state.queue_manager.check_price_crossing(
                best_bid=data.bids[0, 0],
                best_ask=data.asks[0, 0]
            )
            for oid in crossed_orders:
                if oid in state.orders:
                    order = state.orders[oid]
                    # Исполняем по цене лимита (мы выставили лимит, рынок прошел сквозь него)
                    to_fill.append((oid, order.price))
            
            # Задача 212: В режиме realistic исполнение лимитных ордеров происходит
            # ТОЛЬКО через _on_trade -> queue_manager.update_on_trade.
            # Здесь обрабатываем только Chasing логику для ордеров, которые
            # не попали в crossed_orders (не были "перепрыгнуты" рынком).
            for oid, order in list(state.orders.items()):
                if order.order_type == "limit" and oid not in crossed_orders:
                    # Логика Chasing (Re-pegging) - НЕ исполняем ордер здесь через match_limit_order
                    if state.config.chase_mode != "None":
                        # Если цена ушла далеко от нашего лимита
                        best_price = data.bids[0, 0] if order.side == "buy" else data.asks[0, 0]
                        dist_bps = 0.0
                        if order.price > 0:
                            dist_bps = abs(best_price - order.price) / order.price * 10000
                        
                        if dist_bps > state.config.chase_threshold_bps:
                            # Задача 211: Полноценный Re-pegging
                            # Проверяем количество попыток (chase_max_attempts)
                            if order.chase_count < state.config.chase_max_attempts:
                                # 1. Генерируем Cancel событие
                                cancel_event = Event(
                                    timestamp=self.current_time + self.internal_latency,
                                    type=EventType.CANCEL,
                                    data=order.order_id,
                                    symbol=symbol
                                )
                                self.push_event(cancel_event)
                                
                                # 2. Генерируем новый Order событие с актуальной ценой и учетом chase_distance_bps
                                offset_price = best_price * (state.config.chase_distance_bps / 10000)
                                if order.side == "buy":
                                    new_price = best_price + offset_price
                                else:
                                    new_price = best_price - offset_price
                                    
                                repeg_order = OrderData(
                                    order_id=f"{order.order_id}_ch{order.chase_count+1}",
                                    side=order.side,
                                    price=new_price,
                                    amount=order.amount,
                                    order_type="limit",
                                    created_at_ms=self.current_time + self.internal_latency + self.network_latency,
                                    chase_count=order.chase_count + 1,
                                    is_iceberg=order.is_iceberg,
                                    iceberg_parent_id=order.iceberg_parent_id
                                )
                                
                                repeg_event = Event(
                                    timestamp=self.current_time + self.internal_latency + self.network_latency,
                                    type=EventType.ORDER,
                                    data=repeg_order,
                                    symbol=symbol
                                )
                                state.total_orders_placed += 1
                                self.push_event(repeg_event)
                            else:
                                # Превышено макс. кол-во попыток - просто отменяем без перевыставления
                                cancel_event = Event(
                                    timestamp=self.current_time + self.internal_latency,
                                    type=EventType.CANCEL,
                                    data=order.order_id,
                                    symbol=symbol
                                )
                                self.push_event(cancel_event)
            
            for oid, price in to_fill:
                self._fill_order(oid, price, "maker", symbol)

    def _fill_order(self, oid: str, price: float, fill_type: str, symbol: str = ""):
        """Исполнение ордера с поддержкой мульти-инструментальности"""
        state = self.get_state(symbol)
        if oid not in state.orders: return
        order = state.orders.pop(oid)
            
        # Расчет проскальзывания в bps от мида в момент сигнала
        if oid in state.signal_mids:
            mid_at_signal = state.signal_mids.pop(oid)
            # В HFT проскальзывание обычно считают как разницу между мидом в момент засылки сигнала и ценой исполнения.
            # Если купили дороже мида - это положительное проскальзывание (slippage).
            # Для BUY: (fill_price - mid) / mid * 10000
            # Для SELL: (mid - fill_price) / mid * 10000
            slippage_bps = (price - mid_at_signal) / mid_at_signal * 10000 if order.side == "buy" else \
                           (mid_at_signal - price) / mid_at_signal * 10000
            state.slippages.append(slippage_bps)

        fill_event = Event(
            timestamp=self.current_time,
            type=EventType.FILL,
            data=FillData(
                order_id=oid,
                side=order.side,
                price=price,
                amount=order.amount,
                fee_usd=0.0,
                fill_type=fill_type,
                is_iceberg=order.is_iceberg,
                iceberg_parent_id=order.iceberg_parent_id
            ),
            symbol=symbol
        )
        self.push_event(fill_event)

    def _check_limit_timeouts(self, symbol: str = ""):
        """
        Задача 058: Проверка таймаутов лимитных ордеров.
        Если лимитный ордер не исполнен за отведённое время, конвертируем его в маркет.
        """
        state = self.get_state(symbol)
        
        # Проверяем каждый лимитный ордер на таймаут
        orders_to_convert = []
        for oid, order in list(state.orders.items()):
            if order.order_type == "limit":
                # Проверяем, превышен ли таймаут
                time_elapsed = self.current_time - order.created_at_ms
                if time_elapsed >= state.config.limit_timeout_ms:
                    orders_to_convert.append((oid, order))
        
        # Конвертируем таймаутные ордера в маркет
        for oid, order in orders_to_convert:
            self._convert_to_market(oid, order, symbol)

    def _convert_to_market(self, oid: str, order: OrderData, symbol: str = ""):
        """
        Задача 058: Конвертация лимитного ордера в маркет при таймауте.
        Отменяем лимит и создаём маркет ордер.
        """
        state = self.get_state(symbol)
        
        # Удаляем лимитный ордер
        if oid in state.orders:
            del state.orders[oid]
        
        # Увеличиваем счётчик таймаутных ордеров
        state.timeout_orders += 1
        
        # Создаём маркет ордер с тем же ID (или новым)
        market_order_id = f"{oid}_market"
        
        # Сохраняем мид для расчета проскальзывания маркет ордера
        if oid in state.signal_mids:
            state.signal_mids[market_order_id] = state.signal_mids[oid]
            del state.signal_mids[oid]
        elif state.last_market_data:
            state.signal_mids[market_order_id] = state.last_market_data.mid_price
        
        # Генерируем ORDER событие для маркет ордера
        market_event = Event(
            timestamp=self.current_time,
            type=EventType.ORDER,
            data=OrderData(
                order_id=market_order_id,
                side=order.side,
                price=0.0,  # Маркет ордер не имеет цены
                amount=order.amount,
                order_type="market",
                created_at_ms=self.current_time
            ),
            symbol=symbol
        )
        self.push_event(market_event)

    def _on_signal(self, data: SignalData, symbol: str = ""):
        """Обработка сигнала с поддержкой мульти-инструментальности"""
        state = self.get_state(symbol)
        
        # Задача 213: Проверка лимитов риск-менеджмента
        if state.is_drawdown_limit_exceeded():
            print(f"[{symbol}] Trading halted: Drawdown limit exceeded ({state.get_current_drawdown_pct():.2f}%)")
            return
        
        # Добавляем задержку (имитация принятия решения и отправки по сети)
        latency = 0 if self.execution_mode == "ideal" else (self.internal_latency + self.network_latency)
        execution_ts = self.current_time + latency
        
        # Генерируем ORDER событие
        if data.side != 'flat':
            # Логика в зависимости от SorConfig/BotConfig
            # Если сигнал сильный (critical_signal), бьем маркетом
            is_aggressive = data.confidence >= state.config.sor.critical_signal
            
            order_type = "market" if is_aggressive else "limit"
            
            # Расчет цены для лимитки (напр. Best Bid/Ask или с отступом)
            price = 0.0
            amount = 0.0
            if state.last_market_data:
                # Количество базовой валюты исходя из order_size_usd
                amount = state.config.order_size_usd / state.last_market_data.mid_price
                
                if order_type == "limit":
                    # ChaseBest: ставим на Best Bid/Ask
                    if data.side == "buy":
                        price = state.last_market_data.bids[0, 0]
                    else:
                        price = state.last_market_data.asks[0, 0]
                    
                    # Задача 211: Учет Urgency (Срочность)
                    # Если urgency > 0, смещаем лимитную цену ближе к противоположной стороне
                    if state.config.sor.default_urgency > 0:
                        spread = state.last_market_data.asks[0, 0] - state.last_market_data.bids[0, 0]
                        # Смещение: 0.0 -> на лучшей цене, 1.0 -> на цене агрессора (мгновенный fill)
                        urgency_offset = spread * state.config.sor.default_urgency
                        if data.side == "buy":
                            price += urgency_offset
                        else:
                            price -= urgency_offset
            
            # Задача 213: Проверка лимита позиции перед созданием ордера
            side_mult = 1 if data.side == "buy" else -1
            if state.is_position_limit_exceeded(side_mult * amount):
                print(f"[{symbol}] Order rejected: Position limit would be exceeded")
                return
            
            total_amount = amount
            
            # Задача 211: Поддержка Iceberg (Айсберг-ордера)
            if state.config.sor.iceberg_enabled and order_type == "limit":
                parent_id = f"ice_{execution_ts}_{symbol}"
                state.iceberg_remaining[parent_id] = total_amount
                state.iceberg_sides[parent_id] = data.side
                
                # Выставляем первый слайс айсберга
                self._trigger_next_iceberg_slice(parent_id, execution_ts, symbol)
                return
            
            # Simple SOR: Slicing (Параллельное выставление ордеров)
            # Вычисляем максимальный размер слайса (напр. 30% от общей суммы или фиксированный лимит)
            max_slice = total_amount * state.config.sor.max_size_ratio
            if max_slice <= 0: max_slice = total_amount
            
            remaining = total_amount
            slice_idx = 0
            while remaining > 0:
                current_slice = min(remaining, max_slice)
                remaining -= current_slice
                
                slice_ts = execution_ts + (slice_idx * state.config.sor.slice_interval_ms)
                order_id = f"ord_{slice_ts}_{slice_idx}"
                
                # Сохраняем мид для расчета проскальзывания (для всего ордера по частям)
                if state.last_market_data:
                    state.signal_mids[order_id] = state.last_market_data.mid_price
                
                order_event = Event(
                    timestamp=slice_ts,
                    type=EventType.ORDER,
                    data=OrderData(
                        order_id=order_id,
                        side=data.side,
                        price=price,
                        amount=current_slice,
                        order_type=order_type,
                        created_at_ms=slice_ts  # Задача 058: Сохраняем время создания для проверки таймаутов
                    ),
                    symbol=symbol
                )
                state.total_orders_placed += 1
                self.push_event(order_event)
                slice_idx += 1


    def _on_order(self, data: OrderData, symbol: str = ""):
        """Обработка ордера с поддержкой мульти-инструментальности и симуляцией ошибок"""
        state = self.get_state(symbol)
        
        # Задача 215: Проверка на ошибки биржи
        if self.error_simulator:
            error_type, error_data = self.error_simulator.should_fail(
                current_time_ms=self.current_time,
                order_id=data.order_id
            )
            
            if error_type != ExchangeErrorType.NONE:
                # Генерируем событие об ошибке
                error_event = Event(
                    timestamp=self.current_time,
                    type=EventType.EXCHANGE_ERROR,
                    data=error_data,
                    symbol=symbol
                )
                self.push_event(error_event)
                
                # Для RateLimitError: повторяем операцию после backoff
                if error_type == ExchangeErrorType.RATE_LIMIT:
                    retry_time = self.error_simulator.backoff_until_ms
                    retry_event = Event(
                        timestamp=retry_time,
                        type=EventType.ORDER,
                        data=data,
                        symbol=symbol
                    )
                    self.push_event(retry_event)
                else:
                    # Для других ошибок: считаем сигнал потерянным
                    state.lost_trades_count += 1
                
                return  # Не обрабатываем ордер
        
        if data.order_type == "market":
            # Исполняем немедленно по текущему стакану (с учетом проскальзывания)
            if state.last_market_data:
                # В HFT маркет ордер все равно имеет задержку исполнения на бирже
                # Но мы уже учли network_latency в _on_signal -> ORDER.
                # Поэтому считаем исполнение по стакану в момент прихода ордера на биржу.
                
                amount_usd = data.amount * state.last_market_data.mid_price # Условно
                fill_price = calculate_book_price_numba(
                    data.side, amount_usd, state.last_market_data.bids, state.last_market_data.asks
                )
                
                fill_event = Event(
                    timestamp=self.current_time,
                    type=EventType.FILL,
                    data=FillData(
                        order_id=data.order_id,
                        side=data.side,
                        price=fill_price,
                        amount=data.amount,
                        fee_usd=0.0,
                        fill_type="taker"
                    )
                )
                self.push_event(fill_event)
        else:
            # Лимитный ордер - сохраняем и ждем матчинга в _on_market или _on_trade
            state.orders[data.order_id] = data
            
            # Задача 212: Размещаем ордер в очереди
            if state.last_market_data:
                levels = state.last_market_data.bids if data.side == "buy" else state.last_market_data.asks
                volume_at_level = 0.0
                for p, v in levels:
                    if abs(p - data.price) < 1e-8:  # Точное совпадение цены
                        volume_at_level = v
                        break
                
                # Размещаем ордер в очереди
                side = OrderSide.BUY if data.side == "buy" else OrderSide.SELL
                state.queue_manager.place_order(
                    order_id=data.order_id,
                    price=data.price,
                    volume_at_level=volume_at_level,
                    side=side
                )

    def _on_fill(self, data: FillData, symbol: str = ""):
        """Обработка исполнения ордера с поддержкой мульти-инструментальности"""
        state = self.get_state(symbol)

        # Обновление позиции и баланса
        side_mult = 1 if data.side == "buy" else -1

        # Если это закрытие позиции (полное или частичное), считаем PnL
        if state.position != 0 and np.sign(state.position) != side_mult:
            # Упрощенно: закрываем позицию (flip или reduce)
            # В данном бэктестере у нас 1 лот за раз, так что это просто закрытие
            original_side = "buy" if side_mult < 0 else "sell"
            entry_price = state.entry_prices.get(original_side, data.price)

            pnl = calculate_pnl_numba(entry_price, data.price, data.amount, original_side)
            state.gross_pnl += pnl
            state.balance += pnl
        else:
            # Открытие или увеличение
            state.entry_prices[data.side] = data.price

        # Обновление позиции
        state.position += side_mult * data.amount

        # Учет комиссий
        fee_rate = (state.config.maker_fee_bps if data.fill_type == "maker" else state.config.taker_fee_bps) / 10000
        data.fee_usd = data.amount * data.price * fee_rate
        state.balance -= data.fee_usd

        # Задача 211: Если это айсберг-ордер, выставляем следующий слайс
        if data.is_iceberg and data.iceberg_parent_id:
            # Небольшая задержка перед следующим слайсом
            next_slice_ts = self.current_time + self.internal_latency + self.network_latency
            self._trigger_next_iceberg_slice(data.iceberg_parent_id, next_slice_ts, symbol)

        state.trades.append(data)

    def _trigger_next_iceberg_slice(self, parent_id: str, timestamp: int, symbol: str = ""):
        """
        Задача 211: Выставление следующего видимого слайса айсберг-ордера.
        Использует параметры iceberg_randomize и iceberg_price_dev_bps.
        """
        state = self.get_state(symbol)
        if parent_id not in state.iceberg_remaining or state.iceberg_remaining[parent_id] <= 0:
            # Айсберг полностью исполнен
            if parent_id in state.iceberg_remaining:
                del state.iceberg_remaining[parent_id]
                del state.iceberg_sides[parent_id]
            return

        remaining = state.iceberg_remaining[parent_id]
        side = state.iceberg_sides[parent_id]
        
        # 1. Расчет размера слайса с рандомизацией (iceberg_randomize)
        # Базовый размер слайса от общей суммы сигнала
        base_slice_usd = state.config.order_size_usd * state.config.sor.max_size_ratio
        # Рандомизация: +/- iceberg_randomize %
        rand_factor = 1.0 + (np.random.random() * 2 - 1) * state.config.sor.iceberg_randomize
        slice_amount_usd = min(remaining * state.last_market_data.mid_price, base_slice_usd * rand_factor)
        current_slice_amount = slice_amount_usd / state.last_market_data.mid_price

        # 2. Расчет цены с рандомизацией (iceberg_price_dev_bps)
        # Базовая цена (Best Bid/Ask)
        best_price = state.last_market_data.bids[0, 0] if side == "buy" else state.last_market_data.asks[0, 0]
        # Рандомное отклонение в bps
        price_dev_bps = (np.random.random() * 2 - 1) * state.config.sor.iceberg_price_dev_bps
        price_offset = best_price * (price_dev_bps / 10000)
        ice_price = best_price + price_offset

        # 3. Выставление ордера
        order_id = f"ice_sl_{parent_id}_{int(timestamp)}"
        
        # Сохраняем мид для расчета проскальзывания
        state.signal_mids[order_id] = state.last_market_data.mid_price
        
        order_data = OrderData(
            order_id=order_id,
            side=side,
            price=ice_price,
            amount=current_slice_amount,
            order_type="limit",
            created_at_ms=timestamp,
            is_iceberg=True,
            iceberg_parent_id=parent_id
        )
        
        order_event = Event(
            timestamp=timestamp,
            type=EventType.ORDER,
            data=order_data,
            symbol=symbol
        )
        
        # Обновляем остаток в айсберге
        state.iceberg_remaining[parent_id] -= current_slice_amount
        state.total_orders_placed += 1
        self.push_event(order_event)

    def get_metrics(self, symbol: str = "") -> Dict[str, Any]:
        """Получение метрик для символа с поддержкой мульти-инструментальности"""
        state = self.get_state(symbol)

        if not state.trades:
            metrics = {}
        else:
            fills_count = len(state.trades)
            maker_fills = len([t for t in state.trades if t.fill_type == "maker"])
            maker_rate = maker_fills / fills_count if fills_count > 0 else 0

            # Задача 059: Разбивка комиссий по типам (Maker/Taker)
            maker_fees = sum(t.fee_usd for t in state.trades if t.fill_type == "maker")
            taker_fees = sum(t.fee_usd for t in state.trades if t.fill_type == "taker")
            total_fees = maker_fees + taker_fees
            
            avg_slippage = np.mean(state.slippages) if state.slippages else 0.0
            unexecuted_rate = state.total_orders_cancelled / state.total_orders_placed if state.total_orders_placed > 0 else 0.0

            metrics = {
                "total_trades": fills_count,
                "maker_rate": maker_rate,
                "total_fees_usd": total_fees,
                # Задача 059: Метрики комиссий по типам
                "maker_fees_usd": maker_fees,
                "taker_fees_usd": taker_fees,
                "final_balance": state.balance,
                "net_pnl": state.balance - state.config.initial_balance,
                # Задача 059: Gross PnL (разница цен без учета комиссий)
                "gross_pnl": state.gross_pnl,
                "avg_slippage_bps": avg_slippage,
                "unexecuted_rate": unexecuted_rate,
                "total_orders": state.total_orders_placed,
                # Задача 058: Метрика Market Fallback Rate
                "market_fallback_rate": state.timeout_orders / state.total_orders_placed if state.total_orders_placed > 0 else 0.0
            }
        
        # Задача 215: Добавляем метрики ошибок
        if self.error_simulator:
            error_metrics = self.error_simulator.get_metrics()
            metrics.update({
                "lost_trades_count": state.lost_trades_count,
                "error_recovery_time_ms": error_metrics["avg_recovery_time_ms"],
                "max_recovery_time_ms": error_metrics["max_recovery_time_ms"],
                "total_errors": error_metrics["total_errors"],
                "error_rate_pct": error_metrics["error_rate_pct"],
                "resilience_score": self.error_simulator.get_resilience_score()
            })
        
        return metrics

    def _on_trade(self, data: TradeData, symbol: str = ""):
        """
        Задача 212: Обработка события о сделке для обновления очереди лимитных ордеров.
        Задача 213: Поддержка мульти-инструментальности

        Вызывается при получении информации о сделке на рынке.
        Обновляет состояние всех активных лимитных ордеров в очереди.
        """
        state = self.get_state(symbol)

        # Обновляем очередь: проверяем, какие ордера исполнены
        filled_orders = state.queue_manager.update_on_trade(
            trade_price=data.trade_price,
            trade_volume=data.trade_volume
        )

        # Для каждого исполненного ордера генерируем FILL событие
        for order_id in filled_orders:
            if order_id in state.orders:
                order = state.orders[order_id]
                # Исполняем ордер по цене сделки
                self._fill_order(order_id, data.trade_price, "maker", symbol)

    def _on_cancel(self, order_id: str, symbol: str = ""):
        """Отмена ордера с поддержкой мульти-инструментальности и симуляцией ошибок"""
        state = self.get_state(symbol)
        
        # Задача 215: Проверка на ошибки биржи при отмене ордера
        if self.error_simulator:
            error_type, error_data = self.error_simulator.should_fail(
                current_time_ms=self.current_time,
                order_id=order_id
            )
            
            if error_type != ExchangeErrorType.NONE:
                # Генерируем событие об ошибке
                error_event = Event(
                    timestamp=self.current_time,
                    type=EventType.EXCHANGE_ERROR,
                    data=error_data,
                    symbol=symbol
                )
                self.push_event(error_event)
                
                # Для RateLimitError: повторяем операцию после backoff
                if error_type == ExchangeErrorType.RATE_LIMIT:
                    retry_time = self.error_simulator.backoff_until_ms
                    retry_event = Event(
                        timestamp=retry_time,
                        type=EventType.CANCEL,
                        data=order_id,
                        symbol=symbol
                    )
                    self.push_event(retry_event)
                else:
                    # Для других ошибок: считаем сигнал потерянным
                    state.lost_trades_count += 1
                
                return  # Не обрабатываем отмену

        if order_id in state.orders:
            del state.orders[order_id]
            state.total_orders_cancelled += 1
            if order_id in state.signal_mids:
                del state.signal_mids[order_id]

            # Задача 212: Отменяем ордер в очереди
            state.queue_manager.cancel_order(order_id)

    def _on_exchange_error(self, data: ExchangeErrorData, symbol: str = ""):
        """
        Задача 215: Обработка ошибок биржи.
        
        Логирует ошибку и обновляет метрики.
        Для RateLimitError: retry уже запланирован в _on_order.
        """
        state = self.get_state(symbol)
        
        # Логируем ошибку (в реальном бэктесте можно записывать в файл)
        error_msg = f"[{symbol}] Exchange Error at {data.timestamp_ms}ms: " \
                   f"{data.error_type.value} (code {data.error_code}): {data.error_message}"
        
        if data.order_id:
            error_msg += f" | Order ID: {data.order_id}"
        
        # В production можно использовать logging
        # logging.warning(error_msg)
        # Для бэктеста просто пропускаем (метрики уже обновлены в simulator)


    def get_all_trades(self, symbol: str = "") -> List[Dict[str, Any]]:
        """
        Получение всех сделок для символа в формате, пригодном для CSV.
        Задача 213: Поддержка мульти-инструментальности
        
        Returns:
            Список словарей с данными о сделках, включая колонку symbol
        """
        state = self.get_state(symbol)
        trades_list = []
        
        for trade in state.trades:
            trade_dict = {
                "symbol": symbol if symbol else self._primary_symbol,
                "order_id": trade.order_id,
                "side": trade.side,
                "price": trade.price,
                "amount": trade.amount,
                "fee_usd": trade.fee_usd,
                "fill_type": trade.fill_type
            }
            trades_list.append(trade_dict)
        
        return trades_list
    
    def get_all_trades_multi_symbol(self) -> List[Dict[str, Any]]:
        """
        Получение всех сделок для всех символов в формате, пригодном для CSV.
        Задача 213: Поддержка мульти-инструментальности
        
        Returns:
            Список словарей с данными о сделках от всех символов
        """
        all_trades = []
        for symbol in self.states.keys():
            all_trades.extend(self.get_all_trades(symbol))
        
        # Сортируем по времени (если есть timestamp в TradeData)
        return all_trades
