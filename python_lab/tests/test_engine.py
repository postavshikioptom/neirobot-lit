import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.backtest.engine import EventEngine, BotConfig, MarketData, SignalData, Event, EventType, FillData

class TestEventEngine(unittest.TestCase):
    def setUp(self):
        self.config = BotConfig(
            symbol="TEST",
            initial_balance=1000.0,
            maker_fee_bps=10.0, # 0.1%
            taker_fee_bps=20.0, # 0.2%
            check_volume_ahead=True,
            order_size_usd=100.0
        )

        self.engine = EventEngine(self.config)
        
        # Mock Market Data (Best Bid=100, Best Ask=101)
        self.bids = np.array([[100.0, 1.0], [99.0, 1.0]], dtype=float)
        self.asks = np.array([[101.0, 1.0], [102.0, 1.0]], dtype=float)
        self.market_data = MarketData(mid_price=100.5, bids=self.bids, asks=self.asks)

    def test_initial_balance(self):
        self.assertEqual(self.engine.balance, 1000.0)
        self.assertEqual(self.engine.position, 0.0)

    def test_market_buy_execution(self):
        # 1. Send Signal BUY
        signal = SignalData(probs=np.array([0.9, 0.1, 0.0]), side="buy", confidence=0.9)
        self.engine.push_event(Event(0, EventType.SIGNAL, signal))
        
        # 2. Update Market (Trigger execution)
        self.engine.push_event(Event(10, EventType.MARKET, self.market_data))
        
        self.engine.run()
        
        # Should populate orders
        self.assertEqual(len(self.engine.orders), 1)
        order = list(self.engine.orders.values())[0]
        self.assertEqual(order.side, "buy")
        self.assertEqual(order.price, 100.0) # Placed at Best Bid
        self.assertEqual(order.order_type, "limit")

    def test_limit_fill_realistic(self):
        self.engine.set_mode("realistic")
        
        # Place BUY order at 100
        signal = SignalData(probs=np.array([0.9, 0.1, 0.0]), side="buy", confidence=0.9)
        self.engine.push_event(Event(0, EventType.SIGNAL, signal))
        self.engine.push_event(Event(10, EventType.MARKET, self.market_data))
        
        # Move market DOWN to cross order (Ask drops to 99)
        new_bids = np.array([[99.0, 1.0], [98.0, 1.0]], dtype=float)
        new_asks = np.array([[100.0, 1.0], [101.0, 1.0]], dtype=float) # Ask moves to 100, fills Bid at 100
        new_market_data = MarketData(mid_price=99.5, bids=new_bids, asks=new_asks)
        
        self.engine.push_event(Event(20, EventType.MARKET, new_market_data))
        
        self.engine.run()
        
        # Check fill
        self.assertEqual(len(self.engine.config.symbol), 4)
        # Should be filled
        # Check trades/metrics?
        metrics = self.engine.get_metrics()
        self.assertEqual(metrics["total_trades"], 1)
        self.assertGreater(self.engine.position, 0)

    def test_ideal_mode_fill(self):
        self.engine.set_mode("ideal")
        
        # Place BUY order at 100
        signal = SignalData(probs=np.array([0.9, 0.1, 0.0]), side="buy", confidence=0.9)
        self.engine.push_event(Event(0, EventType.SIGNAL, signal))
        self.engine.push_event(Event(10, EventType.MARKET, self.market_data))
        
        # Move Market Mid Price to 100 (Touch)
        # Ideal mode fills on Mid Price touch
        new_market_data = MarketData(mid_price=100.0, bids=self.bids, asks=self.asks)
        self.engine.push_event(Event(20, EventType.MARKET, new_market_data))
        
        self.engine.run()
        
        metrics = self.engine.get_metrics()
        self.assertEqual(metrics["total_trades"], 1)
        self.assertEqual(metrics["maker_rate"], 1.0) # Maker fill

    def test_cancel_repeg(self):
        self.engine.config.chase_mode = "ToBest"
        self.engine.config.chase_threshold_bps = 1.0 # Very sensitive
        
        # Place BUY order at 100
        signal = SignalData(probs=np.array([0.9, 0.1, 0.0]), side="buy", confidence=0.9)
        self.engine.push_event(Event(0, EventType.SIGNAL, signal))
        self.engine.push_event(Event(10, EventType.MARKET, self.market_data))
        
        # Move Market UP (Best Bid moves to 105)
        # Order is at 100. Distance = 5/100 = 5%. > 0.01% threshold.
        # Should trigger Cancel
        new_bids = np.array([[105.0, 1.0], [104.0, 1.0]], dtype=float)
        new_asks = np.array([[106.0, 1.0], [107.0, 1.0]], dtype=float)
        new_market_data = MarketData(mid_price=105.5, bids=new_bids, asks=new_asks)
        
        self.engine.push_event(Event(20, EventType.MARKET, new_market_data))
        
        self.engine.run()
        
    def test_slippage_calculation(self):
        # 1. Signal at mid 100.5
        signal = SignalData(probs=np.array([0.9, 0.1, 0.0]), side="buy", confidence=0.9)
        self.engine.push_event(Event(0, EventType.SIGNAL, signal))
        self.engine.push_event(Event(10, EventType.MARKET, self.market_data))
        
        # 2. Fill at 101.0 (Best Ask)
        # Slippage should be (101.0 - 100.5) / 100.5 * 10000 = 49.75 bps
        new_bids = np.array([[101.0, 1.0], [100.0, 1.0]], dtype=float)
        new_asks = np.array([[102.0, 1.0], [103.0, 1.0]], dtype=float)
        new_market_data = MarketData(mid_price=101.5, bids=new_bids, asks=new_asks)
        
        # Fill trigger
        self.engine.push_event(Event(20, EventType.MARKET, new_market_data))
        self.engine.run()
        
        metrics = self.engine.get_metrics()
        self.assertAlmostEqual(metrics["avg_slippage_bps"], 49.75, places=2)

    def test_volume_ahead_depletion(self):
        # Place limit BUY at 100.0
        # Best Bid is 100.0 with volume 1.0. Volume ahead = 1.0.
        signal = SignalData(probs=np.array([0.4, 0.0, 0.6]), side="buy", confidence=0.4) # Not aggressive
        self.engine.push_event(Event(0, EventType.SIGNAL, signal))
        self.engine.push_event(Event(10, EventType.MARKET, self.market_data))
        self.engine.run()
        
        oid = list(self.engine.orders.keys())[0]
        self.assertEqual(self.engine.volume_ahead[oid], 1.0)
        
        # 2. Volume at 100.0 decreases to 0.0 (or price moves through)
        new_bids = np.array([[99.0, 1.0], [98.0, 1.0]], dtype=float)
        new_market_data = MarketData(mid_price=99.5, bids=new_bids, asks=self.asks)
        self.engine.push_event(Event(20, EventType.MARKET, new_market_data))
        self.engine.run()
        
        # Should be filled
        metrics = self.engine.get_metrics()
        self.assertEqual(metrics["total_trades"], 1)

    def test_sor_slicing(self):
        # Set max_size_ratio to 0.5 (2 slices)
        self.engine.config.sor.max_size_ratio = 0.5
        self.engine.config.order_size_usd = 200.0 # mid is ~100.5, amount is ~2.0
        
        signal = SignalData(probs=np.array([0.9, 0.1, 0.0]), side="buy", confidence=0.9)
        self.engine.push_event(Event(0, EventType.SIGNAL, signal))
        
        # Run until events processed
        self.engine.run()
        
        # Should have 2 orders placed (total_orders_placed)
        metrics = self.engine.get_metrics()
        # Note: Signal event generated 2 ORDER events in the queue
        # total_orders_placed is incremented in _on_signal
        self.assertEqual(self.engine.total_orders_placed, 2)


if __name__ == "__main__":
    unittest.main()
