#!/usr/bin/env python3
"""
Задача 221: Real-time Equity Streamer - Live Viewer
Клиент для визуализации потока equity обновлений в терминале.

Использование:
    python live_equity_viewer.py --port 9001 --symbol BTCUSDT

Зависимости:
    pip install websockets plotext
"""

import asyncio
import json
import argparse
import sys
from collections import deque
from datetime import datetime
import plotext as plt

try:
    import websockets
except ImportError:
    print("Error: websockets library not installed. Run: pip install websockets")
    sys.exit(1)


class EquityViewer:
    """Визуализатор equity в реальном времени"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 9001, max_points: int = 100):
        self.ws_url = f"ws://{host}:{port}/ws"
        self.max_points = max_points
        
        # Буферы для данных
        self.timestamps = deque(maxlen=max_points)
        self.equity_values = deque(maxlen=max_points)
        self.unrealized_pnl = deque(maxlen=max_points)
        self.realized_pnl = deque(maxlen=max_points)
        self.position_sizes = deque(maxlen=max_points)
        
        # Статистика
        self.update_count = 0
        self.start_time = None
        self.last_update = None
        
    async def connect_and_stream(self):
        """Подключается к WebSocket серверу и обрабатывает поток данных"""
        print(f"Connecting to {self.ws_url}...")
        
        try:
            async with websockets.connect(self.ws_url) as websocket:
                print("Connected! Streaming equity updates...")
                self.start_time = datetime.now()
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        self.process_update(data)
                        self.render_chart()
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                    except KeyboardInterrupt:
                        print("\nShutting down...")
                        break
                        
        except websockets.exceptions.WebSocketException as e:
            print(f"WebSocket error: {e}")
            print("Make sure the bot is running and the monitoring port is correct.")
        except ConnectionRefusedError:
            print(f"Connection refused to {self.ws_url}")
            print("Make sure the bot is running and the monitoring port is correct.")
        except Exception as e:
            print(f"Unexpected error: {e}")
    
    def process_update(self, data: dict):
        """Обрабатывает обновление equity"""
        self.update_count += 1
        self.last_update = datetime.now()
        
        # Извлекаем данные
        timestamp = data.get("timestamp", 0)
        total_equity = data.get("total_equity", 0.0)
        unrealized_pnl = data.get("unrealized_pnl", 0.0)
        realized_pnl = data.get("realized_pnl_day", 0.0)
        position_size = data.get("position_size", 0.0)
        
        # Добавляем в буферы
        self.timestamps.append(timestamp)
        self.equity_values.append(total_equity)
        self.unrealized_pnl.append(unrealized_pnl)
        self.realized_pnl.append(realized_pnl)
        self.position_sizes.append(position_size)
    
    def render_chart(self):
        """Отрисовывает график в терминале"""
        plt.clear_figure()
        
        if len(self.equity_values) < 2:
            return
        
        # Конвертируем timestamps в относительное время (секунды от начала)
        if self.timestamps:
            first_ts = self.timestamps[0]
            relative_times = [(ts - first_ts) / 1000.0 for ts in self.timestamps]
        else:
            relative_times = []
        
        # Основной график - Total Equity
        plt.subplot(2, 1)
        plt.plot(relative_times, list(self.equity_values), label="Total Equity", color="green")
        plt.title("Real-time Equity Monitor")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Equity (USDT)")
        
        # Добавляем текущие значения
        if self.equity_values:
            current_equity = self.equity_values[-1]
            current_unrealized = self.unrealized_pnl[-1]
            current_realized = self.realized_pnl[-1]
            current_position = self.position_sizes[-1]
            
            info_text = (
                f"Equity: ${current_equity:.2f} | "
                f"Unrealized PnL: ${current_unrealized:+.2f} | "
                f"Realized PnL: ${current_realized:+.2f} | "
                f"Position: {current_position:+.4f}"
            )
            plt.text(info_text, x=0, y=plt.plot_size()[1] - 2)
        
        # Второй график - PnL
        plt.subplot(2, 2)
        plt.plot(relative_times, list(self.unrealized_pnl), label="Unrealized PnL", color="cyan")
        plt.plot(relative_times, list(self.realized_pnl), label="Realized PnL", color="yellow")
        plt.xlabel("Time (seconds)")
        plt.ylabel("PnL (USDT)")
        plt.title("PnL Breakdown")
        
        # Статистика
        if self.start_time and self.last_update:
            uptime = (self.last_update - self.start_time).total_seconds()
            update_rate = self.update_count / uptime if uptime > 0 else 0
            
            stats_text = f"Updates: {self.update_count} | Rate: {update_rate:.1f}/s | Uptime: {uptime:.0f}s"
            plt.text(stats_text, x=0, y=0)
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Real-time Equity Viewer for Neirobot LiT"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="WebSocket server host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9001,
        help="WebSocket server port (default: 9001)"
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=100,
        help="Maximum number of data points to display (default: 100)"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        help="Trading symbol (for display purposes only)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Real-time Equity Viewer - Neirobot LiT")
    print("=" * 60)
    if args.symbol:
        print(f"Symbol: {args.symbol}")
    print(f"Server: {args.host}:{args.port}")
    print(f"Max points: {args.max_points}")
    print("=" * 60)
    print()
    
    viewer = EquityViewer(host=args.host, port=args.port, max_points=args.max_points)
    
    try:
        asyncio.run(viewer.connect_and_stream())
    except KeyboardInterrupt:
        print("\nShutdown complete.")


if __name__ == "__main__":
    main()
