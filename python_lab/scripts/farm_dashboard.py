#!/usr/bin/env python3
"""
Farm Dashboard - Real-time Bot Monitoring
Задача 226: Дашборд для мониторинга состояния фермы ботов

Использует rich для красивого вывода и psutil для сбора метрик
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Optional

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback

import psutil
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text


def load_farm_config(config_path: str = "farm.toml") -> dict:
    """Загрузка конфигурации фермы"""
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def load_bot_pids(pid_dir: Path = Path("bots/.pids")) -> Dict[str, int]:
    """Загрузка PID файлов ботов"""
    pids = {}
    if not pid_dir.exists():
        return pids
    
    for pid_file in pid_dir.glob("*.pid"):
        symbol = pid_file.stem
        try:
            pid = int(pid_file.read_text().strip())
            pids[symbol] = pid
        except (ValueError, IOError):
            pass
    
    return pids


def get_bot_info(symbol: str, pid: int, cpu_core: Optional[int] = None) -> dict:
    """Получить информацию о боте через psutil"""
    try:
        process = psutil.Process(pid)
        
        if not process.is_running():
            return {
                "symbol": symbol,
                "pid": pid,
                "cpu_core": cpu_core,
                "status": "Dead",
                "memory_mb": 0,
                "cpu_percent": 0,
                "uptime": 0,
            }
        
        # Получение метрик
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent(interval=0.1)
        create_time = process.create_time()
        uptime = time.time() - create_time
        
        # Попытка получить CPU affinity
        if cpu_core is None:
            try:
                affinity = process.cpu_affinity()
                cpu_core = affinity[0] if affinity else None
            except (psutil.AccessDenied, AttributeError):
                cpu_core = None
        
        return {
            "symbol": symbol,
            "pid": pid,
            "cpu_core": cpu_core,
            "status": "Running",
            "memory_mb": memory_info.rss / (1024 * 1024),
            "cpu_percent": cpu_percent,
            "uptime": uptime,
        }
    
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return {
            "symbol": symbol,
            "pid": pid,
            "cpu_core": cpu_core,
            "status": "Dead",
            "memory_mb": 0,
            "cpu_percent": 0,
            "uptime": 0,
        }


def format_uptime(seconds: float) -> str:
    """Форматирование uptime в читаемый вид"""
    if seconds == 0:
        return "N/A"
    
    td = timedelta(seconds=int(seconds))
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    else:
        return f"{minutes}m {seconds}s"


def create_bot_table(bots_info: List[dict]) -> Table:
    """Создание таблицы с информацией о ботах"""
    table = Table(title="🤖 Bot Farm Status", show_header=True, header_style="bold magenta")
    
    # Колонки
    table.add_column("Symbol", style="cyan", no_wrap=True)
    table.add_column("PID", style="blue")
    table.add_column("CPU Core", justify="center")
    table.add_column("RAM (MB)", justify="right")
    table.add_column("CPU %", justify="right")
    table.add_column("Uptime", justify="right")
    table.add_column("Status", justify="center")
    
    # Сортировка по символу
    bots_info.sort(key=lambda x: x["symbol"])
    
    # Добавление строк
    for bot in bots_info:
        # Цветовая индикация статуса
        if bot["status"] == "Running":
            status_text = Text("✓ Running", style="bold green")
        else:
            status_text = Text("✗ Dead", style="bold red")
        
        # Форматирование CPU core
        cpu_core_str = str(bot["cpu_core"]) if bot["cpu_core"] is not None else "N/A"
        
        # Форматирование памяти
        memory_str = f"{bot['memory_mb']:.1f}" if bot["status"] == "Running" else "N/A"
        
        # Форматирование CPU %
        cpu_str = f"{bot['cpu_percent']:.1f}" if bot["status"] == "Running" else "N/A"
        
        # Форматирование uptime
        uptime_str = format_uptime(bot["uptime"])
        
        table.add_row(
            bot["symbol"],
            str(bot["pid"]),
            cpu_core_str,
            memory_str,
            cpu_str,
            uptime_str,
            status_text,
        )
    
    return table


def create_summary_panel(bots_info: List[dict], config: dict) -> Panel:
    """Создание панели с общей статистикой"""
    total = len(bots_info)
    running = sum(1 for b in bots_info if b["status"] == "Running")
    dead = total - running
    
    # Расчет процента мертвых ботов
    dead_percentage = (dead / total * 100) if total > 0 else 0
    
    # Цветовая индикация
    if dead_percentage > config["monitoring"]["critical_alert_threshold"] * 100:
        health_status = Text("🔴 CRITICAL", style="bold red")
    elif dead_percentage > 0:
        health_status = Text("🟡 WARNING", style="bold yellow")
    else:
        health_status = Text("🟢 HEALTHY", style="bold green")
    
    # Общая статистика по ресурсам
    total_memory = sum(b["memory_mb"] for b in bots_info if b["status"] == "Running")
    avg_cpu = sum(b["cpu_percent"] for b in bots_info if b["status"] == "Running") / running if running > 0 else 0
    
    summary_text = Text()
    summary_text.append(f"Total Bots: {total}  ", style="bold")
    summary_text.append(f"Running: {running}  ", style="bold green")
    summary_text.append(f"Dead: {dead}  ", style="bold red")
    summary_text.append(f"Health: ", style="bold")
    summary_text.append(health_status)
    summary_text.append(f"\n\nTotal Memory: {total_memory:.1f} MB  ", style="bold")
    summary_text.append(f"Avg CPU: {avg_cpu:.1f}%", style="bold")
    
    return Panel(summary_text, title="📊 Farm Summary", border_style="blue")


def run_dashboard(config_path: str = "farm.toml", refresh_interval: Optional[int] = None):
    """Запуск интерактивного дашборда"""
    console = Console()
    
    # Загрузка конфигурации
    try:
        config = load_farm_config(config_path)
    except Exception as e:
        console.print(f"[bold red]Failed to load config: {e}[/bold red]")
        sys.exit(1)
    
    # Интервал обновления
    if refresh_interval is None:
        refresh_interval = config["monitoring"]["dashboard_refresh_interval"]
    
    # Список символов из конфигурации
    symbols = config["symbols"]["list"]
    pid_dir = Path("bots/.pids")
    
    console.print("[bold green]Starting Farm Dashboard...[/bold green]")
    console.print(f"Monitoring {len(symbols)} bots")
    console.print(f"Refresh interval: {refresh_interval}s")
    console.print("\nPress Ctrl+C to exit\n")
    
    try:
        with Live(console=console, refresh_per_second=1) as live:
            while True:
                # Загрузка PID файлов
                pids = load_bot_pids(pid_dir)
                
                # Сбор информации о ботах
                bots_info = []
                for symbol in symbols:
                    pid = pids.get(symbol)
                    if pid:
                        info = get_bot_info(symbol, pid)
                        bots_info.append(info)
                    else:
                        # Бот не запущен
                        bots_info.append({
                            "symbol": symbol,
                            "pid": 0,
                            "cpu_core": None,
                            "status": "Not Started",
                            "memory_mb": 0,
                            "cpu_percent": 0,
                            "uptime": 0,
                        })
                
                # Создание layout
                layout = Layout()
                layout.split_column(
                    Layout(create_summary_panel(bots_info, config), size=5),
                    Layout(create_bot_table(bots_info)),
                )
                
                # Обновление дисплея
                live.update(layout)
                
                # Ожидание
                time.sleep(refresh_interval)
    
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Dashboard stopped[/bold yellow]")


def main():
    """CLI интерфейс"""
    parser = argparse.ArgumentParser(
        description="Farm Dashboard - Real-time Bot Monitoring"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="farm.toml",
        help="Path to farm configuration file"
    )
    parser.add_argument(
        "--refresh",
        type=int,
        help="Refresh interval in seconds (overrides config)"
    )
    
    args = parser.parse_args()
    
    run_dashboard(args.config, args.refresh)


if __name__ == "__main__":
    main()
