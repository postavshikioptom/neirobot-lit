#!/usr/bin/env python3
"""
Farm Control - Централизованное управление фермой ботов

Предоставляет:
1. TUI интерфейс для мониторинга всех ботов
2. CLI команды для управления (panic, reload, status)

Использование:
    python farm_ctl.py ui                    # Запуск TUI дашборда
    python farm_ctl.py status --symbol BTC   # Статус конкретного бота
    python farm_ctl.py panic --symbol BTC    # Экстренная остановка
    python farm_ctl.py reload --all          # Перезагрузка всех ботов
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional
import tomli
import httpx
import click
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, DataTable, Static, Button
from textual.reactive import reactive
from rich.text import Text


class BotConfig:
    """Конфигурация бота для подключения"""
    def __init__(self, symbol: str, port: int = 9001):
        self.symbol = symbol
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"


class FarmController:
    """Контроллер для взаимодействия с ботами через HTTP API"""
    
    def __init__(self, bots: List[BotConfig]):
        self.bots = bots
        self.client = httpx.AsyncClient(timeout=5.0)
    
    async def get_status(self, bot: BotConfig) -> Optional[Dict]:
        """Получить статус бота"""
        try:
            response = await self.client.get(f"{bot.base_url}/status")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            return {"error": str(e)}
        return None
    
    async def send_panic(self, bot: BotConfig) -> bool:
        """Отправить команду PANIC"""
        try:
            response = await self.client.post(f"{bot.base_url}/panic")
            return response.status_code == 200
        except Exception:
            return False
    
    async def send_pause(self, bot: BotConfig) -> bool:
        """Отправить команду PAUSE"""
        try:
            response = await self.client.post(f"{bot.base_url}/pause")
            return response.status_code == 200
        except Exception:
            return False
    
    async def send_reload(self, bot: BotConfig) -> bool:
        """Отправить команду RELOAD"""
        try:
            response = await self.client.post(f"{bot.base_url}/reload")
            return response.status_code == 200
        except Exception:
            return False
    
    async def close(self):
        """Закрыть HTTP клиент"""
        await self.client.aclose()


class FarmApp(App):
    """Textual TUI приложение для мониторинга фермы ботов"""
    
    CSS = """
    Screen {
        background: $surface;
    }
    
    DataTable {
        height: 100%;
        margin: 1 2;
    }
    
    .status-header {
        dock: top;
        height: 3;
        background: $primary;
        color: $text;
        content-align: center middle;
        text-style: bold;
    }
    
    .button-panel {
        dock: bottom;
        height: 3;
        background: $panel;
    }
    
    Button {
        margin: 0 1;
    }
    """
    
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("p", "panic_selected", "Panic Selected"),
        ("space", "pause_selected", "Pause Selected"),
    ]
    
    def __init__(self, bots: List[BotConfig]):
        super().__init__()
        self.bots = bots
        self.controller = FarmController(bots)
        self.selected_row = None
    
    def compose(self) -> ComposeResult:
        """Создание структуры UI"""
        yield Header()
        yield Static("🤖 Neirobot Farm Dashboard", classes="status-header")
        yield DataTable(id="farm_table")
        yield Container(
            Button("Refresh", id="btn_refresh", variant="primary"),
            Button("Pause Selected", id="btn_pause", variant="warning"),
            Button("Panic Selected", id="btn_panic", variant="error"),
            Button("Reload Selected", id="btn_reload", variant="success"),
            classes="button-panel"
        )
        yield Footer()
    
    def on_mount(self) -> None:
        """Инициализация при запуске"""
        table = self.query_one(DataTable)
        table.cursor_type = "row"
        
        # Добавляем колонки
        table.add_columns(
            "Symbol",
            "Status", 
            "PnL",
            "Position",
            "Latency (ms)",
            "Uptime (s)"
        )
        
        # Добавляем строки для каждого бота
        for bot in self.bots:
            table.add_row(
                bot.symbol,
                "connecting...",
                "-",
                "-",
                "-",
                "-",
                key=bot.symbol
            )
        
        # Запускаем периодическое обновление каждую секунду
        self.set_interval(1.0, self.update_status)
    
    async def update_status(self) -> None:
        """Обновление статусов всех ботов"""
        table = self.query_one(DataTable)
        
        for bot in self.bots:
            status_data = await self.controller.get_status(bot)
            
            if status_data and "error" not in status_data:
                # Форматируем данные
                status = status_data.get("status", "unknown")
                pnl = status_data.get("pnl")
                position = status_data.get("position")
                latency = status_data.get("latency_ms")
                uptime = status_data.get("uptime_secs", 0)
                
                # Цветовое кодирование статуса
                if status == "running":
                    status_text = Text(status, style="bold green")
                elif status == "emergency":
                    status_text = Text(status, style="bold red")
                elif status == "blocked":
                    status_text = Text(status, style="bold yellow")
                else:
                    status_text = Text(status, style="bold white")
                
                # Цветовое кодирование PnL
                if pnl is not None:
                    pnl_value = float(pnl)
                    if pnl_value > 0:
                        pnl_text = Text(f"+{pnl_value:.2f}", style="bold green")
                    elif pnl_value < 0:
                        pnl_text = Text(f"{pnl_value:.2f}", style="bold red")
                    else:
                        pnl_text = Text(f"{pnl_value:.2f}", style="white")
                else:
                    pnl_text = Text("-", style="dim")
                
                # Обновляем строку
                table.update_cell(
                    bot.symbol, "Status", status_text
                )
                table.update_cell(
                    bot.symbol, "PnL", pnl_text
                )
                table.update_cell(
                    bot.symbol, "Position", 
                    f"{position:.4f}" if position is not None else "-"
                )
                table.update_cell(
                    bot.symbol, "Latency (ms)", 
                    f"{latency:.1f}" if latency is not None else "-"
                )
                table.update_cell(
                    bot.symbol, "Uptime (s)", str(uptime)
                )
            else:
                # Ошибка подключения
                error_msg = status_data.get("error", "connection failed") if status_data else "no response"
                table.update_cell(
                    bot.symbol, "Status", Text(error_msg, style="bold red")
                )
    
    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Обработка выбора строки"""
        self.selected_row = event.row_key.value
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Обработка нажатий кнопок"""
        if event.button.id == "btn_refresh":
            await self.update_status()
        
        elif event.button.id == "btn_pause":
            if self.selected_row:
                bot = next((b for b in self.bots if b.symbol == self.selected_row), None)
                if bot:
                    success = await self.controller.send_pause(bot)
                    if success:
                        self.notify(f"PAUSE sent to {bot.symbol}", severity="warning")
                    else:
                        self.notify(f"Failed to send PAUSE to {bot.symbol}", severity="error")
        
        elif event.button.id == "btn_panic":
            if self.selected_row:
                bot = next((b for b in self.bots if b.symbol == self.selected_row), None)
                if bot:
                    success = await self.controller.send_panic(bot)
                    if success:
                        self.notify(f"PANIC sent to {bot.symbol}", severity="warning")
                    else:
                        self.notify(f"Failed to send PANIC to {bot.symbol}", severity="error")
        
        elif event.button.id == "btn_reload":
            if self.selected_row:
                bot = next((b for b in self.bots if b.symbol == self.selected_row), None)
                if bot:
                    success = await self.controller.send_reload(bot)
                    if success:
                        self.notify(f"RELOAD sent to {bot.symbol}", severity="information")
                    else:
                        self.notify(f"Failed to send RELOAD to {bot.symbol}", severity="error")
    
    def action_refresh(self) -> None:
        """Обновить статусы (горячая клавиша R)"""
        asyncio.create_task(self.update_status())
    
    def action_panic_selected(self) -> None:
        """PANIC для выбранного бота (горячая клавиша P)"""
        if self.selected_row:
            bot = next((b for b in self.bots if b.symbol == self.selected_row), None)
            if bot:
                asyncio.create_task(self.controller.send_panic(bot))
                self.notify(f"PANIC sent to {bot.symbol}", severity="warning")
    
    def action_pause_selected(self) -> None:
        """PAUSE для выбранного бота (горячая клавиша Space)"""
        if self.selected_row:
            bot = next((b for b in self.bots if b.symbol == self.selected_row), None)
            if bot:
                asyncio.create_task(self.controller.send_pause(bot))
                self.notify(f"PAUSE sent to {bot.symbol}", severity="information")
    
    async def on_unmount(self) -> None:
        """Очистка при закрытии"""
        await self.controller.close()


def discover_bots(bots_dir: Path = Path("bots")) -> List[BotConfig]:
    """Автоматическое обнаружение ботов в директории bots/"""
    bots = []
    
    if not bots_dir.exists():
        return bots
    
    for bot_path in bots_dir.iterdir():
        if bot_path.is_dir():
            config_file = bot_path / "config.toml"
            if config_file.exists():
                symbol = bot_path.name
                # Пытаемся прочитать порт из конфигурации
                port = 9001  # Дефолтный порт
                try:
                    with open(config_file, "rb") as f:
                        config_data = tomli.load(f)
                        # Ищем monitoring_port в конфигурации
                        if "monitoring_port" in config_data:
                            port = config_data["monitoring_port"]
                        elif "bot" in config_data and "monitoring_port" in config_data["bot"]:
                            port = config_data["bot"]["monitoring_port"]
                except Exception as e:
                    click.echo(f"Warning: Failed to read port from {config_file}: {e}", err=True)
                
                bots.append(BotConfig(symbol=symbol, port=port))
    
    return bots


@click.group()
def cli():
    """Farm Control - Управление фермой ботов"""
    pass


@cli.command()
def ui():
    """Запустить TUI дашборд"""
    bots = discover_bots()
    
    if not bots:
        click.echo("No bots found in bots/ directory", err=True)
        sys.exit(1)
    
    click.echo(f"Found {len(bots)} bot(s): {', '.join(b.symbol for b in bots)}")
    
    app = FarmApp(bots)
    app.run()


@cli.command()
@click.option("--symbol", required=True, help="Symbol of the bot")
@click.option("--port", type=int, default=None, help="Port of the command server (auto-detect from config if not specified)")
async def status(symbol: str, port: Optional[int]):
    """Получить статус бота"""
    if port is None:
        # Пытаемся автоматически определить порт из конфигурации
        bot_config_path = Path("bots") / symbol / "config.toml"
        if bot_config_path.exists():
            try:
                with open(bot_config_path, "rb") as f:
                    config_data = tomli.load(f)
                    port = config_data.get("monitoring_port", 9001)
            except Exception:
                port = 9001
        else:
            port = 9001
    
    bot = BotConfig(symbol=symbol, port=port)
    controller = FarmController([bot])
    
    try:
        status_data = await controller.get_status(bot)
        if status_data:
            click.echo(f"Status for {symbol}:")
            for key, value in status_data.items():
                click.echo(f"  {key}: {value}")
        else:
            click.echo(f"Failed to get status for {symbol}", err=True)
    finally:
        await controller.close()


@cli.command()
@click.option("--symbol", required=True, help="Symbol of the bot to panic")
@click.option("--port", type=int, default=None, help="Port of the command server (auto-detect from config if not specified)")
async def panic(symbol: str, port: Optional[int]):
    """Отправить команду PANIC боту"""
    if port is None:
        # Пытаемся автоматически определить порт из конфигурации
        bot_config_path = Path("bots") / symbol / "config.toml"
        if bot_config_path.exists():
            try:
                with open(bot_config_path, "rb") as f:
                    config_data = tomli.load(f)
                    port = config_data.get("monitoring_port", 9001)
            except Exception:
                port = 9001
        else:
            port = 9001
    
    bot = BotConfig(symbol=symbol, port=port)
    controller = FarmController([bot])
    
    try:
        click.confirm(
            f"Are you sure you want to PANIC {symbol}? This will close all positions!",
            abort=True
        )
        
        success = await controller.send_panic(bot)
        if success:
            click.echo(f"PANIC command sent to {symbol}")
        else:
            click.echo(f"Failed to send PANIC to {symbol}", err=True)
    finally:
        await controller.close()


@cli.command()
@click.option("--symbol", required=True, help="Symbol of the bot to pause")
@click.option("--port", type=int, default=None, help="Port of the command server (auto-detect from config if not specified)")
async def pause(symbol: str, port: Optional[int]):
    """Отправить команду PAUSE боту (приостановка торговли)"""
    if port is None:
        # Пытаемся автоматически определить порт из конфигурации
        bot_config_path = Path("bots") / symbol / "config.toml"
        if bot_config_path.exists():
            try:
                with open(bot_config_path, "rb") as f:
                    config_data = tomli.load(f)
                    port = config_data.get("monitoring_port", 9001)
            except Exception:
                port = 9001
        else:
            port = 9001
    
    bot = BotConfig(symbol=symbol, port=port)
    controller = FarmController([bot])
    
    try:
        success = await controller.send_pause(bot)
        if success:
            click.echo(f"PAUSE command sent to {symbol}")
        else:
            click.echo(f"Failed to send PAUSE to {symbol}", err=True)
    finally:
        await controller.close()


@cli.command()
@click.option("--symbol", help="Symbol of the bot to reload")
@click.option("--all", "reload_all", is_flag=True, help="Reload all bots")
@click.option("--port", type=int, default=None, help="Port of the command server (auto-detect from config if not specified)")
async def reload(symbol: Optional[str], reload_all: bool, port: Optional[int]):
    """Отправить команду RELOAD боту"""
    if not symbol and not reload_all:
        click.echo("Either --symbol or --all must be specified", err=True)
        sys.exit(1)
    
    if reload_all:
        bots = discover_bots()
    else:
        # Если указан символ, определяем порт
        if port is None:
            bot_config_path = Path("bots") / symbol / "config.toml"
            if bot_config_path.exists():
                try:
                    with open(bot_config_path, "rb") as f:
                        config_data = tomli.load(f)
                        port = config_data.get("monitoring_port", 9001)
                except Exception:
                    port = 9001
            else:
                port = 9001
        bots = [BotConfig(symbol=symbol, port=port)]
    
    controller = FarmController(bots)
    
    try:
        for bot in bots:
            success = await controller.send_reload(bot)
            if success:
                click.echo(f"RELOAD command sent to {bot.symbol}")
            else:
                click.echo(f"Failed to send RELOAD to {bot.symbol}", err=True)
    finally:
        await controller.close()


if __name__ == "__main__":
    # Для async команд используем asyncio
    import inspect
    
    # Создаем обертки для async команд
    original_commands = {}
    for name, command in cli.commands.items():
        if inspect.iscoroutinefunction(command.callback):
            original_commands[name] = command.callback
            # Создаем синхронную обертку
            def make_wrapper(coro_func):
                def wrapper(*args, **kwargs):
                    return asyncio.run(coro_func(*args, **kwargs))
                return wrapper
            command.callback = make_wrapper(command.callback)
    
    cli()
