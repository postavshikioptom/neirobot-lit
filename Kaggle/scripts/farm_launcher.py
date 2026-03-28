#!/usr/bin/env python3
"""
Farm Launcher - Автоматизированный запуск ботов с изоляцией ресурсов (Задача 230)

Использует systemd-run для обеспечения жестких лимитов CPU и памяти.
Поддерживает конфигурацию через TOML файлы и переменные окружения.

Примеры использования:
    # Запустить один бот
    python farm_launcher.py start BTCUSDT

    # Запустить несколько ботов
    python farm_launcher.py start BTCUSDT ETHUSDT DOGEUSDT

    # Запустить с пользовательскими параметрами
    python farm_launcher.py start BTCUSDT --cpu-core 1 --memory-mb 1024

    # Остановить бота
    python farm_launcher.py stop BTCUSDT

    # Просмотр статуса
    python farm_launcher.py status

    # Просмотр логов
    python farm_launcher.py logs BTCUSDT
"""

import argparse
import subprocess
import sys
import os
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class FarmLauncher:
    """Менеджер запуска ботов с изоляцией ресурсов"""

    def __init__(self, project_root: Optional[Path] = None):
        """
        Инициализация FarmLauncher

        Args:
            project_root: Корневая директория проекта (по умолчанию текущая директория)
        """
        self.project_root = project_root or Path.cwd()
        self.binary_path = self.project_root / "target" / "release" / "run-bot"
        self.bots_dir = self.project_root / "bots"
        self.log_dir = self.project_root / "logs" / "farm"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _get_bot_config(self, symbol: str) -> Dict:
        """
        Получить конфигурацию бота из TOML файла

        Args:
            symbol: Символ торговой пары (например, BTCUSDT)

        Returns:
            Словарь с конфигурацией
        """
        config_path = self.bots_dir / symbol / "config.toml"
        
        if not config_path.exists():
            logger.warning(f"Config not found for {symbol}: {config_path}")
            return {}

        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        try:
            with open(config_path, 'rb') as f:
                config = tomllib.load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to parse config for {symbol}: {e}")
            return {}

    def _get_resource_limits(self, symbol: str, 
                            cpu_core: Optional[int] = None,
                            memory_mb: Optional[int] = None) -> Tuple[int, int]:
        """
        Получить лимиты ресурсов для бота

        Args:
            symbol: Символ торговой пары
            cpu_core: Переопределение номера ядра CPU
            memory_mb: Переопределение лимита памяти

        Returns:
            Кортеж (cpu_core, memory_mb)
        """
        config = self._get_bot_config(symbol)
        
        # Получаем значения из конфига
        config_cpu_core = None
        config_memory_mb = 512  # Default
        
        if 'system' in config:
            config_cpu_core = config['system'].get('cpu_core')
            config_memory_mb = config['system'].get('max_memory_mb', 512)
        
        # Переопределяем из параметров если указаны
        final_cpu_core = cpu_core if cpu_core is not None else config_cpu_core
        final_memory_mb = memory_mb if memory_mb is not None else config_memory_mb
        
        return final_cpu_core, final_memory_mb

    def _build_systemd_run_cmd(self, symbol: str, 
                              cpu_core: Optional[int] = None,
                              memory_mb: Optional[int] = None) -> List[str]:
        """
        Построить команду systemd-run с параметрами изоляции

        Args:
            symbol: Символ торговой пары
            cpu_core: Номер ядра CPU для привязки
            memory_mb: Лимит памяти в MB

        Returns:
            Список аргументов для subprocess
        """
        cpu_core, memory_mb = self._get_resource_limits(symbol, cpu_core, memory_mb)
        
        cmd = [
            "systemd-run",
            "--user",
            "--scope",
            f"-p MemoryMax={memory_mb}M",
            f"-p CPUQuota=100%",
        ]
        
        # Добавляем CPU affinity если указано
        if cpu_core is not None:
            cmd.append(f"-p AllowedCPUs={cpu_core}")
        
        # Добавляем имя scope для отслеживания
        cmd.append(f"--unit=neirobot-{symbol}.scope")
        
        # Добавляем сам бинарник и параметры
        cmd.extend([
            str(self.binary_path),
            symbol
        ])
        
        return cmd

    def start(self, symbols: List[str], 
             cpu_core: Optional[int] = None,
             memory_mb: Optional[int] = None,
             detach: bool = True) -> bool:
        """
        Запустить ботов с изоляцией ресурсов

        Args:
            symbols: Список символов для запуска
            cpu_core: Переопределение номера ядра CPU
            memory_mb: Переопределение лимита памяти
            detach: Запустить в фоне (True) или ждать завершения (False)

        Returns:
            True если успешно, False если ошибка
        """
        if not self.binary_path.exists():
            logger.error(f"Binary not found: {self.binary_path}")
            logger.info("Please build the project first: cargo build --release")
            return False

        success = True
        for symbol in symbols:
            try:
                cmd = self._build_systemd_run_cmd(symbol, cpu_core, memory_mb)
                
                logger.info(f"Starting {symbol}...")
                logger.debug(f"Command: {' '.join(cmd)}")
                
                if detach:
                    # Запускаем в фоне
                    result = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    # Даем время на инициализацию
                    time.sleep(1)
                    
                    if result.poll() is not None:
                        # Процесс завершился
                        stdout, stderr = result.communicate()
                        logger.error(f"Failed to start {symbol}")
                        if stderr:
                            logger.error(f"Error: {stderr}")
                        success = False
                    else:
                        logger.info(f"✓ {symbol} started successfully")
                else:
                    # Ждем завершения
                    result = subprocess.run(cmd, check=True)
                    logger.info(f"✓ {symbol} completed")
                    
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to start {symbol}: {e}")
                success = False
            except Exception as e:
                logger.error(f"Unexpected error starting {symbol}: {e}")
                success = False

        return success

    def stop(self, symbols: List[str]) -> bool:
        """
        Остановить ботов

        Args:
            symbols: Список символов для остановки

        Returns:
            True если успешно, False если ошибка
        """
        success = True
        for symbol in symbols:
            try:
                logger.info(f"Stopping {symbol}...")
                
                # Используем systemctl для остановки scope
                cmd = [
                    "systemctl",
                    "--user",
                    "stop",
                    f"neirobot-{symbol}.scope"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info(f"✓ {symbol} stopped successfully")
                else:
                    # Scope может не существовать, это нормально
                    logger.warning(f"Could not stop {symbol} (may not be running)")
                    
            except Exception as e:
                logger.error(f"Error stopping {symbol}: {e}")
                success = False

        return success

    def status(self) -> bool:
        """
        Показать статус всех запущенных ботов

        Returns:
            True если успешно
        """
        try:
            logger.info("Checking bot status...")
            
            # Получаем список всех scope'ов neirobot
            cmd = [
                "systemctl",
                "--user",
                "list-units",
                "--type=scope",
                "--all",
                "--no-pager"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error("Failed to get systemctl status")
                return False
            
            # Парсим вывод
            lines = result.stdout.split('\n')
            neirobot_scopes = [l for l in lines if 'neirobot-' in l]
            
            if not neirobot_scopes:
                logger.info("No running bots found")
                return True
            
            logger.info("\nRunning bots:")
            logger.info("-" * 80)
            logger.info(f"{'SYMBOL':<15} {'STATUS':<20} {'MEMORY':<15} {'CPU':<10}")
            logger.info("-" * 80)
            
            for line in neirobot_scopes:
                parts = line.split()
                if len(parts) >= 2:
                    scope_name = parts[0]
                    status = parts[1] if len(parts) > 1 else "unknown"
                    
                    # Извлекаем символ из имени scope
                    symbol = scope_name.replace('neirobot-', '').replace('.scope', '')
                    
                    # Получаем информацию о ресурсах
                    memory_info = self._get_scope_memory(scope_name)
                    cpu_info = self._get_scope_cpu(scope_name)
                    
                    logger.info(f"{symbol:<15} {status:<20} {memory_info:<15} {cpu_info:<10}")
            
            logger.info("-" * 80)
            return True
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return False

    def _get_scope_memory(self, scope_name: str) -> str:
        """Получить информацию о памяти scope'а"""
        try:
            cmd = [
                "systemctl",
                "--user",
                "show",
                scope_name,
                "-p", "MemoryCurrent"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Парсим вывод вида "MemoryCurrent=123456789"
                for line in result.stdout.split('\n'):
                    if 'MemoryCurrent=' in line:
                        bytes_val = int(line.split('=')[1])
                        mb_val = bytes_val / (1024 * 1024)
                        return f"{mb_val:.1f}MB"
            
            return "N/A"
        except:
            return "N/A"

    def _get_scope_cpu(self, scope_name: str) -> str:
        """Получить информацию о CPU scope'а"""
        try:
            cmd = [
                "systemctl",
                "--user",
                "show",
                scope_name,
                "-p", "CPUUsageNSec"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return "active"
            
            return "N/A"
        except:
            return "N/A"

    def logs(self, symbol: str, lines: int = 50, follow: bool = False) -> bool:
        """
        Показать логи бота

        Args:
            symbol: Символ торговой пары
            lines: Количество строк для показа
            follow: Следить за логами в реальном времени

        Returns:
            True если успешно
        """
        try:
            log_file = self.bots_dir / symbol / "logs" / "bot.log"
            
            if not log_file.exists():
                logger.warning(f"Log file not found: {log_file}")
                return False
            
            if follow:
                # Используем tail -f
                cmd = ["tail", "-f", str(log_file)]
            else:
                # Показываем последние N строк
                cmd = ["tail", "-n", str(lines), str(log_file)]
            
            subprocess.run(cmd)
            return True
            
        except Exception as e:
            logger.error(f"Error reading logs: {e}")
            return False


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description="Farm Launcher - Запуск ботов с изоляцией ресурсов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Запустить один бот
  python farm_launcher.py start BTCUSDT

  # Запустить несколько ботов
  python farm_launcher.py start BTCUSDT ETHUSDT DOGEUSDT

  # Запустить с пользовательскими параметрами
  python farm_launcher.py start BTCUSDT --cpu-core 1 --memory-mb 1024

  # Остановить бота
  python farm_launcher.py stop BTCUSDT

  # Просмотр статуса
  python farm_launcher.py status

  # Просмотр логов
  python farm_launcher.py logs BTCUSDT
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Команда')

    # Команда start
    start_parser = subparsers.add_parser('start', help='Запустить ботов')
    start_parser.add_argument('symbols', nargs='+', help='Символы для запуска')
    start_parser.add_argument('--cpu-core', type=int, help='Номер ядра CPU')
    start_parser.add_argument('--memory-mb', type=int, help='Лимит памяти в MB')
    start_parser.add_argument('--no-detach', action='store_true', help='Не запускать в фоне')

    # Команда stop
    stop_parser = subparsers.add_parser('stop', help='Остановить ботов')
    stop_parser.add_argument('symbols', nargs='+', help='Символы для остановки')

    # Команда status
    status_parser = subparsers.add_parser('status', help='Показать статус')

    # Команда logs
    logs_parser = subparsers.add_parser('logs', help='Показать логи')
    logs_parser.add_argument('symbol', help='Символ')
    logs_parser.add_argument('--lines', type=int, default=50, help='Количество строк')
    logs_parser.add_argument('--follow', action='store_true', help='Следить за логами')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    launcher = FarmLauncher()

    if args.command == 'start':
        success = launcher.start(
            args.symbols,
            cpu_core=args.cpu_core,
            memory_mb=args.memory_mb,
            detach=not args.no_detach
        )
        return 0 if success else 1

    elif args.command == 'stop':
        success = launcher.stop(args.symbols)
        return 0 if success else 1

    elif args.command == 'status':
        success = launcher.status()
        return 0 if success else 1

    elif args.command == 'logs':
        success = launcher.logs(args.symbol, args.lines, args.follow)
        return 0 if success else 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
