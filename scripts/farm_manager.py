#!/usr/bin/env python3
"""
Farm Manager - Multi-Instance Bot Orchestrator
Задача 226: Система оркестрации для управления фермой ботов

Функционал:
- Параллельный запуск ботов через ThreadPoolExecutor
- CPU affinity для изоляции ядер
- Динамическое назначение портов мониторинга
- Graceful shutdown с SIGTERM
- Генерация systemd юнитов
"""

import sys
import os
import signal
import subprocess
import time
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback для Python < 3.11

import psutil


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('farm_manager')


@dataclass
class FarmConfig:
    """Конфигурация фермы из farm.toml"""
    base_monitoring_port: int
    max_parallel_starts: int
    bot_binary: str
    working_directory: str
    cpu_cores_per_bot: int
    memory_limit_mb: int
    enable_cpu_affinity: bool
    dashboard_refresh_interval: int
    critical_alert_threshold: float
    symbols: List[str]

    @classmethod
    def load(cls, config_path: str = "farm.toml") -> "FarmConfig":
        """Загрузка конфигурации из TOML файла"""
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        
        return cls(
            base_monitoring_port=data["farm"]["base_monitoring_port"],
            max_parallel_starts=data["farm"]["max_parallel_starts"],
            bot_binary=data["farm"]["bot_binary"],
            working_directory=data["farm"]["working_directory"],
            cpu_cores_per_bot=data["resources"]["cpu_cores_per_bot"],
            memory_limit_mb=data["resources"]["memory_limit_mb"],
            enable_cpu_affinity=data["resources"]["enable_cpu_affinity"],
            dashboard_refresh_interval=data["monitoring"]["dashboard_refresh_interval"],
            critical_alert_threshold=data["monitoring"]["critical_alert_threshold"],
            symbols=data["symbols"]["list"],
        )


@dataclass
class BotProcess:
    """Информация о запущенном боте"""
    symbol: str
    pid: int
    monitoring_port: int
    cpu_cores: List[int]
    start_time: float
    process: psutil.Process


class FarmManager:
    """Оркестратор для управления фермой ботов"""
    
    def __init__(self, config: FarmConfig):
        self.config = config
        self.bots: Dict[str, BotProcess] = {}
        self.pid_dir = Path("bots/.pids")
        self.pid_dir.mkdir(parents=True, exist_ok=True)
        
        # Доступные CPU ядра
        self.available_cores = list(range(psutil.cpu_count()))
        self.used_cores = set()
        
        # Graceful shutdown handler
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Обработчик сигналов для graceful shutdown"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop_all()
        sys.exit(0)
    
    def _get_next_cpu_cores(self) -> List[int]:
        """Получить следующие свободные CPU ядра согласно cpu_cores_per_bot"""
        if not self.config.enable_cpu_affinity or self.config.cpu_cores_per_bot == 0:
            return []
        
        cores = []
        cores_needed = self.config.cpu_cores_per_bot
        
        # Пытаемся найти свободные ядра
        for core in self.available_cores:
            if core not in self.used_cores and len(cores) < cores_needed:
                cores.append(core)
                self.used_cores.add(core)
        
        # Если недостаточно свободных ядер, используем round-robin
        while len(cores) < cores_needed:
            core = self.available_cores[len(cores) % len(self.available_cores)]
            cores.append(core)
        
        return cores
    
    def _release_cpu_cores(self, cores: List[int]):
        """Освободить CPU ядра"""
        for core in cores:
            if core in self.used_cores:
                self.used_cores.discard(core)
    
    def _get_monitoring_port(self, index: int) -> int:
        """Получить порт мониторинга для бота"""
        return self.config.base_monitoring_port + index
    
    def _save_pid(self, symbol: str, pid: int):
        """Сохранить PID в файл"""
        pid_file = self.pid_dir / f"{symbol}.pid"
        pid_file.write_text(str(pid))
    
    def _load_pid(self, symbol: str) -> Optional[int]:
        """Загрузить PID из файла"""
        pid_file = self.pid_dir / f"{symbol}.pid"
        if pid_file.exists():
            try:
                return int(pid_file.read_text().strip())
            except (ValueError, IOError):
                return None
        return None
    
    def _remove_pid(self, symbol: str):
        """Удалить PID файл"""
        pid_file = self.pid_dir / f"{symbol}.pid"
        if pid_file.exists():
            pid_file.unlink()
    
    def start_bot(self, symbol: str, index: int) -> bool:
        """
        Запустить одного бота
        
        Args:
            symbol: Символ для торговли
            index: Индекс бота в списке (для назначения порта)
        
        Returns:
            True если запуск успешен, False иначе
        """
        try:
            # Проверка что бот уже не запущен
            if symbol in self.bots:
                logger.warning(f"Bot {symbol} is already running")
                return False
            
            # Проверка конфигурации бота
            bot_config = Path(f"bots/{symbol}/config.toml")
            if not bot_config.exists():
                logger.error(f"Bot config not found: {bot_config}")
                return False
            
            # Назначение ресурсов
            cpu_cores = self._get_next_cpu_cores()
            monitoring_port = self._get_monitoring_port(index)
            
            # Команда запуска
            cmd = [
                self.config.bot_binary,
                symbol,
                "--config", str(bot_config),
                "--monitoring-port", str(monitoring_port),
            ]
            
            logger.info(f"Starting bot {symbol} on port {monitoring_port}, CPU cores {cpu_cores}")
            
            # Запуск процесса
            process = subprocess.Popen(
                cmd,
                cwd=self.config.working_directory,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,  # Для корректной обработки сигналов
            )
            
            # Установка CPU affinity
            if cpu_cores:
                try:
                    p = psutil.Process(process.pid)
                    p.cpu_affinity(cpu_cores)
                    logger.info(f"Set CPU affinity for {symbol} to cores {cpu_cores}")
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    logger.warning(f"Failed to set CPU affinity for {symbol}: {e}")
            
            # Установка лимита памяти (если поддерживается)
            if self.config.memory_limit_mb > 0:
                try:
                    p = psutil.Process(process.pid)
                    # На Linux можно использовать rlimit
                    if hasattr(p, 'rlimit'):
                        import resource
                        p.rlimit(resource.RLIMIT_AS, (
                            self.config.memory_limit_mb * 1024 * 1024,
                            self.config.memory_limit_mb * 1024 * 1024
                        ))
                    elif sys.platform == "win32":
                        # Использование Windows Job Objects для лимитов памяти
                        import win32job
                        import win32api
                        import win32con
                        
                        job = win32job.CreateJobObject(None, f"bot_job_{process.pid}")
                        info = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
                        
                        # Установка лимита памяти (в байтах)
                        mem_limit = self.config.memory_limit_mb * 1024 * 1024
                        info['ProcessMemoryLimit'] = mem_limit
                        info['BasicLimitInformation']['LimitFlags'] |= win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY
                        
                        win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, info)
                        
                        # Привязка процесса к Job Object
                        handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, False, process.pid)
                        win32job.AssignProcessToJobObject(job, handle)
                        logger.info(f"Set memory limit for {symbol} to {self.config.memory_limit_mb} MB (Windows Job Object)")
                except Exception as e:
                    logger.warning(f"Failed to set memory limit for {symbol}: {e}")
            
            # Сохранение информации о боте
            bot_process = BotProcess(
                symbol=symbol,
                pid=process.pid,
                monitoring_port=monitoring_port,
                cpu_cores=cpu_cores,
                start_time=time.time(),
                process=psutil.Process(process.pid),
            )
            
            self.bots[symbol] = bot_process
            self._save_pid(symbol, process.pid)
            
            logger.info(f"Bot {symbol} started successfully (PID: {process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start bot {symbol}: {e}")
            if 'cpu_cores' in locals():
                self._release_cpu_cores(cpu_cores)
            return False
    
    def stop_bot(self, symbol: str, timeout: int = 30) -> bool:
        """
        Остановить бота
        
        Args:
            symbol: Символ бота
            timeout: Таймаут ожидания завершения в секундах
        
        Returns:
            True если остановка успешна, False иначе
        """
        try:
            bot = self.bots.get(symbol)
            if not bot:
                # Попытка загрузить PID из файла
                pid = self._load_pid(symbol)
                if pid:
                    try:
                        process = psutil.Process(pid)
                        process.terminate()
                        process.wait(timeout)
                        self._remove_pid(symbol)
                        logger.info(f"Bot {symbol} stopped (PID: {pid})")
                        return True
                    except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                        self._remove_pid(symbol)
                        return False
                
                logger.warning(f"Bot {symbol} is not running")
                return False
            
            logger.info(f"Stopping bot {symbol} (PID: {bot.pid})")
            
            # Отправка SIGTERM для graceful shutdown
            bot.process.terminate()
            
            # Ожидание завершения
            try:
                bot.process.wait(timeout)
                logger.info(f"Bot {symbol} stopped gracefully")
            except psutil.TimeoutExpired:
                logger.warning(f"Bot {symbol} did not stop gracefully, sending SIGKILL")
                bot.process.kill()
                bot.process.wait(5)
            
            # Освобождение ресурсов
            if bot.cpu_cores:
                self._release_cpu_cores(bot.cpu_cores)
            self._remove_pid(symbol)
            del self.bots[symbol]
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop bot {symbol}: {e}")
            return False
    
    def start_all(self) -> Dict[str, bool]:
        """
        Запустить всех ботов параллельно
        
        Returns:
            Словарь {symbol: success}
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.config.max_parallel_starts) as executor:
            futures = {
                executor.submit(self.start_bot, symbol, idx): symbol
                for idx, symbol in enumerate(self.config.symbols)
            }
            
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    success = future.result()
                    results[symbol] = success
                except Exception as e:
                    logger.error(f"Exception starting {symbol}: {e}")
                    results[symbol] = False
        
        # Статистика
        successful = sum(1 for v in results.values() if v)
        logger.info(f"Started {successful}/{len(results)} bots successfully")
        
        return results
    
    def stop_all(self, timeout: int = 30) -> Dict[str, bool]:
        """
        Остановить всех ботов
        
        Args:
            timeout: Таймаут для каждого бота
        
        Returns:
            Словарь {symbol: success}
        """
        results = {}
        symbols = list(self.bots.keys())
        
        for symbol in symbols:
            results[symbol] = self.stop_bot(symbol, timeout)
        
        successful = sum(1 for v in results.values() if v)
        logger.info(f"Stopped {successful}/{len(results)} bots successfully")
        
        return results
    
    def restart_bot(self, symbol: str) -> bool:
        """Перезапустить бота"""
        index = self.config.symbols.index(symbol) if symbol in self.config.symbols else 0
        self.stop_bot(symbol)
        time.sleep(2)  # Небольшая задержка
        return self.start_bot(symbol, index)
    
    def restart_all(self) -> Dict[str, bool]:
        """Перезапустить всех ботов"""
        self.stop_all()
        time.sleep(2)
        return self.start_all()
    
    def get_status(self) -> Dict[str, dict]:
        """
        Получить статус всех ботов
        
        Returns:
            Словарь {symbol: {status, pid, cpu_core, memory_mb, uptime, ...}}
        """
        status = {}
        
        for symbol, bot in self.bots.items():
            try:
                if bot.process.is_running():
                    memory_info = bot.process.memory_info()
                    cpu_percent = bot.process.cpu_percent(interval=0.1)
                    uptime = time.time() - bot.start_time
                    
                    status[symbol] = {
                        "status": "Running",
                        "pid": bot.pid,
                        "cpu_cores": bot.cpu_cores,
                        "monitoring_port": bot.monitoring_port,
                        "memory_mb": memory_info.rss / (1024 * 1024),
                        "cpu_percent": cpu_percent,
                        "uptime": uptime,
                    }
                else:
                    status[symbol] = {
                        "status": "Dead",
                        "pid": bot.pid,
                        "cpu_cores": bot.cpu_cores,
                        "monitoring_port": bot.monitoring_port,
                    }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                status[symbol] = {
                    "status": "Dead",
                    "pid": bot.pid,
                    "cpu_cores": bot.cpu_cores,
                    "monitoring_port": bot.monitoring_port,
                }
        
        return status
    
    def check_health(self) -> tuple[int, int, float]:
        """
        Проверить здоровье фермы
        
        Returns:
            (total_bots, dead_bots, dead_percentage)
        """
        status = self.get_status()
        total = len(status)
        dead = sum(1 for s in status.values() if s["status"] == "Dead")
        percentage = dead / total if total > 0 else 0.0
        
        return total, dead, percentage
    
    def send_critical_alert(self, message: str):
        """
        Отправить критический алерт
        Интеграция с alert_manager (задача 222)
        """
        logger.critical(f"CRITICAL ALERT: {message}")
        
        # 1. Прямая отправка в Telegram через Python (как в alert_manager.rs)
        token = os.environ.get("TELEGRAM_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        
        if token and chat_id:
            try:
                import urllib.request
                import json
                
                # Экранирование для MarkdownV2 (минимальное)
                escaped_msg = message.replace(".", "\\.").replace("-", "\\-").replace("!", "\\!")
                
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                payload = {
                    "chat_id": chat_id,
                    "text": f"🚨 *CRITICAL FARM ALERT*\n\n{escaped_msg}",
                    "parse_mode": "MarkdownV2"
                }
                
                req = urllib.request.Request(
                    url, 
                    data=json.dumps(payload).encode('utf-8'),
                    headers={'Content-Type': 'application/json'},
                    method='POST'
                )
                
                with urllib.request.urlopen(req, timeout=10) as response:
                    if response.getcode() == 200:
                        logger.info("Critical alert sent to Telegram successfully")
            except Exception as e:
                logger.error(f"Failed to send Telegram alert directly: {e}")

        # 2. Попытка вызвать run-bot с --test-alert (для интеграции с Rust AlertManager)
        if self.config.symbols:
            first_symbol = self.config.symbols[0]
            try:
                subprocess.run(
                    [self.config.bot_binary, first_symbol, "--test-alert", message],
                    timeout=10,
                    capture_output=True,
                )
            except Exception as e:
                logger.debug(f"Integration alert skipped: {e}")
    
    def monitor_health(self):
        """Мониторинг здоровья фермы и отправка алертов"""
        total, dead, percentage = self.check_health()
        
        if percentage > self.config.critical_alert_threshold:
            message = f"Farm health critical: {dead}/{total} bots dead ({percentage*100:.1f}%)"
            self.send_critical_alert(message)
    
    def generate_systemd_units(self, output_dir: str = "deploy/generated"):
        """
        Генерация systemd юнитов для каждого символа
        
        Args:
            output_dir: Директория для сохранения юнитов
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Чтение шаблона
        template_path = Path("infra/neirobot@.service")
        if not template_path.exists():
            logger.error(f"Template not found: {template_path}")
            return
        
        template = template_path.read_text()
        
        for idx, symbol in enumerate(self.config.symbols):
            # Замена плейсхолдеров
            unit_content = template.replace("%i", symbol)
            
            # Добавление порта мониторинга в Environment
            monitoring_port = self._get_monitoring_port(idx)
            env_line = f"Environment=\"MONITORING_PORT={monitoring_port}\""
            
            # Добавление лимита памяти в Environment
            memory_limit = self.config.memory_limit_mb
            memory_env_line = f"Environment=\"MEMORY_LIMIT_MB={memory_limit}\""
            
            # Вставка после существующих Environment
            lines = unit_content.split("\n")
            insert_idx = -1
            for i, line in enumerate(lines):
                if line.startswith("Environment="):
                    insert_idx = i
            
            if insert_idx >= 0:
                lines.insert(insert_idx + 1, env_line)
                lines.insert(insert_idx + 2, memory_env_line)
            
            unit_content = "\n".join(lines)
            
            # Сохранение юнита
            unit_file = output_path / f"neirobot-lit-{symbol}.service"
            unit_file.write_text(unit_content)
            logger.info(f"Generated systemd unit: {unit_file}")
        
        logger.info(f"Generated {len(self.config.symbols)} systemd units in {output_dir}")


def main():
    """CLI интерфейс"""
    parser = argparse.ArgumentParser(
        description="Farm Manager - Multi-Instance Bot Orchestrator"
    )
    parser.add_argument(
        "command",
        choices=["start", "stop", "restart", "status", "generate-systemd"],
        help="Command to execute"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Apply command to all bots"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        help="Apply command to specific symbol"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="farm.toml",
        help="Path to farm configuration file"
    )
    
    args = parser.parse_args()
    
    # Загрузка конфигурации
    try:
        config = FarmConfig.load(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)
    
    # Создание менеджера
    manager = FarmManager(config)
    
    # Выполнение команды
    if args.command == "start":
        if args.all:
            results = manager.start_all()
            sys.exit(0 if all(results.values()) else 1)
        elif args.symbol:
            index = config.symbols.index(args.symbol) if args.symbol in config.symbols else 0
            success = manager.start_bot(args.symbol, index)
            sys.exit(0 if success else 1)
        else:
            logger.error("Specify --all or --symbol")
            sys.exit(1)
    
    elif args.command == "stop":
        if args.all:
            results = manager.stop_all()
            sys.exit(0 if all(results.values()) else 1)
        elif args.symbol:
            success = manager.stop_bot(args.symbol)
            sys.exit(0 if success else 1)
        else:
            logger.error("Specify --all or --symbol")
            sys.exit(1)
    
    elif args.command == "restart":
        if args.all:
            results = manager.restart_all()
            sys.exit(0 if all(results.values()) else 1)
        elif args.symbol:
            success = manager.restart_bot(args.symbol)
            sys.exit(0 if success else 1)
        else:
            logger.error("Specify --all or --symbol")
            sys.exit(1)
    
    elif args.command == "status":
        status = manager.get_status()
        for symbol, info in status.items():
            print(f"{symbol}: {info}")
        
        # Проверка здоровья
        manager.monitor_health()
    
    elif args.command == "generate-systemd":
        manager.generate_systemd_units()


if __name__ == "__main__":
    main()
