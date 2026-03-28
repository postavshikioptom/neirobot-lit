#!/usr/bin/env python3
"""
Model Manager - Автоматизированная дистрибуция моделей
Задача 228: Безопасный деплой ONNX моделей с Atomic Push и Backup

Использование:
    python model_manager.py --source path/to/model.onnx --symbols BTC,ETH
    python model_manager.py --source path/to/model.onnx --group scalping
    python model_manager.py --source path/to/model.onnx --all
"""

import argparse
import hashlib
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

try:
    import toml
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
except ImportError as e:
    print(f"Error: Missing required dependency: {e}")
    print("Install with: pip install toml rich")
    sys.exit(1)

console = Console()


def compute_file_hash(file_path: Path) -> str:
    """
    Вычисляет SHA-256 хеш файла
    
    Args:
        file_path: Путь к файлу
        
    Returns:
        Hex-строка с хешем
    """
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_farm_config(farm_config_path: Path) -> dict:
    """
    Загружает конфигурацию фермы из farm.toml
    
    Args:
        farm_config_path: Путь к farm.toml
        
    Returns:
        Словарь с конфигурацией
    """
    if not farm_config_path.exists():
        console.print(f"[red]Error: farm.toml not found at {farm_config_path}[/red]")
        sys.exit(1)
    
    with open(farm_config_path, 'r') as f:
        config = toml.load(f)
    
    return config


def filter_symbols(
    all_symbols: List[str],
    symbols_filter: Optional[List[str]] = None,
    group_filter: Optional[str] = None,
    deploy_all: bool = False
) -> List[str]:
    """
    Фильтрует символы по заданным критериям
    
    Args:
        all_symbols: Полный список символов из farm.toml
        symbols_filter: Список символов для фильтрации (например, ['BTC', 'ETH'])
        group_filter: Группа для фильтрации (например, 'scalping')
        deploy_all: Флаг деплоя на все символы
        
    Returns:
        Отфильтрованный список символов
    """
    if deploy_all:
        return all_symbols
    
    if symbols_filter:
        # Фильтрация по символам
        # Поддерживаем короткие имена (BTC -> BTCUSDT)
        filtered = []
        for symbol_filter in symbols_filter:
            for symbol in all_symbols:
                if symbol_filter.upper() in symbol.upper():
                    filtered.append(symbol)
        return list(set(filtered))  # Убираем дубликаты
    
    if group_filter:
        # Фильтрация по группам из farm.toml
        return []  # Будет заполнено в main()
    
    return all_symbols


def deploy_model(
    source_model: Path,
    symbol: str,
    bots_dir: Path,
    dry_run: bool = False
) -> bool:
    """
    Деплоит модель для одного символа с Atomic Push и Backup
    
    Args:
        source_model: Путь к исходной модели
        symbol: Символ (например, BTCUSDT)
        bots_dir: Директория с ботами
        dry_run: Режим dry-run (без реальных изменений)
        
    Returns:
        True если деплой успешен, False иначе
    """
    model_dir = bots_dir / symbol / "model"
    
    # Проверяем существование директории
    if not model_dir.exists():
        console.print(f"[yellow]Warning: Model directory not found for {symbol}: {model_dir}[/yellow]")
        return False
    
    target_model = model_dir / "model.onnx"
    backup_model = model_dir / "model.onnx.bak"
    tmp_model = model_dir / "model.onnx.tmp"
    hash_file = model_dir / "model.hash"
    
    try:
        # 1. Backup: Копируем текущую модель в .bak (если существует)
        if target_model.exists() and not dry_run:
            console.print(f"  [cyan]Creating backup: {backup_model.name}[/cyan]")
            shutil.copy2(target_model, backup_model)
        
        # 2. Atomic Push: Копируем новую модель в .tmp
        if not dry_run:
            console.print(f"  [cyan]Copying model to temporary file...[/cyan]")
            shutil.copy2(source_model, tmp_model)
        
        # 3. Вычисляем хеш новой модели
        console.print(f"  [cyan]Computing SHA-256 hash...[/cyan]")
        model_hash = compute_file_hash(source_model if dry_run else tmp_model)
        
        # 4. Атомарное переименование .tmp -> .onnx
        if not dry_run:
            console.print(f"  [cyan]Atomic rename: {tmp_model.name} -> {target_model.name}[/cyan]")
            tmp_model.replace(target_model)
        
        # 5. Обновляем файл model.hash
        if not dry_run:
            console.print(f"  [cyan]Updating hash file: {hash_file.name}[/cyan]")
            with open(hash_file, 'w') as f:
                f.write(f"{model_hash}\n")
        
        console.print(f"  [green]✓ Successfully deployed to {symbol}[/green]")
        console.print(f"  [dim]Hash: {model_hash[:16]}...[/dim]")
        return True
        
    except Exception as e:
        console.print(f"  [red]✗ Failed to deploy to {symbol}: {e}[/red]")
        
        # Cleanup: Удаляем временный файл если он остался
        if tmp_model.exists() and not dry_run:
            try:
                tmp_model.unlink()
            except:
                pass
        
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Model Manager - Автоматизированная дистрибуция ONNX моделей",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Деплой на конкретные символы
  python model_manager.py --source model.onnx --symbols BTC,ETH,SOL
  
  # Деплой на группу (TODO: требуется реализация групп)
  python model_manager.py --source model.onnx --group scalping
  
  # Деплой на все символы (ОПАСНО! Используйте с осторожностью)
  python model_manager.py --source model.onnx --all
  
  # Dry-run режим (без реальных изменений)
  python model_manager.py --source model.onnx --symbols BTC --dry-run
        """
    )
    
    parser.add_argument(
        '--source',
        type=Path,
        required=True,
        help='Путь к исходной модели model.onnx'
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        help='Список символов через запятую (например: BTC,ETH,SOL)'
    )
    
    parser.add_argument(
        '--group',
        type=str,
        help='Группа ботов для деплоя (например: scalping, trending)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Деплой на ВСЕ символы (используйте с осторожностью!)'
    )
    
    parser.add_argument(
        '--farm-config',
        type=Path,
        default=Path('farm.toml'),
        help='Путь к farm.toml (по умолчанию: farm.toml)'
    )
    
    parser.add_argument(
        '--bots-dir',
        type=Path,
        default=Path('bots'),
        help='Директория с ботами (по умолчанию: bots/)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Режим dry-run: показать что будет сделано, но не выполнять'
    )
    
    args = parser.parse_args()
    
    # Валидация аргументов
    if not args.source.exists():
        console.print(f"[red]Error: Source model not found: {args.source}[/red]")
        sys.exit(1)
    
    if not args.symbols and not args.group and not args.all:
        console.print("[red]Error: Must specify --symbols, --group, or --all[/red]")
        parser.print_help()
        sys.exit(1)
    
    if args.all:
        console.print("[yellow]⚠ WARNING: Deploying to ALL symbols![/yellow]")
        console.print("[yellow]This is a system-wide operation. Consider Canary testing first.[/yellow]")
        response = input("Are you sure? (yes/no): ")
        if response.lower() != 'yes':
            console.print("[yellow]Deployment cancelled.[/yellow]")
            sys.exit(0)
    
    # Загружаем конфигурацию фермы
    console.print(f"[cyan]Loading farm configuration from {args.farm_config}...[/cyan]")
    farm_config = load_farm_config(args.farm_config)
    
    all_symbols = farm_config.get('symbols', {}).get('list', [])
    if not all_symbols:
        console.print("[red]Error: No symbols found in farm.toml[/red]")
        sys.exit(1)
    
    # Фильтруем символы
    symbols_filter = args.symbols.split(',') if args.symbols else None
    
    # Обработка фильтра по группам
    if args.group:
        groups = farm_config.get('groups', {})
        if args.group not in groups:
            console.print(f"[red]Error: Group '{args.group}' not found in farm.toml[/red]")
            console.print(f"[yellow]Available groups: {', '.join(groups.keys())}")
            sys.exit(1)
        target_symbols = groups[args.group]
    else:
        target_symbols = filter_symbols(
            all_symbols,
            symbols_filter=symbols_filter,
            group_filter=args.group,
            deploy_all=args.all
        )
    
    if not target_symbols:
        console.print("[red]Error: No symbols matched the filter criteria[/red]")
        sys.exit(1)
    
    # Показываем план деплоя
    console.print("\n[bold cyan]Deployment Plan:[/bold cyan]")
    console.print(f"  Source model: {args.source}")
    console.print(f"  Target symbols: {', '.join(target_symbols)}")
    console.print(f"  Total: {len(target_symbols)} bot(s)")
    if args.dry_run:
        console.print("  [yellow]Mode: DRY-RUN (no changes will be made)[/yellow]")
    console.print()
    
    # Выполняем деплой
    start_time = time.time()
    success_count = 0
    failed_count = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Deploying models...", total=len(target_symbols))
        
        for symbol in target_symbols:
            progress.update(task, description=f"[cyan]Deploying to {symbol}...")
            
            if deploy_model(args.source, symbol, args.bots_dir, args.dry_run):
                success_count += 1
            else:
                failed_count += 1
            
            progress.advance(task)
    
    # Итоговый отчет
    elapsed_time = time.time() - start_time
    
    console.print("\n[bold cyan]Deployment Summary:[/bold cyan]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total symbols", str(len(target_symbols)))
    table.add_row("Successful", str(success_count))
    table.add_row("Failed", str(failed_count))
    table.add_row("Elapsed time", f"{elapsed_time:.2f}s")
    
    console.print(table)
    
    if failed_count > 0:
        console.print(f"\n[yellow]⚠ {failed_count} deployment(s) failed. Check logs above.[/yellow]")
        sys.exit(1)
    else:
        console.print("\n[green]✓ All deployments completed successfully![/green]")
        if not args.dry_run:
            console.print("[cyan]Note: Bots will detect model.hash changes and reload automatically.[/cyan]")


if __name__ == "__main__":
    main()
