"""
train_runtime.py — Runtime/bootstrap helpers для train.py.
Вынесено из train.py в рамках задачи 322.3.
"""
import psutil
import torch
import pytorch_lightning as pl
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainPaths:
    base_path: Path
    data_path: Path
    norm_params_path: Path
    checkpoint_dir: Path
    cache_dir: Path


def seed_training(seed: int = 42) -> None:
    """Фиксирует seed для воспроизводимости."""
    pl.seed_everything(seed)


def build_train_paths(module_file: str, symbol: str) -> TrainPaths:
    """Строит все пути для обучения на основе расположения модуля и символа."""
    base_path = Path(module_file).parent.parent.parent
    return TrainPaths(
        base_path=base_path,
        data_path=base_path / "bots" / symbol / "data" / "raw",
        norm_params_path=base_path / "bots" / symbol / "models" / "norm_params.json",
        checkpoint_dir=base_path / "bots" / symbol / "models" / "checkpoints",
        cache_dir=base_path / "bots" / symbol / "models" / "cache",
    )


def build_dataloader_kwargs(args, *, shuffle: bool) -> dict:
    """
    Единый источник правды для kwargs DataLoader.
    Используется в main train, Optuna, CV и distill путях.
    """
    kwargs = {
        'batch_size': args.batch_size,
        'shuffle': shuffle,
        'num_workers': args.num_workers,
        'pin_memory': args.pin_memory,
        'worker_init_fn': None,
    }
    if args.num_workers > 0:
        kwargs['persistent_workers'] = args.persistent_workers
        kwargs['prefetch_factor'] = args.prefetch_factor
    return kwargs


def resolve_trainer_precision(args) -> int | str:
    """Определяет precision для pl.Trainer на основе args.precision_mode."""
    if args.precision_mode == "32":
        return 32
    elif args.precision_mode == "16-mixed":
        return "16-mixed"
    else:  # auto
        return "16-mixed" if torch.cuda.is_available() else 32


def warn_if_dataset_may_exceed_ram(paths: TrainPaths, symbol: str, seq_len: int) -> None:
    """Предупреждает если датасет может не влезть в RAM."""
    mem = psutil.virtual_memory()
    available_ram_gb = mem.available / (1024 ** 3)

    pattern = f"{symbol}_*.parquet"
    files = list(paths.data_path.glob(pattern))
    if files:
        total_size_gb = sum(f.stat().st_size for f in files) / (1024 ** 3)
        decompression_factor = 4
        window_overhead = seq_len / 10
        estimated_ram_gb = total_size_gb * decompression_factor * (1 + window_overhead / 100)

        if estimated_ram_gb > available_ram_gb * 0.7:
            print(f"\n⚠️  WARNING: Dataset size (~{estimated_ram_gb:.2f} GB) may exceed available RAM ({available_ram_gb:.2f} GB)")
            print(f"   Estimated breakdown:")
            print(f"   - Compressed Parquet: {total_size_gb:.2f} GB")
            print(f"   - Decompressed in RAM: {total_size_gb * decompression_factor:.2f} GB")
            print(f"   - With sliding window (seq_len={seq_len}): {estimated_ram_gb:.2f} GB")
            print(f"   Continuing with 'memory' mode as requested...\n")
