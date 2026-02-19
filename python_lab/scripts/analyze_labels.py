#!/usr/bin/env python3
"""
Скрипт для анализа распределения меток (Label Distribution Analysis).

Анализирует:
- Распределение классов (Up/Down/Flat)
- Imbalance Ratio (Max/Min)
- Матрицу переходов между состояниями
- Консистентность между train и val выборками

Использование:
    python scripts/analyze_labels.py --data_path bots/CAKEUSDT/data/raw
"""

import argparse
import json
from pathlib import Path
import polars as pl
import sys

# Добавляем путь к src для импорта
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import analyze_labels


def load_parquet_data(data_path: Path):
    """
    Загружает train.parquet и val.parquet из указанной директории.
    
    Args:
        data_path: Путь к директории с parquet-файлами
    
    Returns:
        tuple: (train_df, val_df) или (None, None) если файлы не найдены
    """
    train_path = data_path / "train.parquet"
    val_path = data_path / "val.parquet"
    
    train_df = None
    val_df = None
    
    if train_path.exists():
        print(f"✓ Загружаем train.parquet из {train_path}")
        train_df = pl.read_parquet(train_path)
        print(f"  Размер: {len(train_df)} строк")
    else:
        print(f"⚠️  train.parquet не найден: {train_path}")
    
    if val_path.exists():
        print(f"✓ Загружаем val.parquet из {val_path}")
        val_df = pl.read_parquet(val_path)
        print(f"  Размер: {len(val_df)} строк")
    else:
        print(f"⚠️  val.parquet не найден: {val_path}")
    
    return train_df, val_df


def check_consistency(train_result, val_result, threshold=5.0):
    """
    Проверяет консистентность распределения меток между train и val.
    
    Args:
        train_result: Результат analyze_labels для train
        val_result: Результат analyze_labels для val
        threshold: Порог различия в процентах для предупреждения
    """
    print("\n" + "="*60)
    print("=== Проверка консистентности Train vs Val ===")
    print("="*60)
    
    train_dist = {item["label"]: item["percentage"] for item in train_result["distribution"]}
    val_dist = {item["label"]: item["percentage"] for item in val_result["distribution"]}
    
    all_labels = set(train_dist.keys()) | set(val_dist.keys())
    
    inconsistent = False
    
    for label in sorted(all_labels):
        train_pct = train_dist.get(label, 0.0)
        val_pct = val_dist.get(label, 0.0)
        diff = abs(train_pct - val_pct)
        
        status = "✓" if diff <= threshold else "⚠️"
        
        print(f"{status} Класс {label}: Train={train_pct:.2f}%, Val={val_pct:.2f}%, Diff={diff:.2f}%")
        
        if diff > threshold:
            inconsistent = True
    
    if inconsistent:
        print(f"\n⚠️  WARNING: Обнаружены различия >={threshold}% между train и val!")
        print("   Это может указывать на проблемы с разделением данных.")
    else:
        print(f"\n✓ Распределения train и val консистентны (различия <{threshold}%)")


def save_metadata(output_path: Path, train_result, val_result=None):
    """
    Сохраняет метаданные анализа в JSON-файл.
    
    Args:
        output_path: Путь для сохранения metadata.json
        train_result: Результат analyze_labels для train
        val_result: Результат analyze_labels для val (опционально)
    """
    metadata = {
        "train": {
            "distribution": train_result["distribution"],
            "imbalance_ratio": train_result["imbalance_ratio"]
        }
    }
    
    if val_result is not None:
        metadata["val"] = {
            "distribution": val_result["distribution"],
            "imbalance_ratio": val_result["imbalance_ratio"]
        }
    
    metadata_path = output_path / "label_analysis_metadata.json"
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Метаданные сохранены: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Анализ распределения меток и матрицы переходов"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Путь к директории с parquet-файлами (train.parquet, val.parquet)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Путь для сохранения графиков и метаданных (по умолчанию: data_path)"
    )
    parser.add_argument(
        "--consistency_threshold",
        type=float,
        default=5.0,
        help="Порог различия в %% для предупреждения о несоответствии train/val (по умолчанию: 5.0)"
    )
    
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    
    if not data_path.exists():
        print(f"❌ Ошибка: Путь не существует: {data_path}")
        sys.exit(1)
    
    # Определяем output_dir
    output_dir = Path(args.output_dir) if args.output_dir else data_path
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("=== Анализ распределения меток ===")
    print("="*60)
    print(f"Путь к данным: {data_path}")
    print(f"Путь для сохранения: {output_dir}")
    print("="*60)
    
    # Загружаем данные
    train_df, val_df = load_parquet_data(data_path)
    
    if train_df is None and val_df is None:
        print("\n❌ Ошибка: Не найдено ни одного parquet-файла!")
        sys.exit(1)
    
    # Анализируем train
    train_result = None
    if train_df is not None:
        print("\n" + "="*60)
        print("=== Анализ TRAIN ===")
        print("="*60)
        train_result = analyze_labels(
            train_df, 
            output_dir=output_dir / "train",
            save_plots=True
        )
    
    # Анализируем val
    val_result = None
    if val_df is not None:
        print("\n" + "="*60)
        print("=== Анализ VAL ===")
        print("="*60)
        val_result = analyze_labels(
            val_df,
            output_dir=output_dir / "val",
            save_plots=True
        )
    
    # Проверяем консистентность
    if train_result is not None and val_result is not None:
        check_consistency(train_result, val_result, args.consistency_threshold)
    
    # Сохраняем метаданные
    if train_result is not None:
        save_metadata(output_dir, train_result, val_result)
    
    print("\n" + "="*60)
    print("✓ Анализ завершен успешно!")
    print("="*60)


if __name__ == "__main__":
    main()
