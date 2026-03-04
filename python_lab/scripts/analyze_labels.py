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
from features import FeatureEngineer
from labels import Labeler


def load_parquet_data(data_path: Path, horizon: int = 100, threshold: float = 0.0005):
    """
    Загружает train.parquet и val.parquet из указанной директории.
    Если колонки 'label' нет, рассчитывает её на лету.
    
    Args:
        data_path: Путь к директории с parquet-файлами
        horizon: Горизонт для Labeler
        threshold: Порог для Labeler
    
    Returns:
        tuple: (train_df, val_df) или (None, None) если файлы не найдены
    """
    train_path = data_path / "train.parquet"
    val_path = data_path / "val.parquet"
    
    def process_file(path: Path, name: str):
        if not path.exists():
            print(f"[WARN] {name} не найден: {path}")
            return None
        
        print(f"[INFO] Загружаем {name} из {path}")
        df = pl.read_parquet(path)
        
        if "label" not in df.columns:
            print(f"  [WARN] Колонка 'label' не найдена в {name}. Генерируем на лету (h={horizon}, t={threshold})...")
            # 1. Расчет mid_price и базовых признаков
            fe = FeatureEngineer(n_levels=50)
            df = fe.transform(df)
            
            # 2. Расчет меток
            labeler = Labeler(horizon=horizon, threshold=threshold)
            df = labeler.add_labels(df)
            print(f"  [INFO] Метки для {name} сгенерированы. Размер после разметки: {len(df)}")
        else:
            print(f"  [INFO] Размер {name}: {len(df)} строк")
            
        return df

    train_df = process_file(train_path, "train.parquet")
    val_df = process_file(val_path, "val.parquet")
    
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
        
        status = "[OK]" if diff <= threshold else "[WARN]"
        
        print(f"{status} Класс {label}: Train={train_pct:.2f}%, Val={val_pct:.2f}%, Diff={diff:.2f}%")
        
        if diff > threshold:
            inconsistent = True
    
    if inconsistent:
        print(f"\n[WARN] WARNING: Обнаружены различия >={threshold}% между train и val!")
        print("   Это может указывать на проблемы с разделением данных.")
    else:
        print(f"\n[OK] Распределения train и val консистентны (различия <{threshold}%)")


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
    
    print(f"\n[OK] Метаданные сохранены: {metadata_path}")


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
    parser.add_argument(
        "--horizon",
        type=int,
        default=100,
        help="Горизонт предсказания (K) для генерации меток (default: 100)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0005,
        help="Порог доходности для меток (default: 0.0005)"
    )
    
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    
    if not data_path.exists():
        print(f"[ERROR] Ошибка: Путь не существует: {data_path}")
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
    train_df, val_df = load_parquet_data(data_path, horizon=args.horizon, threshold=args.threshold)
    
    if train_df is None and val_df is None:
        print("\n[ERROR] Ошибка: Не найдено ни одного parquet-файла!")
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
    print("[OK] Анализ завершен успешно!")
    print("="*60)


if __name__ == "__main__":
    main()
