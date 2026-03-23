"""
train_metadata.py — Обновление metadata.json после нормализации.
Вынесено из train.py в рамках задачи 322.4.
"""
import json
import datetime
from datetime import UTC
from pathlib import Path


def update_model_metadata(base_path, symbol, args, winsor_limits, norm_params_path):
    """
    Обновляет или создает metadata.json с параметрами нормализации (Задача 240/056).
    Путь сохранения: bots/<SYMBOL>/models/metadata.json
    """
    metadata_path = Path(base_path) / "bots" / symbol / "models" / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    # Загружаем существующие метаданные или создаем новые
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {
            "metadata_version": "1.1.0",
            "model_name": "LiT",
            "export_timestamp": datetime.datetime.now(UTC).isoformat() + "Z",
        }

    # Загружаем сохраненные параметры нормализации
    norm_params_path = Path(norm_params_path)
    if norm_params_path.exists():
        with open(norm_params_path, 'r') as f:
            norm_data = json.load(f)

        if isinstance(norm_data, dict) and "params" in norm_data:
            params = norm_data["params"]
        else:
            params = norm_data

        metadata["normalization"] = {
            "scaler_type": args.scaler_type,
            "winsor_limits": winsor_limits,
            "params": params
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"[{symbol}] Metadata updated with normalization params at {metadata_path}")
