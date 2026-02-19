#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работы LOBCalibrationDataReader.

Использование:
    python python_lab/scripts/test_quantization.py \
        --model_path bots/CAKEUSDT/model/lit.onnx \
        --data_path bots/CAKEUSDT/data/raw \
        --symbol CAKEUSDT
"""

import argparse
import sys
from pathlib import Path

# Добавляем путь к src для импорта модулей
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset import LOBCalibrationDataReader


def main():
    parser = argparse.ArgumentParser(
        description="Тест калибровочного загрузчика данных для квантования"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Путь к ONNX модели"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Путь к директории с parquet данными"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Торговый символ (например, BTCUSDT)"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Количество снапшотов для теста (default: 10)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Testing LOBCalibrationDataReader")
    print("=" * 80)
    
    # Создаем калибровочный загрузчик
    print(f"\n[1/3] Creating calibration data reader...")
    reader = LOBCalibrationDataReader(
        onnx_model_path=args.model_path,
        data_path=args.data_path,
        symbol=args.symbol,
        seq_len=100,
        n_past_returns=0,
        n_samples=args.n_samples,
        val_split=0.8
    )
    
    print(f"✓ Reader created successfully")
    print(f"  - Input name: {reader.input_name}")
    print(f"  - Calibration samples: {len(reader.calibration_data)}")
    
    # Тестируем get_next()
    print(f"\n[2/3] Testing get_next() method...")
    batch_count = 0
    while True:
        batch = reader.get_next()
        if batch is None:
            break
        
        batch_count += 1
        
        # Проверяем структуру батча
        assert reader.input_name in batch, f"Input name '{reader.input_name}' not in batch"
        data = batch[reader.input_name]
        
        print(f"  Batch {batch_count}: shape = {data.shape}, dtype = {data.dtype}")
    
    print(f"✓ Processed {batch_count} batches")
    
    # Тестируем rewind()
    print(f"\n[3/3] Testing rewind() method...")
    reader.rewind()
    
    first_batch = reader.get_next()
    assert first_batch is not None, "First batch after rewind is None"
    
    print(f"✓ Rewind successful")
    print(f"  First batch shape: {first_batch[reader.input_name].shape}")
    
    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)
    print("\nYou can now run quantization:")
    print(f"  python python_lab/scripts/quantize_onnx.py \\")
    print(f"      --input {args.model_path} \\")
    print(f"      --output {Path(args.model_path).parent / 'lit_int8.onnx'} \\")
    print(f"      --data_path {args.data_path} \\")
    print(f"      --symbol {args.symbol}")


if __name__ == "__main__":
    main()
