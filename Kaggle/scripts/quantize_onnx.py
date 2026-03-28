#!/usr/bin/env python3
"""
Скрипт квантования ONNX модели из FP32 в INT8 с автоматическим контролем качества.

Использует статическое квантование (PTQ) с калибровкой на валидационной выборке.
Автоматически исключает проблемные узлы при падении качества (MCC).

Задача №157: Экспорт квантованной модели (INT8) в ONNX
"""

import argparse
import time
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
from typing import List, Tuple, Optional
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat
from sklearn.metrics import matthews_corrcoef
import sys

# Добавляем путь к src для импорта модулей
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset import LOBCalibrationDataReader, get_val_loader


def evaluate_model_mcc(
    model_path: str,
    data_path: str,
    symbol: str,
    seq_len: int = 100,
    n_past_returns: int = 0,
    batch_size: int = 256,
    max_batches: Optional[int] = None
) -> float:
    """
    Вычисляет MCC (Matthews Correlation Coefficient) для ONNX модели.
    
    Args:
        model_path: путь к ONNX модели
        data_path: путь к директории с данными
        symbol: торговый символ
        seq_len: длина последовательности
        n_past_returns: количество past returns каналов
        batch_size: размер батча
        max_batches: максимальное количество батчей для оценки (None = все)
    
    Returns:
        float: значение MCC
    """
    print(f"[Evaluate] Loading model: {model_path}")
    
    # Создаем ONNX Runtime сессию
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # Загружаем валидационные данные
    val_loader = get_val_loader(
        data_path=data_path,
        symbol=symbol,
        seq_len=seq_len,
        n_past_returns=n_past_returns,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Для стабильности используем 0
        data_mode="memory",
        val_split=0.8
    )
    
    # Собираем предсказания и истинные метки
    all_preds = []
    all_labels = []
    
    for batch_idx, batch in enumerate(val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        
        x, y, _, _, _ = batch
        
        # Инференс
        outputs = session.run(None, {input_name: x.numpy()})
        logits = outputs[0]  # (batch_size, num_classes)
        
        # Предсказания
        preds = np.argmax(logits, axis=1)
        
        all_preds.extend(preds)
        all_labels.extend(y.numpy())
    
    # Вычисляем MCC
    mcc = matthews_corrcoef(all_labels, all_preds)
    print(f"[Evaluate] MCC: {mcc:.4f}")
    
    return mcc


def benchmark_model(
    model_path: str,
    data_path: str,
    symbol: str,
    seq_len: int = 100,
    n_past_returns: int = 0,
    warmup_runs: int = 1000,
    measurement_runs: int = 1000
) -> Tuple[float, float]:
    """
    Бенчмарк производительности ONNX модели.
    
    Args:
        model_path: путь к ONNX модели
        data_path: путь к директории с данными
        symbol: торговый символ
        seq_len: длина последовательности
        n_past_returns: количество past returns каналов
        warmup_runs: количество прогонов для прогрева
        measurement_runs: количество прогонов для замера
    
    Returns:
        Tuple[float, float]: (mean_latency_ms, std_latency_ms)
    """
    print(f"[Benchmark] Loading model: {model_path}")
    
    # Создаем ONNX Runtime сессию
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # Загружаем один батч данных для бенчмарка
    val_loader = get_val_loader(
        data_path=data_path,
        symbol=symbol,
        seq_len=seq_len,
        n_past_returns=n_past_returns,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        data_mode="memory",
        val_split=0.8
    )
    
    # Берем первый батч
    x, _, _, _, _ = next(iter(val_loader))
    input_data = x.numpy()
    
    # Warmup
    print(f"[Benchmark] Warmup: {warmup_runs} runs...")
    for _ in range(warmup_runs):
        session.run(None, {input_name: input_data})
    
    # Measurement
    print(f"[Benchmark] Measurement: {measurement_runs} runs...")
    latencies = []
    for _ in range(measurement_runs):
        start = time.perf_counter()
        session.run(None, {input_name: input_data})
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # в миллисекундах
    
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    
    print(f"[Benchmark] Mean latency: {mean_latency:.3f} ms ± {std_latency:.3f} ms")
    
    return mean_latency, std_latency


def get_problematic_nodes(model_path: str) -> List[str]:
    """
    Возвращает список потенциально проблемных узлов для квантования.
    Обычно это Softmax, LayerNorm и финальный Linear слой.
    
    Args:
        model_path: путь к ONNX модели
    
    Returns:
        List[str]: список имен узлов для исключения
    """
    model = onnx.load(model_path)
    problematic_nodes = []
    
    # Ищем Softmax, LayerNorm узлы
    for node in model.graph.node:
        if node.op_type in ['Softmax', 'LayerNorm']:
            problematic_nodes.append(node.name)
    
    # Ищем финальный Linear/MatMul слой (последний перед выходом)
    output_names = [output.name for output in model.graph.output]
    for node in reversed(model.graph.node):
        if node.op_type in ['MatMul', 'Gemm']:
            # Проверяем, что выход этого узла идет в output
            if any(out in output_names for out in node.output):
                problematic_nodes.append(node.name)
                break
    
    return problematic_nodes


def quantize_model(
    input_model_path: str,
    output_model_path: str,
    data_path: str,
    symbol: str,
    seq_len: int = 100,
    n_past_returns: int = 0,
    n_calibration_samples: int = 1000,
    nodes_to_exclude: Optional[List[str]] = None
) -> None:
    """
    Выполняет статическое квантование ONNX модели.
    
    Args:
        input_model_path: путь к FP32 модели
        output_model_path: путь для сохранения INT8 модели
        data_path: путь к директории с данными
        symbol: торговый символ
        seq_len: длина последовательности
        n_past_returns: количество past returns каналов
        n_calibration_samples: количество снапшотов для калибровки
        nodes_to_exclude: список узлов для исключения из квантования
    """
    print(f"[Quantize] Input model: {input_model_path}")
    print(f"[Quantize] Output model: {output_model_path}")
    print(f"[Quantize] Calibration samples: {n_calibration_samples}")
    
    if nodes_to_exclude:
        print(f"[Quantize] Nodes to exclude: {nodes_to_exclude}")
    
    # Создаем калибровочный загрузчик данных
    calibration_data_reader = LOBCalibrationDataReader(
        onnx_model_path=input_model_path,
        data_path=data_path,
        symbol=symbol,
        seq_len=seq_len,
        n_past_returns=n_past_returns,
        n_samples=n_calibration_samples,
        val_split=0.8
    )
    
    # Выполняем квантование
    print("[Quantize] Starting quantization...")
    quantize_static(
        model_input=input_model_path,
        model_output=output_model_path,
        calibration_data_reader=calibration_data_reader,
        quant_format=QuantFormat.QDQ,  # QDQ формат для лучшей совместимости
        activation_type=QuantType.QInt8,  # Symmetric для активаций
        weight_type=QuantType.QUInt8,  # Asymmetric для весов
        per_channel=True,  # Per-channel квантование для весов (критично для Трансформеров!)
        reduce_range=False,  # Не нужно для современных CPU с VNNI
        nodes_to_exclude=nodes_to_exclude or []
    )
    
    print(f"[Quantize] Quantized model saved to: {output_model_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Квантование ONNX модели из FP32 в INT8 с автоматическим контролем качества"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Путь к FP32 ONNX модели"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Путь для сохранения INT8 ONNX модели"
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
        "--max_mcc_drop",
        type=float,
        default=0.02,
        help="Максимально допустимое падение MCC (default: 0.02)"
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=100,
        help="Длина последовательности (default: 100)"
    )
    parser.add_argument(
        "--n_past_returns",
        type=int,
        default=0,
        help="Количество past returns каналов (default: 0)"
    )
    parser.add_argument(
        "--n_calibration_samples",
        type=int,
        default=1000,
        help="Количество снапшотов для калибровки (default: 1000)"
    )
    parser.add_argument(
        "--skip_benchmark",
        action="store_true",
        help="Пропустить бенчмарк производительности"
    )
    parser.add_argument(
        "--skip_quality_check",
        action="store_true",
        help="Пропустить проверку качества (Auto-Exclude)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ONNX Model Quantization (FP32 -> INT8)")
    print("=" * 80)
    
    # 1. Замер baseline MCC на FP32 модели
    if not args.skip_quality_check:
        print("\n[Step 1/5] Evaluating baseline MCC on FP32 model...")
        baseline_mcc = evaluate_model_mcc(
            model_path=args.input,
            data_path=args.data_path,
            symbol=args.symbol,
            seq_len=args.seq_len,
            n_past_returns=args.n_past_returns,
            max_batches=50  # Ограничиваем для скорости
        )
    else:
        baseline_mcc = None
        print("\n[Step 1/5] Skipping baseline MCC evaluation...")
    
    # 2. Выполнение квантования
    print("\n[Step 2/5] Quantizing model...")
    nodes_to_exclude = []
    attempt = 1
    max_attempts = 3
    
    while attempt <= max_attempts:
        print(f"\n--- Quantization Attempt {attempt}/{max_attempts} ---")
        
        # Квантуем модель
        quantize_model(
            input_model_path=args.input,
            output_model_path=args.output,
            data_path=args.data_path,
            symbol=args.symbol,
            seq_len=args.seq_len,
            n_past_returns=args.n_past_returns,
            n_calibration_samples=args.n_calibration_samples,
            nodes_to_exclude=nodes_to_exclude if nodes_to_exclude else None
        )
        
        # 3. Проверка качества (Auto-Exclude)
        if not args.skip_quality_check:
            print("\n[Step 3/5] Evaluating quantized model MCC...")
            quantized_mcc = evaluate_model_mcc(
                model_path=args.output,
                data_path=args.data_path,
                symbol=args.symbol,
                seq_len=args.seq_len,
                n_past_returns=args.n_past_returns,
                max_batches=50
            )
            
            mcc_drop = baseline_mcc - quantized_mcc
            print(f"\n[Quality Check] Baseline MCC: {baseline_mcc:.4f}")
            print(f"[Quality Check] Quantized MCC: {quantized_mcc:.4f}")
            print(f"[Quality Check] MCC Drop: {mcc_drop:.4f} (threshold: {args.max_mcc_drop:.4f})")
            
            if mcc_drop > args.max_mcc_drop:
                print(f"\n[Warning] MCC drop ({mcc_drop:.4f}) exceeds threshold ({args.max_mcc_drop:.4f})")
                
                if attempt < max_attempts:
                    # Добавляем проблемные узлы в список исключений
                    problematic_nodes = get_problematic_nodes(args.input)
                    
                    if not problematic_nodes:
                        print("[Error] No problematic nodes found to exclude. Stopping.")
                        break
                    
                    # Добавляем по одному узлу за раз
                    next_node = problematic_nodes[min(attempt - 1, len(problematic_nodes) - 1)]
                    if next_node not in nodes_to_exclude:
                        nodes_to_exclude.append(next_node)
                    
                    print(f"[Auto-Exclude] Adding node to exclusion list: {next_node}")
                    print(f"[Auto-Exclude] Current exclusion list: {nodes_to_exclude}")
                    print(f"[Auto-Exclude] Retrying quantization...")
                    
                    attempt += 1
                    continue
                else:
                    print(f"[Error] Max attempts ({max_attempts}) reached. Quality threshold not met.")
                    print(f"[Suggestion] Consider:")
                    print(f"  1. Increasing --max_mcc_drop threshold")
                    print(f"  2. Using more calibration samples (--n_calibration_samples)")
                    print(f"  3. Manually excluding more nodes")
                    break
            else:
                print(f"[Success] MCC drop within acceptable range!")
                break
        else:
            print("\n[Step 3/5] Skipping quality check...")
            break
    
    # 4. Бенчмарк производительности
    if not args.skip_benchmark:
        print("\n[Step 4/5] Benchmarking performance...")
        
        print("\n--- FP32 Model ---")
        fp32_mean, fp32_std = benchmark_model(
            model_path=args.input,
            data_path=args.data_path,
            symbol=args.symbol,
            seq_len=args.seq_len,
            n_past_returns=args.n_past_returns,
            warmup_runs=1000,
            measurement_runs=1000
        )
        
        print("\n--- INT8 Model ---")
        int8_mean, int8_std = benchmark_model(
            model_path=args.output,
            data_path=args.data_path,
            symbol=args.symbol,
            seq_len=args.seq_len,
            n_past_returns=args.n_past_returns,
            warmup_runs=1000,
            measurement_runs=1000
        )
        
        speedup = fp32_mean / int8_mean
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)
        print(f"FP32 Model: {fp32_mean:.3f} ms ± {fp32_std:.3f} ms")
        print(f"INT8 Model: {int8_mean:.3f} ms ± {int8_std:.3f} ms")
        print(f"Speedup: {speedup:.2f}x")
        print("=" * 80)
    else:
        print("\n[Step 4/5] Skipping benchmark...")
    
    # 5. Финальный отчет
    print("\n[Step 5/5] Quantization complete!")
    print(f"\nOutput model: {args.output}")
    
    if nodes_to_exclude:
        print(f"Excluded nodes: {nodes_to_exclude}")
    
    print("\nNext steps:")
    print(f"  1. Update config.toml: model_path = \"model/lit_int8.onnx\"")
    print(f"  2. Verify ONNX Runtime settings in src/ml/onnx.rs")
    print(f"  3. Test inference with the quantized model")


if __name__ == "__main__":
    main()
