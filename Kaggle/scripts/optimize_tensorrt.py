import onnxruntime as ort
import json
import argparse
import time
import numpy as np
from pathlib import Path
import sys

def check_dependencies():
    """Проверяет наличие необходимых библиотек."""
    try:
        import tensorrt as trt
        trt_version = trt.__version__
        print(f"✅ TensorRT version: {trt_version}")
        major = int(trt_version.split('.')[0])
        if major < 10:
            print(f"⚠️ Warning: TensorRT version {trt_version} is < 10.0. Recommended version is 10.0+")
    except ImportError:
        print("⚠️ Warning: tensorrt python package not found. Optimization might fail if ONNX Runtime cannot find TRT libraries.")

def optimize(onnx_path, opt_level=3):
    check_dependencies()
    # 1. Читаем размерность из метаданных (056)
    onnx_path = Path(onnx_path)
    meta_path = onnx_path.parent / "metadata.json"
    
    if not meta_path.exists():
        print(f"❌ Error: metadata.json not found at {meta_path}")
        sys.exit(1)
        
    with open(meta_path) as f:
        meta = json.load(f)
    
    if 'model_params' not in meta:
        print(f"❌ Error: 'model_params' not found in {meta_path}")
        sys.exit(1)
        
    params = meta['model_params']
    seq_len = params['seq_len']
    n_levels = params['n_levels']
    n_channels = params['in_channels']
    
    # 3D профиль: Batch x Seq x Features (Features = Channels * Levels)
    # Это соответствует ожиданиям Rust-кода и логике патчинга
    feats = n_channels * n_levels
    shape_str = f"input:1x{seq_len}x{feats}" # Жесткий профиль для Batch=1
    
    # Кэш сохраняем рядом с моделью
    cache_dir = onnx_path.parent / "trt_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 2. Настройка TensorRT провайдера
    providers = [
        ('TensorrtExecutionProvider', {
            'device_id': 0,
            'trt_max_workspace_size': 1073741824, # 1GB
            'trt_fp16_enable': True,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': str(cache_dir),
            'trt_builder_optimization_level': opt_level,
            # Фиксируем профиль, чтобы исключить runtime overhead
            'trt_profile_min_shape': shape_str,
            'trt_profile_opt_shape': shape_str,
            'trt_profile_max_shape': shape_str,
            'trt_dla_enable': False,
        }),
        'CUDAExecutionProvider'
    ]

    print(f"🚀 Building TRT Engine with profile {shape_str}...")
    try:
        # ort.InferenceSession инициализирует билд движка при первом запуске с этими провайдерами
        session = ort.InferenceSession(str(onnx_path), providers=providers)
    except Exception as e:
        print(f"❌ Error during session creation: {e}")
        sys.exit(1)

    # 3. Прогрев (Warm-up) - 10 прогонов для стабилизации кеша и выбора ядер
    dummy_input = np.random.randn(1, seq_len, feats).astype(np.float32)
    latencies = []
    
    for i in range(10):
        start = time.perf_counter()
        session.run(None, {'input': dummy_input})
        latency = time.perf_counter() - start
        latencies.append(latency)
        print(f"  Warm-up run {i+1}/10: {latency*1000:.3f}ms")
    
    print(f"✓ Optimization complete. Avg Latency: {np.mean(latencies)*1000:.3f}ms")
    print(f"📁 Engine cache saved to: {cache_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize ONNX model using TensorRT")
    parser.add_argument("--onnx_path", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--opt_level", type=int, default=3, help="TRT builder optimization level (default: 3)")
    
    args = parser.parse_args()
    optimize(args.onnx_path, args.opt_level)
