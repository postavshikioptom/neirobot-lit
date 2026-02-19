# Задача 129: Оптимизация инференса через NVIDIA TensorRT (v2.0)

## 1. Скрипт оптимизации `python_lab/scripts/optimize_tensorrt.py`
Скрипт должен создать оптимизированный план (engine) под конкретное железо с фиксированным профилем.

```python
import onnxruntime as ort
import json
import argparse
import time
import numpy as np
from pathlib import Path

def optimize(onnx_path, cache_dir, opt_level=3):
    # 1. Читаем размерность из метаданных (056)
    meta_path = Path(onnx_path).parent / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    
    seq_len = meta['input_shape'][1]
    feats = meta['input_shape'][2]
    shape_str = f"input:1x{seq_len}x{feats}" # Жесткий профиль для Batch=1

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
    session = ort.InferenceSession(onnx_path, providers=providers)

    # 3. Прогрев (Warm-up) - 10 прогонов для стабилизации кеша и выбора ядер
    dummy_input = np.random.randn(1, seq_len, feats).astype(np.float32)
    latencies = []
    for _ in range(10):
        start = time.perf_counter()
        session.run(None, {'input': dummy_input})
        latencies.append(time.perf_counter() - start)
    
    print(f"✓ Optimization complete. Avg Latency: {np.mean(latencies)*1000:.3f}ms")
```

## 2. Интеграция в Rust (`src/ml/onnx.rs`)
Кэш должен быть изолирован для каждого торгового инструмента (символа), чтобы избежать конфликтов при одновременном запуске нескольких ботов.

```rust
// В src/ml/onnx.rs
let symbol = config.symbol.clone();
let cache_path = format!("bots/{}/model/trt_cache", symbol);

let trt_options = TensorRTExecutionProvider::default()
    .with_engine_cache(true)
    .with_engine_cache_path(&cache_path)
    .with_fp16(true)
    .with_builder_optimization_level(3); // Соответствует Python скрипту

session_builder.with_execution_providers(vec![
    trt_options.build(),
    CUDAExecutionProvider::default().build()
])?;
```

## 3. Спорные моменты и корректировки (Grok/Zencoder)

*   **Fixed Profiles**: Полностью согласен с Grok. Использование `trt_profile_min/opt/max_shape` с идентичными значениями — это единственный способ заставить TRT вырезать весь динамический код из графа.
*   **Optimization Level**: Переходим на уровень **3** по умолчанию. Уровень 5 дает выигрыш в микросекунды, но увеличивает время сборки движка до 10 минут на картах серии RTX, что мешает быстрой итерации.
*   **Warm-up**: 10 прогонов обязательны. TensorRT использует первые запуски для замера производительности различных реализаций ядер (kernels) прямо на вашем GPU.
*   **Hardware Lock**: В [./docs/000-architecture.md](./docs/000-architecture.md) необходимо добавить пометку, что папка `trt_cache` **не должна** переноситься между серверами с разными моделями GPU (например, с RTX 3090 на A100).

## 4. Инструкции для Gemini (Coder AI):
1.  **Dependencies**: Проверить наличие `tensorrt>=10.0` в окружении.
2.  **Scripts**: Создать `python_lab/scripts/optimize_tensorrt.py` с аргументами `--onnx_path` и `--opt_level`.
3.  **Metadata**: Скрипт обязан падать с ошибкой, если рядом с `.onnx` нет файла `metadata.json`.
4.  **Rust**: Обновить [./src/ml/onnx.rs](./src/ml/onnx.rs), пробросив путь к кэшу через конфигурацию бота.

**Результат**: Мы получаем «выжатый» до предела инференс, который использует 100% возможностей Tensor Cores, обеспечивая минимально возможный `tick-to-trade` latency.
