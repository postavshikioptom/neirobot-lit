# Задача 128: Оптимизация и квантование модели в FP16 (v2.0)

## 1. Конвертация в `python_lab/scripts/export_onnx.py`
Вместо кастомных конвертеров используем `onnxsim`, который корректно схлопывает узлы и выполняет `Cast` в FP16, сохраняя стабильность графа.

```python
import onnx
from onnxsim import simplify

def convert_to_fp16(input_path, output_path):
    """
    Использует onnx-simplifier для конвертации графа в FP16.
    """
    # Загружаем модель
    model = onnx.load(input_path)
    
    # Simplify + FP16 conversion
    # skip_fuse_bn=False позволяет эффективно сжать слои нормализации
    model_opt, check = simplify(model, convert_fp16=True)
    
    if not check:
        raise RuntimeError("Simplified ONNX model could not be validated")
        
    onnx.save(model_opt, output_path)
    print(f"✓ Model optimized and converted to FP16: {output_path}")
```

## 2. Валидация стабильности (NaN/Precision check)
В глубоких трансформерах переход в FP16 часто вызывает `NaN` в слоях `LayerNorm`. Gemini должен реализовать проверку выходов.

```python
import numpy as np
import onnxruntime as ort

def validate_precision(fp32_path, fp16_path, dummy_input):
    s32 = ort.InferenceSession(fp32_path)
    s16 = ort.InferenceSession(fp16_path)
    
    out32 = s32.run(None, {'input': dummy_input})[0]
    out16 = s16.run(None, {'input': dummy_input})[0]
    
    # Проверка на NaN
    if np.isnan(out16).any():
        print("❌ CRITICAL: FP16 model produced NaN. Fallback to FP32 recommended.")
        return False
        
    # Расчет MSE
    mse = np.mean((out32 - out16.astype(np.float32))**2)
    print(f"FP16 vs FP32 MSE: {mse:.2e}")
    
    return mse < 1e-3
```

## 3. Обновление метаданных (`metadata.json`)
Согласно архитектуре (056), при экспорте в `metadata.json` должно записываться поле `precision`:

```json
{
  "model_name": "lit_transformer_v1",
  "precision": "fp16",
  "onnx_opset": 18,
  "input_shape": [1, 50, 40]
}
```

## 4. Интеграция в Rust (`src/ml/onnx.rs`)
Обнови конфигурацию провайдера для использования аппаратного ускорения FP16 на GPU (TensorRT/CUDA).

```rust
// Использование ort 2.0 API
let mut session_builder = Session::builder()?;

if model_path.contains("_fp16") {
    // Включаем поддержку fp16 на уровне провайдера исполнения
    session_builder = session_builder.with_execution_providers(vec![
        CUDAExecutionProvider::default()
            .with_fp16(true) 
            .build()
    ])?;
    tracing::info!("ONNX: FP16 acceleration enabled for CUDA");
}
```

## 5. Инструкции для Gemini (Coder AI):
1.  **Dependencies**: Добавить `onnx-simplifier` в `requirements.txt`.
2.  **Scripts**: Обновить `python_lab/scripts/export_onnx.py`. Добавить аргумент `--fp16`.
3.  **Validation**: Интегрировать функцию `validate_precision` в процесс экспорта. Если валидация не проходит — не сохранять FP16 файл и выводить ошибку.
4.  **Rust**: В [./src/ml/onnx.rs](./src/ml/onnx.rs) реализовать детекцию суффикса `_fp16` и настройку `CUDAExecutionProvider`.

**Результат**: Мы получаем безопасный и быстрый переход на FP16, который экономит 50% видеопамяти и сокращает время инференса без риска получить `NaN` в продакшене.

