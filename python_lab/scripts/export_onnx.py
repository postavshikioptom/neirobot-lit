import torch
import torch.onnx
import json
import argparse
import subprocess
import datetime
import math
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
from python_lab.src.train import LiTModule

# Для оптимизации и FP16
try:
    import onnxsim
except ImportError:
    onnxsim = None

try:
    from onnxconverter_common import float16
except ImportError:
    float16 = None

def get_git_hash():
    """Возвращает текущий git hash коммита."""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"

def validate_normalization(params):
    """Проверяет параметры нормализации на наличие NaN/Inf."""
    for feat, values in params.items():
        mean = values.get("mean", 0.0)
        std = values.get("std", 1.0)
        if not (math.isfinite(mean) and math.isfinite(std)):
            raise ValueError(f"Invalid normalization params for {feat}: mean={mean}, std={std}")
        if std <= 0:
            raise ValueError(f"Standard deviation must be positive for {feat}, got {std}")

def convert_to_fp16(input_path, output_path):
    """
    Использует onnx-simplifier для конвертации графа в FP16.
    """
    if onnxsim is None:
        print("❌ Error: onnxsim not installed.")
        return False
    
    print(f"Converting {input_path} to FP16 using onnxsim...")
    model = onnx.load(str(input_path))
    
    # Simplify + FP16 conversion
    model_opt, check = onnxsim.simplify(model, convert_fp16=True)
    
    if not check:
        print("❌ Error: Simplified ONNX model could not be validated")
        return False
        
    onnx.save(model_opt, str(output_path))
    print(f"✓ Model optimized and converted to FP16: {output_path}")
    return True

def validate_precision(fp32_path, fp16_path, dummy_input):
    """Сравнивает точность ONNX FP16 модели с ONNX FP32."""
    print("Verifying FP16 model precision vs FP32...")
    
    providers = ['CPUExecutionProvider']
    s32 = ort.InferenceSession(str(fp32_path), providers=providers)
    s16 = ort.InferenceSession(str(fp16_path), providers=providers)
    
    input_name = s32.get_inputs()[0].name
    ort_inputs = {input_name: dummy_input.numpy().astype(np.float32)}
    
    out32 = s32.run(None, ort_inputs)[0]
    out16 = s16.run(None, ort_inputs)[0]
    
    # Проверка на NaN
    if np.isnan(out16).any():
        print("❌ CRITICAL: FP16 model produced NaN. Fallback to FP32 recommended.")
        return False
        
    # Расчет MSE
    mse = np.mean((out32 - out16.astype(np.float32))**2)
    print(f"  FP16 vs FP32 MSE: {mse:.2e}")
    
    if mse >= 1e-3:
        print(f"❌ FP16 deviation is too high (MSE: {mse:.2e} >= 1e-3)")
        return False
        
    print("✓ FP16 Verification successful.")
    return True

def export():
    parser = argparse.ArgumentParser(description="Export LiT model to ONNX with FP16 support")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint file")
    parser.add_argument("--config_path", type=str, required=True, help="Path to config file (YAML/TOML)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save exported files")
    parser.add_argument("--fp16", action="store_true", help="Convert model to FP16 for faster inference")
    parser.add_argument("--signal_only", action="store_true", default=True, help="Export only logits head (exclude volatility)")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    config_path = Path(args.config_path)
    models_dir = Path(args.output_dir)
    
    # Создаем директорию вывода, если её нет
    models_dir.mkdir(parents=True, exist_ok=True)
    
    norm_params_path = models_dir / "norm_params.json"
    output_onnx = models_dir / "model.onnx"
    output_metadata = models_dir / "metadata.json"

    # 1. Проверка существования файлов
    if not model_path.exists():
        print(f"Error: Model checkpoint not found at {model_path}")
        return
    
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return

    print(f"Loading checkpoint: {model_path}")

    # 2. Загрузка модели
    try:
        module = LiTModule.load_from_checkpoint(str(model_path), map_location="cpu")
        module.eval()
        module.freeze()
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    hparams = module.hparams
    seq_len = hparams.seq_len
    d_model = hparams.get("d_model", 64)
    nhead = hparams.get("nhead", 4)
    num_layers = hparams.get("num_layers", 2)
    past_returns_lags = hparams.get("past_returns_lags", [])
    n_channels = 3 + len(past_returns_lags)
    
    # Динамически извлекаем n_levels из модели (подзадача 2)
    n_levels = hparams.get("n_levels", 50)
    
    # Динамически извлекаем patch_size из модели (подзадача 3)
    patch_size = hparams.get("patch_size", None)
    
    # Динамически определяем output_classes (подзадача 4)
    output_classes = hparams.get("num_classes", hparams.get("output_classes", 3))

    # 3. Загрузка параметров нормализации
    if not norm_params_path.exists():
        print(f"Error: Normalization params not found at {norm_params_path}")
        return
    
    with open(norm_params_path, 'r') as f:
        norm_params = json.load(f)
    
    try:
        validate_normalization(norm_params)
    except ValueError as e:
        print(f"Validation failed: {e}")
        return

    # 4. Формирование списков mean/std и robust параметров (Задача 240)
    means = []
    stds = []
    medians = []  # Задача 240
    iqrs = []  # Задача 240
    order = ["p", "v", "i"]
    for prefix in order:
        for i in range(n_levels):
            feat_name = f"feat_{prefix}_{i}"
            if feat_name not in norm_params:
                print(f"Error: Missing normalization params for {feat_name}")
                return
            means.append(norm_params[feat_name]["mean"])
            stds.append(norm_params[feat_name]["std"])
            # Задача 240: Извлекаем медиану и IQR (с fallback на None если не существуют)
            medians.append(norm_params[feat_name].get("median", None))
            iqrs.append(norm_params[feat_name].get("iqr", None))

    # 5. Экспорт в ONNX (FP32)
    # Обертка для модели, чтобы она принимала 3D тензор [1, seq_len, features]
    # и преобразовывала его в 4D [1, seq_len, channels, levels], который ожидает LiTModel
    class ModelWrapper(torch.nn.Module):
        def __init__(self, base_model, n_channels, n_levels, signal_only=True):
            super().__init__()
            self.base_model = base_model
            self.n_channels = n_channels
            self.n_levels = n_levels
            self.signal_only = signal_only
            
        def forward(self, x):
            # x: [1, seq_len, n_channels * n_levels] -> [1, seq_len, n_channels, n_levels]
            x = x.view(x.shape[0], x.shape[1], self.n_channels, self.n_levels)
            outputs = self.base_model(x)
            
            if self.signal_only:
                # Возвращаем только логиты (первый элемент кортежа (logits, vol))
                return outputs[0]
            return outputs

    wrapped_model = ModelWrapper(module.model, n_channels, n_levels, signal_only=args.signal_only)
    dummy_input_3d = torch.randn(1, seq_len, n_channels * n_levels)
    
    # Определяем имена выходов
    output_names = ['logits'] if args.signal_only else ['logits', 'volatility']
    
    print(f"Exporting ONNX to {output_onnx} (opset 18), input shape: {dummy_input_3d.shape}...")
    torch.onnx.export(
        wrapped_model,
        dummy_input_3d,
        str(output_onnx),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=output_names
    )

    # ... (пропуск упрощения и FP16) ...

    # Упрощение графа через onnxsim
    if onnxsim is not None:
        print("Simplifying ONNX graph...")
        try:
            model_simp, check = onnxsim.simplify(str(output_onnx))
            if check:
                onnx.save(model_simp, str(output_onnx))
                print("✓ ONNX simplified")
            else:
                print("⚠️  ONNX simplification check failed, keeping original")
        except Exception as e:
            print(f"⚠️  Simplification failed: {e}")

    # Задача 098: Фиксация динамических размерностей
    # Post-export Fix: Если модель была экспортирована с динамическими осями ранее,
    # добавить шаг принудительной фиксации
    try:
        from onnxruntime.tools import make_dynamic_shape_fixed
        print("Fixing dynamic axes in exported ONNX model...")
        fixed_output = models_dir / "model_fixed.onnx"
        make_dynamic_shape_fixed.make_dynamic_shape_fixed('batch', 1, str(output_onnx), str(fixed_output))
        output_onnx = fixed_output
        print("✓ Dynamic batch dimension fixed to 1")
    except ImportError:
        print("⚠️  onnxruntime.tools not available, skipping dynamic shape fix")
    except Exception as e:
        print(f"⚠️  Failed to fix dynamic shapes: {e}")

    # 6. Квантование FP16
    final_precision = "fp32"
    if args.fp16:
        fp16_onnx = models_dir / "model.fp16.onnx"
        if convert_to_fp16(output_onnx, fp16_onnx):
            if validate_precision(output_onnx, fp16_onnx, dummy_input_3d):
                output_onnx = fp16_onnx
                final_precision = "fp16"
            else:
                print("❌ FP16 validation failed. Falling back to FP32.")
                if fp16_onnx.exists():
                    fp16_onnx.unlink()

    # 7. Сборка расширенных метаданных
    # Температура для калибровки (по умолчанию 1.0, если не задана в чекпоинте)
    temperature = hparams.get("temperature", 1.0)
    
    # Задача 240: Параметры нормализации
    norm_params_metadata = {
        "scaler_type": hparams.get("scaler_type", "zscore"),
        "mean": means,
        "std": stds,
        "median": medians,
        "iqr": iqrs,
        "winsor_limits": hparams.get("winsor_limits", None)
    }
    
    metadata = {
        "metadata_version": "1.1.0",
        "git_hash": get_git_hash(),
        "export_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "temperature": temperature,
        "temperature_embedded": False,
        "model_name": "LiT",
        "model_params": {
            "architecture": "LiT-Transformer",
            "seq_len": seq_len,
            "n_levels": n_levels,
            "in_channels": n_channels,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "patch_size": patch_size,
            "feature_order": ["price", "volume", "imbalance"],
            "output_classes": output_classes,
            "label_map": {"0": "Flat", "1": "Up", "2": "Down"},
            "precision": final_precision,
            "quantized": args.fp16,
            "onnx_opset": 18
        },
        "normalization": norm_params_metadata,
        "onnx_file": output_onnx.name
    }

    with open(output_metadata, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Extended metadata saved to {output_metadata} (Precision: {final_precision})")
    print("Export completed successfully!")

if __name__ == "__main__":
    export()
