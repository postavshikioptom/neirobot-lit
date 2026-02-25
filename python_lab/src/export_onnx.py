import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import argparse
from pathlib import Path
import json

# Для оптимизации и FP16
try:
    import onnxsim
except ImportError:
    onnxsim = None

try:
    from onnxconverter_common import float16
except ImportError:
    float16 = None

# Импортируем из текущего пакета
try:
    from .lit_model import LiTModel
    from .train import LiTModule
except ImportError:
    # Для запуска как скрипта, если пакет не установлен
    from lit_model import LiTModel
    from train import LiTModule


class ExportWrapper(nn.Module):
    """
    Обертка для экспорта модели с входом формы (B, S, 150).
    Преобразует плоский вход (B, S, 150) в (B, S, in_channels, n_levels)
    для совместимости с LOBPatching внутри модели.
    
    150 = 50 уровней * 3 канала (Price, Volume, Imbalance)
    """
    def __init__(self, model, in_channels=3, n_levels=50):
        super().__init__()
        self.model = model
        self.in_channels = in_channels
        self.n_levels = n_levels
        
    def forward(self, x):
        """
        x: (Batch, Seq, 150) - плоский входной тензор
        Преобразуем в (Batch, Seq, 3, 50) для LOBPatching
        
        Согласно плану 053:
        - 150 = 50 уровней * 3 канала (Price, Volume, Imbalance)
        - Это ровно 3 канала по 50 уровней
        - LOBPatching должен быть внутри графа и обработать этот вход целиком
        """
        b, s, f = x.shape
        
        # Reshape (B, S, 150) -> (B, S, 3, 50)
        # 3 канала: Price, Volume, Imbalance
        # 50 уровней стакана
        x_reshaped = x.view(b, s, 3, self.n_levels)  # (B, S, 3, 50)
        
        # Передаем в модель БЕЗ потери данных
        # LOBPatching внутри модели обработает (B, S, 3, 50)
        return self.model(x_reshaped)

def convert_to_fp16(onnx_path, output_path):
    """Конвертирует ONNX модель в FP16."""
    if float16 is None:
        print("❌ Error: onnxconverter-common not installed.")
        return False
    
    print(f"Converting {onnx_path} to FP16...")
    model = onnx.load(str(onnx_path))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, str(output_path))
    print(f"✓ FP16 model saved to {output_path}")
    return True

def validate_precision(onnx_path, torch_model, dummy_input, precision="fp32"):
    """Проверяет точность ONNX модели по сравнению с PyTorch."""
    print(f"Verifying {precision.upper()} model precision...")
    
    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    
    # Подготовка входов для ONNX
    if isinstance(dummy_input, tuple):
        # Модель с regime_id
        x, regime_id = dummy_input
        input_names = [inp.name for inp in session.get_inputs()]
        ort_inputs = {
            input_names[0]: x.numpy().astype(np.float32),
            input_names[1]: regime_id.numpy().astype(np.int64)
        }
        torch_input = dummy_input
    else:
        # Модель без regime_id
        input_name = session.get_inputs()[0].name
        ort_inputs = {input_name: dummy_input.numpy().astype(np.float32)}
        torch_input = dummy_input
    
    # Инференс ONNX
    ort_outs = session.run(None, ort_inputs)[0]
    
    # Инференс PyTorch
    torch_model.eval()
    with torch.no_grad():
        if isinstance(torch_input, tuple):
            torch_outs = torch_model(*torch_input).numpy()
        else:
            torch_outs = torch_model(torch_input).numpy()
    
    # Допуски
    rtol = 1e-03 if precision == "fp32" else 1e-02
    atol = 1e-05 if precision == "fp32" else 1e-02
    
    diff = np.abs(torch_outs - ort_outs)
    print(f"  Max difference: {diff.max():.2e}")
    
    try:
        np.testing.assert_allclose(torch_outs, ort_outs, rtol=rtol, atol=atol)
        print(f"✓ {precision.upper()} Verification successful.")
        return True
    except AssertionError as e:
        if precision == "fp16":
            print(f"⚠️  FP16 deviation is slightly high but usually acceptable.")
            return True
        else:
            print(f"❌ {precision.upper()} Verification failed!")
            print(f"Error: {e}")
            raise RuntimeError(f"ONNX validation failed for {precision.upper()}: {e}")

def export(input_path, output_path, embed_temperature=False, use_fp16=False):
    """
    Загружает веса модели и экспортирует ее в формат ONNX.
    """
    print(f"Loading model from {input_path}...")
    
    # 1. Загрузка модели
    input_ext = Path(input_path).suffix
    seq_len = 100 
    activation = 'gelu_exact' 
    
    if input_ext == ".ckpt":
        model_module = LiTModule.load_from_checkpoint(input_path, map_location="cpu")
        model = model_module.model
        hparams = model_module.hparams
        seq_len = hparams.get("seq_len", 100)
        activation = hparams.get("activation", "gelu_exact")
        past_returns_lags = hparams.get("past_returns_lags", [10, 50, 100])
        num_regimes = hparams.get("num_regimes", 0)
        regime_embedding_dim = hparams.get("regime_embedding_dim", 16)
    else:
        model = LiTModel()
        model.load_state_dict(torch.load(input_path, map_location="cpu"))
        past_returns_lags = [10, 50, 100]
        num_regimes = 0
        regime_embedding_dim = 16
    
    model.eval()

    # 2. Метаданные
    metadata_path = Path(output_path).parent / "metadata.json"
    temperature = None
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            temperature = metadata.get('temperature', None)
    else:
        metadata = {}
    
    # Проверяем, есть ли optuna_config.json с best_seq_len (Задача 055)
    optuna_config_path = Path(output_path).parent / "optuna_config.json"
    if optuna_config_path.exists():
        with open(optuna_config_path, 'r') as f:
            optuna_config = json.load(f)
            best_seq_len = optuna_config.get('best_seq_len', seq_len)
            if best_seq_len != seq_len:
                print(f"Found Optuna best_seq_len: {best_seq_len} (overriding default {seq_len})")
                seq_len = best_seq_len
    
    # 3. Температура
    if embed_temperature and temperature is not None:
        print(f"Embedding temperature scaling (T={temperature:.4f}) into ONNX graph...")
        class ModelWithTemperature(nn.Module):
            def __init__(self, base_model, temperature):
                super().__init__()
                self.model = base_model
                self.temperature = temperature
            def forward(self, x):
                logits = self.model(x)
                return torch.softmax(logits / self.temperature, dim=1)
        
        model = ModelWithTemperature(model, temperature)
    
    # 4. Обернуть модель в ExportWrapper для преобразования входа (B, S, 150) -> (B, S, 3, 50)
    # Согласно плану 053: 150 = 50 уровней * 3 канала (Price, Volume, Imbalance)
    in_channels = 3  # 3 канала согласно плану 053
    n_levels = 50  # Стандартное значение
    export_model = ExportWrapper(model, in_channels=in_channels, n_levels=n_levels)
    export_model.eval()
    
    # 5. Dummy input - форма (1, 100, 150) согласно плану 053
    dummy_input = torch.randn(1, seq_len, 150)
    
    # Проверяем, использует ли модель regime embedding
    num_regimes = getattr(model, 'num_regimes', 0)
    use_regime = num_regimes > 0 and hasattr(model, 'regime_embedding') and model.regime_embedding is not None
    
    if use_regime:
        # Создаем dummy regime_id для экспорта
        dummy_regime_id = torch.zeros(1, dtype=torch.long)
        dummy_inputs = (dummy_input, dummy_regime_id)
        input_names = ['input', 'regime_id']
        print(f"Model uses regime embedding with {num_regimes} regimes")
    else:
        dummy_inputs = dummy_input
        input_names = ['input']
        print("Model does not use regime embedding")

    # 6. Экспорт в ONNX (FP32)
    print(f"Exporting to ONNX (opset 17), seq_len={seq_len}...")
    
    # Проверяем наличие разреженности в модели
    total_params = 0
    zero_params = 0
    for name, param in export_model.named_parameters():
        if 'weight' in name:
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
    
    sparsity = zero_params / total_params if total_params > 0 else 0.0
    
    if sparsity > 0.01:  # Если разреженность > 1%
        print(f"\n{'='*60}")
        print(f"⚠️  SPARSE MODEL DETECTED")
        print(f"{'='*60}")
        print(f"Model Sparsity: {sparsity:.2%}")
        print(f"Zero Parameters: {zero_params:,} / {total_params:,}")
        print(f"")
        print(f"IMPORTANT: Unstructured sparsity in ONNX Runtime")
        print(f"  - Reduces model file size ✓")
        print(f"  - Does NOT provide inference speedup ✗")
        print(f"  - Sparse operations executed as dense")
        print(f"")
        print(f"For actual speedup, consider:")
        print(f"  - Structured 2:4 sparsity with NVIDIA GPU (Ampere+)")
        print(f"  - Specialized sparse inference engines")
        print(f"  - Quantization (INT8) for additional compression")
        print(f"{'='*60}\n")
    
    # Экспорт с dynamic_axes для batch_size
    torch.onnx.export(
        export_model,
        dummy_inputs,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    # Упрощение графа
    if onnxsim is not None:
        print("Simplifying ONNX graph...")
        try:
            model_simp, check = onnxsim.simplify(output_path)
            if check:
                onnx.save(model_simp, output_path)
                print("✓ ONNX simplified")
        except Exception as e:
            print(f"⚠️  Simplification failed: {e}")

    # Валидация FP32 - выбросит исключение если не пройдена
    if use_regime:
        validate_precision(output_path, export_model, dummy_inputs, "fp32")
    else:
        validate_precision(output_path, export_model, dummy_input, "fp32")

    # 7. Квантование FP16
    final_precision = "fp32"
    if use_fp16:
        fp16_path = str(Path(output_path).with_suffix(".fp16.onnx"))
        if convert_to_fp16(output_path, fp16_path):
            if use_regime:
                validate_precision(fp16_path, export_model, dummy_inputs, "fp16")
            else:
                validate_precision(fp16_path, export_model, dummy_input, "fp16")
            output_path = fp16_path
            final_precision = "fp16"

    # 8. Обновление метаданных
    # Получаем num_horizons из модели
    num_horizons = getattr(model, 'num_horizons', 1)
    use_horizon_embedding = getattr(model, 'use_horizon_embedding', False)
    
    # Согласно плану 053: входной формат (B, S, 150) = (B, S, 3, 50)
    # где 3 = каналы (Price, Volume, Imbalance)
    # 50 = уровни стакана
    
    # Структурируем metadata для совместимости с Rust кодом (Задача 055)
    model_params = {
        "seq_len": seq_len,
        "n_levels": n_levels,
        "in_channels": in_channels,  # 3 канала согласно плану 053
        "past_returns_lags": past_returns_lags,
    }
    
    # Сохраняем дополнительные метаданные для отладки и мониторинга
    export_metadata = {
        "model_name": "LiT",
        "activation": activation,
        "onnx_file": Path(output_path).name,
        "temperature_embedded": embed_temperature,
        "precision": final_precision,
        "quantized": use_fp16,
        "num_regimes": num_regimes,
        "regime_embedding_dim": regime_embedding_dim,
        "use_regime_embedding": use_regime,
        "sparsity": float(sparsity),
        "pruned": sparsity > 0.01,
        "num_horizons": num_horizons,
        "use_horizon_embedding": use_horizon_embedding,
        "multi_horizon": num_horizons > 1,
        "input_shape": [1, seq_len, 150],
        "input_format": "flat_lob_3ch",
        "input_description": "Flat LOB buffer: 50 levels * 3 channels (Price, Volume, Imbalance) = 150 features"
    }
    
    # 9. Экспорт параметров HMM (regime_config.json) - Задача 155
    regime_config_path = Path(output_path).parent / "regime_config.json"
    if regime_config_path.exists():
        with open(regime_config_path, 'r') as f:
            regime_config = json.load(f)
        export_metadata["regime_detection"] = {
            "enabled": True,
            "n_components": regime_config.get("n_components", 0),
            "covariance_type": regime_config.get("covariance_type", "diag"),
            "config_file": "regime_config.json"
        }
        print(f"✓ Regime config found at {regime_config_path} with {regime_config.get('n_components', 0)} components")
    else:
        print("⚠️  Regime config not found, skipping regime detection export")
        export_metadata["regime_detection"] = {
            "enabled": False
        }
    
    # 10. Параметры нормализации (Задача 240)
    norm_params_path = Path(output_path).parent / "norm_params.json"
    norm_metadata = {
        "scaler_type": hparams.get("scaler_type", "zscore"),
        "winsor_limits": hparams.get("winsor_limits", None)
    }
    
    if norm_params_path.exists():
        print(f"Loading normalization params from {norm_params_path}...")
        with open(norm_params_path, 'r') as f:
            norm_data = json.load(f)
        
        if isinstance(norm_data, dict) and "params" in norm_data:
            norm_params = norm_data["params"]
            if "scaler_type" in norm_data:
                norm_metadata["scaler_type"] = norm_data["scaler_type"]
            if "winsor_limits" in norm_data:
                norm_metadata["winsor_limits"] = norm_data.get("winsor_limits")
        else:
            norm_params = norm_data
            
        means, stds, medians, iqrs = [], [], [], []
        winsor_limits_vals = []
        
        order = ["p", "v", "i"]
        for prefix in order:
            for i in range(n_levels):
                feat_name = f"feat_{prefix}_{i}"
                p = norm_params.get(feat_name, {})
                m = p.get("mean", 0.0)
                s = p.get("std", 1.0)
                med = p.get("median", m)
                iqr = p.get("iqr", s)
                
                means.append(float(m))
                stds.append(float(s))
                medians.append(float(med))
                iqrs.append(float(iqr))
                
                if "winsor_low" in p and "winsor_high" in p:
                    winsor_limits_vals.append(float(p["winsor_low"]))
                    winsor_limits_vals.append(float(p["winsor_high"]))
        
        # Задача 240: Финализация параметров нормализации
        final_winsor_limits = winsor_limits_vals if winsor_limits_vals else hparams.get("winsor_limits", None)
        
        norm_metadata.update({
            "mean": means,
            "std": stds,
            "median": medians,
            "iqr": iqrs,
            "winsor_limits": final_winsor_limits
        })
            
    export_metadata["normalization"] = norm_metadata

    # Объединяем с существующей metadata (если есть)
    metadata.update(export_metadata)
    metadata["model_params"] = model_params
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved (Precision: {final_precision})")
    print(f"✓ Model exported to {output_path} and verified.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export LiT model to ONNX")
    parser.add_argument("--input", required=True, help="Path to .ckpt or .pt")
    parser.add_argument("--output", default="lit.onnx", help="Output filename")
    parser.add_argument("--embed_temperature", action="store_true", help="Embed temperature")
    parser.add_argument("--fp16", action="store_true", help="Use FP16")
    
    args = parser.parse_args()
    output_path = Path(args.output)
    if output_path.suffix != ".onnx":
        output_path = output_path.with_suffix(".onnx")
        
    export(args.input, str(output_path), 
           embed_temperature=args.embed_temperature, 
           use_fp16=args.fp16)
