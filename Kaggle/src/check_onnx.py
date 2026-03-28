import torch
import polars as pl
import onnxruntime as ort
import json
import numpy as np
import argparse
from pathlib import Path

# Импорт из текущего пакета
try:
    from .lit_model import LiTModel
    from .train import LiTModule
    from .features import FeatureEngineer
    from .normalization import Normalizer
    from .export_onnx import ExportWrapper
except ImportError:
    # Для запуска как скрипта
    from lit_model import LiTModel
    from train import LiTModule
    from features import FeatureEngineer
    from normalization import Normalizer
    from export_onnx import ExportWrapper

def check_onnx(onnx_path, pt_path, data_path, norm_path, n_windows=10):
    print(f"Loading data from {data_path}...")
    df = pl.read_parquet(data_path)
    
    # 1. Предобработка данных (фичи + нормализация)
    print("Preprocessing data...")
    fe = FeatureEngineer(n_levels=50)
    df_feat = fe.transform(df)
    
    normalizer = Normalizer(norm_path)
    normalizer.load()
    df_norm = normalizer.transform(df_feat)
    
    # Получаем признаки для проверки
    feat_cols = [c for c in df_norm.columns if c.startswith("feat_")]
    data_array = df_norm.select(feat_cols).to_numpy().astype(np.float32)
    
    # 2. Загрузка моделей
    print(f"Loading PyTorch model from {pt_path}...")
    model_module = LiTModule.load_from_checkpoint(pt_path, map_location="cpu")
    base_model = model_module.model.eval()
    
    # Оборачиваем модель в ExportWrapper для преобразования входа (B, S, 200) -> (B, S, 4, 50)
    # Согласно плану 031: 200 = 50 уровней * 2 стороны * 2 параметра (price, volume)
    model_pt = ExportWrapper(base_model, in_channels=4, n_levels=50)
    model_pt.eval()
    
    print(f"Loading ONNX model from {onnx_path}...")
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # 3. Цикл валидации по случайным окнам
    print(f"Starting validation on {n_windows} random windows...")
    seq_len = 100
    
    for i in range(n_windows):
        # Выбираем случайную точку для окна
        start_idx = np.random.randint(0, len(data_array) - seq_len)
        window = data_array[start_idx : start_idx + seq_len]
        
        # Формируем входной тензор (Batch, Seq, Feat) = (1, 100, 200)
        # Это плоский формат LOB: 50 уровней * 2 стороны * 2 параметра (price, volume)
        input_tensor = np.expand_dims(window, axis=0).astype(np.float32)
        
        # Инференс PyTorch
        # ExportWrapper преобразует (1, 100, 200) -> (1, 100, 4, 50) для LiTModel
        with torch.no_grad():
            pt_output = model_pt(torch.from_numpy(input_tensor))
            
            # Распаковываем кортеж (logits, vol) и берем только logits
            if isinstance(pt_output, tuple):
                pt_logits, pt_vol = pt_output
                pt_out = pt_logits.numpy()
            else:
                # На случай, если модель возвращает только logits
                pt_out = pt_output.numpy()
            
        # Инференс ONNX
        # ONNX модель экспортирована с ExportWrapper, поэтому она ожидает (1, 100, 200)
        # и возвращает только logits (не кортеж)
        ort_inputs = {input_name: input_tensor}
        onnx_out = session.run(None, ort_inputs)[0]
        
        # Сравнение
        try:
            np.testing.assert_allclose(pt_out, onnx_out, atol=1e-6)
            print(f"Window {i+1}/{n_windows} (start={start_idx}): MATCHED")
        except AssertionError as e:
            max_diff = np.abs(pt_out - onnx_out).max()
            print(f"Window {i+1}/{n_windows} (start={start_idx}): MISMATCH! Max diff: {max_diff}")
            raise e

    print("\n✓ Final Validation Passed: PyTorch and ONNX models produce identical results on real data.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify ONNX model against PyTorch on real data")
    parser.add_argument("--onnx", required=True, help="Path to exported .onnx model")
    parser.add_argument("--pt", required=True, help="Path to PyTorch Lightning .ckpt file")
    parser.add_argument("--data", required=True, help="Path to real data in .parquet format")
    parser.add_argument("--norm", required=True, help="Path to normalization parameters .json")
    parser.add_argument("--n_windows", type=int, default=10, help="Number of random windows to check")
    
    args = parser.parse_args()
    check_onnx(args.onnx, args.pt, args.data, args.norm, args.n_windows)
