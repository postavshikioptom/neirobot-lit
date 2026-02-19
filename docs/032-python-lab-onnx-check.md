# 032 - Python Lab ONNX Real Data Check
Цель задачи: Провести финальную валидацию экспортированной модели lit.onnx на реальных данных. Скрипт должен гарантировать, что предсказания ONNX и PyTorch идентичны (разница < 1e-6) при подаче одинаковых окон из Parquet-файла.

Файлы: python_lab/src/check_onnx.py (создать)

Инструкции для Gemini:

CLI: Аргументы --onnx, --pt (путь к .ckpt или .pt), --data (.parquet), --norm (norm.json), --n_windows (количество случайных окон для проверки, по умолчанию 10).
Импорт логики: Использовать функции из предыдущих задач (022, 024). Если они не вынесены в модули, реализовать их внутри скрипта (relative prices, log volumes, Z-score).
Загрузка моделей:
PyTorch: LiTModule.load_from_checkpoint(args.pt).model.eval().
ONNX: ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider']).
Цикл валидации:
Выбрать случайную стартовую точку в данных.
Сформировать окно (1, 100, 200).
Прогнать через обе модели.
Сравнить результаты через np.testing.assert_allclose(pt_out, onnx_out, atol=1e-6).
import torch
import polars as pl
import onnxruntime as ort
import json
import numpy as np
import argparse
from src.lit_model import LiTModel
from src.train import LiTModule
# Импорт функций из 022/024 (предполагаем наличие src/features.py или src/dataset.py)
# from src.dataset import create_features, apply_normalization 

def check_onnx(onnx_path, pt_path, data_path, norm_path, n_windows=10):
    # 1. Load Norm & Data
    with open(norm_path, 'r') as f:
        norm = json.load(f)
    df = pl.read_parquet(data_path)
    
    # 2. Preprocess full DF
    # features = create_features(df)
    # normalized = apply_normalization(features, norm)
    # (Здесь должна быть ваша реализация из задач 022/024)
    
    # 3. Load Models
    model_pt = LiTModule.load_from_checkpoint(pt_path).model.eval()
    session = ort.InferenceSession(onnx_path)
    
    # 4. Random Window Loop
    for i in range(n_windows):
        start = np.random.randint(0, len(normalized) - 100)
        window = normalized[start : start + 100].to_numpy().astype(np.float32)
        input_tensor = np.expand_dims(window, axis=0)

        # PyTorch Inference
        with torch.no_grad():
            pt_out = model_pt(torch.from_numpy(input_tensor)).numpy()

        # ONNX Inference
        ort_inputs = {session.get_inputs()[0].name: input_tensor}
        onnx_out = session.run(None, ort_inputs)[0]

        # Final Comparison
        try:
            np.testing.assert_allclose(pt_out, onnx_out, atol=1e-6)
            print(f"Window {i+1}: MATCHED")
        except AssertionError as e:
            print(f"Window {i+1}: MISMATCH! Diff: {np.abs(pt_out - onnx_out).max()}")
            raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--pt", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--norm", required=True)
    args = parser.parse_args()
    check_onnx(args.onnx, args.pt, args.data, args.norm)
Технические требования:

Сравнение: Обязательный assert с точностью 1e-6. Если условие не выполняется — скрипт должен завершаться с ошибкой.
Данные: Использовать random_start для проверки разных участков рынка.
ONNX Providers: Использовать CPUExecutionProvider для стабильности теста.