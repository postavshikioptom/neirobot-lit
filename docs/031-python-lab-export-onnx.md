# 031 - Python Lab Export ONNX
Цель задачи: Реализовать CLI-скрипт для экспорта обученной модели в формат ONNX. Скрипт должен поддерживать загрузку весов (как из чекпоинтов Lightning, так и из чистых файлов весов), задавать корректную форму входного тензора и проверять результат через onnxruntime.

Файлы: python_lab/src/export_onnx.py (создать)

Инструкции для Gemini:

CLI Интерфейс: Использовать argparse для указания пути к модели (--input) и выходного файла (--output, по умолчанию lit.onnx).
Загрузка модели:
Если используется Lightning (задача 027), загружать через LiTModule.load_from_checkpoint.
Если чистый PyTorch — создать экземпляр LiTModel и загрузить state_dict.
Модель должна быть переведена в .eval().
Параметры экспорта:
dummy_input: форма (1, 100, 200), где 1 — batch, 100 — seq_len, 200 — признаки (50 уровней стакана * 2 стороны * 2 параметра p/v).
opset_version=17 (для поддержки последних функций Transformer).
dynamic_axes: только для batch_size (индекс 0).
Валидация: Запустить инференс через onnxruntime и сравнить вывод с PyTorch через np.testing.assert_allclose.
import torch
import onnx
import onnxruntime as ort
import numpy as np
import argparse
from .lit_model import LiTModel
from .train import LiTModule  # Если используется Lightning

def export(input_path, output_path):
    # 1. Загрузка (поддержка Lightning .ckpt)
    # Если в 027 использовали чистый Torch, заменить на LiTModel() + load_state_dict
    model_module = LiTModule.load_from_checkpoint(input_path)
    model = model_module.model
    model.eval()

    # 2. Dummy input (Batch, Seq, Feat)
    # 200 = (50 asks + 50 bids) * 2 (price, vol)
    dummy_input = torch.randn(1, 100, 200)

    # 3. Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    # 4. Проверка
    session = ort.InferenceSession(output_path)
    ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outs = session.run(None, ort_inputs)[0]
    
    with torch.no_grad():
        torch_outs = model(dummy_input).numpy()
    
    np.testing.assert_allclose(torch_outs, ort_outs, rtol=1e-03, atol=1e-05)
    print(f"✓ Model exported to {output_path} and verified.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to .ckpt or .pt file")
    parser.add_argument("--output", default="lit.onnx", help="Output filename")
    args = parser.parse_args()
    export(args.input, args.output)
Технические требования:

Вход: Слой LOBPatching (задача 026) должен быть внутри экспортируемого графа, поэтому вход — это именно (B, 100, 200).
Названия: Входной тензор — input, выходной — output (нужно для Rust).
Ошибка: Если разница между Torch и ONNX больше 1e-5, скрипт должен падать с ошибкой.
Почему это важно: Грок прав в том, что нам нужен CLI для автоматизации (например, в CI/CD или скриптах обучения). Использование lit.onnx как стандартного имени упростит загрузку в Rust. Форма (1, 100, 200) — это "сырой" вход в нейронку со всеми уровнями, что позволяет Rust-части просто подавать буфер стакана без доп. обработки в токены.