"""
Тест паритета между Python и ONNX для разных функций активации.
Проверяет, что экспорт в ONNX корректен для всех типов активаций (GELU exact/tanh, SiLU, ReLU).
"""
import torch
import onnxruntime as ort
import numpy as np
import sys
from pathlib import Path

# Добавляем путь к src для импорта модулей
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lit_model import LiTModel

def test_activation_parity(activation_type: str, tolerance: float = 1e-6):
    """
    Тестирует паритет между PyTorch и ONNX для заданной функции активации.
    
    Args:
        activation_type: Тип активации ('relu', 'gelu_exact', 'gelu_tanh', 'silu')
        tolerance: Максимально допустимая разница между предсказаниями
    """
    print(f"\n{'='*60}")
    print(f"Testing Activation: {activation_type.upper()}")
    print(f"{'='*60}")
    
    # 1. Создаем модель с заданной активацией
    model = LiTModel(
        seq_len=100,
        in_channels=6,  # 3 базовых + 3 past returns
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        activation=activation_type
    )
    model.eval()
    
    # 2. Создаем тестовый вход
    batch_size = 8
    dummy_input = torch.randn(batch_size, 100, 6, 50)
    
    # 3. Получаем предсказание в PyTorch
    with torch.no_grad():
        torch_output = model(dummy_input).numpy()
    
    # 4. Экспортируем в ONNX
    temp_onnx_path = Path(__file__).parent.parent / "temp" / f"test_{activation_type}.onnx"
    temp_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting to ONNX (opset 17)...")
    torch.onnx.export(
        model,
        dummy_input,
        str(temp_onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # 5. Загружаем ONNX модель и делаем предсказание
    print(f"Loading ONNX model...")
    session = ort.InferenceSession(str(temp_onnx_path), providers=['CPUExecutionProvider'])
    
    input_name = session.get_inputs()[0].name
    ort_inputs = {input_name: dummy_input.numpy().astype(np.float32)}
    onnx_output = session.run(None, ort_inputs)[0]
    
    # 6. Сравниваем результаты
    max_diff = np.abs(torch_output - onnx_output).max()
    mean_diff = np.abs(torch_output - onnx_output).mean()
    
    print(f"\nParity Check Results:")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")
    print(f"  Tolerance: {tolerance:.2e}")
    
    # 7. Проверяем, что разница в пределах допустимой
    try:
        np.testing.assert_allclose(torch_output, onnx_output, rtol=1e-03, atol=tolerance)
        print(f"✓ Parity test PASSED for {activation_type}")
        return True
    except AssertionError:
        print(f"✗ Parity test FAILED for {activation_type}")
        print(f"  Difference {max_diff:.2e} exceeds tolerance {tolerance:.2e}")
        return False

def test_gradient_flow(activation_type: str):
    """
    Проверяет, что градиенты не затухают при подаче сильно отрицательных тензоров.
    Это важно для GELU и SiLU, которые имеют ненулевой градиент для отрицательных значений.
    """
    print(f"\n{'='*60}")
    print(f"Testing Gradient Flow: {activation_type.upper()}")
    print(f"{'='*60}")
    
    # Создаем модель
    model = LiTModel(
        seq_len=100,
        in_channels=3,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.0,  # Отключаем dropout для стабильности
        activation=activation_type
    )
    model.train()
    
    # Создаем вход с сильно отрицательными значениями (Z-score нормализация может давать такие значения)
    negative_input = torch.randn(4, 100, 3, 50) * 3.0 - 5.0  # Среднее около -5
    negative_input.requires_grad = True
    
    # Прямой проход
    output = model(negative_input)
    loss = output.sum()
    
    # Обратный проход
    loss.backward()
    
    # Проверяем градиенты
    grad_norm = negative_input.grad.norm().item()
    grad_mean = negative_input.grad.abs().mean().item()
    
    print(f"\nGradient Statistics:")
    print(f"  Gradient norm: {grad_norm:.4f}")
    print(f"  Gradient mean (abs): {grad_mean:.6f}")
    
    # Проверяем, что градиенты не нулевые (не затухли)
    if grad_norm > 1e-6 and grad_mean > 1e-8:
        print(f"✓ Gradient flow test PASSED for {activation_type}")
        return True
    else:
        print(f"✗ Gradient flow test FAILED for {activation_type}")
        print(f"  Gradients are too small (possible vanishing gradient)")
        return False

def test_all_activations():
    """
    Запускает все тесты для всех поддерживаемых активаций.
    """
    activations = ['relu', 'gelu_exact', 'gelu_tanh', 'silu']
    
    print("\n" + "="*60)
    print("ACTIVATION PARITY TEST SUITE")
    print("="*60)
    
    parity_results = {}
    gradient_results = {}
    
    for activation in activations:
        parity_results[activation] = test_activation_parity(activation)
        gradient_results[activation] = test_gradient_flow(activation)
    
    # Итоговый отчет
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    print("\nParity Tests:")
    for activation, passed in parity_results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {activation:15s}: {status}")
    
    print("\nGradient Flow Tests:")
    for activation, passed in gradient_results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {activation:15s}: {status}")
    
    # Проверяем, что все тесты прошли
    all_passed = all(parity_results.values()) and all(gradient_results.values())
    
    if all_passed:
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        return 0
    else:
        print("\n" + "="*60)
        print("SOME TESTS FAILED ✗")
        print("="*60)
        return 1

if __name__ == "__main__":
    try:
        exit_code = test_all_activations()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
