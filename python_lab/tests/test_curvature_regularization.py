"""
Тесты для Curvature Regularization (Задача 238)
"""
import torch
import pytest
from python_lab.src.lit_model import LiTModel, compute_curvature_penalty, apply_input_noise


def test_apply_input_noise():
    """Тест функции apply_input_noise"""
    # Создаем тестовые данные
    x = torch.randn(8, 100, 3, 50)
    std = 0.005
    
    # Применяем шум
    x_noisy = apply_input_noise(x, std=std)
    
    # Проверяем что форма не изменилась
    assert x_noisy.shape == x.shape
    
    # Проверяем что данные изменились (с высокой вероятностью)
    assert not torch.allclose(x, x_noisy)
    
    # Проверяем что разница имеет примерно правильное std
    diff = x_noisy - x
    actual_std = diff.std().item()
    # Допускаем отклонение в 3 раза (статистическая вариация)
    assert 0.001 < actual_std < 0.015


def test_compute_curvature_penalty():
    """Тест функции compute_curvature_penalty"""
    # Создаем простую модель
    model = LiTModel(seq_len=100, in_channels=3, d_model=64, nhead=4, num_layers=2)
    model.eval()
    
    # Создаем тестовые данные
    x = torch.randn(4, 100, 3, 50)
    
    # Получаем предсказания
    with torch.no_grad():
        logits, vol = model(x)
    
    # Вычисляем curvature penalty
    penalty = compute_curvature_penalty(model, x, logits, lambda_=1e-4, epsilon=1e-3)
    
    # Проверяем что penalty - скаляр
    assert penalty.dim() == 0
    
    # Проверяем что penalty положительный
    assert penalty.item() >= 0
    
    # Проверяем что penalty имеет разумную величину (не NaN, не Inf)
    assert torch.isfinite(penalty)


def test_curvature_penalty_with_tuple_output():
    """Тест что compute_curvature_penalty корректно обрабатывает кортеж (logits, vol)"""
    model = LiTModel(seq_len=100, in_channels=3, d_model=64, nhead=4, num_layers=2)
    model.eval()
    
    x = torch.randn(4, 100, 3, 50)
    
    with torch.no_grad():
        outputs = model(x)  # Возвращает кортеж (logits, vol)
    
    # Передаем кортеж как outputs
    penalty = compute_curvature_penalty(model, x, outputs, lambda_=1e-4, epsilon=1e-3)
    
    assert penalty.dim() == 0
    assert penalty.item() >= 0
    assert torch.isfinite(penalty)


def test_curvature_penalty_gradient():
    """Тест что curvature penalty поддерживает градиенты"""
    model = LiTModel(seq_len=100, in_channels=3, d_model=64, nhead=4, num_layers=2)
    model.train()
    
    x = torch.randn(4, 100, 3, 50)
    
    # Forward pass
    logits, vol = model(x)
    
    # Вычисляем penalty
    penalty = compute_curvature_penalty(model, x, logits, lambda_=1e-4, epsilon=1e-3)
    
    # Проверяем что можем вычислить градиенты
    penalty.backward()
    
    # Проверяем что градиенты не None для параметров модели
    has_gradients = False
    for param in model.parameters():
        if param.grad is not None:
            has_gradients = True
            break
    
    assert has_gradients, "Model parameters should have gradients after backward()"


if __name__ == "__main__":
    print("Running Curvature Regularization tests...")
    test_apply_input_noise()
    print("✓ test_apply_input_noise passed")
    
    test_compute_curvature_penalty()
    print("✓ test_compute_curvature_penalty passed")
    
    test_curvature_penalty_with_tuple_output()
    print("✓ test_curvature_penalty_with_tuple_output passed")
    
    test_curvature_penalty_gradient()
    print("✓ test_curvature_penalty_gradient passed")
    
    print("\nAll tests passed! ✓")
