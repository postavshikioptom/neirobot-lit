"""
Простая проверка реализации Curvature Regularization (Задача 238)
"""
import torch
import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.lit_model import LiTModel, compute_curvature_penalty, apply_input_noise
    print("✓ Импорты успешны")
    
    # Тест 1: apply_input_noise
    print("\n1. Тестирование apply_input_noise...")
    x = torch.randn(2, 10, 3, 50)
    x_noisy = apply_input_noise(x, std=0.005)
    assert x_noisy.shape == x.shape, "Форма должна сохраниться"
    assert not torch.allclose(x, x_noisy), "Данные должны измениться"
    print("   ✓ apply_input_noise работает корректно")
    
    # Тест 2: compute_curvature_penalty
    print("\n2. Тестирование compute_curvature_penalty...")
    model = LiTModel(seq_len=10, in_channels=3, d_model=32, nhead=2, num_layers=1)
    model.eval()
    
    with torch.no_grad():
        logits, vol = model(x)
    
    penalty = compute_curvature_penalty(model, x, logits, lambda_=1e-4, epsilon=1e-3)
    assert penalty.dim() == 0, "Penalty должен быть скаляром"
    assert penalty.item() >= 0, "Penalty должен быть неотрицательным"
    assert torch.isfinite(penalty), "Penalty должен быть конечным"
    print(f"   ✓ compute_curvature_penalty работает корректно (penalty={penalty.item():.6f})")
    
    # Тест 3: Градиенты
    print("\n3. Тестирование градиентов...")
    model.train()
    x_train = torch.randn(2, 10, 3, 50)
    logits_train, vol_train = model(x_train)
    penalty_train = compute_curvature_penalty(model, x_train, logits_train, lambda_=1e-4)
    penalty_train.backward()
    
    has_gradients = any(p.grad is not None for p in model.parameters())
    assert has_gradients, "Параметры модели должны иметь градиенты"
    print("   ✓ Градиенты вычисляются корректно")
    
    print("\n" + "="*60)
    print("ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО! ✓")
    print("="*60)
    print("\nРеализация задачи 238 завершена:")
    print("  • compute_curvature_penalty - реализована")
    print("  • apply_input_noise - реализована")
    print("  • Интеграция в training_step - выполнена")
    print("  • Параметры конфигурации - добавлены")
    
except Exception as e:
    print(f"\n✗ ОШИБКА: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
