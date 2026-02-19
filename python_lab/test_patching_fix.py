#!/usr/bin/env python3
"""
Тест для проверки исправления динамических размерностей в LOBPatching.
Проверяет, что слой работает с разными параметрами in_channels и n_levels.
"""

import sys
import torch
sys.path.insert(0, 'python_lab/src')

from layers import LOBPatching
from lit_model import LiTModel

def test_lob_patching_default():
    """Тест LOBPatching с параметрами по умолчанию (in_channels=3, n_levels=50)"""
    print("=" * 60)
    print("TEST 1: LOBPatching с параметрами по умолчанию")
    print("=" * 60)
    
    # Параметры по умолчанию
    seq_len = 100
    in_channels = 3
    n_levels = 50
    d_model = 64
    
    # Ожидаемые размеры
    num_features = in_channels * n_levels  # 3 * 50 = 150
    num_patches = num_features // 2  # 150 // 2 = 75
    
    print(f"Параметры:")
    print(f"  seq_len: {seq_len}")
    print(f"  in_channels: {in_channels}")
    print(f"  n_levels: {n_levels}")
    print(f"  d_model: {d_model}")
    print(f"\nОжидаемые размеры:")
    print(f"  num_features: {num_features}")
    print(f"  num_patches: {num_patches}")
    
    # Создаем слой
    patching = LOBPatching(seq_len=seq_len, n_levels=n_levels, in_channels=in_channels, d_model=d_model)
    
    # Проверяем вычисленные размеры
    print(f"\nВычисленные размеры в слое:")
    print(f"  patching.num_features: {patching.num_features}")
    print(f"  patching.num_patches: {patching.num_patches}")
    
    assert patching.num_features == num_features, f"num_features mismatch: {patching.num_features} != {num_features}"
    assert patching.num_patches == num_patches, f"num_patches mismatch: {patching.num_patches} != {num_patches}"
    
    # Проверяем размеры параметров
    print(f"\nРазмеры параметров:")
    print(f"  level_pos_emb: {patching.level_pos_emb.shape}")
    print(f"  time_pos_emb: {patching.time_pos_emb.shape}")
    
    assert patching.level_pos_emb.shape == (1, num_patches, d_model), \
        f"level_pos_emb shape mismatch: {patching.level_pos_emb.shape}"
    assert patching.time_pos_emb.shape == (1, seq_len, d_model), \
        f"time_pos_emb shape mismatch: {patching.time_pos_emb.shape}"
    
    # Тестируем forward
    batch_size = 8
    x = torch.randn(batch_size, seq_len, in_channels, n_levels)
    print(f"\nВходной тензор: {x.shape}")
    
    output = patching(x)
    print(f"Выходной тензор: {output.shape}")
    
    expected_output_shape = (batch_size, seq_len, d_model)
    assert output.shape == expected_output_shape, \
        f"Output shape mismatch: {output.shape} != {expected_output_shape}"
    
    print("\n✓ TEST 1 PASSED\n")


def test_lob_patching_custom():
    """Тест LOBPatching с пользовательскими параметрами"""
    print("=" * 60)
    print("TEST 2: LOBPatching с пользовательскими параметрами")
    print("=" * 60)
    
    # Пользовательские параметры
    seq_len = 50
    in_channels = 6
    n_levels = 100
    d_model = 128
    
    # Ожидаемые размеры
    num_features = in_channels * n_levels  # 6 * 100 = 600
    num_patches = num_features // 2  # 600 // 2 = 300
    
    print(f"Параметры:")
    print(f"  seq_len: {seq_len}")
    print(f"  in_channels: {in_channels}")
    print(f"  n_levels: {n_levels}")
    print(f"  d_model: {d_model}")
    print(f"\nОжидаемые размеры:")
    print(f"  num_features: {num_features}")
    print(f"  num_patches: {num_patches}")
    
    # Создаем слой
    patching = LOBPatching(seq_len=seq_len, n_levels=n_levels, in_channels=in_channels, d_model=d_model)
    
    # Проверяем вычисленные размеры
    print(f"\nВычисленные размеры в слое:")
    print(f"  patching.num_features: {patching.num_features}")
    print(f"  patching.num_patches: {patching.num_patches}")
    
    assert patching.num_features == num_features, f"num_features mismatch: {patching.num_features} != {num_features}"
    assert patching.num_patches == num_patches, f"num_patches mismatch: {patching.num_patches} != {num_patches}"
    
    # Проверяем размеры параметров
    print(f"\nРазмеры параметров:")
    print(f"  level_pos_emb: {patching.level_pos_emb.shape}")
    print(f"  time_pos_emb: {patching.time_pos_emb.shape}")
    
    assert patching.level_pos_emb.shape == (1, num_patches, d_model), \
        f"level_pos_emb shape mismatch: {patching.level_pos_emb.shape}"
    assert patching.time_pos_emb.shape == (1, seq_len, d_model), \
        f"time_pos_emb shape mismatch: {patching.time_pos_emb.shape}"
    
    # Тестируем forward
    batch_size = 4
    x = torch.randn(batch_size, seq_len, in_channels, n_levels)
    print(f"\nВходной тензор: {x.shape}")
    
    output = patching(x)
    print(f"Выходной тензор: {output.shape}")
    
    expected_output_shape = (batch_size, seq_len, d_model)
    assert output.shape == expected_output_shape, \
        f"Output shape mismatch: {output.shape} != {expected_output_shape}"
    
    print("\n✓ TEST 2 PASSED\n")


def test_lit_model_default():
    """Тест LiTModel с параметрами по умолчанию"""
    print("=" * 60)
    print("TEST 3: LiTModel с параметрами по умолчанию")
    print("=" * 60)
    
    # Параметры по умолчанию
    seq_len = 100
    in_channels = 3
    d_model = 64
    
    print(f"Параметры:")
    print(f"  seq_len: {seq_len}")
    print(f"  in_channels: {in_channels}")
    print(f"  d_model: {d_model}")
    
    # Создаем модель
    model = LiTModel(seq_len=seq_len, in_channels=in_channels, d_model=d_model)
    
    # Проверяем, что LOBPatching инициализирован правильно
    print(f"\nПараметры LOBPatching в модели:")
    print(f"  num_features: {model.patching.num_features}")
    print(f"  num_patches: {model.patching.num_patches}")
    
    # Тестируем forward
    batch_size = 8
    n_levels = 50  # N_LEVELS из lit_model.py
    x = torch.randn(batch_size, seq_len, in_channels, n_levels)
    print(f"\nВходной тензор: {x.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"Выходной тензор (logits, vol): {output[0].shape}, {output[1].shape}")
    
    expected_logits_shape = (batch_size, 3)
    expected_vol_shape = (batch_size, 1)
    
    assert output[0].shape == expected_logits_shape, \
        f"Logits shape mismatch: {output[0].shape} != {expected_logits_shape}"
    assert output[1].shape == expected_vol_shape, \
        f"Vol shape mismatch: {output[1].shape} != {expected_vol_shape}"
    
    print("\n✓ TEST 3 PASSED\n")


def test_lit_model_multi_horizon():
    """Тест LiTModel с multi-horizon"""
    print("=" * 60)
    print("TEST 4: LiTModel с multi-horizon (3 горизонта)")
    print("=" * 60)
    
    # Параметры
    seq_len = 100
    in_channels = 3
    d_model = 64
    num_horizons = 3
    
    print(f"Параметры:")
    print(f"  seq_len: {seq_len}")
    print(f"  in_channels: {in_channels}")
    print(f"  d_model: {d_model}")
    print(f"  num_horizons: {num_horizons}")
    
    # Создаем модель
    model = LiTModel(seq_len=seq_len, in_channels=in_channels, d_model=d_model, num_horizons=num_horizons)
    
    # Тестируем forward
    batch_size = 8
    n_levels = 50
    x = torch.randn(batch_size, seq_len, in_channels, n_levels)
    print(f"\nВходной тензор: {x.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"Выходной тензор (logits, vol): {output[0].shape}, {output[1].shape}")
    
    expected_logits_shape = (batch_size, num_horizons, 3)
    expected_vol_shape = (batch_size, 1)
    
    assert output[0].shape == expected_logits_shape, \
        f"Logits shape mismatch: {output[0].shape} != {expected_logits_shape}"
    assert output[1].shape == expected_vol_shape, \
        f"Vol shape mismatch: {output[1].shape} != {expected_vol_shape}"
    
    print("\n✓ TEST 4 PASSED\n")


if __name__ == "__main__":
    try:
        test_lob_patching_default()
        test_lob_patching_custom()
        test_lit_model_default()
        test_lit_model_multi_horizon()
        
        print("=" * 60)
        print("✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
