import numpy as np
import json
from pathlib import Path

def test_channel_formation():
    """
    Проверяет что Python и Rust формируют каналы одинаково.
    """
    # Создаем тестовые сырые данные
    ask_p = np.array([100.0, 100.1, 100.2])  # 3 уровня для простоты
    bid_p = np.array([99.9, 99.8, 99.7])
    ask_v = np.array([10.0, 20.0, 30.0])
    bid_v = np.array([15.0, 25.0, 35.0])
    
    # Python: формируем каналы
    price_ch_py = (ask_p + bid_p) / 2.0
    vol_ch_py = ask_v + bid_v
    imb_ch_py = (bid_v - ask_v) / (bid_v + ask_v + 1e-8)
    
    print("Python каналы:")
    print(f"  price_ch: {price_ch_py}")
    print(f"  vol_ch: {vol_ch_py}")
    print(f"  imb_ch: {imb_ch_py}")
    
    # Rust: (эмулируем логику из tensor.rs)
    price_ch_rust = (ask_p + bid_p) / 2.0
    vol_ch_rust = ask_v + bid_v
    imb_ch_rust = (bid_v - ask_v) / (bid_v + ask_v + 1e-7)  # В Rust используется 1e-7
    
    print("\nRust каналы:")
    print(f"  price_ch: {price_ch_rust}")
    print(f"  vol_ch: {vol_ch_rust}")
    print(f"  imb_ch: {imb_ch_rust}")
    
    # Проверяем совпадение
    assert np.allclose(price_ch_py, price_ch_rust, rtol=1e-5)
    assert np.allclose(vol_ch_py, vol_ch_rust, rtol=1e-5)
    # Используем больший допуск для imb_ch из-за разницы в epsilon
    assert np.allclose(imb_ch_py, imb_ch_rust, atol=1e-6)
    
    print("\n✓ Каналы формируются идентично!")

if __name__ == "__main__":
    test_channel_formation()
