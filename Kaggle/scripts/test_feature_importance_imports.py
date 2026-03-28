"""
Тестовый скрипт для проверки корректности импортов feature_importance.py
"""

import sys
from pathlib import Path

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    # Проверяем импорты из feature_importance
    from dataset import LOBPyTorchDataset
    from lit_model import LiTModel
    from utils import plot_feature_importance_bar, plot_lob_importance_heatmap
    
    print("✓ Все импорты успешны!")
    print("✓ LOBPyTorchDataset:", LOBPyTorchDataset)
    print("✓ LiTModel:", LiTModel)
    print("✓ plot_feature_importance_bar:", plot_feature_importance_bar)
    print("✓ plot_lob_importance_heatmap:", plot_lob_importance_heatmap)
    
except ImportError as e:
    print(f"✗ Ошибка импорта: {e}")
    sys.exit(1)

print("\n✓ Все проверки пройдены!")
