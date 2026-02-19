import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def run_shap_analysis(model, train_data, test_data, depth, seq_len=1, past_returns_lags=None, save_path=None):
    """
    model: PyTorch модель в eval mode.
    train_data: тензор для калибровки (background).
    test_data: тензор для анализа.
    depth: глубина стакана (n_levels).
    seq_len: длина последовательности.
    past_returns_lags: список лагов (для именования).
    """
    if past_returns_lags is None:
        past_returns_lags = [10, 50, 100]
        
    # 1. Генерация имен фич с учетом временных шагов
    # Базовые имена для одного временного шага
    base_names = []
    for i in range(depth):
        base_names.extend([f'ask_p_{i}', f'ask_v_{i}', f'bid_p_{i}', f'bid_v_{i}'])
    
    # Добавляем лаги past returns к базовым именам
    for lag in past_returns_lags:
        base_names.append(f'r_{lag}')
        
    # Итоговые имена с учетом времени (t-0, t-1, ...)
    feature_names = []
    # В тензоре LiTModel время идет от старых (t-99) к новым (t-0)
    for t in range(seq_len - 1, -1, -1):
        suffix = f" (t-{t})"
        for name in base_names:
            feature_names.append(name + suffix)
    
    # 2. Подготовка фоновых данных
    background_data = train_data.cpu().numpy()
    # Сплющиваем для kmeans (B, Seq*Channels*Levels)
    flat_background_data = background_data.reshape(background_data.shape[0], -1)
    
    background = shap.kmeans(flat_background_data, 100).data
    background = torch.from_numpy(background).to(train_data.device)
    
    # Обертка для модели
    input_shape = test_data.shape[1:]
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model, input_shape):
            super().__init__()
            self.model = model
            self.input_shape = input_shape
        def forward(self, x):
            return self.model(x.view(-1, *self.input_shape))
            
    wrapped_model = ModelWrapper(model, input_shape)
    
    # 3. Инициализация GradientExplainer
    explainer = shap.GradientExplainer(wrapped_model, background)
    
    # Считаем SHAP values
    test_data_flat = test_data.view(test_data.shape[0], -1)
    shap_values = explainer.shap_values(test_data_flat)
    
    # Дозаполнение имен для дополнительных каналов (например, past returns)
    total_features = test_data_flat.shape[1]
    if len(feature_names) < total_features:
        for i in range(len(feature_names), total_features):
            feature_names.append(f'extra_feat_{i}')
    elif len(feature_names) > total_features:
        feature_names = feature_names[:total_features]

    return shap_values, feature_names

def prune_features(shap_values, feature_names, threshold=0.01):
    """
    Агрегирует важность по базовым признакам и уровням стакана.
    """
    import re
    # 1. Считаем абсолютную важность каждого элемента тензора
    # shap_values shape: [classes][samples, flat_features]
    abs_shap_flat = np.mean([np.abs(v).mean(0) for v in shap_values], axis=0)
    
    # 2. Агрегируем важность по "базовым" именам и по "уровням"
    aggregated_importance = {}
    level_importance = {} # Группировка по i в ask_p_i
    
    for name, imp in zip(feature_names, abs_shap_flat):
        # 'ask_v_19 (t-5)' -> 'ask_v_19'
        base_name = name.split(' (t-')[0]
        aggregated_importance[base_name] = aggregated_importance.get(base_name, 0) + imp
        
        # Группировка по уровню стакана
        # Ищем номер уровня в имени (например, _19)
        match = re.search(r'_(\d+)$', base_name)
        if match:
            level = int(match.group(1))
            level_name = f"Level {level}"
            level_importance[level_name] = level_importance.get(level_name, 0) + imp
        elif base_name.startswith('r_'):
            level_importance['Past Returns'] = level_importance.get('Past Returns', 0) + imp
        
    # 3. Нормализуем агрегированную важность признаков
    total_impact = sum(aggregated_importance.values())
    if total_impact == 0:
        importance_pct = {k: 0.0 for k in aggregated_importance}
    else:
        importance_pct = {k: v / total_impact for k, v in aggregated_importance.items()}
        
    # 4. Нормализуем важность уровней
    total_level_impact = sum(level_importance.values())
    level_pct = {k: v / total_level_impact for k, v in level_importance.items()} if total_level_impact > 0 else {}
        
    to_keep = [name for name, pct in importance_pct.items() if pct >= threshold]
    to_drop = [name for name, pct in importance_pct.items() if pct < threshold]
    
    # Сортируем по важности
    sorted_importance = dict(sorted(importance_pct.items(), key=lambda x: x[1], reverse=True))
    sorted_levels = dict(sorted(level_pct.items(), key=lambda x: x[1], reverse=True))
    
    return to_keep, to_drop, sorted_importance, sorted_levels

def plot_shap_results(shap_values, test_data, feature_names, save_path):
    """
    Генерирует графики важности признаков для каждого класса.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    classes = ['Flat', 'Up', 'Down']
    test_data_flat = test_data.view(test_data.shape[0], -1).cpu().numpy()
    
    for i, class_name in enumerate(classes):
        if i >= len(shap_values):
            break
            
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values[i], 
            test_data_flat, 
            feature_names=feature_names, 
            plot_type="bar",
            max_display=20,
            show=False
        )
        plt.title(f"Feature Importance for {class_name}")
        plt.tight_layout()
        plt.savefig(save_path / f"shap_bar_{class_name.lower()}.png")
        plt.close()
        
    print(f"✓ SHAP summary plots saved to: {save_path}")
