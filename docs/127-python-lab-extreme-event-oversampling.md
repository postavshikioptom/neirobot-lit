# Задача 127: Оверсэмплинг экстремальных событий для временных рядов (v2.0)

## 1. Модуль балансировки в `python_lab/src/dataset.py`
Реализуй функцию, которая работает с 3D тензорами `(N, Seq_Len, Features)`, сохраняя временную структуру при использовании классических алгоритмов через `reshape`.

```python
import numpy as np
import torch
from smote_variants import BGMM_SMOTE  # pip install smote-variants

def balance_dataset(features, labels, method='bgmm', ratio=0.5):
    """
    features: (N, Seq_Len, Feats)
    labels: (N,)
    ratio: целевая доля миноритарных классов относительно мажоритарного (напр. 0.5 = 1:2)
    """
    orig_shape = features.shape # (N, S, F)
    n_samples = orig_shape[0]
    
    # 1. Flatten для совместимости с библиотеками оверсэмплинга
    features_2d = features.reshape(n_samples, -1)
    
    # 2. Определение стратегии (не делаем 1:1, чтобы избежать оверфиттинга)
    counts = np.bincount(labels)
    maj_class = np.argmax(counts)
    target_count = int(counts[maj_class] * ratio)
    
    # Стратегия: балансируем только Up (1) и Down (2) до target_count
    sampling_strategy = {1: max(counts[1], target_count), 2: max(counts[2], target_count)}
    
    if method == 'bgmm':
        sampler = BGMM_SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    elif method == 'smote':
        from imblearn.over_sampling import SMOTE
        sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    else:
        return features, labels

    X_res, y_res = sampler.sample(features_2d, labels)
    
    # 3. Reshape обратно в последовательности
    X_res = X_res.reshape(-1, orig_shape[1], orig_shape[2])
    
    # 4. Фильтрация физически невозможных стаканов
    mask = [validate_lob_sequence(seq) for seq in X_res]
    return X_res[mask], y_res[mask]
```

## 2. LOB-валидатор (Guard)
Мы должны гарантировать, что синтетические данные не нарушают правила биржи. Проверяем каждый шаг в последовательности.

```python
def validate_lob_sequence(sequence):
    """
    sequence: (Seq_Len, Features)
    Проверяет: цены > 0, объемы > 0, ask_0 > bid_0 (no cross).
    """
    # Предполагаем структуру из 022: [ask_p_0, ask_v_0, bid_p_0, bid_v_0, ...]
    for step in sequence:
        ask_p0, ask_v0 = step[0], step[1]
        bid_p0, bid_v0 = step[2], step[3]
        
        if ask_p0 <= bid_p0: return False # Спред схлопнулся или отрицательный
        if np.any(step < 0): return False  # Отрицательные цены/объемы
        
    return True
```

## 3. Спорные моменты и корректировки (Grok + Zencoder)

*   **SMOTE vs Time-Series**: Обычный SMOTE на flattened данных — это компромисс. Grok прав: лучше использовать `BGMM_SMOTE` (Bayesian Gaussian Mixture Model), так как он лучше моделирует распределение в финансовом пространстве признаков.
*   **Validation Leakage**: Это критическая ошибка. **Балансировка применяется только к `train_set`**. Валидационный и тестовый наборы должны отражать реальное рыночное распределение (где 90% времени ничего не происходит).
*   **Ratio 1:1**: Категорически против. Если мы сделаем классы равными (33/33/33), модель начнет «галлюцинировать» лонги там, где их нет. Оптимальное `ratio` — 0.3-0.5.
*   **Normalization**: Синтетические данные могут изменить `mean` и `std`. Сначала делаем оверсэмплинг, затем — финальную нормализацию (Z-Score) на основе параметров тренировочного сета.

## 4. Инструкции для Gemini (Coder AI):
1.  **Dependencies**: Добавить `smote-variants` и `imbalanced-learn`.
2.  **python_lab/src/dataset.py**: Реализовать `balance_dataset` и `validate_lob_sequence`.
3.  **python_lab/train.py**: Вызывать балансировку **после** разделения на Train/Val, но **до** создания DataLoader.
4.  **Log**: Выводить количество отбракованных сэмплов (инвалидных стаканов) после валидатора.

**Результат**: Мы получаем «умное» увеличение числа примеров с сильными движениями цены, что поднимет F1-score для классов `Up` и `Down` без разрушения физики стакана.
