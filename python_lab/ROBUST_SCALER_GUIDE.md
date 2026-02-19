# Руководство по Robust Scaler и Winsorization (Задача 240)

## Обзор

Реализованы продвинутые методы нормализации данных, устойчивые к выбросам:
- **Winsorization** - клиппинг экстремальных значений по перцентилям
- **Robust Scaler** - масштабирование на основе медианы и IQR вместо среднего и стандартного отклонения

## Python API

### 1. Винзоризация (Winsorization)

```python
from src.dataset import apply_winsorization

# Ограничение значений по 1% и 99% перцентилям
df_clipped = apply_winsorization(df, limits=(0.01, 0.99))
```

### 2. Robust Scaling

```python
from src.dataset import fit_robust_params, apply_robust_scaling

# Вычисление параметров на тренировочных данных
params = fit_robust_params(df_train)

# Применение масштабирования
df_scaled = apply_robust_scaling(df_test, params)
```

### 3. Комбинированный workflow

```python
# 1. Винзоризация для удаления экстремальных выбросов
df_winsor = apply_winsorization(df_train, limits=(0.05, 0.95))

# 2. Вычисление robust параметров
params = fit_robust_params(df_winsor)

# 3. Применение масштабирования
df_scaled = apply_robust_scaling(df_winsor, params)
```

## Rust Integration

Параметры нормализации сохраняются в `metadata.json`:

```json
{
  "normalization": {
    "scaler_type": "robust",
    "median": [0.1, 0.2, ...],
    "iqr": [0.5, 0.6, ...]
  }
}
```

Rust автоматически выбирает правильный алгоритм нормализации при загрузке модели.

## Конфигурация

Поддерживаемые типы скейлеров:
- `"zscore"` - стандартная Z-score нормализация (по умолчанию)
- `"robust"` - Robust Scaler (медиана + IQR)
- `"winsor_robust"` - винзоризация + Robust Scaler
