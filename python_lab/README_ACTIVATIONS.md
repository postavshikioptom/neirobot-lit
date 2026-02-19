# Функции активации в LiT модели

## Обзор

Модель LiT теперь поддерживает выбор функции активации для улучшения сходимости и способности обрабатывать зашумленные данные LOB.

## Поддерживаемые активации

### 1. ReLU (Rectified Linear Unit)
- **Тип**: `relu`
- **Описание**: Классическая активация, обнуляет отрицательные значения
- **Формула**: `f(x) = max(0, x)`
- **Плюсы**: Быстрая, простая
- **Минусы**: Проблема "мертвых нейронов" для отрицательных значений

### 2. GELU (Gaussian Error Linear Unit) - Exact
- **Тип**: `gelu_exact`
- **Описание**: Точная GELU через функцию ошибок (erf)
- **Формула**: `f(x) = x * Φ(x)`, где Φ - функция распределения Гаусса
- **Плюсы**: Ненулевой градиент для отрицательных значений, лучше для нормализованных данных
- **Минусы**: Немного медленнее ReLU
- **Рекомендуется**: Для большинства задач с LOB данными

### 3. GELU (Gaussian Error Linear Unit) - Tanh Approximation
- **Тип**: `gelu_tanh`
- **Описание**: Аппроксимация GELU через tanh (быстрее)
- **Формула**: `f(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`
- **Плюсы**: Быстрее exact GELU, близкие результаты
- **Минусы**: Небольшая потеря точности
- **Рекомендуется**: Для ускорения инференса

### 4. SiLU (Sigmoid Linear Unit / Swish)
- **Тип**: `silu`
- **Описание**: Активация с сигмоидным взвешиванием
- **Формула**: `f(x) = x * σ(x)`, где σ - сигмоида
- **Плюсы**: Часто превосходит GELU в задачах временных рядов
- **Минусы**: Немного медленнее ReLU
- **Рекомендуется**: Для экспериментов, может дать лучший MCC

## Использование

### Обучение с выбором активации

```bash
# GELU (exact) - рекомендуется по умолчанию
python -m python_lab.scripts.train --symbol BTCUSDT --activation gelu_exact

# GELU (tanh) - для ускорения
python -m python_lab.scripts.train --symbol BTCUSDT --activation gelu_tanh

# SiLU - для экспериментов
python -m python_lab.scripts.train --symbol BTCUSDT --activation silu

# ReLU - базовая линия
python -m python_lab.scripts.train --symbol BTCUSDT --activation relu
```

### Подбор гиперпараметров с Optuna

Optuna автоматически перебирает все типы активаций:

```bash
python -m python_lab.scripts.tune --symbol BTCUSDT --trials 50
```

Optuna выберет лучшую активацию на основе метрики MCC.

### Экспорт в ONNX

Экспорт автоматически сохраняет тип активации в метаданных:

```bash
python -m python_lab.scripts.export_onnx --input model.ckpt --output model.onnx
```

Метаданные будут содержать поле `"activation"` для использования в Rust.

## Тестирование

### Тест паритета активаций

Проверяет, что все активации корректно экспортируются в ONNX:

```bash
python python_lab/scripts/test_activation_parity.py
```

Тест проверяет:
1. **Parity**: Разница между PyTorch и ONNX < 1e-6
2. **Gradient Flow**: Градиенты не затухают для отрицательных значений

### Ожидаемые результаты

```
ACTIVATION PARITY TEST SUITE
============================================================

Testing Activation: RELU
  Max difference: 1.19e-07
  Mean difference: 2.34e-08
  ✓ Parity test PASSED for relu
  ✓ Gradient flow test PASSED for relu

Testing Activation: GELU_EXACT
  Max difference: 8.94e-08
  Mean difference: 1.87e-08
  ✓ Parity test PASSED for gelu_exact
  ✓ Gradient flow test PASSED for gelu_exact

Testing Activation: GELU_TANH
  Max difference: 9.12e-08
  Mean difference: 1.92e-08
  ✓ Parity test PASSED for gelu_tanh
  ✓ Gradient flow test PASSED for gelu_tanh

Testing Activation: SILU
  Max difference: 7.65e-08
  Mean difference: 1.54e-08
  ✓ Parity test PASSED for silu
  ✓ Gradient flow test PASSED for silu

ALL TESTS PASSED ✓
```

## Рекомендации

### Для начала работы
- Используйте `gelu_exact` как базовую линию
- Это современная активация с хорошими свойствами для нормализованных данных

### Для оптимизации
- Запустите Optuna для автоматического подбора лучшей активации
- Сравните MCC для разных активаций на вашем датасете

### Для продакшена
- Если `gelu_exact` и `gelu_tanh` дают близкие результаты, используйте `gelu_tanh` для ускорения
- Если `silu` дает лучший MCC, используйте её

## Технические детали

### ONNX Opset
- Используется **opset 17** для нативной поддержки GELU и SiLU
- Операторы не разлагаются на примитивы, что ускоряет инференс

### Применение в архитектуре
Активация применяется во всех слоях модели:
1. **TransformerEncoder**: В feedforward слоях через callable функцию (полная поддержка SiLU)
2. **Classification Head**: В MLP между линейными слоями
3. **LOBPatching**: Не содержит нелинейностей (только Conv1d и LayerNorm)

### Consistency
Одна и та же активация используется во всех слоях модели для консистентности.

## Интеграция с Rust

После экспорта модели в ONNX, файл `metadata.json` будет содержать:

```json
{
  "model_name": "LiT",
  "seq_len": 100,
  "n_levels": 50,
  "in_channels": 6,
  "past_returns_lags": [10, 50, 100],
  "activation": "gelu_exact",
  "onnx_file": "lit.onnx"
}
```

Rust код может использовать это поле для логирования и валидации.
