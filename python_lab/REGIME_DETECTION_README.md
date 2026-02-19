# Market Regime Detection через HMM (Задача 155)

## Обзор

Реализована система определения режимов рынка на основе Hidden Markov Models (HMM) для улучшения производительности модели LiT в различных рыночных условиях.

## Компоненты

### 1. Расчет признаков режима (`python_lab/src/dataset.py`)

Реализованы функции для вычисления признаков на скользящих окнах:

- **`compute_intensity(timestamps, window=1000)`**: Количество событий в окне
- **`compute_volatility(mid_prices, window=1000)`**: Логарифм стандартного отклонения цен
- **`compute_spread_zscore(ask_prices, bid_prices, window=1000)`**: Нормализованный спред
- **`compute_ofi(ask_prices, ask_volumes, bid_prices, bid_volumes, window=1000)`**: Order Flow Imbalance
- **`compute_regime_features(df, window=1000)`**: Объединяет все признаки в матрицу

### 2. Детектор режимов (`python_lab/src/regime.py`)

Класс `RegimeDetector` на основе `hmmlearn.hmm.GaussianHMM`:

- **`fit(features)`**: Обучение HMM на признаках
- **`predict_state(features)`**: Online inference для текущего момента
- **`predict_states(features)`**: Предсказание последовательности режимов
- **`compute_silhouette_score(features)`**: Оценка качества кластеризации
- **`save(path)` / `load(path)`**: Сохранение/загрузка параметров HMM

Функция `optimize_n_components_optuna()` для автоматического подбора количества состояний через Optuna.

### 3. Интеграция в модель (`python_lab/src/lit_model.py`)

Модель `LiTModel` расширена:

- **`num_regimes`**: Количество режимов рынка (0 = отключено)
- **`regime_embedding_dim`**: Размерность embedding для режимов
- **`regime_embedding`**: `nn.Embedding(num_regimes, regime_embedding_dim)`
- **`regime_projection`**: Проекция для интеграции с патчами

В `forward()` добавлен параметр `regime_id`, который конкатенируется с выходом Patching Layer.

### 4. Обучение с Regime-Weighted Loss (`python_lab/src/train.py`)

Класс `LiTModule` расширен:

- **`use_regime_weighting`**: Флаг для включения взвешивания по режимам
- **`regime_weights`**: Веса для каждого режима (обратно пропорционально частоте)

В `training_step()` лосс взвешивается по режимам:
```python
combined_weights = time_weights * regime_weights[regime_id]
loss = (loss_raw * combined_weights).mean()
```

В `on_validation_epoch_end()` логируются метрики отдельно для каждого режима.

### 5. Экспорт параметров

Параметры HMM сохраняются в `regime_config.json`:
- `startprob`: Начальные вероятности состояний
- `transmat`: Матрица переходов между состояниями
- `means`: Средние значения признаков для каждого состояния
- `covars`: Ковариационные матрицы для каждого состояния
- `feature_means` / `feature_stds`: Параметры нормализации признаков

## Использование

### Обучение с определением режимов

```bash
python -m src.train --symbol BTCUSDT --data_mode memory
```

Режимы автоматически определяются для `memory` и `memmap` режимов. Для `streaming` режима используются dummy значения (все семплы в режиме 0).

### Параметры

- Количество состояний оптимизируется автоматически через Optuna (2-6 состояний, 10 попыток)
- Размер окна для признаков: 1000 событий
- Веса режимов вычисляются обратно пропорционально частоте

### Логирование

В TensorBoard логируются:
- `val_mcc_regime_{i}`: MCC для режима i
- `val_f1_regime_{i}`: F1-score для режима i

В консоли выводится:
```
Metrics by Market Regime:
  Regime 0: MCC=0.4523, F1=0.6234, Samples=12345
  Regime 1: MCC=0.3891, F1=0.5678, Samples=5678
  Regime 2: MCC=0.5123, F1=0.6789, Samples=2345
```

## Преимущества HMM над K-Means

1. **Временная последовательность**: HMM учитывает инерцию рыночных режимов
2. **Вероятностные переходы**: Модель переходов между состояниями
3. **Online inference**: Эффективное предсказание текущего состояния

## Интеграция с Rust

### Rust-реализация (`src/ml/regime.rs`)

Модуль `regime.rs` реализует предсказание режимов на основе параметров HMM:

```rust
use crate::ml::regime::RegimePredictor;

// Загрузка конфигурации HMM
let predictor = RegimePredictor::load("bots/BTCUSDT/models/regime_config.json")?;

// Предсказание режима
let features = vec![intensity, volatility, spread_zscore, ofi];
let regime_id = predictor.predict_state(&features)?;
```

### Расчет признаков (`src/ml/tensor.rs`)

Класс `RegimeFeatureCalculator` вычисляет признаки из истории снапшотов:

```rust
use crate::ml::tensor::RegimeFeatureCalculator;

// Создание калькулятора с окном 1000 событий
let mut calc = RegimeFeatureCalculator::new(1000);

// Добавление снапшотов
calc.push(&orderbook, timestamp_ms);

// Вычисление признаков
if calc.is_ready() {
    let features = calc.compute_features().unwrap();
    let regime_id = predictor.predict_state(&features)?;
}
```

### Интеграция с ONNX (`src/ml/onnx.rs`)

ONNX-модель принимает `regime_id` как второй вход (если `num_regimes > 0`):

```rust
// Если модель использует regime embedding
if metadata.use_regime_embedding {
    let regime_id = predictor.predict_state(&features)?;
    let outputs = session.run(vec![
        input_tensor,
        regime_id_tensor
    ])?;
}
```

### Алгоритм HMM Inference

Rust-реализация использует упрощенный алгоритм для online inference:

1. **Нормализация признаков**: `(x - mean) / std`
2. **Вычисление log-вероятности** для каждого состояния:
   ```
   log P(state | features) = log P(state) + log P(features | state)
   ```
3. **Выбор состояния** с максимальной вероятностью

Для полноценного HMM inference с учетом последовательности можно реализовать алгоритм Витерби, но для online inference достаточно текущего наблюдения.

## Зависимости

Добавлено в `requirements.txt`:
```
hmmlearn>=0.3.0  # Hidden Markov Models для определения режимов рынка
```
