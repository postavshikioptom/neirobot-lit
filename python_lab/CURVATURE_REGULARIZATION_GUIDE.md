# Руководство по Curvature Regularization (Задача 238)

## Обзор

Curvature Regularization - это механизм регуляризации поверхности функции потерь для повышения обобщающей способности модели LiT. Реализация включает два компонента:

1. **Curvature Penalty** - штраф за кривизну функции потерь
2. **Input Noise Injection** - аугментация входных данных гауссовым шумом

## Цель

Снизить чувствительность модели к незначительным флуктуациям входов (LOB-снапшотов) и предотвратить переобучение на микро-шуме стакана.

## Использование

### Параметры командной строки

```bash
python -m python_lab.src.train \
    --symbol BTCUSDT \
    --use_curvature_reg \
    --curvature_lambda 1e-4 \
    --input_noise_std 0.005
```

### Параметры

- `--use_curvature_reg` - включить регуляризацию кривизны (по умолчанию: True)
- `--curvature_lambda` - коэффициент штрафа за кривизну (рекомендуется: 1e-4 до 1e-3)
- `--input_noise_std` - стандартное отклонение шума для аугментации (рекомендуется: 0.005)

### Примеры использования

#### Базовое обучение с curvature regularization

```bash
python -m python_lab.src.train \
    --symbol BTCUSDT \
    --epochs 50 \
    --use_curvature_reg \
    --curvature_lambda 1e-4 \
    --input_noise_std 0.005
```

#### Отключение curvature regularization

```bash
python -m python_lab.src.train \
    --symbol BTCUSDT \
    --epochs 50 \
    --curvature_lambda 0
```

Примечание: Установка `curvature_lambda=0` эффективно отключает штраф за кривизну.

#### Только noise injection без curvature penalty

```bash
python -m python_lab.src.train \
    --symbol BTCUSDT \
    --epochs 50 \
    --curvature_lambda 0 \
    --input_noise_std 0.01
```

#### Агрессивная регуляризация

```bash
python -m python_lab.src.train \
    --symbol BTCUSDT \
    --epochs 50 \
    --use_curvature_reg \
    --curvature_lambda 1e-3 \
    --input_noise_std 0.01
```

## Технические детали

### Curvature Penalty

Функция `compute_curvature_penalty` аппроксимирует кривизну поверхности функции потерь через конечные разности:

1. Генерирует случайное направление шума `v`
2. Нормализует вектор `v`
3. Вычисляет возмущенные входы: `perturbed_inputs = inputs + epsilon * v`
4. Получает предсказания на возмущенных входах
5. Вычисляет L2-штраф за разницу предсказаний

Это вычислительно эффективнее прямого вычисления Гессиана.

### Input Noise Injection

Функция `apply_input_noise` добавляет гауссов шум к входным данным:

```python
noise = torch.randn_like(x) * std
x_noisy = x + noise
```

Применяется только во время обучения (не валидации).

## Мониторинг

Во время обучения логируются следующие метрики:

- `train_loss_reg` - значение curvature penalty
- `train_loss` - общий loss (включая reg_loss)
- `train_loss_cls` - classification loss
- `train_loss_vol` - volatility loss

Просмотр в TensorBoard:

```bash
tensorboard --logdir runs/BTCUSDT/
```

## Ожидаемые результаты

1. **Уменьшение Confidence Jitter** - снижение "дрожания" вероятностей при стабильных рыночных условиях
2. **Устойчивость к спуфингу** - повышение устойчивости к фиктивным заявкам в стакане
3. **Улучшение MCC** - повышение Matthews Correlation Coefficient на валидационном сете с шумом

## Рекомендации по настройке

### Начальные значения

Для большинства случаев рекомендуются значения по умолчанию:
- `curvature_lambda = 1e-4`
- `input_noise_std = 0.005`

### Подбор параметров

1. **Если модель переобучается:**
   - Увеличьте `curvature_lambda` до 5e-4 или 1e-3
   - Увеличьте `input_noise_std` до 0.01

2. **Если модель недообучается:**
   - Уменьшите `curvature_lambda` до 5e-5 или 1e-5
   - Уменьшите `input_noise_std` до 0.001

3. **Мониторинг:**
   - Следите за `train_loss_reg` в TensorBoard
   - Если reg_loss слишком большой (> 0.1), уменьшите `curvature_lambda`
   - Если reg_loss слишком маленький (< 1e-6), увеличьте `curvature_lambda`

## Совместимость

Curvature Regularization совместим со всеми режимами обучения:
- Обычное обучение (`--mode train`)
- Knowledge Distillation (`--mode distill`)
- Cross-Validation (`--mode cv`)

## Тестирование

Запуск тестов:

```bash
python python_lab/test_implementation.py
```

Или через pytest:

```bash
pytest python_lab/tests/test_curvature_regularization.py -v
```

## Ссылки

- Задача 238: `docs/238-python-lab-loss-function-curvature-regularization.md`
- Реализация: `python_lab/src/lit_model.py`, `python_lab/src/train.py`
- Тесты: `python_lab/tests/test_curvature_regularization.py`
