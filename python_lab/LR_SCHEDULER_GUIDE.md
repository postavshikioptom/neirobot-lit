# Руководство по использованию LR Schedulers

## Обзор

В проекте реализована гибкая система управления Learning Rate (LR) с поддержкой 5 стратегий:

1. **OneCycleLR** - Быстрая сходимость с циклическим momentum
2. **CosineAnnealingWithWarmup** - Плавное затухание с разогревом
3. **ReduceLROnPlateau** - Адаптивное снижение при стагнации (по умолчанию)
4. **StepLR** - Ступенчатое снижение
5. **None** - Константный LR

## Использование в командной строке

### Базовый запуск с OneCycleLR

```bash
python -m python_lab.scripts.train \
    --symbol BTCUSDT \
    --scheduler onecycle \
    --div_factor 25.0 \
    --pct_start 0.3 \
    --epochs 50
```

### CosineAnnealing с Warmup

```bash
python -m python_lab.scripts.train \
    --symbol BTCUSDT \
    --scheduler cosine \
    --div_factor 25.0 \
    --epochs 100
```

### ReduceLROnPlateau (адаптивный)

```bash
python -m python_lab.scripts.train \
    --symbol BTCUSDT \
    --scheduler plateau \
    --plateau_factor 0.5 \
    --plateau_patience 5 \
    --epochs 100
```

### StepLR (простой)

```bash
python -m python_lab.scripts.train \
    --symbol BTCUSDT \
    --scheduler step \
    --step_size 10 \
    --gamma 0.5 \
    --epochs 100
```

### Без scheduler (константный LR)

```bash
python -m python_lab.scripts.train \
    --symbol BTCUSDT \
    --scheduler none \
    --epochs 50
```

## Параметры schedulers

### OneCycleLR

- `--div_factor` (default: 25.0) - Начальный LR = max_lr / div_factor
- `--final_div_factor` (default: 10000.0) - Финальный LR = max_lr / final_div_factor
- `--pct_start` (default: 0.3) - Процент цикла на увеличение LR (30%)
- Автоматически использует `cycle_momentum=True` для AdamW

**Когда использовать**: Для быстрой сходимости за меньшее количество эпох (20-50).

### CosineAnnealingWithWarmup

- `--div_factor` (default: 25.0) - Начальный LR для warmup = max_lr / div_factor
- Warmup: первые 10% шагов
- Cosine annealing: оставшиеся 90% шагов

**Когда использовать**: Для длительного обучения (100+ эпох) с плавным затуханием.

### ReduceLROnPlateau

- `--plateau_factor` (default: 0.5) - Коэффициент снижения LR
- `--plateau_patience` (default: 5) - Количество эпох без улучшения val_mcc
- Мониторит метрику: `val_mcc` (Matthews Correlation Coefficient)

**Когда использовать**: Для зашумленных LOB-данных, когда нужна адаптация к плато.

### StepLR

- `--step_size` (default: 10) - Снижение LR каждые N эпох
- `--gamma` (default: 0.5) - Коэффициент снижения (new_lr = lr * gamma)

**Когда использовать**: Простой baseline для сравнения.

### None

Константный LR на протяжении всего обучения.

**Когда использовать**: Для ablation studies и сравнения с динамическими schedulers.

## Мониторинг в TensorBoard

Все schedulers логируют:
- `lr` - текущий learning rate (на каждом шаге)
- `momentum` - текущий momentum (beta1 для AdamW, на каждом шаге)

Запуск TensorBoard:

```bash
tensorboard --logdir tb_logs
```

## Интеграция с Optuna (HPO)

В `tune.py` автоматически подбираются:
- Тип scheduler: `["onecycle", "plateau", "cosine", "step", "none"]`
- `div_factor`: [10.0, 40.0] для OneCycle/Cosine
- `pct_start`: [0.2, 0.4] для OneCycle
- `plateau_patience`: [3, 7] для Plateau
- И другие параметры

Запуск HPO:

```bash
python -m python_lab.src.tune --symbol BTCUSDT --trials 50
```

## Тестирование schedulers

Визуализация всех schedulers:

```bash
python python_lab/scripts/test_lr_scheduler.py
```

Создаст график `lr_scheduler_comparison.png` с кривыми LR и momentum для всех стратегий.

## Рекомендации

### Для быстрых экспериментов (20-50 эпох)
```bash
--scheduler onecycle --div_factor 25.0 --pct_start 0.3
```

### Для длительного обучения (100+ эпох)
```bash
--scheduler cosine --div_factor 25.0
```

### Для нестабильных данных
```bash
--scheduler plateau --plateau_factor 0.5 --plateau_patience 5
```

### Для baseline
```bash
--scheduler none
```

## Примеры из документации

### Stability Test (проверка на NaN)

```bash
# Высокий initial_lr с warmup (должен быть стабильным)
python -m python_lab.scripts.train \
    --symbol BTCUSDT \
    --scheduler cosine \
    --lr 1e-3 \
    --div_factor 10.0 \
    --epochs 20
```

### Ablation Study (OneCycle vs Constant)

```bash
# Constant LR
python -m python_lab.scripts.train --symbol BTCUSDT --scheduler none --epochs 50

# OneCycle LR
python -m python_lab.scripts.train --symbol BTCUSDT --scheduler onecycle --epochs 50
```

Ожидается: OneCycle даст более высокий MCC при меньшем количестве эпох.

## Troubleshooting

### Ошибка: "Tried to step X times. The specified number of total steps is Y"

**Причина**: Несоответствие между `total_steps` и фактическим количеством шагов.

**Решение**: Используется `self.trainer.estimated_stepping_batches` для автоматического расчета. Если проблема сохраняется, проверьте `max_epochs` и размер датасета.

### Ошибка: NaN loss при OneCycleLR

**Причина**: Слишком высокий `max_lr` или низкий `div_factor`.

**Решение**: 
```bash
--lr 1e-4 --div_factor 25.0  # Более консервативные значения
```

### Plateau не снижает LR

**Причина**: Метрика `val_mcc` продолжает улучшаться.

**Решение**: Уменьшите `--plateau_patience` или проверьте, что модель действительно на плато.

## Дополнительные параметры

### Weight Decay

```bash
--weight_decay 1e-5  # По умолчанию
```

Контролирует L2-регуляризацию в AdamW optimizer.

## Ссылки

- [PyTorch OneCycleLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html)
- [PyTorch SequentialLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.SequentialLR.html)
- [Super-Convergence Paper](https://arxiv.org/abs/1708.07120)
