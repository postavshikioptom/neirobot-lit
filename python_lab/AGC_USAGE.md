# Adaptive Gradient Clipping (AGC) - Руководство по использованию

## Описание

Adaptive Gradient Clipping (AGC) — это метод стабилизации обучения нейронных сетей на волатильных данных, предложенный в статье NFNet (Brock et al., 2021). AGC масштабирует градиенты пропорционально норме весов, предотвращая взрывные градиенты без подавления обучения.

## Использование

### Базовый пример

```bash
# Обучение с AGC (рекомендуется для волатильных пар)
python -m python_lab.src.train \
    --symbol FARTCOINUSDT \
    --clip_mode agc \
    --clip_val 0.01 \
    --epochs 50
```

### Режимы клиппинга

#### 1. Без клиппинга (по умолчанию)
```bash
python -m python_lab.src.train --symbol BTCUSDT --clip_mode none
```

#### 2. Стандартный gradient clipping
```bash
# Для стабильных инструментов (BTC, ETH)
python -m python_lab.src.train \
    --symbol BTCUSDT \
    --clip_mode norm \
    --clip_val 1.0
```

#### 3. Adaptive Gradient Clipping (AGC)
```bash
# Для волатильных инструментов (PEPE, FART, CAKE)
python -m python_lab.src.train \
    --symbol FARTCOINUSDT \
    --clip_mode agc \
    --clip_val 0.01
```

## Параметры

### --clip_mode
- `none` - без клиппинга (по умолчанию)
- `norm` - стандартный gradient clipping по глобальной норме
- `agc` - адаптивный послойный клиппинг

### --clip_val
- Для `norm`: обычно 1.0
- Для `agc`: 0.01-0.1 (меньше = агрессивнее)

## Рекомендации по выбору clip_val

### Для AGC режима:

| Тип инструмента | Рекомендуемый clip_val | Обоснование |
|----------------|------------------------|-------------|
| BTC, ETH (высокая ликвидность) | 0.05-0.1 | Более мягкий клиппинг, стабильные данные |
| Средние альткоины | 0.02-0.05 | Умеренный клиппинг |
| Мемкоины (PEPE, FART) | 0.01-0.02 | Агрессивный клиппинг для защиты от спайков |

## Мониторинг

AGC автоматически логирует статистику каждые 100 шагов:

```
[Step 1000] Gradient Stats:
  Clipped: 23.5% (47/200)
  Max Ratio (All): 0.0234
  Max Ratio (Attention): 0.0189
  Global Grad Norm: 0.4521
```

### Интерпретация метрик:

- **Clipped %**: Процент параметров, к которым применен клиппинг
  - < 10%: Клиппинг почти не работает, можно уменьшить clip_val
  - 10-30%: Оптимальный диапазон
  - > 50%: Слишком агрессивный клиппинг, модель "задыхается"

- **Max Ratio (Attention)**: Максимальное отношение ||G|| / ||W|| в Attention слоях
  - Критично для Transformer архитектур
  - Должно быть близко к clip_val

## Примеры команд

### Обучение с AGC и OneCycle scheduler
```bash
python -m python_lab.src.train \
    --symbol FARTCOINUSDT \
    --clip_mode agc \
    --clip_val 0.01 \
    --scheduler onecycle \
    --epochs 50 \
    --batch_size 128
```

### Knowledge Distillation с AGC
```bash
python -m python_lab.src.train \
    --symbol BTCUSDT \
    --mode distill \
    --teacher_path bots/BTCUSDT/models/checkpoints/teacher_lit.pt \
    --clip_mode agc \
    --clip_val 0.02 \
    --alpha 0.9 \
    --temperature 3.0
```

### Cross-Validation с AGC
```bash
python -m python_lab.src.train \
    --symbol CAKEUSDT \
    --mode cv \
    --n_splits 5 \
    --clip_mode agc \
    --clip_val 0.03 \
    --purge_buffer_events 100 \
    --embargo_buffer_events 50
```

## Технические детали

### Алгоритм AGC

1. Для каждого параметра вычисляется отношение:
   ```
   ratio = ||G|| / max(||W||, eps)
   ```
   где eps = 1e-6 для защиты от деления на ноль

2. Если ratio > clip_factor, градиент масштабируется:
   ```
   G_new = G * (clip_factor * ||W||) / ||G||
   ```

### Исключения

AGC **не применяется** к:
- Одномерным параметрам (bias)
- LayerNorm (веса и смещения)
- Embeddings (включая многомерные: cls_token, pos_emb, level_pos_emb)

Это критично для стабильности обучения, так как эти параметры отвечают за калибровку активаций и позиционное кодирование.

## Интеграция с Optuna

AGC полностью интегрирован в процесс гиперпараметрической оптимизации:

```bash
# Запуск HPO с автоматическим подбором clip_mode и clip_val
python -m python_lab.src.tune \
    --symbol FARTCOINUSDT \
    --trials 50
```

Optuna автоматически подбирает:
- `clip_mode`: none, norm, или agc
- `clip_val`: 
  - Для agc: 0.01-0.1
  - Для norm: 0.5-2.0

Гипотеза: для волатильных инструментов (PEPE, FART) Optuna выберет более агрессивный клиппинг (меньший clip_val), а для стабильных (BTC) - более мягкий.

## Ссылки

- [NFNet Paper](https://arxiv.org/abs/2102.06171) - Brock et al., "High-Performance Large-Scale Image Recognition Without Normalization" (2021)
- [Hugging Face Implementation](https://huggingface.co/nux1111/comfyui_controlnet_aux/blob/main/src/custom_timm/utils/agc.py)
