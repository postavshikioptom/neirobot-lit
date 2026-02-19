# Knowledge Distillation Guide (Задача 151)

## Обзор

Knowledge Distillation (дистилляция знаний) - это техника сжатия модели, которая позволяет передать знания от большой "teacher" модели к компактной "student" модели. Это критично для достижения минимальной латентности в Rust-инференсе.

## Преимущества для LOB данных

- **Сжатие модели**: уменьшение количества параметров в 4-10 раз
- **Ускорение инференса**: speedup 3-5x на GPU, еще больше на CPU
- **Сохранение точности**: MCC retention 95-98% при правильных параметрах
- **Лучшая генерализация**: soft labels от teacher содержат информацию о близости классов

## Workflow

### Шаг 1: Обучение Teacher модели

Сначала обучите тяжелую teacher модель с большими параметрами:

```bash
python -m python_lab.src.train \
    --symbol BTCUSDT \
    --mode train \
    --epochs 50 \
    --batch_size 128 \
    --seq_len 100 \
    --d_model 256 \
    --nhead 8 \
    --num_layers 8 \
    --dropout 0.1 \
    --activation gelu_exact \
    --scheduler onecycle
```

После обучения модель будет **автоматически сохранена** в `bots/BTCUSDT/models/teacher_lit.pt`.

Вы увидите сообщение:
```
============================================================
✓ Teacher model automatically saved to:
  bots/BTCUSDT/models/teacher_lit.pt
  Source: bots/BTCUSDT/models/checkpoints/lit-epoch=XX-val_mcc=0.XXXX.ckpt

Next step: Use this teacher for distillation:
  python -m python_lab.src.train \
    --symbol BTCUSDT \
    --mode distill \
    --teacher_path bots/BTCUSDT/models/teacher_lit.pt
============================================================
```

**Примечание**: Если вы хотите сохранить checkpoint вручную, используйте команды из предыдущей версии документации.

### Шаг 2: Distillation в Student модель

Теперь дистиллируйте знания в компактную student модель:

```bash
python -m python_lab.src.train \
    --symbol BTCUSDT \
    --mode distill \
    --teacher_path bots/BTCUSDT/models/teacher_lit.pt \
    --student_d_model 64 \
    --student_nhead 4 \
    --student_num_layers 2 \
    --alpha 0.9 \
    --temperature 3.0 \
    --epochs 30 \
    --batch_size 128 \
    --scheduler cosine
```

После обучения student модель будет **автоматически сохранена** в `bots/BTCUSDT/models/lit.pt`.

Вы увидите сообщение:
```
============================================================
✓ Student model automatically saved to:
  bots/BTCUSDT/models/lit.pt
  Source: bots/BTCUSDT/models/checkpoints/lit-epoch=XX-val_mcc=0.XXXX.ckpt

Next step: Export to ONNX:
  python -m python_lab.scripts.export_onnx \
    --checkpoint bots/BTCUSDT/models/lit.pt
============================================================
```

### Шаг 3: Анализ результатов

После обучения вы увидите таблицу сравнения:

```
KNOWLEDGE DISTILLATION: Teacher vs Student Comparison
==============================================================
Metric                    Teacher         Student         Improvement    
----------------------------------------------------------------------
MCC                       0.6234          0.6102          97.88%
Latency (ms)              12.45           3.21            3.88x
Parameters                1,234,567       156,789         7.87x
----------------------------------------------------------------------
```

Метрики также сохраняются в `bots/SYMBOL/models/distillation_metrics.json`.

## Параметры Distillation

### --alpha (default: 0.9)

Вес soft loss в комбинированной функции потерь:
- `loss = alpha * soft_loss + (1 - alpha) * hard_loss`
- **Рекомендация для LOB**: 0.9 (высокий вес на soft loss)
- Почему: soft labels от teacher содержат информацию о близости классов (Up vs Down vs Flat), которую теряет обычная кросс-энтропия

### --temperature (default: 3.0)

Температура для размягчения логитов:
- Более высокая T → более мягкие распределения → больше информации о близости классов
- Более низкая T → более резкие распределения → ближе к hard labels
- **Рекомендация для LOB**: 2.0-5.0
- Почему: LOB данные зашумлены, размягчение помогает student не переобучаться на шуме

### Student архитектура

Типичные конфигурации:

| Конфигурация | d_model | nhead | num_layers | Compression | Speedup |
|--------------|---------|-------|------------|-------------|---------|
| Tiny         | 32      | 2     | 1          | ~15x        | ~6x     |
| Small        | 64      | 4     | 2          | ~8x         | ~4x     |
| Medium       | 128     | 4     | 3          | ~4x         | ~2.5x   |

## Интеграция с другими фичами

### Time Weighting (Задача 123)

Distillation совместим с time weighting:
```bash
--mode distill --use_time_weighting --half_life_hours 24.0
```

### Data Augmentation (Задача 124)

Аугментация применяется только к student:
```bash
--mode distill --use_symmetric_flip --volume_jitter_range 0.1
```

### LR Schedulers (Задача 095)

Рекомендуется использовать cosine scheduler для distillation:
```bash
--mode distill --scheduler cosine --div_factor 25.0
```

## Troubleshooting

### Student MCC значительно ниже Teacher

- Увеличьте `--temperature` (попробуйте 4.0-5.0)
- Увеличьте `--alpha` (попробуйте 0.95)
- Увеличьте количество эпох
- Попробуйте более крупную student архитектуру

### Student не сходится

- Уменьшите learning rate
- Используйте warmup scheduler (cosine или onecycle)
- Проверьте, что teacher модель загружена корректно

### Латентность не улучшилась

- Уменьшите student архитектуру (меньше d_model, num_layers)
- Проверьте, что замер проводится на правильном устройстве (GPU vs CPU)
- Убедитесь, что используется mixed precision (16-mixed)

## Следующие шаги

После успешной distillation:

1. **Экспорт в ONNX** (Задача 031):
   ```bash
   python -m python_lab.scripts.export_onnx \
       --checkpoint bots/BTCUSDT/models/checkpoints/best_student.ckpt
   ```

2. **Оптимизация TensorRT** (Задача 032):
   ```bash
   python -m python_lab.scripts.optimize_tensorrt \
       --onnx bots/BTCUSDT/models/lit.onnx
   ```

3. **Интеграция в Rust** (Задача 033):
   - Используйте оптимизированную ONNX модель в Rust инференсе
   - Ожидайте латентность <5ms на GPU, <20ms на CPU

## Optuna Tuning (Задача 151 + 030)

Для автоматического поиска оптимальных параметров distillation используйте Optuna:

```bash
python -m python_lab.src.tune \
    --symbol BTCUSDT \
    --mode distill \
    --teacher_path bots/BTCUSDT/models/teacher_lit.pt \
    --trials 50
```

Optuna будет оптимизировать:
- `alpha` (0.7-0.95) - вес soft loss
- `temperature` (2.0-5.0) - температура для размягчения логитов
- `d_model` (32, 64, 128) - размер student модели
- `nhead` (4, 8) - количество attention heads
- `num_layers` (1, 2, 3) - количество transformer слоев
- Другие гиперпараметры (lr, dropout, scheduler и т.д.)

После завершения тюнинга используйте лучшие параметры для финального обучения:

```bash
# Пример с лучшими параметрами из Optuna
python -m python_lab.src.train \
    --symbol BTCUSDT \
    --mode distill \
    --teacher_path bots/BTCUSDT/models/teacher_lit.pt \
    --student_d_model 64 \
    --student_nhead 4 \
    --student_num_layers 2 \
    --alpha 0.87 \
    --temperature 3.5 \
    --epochs 50
```

## Ссылки

- [Hinton et al. 2015 - Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [PyTorch KL Divergence Loss](https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html)
- [Задача 000 - Architecture](../docs/000-architecture.md)
