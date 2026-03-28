# Задача 328: Проверить objective, calibration и decision rule после нового target

**Дата создания:** 27.03.2026  
**Статус:** ПЛАН  
**Категория:** Обучение Python  
**Зависимости:** 327-python-train-execution-aware-event-time-labeling

## Цель

После фикса нового label contract нужно проверить две вещи:

1. не душит ли directional signal текущий objective;
2. не теряется ли сигнал на этапе принятия решения.

Задача покрывает:

- ablation `CE vs Focal`
- `class_weights on/off`
- `cls_only vs multi_task`
- post-hoc calibration
- class-specific decision thresholds
- `no-trade reject option`

## Контекст и проблема

Сейчас модель валидационно использует почти чистый argmax по logits. Calibration уже считается, но не применяется. При этом по логам есть подозрение, что backbone лучше учит volatility target, чем directional target. Это значит, что без ablation по objective и без calibrated decision rule можно неправильно оценить потенциал модели.

## Подзадачи

### 328.1: Добавить явные objective flags

**Файлы:**
- `python_lab/src/train_cli.py`
- `python_lab/src/train_module.py`

**Что добавить:**
- `--use_class_weights`
- `--multi_task`
- `--cls_loss_weight`
- `--vol_loss_weight`

**Матрица ablation:**
- `CE + cls_only`
- `CE + multi_task`
- `Focal + cls_only`
- `Focal + multi_task`
- отдельно `class_weights on/off`

### 328.2: Явно разделить `cls_only` и `multi_task`

**Файл:** `python_lab/src/train_module.py`

**Что поменять:**

`training_step` и `validation_step` должны поддерживать:

```python
total_loss = cls_loss_weight * loss_cls + vol_loss_weight * loss_vol
```

При `multi_task=False` `loss_vol` не должен участвовать в total loss.

### 328.3: Реализовать post-hoc calibration

**Файлы:**
- `python_lab/src/utils.py`
- `python_lab/src/train_module.py`

**Что добавить:**

Helper:

```python
fit_temperature_scaler(logits_val, y_val)
```

**Артефакт:**

`artifacts/<symbol>/calibration/temperature_scaler.json`

**Логировать:**
- `ece_before`
- `ece_after`
- `mce_before`
- `mce_after`

### 328.4: Реализовать decision rule поверх calibrated probs

**Файлы:**
- `python_lab/src/train_module.py`
- `python_lab/src/train_cli.py`

**Что добавить в CLI:**
- `--decision_rule`
- `--decision_confidence`
- `--decision_hold_threshold`
- `--flat_prob_threshold`
- `--up_prob_threshold`
- `--down_prob_threshold`
- `--margin_threshold`

**Режимы decision rule:**
- `argmax`
- `confidence_gap`
- `class_specific_thresholds`
- `flat_bias`

### 328.5: Разделить argmax-metrics и decision-rule-metrics

**Файл:** `python_lab/src/train_module.py`

**Что сделать:**

На validation path считать две независимые метрики:

- `argmax_metrics`
- `decision_rule_metrics`

Если выбран любой rule кроме `argmax`, primary summary и trading report должны относиться к `decision_rule_metrics`.

### 328.6: Сохранять objective ablation table

**Файлы:**
- `python_lab/src/train.py`
- `python_lab/src/baselines.md`

**Артефакт:**

`objective_ablation.csv`

**Колонки:**
- `loss_type`
- `class_weights`
- `multi_task`
- `mcc_primary`
- `coverage_directional`
- `net_edge_total`
- `ece`

## Верификация

1. Есть минимум одна полная objective ablation matrix.
2. Есть `temperature_scaler.json`.
3. `ece_after <= ece_before`.
4. `decision_rule_metrics` считаются отдельно от `argmax_metrics`.

## Критерии приёмки

1. Есть documented ablation `CE/Focal`, `weights on/off`, `cls_only/multi_task`.
2. Calibration применяется не только для отчёта, но и для decision rule.
3. Trading report после 328 умеет работать с no-trade logic.

## Что не делать

- Не менять labels в этой задаче.
- Не менять LiT architecture.
- Не добавлять новые channels.
