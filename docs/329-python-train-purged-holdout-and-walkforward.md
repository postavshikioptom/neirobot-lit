# Задача 329: Перевести evaluation на purged holdout и walk-forward

**Дата создания:** 27.03.2026  
**Статус:** ПЛАН  
**Категория:** Обучение Python  
**Зависимости:** 328-python-train-objective-calibration-decision-rule

## Цель

Заменить слабый основной split на честный evaluation-контур для overlapping windows и forward labels:

- `purged_holdout`
- `walk_forward`
- режимный анализ по корзинам рынка

## Контекст и проблема

Текущий chronological split без purge/embargo для такой задачи слишком слаб. При наличии seq windows, forward labels и overlapping samples нужно гарантировать, что near-boundary leakage не попадает в validation/test.

## Подзадачи

### 329.1: Добавить split-strategy в CLI

**Файлы:**
- `python_lab/src/train_cli.py`
- `python_lab/src/train.py`

**Что добавить:**
- `--split_strategy`
- `--embargo_seconds`
- `--purge_buffer_events`
- `--embargo_buffer_events`
- `--holdout_days`
- `--training_window_days`

**Режимы:**
- `chronological`
- `purged_holdout`
- `walk_forward`

### 329.2: Реализовать effective purge rule

**Файл:** `python_lab/src/train_data.py`

**Ключевая формула:**

```python
effective_purge_events = max(seq_len, max_horizon, max_lag, purge_buffer_events)
```

То есть purge должен учитывать не только секунды, но и структуру sample overlap.

### 329.3: Реализовать `purged_holdout` helper

**Файл:** `python_lab/src/train_data.py`

**Что должен делать helper:**

- строить `train_indices`, `val_indices`, `test_indices`
- вырезать boundary leakage по времени
- вырезать boundary leakage по событиям/окнам
- логировать `effective_purge_events`

### 329.4: Реализовать walk-forward runner

**Файлы:**
- `python_lab/src/train.py`
- `python_lab/src/train_cv.py` или новый runner

**Что должно логироваться по каждому окну:**
- `train_range`
- `val_range`
- `test_range`
- `effective_purge_events`
- `mcc_primary`
- `coverage_directional`
- `net_edge_total`

### 329.5: Добавить режимный анализ качества

**Файлы:**
- `python_lab/src/train_module.py`
- `python_lab/src/train_data.py`

**Разрезы минимум по:**
- `spread`
- `volatility`
- `activity`

Для каждой корзины логировать:
- `samples`
- `coverage_directional`
- `mcc_primary`
- `net_edge_total`

### 329.6: Согласовать single-run и CV по одному metric contract

**Файлы:**
- `python_lab/src/train_cv.py`
- `python_lab/src/train_module.py`

**Требование:**

И обычный holdout, и CV/walk-forward должны использовать тот же `val_mcc_primary` и тот же trading report contract из 325.

## Верификация

1. `effective_purge_events >= max(seq_len, max_horizon, max_lag)`.
2. Holdout split не имеет overlap leakage.
3. Walk-forward создаёт несколько окон и логирует их диапазоны.
4. Есть режимный анализ по spread/vol/activity.

## Критерии приёмки

1. Основной split больше не сводится к старому 70/15/15 без purge.
2. Есть рабочий `walk_forward`.
3. Есть режимные quality reports.

## Что не делать

- Не менять labels в этой задаче.
- Не менять architecture.
