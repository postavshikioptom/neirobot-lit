# Задача 330: Добавить channel attribution logging по эпохам

**Дата создания:** 27.03.2026  
**Статус:** ПЛАН  
**Категория:** Обучение Python  
**Зависимости:** 329-python-train-purged-holdout-and-walkforward

## Цель

Добавить вторичную диагностику, которая показывает вклад входных каналов в решения `Flat / Up / Down`. Это не лечение качества, а инструмент анализа после починки labels, metrics и evaluation.

## Контекст и проблема

Пользователь хочет видеть, какой канал больше влияет на решения модели. Это полезно, но только как post-hoc instrumentation. Главное требование: attribution не должен менять train loss и не должен ломать stable pipeline.

## Подзадачи

### 330.1: Добавить attribution flags и helper

**Файлы:**
- `python_lab/src/train_cli.py`
- `python_lab/src/utils.py`

**Что добавить:**
- `--enable_channel_attribution`
- `--channel_attribution_samples`
- `--channel_attribution_method` with choices `grad_x_input`, `occlusion`

### 330.2: Зафиксировать реальный channel contract проекта

**Файлы:**
- `python_lab/src/dataset.py`
- `python_lab/src/train_module.py`

**Что добавить:**

Единый ordered список из 11 каналов:

- `microprice_dev`
- `volume`
- `imbalance`
- `ofi`
- `vib`
- `ret_10`
- `ret_50`
- `ret_100`
- `spread`
- `delta_imb`
- `delta_spread`

Нельзя использовать сокращённый ручной список.

### 330.3: Интегрировать attribution в `on_validation_epoch_end`

**Файл:** `python_lab/src/train_module.py`

**Что считать:**
- `general attribution`
- attribution by `predicted class`
- attribution by `true class`
- attribution on `correct`
- attribution on `wrong`

### 330.4: Сохранять артефакты в JSON и CSV

**Файлы:**
- `python_lab/src/train_module.py`
- `docs/train_logs.md`

**Артефакты:**
- `artifacts/<symbol>/attribution/epoch_{epoch}.json`
- `artifacts/<symbol>/attribution/epoch_{epoch}.csv`

**Для каждого канала хранить:**
- `mean_abs_attr`
- `signed_attr_mean`
- `rank`
- `group`

### 330.5: Печатать только high-signal summary

**Файл:** `python_lab/src/train_module.py`

**Что печатать в stdout:**
- top-5 каналов для `Flat`
- top-5 каналов для `Up`
- top-5 каналов для `Down`

## Верификация

1. Attribution включается отдельным флагом.
2. На эпохе создаются JSON/CSV артефакты.
3. В attribution ровно 11 каналов и порядок совпадает с dataset contract.
4. Есть разрезы `predicted/true/correct/wrong`.

## Критерии приёмки

1. Attribution не влияет на train loss.
2. Логи показывают вклад каналов по классам.
3. Есть machine-readable артефакты по эпохам.

## Что не делать

- Не менять LiT architecture.
- Не создавать новые channels.
- Не использовать attribution как training signal.
