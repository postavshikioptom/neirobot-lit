# Задача 331: Заморозить мёртвые ветки и упростить stable pipeline

**Дата создания:** 27.03.2026  
**Статус:** ПЛАН  
**Категория:** Обучение Python  
**Зависимости:** 330-python-train-channel-attribution-logging

## Цель

После задач 325-330 нужно сократить лишние degrees of freedom и перевести проект в один стабильный baseline path, чтобы следующие реализации не путались между experimental и stable ветками.

## Контекст и проблема

В проекте накопились ветки:

- multi-horizon
- distillation
- legacy split paths
- deprecated CLI flags
- разрозненные артефакты

Часть из них не помогла качеству, но усложнила reasoning и сопровождение.

## Подзадачи

### 331.1: Добавить baseline profile

**Файлы:**
- `python_lab/src/train_cli.py`
- `python_lab/src/config_profiles.py` при необходимости

**Что добавить:**

Профиль:

`--profile lit_scalping_baseline`

Он должен фиксировать:

- single horizon
- execution-aware label mode
- purged holdout split
- один approved loss
- один approved decision rule
- attribution off by default

### 331.2: Заморозить experimental branches, не удаляя код

**Файлы:**
- `python_lab/src/train_cli.py`
- `python_lab/src/train_module.py`
- `python_lab/src/train.py`

**Что сделать:**

Через `--freeze_experimental_features` отключать по умолчанию:

- multi-horizon path
- distillation path
- лишние legacy branches

### 331.3: Добавить startup invariant checks

**Файлы:**
- `python_lab/src/train.py`
- `python_lab/src/train_cli.py`

**Что проверять до первой эпохи:**
- `label contract version`
- `metrics contract version`
- `split_strategy`
- совместимость `decision_rule` с calibration config
- отсутствие multi-horizon внутри stable profile

При нарушении инварианта запуск должен падать сразу.

### 331.4: Разделить stable и experimental CLI

**Файл:** `python_lab/src/train_cli.py`

**Что сделать:**

Флаги разделить на:
- `stable`
- `experimental`
- `deprecated`

Чтобы новый исполнитель плана не путал production baseline с историческими экспериментами.

### 331.5: Зафиксировать единое дерево артефактов

**Файлы:**
- `python_lab/src/train.py`
- `python_lab/src/baselines.md`

**Структура:**
- `artifacts/<symbol>/labels`
- `artifacts/<symbol>/validation`
- `artifacts/<symbol>/calibration`
- `artifacts/<symbol>/attribution`
- `artifacts/<symbol>/walk_forward`

### 331.6: Документировать pipeline_state

**Файлы:**
- `docs/train_logs.md`
- `python_lab/src/baselines.md`

**Что фиксировать:**
- какой profile активен
- какие ветки frozen
- какой metrics contract активен
- какой label contract активен
- какой split_strategy активен

## Верификация

1. `--profile lit_scalping_baseline` поднимает один и тот же stable contract.
2. `--freeze_experimental_features` реально отключает лишние ветки.
3. Нарушение invariant checks даёт ранний `ValueError`.
4. Артефакты сохраняются в единое дерево.

## Критерии приёмки

1. Есть один baseline profile.
2. Есть startup invariant checks.
3. Experimental branches отделены от stable path.
4. Артефакты и docs структурированы единообразно.

## Что не делать

- Не удалять полезный код без замены на gated branch.
- Не менять LiT architecture.
