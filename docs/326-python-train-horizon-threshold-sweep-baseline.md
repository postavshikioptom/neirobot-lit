# Задача 326: Построить baseline sweep по horizon x threshold

**Дата создания:** 27.03.2026  
**Статус:** ПЛАН  
**Категория:** Обучение Python  
**Зависимости:** 325-python-train-fix-metrics-validation-contract

## Цель

Перед изменением labels нужно формально измерить поверхность задачи: как меняются class balance, trade share, horizon-time и spread-context при разных `(horizon, threshold)`. Задача не меняет LiT и не переводит основной train на новый target, а создаёт baseline grid и shortlist кандидатов.

## Контекст и проблема

Локальный анализ уже показал:

- `h=10, thr=0.0005` -> около `95.8 / 2.1 / 2.1`
- `h=20, thr=0.0005` -> около `91.1 / 4.3 / 4.6`
- `h=50, thr=0.0005` -> около `79.0 / 10.1 / 10.9`
- `h=100, thr=0.0005` -> около `65.3 / 16.7 / 18.0`

Также известно:

- median spread около `32.54 bps`
- текущий threshold `0.0005 = 5 bps`
- значит текущий target живёт внутри спреда
- текущий dynamic threshold (`rolling_std * 0.5`) даёт около `21 / 39 / 39` и не годится как быстрый replacement

Эти факты должны быть превращены в reproducible sweep, а не оставаться разрозненными вычислениями.

## Подзадачи

### 326.1: Добавить отдельный sweep-runner и CLI

**Файлы:**
- `python_lab/src/train_cli.py`
- `python_lab/src/train.py`

**Что добавить в CLI:**
- `--horizon_sweep`
- `--threshold_sweep`
- `--sweep_baseline_path`
- `--sweep_use_event_rows`
- `--sweep_train_topk`
- `--sweep_epochs`
- `--sweep_limit_train_batches`
- `--sweep_limit_val_batches`

**Требование:**

Если включён sweep-режим, основной train loop не должен стартовать автоматически.

### 326.2: Реализовать `collect_sweep_baseline`

**Файл:** `python_lab/src/train_data.py`

**Что считать для каждой точки grid:**

- `share_flat`
- `share_up`
- `share_down`
- `trade_share`
- `row_time_seconds`
- `event_time_seconds`, если доступен `last_update_id`
- `median_spread_bps`
- `threshold_bps`
- `threshold_to_spread_ratio`
- `subspread_target`

**Минимальная grid-сетка:**

- `horizons = [10, 20, 50, 100]`
- `thresholds = [0.0001, 0.0002, 0.0003, 0.0005, 0.0008, 0.0010, 0.0015]`

### 326.3: Зафиксировать dynamic threshold как unsafe reference

**Файлы:**
- `python_lab/src/train_data.py`
- `docs/baselines.md`

**Что добавить:**

Отдельный блок `dynamic_threshold_reference`, где явно фиксируется, что текущая реализация `rolling_std * 0.5` при `h=100` даёт распределение около `21 / 39 / 39` и не должна становиться новым default.

### 326.4: Логировать relation threshold к spread

**Файлы:**
- `python_lab/src/train_data.py`
- `docs/baselines.md`

**Что добавить:**

Для каждой строки sweep-таблицы сохранять:

```text
threshold_bps
median_spread_bps
threshold_to_spread_ratio
subspread_target
```

Если `threshold_to_spread_ratio < 1.0`, точка должна быть явно помечена как `subspread_target=true`.

### 326.5: Добавить mini-train sweep на top-k кандидатов

**Файлы:**
- `python_lab/src/train.py`
- `python_lab/src/train_data.py`

**Что делать:**

После dry-run grid отбирать `top-k` кандидатов по фильтрам:

- `subspread_target = false`
- разумный `trade_share`
- разумный class balance

Для этих кандидатов запускать короткое обучение на одной и той же конфигурации LiT и сохранять:

- `mini_train_mcc`
- `mini_train_coverage_directional`
- `mini_train_net_edge_total`

### 326.6: Сохранить результаты в CSV и JSON

**Файлы:**
- `python_lab/src/train_data.py`
- `docs/train_logs.md`

**Артефакты:**
- `docs/sweep_baseline.csv`
- `docs/sweep_baseline.json`
- `docs/baselines.md` update

## Верификация

1. В sweep-таблице есть `h=10,20,50,100`.
2. Для `h=100,thr=0.0005` видно около `65.3/16.7/18.0`.
3. Для `h=100,thr=0.0015` видно возврат к `~90%+ Flat`.
4. Для `thr=0.0005` видно `subspread_target=true`.
5. Dynamic threshold reference сохранён отдельно и не подменяет static grid.

## Критерии приёмки

1. Sweep работает без запуска обычного train.
2. Есть CSV и JSON с полной grid-статистикой.
3. В отчёте есть `threshold_to_spread_ratio`.
4. Есть shortlist кандидатов для 327.

## Что не делать

- Не менять LiT architecture.
- Не менять channels.
- Не включать новый target в основной train на этой задаче.
