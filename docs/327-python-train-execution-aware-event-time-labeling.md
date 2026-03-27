# Задача 327: Пересобрать labels в execution-aware и event-time contract

**Дата создания:** 27.03.2026  
**Статус:** ПЛАН  
**Категория:** Обучение Python  
**Зависимости:** 326-python-train-horizon-threshold-sweep-baseline

## Цель

Перевести labels от legacy row-shift разметки к новому контракту, который учитывает:

- `label_mode`
- `time_mode`
- spread floor
- fee/slippage floor
- event-time или physical-time indexing

Задача должна решить не только проблему времени, но и проблему неторгуемого target внутри спреда.

## Контекст и проблема

Сейчас `Labeler` делает простое `.shift(-horizon)` и сравнивает future mid return с fixed threshold. Проблемы:

- horizon выражен в строках, а не в событиях/мс;
- threshold не учитывает spread, fee, slippage;
- target часто живёт внутри спреда;
- текущий dynamic threshold `rolling_std * K` локально даёт около `21/39/39` и не годится как новый default.

## Подзадачи

### 327.1: Расширить `Labeler` до нового contract API

**Файлы:**
- `python_lab/src/labels.py`
- `python_lab/src/train_cli.py`
- `python_lab/src/train_data.py`

**Что добавить в `Labeler.__init__`:**
- `label_mode`
- `time_mode`
- `event_time_column`
- `cost_floor_bps`
- `fee_bps`
- `slippage_bps`
- `use_spread_floor`

**Обязательные режимы:**
- `label_mode="legacy_mid_return"`
- `label_mode="execution_mid_return"`

**time_mode:**
- `row`
- `event`
- `ms`

### 327.2: Реализовать `effective_threshold`

**Файл:** `python_lab/src/labels.py`

**Новая логика:**

В execution-aware режиме использовать:

```python
threshold_static = self.threshold
threshold_spread = spread_bps / 10000.0 if use_spread_floor else 0.0
threshold_cost = (cost_floor_bps + 2 * fee_bps + slippage_bps) / 10000.0
effective_threshold = max(threshold_static, threshold_spread, threshold_cost)
```

Именно `effective_threshold`, а не голый static threshold, должен участвовать в разметке.

### 327.3: Реализовать event/ms future index lookup

**Файл:** `python_lab/src/labels.py`

**Что добавить:**

Helper:

```python
def _map_future_indices(df, horizon_value, time_mode):
    ...
```

**Режимы:**

- `row`: старое поведение
- `event`: переход по update events
- `ms`: поиск будущего индекса по timestamp + horizon_ms

Важно: не смешивать event/time semantics с догадкой, что `h=10` означает `10 секунд`. Это разные единицы измерения.

### 327.4: Явно запретить новый label mode + старый dynamic threshold

**Файлы:**
- `python_lab/src/labels.py`
- `python_lab/src/train_cli.py`

**Требование:**

Если пользователь запускает:

```bash
--label_mode execution_mid_return --dynamic_threshold True
```

код должен падать с `ValueError`. Старый `dynamic_threshold` разрешён только в legacy/debug режиме.

### 327.5: Согласовать class weights с новым label contract

**Файлы:**
- `python_lab/src/train_data.py`
- `python_lab/src/train_module.py`

**Что сделать:**

После смены labels class weights считать только:

- по train split
- по тем же label columns, что реально подаются в dataset

Если class weights посчитаны по legacy labels, а датасет использует execution labels, запуск должен падать.

### 327.6: Добавить diagnostics и verification script

**Файлы:**
- `python_lab/src/labels.py`
- `scripts/event_time_label_check.py`
- `docs/train_logs.md`

**Что логировать:**
- `label_mode`
- `time_mode`
- `effective_threshold_p50`
- `effective_threshold_p95`
- `row_gap_median_seconds`
- `event_gap_median_seconds`

**Скрипт проверки должен сравнивать:**
- `label_row`
- `label_event`
- `label_exec`

и печатать:
- `counts(row)`
- `counts(event)`
- `counts(exec)`
- `num_different`

## Верификация

1. Запустить `--label_mode execution_mid_return --time_mode event`.
2. Проверить, что появляются execution-aware label columns.
3. Проверить, что `effective_threshold` логируется.
4. Проверить, что сочетание нового label mode и старого `dynamic_threshold` падает с ошибкой.
5. Проверить, что verification script показывает различие между legacy и execution labels.

## Критерии приёмки

1. Новый label contract поддерживает `label_mode` и `time_mode`.
2. В execution-aware режиме используется `effective_threshold`.
3. Event/ms indexing заменяет слепой row-shift там, где требуется.
4. Старый dynamic threshold не может тихо стать частью нового contract.

## Что не делать

- Не менять LiT architecture.
- Не менять channels.
- Не подменять execution-aware labels старым `rolling_std * K`.
