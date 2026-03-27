# Задача 325: Починить контракт метрик и валидации для LiT

**Дата создания:** 27.03.2026  
**Статус:** ПЛАН  
**Категория:** Обучение Python  
**Зависимости:** 324-python-train-stabilization-dynamic-ofi-delta

## Цель

Привести validation-контур к одному честному контракту измерений. До любых новых экспериментов нужно убрать двойное логирование MCC, правильно считать short-edge, отделить quality-метрики от trade-метрик и зафиксировать один primary monitor key для checkpoint/early stopping.

## Контекст и проблема

Сейчас в [python_lab/src/train_module.py](D:\MAX\PYTHON\NEURAL-BOTS\neirobot-lit\python_lab\src\train_module.py):

- `val_mcc` логируется и в `validation_step`, и в `on_validation_epoch_end`;
- `edge_down` считается без sign flip для short;
- `DA без Flat` печатается без coverage и без baseline-интерпретации;
- `Hit Rate` по факту ближе к precision, а не к полному class report;
- нет `net_edge_after_costs`;
- `ModelCheckpoint`/`EarlyStopping` не зафиксированы на одном primary key.

На текущем качестве это особенно критично: MCC около `0.0189`, DA без Flat около `0.1173`, поэтому даже маленькая логическая ошибка в метриках искажает всю картину.

## Подзадачи

### 325.1: Зафиксировать один metric contract и передавать его через CLI

**Файлы:**
- `python_lab/src/train_cli.py`
- `python_lab/src/train_module.py`

**Что добавить в CLI:**
- `--metric_contract` with choices `standard`, `hft`, `strict`
- `--metric_log_prefix`
- `--metric_directional_base`
- `--report_fee_bps`
- `--report_slippage_bps`
- `--report_half_spread_bps`

**Требование:**

Все эти поля должны попадать в `self.hparams` и сохраняться в checkpoint, чтобы validation contract можно было воспроизвести.

### 325.2: Убрать двойной источник правды для MCC/F1

**Файл:** `python_lab/src/train_module.py`

**Что поменять:**

1. Удалить прямое логирование `val_mcc` и `val_f1_macro` из `validation_step`.
2. В `validation_step` только накапливать:
   - `logits`
   - `labels`
   - `f_ret`
   - `imbalance`
   - `regime_id`
3. Ввести helper:
   - `_accumulate_validation_outputs(...)`
   - `_finalize_validation_metrics(...)`
4. Считать все epoch-метрики только в `on_validation_epoch_end`.

**Ключевая замена:**

```python
# БЫЛО
self.log("val_mcc", self.mcc(logits, y), ...)

# СТАЛО
self._validation_accumulator.append(
    {
        "logits": logits.detach().cpu(),
        "labels": y.detach().cpu(),
        "f_ret": f_ret.detach().cpu(),
        "imbalance": imbalance.detach().cpu(),
    }
)
```

### 325.3: Ввести один primary monitor key

**Файлы:**
- `python_lab/src/train.py`
- `python_lab/src/train_cv.py`
- `python_lab/src/train_module.py`

**Что поменять:**

1. Ввести единый ключ `val_mcc_primary`.
2. Все `ModelCheckpoint` и `EarlyStopping` должны мониторить только его.
3. Вспомогательные поля логировать отдельно:
   - `val_mcc_torch`
   - `val_mcc_np`
   - `val_direction_mcc`
   - `val_f1_macro_torch`
   - `val_f1_macro_np`

### 325.4: Исправить semantics short-edge и добавить cost-aware edge

**Файлы:**
- `python_lab/src/train_module.py`
- `python_lab/src/utils.py`

**Что поменять:**

1. Вынести helper `compute_directional_metrics(...)`.
2. Для `Down` считать:

```python
edge_down_signed = np.mean(-f_ret[preds == 2]) if np.any(preds == 2) else 0.0
```

3. Добавить:
   - `coverage_directional`
   - `coverage_long`
   - `coverage_short`
   - `trade_count_total`
   - `trade_count_long`
   - `trade_count_short`
   - `gross_edge_long`
   - `gross_edge_short`
   - `gross_edge_total`
   - `net_edge_long`
   - `net_edge_short`
   - `net_edge_total`

**Формула costs:**

```python
roundtrip_cost = (2 * half_spread_bps + 2 * fee_bps + slippage_bps) / 10000.0
net_edge_long = gross_edge_long - roundtrip_cost
net_edge_short = gross_edge_short - roundtrip_cost
```

### 325.5: Разделить precision/recall и убрать двусмысленный `Hit Rate`

**Файлы:**
- `python_lab/src/train_module.py`
- `python_lab/src/utils.py`

**Что поменять:**

Вместо одного `Hit Rate` логировать отдельно:

- `precision_flat`, `precision_up`, `precision_down`
- `recall_flat`, `recall_up`, `recall_down`

Поля вида `false_up`/`false_down` оставить только как вторичную market-specific диагностику, а не как основную class table.

### 325.6: Сохранить единый epoch validation report

**Файл:** `python_lab/src/train_module.py`

**Что добавить:**

На каждой эпохе сохранять JSON-артефакт:

`artifacts/<symbol>/validation/validation_report_epoch_{epoch}.json`

Минимальная структура:

```json
{
  "epoch": 0,
  "quality": {},
  "calibration": {},
  "coverage": {},
  "trade": {},
  "class_metrics": {},
  "regime_metrics": {}
}
```

### 325.7: Обновить docs/train_logs.md под новый contract

**Файл:** `docs/train_logs.md`

**Что добавить:**

Описание полей:

- `val_mcc_primary`
- `val_direction_mcc`
- `val_da_without_flat`
- `coverage_directional`
- `gross_edge_total`
- `net_edge_total`
- `val_ece`
- `val_mce`

## Верификация

1. Запустить 1 эпоху и убедиться, что `val_mcc` больше не логируется в `validation_step`.
2. Проверить, что `val_mcc_primary` существует ровно один раз на эпоху.
3. Проверить, что `edge_down_signed` меняет знак корректно.
4. Проверить, что JSON-отчёт по эпохе создаётся.
5. Проверить, что `coverage_directional` и `net_edge_total` присутствуют в логе и JSON.

## Критерии приёмки

1. Есть один primary metric key: `val_mcc_primary`.
2. Все epoch metrics считаются в одном месте.
3. `edge_down` исправлен на short-edge.
4. В отчёте есть coverage и cost-aware edge.
5. Таблица class metrics разделяет precision и recall.

## Что не делать

- Не менять labels.
- Не менять LiT architecture.
- Не добавлять новые каналы.
