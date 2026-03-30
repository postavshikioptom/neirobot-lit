# Задача 333: python-lit-target-coverage-threshold-calibration

## Главная задача (ОДНА)
Внедрить **автокалибровку decision-threshold на валидации под целевой `coverage_directional` с учетом `net_edge_total`**, чтобы убрать деградацию `val_mcc_primary` после ранних эпох и стабилизировать execution-quality без изменения архитектуры LiT.

## Почему это нужно (анализ проблемы)
1. По запуску из `output.txt` (дата 2026-03-30):
- Epoch 1: `val_mcc_primary=0.0110`, `coverage_directional=0.9434`, `ECE=0.3277`.
- Epoch 5: `val_mcc_primary=0.0044`, `coverage_directional=0.5978`.
- Epoch 10: `val_mcc_primary=-0.0031`, `coverage_directional=0.5029`.
2. В логах `conf_wrong > conf_correct`, то есть текущие фиксированные пороги decision-rule не удерживают качественную зону сигналов.
3. В задаче 332 уже сделаны execution-aware labels + quality-gate + ablation правил, но пороги (`decision_hold_threshold`, `margin_threshold`, class thresholds) остаются статичными и не адаптируются к текущему распределению уверенности.

## Ограничения
1. **Не менять** `python_lab/src/lit_model.py` и `python_lab/src/layers.py`.
2. Не менять архитектуру проекта и не переходить на другую модель.
3. Все правки только в training/validation contract и CLI.
4. **Не создавать новый baseline**: использовать уже существующий baseline-контракт из `python_lab/src/baselines.md` и `docs/train_logs.md` как frozen reference.
5. Не дублировать baseline-секции; в `python_lab/src/baselines.md` только обновлять/дополнять блоки сравнения для задачи 333.

## Подзадачи (конкретные правки)

### 1) Расширить CLI-контракт для задачи 333
**Файл:** `python_lab/src/train_cli.py`

1. В константы профилей добавить:
- `TASK333_PROFILE = "task333_target_coverage_threshold"`.
2. В `PROFILE_OVERRIDES` добавить блок `TASK333_PROFILE`:
- `profile`: single-horizon 100, `label_mode=execution_mid_return`, `decision_rule=flat_bias`, `freeze_experimental_features=True`.
- Включить новый режим калибровки порогов (см. ниже).
3. В `build_train_parser()` добавить новые флаги:
- `--decision_threshold_calibration` (`choices=["off","target_coverage"]`, default `off`)
- `--decision_threshold_target_coverage` (float, default `0.35`)
- `--decision_threshold_target_tolerance` (float, default `0.05`)
- `--decision_threshold_min_coverage` (float, default `0.18`)
- `--decision_threshold_max_coverage` (float, default `0.75`)
- `--decision_threshold_opt_metric` (`choices=["net_edge_total","val_mcc_primary"]`, default `net_edge_total`)
- `--decision_threshold_quantiles` (str, default `"0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90"`)
4. В `validate_train_args(...)` добавить проверки:
- все coverage-параметры в `[0,1]`;
- `target_tolerance > 0`;
- `min_coverage <= target_coverage <= max_coverage`;
- quantiles распарсить в sorted unique list, каждый в `(0,1)`.

**Что заменить:**
- Текущая логика только статических порогов (`decision_hold_threshold`, `margin_threshold`, `*_prob_threshold`) остается как fallback.
- Новая логика активируется только при `--decision_threshold_calibration target_coverage`.

---

### 2) Добавить утилиту автоподбора порога по валидации
**Файл:** `python_lab/src/utils.py`

1. Добавить новую функцию:
`def calibrate_decision_thresholds_for_target_coverage(...):`

Сигнатура (примерно):
- `probs: np.ndarray` (N,3)
- `y_true: np.ndarray`
- `logits: np.ndarray`
- `f_ret: np.ndarray`
- `imbalance: np.ndarray`
- `rule: str`
- `base_params: dict`
- `target_coverage: float`
- `target_tolerance: float`
- `min_coverage: float`
- `max_coverage: float`
- `quantiles: list[float]`
- `opt_metric: str`
- `directional_base: str`
- `fee_bps/slippage_bps/half_spread_bps`

2. Внутри функции:
- строить набор кандидатов порога `tau` через квантили по `top1_prob` (или по `max(p_up,p_down)` для `flat_bias`);
- для каждого `tau` прогонять `apply_decision_rule(...)` с override параметром;
- считать метрики через существующий `compute_directional_metrics(...)` + `safe_matthews_corrcoef(...)`;
- отфильтровать кандидаты по коридору покрытия: `[max(min_coverage, target_coverage-target_tolerance), min(max_coverage, target_coverage+target_tolerance)]`;
- выбрать лучший кандидат по `opt_metric` (`net_edge_total` либо `val_mcc_primary`), tie-breaker: ближе к target coverage, затем выше `net_edge_total`.
- если валидных кандидатов нет, вернуть fallback со статическими параметрами.

3. Обновить `apply_decision_rule(...)`:
- добавить опциональный аргумент `threshold_overrides: dict | None`;
- если overrides передан, подменять `decision_confidence/decision_hold_threshold/margin_threshold/*_prob_threshold` локально без изменения старого API.

4. Возвращаемый объект калибратора (dict):
- `selected_thresholds`
- `selected_metrics`
- `target_coverage`
- `coverage_error`
- `used_fallback`
- `candidate_table` (top-N кратко)

---

### 3) Подключить автокалибровку в validation epoch-end
**Файл:** `python_lab/src/train_module.py`

1. В `on_validation_epoch_end()` в single-horizon и multi-horizon ветках:
- после temperature scaling (`scaled_probs`) и до расчета `decision_metrics` вставить вызов новой функции `calibrate_decision_thresholds_for_target_coverage(...)`, если `self.hparams.decision_threshold_calibration == "target_coverage"`.
2. Использовать `selected_thresholds` для получения `decision_pred` (через `apply_decision_rule(..., threshold_overrides=...)`).
3. Логировать новые метрики:
- `decision_threshold_auto_applied` (0/1)
- `decision_threshold_target_coverage`
- `decision_threshold_selected_coverage`
- `decision_threshold_coverage_error`
- `decision_threshold_used_fallback` (0/1)
- `decision_threshold_selected_confidence` (или hold threshold)
4. В `finalized` добавить раздел:
- `decision_threshold_calibration: {mode, target, selected_thresholds, selected_metrics, used_fallback, candidate_table_topk}`
5. В JSON-артефакт `validation_report_epoch_{epoch}.json` сохранить новый раздел без сокращений (чтобы можно было анализировать в `docs/train_logs.md`).

**Что именно меняется в поведении:**
- Сейчас: один фиксированный порог на весь run.
- После 333: порог пересчитывается на каждой validation-эпохе на основе текущего распределения confidence и торговых метрик.

---

### 4) Привязать profile 333 и отчётность
**Файл:** `python_lab/src/train.py`

1. В startup invariant checks добавить проверку:
- для `profile == task333_target_coverage_threshold` обязательно:
  - `decision_threshold_calibration == target_coverage`
  - `decision_rule != argmax`
  - `quality_gate_enabled == True`
2. В `_build_pipeline_state(...)` добавить поля:
- `decision_threshold_calibration_mode`
- `decision_threshold_target_coverage`
3. В `_append_task332_run_report(...)` не менять старый блок; добавить новый аналог:
- `_append_task333_run_report(...)` c полями:
  - `decision_threshold_calibration`
  - `target_coverage`
  - `selected_thresholds`
  - `coverage_error`
  - `val_mcc_primary`
  - `net_edge_total`
  - `quality_gate`
4. Записывать отчёт одновременно в:
- `docs/train_logs.md`
- `python_lab/src/baselines.md`
5. Явно зафиксировать правило в отчёте:
- baseline берётся из уже существующего `python_lab/src/baselines.md` (задача 331/332),
- новый baseline-run не создаётся, выполняется только сравнение `baseline vs calibrated`.

---

### 5) Добавить тесты на новый контракт
**Файлы:**
- `python_lab/tests/test_decision_threshold_calibration.py` (новый)
- `python_lab/tests/test_train_data_purged_holdout.py` (точечное дополнение при необходимости импорта)

Тест-кейсы:
1. `test_threshold_calibration_hits_target_coverage_band`:
- synthetic probs + labels, целевой coverage 0.40 ± 0.05;
- assert выбранный порог попал в коридор и `used_fallback=False`.
2. `test_threshold_calibration_fallback_when_no_candidate`:
- задать невозможный коридор coverage;
- assert `used_fallback=True`, предсказания равны old static behavior.
3. `test_apply_decision_rule_threshold_overrides_backward_compatible`:
- без overrides и с overrides;
- verify старый вызов не ломается.

---

## Команды проверки после внедрения
1. Unit tests:
`python -m pytest python_lab/tests/test_decision_threshold_calibration.py -q`
2. Sanity train (1-2 эпохи):
`python -m python_lab.src.train --profile task333_target_coverage_threshold --symbol BTCUSDT --epochs 2`
3. Проверить артефакты:
- `artifacts/BTCUSDT/validation/validation_report_epoch_*.json`
- `docs/train_logs.md`

## Критерий готовности задачи 333
1. На baseline-датасете `val_mcc_primary` не уходит в отрицательную область к 10 эпохе (минимум: `>= 0.0`).
2. `coverage_directional` удерживается в целевом коридоре (`target ± tolerance`) минимум на 70% эпох.
3. `net_edge_total` на best-epoch не отрицательный.
4. Нет изменений в LiT-архитектуре (`lit_model.py`, `layers.py` untouched).
5. В артефактах нет отдельного "нового baseline"; есть только сравнение с уже существующим baseline.

## Внешние ориентиры (использованы при планировании)
1. Calibrated selective classification (threshold tuning под target coverage):
https://arxiv.org/html/2208.12084v2
2. Практический код calibrate selector threshold:
https://github.com/ajfisch/calibrated-selective-classification
3. PyTorch Lightning EarlyStopping/ModelCheckpoint contracts:
https://lightning.ai/docs/pytorch/stable/common/early_stopping.html
https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html
4. TorchMetrics CalibrationError (ECE/MCE):
https://lightning.ai/docs/torchmetrics/stable/classification/calibration_error.html
5. Trading-oriented confidence threshold sweep (competitive pattern):
https://github.com/KuznetsovKarazin/crypto-confidence-execution
