# Baseline Sweep

Этот файл заполняется и перезаписывается sweep runner'ом из `python_lab/src/train.py`.

Ожидаемые разделы:

- `Static Grid` с полями `share_flat`, `share_up`, `share_down`, `trade_share`,
  `row_time_seconds`, `event_time_seconds`, `median_spread_bps`,
  `threshold_bps`, `threshold_to_spread_ratio`, `subspread_target`.
- `Dynamic Threshold Reference` как отдельный unsafe reference для текущей схемы
  `rolling_std * 0.5` при `h=100`.
- `Shortlist` top-k кандидатов для следующей задачи с полями
  `mini_train_mcc`, `mini_train_coverage_directional`, `mini_train_net_edge_total`.

Важно:

- `dynamic_threshold_reference` не должен подменять static grid и не становится новым default.
- Если `threshold_to_spread_ratio < 1.0`, строка baseline должна быть помечена как `subspread_target=true`.

## Objective Ablation

Objective ablation сохраняется в `objective_ablation.csv` в корне репозитория.

Ожидаемые колонки:

- `loss_type`
- `class_weights`
- `multi_task`
- `mcc_primary`
- `coverage_directional`
- `net_edge_total`
- `ece`

## Stable Artifact Tree (Задача 331)

Единая структура артефактов для stable pipeline:

- `artifacts/<symbol>/labels`
- `artifacts/<symbol>/validation`
- `artifacts/<symbol>/calibration`
- `artifacts/<symbol>/attribution`
- `artifacts/<symbol>/walk_forward`

## Pipeline State (Задача 331)

Для каждого baseline-запуска фиксируется:

- `profile`
- `frozen_branches`
- `metrics_contract` и `metrics_contract_version`
- `label_contract_mode` и `label_contract_version`
- `split_strategy`

## Task 332 Execution Recalibration

- Baseline control point (from task 331): `val_mcc_primary=0.0110`, `macro_f1=0.3239`, `coverage_directional=0.5029`, `da_without_flat=0.1720`, `net_edge_total~0`.
- Label contract: `label_mode=execution_mid_return`, `use_spread_floor=true`, non-zero `cost_floor_bps/fee_bps/slippage_bps`.
- Decision-rule ablation (same model, no architecture change): `argmax`, `confidence_gap`, `class_specific_thresholds`, `flat_bias`.
- Narrow threshold sweep: local band around current `--threshold` (no broad research grid).
- Quality-gate (run acceptable only if all true): MCC growth, non-zero/min coverage, non-negative net edge.
- Report fields per run: `val_mcc_primary`, `val_f1_macro_np`, `coverage_directional`, `val_da_without_flat`, `net_edge_total`, `decision_rule`, `effective_threshold`, `quality_gate.passed`.

### Production-safe flags (332)

```bash
python -m python_lab.src.train \
  --profile task332_execution_recalibration \
  --decision_rule flat_bias \
  --decision_rule_ablation \
  --quality_gate_enabled \
  --quality_gate_min_coverage_directional 0.18 \
  --quality_gate_require_non_negative_net_edge \
  --quality_gate_require_mcc_growth
```
