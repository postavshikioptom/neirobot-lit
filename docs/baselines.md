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
