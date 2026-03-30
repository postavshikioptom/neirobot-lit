# Задача 332: Execution-Aware Recalibration (без смены LiT)

## Почему текущие метрики слабые (по `output.txt`)
1. `val_mcc_primary` деградирует по эпохам: `0.0110 -> 0.0044 -> -0.0031`.
2. При этом `Macro-F1` растёт (`0.2110 -> 0.3144 -> 0.3239`), то есть оптимизация уходит в «красивый F1», но не в устойчивый directional signal.
3. `coverage_directional` падает (`0.9434 -> 0.5978 -> 0.5029`), а `DA без Flat` остаётся низким (`~0.17`), `net_edge_total` около нуля.
4. В `python_lab/src/labels.py` и `python_lab/src/train_cli.py` execution-aware режим есть, но по умолчанию costs/spread-floor отключены (`cost_floor_bps=fee_bps=slippage_bps=0`, `use_spread_floor=False`), decision rule остаётся `argmax`.

Вывод: главная причина не в архитектуре LiT, а в несоответствии `label contract + decision rule` реальной торговой постановке.

## Одна ключевая задача 332
Привести `execution_mid_return` и правило принятия сигнала к cost-aware контракту, чтобы оптимизация шла в `val_mcc_primary` и положительный `net_edge_total`, а не только в macro-F1.

## Подзадачи (малые, последовательные)
1. Зафиксировать baseline-запуск 331 как контрольную точку.
Файлы: `output.txt`, `docs/train_logs.md`.

2. Добавить отдельный профиль `task332_execution_recalibration`.
Файл: `python_lab/src/train_cli.py`.
Профиль должен фиксировать: `label_mode=execution_mid_return`, `split_strategy=purged_holdout`, `decision_rule` не `argmax` (для абляции), монитор `val_mcc_primary` остаётся прежним.

3. Включить cost-aware порог в labels.
Файлы: `python_lab/src/train_cli.py`, `python_lab/src/train_data.py`, `python_lab/src/labels.py`.
Минимум: активировать `use_spread_floor` и ненулевые `fee_bps/slippage_bps` для label generation.

4. Провести узкую абляцию decision-rule без изменения модели.
Файлы: `python_lab/src/train_cli.py`, `python_lab/src/train_module.py`.
Сетка: `argmax`, `confidence_gap`, `class_specific_thresholds`, `flat_bias` (в рамках уже существующей реализации).

5. Подобрать пороги только в малом диапазоне вокруг текущего.
Файлы: `python_lab/src/train_data.py`, `python_lab/src/train.py`.
Использовать существующий sweep-контур из задач 326/328, но ограничить кандидаты, чтобы не расползаться в широкий ресёрч.

6. Ввести quality-gate после эпохи/запуска.
Файлы: `python_lab/src/train.py`, `python_lab/src/train_module.py`.
Run считается приемлемым только если одновременно:
`val_mcc_primary` растёт, `coverage_directional` не уходит в ноль, `net_edge_total >= 0`.

7. Обновить отчётность под контракт 332.
Файлы: `docs/train_logs.md`, `python_lab/src/baselines.md`.
Фиксировать по каждому запуску: `val_mcc_primary`, `macro_f1`, `coverage_directional`, `da_without_flat`, `net_edge_total`, `decision_rule`, `effective_threshold`.

8. Зафиксировать финальный «production-safe» набор флагов 332.
Файлы: `python_lab/src/baselines.md`, `docs/000-tasks_list.md` (при необходимости регистрации выполнения).

## Критерий успеха
1. На том же символе и том же датасете `best val_mcc_primary >= 0.021` (минимум +0.010 к текущему baseline `0.0110`).
2. `coverage_directional >= 0.18` и не деградирует ступенчато по эпохам.
3. `net_edge_total >= 0` на лучшей эпохе.
4. Без изменения архитектуры (`python_lab/src/lit_model.py`, `python_lab/src/layers.py` не меняются).

## Внешние ориентиры
1. LiT paper (feature/label/split практики): https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1616485/full
2. TLOB (horizon-bias и cost-aware threshold обсуждение): https://ar5iv.labs.arxiv.org/html/2502.15757
3. Deep LOB microstructural guide (event-time, split, leakage): https://discovery.ucl.ac.uk/id/eprint/10218102/1/Deep%20limit%20order%20book%20forecasting%20a%20microstructural%20guide.pdf
4. PyTorch Lightning callbacks (monitor `val_mcc_primary`): https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html
