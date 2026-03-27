
## Задача 320 | Дата: 2026-03-23 | Эпох: 0 (валидация не завершена)

### Изменения (из docs/000-tasks_list.md):
Устранить silent-stall после 319 на границе train→validation. Добавить phase-логи, guards для sanity-check и тяжелых epoch-end артефактов, вынести DataLoader/Trainer knobs в CLI, исправить split full_dataset/val/test на eval-mode и добавить диагностику saturation каналов до/после clamp.

### Использованные каналы (11):
0: MicropriceDev, 1: Vol, 2: Imb, 3: OFI, 4: VIB, 5: Ret_10, 6: Ret_50, 7: Ret_100, 8: Spread, 9: DeltaImb, 10: DeltaSpread

### Лучшие метрики:
- MCC: N/A (epoch N/A)
- Macro-F1: N/A (epoch N/A)
- DA: N/A
- Hit Rate Up: N/A

### Стартовая статистика каналов (до завершения Epoch 0, sample idx=100):
| Канал | mean | std | min | max |
|-------|------|-----|-----|-----|
| MicropriceDev | 0.0287 | 0.2532 | -0.4134 | 0.3673 |
| Vol | 0.3170 | 0.6939 | -2.3208 | 2.3194 |
| Imb | 0.0456 | 0.5653 | -1.6142 | 3.8219 |
| OFI | -0.0126 | 1.6742 | -3.0000 | 3.0000 |
| VIB | 0.2431 | 0.3326 | -0.3540 | 0.8900 |
| Ret_10 | 0.1466 | 0.3418 | -0.7849 | 1.0469 |
| Ret_50 | 0.5327 | 0.4242 | 0.0000 | 1.2702 |
| Ret_100 | 0.5133 | 0.4493 | -0.4097 | 0.8708 |
| Spread | 1.2839 | 0.4149 | 0.9494 | 2.5154 |
| DeltaImb | -0.1500 | 1.7170 | -3.0000 | 3.0000 |
| DeltaSpread | -0.0600 | 1.2715 | -3.0000 | 3.0000 |

### Аномалия скорости:
- Текущий запуск не завис: в `Epoch 0/49` прогресс дошел до `176/4055` за `0:11:12`, скорость держится около `0.26 it/s`, ETA около `4:08:00`.
- Это радикально медленнее исторических запусков из сохраненных логов: `4m19s` (после 315-4), `5m29s-5m31s` (после 305), `8m51s-8m54s` (после 318).
- В логе есть предупреждения Lightning про `train_dataloader`/`val_dataloader` с низким `num_workers`.
- Дополнительно видны `Non-finite gradient` в `model.vol_regressor.*` и экстремальные сырые значения OFI/Delta-каналов до clamp.

## Задача 321 | Дата: 2026-03-23 | Эпох: 0 (валидация не завершена)

### Изменения (из docs/000-tasks_list.md):
N/A. В `output.txt` указано `ПОСЛЕ ЗАДАЧИ 321`, но задача `321` пока отсутствует в `docs/000-tasks_list.md`. Для анализа использован фактический лог текущего запуска.

### Использованные каналы (11):
0: MicropriceDev, 1: Vol, 2: Imb, 3: OFI, 4: VIB, 5: Ret_10, 6: Ret_50, 7: Ret_100, 8: Spread, 9: DeltaImb, 10: DeltaSpread

### Лучшие метрики:
- MCC: N/A (epoch N/A)
- Macro-F1: N/A (epoch N/A)
- DA: N/A
- Hit Rate Up: N/A

### Стартовая статистика каналов (до завершения Epoch 0, sample idx=100):
| Канал | mean | std | min | max |
|-------|------|-----|-----|-----|
| MicropriceDev | 0.0287 | 0.2532 | -0.4134 | 0.3673 |
| Vol | 0.3170 | 0.6939 | -2.3208 | 2.3194 |
| Imb | 0.0456 | 0.5653 | -1.6142 | 3.8219 |
| OFI | -0.0126 | 1.6742 | -3.0000 | 3.0000 |
| VIB | 0.2431 | 0.3326 | -0.3540 | 0.8900 |
| Ret_10 | 0.1466 | 0.3418 | -0.7849 | 1.0469 |
| Ret_50 | 0.5327 | 0.4242 | 0.0000 | 1.2702 |
| Ret_100 | 0.5133 | 0.4493 | -0.4097 | 0.8708 |
| Spread | 1.2839 | 0.4149 | 0.9494 | 2.5154 |
| DeltaImb | -0.1500 | 1.7170 | -3.0000 | 3.0000 |
| DeltaSpread | -0.0600 | 1.2715 | -3.0000 | 3.0000 |

### Критические наблюдения:
- В текущем запуске снова нет завершенной валидации, поэтому качество модели пока сравнить нельзя.
- В логе сохраняются экстремальные сырые значения перед clamp: `OFI` до `~1.69e8`, `DeltaImb` до `~7.67e7`, `DeltaSpread` до `~3.34e5`.
- После clamp насыщение остается заметным: `OFI` около `31.14%`, `DeltaImb` около `33.00%`, `DeltaSpread` около `18.00%`.
- В хвосте лога появился новый симптом: в `Sample Normalized Tensor` заметен большой нулевой блок, после чего есть прямое замечание `вижу нули здесь в данных почемуто появились, раньше не было`.
- Контекстно это продолжение регрессии после задач `319-320`: проблема остается не в полном зависании, а в аномально медленном и, вероятно, численно нестабильном старте эпохи.

## Задача 322 | Дата: 2026-03-23 | Эпох: N/A (ревью рефакторинга, не лог обучения)

### Изменения (из docs/000-tasks_list.md):
N/A. В `docs/000-tasks_list.md` задача `322` не найдена; описание взято из `output.txt` (`ПОСЛЕ ЗАДАЧИ 322`).

### Использованные каналы:
N/A

### Лучшие метрики:
- MCC: N/A (epoch N/A)
- Macro-F1: N/A (epoch N/A)
- DA: N/A
- Hit Rate Up: N/A

### Ключевые недоделки и риски для сохранения поведения:
- `train_module.py` все еще содержит дубликат `enable_dropout`; должен остаться только в `train_postprocess.py`.
- Конфиг горизонтов еще не централизован: `train.py` не вызывает `resolve_horizon_config`, а `train_data.py` держит дублирующий `_resolve_horizons`.
- Это критично для поведения, потому что дубликат в `train_data.py` не нормализует `horizon_weights` и не выставляет `equal weights` по умолчанию.
- `prepare_training_data(...)` пока не принимает `horizons`, `num_horizons`, `horizon_weights`, поэтому orchestration еще не полностью вынесен вверх.
- `update_model_metadata(...)` все еще вызывается внутри `train_data.py`, а не из `train.py`; side effect не вынесен в entrypoint.

### Что важно не сломать:
- Сохранить прежнюю семантику multi-horizon: нормализация весов и `equal weights` по умолчанию.
- Сохранить порядок этапов: parse CLI -> resolve horizons -> prepare data -> fit/save normalizer -> update metadata -> training/postprocess.
- Не дублировать `enable_dropout` и другие post-training side effects между модулями.

## Задача 324 | Дата: 2026-03-25 | Эпох: 10

### Изменения (из docs/000-tasks_list.md):
Стабилизировать dynamic feature contract OFI, DeltaImb, DeltaSpread для LiT. Перевести DeltaImb и DeltaSpread на тот же event-consistent источник, что и OFI. Переделать fit dynamic-normalizer. Синхронизировать train-fit

### Использованные каналы (11):
0: MicropriceDev, 1: Vol, 2: Imb, 3: OFI, 4: VIB, 5: Ret_10, 6: Ret_50, 7: Ret_100, 8: Spread, 9: DeltaImb, 10: DeltaSpread

### Лучшие метрики:
- MCC: 0.0189 (epoch 10)
- Macro-F1: 0.3310 (epoch 10)
- DA: 0.1173 (epoch 10)
- Hit Rate Up: 11.13% (epoch 10)

### Статистика каналов (после CLAMP, усреднение по последней эпохе):
| Канал | mean | std | min | max |
|-------|------|-----|-----|-----|
| MicropriceDev | 0.0314 | 0.2516 | -0.4134 | 0.3673 |
| Vol | 0.3166 | 0.6947 | -2.3208 | 2.3194 |
| Imb | 0.0464 | 0.5670 | -1.6142 | 3.8219 |
| OFI | -0.1268 | 0.5251 | -1.3288 | 1.6212 |
| VIB | 0.2453 | 0.3314 | -0.3540 | 0.8900 |
| Ret_10 | 0.1518 | 0.3474 | -0.7849 | 1.0469 |
| Ret_50 | 0.5006 | 0.4139 | 0.0000 | 1.2702 |
| Ret_100 | 0.5502 | 0.3984 | -0.4097 | 0.8708 |
| Spread | 1.2600 | 0.4027 | 0.9494 | 2.5154 |
| DeltaImb | 0.0220 | 0.6669 | -3.2487 | 3.2439 |
| DeltaSpread | -0.0001 | 0.0246 | -0.0669 | 0.1284 |

---

[Сюда будут добавляться новые записи]

## Задача 324 | Дата: 2026-03-26 | Эпох: 10

### Изменения (из docs/000-tasks_list.md):
Стабилизировать dynamic feature contract OFI, DeltaImb, DeltaSpread для LiT. Перевести DeltaImb и DeltaSpread на тот же event-consistent источник, что и OFI. Переделать fit dynamic-normalizer. Синхронизировать train-fit

### Использованные каналы (11):
0: MicropriceDev, 1: Vol, 2: Imb, 3: OFI, 4: VIB, 5: Ret_10, 6: Ret_50, 7: Ret_100, 8: Spread, 9: DeltaImb, 10: DeltaSpread

### Эпохи и метрики:
- Epoch 1: MCC=-0.0077, Macro-F1=0.2000
- Epoch 5: MCC=0.0099, Macro-F1=0.2903
- Epoch 10: MCC=0.0189, Macro-F1=0.3310

### Лучшие метрики:
- MCC: 0.0189 (epoch 10)
- Macro-F1: 0.3310 (epoch 10)
- DA: 0.1173
- Hit Rate Up: 11.13%

### Статистика каналов (после CLAMP, усреднение по последней эпохе):
| Канал | mean | std | min | max |
|-------|------|-----|-----|-----|
| MicropriceDev | 0.0314 | 0.2516 | -0.4134 | 0.3673 |
| Vol | 0.3166 | 0.6946 | -2.3208 | 2.3194 |
| Imb | 0.0464 | 0.5654 | -1.6142 | 3.8219 |
| OFI | -0.1268 | 0.5251 | -1.3288 | 1.6212 |
| VIB | 0.2453 | 0.3326 | -0.3540 | 0.8900 |
| Ret_10 | 0.1518 | 0.3474 | -0.7849 | 1.0469 |
| Ret_50 | 0.5299 | 0.4232 | 0.0000 | 1.2702 |
| Ret_100 | 0.5174 | 0.4445 | -0.4097 | 0.8708 |
| Spread | 1.2839 | 0.4150 | 0.9494 | 2.5154 |
| DeltaImb | 0.0236 | 0.6674 | -3.2487 | 3.2439 |
| DeltaSpread | -0.0001 | 0.0281 | -0.0669 | 0.1284 |

---

## Контракт validation-метрик (Задача 325)

Начиная с задачи 325 для обучения используется единый epoch-level contract:

- `val_mcc_primary`: единственный primary monitor key для `ModelCheckpoint` и `EarlyStopping`.
- `val_mcc_torch`, `val_mcc_np`: вторичные диагностические версии MCC из разных реализаций.
- `val_direction_mcc`: directional MCC по предсказаниям вне `Flat`.
- `val_f1_macro_torch`, `val_f1_macro_np`: вторичные диагностические версии macro-F1.
- `val_da_without_flat`: directional accuracy только по `pred != Flat`.
- `coverage_directional`, `coverage_long`, `coverage_short`: покрытие сигналов.
- `gross_edge_total`, `net_edge_total`: общий edge до и после transaction costs.
- `val_ece`, `val_mce`: calibration metrics.

Дополнительно:

- таблица class metrics больше не использует двусмысленный `Hit Rate`; вместо него фиксируются `precision_flat|up|down` и `recall_flat|up|down`;
- short-edge считается со sign flip для класса `Down`;
- epoch report сохраняется в `artifacts/<symbol>/validation/validation_report_epoch_{epoch}.json`.

## Validation Metric Contract (Задача 325)

- `val_mcc_primary`: единственный primary key для `ModelCheckpoint` и `EarlyStopping`; должен логироваться ровно один раз на эпоху.
- `val_direction_mcc`: MCC только по directional-предсказаниям (`pred != Flat`) для отдельного контроля качества direction-only части.
- `val_da_without_flat`: directional accuracy по подмножеству `pred != Flat`; это trade/coverage метрика, а не primary quality key.
- `coverage_directional`: доля validation-сэмплов, где модель дала directional-сигнал (`pred != Flat`).
- `gross_edge_total`: средний directional gross edge до издержек, где `short` считается со сменой знака.
- `net_edge_total`: средний directional edge после roundtrip-costs.
- `val_ece`: Expected Calibration Error по logits всей validation-эпохи.
- `val_mce`: Maximum Calibration Error по logits всей validation-эпохи.

## Задача 324 | Дата: 2026-03-26 | Эпох: 10

### Изменения (из docs/000-tasks_list.md):
Стабилизировать dynamic feature contract OFI, DeltaImb, DeltaSpread для LiT. Перевести DeltaImb и DeltaSpread на тот же event-consistent источник, что и OFI. Переделать fit dynamic-normalizer. Синхронизировать train-fit

### Использованные каналы (11):
0: MicropriceDev, 1: Vol, 2: Imb, 3: OFI, 4: VIB, 5: Ret_10, 6: Ret_50, 7: Ret_100, 8: Spread, 9: DeltaImb, 10: DeltaSpread

### Эпохи и метрики:
- Epoch 1: MCC=-0.0077, Macro-F1=0.2000
- Epoch 5: MCC=0.0099, Macro-F1=0.2903
- Epoch 10: MCC=0.0189, Macro-F1=0.3310

### Лучшие метрики:
- MCC: 0.0189 (epoch 10)
- Macro-F1: 0.3310 (epoch 10)
- DA: 0.1173
- Hit Rate Up: 11.13%

### Статистика каналов (после CLAMP, усреднение по последней эпохе):
| Канал | mean | std | min | max |
|-------|------|-----|-----|-----|
| MicropriceDev | 0.0314 | 0.2516 | -0.4134 | 0.3673 |
| Vol | 0.3166 | 0.6946 | -2.3208 | 2.3194 |
| Imb | 0.0464 | 0.5654 | -1.6142 | 3.8219 |
| OFI | -0.1268 | 0.5251 | -1.3288 | 1.6212 |
| VIB | 0.2453 | 0.3326 | -0.3540 | 0.8900 |
| Ret_10 | 0.1518 | 0.3474 | -0.7849 | 1.0469 |
| Ret_50 | 0.5299 | 0.4232 | 0.0000 | 1.2702 |
| Ret_100 | 0.5174 | 0.4445 | -0.4097 | 0.8708 |
| Spread | 1.2839 | 0.4150 | 0.9494 | 2.5154 |
| DeltaImb | 0.0236 | 0.6674 | -3.2487 | 3.2439 |
| DeltaSpread | -0.0001 | 0.0281 | -0.0669 | 0.1284 |
[Обновленный validation contract]

- `val_mcc_primary`: единственный primary monitor key для `ModelCheckpoint` и `EarlyStopping`.
- `val_mcc_np`, `val_mcc_torch`: вспомогательные реализации MCC для сверки контрактов epoch-end.
- `val_direction_mcc`: directional MCC по directional predictions без Flat.
- `val_f1_macro_np`, `val_f1_macro_torch`: macro-F1, посчитанный одним местом в конце эпохи.
- `val_da_without_flat`: directional accuracy только по `pred != Flat`.
- `coverage_directional`, `coverage_long`, `coverage_short`: доля всех примеров, в которых модель открывает directional / long / short сигнал.
- `gross_edge_total`: средний edge по directional сделкам до costs.
- `net_edge_total`: средний edge по directional сделкам после `report_fee_bps`, `report_slippage_bps`, `report_half_spread_bps`.
- `val_ece`, `val_mce`: calibration metrics из единого epoch-end отчета.
- `precision_flat|up|down`, `recall_flat|up|down`: явная class table вместо двусмысленного `Hit Rate`.
- JSON-артефакт эпохи сохраняется в `artifacts/<symbol>/validation/validation_report_epoch_{epoch}.json`.

## Задача 326 | Baseline sweep

- Артефакты baseline sweep теперь генерируются отдельным sweep-режимом в `python_lab/src/train.py`.
- Ожидаемые выходы: `docs/sweep_baseline.csv`, `docs/sweep_baseline.json`, `docs/baselines.md`.
- Sweep сохраняет relation threshold-to-spread и отдельный `dynamic_threshold_reference`.
- Для shortlist top-k кандидатов sweep-режим пишет `mini_train_mcc`, `mini_train_coverage_directional`, `mini_train_net_edge_total`.

## Sweep Baseline Contract (Задача 326)

- baseline sweep строится отдельно от обычного train loop;
- grid должен сохранять `share_flat`, `share_up`, `share_down`, `trade_share`;
- в отчёте обязаны быть `threshold_bps`, `median_spread_bps`, `threshold_to_spread_ratio`, `subspread_target`;
- `rolling_std * 0.5` для `h=100` фиксируется отдельно как unsafe reference и не подменяет static grid;
- shortlist top-k кандидатов должен содержать `mini_train_mcc`, `mini_train_coverage_directional`, `mini_train_net_edge_total`.
