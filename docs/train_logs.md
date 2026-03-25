
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

## Задача 324 | Дата: 2026-03-25 | Эпох: 0 (ошибка до обучения)

### Изменения (из docs/000-tasks_list.md):
Стабилизировать dynamic feature contract OFI, DeltaImb, DeltaSpread для LiT. Перевести DeltaImb и DeltaSpread на тот же event-consistent источник, что и OFI. Переделать fit dynamic-normalizer. Синхронизировать train-fit

### Использованные каналы (11):
0: MicropriceDev, 1: Vol, 2: Imb, 3: OFI, 4: VIB, 5: Ret_10, 6: Ret_50, 7: Ret_100, 8: Spread, 9: DeltaImb, 10: DeltaSpread

### Лучшие метрики:
- MCC: N/A (epoch N/A)
- Macro-F1: N/A (epoch N/A)
- DA: N/A
- Hit Rate Up: N/A

### Причина остановки:
- RuntimeError: feat_update_id not found in DataFrame; required for OFI calculation
