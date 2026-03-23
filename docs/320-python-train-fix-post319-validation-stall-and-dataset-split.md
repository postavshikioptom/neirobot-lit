# Задача 320: Устранение silent-stall после 319 в validation/epoch_end и исправление train/eval split

**Дата создания:** 23.03.2026
**Статус:** ЗАПЛАНИРОВАНО
**Категория:** Обучение Python
**Зависимости:** Задача 319

## Цель

Вернуть стабильный запуск `python_lab/src/train.py` после изменений задачи 319, убрать ложное и реальное "зависание" на границе `train -> validation`, сделать фазы обучения наблюдаемыми, гарантированно исключить train-mode логику из `val/test`, и добавить отдельную диагностику saturation каналов до и после clamp.

Ожидаемый результат:
- обучение на Kaggle GPU снова доходит до обычной валидации без долгой немой паузы;
- в логах видно, где именно находится пайплайн: train-loop, validation-loop или `on_validation_epoch_end`;
- `val/test` больше не используют `is_train=True` и не запускают train-аугментации;
- тяжелые epoch-end артефакты больше не запускаются в sanity-check и по умолчанию не стартуют на `epoch 0`;
- saturation каналов измеряется явно, а не по косвенным `min/max=-5/5`.

## Контекст и проблема

После задачи 319 обучение стало доходить до логов вида:

```text
Metrics by Market Regime:
  Regime 0: MCC=-0.1140, F1=0.4182, Samples=128
✓ Logged 1000 embeddings to TensorBoard Projector
[BATCH 0] ...
...
[BATCH 500] ...
```

После этого наступает длинная тишина, которая выглядит как зависание.

Важно: пользователь запускает обучение не на локальном ПК, а на внешнем Kaggle GPU сервере с GPU около 16 GB. До последних изменений там такого поведения не было. Значит проблема не сводится к слабому локальному железу и не должна описываться как чисто Windows-специфичная. Корень проблемы нужно искать в логике пайплайна `train.py`, `dataset.py` и `utils.py`.

### Что сейчас видно по коду

1. `Metrics by Market Regime ... Samples=128` печатается в `python_lab/src/train.py:860-874` внутри `on_validation_epoch_end`.
2. `✓ Logged 1000 embeddings to TensorBoard Projector` печатается в `python_lab/src/utils.py:1548`.
3. `log_embeddings()` в `python_lab/src/utils.py:1456-1550` заново итерируется по validation dataloader и делает дополнительный `forward` через hook на `model.patching`.
4. В `on_validation_epoch_end` без защитных guard'ов выполняются тяжелые действия:
   - Reliability Diagram: `python_lab/src/train.py:804-813`
   - Confusion Matrix и PR curves: `python_lab/src/train.py:830-843`
   - Gradient norms и TensorBoard Projector embeddings: `python_lab/src/train.py:880-897`
5. В `pl.Trainer(...)` progress bar отключен: `python_lab/src/train.py:2572-2581`. Поэтому длинная validation-пауза выглядит как полный hang.
6. Есть критическая ошибка режима датасета:
   - `full_dataset` создается с `is_train=True` в `python_lab/src/train.py:1964-1982`
   - `train_ds = TrainSubset(full_dataset, train_indices)` в `python_lab/src/train.py:2042`
   - `val_ds = Subset(full_dataset, val_indices)` и `test_ds = Subset(full_dataset, test_indices)` в `python_lab/src/train.py:2043-2044`
   - при этом аугментация завязана на `self.is_train` в `python_lab/src/dataset.py:1589-1597`
7. Saturation каналов реальна, а не кажется по логам:
   - общий clamp `[-5, 5]` выполняется в `python_lab/src/dataset.py:1654`
   - для `OFI`, `DeltaImb`, `DeltaSpread` есть дополнительный clamp `[-3, 3]` в `python_lab/src/dataset.py:1656-1658`

### Рабочая гипотеза

Текущий "hang" складывается из нескольких причин сразу:

1. После train-loop идет длинная validation-фаза без промежуточных логов.
2. `on_validation_epoch_end` выполняет тяжелые действия, часть из которых повторно проходит по validation dataloader.
3. Валидация и тест, вероятно, работают через dataset, созданный в train-mode, из-за чего туда протекают train-аугментации и лишняя train-логика.
4. Визуально это усиливается тем, что progress bar отключен.
5. Saturation каналов и нестабильность `vol_head` ухудшают качество обучения, но это отдельная проблема и не выглядит главным источником самого stall.

### Почему Kaggle GPU не снимает проблему

Даже быстрый GPU не ускоряет:
- Python-логику внутри `on_validation_epoch_end`;
- повторный проход `log_embeddings()` по validation dataloader;
- лишние train-аугментации в `val/test`, если dataset остается в `is_train=True`;
- слишком тяжелые epoch-end артефакты на `epoch 0`;
- немую паузу из-за отключенного progress bar и отсутствия phase-логов.

Именно поэтому на Kaggle это выглядит как зависание не хуже, чем локально.

---

## Подзадачи

### Подзадача 320.1: Сделать фазы train/validation наблюдаемыми

**Файл:** `python_lab/src/train.py`

**Проблема:** Сейчас в коде есть `on_train_epoch_start` / `on_train_epoch_end`, но нет явных логов начала validation, начала `on_validation_epoch_end` и прогресса по validation batches. Из-за этого невозможно отличить реальный hang от длинной, но живой фазы.

**Что менять:**

1. В `on_train_epoch_start` (`python_lab/src/train.py:351-375`) добавить явный лог старта эпохи обучения.
2. В `on_train_epoch_end` (`python_lab/src/train.py:376-385`) добавить явный лог завершения train-loop и длительности.
3. Добавить новый hook `on_validation_epoch_start` сразу перед `validation_step`, чтобы печатать:
   - текущую эпоху;
   - признак `sanity_checking`;
   - timestamp старта validation.
4. В `validation_step` (`python_lab/src/train.py:655-720`) добавить лог для первого validation batch и затем лог каждые `N` батчей через новый аргумент `--val_batch_log_interval`.
5. В начале `on_validation_epoch_end` (`python_lab/src/train.py:722`) печатать:
   - вошли ли мы в `sanity_checking`;
   - сколько validation сэмплов накоплено;
   - сколько времени занял validation-loop до входа в epoch_end.
6. В конце `on_validation_epoch_end` печатать отдельный лог завершения epoch-end hook и его длительность.

**Ключевая замена:**

```python
# БЫЛО
def on_train_epoch_start(self):
    self._grad_warn_prints = 0
    self._vol_diag_prints = 0
    import time
    self.epoch_start_time = time.time()

# СТАЛО
def on_train_epoch_start(self):
    self._grad_warn_prints = 0
    self._vol_diag_prints = 0
    import time
    self.epoch_start_time = time.time()
    print(f"\n[TRAIN] Epoch {self.current_epoch} started")

def on_validation_epoch_start(self):
    import time
    self.validation_start_time = time.time()
    is_sanity = bool(getattr(self.trainer, 'sanity_checking', False))
    phase = 'SANITY' if is_sanity else 'VAL'
    print(f"\n[{phase}] Epoch {self.current_epoch} validation started")
```

**Дополнительно заменить в `validation_step`:**

```python
# СТАЛО
if batch_idx == 0:
    is_sanity = bool(getattr(self.trainer, 'sanity_checking', False))
    phase = 'SANITY' if is_sanity else 'VAL'
    print(f"[{phase}] first validation batch received")
elif self.hparams.get('val_batch_log_interval', 100) > 0 and batch_idx % self.hparams['val_batch_log_interval'] == 0:
    is_sanity = bool(getattr(self.trainer, 'sanity_checking', False))
    phase = 'SANITY' if is_sanity else 'VAL'
    print(f"[{phase}] batch {batch_idx}")
```

**Верификация:**
- После последнего training batch в консоли сразу появляется `[VAL] ... validation started`.
- Если зависание происходит внутри validation-loop, в логах видно последний validation batch.
- Если зависание происходит внутри `on_validation_epoch_end`, виден вход в этот hook.

---

### Подзадача 320.2: Вынести DataLoader и Trainer knobs в CLI и убрать hardcode

**Файл:** `python_lab/src/train.py`

**Проблема:** Сейчас DataLoader конфиг захардкожен в нескольких местах:
- trial loaders: `python_lab/src/train.py:1264-1288`
- main loaders: `python_lab/src/train.py:2235-2273`
- recreated loaders: `python_lab/src/train.py:2748-2769`
- CV fold loaders: `python_lab/src/train.py:2832-2855`

Сейчас там используются фиксированные:
- `num_workers = 4`
- `pin_memory = True`
- `prefetch_factor = 2` (в основном пути)
- `persistent_workers = True if num_workers > 0 else False`

Для отладки hang это плохой вариант: параметры нельзя быстро ослабить с CLI, а behavior отличается между режимами.

**Что менять:**

1. В `argparse` рядом с блоком `data_mode` (`python_lab/src/train.py:1550-1555`) добавить:
   - `--num_workers` (default `0`)
   - `--pin_memory / --no-pin_memory` (default `True`)
   - `--persistent_workers / --no-persistent_workers` (default `False`)
   - `--prefetch_factor` (default `2`)
   - `--num_sanity_val_steps` (default `0`)
   - `--enable_progress_bar / --no-enable_progress_bar` (default `True`)
   - `--enable_tb_embeddings / --no-enable_tb_embeddings` (default `False`)
   - `--enable_epoch_end_plots / --no-enable_epoch_end_plots` (default `False`)
   - `--skip_epoch0_artifacts / --no-skip_epoch0_artifacts` (default `True`)
   - `--val_batch_log_interval` (default `100`)
2. Вынести сборку kwargs в helper, например `build_dataloader_kwargs(args, shuffle: bool)`.
3. Во всех четырех местах создания DataLoader заменить ручную сборку на один helper.
4. В `pl.Trainer(...)` (`python_lab/src/train.py:2572-2581`) пробросить:
   - `enable_progress_bar=args.enable_progress_bar`
   - `num_sanity_val_steps=args.num_sanity_val_steps`

**Ключевая замена для DataLoader:**

```python
# БЫЛО
num_workers = 4
worker_init_fn = None

train_loader = DataLoader(
    train_ds,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True if num_workers > 0 else False,
    worker_init_fn=worker_init_fn
)

# СТАЛО
def build_dataloader_kwargs(args, *, shuffle):
    kwargs = {
        'batch_size': args.batch_size,
        'shuffle': shuffle,
        'num_workers': args.num_workers,
        'pin_memory': args.pin_memory,
        'worker_init_fn': None,
    }
    if args.num_workers > 0:
        kwargs['persistent_workers'] = args.persistent_workers
        kwargs['prefetch_factor'] = args.prefetch_factor
    return kwargs

train_loader = DataLoader(train_ds, **build_dataloader_kwargs(args, shuffle=True))
val_loader = DataLoader(val_ds, **build_dataloader_kwargs(args, shuffle=False))
test_loader = DataLoader(test_ds, **build_dataloader_kwargs(args, shuffle=False))
```

**Ключевая замена для Trainer:**

```python
# БЫЛО
trainer = pl.Trainer(
    max_epochs=args.epochs,
    callbacks=callbacks,
    logger=logger,
    accelerator="auto",
    devices=1,
    precision=trainer_precision,
    log_every_n_steps=100,
    accumulate_grad_batches=args.accumulate_grad_batches,
    enable_progress_bar=False
)

# СТАЛО
trainer = pl.Trainer(
    max_epochs=args.epochs,
    callbacks=callbacks,
    logger=logger,
    accelerator="auto",
    devices=1,
    precision=trainer_precision,
    log_every_n_steps=100,
    accumulate_grad_batches=args.accumulate_grad_batches,
    enable_progress_bar=args.enable_progress_bar,
    num_sanity_val_steps=args.num_sanity_val_steps
)
```

**Базовый безопасный профиль для Kaggle до стабилизации:**
- `--num_workers 0`
- `--persistent_workers False`
- `--pin_memory True`
- `--num_sanity_val_steps 0`
- `--enable_tb_embeddings False`
- `--enable_epoch_end_plots False`
- `--skip_epoch0_artifacts True`

**Верификация:**
- Один и тот же CLI управляет behavior во всех train/distill/cv путях.
- Можно быстро отключить sanity validation и тяжелые epoch-end артефакты без правок кода.
- Появляется безопасный baseline-конфиг для Kaggle.

---

### Подзадача 320.3: Защитить `on_validation_epoch_end` от тяжелых артефактов в sanity-check и на `epoch 0`

**Файлы:**
- `python_lab/src/train.py`
- `python_lab/src/utils.py`

**Проблема:** Сейчас `on_validation_epoch_end` безусловно выполняет тяжелые действия, а `log_embeddings()` делает дополнительный проход по validation dataloader. Это особенно плохо в двух случаях:
- sanity validation до реального обучения;
- первая реальная эпоха (`epoch 0`), когда и так дороже всего понять, где зависает пайплайн.

**Что менять в `train.py`:**

1. В начале `on_validation_epoch_end` добавить быстрый guard:

```python
if not self.val_y_true:
    print("[VAL] No validation outputs collected; skipping epoch_end")
    return
```

2. Ввести флаги:

```python
is_sanity = bool(getattr(self.trainer, 'sanity_checking', False))
skip_epoch0_artifacts = self.hparams.get('skip_epoch0_artifacts', True) and self.current_epoch == 0
skip_heavy_artifacts = is_sanity or skip_epoch0_artifacts
```

3. Вынести heavy artifacts под guard:
   - Reliability Diagram (`python_lab/src/train.py:804-813`)
   - Confusion Matrix / PR Curves (`python_lab/src/train.py:830-843`)
   - Gradient norms / embeddings (`python_lab/src/train.py:880-897`)

4. Оставить базовые метрики (`MCC`, `F1`, `Vol-MSE`, `Vol-MAE`) включенными, но артефакты делать опциональными.

**Ключевая замена блока heavy artifacts:**

```python
# БЫЛО
if self.current_epoch % 20 == 0:
    plot_reliability_diagram(...)

if self.logger and hasattr(self.logger, 'experiment'):
    writer = self.logger.experiment
    if self.current_epoch % 20 == 0:
        plot_confusion_matrix_tensorboard(...)
        plot_pr_curves_tensorboard(...)
    from .utils import log_gradient_norms
    log_gradient_norms(self.model, writer, self.current_epoch)
    if self.current_epoch % 30 == 0:
        from .utils import log_embeddings
        val_dataloader = self.trainer.val_dataloaders
        if val_dataloader is not None:
            log_embeddings(self.model, val_dataloader, writer, self.current_epoch, max_samples=tb_embedding_samples)

# СТАЛО
is_sanity = bool(getattr(self.trainer, 'sanity_checking', False))
skip_epoch0_artifacts = self.hparams.get('skip_epoch0_artifacts', True) and self.current_epoch == 0
skip_heavy_artifacts = is_sanity or skip_epoch0_artifacts

if self.hparams.get('enable_epoch_end_plots', False) and not skip_heavy_artifacts:
    if self.current_epoch % 20 == 0:
        plot_reliability_diagram(...)
        plot_confusion_matrix_tensorboard(...)
        plot_pr_curves_tensorboard(...)

if self.logger and hasattr(self.logger, 'experiment'):
    writer = self.logger.experiment
    from .utils import log_gradient_norms
    if not is_sanity:
        log_gradient_norms(self.model, writer, self.current_epoch)

    if self.hparams.get('enable_tb_embeddings', False) and not skip_heavy_artifacts and self.current_epoch % 30 == 0:
        from .utils import log_embeddings
        val_dataloaders = self.trainer.val_dataloaders
        val_dataloader = val_dataloaders[0] if isinstance(val_dataloaders, (list, tuple)) else val_dataloaders
        if val_dataloader is not None:
            log_embeddings(self.model, val_dataloader, writer, self.current_epoch, max_samples=tb_embedding_samples)
```

**Что менять в `utils.py`:**

1. В `log_embeddings()` (`python_lab/src/utils.py:1456-1550`) добавить явные прогресс-логи каждые `10` батчей.
2. Если передан список dataloader'ов, брать первый.
3. Восстанавливать исходный режим модели, а не всегда вызывать `model.train()` в конце.

**Ключевая замена в `utils.py`:**

```python
# БЫЛО
model.eval()
...
for batch_idx, batch in enumerate(dataloader):
    ...
model.train()

# СТАЛО
was_training = model.training
if isinstance(dataloader, (list, tuple)):
    dataloader = dataloader[0] if dataloader else None
if dataloader is None:
    return

model.eval()
...
for batch_idx, batch in enumerate(dataloader):
    if batch_idx == 0 or batch_idx % 10 == 0:
        print(f"[TB_EMB] batch {batch_idx}")
    ...
if was_training:
    model.train()
```

**Верификация:**
- В sanity-check не строятся картинки и не запускается `log_embeddings()`.
- На `epoch 0` по умолчанию пропускаются тяжелые артефакты.
- Если включить `--enable_tb_embeddings`, в логах видно прогресс самого сбора embeddings.

---

### Подзадача 320.4: Исправить `train/val/test` split так, чтобы `val/test` всегда работали в eval-mode

**Файлы:**
- `python_lab/src/train.py`
- косвенно проверяется `python_lab/src/dataset.py`

**Проблема:** Сейчас базовый `full_dataset` создается с `is_train=True`, а затем `val_ds` и `test_ds` строятся как `Subset(full_dataset, ...)`. Из-за этого validation/test могут идти через dataset, который живет в train-mode и запускает train-аугментации из `python_lab/src/dataset.py:1589-1597`.

Дополнительная проблема: если просто переключить `full_dataset` в `is_train=False`, то текущая sample-based NaN проверка (`python_lab/src/train.py:1994-2026`) начнет обращаться к `full_dataset[i]` до ручного `normalizer.fit()` на train split (`python_lab/src/train.py:2049-2057`). Значит нужно не только изменить `is_train`, но и переставить порядок блоков.

**Что менять:**

1. В конструкторе `full_dataset` (`python_lab/src/train.py:1964-1982`) заменить `is_train=True` на `is_train=False`.
2. Оставить `TrainSubset(full_dataset, train_indices)` для train.
3. Оставить обычные `Subset(full_dataset, val_indices)` и `Subset(full_dataset, test_indices)` для validation/test.
4. Переместить блок sample-based NaN проверки (`python_lab/src/train.py:1989-2026`) после ручного `normalizer.fit()` (`python_lab/src/train.py:2049-2057`), чтобы `__getitem__` вызывался уже после fit.
5. Для CV пути заменить:
   - `fold_train_ds = Subset(full_dataset, train_idx)` на `TrainSubset(full_dataset, list(train_idx))`
   - `fold_val_ds = Subset(full_dataset, list(val_idx))` оставить как есть
6. Для Optuna path dataset уже создается с `is_train=False` (`python_lab/src/train.py:1239-1255`), поэтому здесь нужно только унифицировать DataLoader config, а split не трогать в этой задаче.

**Ключевая замена конструктора dataset:**

```python
# БЫЛО
full_dataset = LOBDataset(
    df,
    seq_len=args.seq_len,
    ...
    data_mode="memory",
    is_train=True,
    ...
    normalizer=normalizer,
    **time_weighting_params
)

# СТАЛО
full_dataset = LOBDataset(
    df,
    seq_len=args.seq_len,
    ...
    data_mode="memory",
    is_train=False,
    ...
    normalizer=normalizer,
    **time_weighting_params
)
```

**Ключевая замена порядка блоков:**

```python
# БЫЛО
full_dataset = LOBDataset(...)
sample = full_dataset[i]  # NaN check
...
train_ds = TrainSubset(full_dataset, train_indices)
val_ds = Subset(full_dataset, val_indices)
...
normalizer.fit(train_channels_df)

# СТАЛО
full_dataset = LOBDataset(..., is_train=False, ...)
train_ds = TrainSubset(full_dataset, train_indices)
val_ds = Subset(full_dataset, val_indices)
test_ds = Subset(full_dataset, test_indices)

train_channels_df = full_dataset._compute_channels_for_normalization(train_indices_for_fit)
normalizer.fit(train_channels_df, winsor_limits=winsor_limits)
normalizer.save(...)

# Только после fit:
for i in range(0, nan_check_samples, 10):
    sample = train_ds[i] if i < len(train_ds) else full_dataset[i]
```

**Ключевая замена для CV path:**

```python
# БЫЛО
fold_train_ds = Subset(full_dataset, train_idx)
fold_val_ds = Subset(full_dataset, val_idx)

# СТАЛО
fold_train_ds = TrainSubset(full_dataset, list(train_idx))
fold_val_ds = Subset(full_dataset, list(val_idx))
```

**Верификация:**
- Аугментации больше не применяются в `val/test`.
- `full_dataset[i]` не вызывается до ручного `normalizer.fit()`.
- CV train split сохраняет аугментацию, validation split остается в eval-mode.

---

### Подзадача 320.5: Добавить явную диагностику saturation каналов до и после clamp

**Файл:** `python_lab/src/dataset.py`

**Проблема:** Сейчас по логам видно много значений `min/max=-5/5` и `-3/3`, но код не измеряет долю реально clipped значений. Поэтому невозможно понять, идет ли речь о редких выбросах или о массовой потере сигнала.

**Что менять:**

1. В `_process_sample` рядом с блоком `x_final = torch.stack(...)` (`python_lab/src/dataset.py:1644-1658`) сохранить копию до clamp:

```python
x_pre_clip = x_final.clone()
```

2. Добавить helper, например `_log_clip_saturation(x_pre_clip, x_post_clip, idx)`, который по каждому каналу считает:
   - долю значений ниже нижнего лимита;
   - долю значений выше верхнего лимита;
   - итоговую долю clipped значений;
   - `min/max/mean/std` до clamp и после clamp.
3. Для каналов `OFI`, `DeltaImb`, `DeltaSpread` использовать лимит `3.0`, для остальных `5.0`.
4. Логировать только ограниченное число раз, например для первых 1-2 отладочных сэмплов или через отдельный счетчик, чтобы не забить лог.
5. Отдельно вывести summary в `training_step` для первого train batch первой эпохи, чтобы свести saturation dataset-level диагностику и реальный batch-level input.

**Ключевая замена блока clamp:**

```python
# БЫЛО
x_final = torch.clamp(x_final, -5.0, 5.0)
x_final[:, 3, :] = torch.clamp(x_final[:, 3, :], -3.0, 3.0)
x_final[:, 9, :] = torch.clamp(x_final[:, 9, :], -3.0, 3.0)
x_final[:, 10, :] = torch.clamp(x_final[:, 10, :], -3.0, 3.0)

# СТАЛО
x_pre_clip = x_final.clone()

x_final = torch.clamp(x_final, -5.0, 5.0)
x_final[:, 3, :] = torch.clamp(x_final[:, 3, :], -3.0, 3.0)
x_final[:, 9, :] = torch.clamp(x_final[:, 9, :], -3.0, 3.0)
x_final[:, 10, :] = torch.clamp(x_final[:, 10, :], -3.0, 3.0)

if idx is not None and self._clip_diag_prints < self.max_clip_diag_prints:
    self._log_clip_saturation(x_pre_clip, x_final, idx)
    self._clip_diag_prints += 1
```

**Что обязательно вывести в лог:**
- `Channel name`
- `clip_limit`
- `% below limit`
- `% above limit`
- `% total clipped`
- `pre[min,max,std]`
- `post[min,max,std]`

**Верификация:**
- По каждому каналу видно, насколько проблема массовая.
- Можно отдельно сравнить `MicropriceDev`, `Vol`, `Imb`, `Ret_*`, `Spread`, `OFI`, `DeltaImb`, `DeltaSpread`.

---

### Подзадача 320.6: Отдельным follow-up разобрать `vol_head` и non-finite gradients

**Файлы:**
- `python_lab/src/train.py`
- `python_lab/src/lit_model.py`

**Приоритет:** НИЗКИЙ, только после снятия hang/silent-stall

**Проблема:** В логах уже есть:
- `Non-finite gradient in model.vol_regressor.weight`
- `Non-finite gradient in model.vol_regressor.bias`
- `vol_target` уходит существенно ниже текущего clamp окна

При этом в `python_lab/src/train.py:585-586` target и pred режутся clamp'ом перед `vol_loss`. Это может скрывать распределение tails и ухудшать стабильность `vol_head`.

**Что менять позже:**

1. Добавить отдельную диагностику доли `vol_target`, выходящей за `[-vol_clamp_val, +vol_clamp_val]`.
2. Проверить, нужен ли другой loss для volatility branch:
   - `SmoothL1` с более мягкой чувствительностью к tails;
   - symlog/standardization только для volatility target;
   - отдельный gradient clipping для `vol_regressor`.
3. Разобрать, нужно ли менять detach/подачу признаков в `vol_head`.

**Верификация:**
- После снятия hang исчезает ли деградация `vol_head`.
- Пропадают ли non-finite gradients в `vol_regressor`.

---

## Порядок внедрения

1. Сначала выполнить `320.1 + 320.2 + 320.3`.
2. Затем выполнить `320.4`, потому что без правильного `is_train=False` для `val/test` валидация остается логически неверной.
3. После этого выполнить `320.5`, чтобы увидеть реальный масштаб saturation.
4. Только затем переходить к `320.6`.

---

## Критерии приемки

1. На Kaggle GPU обучение перестает уходить в немую паузу без фазовых логов.
2. При запуске с безопасным профилем:
   - `--num_workers 0`
   - `--num_sanity_val_steps 0`
   - `--enable_tb_embeddings False`
   - `--enable_epoch_end_plots False`
   - `--skip_epoch0_artifacts True`
   лог явно показывает переходы `TRAIN -> VAL -> on_validation_epoch_end`.
3. В `val/test` не применяется train-аугментация.
4. В `epoch 0` по умолчанию не запускается повторный проход по `val_dataloader` для embeddings.
5. В логах есть saturation summary по каналам, а не только `min/max=-5/5`.

---

## Короткий ожидаемый эффект

После выполнения задачи 320 обучение снова станет дебажимым и контролируемым:
- если pipeline реально виснет, это будет видно по последней фазе;
- если он просто долго считает validation/epoch_end, это тоже будет видно;
- validation/test перестанут загрязняться train-mode поведением;
- saturation каналов можно будет оценивать количественно, а не на глаз.
