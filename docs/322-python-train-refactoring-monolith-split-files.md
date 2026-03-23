# Рекомендуемое имя файла: `docs/322-python-train-monolith-split-plan.md`

# Задача 322: Рефакторинг монолитного `python_lab/src/train.py` на отдельные файлы

**Дата создания:** 23.03.2026  
**Статус:** ЗАПЛАНИРОВАНО  
**Категория:** Обучение Python  
**Зависимости:** Задачи 151, 303, 320; актуальная схема в `docs/000-architecture.md`

## Цель

Разделить монолитный файл `python_lab/src/train.py` на несколько осмысленных модулей без изменения внешнего интерфейса запуска:

- `python -m python_lab.src.train ...` остается основной точкой входа;
- все текущие CLI-флаги и сценарии `train / distill / cv / optuna` сохраняются;
- новый код раскладывается по отдельным файлам в той же папке `python_lab/src/`, без подпапок;
- из `train.py` уходит orchestration-хаос, а сам файл превращается в тонкий entrypoint;
- критичный порядок инициализации данных, нормализации, split, model/trainer и postprocess не меняется.

Ожидаемый результат:

- `python_lab/src/train.py` становится коротким и читаемым;
- чувствительные блоки `Normalizer`, `split/eval-mode`, `LiTModule`, `Optuna`, `CV`, `postprocess` оказываются изолированы;
- уменьшается риск новых регрессий из-за случайной перестановки кода;
- в будущем новые изменения можно вносить точечно, не раздувая один файл до 4000+ строк.

## Контекст и проблема

Сейчас `python_lab/src/train.py` имеет 3496 строк и одновременно содержит:

1. top-level training core:
   - `TrainSubset` (`39-51`)
   - `ProfilerCallback` (`113-164`)
   - `compute_hft_metrics` (`166-234`)
   - `LiTModule` (`237-1194`)
   - `enable_dropout` (`1196-1198`)
2. Optuna:
   - `objective_seq_len_search` (`1200-1541`)
3. metadata side effect:
   - `update_model_metadata` (`1543-1580`)
4. основной orchestration:
   - `train()` (`1582-3619`)

Внутри `train()` сейчас смешаны:

- `argparse` и нормализация аргументов;
- runtime/bootstrap: `winsor_limits`, `seed`, пути, RAM check, `build_dataloader_kwargs`, `precision`;
- `LOBDataLoader -> FeatureEngineer -> Labeler -> horizons -> Normalizer`;
- `full_dataset -> chronological split -> normalizer.fit/save -> metadata update -> NaN diagnostics`;
- `DataLoader` и `class weights`;
- сборка обычной модели и distillation-ветки;
- callbacks/logger/trainer;
- Optuna seq_len search;
- CV режим;
- обычный train/distill путь;
- MC Dropout;
- pruning;
- final holdout evaluation;
- teacher-vs-student comparison;
- финальное копирование лучшего checkpoint в `teacher_lit.pt` / `lit.pt`.

### Почему это уже опасно

Проблема не только в размере файла. В одном месте смешаны четыре уровня ответственности:

1. training core (`LiTModule`, hooks, analytics, loss/scheduler/distillation);
2. data bootstrap (`load_data`, `FeatureEngineer`, `Labeler`, `Normalizer`, split);
3. runtime/orchestration (`paths`, `trainer`, `logger`, `callbacks`, `precision`);
4. отдельные режимы (`train`, `distill`, `cv`, `optuna`, `pruning`, `holdout`).

Именно эта смесь уже приводила к регрессиям:

- ошибкам порядка `dataset -> split -> eval-mode`;
- проблемам около `Normalizer.fit()` и channel-space;
- silent-stall на границе `train -> validation`;
- дрейфу логики между основным train, CV и Optuna;
- скрытым поломкам из-за тяжелых действий в `on_validation_epoch_end`;
- симптомам численной нестабильности вокруг `vol_regressor` и saturation каналов.

### Что нельзя сломать при рефакторинге

1. Порядок должен остаться таким:

```text
raw data
-> FeatureEngineer
-> Labeler
-> split
-> normalizer.fit(train only, channel-space)
-> dataset/dataloaders
-> model/module
-> trainer
-> fit
-> validation/postprocess
```

2. Нельзя менять batch contract:

```python
(x, y, ts, mid, label, extra_data)
```

где `extra_data` содержит `vol`, `weight`, `regime_id`, `f_ret`.

3. Нельзя разносить ABI модели и датасета:

- `python_lab/src/lit_model.py` остается источником правды для `LiTModel`, `LiTConfig`, формы `(B, S, C, 50)`, `in_channels=11`, `num_horizons`, `use_horizon_embedding`;
- `python_lab/src/dataset.py` остается источником правды для channel synthesis, `feature_order`, `past_ret_cache`, `ofi_cache`, `vib_cache`, `normalize_channel()` и train-time sample semantics.

4. Нельзя допустить расхождение `train / distill / cv / optuna` по `DataLoader` и `Trainer` kwargs.

5. Нельзя механически переносить code blocks, если они завязаны на текущий порядок инициализации:

- fit/save `Normalizer`;
- update `metadata.json`;
- `TrainSubset` vs `Subset`;
- тяжелые validation artifacts;
- distillation bootstrap;
- final copy/export лучшего checkpoint.

## Рабочая гипотеза

Проблема решается не “косметическим” разрезанием файла, а разделением по устойчивым зонам ответственности:

- `train.py` оставить только как thin entrypoint;
- `LiTModule` и training-specific hooks унести в отдельный training core;
- data bootstrap унести в отдельный data orchestration модуль;
- model build для teacher/student унести в factory;
- Optuna, CV и post-training логику отделить в самостоятельные модули;
- metadata side effect вынести отдельно, чтобы не размазывать запись `metadata.json` по data pipeline.

Целевая раскладка:

- `train.py`
- `train_cli.py`
- `train_runtime.py`
- `train_data.py`
- `train_metadata.py`
- `train_module.py`
- `train_model_factory.py`
- `train_optuna.py`
- `train_cv.py`
- `train_postprocess.py`

## Подзадачи

### Подзадача 322.1: Вынести training core в `train_module.py`

**Файлы:** `python_lab/src/train.py`, `python_lab/src/train_module.py`

**Что переносим из `train.py`:**

- `TrainSubset` (`39-51`)
- `ProfilerCallback` (`113-164`)
- `compute_hft_metrics` (`166-234`)
- весь `LiTModule` (`237-1194`)

**Что должно оказаться в новом файле:**

```python
class TrainSubset(...)
class ProfilerCallback(...)
def compute_hft_metrics(...)
class LiTModule(pl.LightningModule): ...
```

**Что заменить в `train.py`:**

```python
# БЫЛО
class TrainSubset(...)
class ProfilerCallback(...)
def compute_hft_metrics(...)
class LiTModule(pl.LightningModule): ...

# СТАЛО
from .train_module import (
    LiTModule,
    ProfilerCallback,
    TrainSubset,
    compute_hft_metrics,
)
```

**Что важно не потерять:**

- все методы `LiTModule` должны переехать без изменения сигнатур:
  - `training_step`
  - `validation_step`
  - `on_validation_epoch_end`
  - `configure_optimizers`
  - `configure_gradient_clipping`
- не менять использование:
  - `DistillationLoss`
  - `CalibrationMetrics`
  - `FocalLoss`
  - `save_confusion_matrices`
  - `plot_reliability_diagram`
- оставить batch contract прежним;
- не разрывать `vol_regressor` и связанные gradient/validation diagnostics на разные модули.

**Дополнительная замена импортов в новом файле:**

В `train_module.py` перенести импорты, которые реально нужны `LiTModule`:

- `torch`, `torch.nn`, `pytorch_lightning`
- `numpy`
- `sklearn.metrics`
- `torchmetrics.classification`
- `torch.profiler`
- `Path`
- `LiTModel`
- `compute_metrics`, `FocalLoss`, `CalibrationMetrics`, `save_confusion_matrices`, `plot_reliability_diagram`

В `train.py` после переноса удалить эти тяжелые импорты, если они больше не используются вне orchestration.

---

### Подзадача 322.2: Вынести CLI и нормализацию аргументов в `train_cli.py`

**Файлы:** `python_lab/src/train.py`, `python_lab/src/train_cli.py`

**Что переносим из `train()`**

Из начала `train()` перенести весь блок `argparse` (`1583-1729`) в отдельный модуль.

**Новый API:**

```python
def build_train_parser() -> argparse.ArgumentParser: ...
def parse_train_args(argv=None): ...
def parse_winsor_limits(raw_value: str) -> tuple[float, float]: ...
def resolve_horizon_config(args) -> tuple[object, int, list[float] | None]: ...
```

**Что заменить в `train.py`:**

```python
# БЫЛО
def train():
    parser = argparse.ArgumentParser(...)
    parser.add_argument(...)
    ...
    args = parser.parse_args()
    winsor_limits = ...

# СТАЛО
from .train_cli import parse_train_args, parse_winsor_limits, resolve_horizon_config

def train():
    args = parse_train_args()
    winsor_limits = parse_winsor_limits(args.winsor_limits)
    horizons, num_horizons, horizon_weights = resolve_horizon_config(args)
```

**Что обязательно сохранить один в один:**

- все текущие `parser.add_argument(...)`;
- все `choices=...`;
- все текущие `default=...`;
- поведение `BooleanOptionalAction`;
- help-тексты можно переносить без изменения смысла;
- пользовательский CLI должен остаться совместимым побайтно по именам флагов.

**Что нельзя делать:**

- нельзя переименовывать флаги;
- нельзя переносить часть флагов в новые файлы и менять способ вызова;
- нельзя менять default для `--mode`, `--scheduler`, `--precision_mode`, `--num_workers`, `--enable_progress_bar` и соседних train-control флагов.

---

### Подзадача 322.3: Вынести runtime/bootstrap в `train_runtime.py`

**Файлы:** `python_lab/src/train.py`, `python_lab/src/train_runtime.py`

**Что переносим из `train()`**

Перенести в отдельный модуль все runtime helper-и из блока сразу после `args = parser.parse_args()`:

- `build_dataloader_kwargs(...)`
- `pl.seed_everything(42)`
- `base_path`, `data_path`, `norm_params_path`, `checkpoint_dir`, `cache_dir`
- RAM availability check и rough dataset size estimate
- выбор `trainer_precision`
- общие сборщики `Trainer` kwargs

**Новый API:**

```python
@dataclass
class TrainPaths:
    base_path: Path
    data_path: Path
    norm_params_path: Path
    checkpoint_dir: Path
    cache_dir: Path

def seed_training(seed: int = 42) -> None: ...
def build_train_paths(module_file: str, symbol: str) -> TrainPaths: ...
def build_dataloader_kwargs(args, *, shuffle: bool) -> dict: ...
def resolve_trainer_precision(args) -> int | str: ...
def warn_if_dataset_may_exceed_ram(paths: TrainPaths, symbol: str, seq_len: int) -> None: ...
```

**Что заменить в `train.py`:**

```python
# БЫЛО
pl.seed_everything(42)
base_path = Path(__file__).parent.parent.parent
data_path = ...
checkpoint_dir = ...

def build_dataloader_kwargs(shuffle: bool):
    ...

if args.precision_mode == "32":
    trainer_precision = 32
...

# СТАЛО
from .train_runtime import (
    build_dataloader_kwargs,
    build_train_paths,
    resolve_trainer_precision,
    seed_training,
    warn_if_dataset_may_exceed_ram,
)

seed_training()
paths = build_train_paths(__file__, args.symbol)
warn_if_dataset_may_exceed_ram(paths, args.symbol, args.seq_len)
trainer_precision = resolve_trainer_precision(args)
```

**Что важно:**

- `build_dataloader_kwargs()` должен стать единым источником правды для:
  - main train loaders
  - recreated loaders после Optuna
  - CV fold loaders
  - trial loaders в Optuna objective

---

### Подзадача 322.4: Вынести metadata side effect в `train_metadata.py`

**Файлы:** `python_lab/src/train.py`, `python_lab/src/train_metadata.py`

**Что переносим:**

- `update_model_metadata(...)` (`1543-1580`)

**Новый API:**

```python
def update_model_metadata(paths, symbol, args, winsor_limits, norm_params_path): ...
```

или сохранить исходную сигнатуру, если это упростит перенос без риска.

**Что заменить в `train.py`:**

```python
# БЫЛО
def update_model_metadata(...): ...
...
update_model_metadata(base_path, args.symbol, args, winsor_limits, norm_params_path)

# СТАЛО
from .train_metadata import update_model_metadata
...
update_model_metadata(paths.base_path, args.symbol, args, winsor_limits, paths.norm_params_path)
```

**Что важно:**

- не менять содержимое `metadata.json`;
- не менять формат поля `normalization`;
- не менять путь сохранения:
  - `bots/<SYMBOL>/models/metadata.json`

---

### Подзадача 322.5: Вынести data orchestration в `train_data.py`

**Файлы:** `python_lab/src/train.py`, `python_lab/src/train_data.py`

**Что переносим из `train()`**

Перенести в отдельный модуль последовательность:

1. `LOBDataLoader(...)` и `loader.load_data(...)` (`около 1828`)
2. `FeatureEngineer(...).transform(...)` (`около 1881`)
3. `Labeler(...).add_labels(...)` (`1918-1928`)
4. `Normalizer(...)` (`1931`)
5. создание `full_dataset = LOBDataset(...)` (`2053+`)
6. chronological split `train/val/test`
7. `normalizer.fit(...)`, `save(...)`, `update_model_metadata(...)` (`2099-2107`)
8. sample-based NaN diagnostics
9. `DataLoader(...)` для `train_loader`, `val_loader`, `test_loader`
10. расчет class weights

**Новый API:**

```python
@dataclass
class PreparedTrainingData:
    df: object
    normalizer: Normalizer
    full_dataset: LOBDataset
    train_ds: object
    val_ds: object
    test_ds: object
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_weights: np.ndarray
    past_returns_lags: list[int]
    in_channels: int
    num_horizons: int
    horizon_weights: list[float] | None
    regime_detector: object | None
    regime_weights: object | None
    num_regimes: int

def prepare_training_data(args, paths, winsor_limits) -> PreparedTrainingData: ...
```

**Что заменить в `train.py`:**

```python
# БЫЛО
loader = LOBDataLoader(...)
df = loader.load_data(...)
fe = FeatureEngineer(...)
df = fe.transform(df)
labeler = Labeler(...)
df = labeler.add_labels(df)
normalizer = Normalizer(...)
full_dataset = LOBDataset(...)
...
train_loader = DataLoader(...)
val_loader = DataLoader(...)
test_loader = DataLoader(...)
weights = ...

# СТАЛО
from .train_data import prepare_training_data

prepared = prepare_training_data(args, paths, winsor_limits)
```

**Что нельзя менять внутри `prepare_training_data()`:**

- `FeatureEngineer` должен идти до `Labeler`;
- `Labeler` должен идти до `LOBDataset`;
- `normalizer.fit()` должен идти только на train-части и только на channel-space;
- `TrainSubset` должен использоваться только для train-части;
- `val_ds` и `test_ds` должны остаться обычными `Subset`, без train-mode augmentation;
- class weights нужно считать от train-части;
- `build_dataloader_kwargs()` должен вызываться из `train_runtime.py`, а не дублироваться локально.

**Какие локальные helper-и стоит завести в `train_data.py`:**

```python
def parse_past_returns_lags(raw: str) -> list[int]: ...
def build_full_dataset(...): ...
def split_dataset_chronologically(full_dataset): ...
def fit_normalizer_on_train_subset(...): ...
def run_normalized_nan_checks(...): ...
def compute_class_weights(...): ...
```

**Важно:**

Не переносить channel synthesis из `dataset.py` в `train_data.py`.  
`train_data.py` только orchestrates вызовы `LOBDataset` и `Normalizer`, но не дублирует их внутреннюю логику.

---

### Подзадача 322.6: Вынести сборку teacher/student модели в `train_model_factory.py`

**Файлы:** `python_lab/src/train.py`, `python_lab/src/train_model_factory.py`

**Что переносим из `train()`**

Перенести две ветки:

- distillation path (`2392-2488`)
- обычный train path (`2491-2537`)

**Новый API:**

```python
@dataclass
class BuiltTrainingModel:
    module: LiTModule
    teacher_model: object | None
    model_class_weights: object | None

def build_training_module(
    args,
    *,
    in_channels,
    past_returns_lags,
    num_horizons,
    horizon_weights,
    model_class_weights,
    regime_detector,
    regime_weights,
    num_regimes,
    winsor_limits,
) -> BuiltTrainingModel: ...
```

**Что заменить в `train.py`:**

```python
# БЫЛО
if args.mode == "distill":
    ...
    model = LiTModule(...)
else:
    ...
    model = LiTModule(...)

# СТАЛО
from .train_model_factory import build_training_module

built = build_training_module(
    args,
    in_channels=prepared.in_channels,
    past_returns_lags=prepared.past_returns_lags,
    num_horizons=prepared.num_horizons,
    horizon_weights=prepared.horizon_weights,
    model_class_weights=model_class_weights,
    regime_detector=prepared.regime_detector,
    regime_weights=prepared.regime_weights,
    num_regimes=prepared.num_regimes,
    winsor_limits=winsor_limits,
)
model = built.module
teacher_model = built.teacher_model
```

**Что важно:**

- `LiTConfig` остается в `lit_model.py`;
- `teacher_module = LiTModule.load_from_checkpoint(...)` не менять по смыслу;
- сохранить использование:
  - `teacher_lit.pt`
  - `student_d_model`, `student_nhead`, `student_num_layers`
  - `alpha`, `temperature`
- не менять связку `teacher_module.model`.

---

### Подзадача 322.7: Вынести Optuna seq_len search в `train_optuna.py`

**Файлы:** `python_lab/src/train.py`, `python_lab/src/train_optuna.py`

**Что переносим:**

- `objective_seq_len_search(...)` (`1200-1541`)
- блок внутри `train()` начиная с `if args.optuna_seq_len_search:` (`2647+`)

**Новый API:**

```python
def objective_seq_len_search(...): ...
def run_optuna_seq_len_search(args, paths, prepared, winsor_limits): ...
```

**Что заменить в `train.py`:**

```python
# БЫЛО
if args.optuna_seq_len_search:
    ...
    study = optuna.create_study(...)
    study.optimize(...)
    ...
    full_dataset = LOBDataset(...)
    train_loader = DataLoader(...)
    val_loader = DataLoader(...)

# СТАЛО
from .train_optuna import run_optuna_seq_len_search

if args.optuna_seq_len_search:
    prepared = run_optuna_seq_len_search(args, paths, prepared, winsor_limits)
```

**Что важно:**

- `objective_seq_len_search()` обязан использовать те же factory-и `build_dataloader_kwargs()`, `LiTModule`, `LiTConfig`, `Normalizer`, что и основной pipeline;
- recreated dataset после поиска должен собираться той же логикой, а не отдельной копией;
- не допускать, чтобы Optuna path начал жить по другой схеме split/normalizer/trainer.

---

### Подзадача 322.8: Вынести CV режим в `train_cv.py`

**Файлы:** `python_lab/src/train.py`, `python_lab/src/train_cv.py`

**Что переносим:**

Весь блок:

- `if args.mode == "cv": ...`
- fold loop;
- fold-level `ModelCheckpoint`, `TensorBoardLogger`, `pl.Trainer`;
- final holdout evaluation лучшей fold-модели;
- запись `cv_results.json`

**Новый API:**

```python
def run_cross_validation(args, paths, prepared, winsor_limits): ...
```

**Что заменить в `train.py`:**

```python
# БЫЛО
if args.mode == "cv":
    ...
else:
    ...

# СТАЛО
from .train_cv import run_cross_validation

if args.mode == "cv":
    run_cross_validation(args, paths, prepared, winsor_limits)
    return
```

**Что важно:**

- fold-модели должны создаваться fresh, без reuse основного `model`;
- `PurgedKFold` и buffer-параметры оставить без изменения;
- fold-level `Trainer` должен использовать тот же runtime helper для precision и shared kwargs;
- не менять логику выбора лучшего fold и итогового holdout.

---

### Подзадача 322.9: Вынести post-training этапы в `train_postprocess.py`

**Файлы:** `python_lab/src/train.py`, `python_lab/src/train_postprocess.py`

**Что переносим:**

1. `enable_dropout` (`1196-1198`)
2. обычный post-fit path:
   - MC Dropout uncertainty (`после trainer.fit(model, ...)`)
   - pruning loop
   - final holdout evaluation
   - `teacher-vs-student`
   - final copy/export hints

**Новый API:**

```python
def enable_dropout(m): ...
def run_mc_dropout_uncertainty(...): ...
def run_model_pruning(...): ...
def run_holdout_evaluation(...): ...
def compare_teacher_student(...): ...
def copy_best_checkpoint_to_target(...): ...
```

**Что заменить в `train.py`:**

```python
# БЫЛО
trainer.fit(model, train_loader, val_loader)
...
MC Dropout block
...
pruning block
...
final holdout evaluation
...
teacher-vs-student
...
copy best checkpoint to teacher_lit.pt / lit.pt

# СТАЛО
from .train_postprocess import (
    copy_best_checkpoint_to_target,
    run_holdout_evaluation,
    run_mc_dropout_uncertainty,
    run_model_pruning,
)

trainer.fit(model, prepared.train_loader, prepared.val_loader)
run_mc_dropout_uncertainty(...)
run_model_pruning(...)
evaluation_result = run_holdout_evaluation(...)
copy_best_checkpoint_to_target(...)
```

**Что важно:**

- `enable_dropout()` использовать только здесь, не оставлять в `train.py`;
- pruning должен работать с тем же `checkpoint_callback`, что и основной train;
- путь сохранения:
  - `bots/<SYMBOL>/models/teacher_lit.pt`
  - `bots/<SYMBOL>/models/lit.pt`
  должен остаться прежним;
- сравнение teacher/student должно запускаться только в `distill` режиме.

---

### Подзадача 322.10: Сжать `train.py` до thin entrypoint

**Файлы:** `python_lab/src/train.py`

**Что должно остаться в финальном `train.py`:**

- импорт новых модулей;
- `def train():`
- последовательный вызов orchestration-функций;
- `if __name__ == "__main__": train()`

**Ожидаемая структура финального файла:**

```python
from .train_cli import parse_train_args, parse_winsor_limits
from .train_runtime import build_train_paths, seed_training
from .train_data import prepare_training_data
from .train_model_factory import build_training_module
from .train_optuna import run_optuna_seq_len_search
from .train_cv import run_cross_validation
from .train_postprocess import (
    run_holdout_evaluation,
    run_mc_dropout_uncertainty,
    run_model_pruning,
    copy_best_checkpoint_to_target,
)

def train():
    args = parse_train_args()
    winsor_limits = parse_winsor_limits(args.winsor_limits)
    seed_training()
    paths = build_train_paths(__file__, args.symbol)
    prepared = prepare_training_data(args, paths, winsor_limits)

    if args.optuna_seq_len_search:
        prepared = run_optuna_seq_len_search(args, paths, prepared, winsor_limits)

    if args.mode == "cv":
        run_cross_validation(args, paths, prepared, winsor_limits)
        return

    built = build_training_module(...)
    trainer_bundle = ...
    trainer.fit(...)
    run_mc_dropout_uncertainty(...)
    run_model_pruning(...)
    run_holdout_evaluation(...)
    copy_best_checkpoint_to_target(...)

if __name__ == "__main__":
    train()
```

**Целевое ограничение:**

После рефакторинга `train.py` не должен содержать 1000+ строк.  
Он должен остаться orchestration-файлом, а не новым монолитом.

## Порядок выполнения рефакторинга

Делать именно в таком порядке:

1. `train_metadata.py`
2. `train_cli.py`
3. `train_runtime.py`
4. `train_module.py`
5. `train_data.py`
6. `train_model_factory.py`
7. `train_optuna.py`
8. `train_cv.py`
9. `train_postprocess.py`
10. зачистка и упрощение `train.py`

Почему именно так:

- сначала выносятся безопасные helper-и и side effect;
- затем фиксируется единый runtime/config слой;
- только потом выносится `LiTModule` и data orchestration;
- Optuna/CV/postprocess переходят на уже готовые shared factory-и;
- в конце `train.py` сжимается без дублирования логики.

## Что нельзя сломать

1. `train.py` должен остаться точкой входа с тем же CLI.
2. Нельзя менять формат batch.
3. Нельзя менять пути сохранения best model artifacts.
4. Нельзя дублировать `build_dataloader_kwargs()` по модулям.
5. Нельзя переносить внутреннюю channel logic из `dataset.py`.
6. Нельзя переносить `LiTConfig`/`LiTModel` из `lit_model.py`.
7. Нельзя менять порядок `FeatureEngineer -> Labeler -> split -> normalizer.fit(train only)`.
8. Нельзя допускать циклические импорты вида:

```text
train_data -> train_module -> train_data
train_postprocess -> train_model_factory -> train_postprocess
```

Правильное направление:

```text
train.py
-> train_cli / train_runtime / train_data / train_model_factory / train_optuna / train_cv / train_postprocess
-> dataset / features / labels / normalization / lit_model / utils
```

## Верификация

### Структурная верификация

1. После переноса каждый новый файл импортируется отдельно:

```bash
python -c "from python_lab.src import train_cli, train_runtime, train_data, train_metadata, train_module, train_model_factory, train_optuna, train_cv, train_postprocess"
```

2. `python -m python_lab.src.train --help` показывает тот же набор CLI-флагов, что и до рефакторинга.

3. В `train.py` не остается локальных копий:

- `TrainSubset`
- `ProfilerCallback`
- `LiTModule`
- `objective_seq_len_search`
- `update_model_metadata`
- `enable_dropout`

### Поведенческая верификация

1. Основной `train` путь все еще проходит через:

```text
parse args
-> seed/path bootstrap
-> load/feature engineer/labeler
-> dataset split
-> normalizer.fit(train only)
-> dataloaders
-> model build
-> trainer.fit
-> holdout/postprocess
```

2. `distill` путь:

- все еще требует `--teacher_path`;
- все еще грузит teacher через `LiTModule.load_from_checkpoint(...)`;
- все еще сохраняет student как `lit.pt`.

3. `cv` путь:

- все еще делает purged k-fold;
- все еще пишет `cv_results.json`;
- все еще оценивает holdout лучшей fold-моделью.

4. `optuna` путь:

- все еще использует ту же dataset/model/runtime логику, что и основной train;
- не создает отдельной, расходящейся версии pipeline.

### Архитектурная верификация

1. `docs/000-architecture.md` соответствует фактической раскладке файлов.
2. В новых модулях нет циклических импортов.
3. `train.py` остается thin entrypoint, а не новым большим файлом-оркестратором на сотни строк.

## Ожидаемый итог

После выполнения задачи 322:

- `python_lab/src/train.py` перестанет быть монолитом;
- при будущих изменениях можно будет отдельно править:
  - CLI
  - data bootstrap
  - model build
  - Optuna
  - CV
  - post-training steps
- риск сломать critical order и train/val behavior заметно снизится;
- дальнейшие задачи по `LiTModule`, `Normalizer`, `distillation`, `pruning`, `holdout` можно будет делать точечно, а не через редактирование одного огромного файла.
