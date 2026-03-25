# Задача 323: Исправление OFI/Delta pipeline в Python train

**Дата создания:** 24.03.2026  
**Статус:** ЗАПЛАНИРОВАНО  
**Категория:** Обучение Python  
**Зависимости:** Задачи 314, 315, 318, 319, 322

## Цель

Заменить текущий неустойчивый pipeline формирования и масштабирования каналов `OFI`, `DeltaImb`, `DeltaSpread` в `python_lab`, чтобы обучение LiT получало эти каналы из корректно инициализированного `update_id`/`ofi_cache` и в устойчивом масштабе без fallback на `diff(imbalance)` в основном пути.

## Краткий контекст

Parquet-файлы уже считаются целостными. Источник сбоя находится внутри Python-пайплайна после загрузки данных.  
В задачах 314, 315, 318 и 319 OFI и delta-каналы уже менялись, но текущее состояние всё ещё содержит два неудачных решения:

1. `OFI` строится через fallback `diff(imbalance)`, если `update_id_raw` или `ofi_cache` не готовы к моменту расчёта каналов.
2. `OFI`, `DeltaImb`, `DeltaSpread` проходят через тот же общий контур масштабирования, что и статичные каналы, после чего дополнительно жёстко режутся clamp'ом, из-за чего динамические каналы становятся нестабильными и частично насыщенными.

Ниже зафиксировано одно ТЗ на замену этого пайплайна без изменения архитектуры LiT, labels, loss и Rust-части.

---

## Подзадачи

### Подзадача 323.1: Перенести инициализацию `update_id_raw` и `ofi_cache` в начало memory-pipeline

**Файлы:**
- `python_lab/src/dataset.py`

**Изменения:**

1. В `LOBDataset._init_memory_mode` переставить порядок инициализации так, чтобы `self.update_id_raw` и `self.ofi_cache` создавались до вызовов:
   - `self._compute_channels_for_normalization(...)`
   - `_fit_normalizer_on_train(...)` через `train_data.py`
   - первой выборки `__getitem__`

2. В начале `LOBDataset._init_memory_mode` сразу после чтения `DataFrame` добавить явное извлечение:
   - `last_update_id` из исходного `df`
   - fallback-переименование `timestamp -> timestamp_ms` оставить как есть
   - сохранить `last_update_id` в `self.update_id_raw` с dtype `np.int64`

3. В `LOBDataset._init_memory_mode` сразу после подготовки `bid_p_matrix`, `ask_p_matrix`, `bid_v_matrix`, `ask_v_matrix` вычислять `self.ofi_cache` через отдельную функцию precompute, а не внутри `_process_sample`.

4. Удалить из `_init_memory_mode` ветку, где при отсутствии `update_id_raw` пишется диагностическое сообщение и активируется запасной путь через `diff(imbalance)`.

5. После этой перестановки `self.ofi_cache` должен существовать всегда в режиме `memory`, если в `df` есть `last_update_id`.

**Точная замена логики:**

- Было:
  - сначала формируется `x_raw`
  - затем часть каналов строится без гарантированного `update_id_raw`
  - затем в `_process_sample` и `_compute_channels_for_normalization` возможен fallback

- Должно стать:
  - `self.update_id_raw`
  - `self.ofi_cache`
  - `self.vib_cache`
  - `self.past_ret_cache`
  - `self.x_raw`
  - `normalizer.fit(...)`

---

### Подзадача 323.2: Убрать `diff(imbalance)` из основного пути формирования OFI

**Файлы:**
- `python_lab/src/dataset.py`

**Изменения:**

1. В функции `compute_ofi_from_lob` оставить один рабочий алгоритм расчёта `OFI` по `bid_p`, `ask_p`, `bid_v`, `ask_v`, `update_ids`.

2. В `compute_ofi_from_lob_cache` оставить восстановление сырых объёмов из `log1p`, затем передавать их в `compute_ofi_from_lob`.

3. В `LOBDataset._calculate_6_channels_raw` убрать резервную ветку:
   ```python
   ofi_raw = torch.diff(imb_ch_raw, dim=0, prepend=imb_ch_raw[:1])
   ```
   из основного исполнения.

4. В `LOBDataset._calculate_6_channels_raw` принимать `ofi_precomputed` как обязательный источник `OFI` для `memory`-режима и строить канал только из него:
   - `ofi_raw = ofi_precomputed.unsqueeze(-1).expand(-1, 50)`

5. В `_process_sample` и `_compute_channels_for_normalization` передавать в `_calculate_6_channels_raw` только `ofi_precomputed`, взятый из `self.ofi_cache`.

6. Для аварийного сценария отсутствия `ofi_cache` оставить не `diff(imbalance)`, а жёсткое исключение:
   - `raise RuntimeError("OFI cache is not initialized before channel construction")`

**Точная замена по функциям:**

- `LOBDataset._calculate_6_channels_raw`
- `LOBDataset._process_sample`
- `LOBDataset._compute_channels_for_normalization`

---

### Подзадача 323.3: Отделить масштабирование OFI/Delta от общего normalizer-пайплайна

**Файлы:**
- `python_lab/src/dataset.py`
- `python_lab/src/normalization.py`

**Изменения:**

1. В `LOBDataset._calculate_6_channels_raw` оставить `OFI`, `DeltaImb`, `DeltaSpread` в сыром динамическом виде до общего stack каналов.

2. В `LOBDataset._process_sample` заменить текущую схему:
   - `normalize_channel(...)`
   - затем общий clamp `[-5, 5]`
   - затем отдельный clamp `[-3, 3]`

   на новый порядок для динамических каналов:
   - `OFI`: `symlog_transform(ofi_raw)`
   - `DeltaImb`: `symlog_transform(di_raw)`
   - `DeltaSpread`: `symlog_transform(ds_raw)`
   - затем отдельная робастная нормализация этих трёх каналов по параметрам, обученным только на этих же каналах
   - затем единый мягкий clamp `[-4, 4]` без второго дополнительного clamp только для OFI/Delta

3. Для статичных каналов `MicropriceDev`, `Vol`, `Imb`, `VIB`, `Ret_10`, `Ret_50`, `Ret_100`, `Spread` оставить текущий путь `normalize_channel(...)`.

4. В `LOBDataset._compute_channels_for_normalization` перестать обучать OFI/Delta через тот же общий `DataFrame` `feat_0..feat_549`, как будто это обычные каналы без временной динамики.

5. Добавить в `Normalizer` отдельную группу параметров для динамических каналов:
   - `dynamic_params["ofi"]`
   - `dynamic_params["delta_imb"]`
   - `dynamic_params["delta_spread"]`

6. В `Normalizer.fit(...)` добавить приём отдельного массива динамических каналов и сохранять для них:
   - `median`
   - `iqr`
   - `winsor_low`
   - `winsor_high`

7. В `Normalizer.save(...)` и `Normalizer.load(...)` сериализовать `dynamic_params` в тот же JSON рядом с обычными `params`.

8. В `LOBDataset._process_sample` добавить отдельную функцию применения этой нормализации, например:
   - `_normalize_dynamic_channel(channel_data, channel_name)`

**Точная замена масштабирующего контура:**

- Было:
  - динамические каналы нормализуются как статичные по `feat_150+`
  - затем дважды режутся clamp'ом

- Должно стать:
  - `symlog`
  - `robust normalize` по отдельным `dynamic_params`
  - один clamp `[-4, 4]`

---

### Подзадача 323.4: Привязать fit normalizer к уже исправленному OFI/Delta channel-space

**Файлы:**
- `python_lab/src/train_data.py`
- `python_lab/src/dataset.py`

**Изменения:**

1. В `_fit_normalizer_on_train(...)` оставить вызов `full_dataset._compute_channels_for_normalization(train_indices_for_fit)`, но изменить сам `LOBDataset._compute_channels_for_normalization(...)` так, чтобы этот метод:
   - использовал уже готовый `self.ofi_cache`
   - использовал ту же ветку `symlog + dynamic robust stats` для `OFI`, `DeltaImb`, `DeltaSpread`
   - возвращал channel-space без ветки `diff(imbalance)`

2. В `PreparedTrainingData` инициализация `normalizer` остаётся прежней, но fit должен происходить уже после готовых:
   - `self.update_id_raw`
   - `self.ofi_cache`
   - `self.vib_cache`
   - `self.past_ret_cache`

3. В `train_data.py` не добавлять новый этап разметки, loss или labels. Изменение ограничить только тем, что `normalizer.fit(...)` получает исправленный набор каналов.

4. После `normalizer.save(...)` в артефакте `norm.json` должны появиться отдельные секции для динамических каналов, а не только общий список `feat_*`.

---

### Подзадача 323.5: Заменить диагностический вывод каналов на итоговый post-transform масштаб

**Файлы:**
- `python_lab/src/dataset.py`
- `python_lab/src/train_module.py`

**Изменения:**

1. В `LOBDataset._process_sample` изменить диагностику `[ДИАГНОСТИКА 316]` так, чтобы для каналов `OFI`, `DeltaImb`, `DeltaSpread` печатались значения уже после:
   - `symlog`
   - отдельной dynamic normalizer
   - финального clamp

2. В `_log_clip_saturation(...)` заменить лимиты:
   - для `OFI`, `DeltaImb`, `DeltaSpread` использовать только новый единый лимит `4.0`
   - убрать старое разделение `5.0` и `3.0` для этих каналов

3. В `LiTModule.log_channel_statistics(...)` оставить те же названия каналов, но логируемые диапазоны должны соответствовать новому масштабу без старого двойного clamp-контура.

4. Из `output.txt` после реализации должны исчезнуть старые паттерны, где `OFI` и delta-каналы массово упираются в `-3.0000` или `3.0000` уже на первых диагностических сэмплах.

---

### Подзадача 323.6: Зафиксировать новый контракт OFI/Delta pipeline внутри dataset.py

**Файлы:**
- `python_lab/src/dataset.py`

**Изменения:**

1. В заголовке `LOBDataset._calculate_6_channels_raw` обновить docstring:
   - `OFI` берётся из `self.ofi_cache`, рассчитанного до построения выборок
   - `DeltaImb` и `DeltaSpread` проходят отдельный dynamic scaling pipeline

2. В заголовке `_process_sample` добавить комментарий с новым порядком:
   - raw LOB
   - cached OFI
   - raw delta channels
   - static normalize
   - dynamic symlog + robust normalize
   - one-pass clamp

3. В `_compute_channels_for_normalization` добавить комментарий, что этот метод обязан использовать тот же OFI/Delta pipeline, что и `__getitem__`, без отдельной упрощённой ветки.

4. Внутри `dataset.py` убрать текстовые сообщения, в которых OFI называется fallback-каналом на базе imbalance. Оставить только контракт cached OFI.

---

## Критерий завершения

После реализации в логах обучения должны одновременно выполняться следующие изменения:

- строки вида `OFI will use fallback` и `feat_update_id not found` отсутствуют;
- в диагностике первых сэмплов `OFI`, `DeltaImb`, `DeltaSpread` больше не упираются массово в `-3.0000` и `3.0000`;
- `Channel 3 (OFI)`, `Channel 9 (DeltaImb)`, `Channel 10 (DeltaSpread)` имеют конечные средние и стандартные отклонения без резкого saturation на старте эпохи;
- `Epoch 1` и `Epoch 5` завершаются без отрицательного перекоса уверенности на ложных directional-сигналах, вызванного нестабильным OFI/Delta input;
- `MCC`, `Macro-F1` и `Directional Accuracy` перестают деградировать относительно текущего запуска из `output.txt` уже на первых завершённых эпохах.
