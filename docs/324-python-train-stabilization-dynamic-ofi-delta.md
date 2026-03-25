# Задача 324: Стабилизировать dynamic feature contract OFI, DeltaImb, DeltaSpread для LiT

**Дата создания:** 25.03.2026  
**Статус:** ЗАПЛАНИРОВАНО  
**Категория:** Обучение Python  
**Зависимости:** задачи 318, 319, 320, 321, 322, 323

## Цель задачи

Исправить один конкретный источник деградации обучения LiT: привести каналы `OFI`, `DeltaImb`, `DeltaSpread` к одному и тому же корректному контракту между:

- построением сырых временных рядов в `python_lab/src/dataset.py`,
- fit нормализатора на train split в `python_lab/src/train_data.py`,
- применением нормализации в `__getitem__` и валидационной диагностикой.

Новая задача не меняет архитектуру LiT, не меняет loss, labels, Rust-выгрузку и не добавляет новую модель. Она только доводит до конца pipeline динамических каналов, который после задачи 323 все еще остается частично неконсистентным.

## Почему это приоритет №1

Последний лог обучения в `output.txt` показывает, что именно dynamic-каналы ломают вход модели:

- `OFI` упирается в clamp на `29%` значений;
- `DeltaImb` упирается в clamp на `33%` значений;
- `DeltaSpread` упирается в clamp на `16%` значений;
- при этом `MCC` остается около `0.01`, `Directional Accuracy` без Flat около `0.10`, а directional edge почти нулевой или отрицательный.

Код подтверждает причину:

- в `python_lab/src/train_data.py` функция `_fit_normalizer_on_train(...)` обучает dynamic-normalizer не на полном train-распределении каналов, а только на трех срезах `arr[:,150]`, `arr[:,450]`, `arr[:,500]`;
- в `python_lab/src/dataset.py` каналы `DeltaImb` и `DeltaSpread` строятся через `torch.diff(...)` по соседним строкам окна, то есть по локальной нарезке sample, а не по event-consistent потоку с учетом `last_update_id`;
- в результате LiT получает не устойчивый микроструктурный сигнал, а частично насыщенный и частично артефактный шум.

Пока этот контракт не исправлен, любые следующие попытки трогать threshold, loss, class weights или саму LiT-архитектуру будут лечить симптомы, а не причину.

---

## Подзадачи

### 324.1. Перевести `DeltaImb` и `DeltaSpread` на тот же event-consistent источник, что и `OFI` - ЗАВЕРШЕНО

**Файлы:**
- `python_lab/src/dataset.py`

**Что поменять:**

1. В memory-pipeline рядом с `self.ofi_cache` добавить предрасчет:
   - `self.delta_imb_cache`
   - `self.delta_spread_cache`

2. Считать эти кэши по полному хронологическому ряду датасета, а не внутри отдельного sample-окна.

3. При расчете использовать `self.update_id_raw`:
   - если между соседними строками нет нового обновления стакана, не считать это валидным event-delta;
   - для повторов snapshot или артефактов ресемплинга не делать обычный `diff`, а занулять вклад этого шага.

4. В `LOBDataset._calculate_6_channels_raw(...)` удалить текущую логику:
   - `delta_imb = torch.diff(imb_ch_raw[:, 0], prepend=...)`
   - `delta_spread = torch.diff(spread_1d, prepend=...)`

5. Вместо этого брать уже готовые значения из:
   - `self.delta_imb_cache[start:end]`
   - `self.delta_spread_cache[start:end]`

6. Расширение этих двух каналов до формы `(seq_len, 50)` оставить таким же, как у `OFI`: один временной ряд на шаг времени, размноженный по level-оси только на финальном этапе сборки channel tensor.

**Зачем это нужно:**

Сейчас `OFI` имеет event-смысл, а `DeltaImb` и `DeltaSpread` имеют sample-local смысл. Эти три канала не согласованы между собой и подмешивают в модель разные типы времени.

---

### 324.2. Переделать fit dynamic-normalizer так, чтобы он видел полное train-распределение, а не три суррогатных столбца - ЗАВЕРШЕНО

**Файлы:**
- `python_lab/src/train_data.py`
- `python_lab/src/normalization.py`

**Что поменять:**

1. В `_fit_normalizer_on_train(...)` убрать текущий выбор:
   - `arr[:, 150]` для `OFI`
   - `arr[:, 450]` для `DeltaImb`
   - `arr[:, 500]` для `DeltaSpread`

2. Вместо этого собрать `dynamic_data` из полного train split:
   - брать все временные значения `self.ofi_cache`, `self.delta_imb_cache`, `self.delta_spread_cache`, которые реально входят в train-окна;
   - применять к ним тот же `symlog`, который потом используется в `__getitem__`;
   - fit делать отдельно по каждому dynamic-каналу на всей train-выборке, а не по одному representative column.

3. В `Normalizer.fit(...)` оставить отдельный блок `dynamic_params`, но заполнять его уже настоящими train-распределениями каналов, а не прокси-срезами.

4. Для каждого dynamic-канала хранить минимум:
   - `median`
   - `iqr`
   - `q01`
   - `q99`

5. В `Normalizer.save(...)` и `Normalizer.load(...)` проверить, что эти параметры сериализуются и восстанавливаются отдельно от обычных `feat_*`.

**Зачем это нужно:**

При текущем fit медиана и IQR оцениваются по слишком узкому подмножеству данных. Для heavy-tail каналов это почти гарантирует плохой scale и последующий mass clipping.

---

### 324.3. Синхронизировать train-fit pipeline и runtime pipeline для dynamic-каналов - ЗАВЕРШЕНО

**Файлы:**
- `python_lab/src/dataset.py`
- `python_lab/src/train_data.py`

**Что поменять:**

1. В `LOBDataset._compute_channels_for_normalization(...)` использовать ту же ветку dynamic-каналов, что и в `_process_sample(...)`.

2. Не допускать, чтобы fit normalizer работал по упрощенному пути, а `__getitem__` по другому.

3. Один и тот же порядок должен быть зафиксирован в обоих местах:
   - взять сырой cached signal;
   - применить `symlog`;
   - применить `transform_dynamic(...)`;
   - применить единый финальный clamp.

4. В `dataset.py` явно оформить это через отдельную вспомогательную функцию, чтобы один и тот же код использовался и для fit, и для обычной выдачи sample.

**Зачем это нужно:**

Сейчас train-fit и runtime визуально похожи, но не идентичны. Для нестабильных каналов этого достаточно, чтобы модель училась на одном распределении, а видеть на входе другое.

---

### 324.4. Заменить per-sample clip-диагностику на диагностику по всему train split - ЗАВЕРШЕНО

**Файлы:**
- `python_lab/src/dataset.py`
- `python_lab/src/train_module.py`

**Что поменять:**

1. Сохранить текущий локальный sample-debug, но перестать использовать его как главный индикатор качества канала.

2. Добавить агрегированную диагностику по train split для `OFI`, `DeltaImb`, `DeltaSpread` после полного пайплайна преобразования:
   - `min`, `max`, `mean`, `std`
   - `p01`, `p50`, `p99`
   - доля значений `<= -limit`
   - доля значений `>= limit`
   - общая доля saturation
   - доля точных нулей

3. Печатать эту сводку один раз сразу после fit normalizer и до старта первой эпохи.

4. Для dynamic-каналов не использовать старую интерпретацию "sample выглядит нормально", если train-агрегаты уже показывают клиппинг на десятках процентов.

**Зачем это нужно:**

Текущая диагностика видит только отдельный sample. Это поздний и слабый сигнал. Нам нужен агрегатный контроль именно на train split, где оценивается normalizer.

---

### 324.5. Добавить hard-guard, который не даст запускать обучение на заведомо сломанном dynamic-scale - ЗАВЕРШЕНО

**Файлы:**
- `python_lab/src/train_data.py`
- `python_lab/src/train_cli.py`

**Что поменять:**

1. После fit dynamic-normalizer и расчета train-диагностики добавить проверку качества scale.

2. Если для любого из каналов выполняется хотя бы одно условие:
   - saturation после финального clamp больше `10%`,
   - доля точных нулей аномально высокая и не объясняется market inactivity,
   - `q99 - q01` после transform слишком мал и канал фактически схлопнулся,
   то обучение не запускать молча.

3. Сделать это либо через:
   - `raise RuntimeError(...)`, либо
   - отдельный CLI-флаг вроде `--allow-bad-dynamic-scale` со значением по умолчанию `False`.

4. В тексте ошибки выводить конкретный канал и его train-статистику, а не общий "normalization issue".

**Зачем это нужно:**

Сейчас пайплайн пропускает очевидно плохой scale и только потом тратит часы на обучение, которое предсказуемо дает почти нулевой directional signal.

---

### 324.6. Провести узкую причинно-следственную проверку на одном повторном запуске - ЗАВЕРШЕНО

**Файлы:**
- `python_lab/src/train_cli.py`
- `docs/train_logs.md`
- `output.txt` нового запуска

**Что поменять в процессе проверки:**

1. Не менять:
   - архитектуру LiT,
   - labels,
   - threshold,
   - loss,
   - optimizer,
   - batch size.

2. Повторить запуск с теми же hyperparameters, что и в текущем `output.txt`, чтобы сравнение было честным.

3. Зафиксировать в логах отдельно:
   - saturation по `OFI`, `DeltaImb`, `DeltaSpread` до и после исправления;
   - `MCC`, `Macro-F1`, `ECE`, `Directional Accuracy`;
   - directional edge по `Up` и `Down`.

4. В `docs/train_logs.md` добавить краткую таблицу "до / после" именно по этой задаче.

**Критерий приемки этой подзадачи:**

- `OFI` saturation падает заметно ниже текущих `29%`;
- `DeltaImb` saturation падает заметно ниже текущих `33%`;
- `DeltaSpread` saturation падает заметно ниже текущих `16%`;
- `MCC` и `Directional Accuracy` уходят от практически нулевого baseline без изменения LiT и без правок labels.

---

### 324.7. Сделать одну узкую абляцию, чтобы доказать, что проблема была именно в dynamic feature contract - ЗАВЕРШЕНО

**Файлы:**
- `python_lab/src/train_cli.py`
- `docs/train_logs.md`

**Что поменять в рамках эксперимента:**

1. После исправления пайплайна сделать ровно одну контрольную абляцию:
   - запуск с теми же параметрами, но без `OFI`, `DeltaImb`, `DeltaSpread`,
   - либо запуск с ними же после исправления.

2. Сравнить две конфигурации только по:
   - `MCC`
   - `Directional Accuracy`
   - directional edge
   - calibration (`ECE`)

3. Если исправленный вариант с dynamic-каналами не становится лучше абляции, не переходить сразу к новым идеям, а сначала повторно проверить:
   - event-consistency,
   - fit normalizer,
   - saturation guards.

**Зачем это нужно:**

Эта абляция отделяет реальный signal recovery от случайного улучшения и не дает снова обвинить LiT-архитектуру без доказательств.

---
=========
# Дополнителньые подзадачи - исправления ошибок.
### 324.8: Исправление ошибок предыдущих подзадач - ЗАВЕРШЕНО
В normalization.py в Normalizer.fit(..., dynamic_data=...) (блок, где сейчас q25/q75/iqr для dynamic) заменить расчёт iqr на защищённый.
Было:
q25 = float(np.quantile(arr, 0.25))
q75 = float(np.quantile(arr, 0.75))
iqr = float(q75 - q25) if not math.isnan(q75 - q25) else 1.0
entry = {"median": median, "iqr": iqr}
Сделать:

q10 = float(np.quantile(arr, 0.10))
q25 = float(np.quantile(arr, 0.25))
q75 = float(np.quantile(arr, 0.75))
q90 = float(np.quantile(arr, 0.90))

raw_iqr = float(q75 - q25)
alt_iqr = float(q90 - q10)
iqr_floor = 1e-3
iqr = raw_iqr if np.isfinite(raw_iqr) and raw_iqr >= iqr_floor else max(alt_iqr, iqr_floor)

entry = {"median": median, "iqr": iqr}
Там же, в том же блоке dynamic, всегда сохранять pre-clip границы для runtime (не зависеть от winsor_limits).
Добавить в entry:
entry["preclip_low"] = float(np.quantile(arr, 0.005))
entry["preclip_high"] = float(np.quantile(arr, 0.995))
И уже существующие winsor_low/winsor_high оставить как есть (если winsor_limits переданы).

В normalization.py в transform_dynamic() синхронизировать scaling со static robust-веткой.
Было:
return (data - median_t) / (iqr_t + self.eps)
# ...
return (data - median) / (iqr + self.eps)
Сделать:

scale_t = (iqr_t + self.eps) / self.scale_multiplier
return (data - median_t) / scale_t
# ...
scale = (iqr + self.eps) / self.scale_multiplier
return (data - median) / scale
В train_data.py в _log_dynamic_train_diagnostics() привести формулу к той же, что будет в runtime.
Было:
normed = (sym_arr - median) / (iqr + eps)
Сделать:

scale = (iqr + eps) / normalizer.scale_multiplier
normed = (sym_arr - median) / scale
В dataset.py в _apply_dynamic_transform() добавить мягкий pre-clip в symlog-пространстве перед нормализацией.
Было:
sym = symlog_transform(raw)
normed = self.normalizer.transform_dynamic(sym, channel_name)
clipped = torch.clamp(normed, -4.0, 4.0)
Сделать:

sym = symlog_transform(raw)
p = self.normalizer.dynamic_params.get(channel_name, {})
low = p.get("preclip_low")
high = p.get("preclip_high")
if low is not None and high is not None:
    low_t = torch.tensor(low, device=sym.device, dtype=sym.dtype)
    high_t = torch.tensor(high, device=sym.device, dtype=sym.dtype)
    sym = torch.clamp(sym, low_t, high_t)

normed = self.normalizer.transform_dynamic(sym, channel_name)
clipped = torch.clamp(normed, -4.0, 4.0)
В train_data.py в _fit_normalizer_on_train() добавить диагностический print до normalizer.fit(...) по каждому dynamic-каналу: raw p01/p50/p99, sym p01/p50/p99, будущий iqr.
Это нужно, чтобы видеть причину saturation до guard-а и быстро валидировать фикс.

------------
Все 5 правок задачи 324.8 выполнены:

normalization.py — защищённый IQR с fallback на Q10-Q90 + всегда сохраняем preclip_low/high
normalization.py — transform_dynamic синхронизирован со static robust-веткой через scale_multiplier
train_data.py — _log_dynamic_train_diagnostics использует ту же формулу что и runtime
dataset.py — _apply_dynamic_transform добавляет мягкий pre-clip перед нормализацией
train_data.py — диагностический print перед normalizer.fit() с raw/sym p01/p50/p99 и будущим IQR

## Задача 324.9: Исправление - у вас сейчас узкое место одно: `delta_imb` (sat 19.18%).  
Что править конкретно:

1. В [normalization.py](/D:/MAX/PYTHON/NEURAL-BOTS/neirobot-lit/python_lab/src/normalization.py) в `Normalizer.fit(..., dynamic_data=...)` изменить выбор `iqr` для dynamic-каналов.
Сейчас `delta_imb` берёт слишком узкий `q75-q25=0.0045`, из-за этого scale слишком маленький.
Заменить логику:
```python
raw_iqr = q75 - q25
alt_iqr = q90 - q10
iqr_floor = 1e-3

# было: iqr = raw_iqr (если не NaN)
# сделать:
if not np.isfinite(raw_iqr):
    iqr = max(alt_iqr, iqr_floor)
elif raw_iqr < max(iqr_floor, 0.35 * alt_iqr):
    iqr = max(alt_iqr, iqr_floor)   # fallback на широкий robust-range
else:
    iqr = max(raw_iqr, iqr_floor)
```

2. В том же блоке `dynamic_data` сохранить preclip-границы (для runtime и диагностики), например:
```python
entry["preclip_low"] = float(np.quantile(arr, 0.01))
entry["preclip_high"] = float(np.quantile(arr, 0.99))
```
(для `delta_imb` это особенно важно).

3. В [normalization.py](/D:/MAX/PYTHON/NEURAL-BOTS/neirobot-lit/python_lab/src/normalization.py) в `transform_dynamic()` убрать `scale_multiplier` из dynamic-ветки.
Для dynamic должно быть:
```python
scale = iqr + self.eps
x = (data - median) / scale
```
Иначе `scale_multiplier=1.5` дополнительно раздувает значения и поднимает saturation.

4. В [dataset.py](/D:/MAX/PYTHON/NEURAL-BOTS/neirobot-lit/python_lab/src/dataset.py) в `_apply_dynamic_transform()` применить preclip из `dynamic_params` до `transform_dynamic()`:
```python
sym = symlog_transform(raw)
p = self.normalizer.dynamic_params.get(channel_name, {})
if "preclip_low" in p and "preclip_high" in p:
    sym = torch.clamp(sym, p["preclip_low"], p["preclip_high"])
normed = self.normalizer.transform_dynamic(sym, channel_name)
clipped = torch.clamp(normed, -4.0, 4.0)
```

5. В [train_data.py](/D:/MAX/PYTHON/NEURAL-BOTS/neirobot-lit/python_lab/src/train_data.py) в `_log_dynamic_train_diagnostics()` считать метрики точно тем же путём, что runtime:
- preclip `sym_arr` (если есть `preclip_low/high`),
- потом нормализация,
- потом saturation относительно `clip_limit`.
Иначе guard проверяет не тот pipeline.

6. После этого повторный прогон: если всё ок, `delta_imb saturation` должен уйти ниже 10%.  
Если всё ещё чуть выше, последний точечный тюнинг: для `delta_imb` повысить fallback-порог с `0.35*alt_iqr` до `0.5*alt_iqr`.



==========

## Критерий завершения задачи

Задача считается завершенной, когда одновременно выполнены все условия:

- `OFI`, `DeltaImb`, `DeltaSpread` строятся из одного согласованного event-time контракта;
- fit normalizer для dynamic-каналов больше не зависит от `arr[:,150]`, `arr[:,450]`, `arr[:,500]`;
- train split логирует агрегированную диагностику saturation до старта обучения;
- пайплайн останавливает запуск при заведомо сломанном scale;
- повторный запуск показывает, что проблема poor metrics действительно была связана с dynamic feature contract, а не с LiT-моделью как таковой.

## Внешние ориентиры

- Cont, Kukanov, Stoikov: OFI как event-driven сигнал дисбаланса книги заявок  
  https://www.smallake.kr/wp-content/uploads/2016/05/Kukanov.pdf

- DeepLOB: динамическая нормализация LOB-данных и важность корректного preprocessing  
  https://ora.ox.ac.uk/objects/uuid:4411af59-2657-4e3e-8ee2-81032c37671c

- Официальная документация `RobustScaler`, чтобы не путать устойчивое масштабирование с решением проблемы тяжелых хвостов само по себе  
  https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html