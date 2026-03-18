# Задача 318: Добавление временной динамики в микроструктурные индикаторы LOB

**Дата создания:** 18.03.2026
**Статус:** ЗАПЛАНИРОВАНО
**Категория:** Обучение Python
**Зависимости:** Задача 317 (9 каналов)

## Цель

Добавить временную динамику в микроструктурные индикаторы LOB, чтобы модель LiT могла различать паттерны изменения, а не видеть статические снимки. Ожидаемый эффект: рост Directional Accuracy с 10.6% к 20%+, рост MCC с ~0 к 0.05-0.1.

## Контекст и проблема

Модель видит 100 тиков (10 сек при 10 тик/сек). За это время:
- Imbalance几乎不变
- Spread几乎不变
- VIB几乎不变
- Текущий OFI = `diff(imbalance)` — это просто разность Imbalance каналов, а не настоящий Order Flow Imbalance

Модель получает 100 почти одинаковых кадров и не может выделить динамические паттерны.

## Подзадачи

### Подзадача 318.1: Сохранить `last_update_id` в пайплайне данных

**Проблема:** `last_update_id` удаляется в `load_multi_symbol_data` (строка 1620-1621 dataset.py). Для расчёта настоящего OFI нужно знать моменты обновления стакана.

**Файлы:**
- `python_lab/src/dataset.py`

**Изменения:**

1. **Функция `load_multi_symbol_data` (строка 1620-1621):**
   - Убрать `"last_update_id"` из списка `meta_cols`
   - Добавить `"last_update_id"` в `rename_map` как `feat_update_id` (строка 1605-1610)
   - Итого: колонка `feat_update_id` попадёт в feature-матрицу

2. **`LOBDataset._setup_feature_indices` (строка 785-793):**
   - Добавить `self.update_id_idx = get_idx("feat_update_id")` для быстрого доступа

3. **`LOBDataset._init_memory_mode` (строка 813-852):**
   - Добавить извлечение `feat_update_id` из DataFrame ДО удаления мета-колонок
   - Сохранить как `self.update_id_raw` (numpy array, int64) для расчёта OFI

**Верификация:**
- `feat_update_id` присутствует в `self.feat_cols`
- `self.update_id_raw` имеет shape (N,) и dtype int64

---

### Подзадача 318.2: Переписать OFI — настоящий Cont-Kukanov-Stoikov Order Flow Imbalance

**Проблема:** Текущий код (строка 1339-1340):
```python
ofi_raw = torch.diff(imb_ch_raw, dim=0, prepend=imb_ch_raw[:1])
```
Это просто разность Imbalance, а не OFI.

**Настоящий OFI:**
OFI = Cumulative net order flow: при каждом обновлении стакана сравниваем объёмы на каждом уровне до и после изменения. Рост объёма на bid-side = positive OFI, рост на ask-side = negative OFI.

Поскольку `last_update_id` меняется при каждом обновлении стакана, мы можем определить "моменты обновления". Но для эффективности на уровне numpy/pandas:

**Файлы:**
- `python_lab/src/dataset.py`

**Изменения:**

1. **Новая функция `compute_ofi_from_lob` (добавить после `compute_depth_imbalance_globally`, ~строка 93):**

```python
def compute_ofi_from_lob(bid_p: np.ndarray, ask_p: np.ndarray,
                         bid_v: np.ndarray, ask_v: np.ndarray,
                         update_ids: np.ndarray) -> np.ndarray:
    """
    Векторизованный расчёт OFI (Order Flow Imbalance) по Cont-Kukanov-Stoikov.

    OFI = cumulative net volume flowing through bid/ask levels.
    При обновлении стакана (change in update_id):
      - Если bid价格上升 (bid_p[0] увеличился) → buyer агрессия → +bid_v[0]
      - Если ask价格下降 (ask_p[0] уменьшился) → seller агрессия → -ask_v[0]
      - Если spread сжимается с обеих сторон → оба объёма добавляются

    Упрощённый векторизованный метод:
    - Определяем change_points через diff(update_ids) > 0
    - Для каждого change_point: OFI += bid_v[0] - ask_v[0]
    - Between change_points: OFI повторяет последнее значение (constant hold)

    Args:
        bid_p, ask_p: (N, 50) best bid/ask prices (relative)
        bid_v, ask_v: (N, 50) raw volumes per level (NOT log1p)
        update_ids: (N,) last_update_id values
    Returns:
        (N,) OFI values, cumulative running sum
    """
    n = len(update_ids)
    ofi = np.zeros(n, dtype=np.float64)

    # Определяем точки обновления стакана
    id_diff = np.diff(update_ids, prepend=update_ids[0])
    is_update = id_diff > 0  # bool mask

    # При обновлении: OFI_delta = sum(bid_v[0:3]) - sum(ask_v[0:3])
    # Берём первые 3 уровня (глубина агрессии)
    depth = 3
    bid_flow = bid_v[:, :depth].sum(axis=1)  # (N,)
    ask_flow = ask_v[:, :depth].sum(axis=1)  # (N,)
    delta = np.where(is_update, bid_flow - ask_flow, 0.0)

    # Кумулятивная сумма
    ofi = np.cumsum(delta).astype(np.float32)

    return ofi
```

2. **Функция `compute_ofi_from_lob_cache` — глобальный precompute (добавить рядом):**

```python
def compute_ofi_from_lob_cache(bid_p: np.ndarray, ask_p: np.ndarray,
                                bid_v: np.ndarray, ask_v: np.ndarray,
                                update_ids: np.ndarray) -> np.ndarray:
    """
    Глобальная версия compute_ofi_from_lob для precompute при инициализации.
    Использует сырые (не log) объёмы.

    Принимает LOG1P объёмы и преобразует их: exp(x) - 1
    Принимает relative prices напрямую (feat_ask_p_i, feat_bid_p_i)
    """
    # Восстанавливаем сырые объёмы из log1p
    bid_v_raw = np.exp(np.clip(bid_v, None, 20.0)) - 1.0
    ask_v_raw = np.exp(np.clip(ask_v, None, 20.0)) - 1.0

    return compute_ofi_from_lob(ask_p, bid_p, ask_v_raw, bid_v_raw, update_ids)
```

3. **`_init_memory_mode` (строка 837-850) — добавить precompute OFI cache:**

После расчёта `self.vib_cache` и `self.past_ret_cache`, добавить:
```python
# 4. Расчёт OFI из сырых данных (Задача 318)
bid_p_matrix = df.select([f"feat_bid_p_{i}" for i in range(self.n_levels)]).to_numpy().astype(np.float64)
ask_p_matrix = df.select([f"feat_ask_p_{i}" for i in range(self.n_levels)]).to_numpy().astype(np.float64)

self.ofi_cache = compute_ofi_from_lob_cache(
    bid_p_matrix, ask_p_matrix, bid_v_matrix, ask_v_matrix,
    self.update_id_raw
)
print(f"[DEBUG] ofi raw: min={self.ofi_cache.min():.6f}, max={self.ofi_cache.max():.6f}")
```

4. **`_calculate_6_channels_raw` (строка 1339-1340) — заменить OFI:**

Заменить:
```python
# БЫЛО:
ofi_raw = torch.diff(imb_ch_raw, dim=0, prepend=imb_ch_raw[:1])
```
На:
```python
# СТАЛО: OFI из precomputed cache (Задача 318)
# Если ofi_cache доступен, используем его; иначе fallback на diff(imbalance)
if ofi_precomputed is not None:
    ofi_raw = ofi_precomputed.unsqueeze(-1).expand(-1, 50)
else:
    ofi_raw = torch.diff(imb_ch_raw, dim=0, prepend=imb_ch_raw[:1])
```

5. **`_calculate_6_channels_raw` — добавить параметр `ofi_precomputed`:**

Сигнатура функции (строка 1293):
```python
def _calculate_6_channels_raw(self, ask_p, ask_v, bid_p, bid_v, vib_raw=None, pr_raw=None, ofi_precomputed=None):
```

6. **`_process_sample` (строка 1463-1483) — передать OFI cache:**

После подготовки `vib_raw` (строка 1463-1464), добавить:
```python
# Подготовка OFI из precomputed cache (Задача 318)
ofi_precomp = None
if idx is not None and hasattr(self, 'ofi_cache') and self.ofi_cache is not None:
    ofi_precomp = torch.from_numpy(self.ofi_cache[idx : idx + self.seq_len].copy()).float()
```

И передать в вызов (строка 1483):
```python
p_raw, v_raw, i_raw, o_raw, vi_raw, ret10_raw, ret50_raw, ret100_raw, sp_raw = self._calculate_6_channels_raw(
    ask_p, ask_v, bid_p, bid_v, vib_raw, pr_raw, ofi_precomputed=ofi_precomp
)
```

7. **`_compute_channels_for_normalization` (строка 1395-1439) — передать OFI:**

В строке ~1426 добавить извлечение OFI cache для нормализации:
```python
ofi_precomp = None
if hasattr(self, 'ofi_cache') and self.ofi_cache is not None:
    if isinstance(data, list):
        ofi_precomp = torch.from_numpy(self.ofi_cache[data]).float()
    else:
        ofi_precomp = torch.from_numpy(self.ofi_cache).float()
```

И передать в `_calculate_6_channels_raw` (строка ~1436).

**Верификация:**
- OFI не равен `diff(Imbalance)` — проверить корреляцию < 0.5
- OFI кумулятивный (растёт/падает монотонно между обновлениями стакана)
- Нормализация OFI применяется корректно

---

### Подзадача 318.3: Исправить MicropriceDev — делить на spread_width/2

**Проблема:** Текущий код (строка 1329):
```python
microprice_dev = microprice / (spread_width + eps)
```
Даёт диапазон [-0.5, 0.5] вместо [-1, 1], уменьшая амплитуду сигнала в 2 раза.

**Файл:** `python_lab/src/dataset.py`

**Изменения:**

1. **Строка 1329:**
```python
# БЫЛО:
microprice_dev = microprice / (spread_width + eps)

# СТАЛО:
microprice_dev = microprice / (spread_width / 2.0 + eps)
```

**Верификация:**
- После нормализации: проверить что channel 0 имеет std > текущего (сигнал усилен)
- Убедиться что модель не получает NaN/Inf

---

### Подзадача 318.4: Добавить temporal derivative features (ΔImbalance, ΔSpread, Cumulative OFI, Imbalance Acceleration)

**Цель:** Заменить дублирование каналов (Imbalance, Spread, OFI, VIB) на их производные, чтобы модель видела динамику ВНУТРИ окна 100 тиков.

**Подход:** Добавить 4 новых канала (всего 13 вместо 9). Или заменить 4 существующих на их комбинации статика+динамика.

**Рекомендуемый подход — РАСШИРЕНИЕ до 13 каналов:**

Новые каналы:
- ch[9] = ΔImb (rate of change of imbalance, т.е. diff по времени)
- ch[10] = ΔSpread (spread velocity, diff по времени)
- ch[11] = CumOFI (cumulative OFI внутри окна — сброс на начало каждого окна)
- ch[12] = ImbAccel (2nd derivative of imbalance — acceleration)

Это не требует замены существующих каналов, только расширение `in_channels`.

**Файлы:**
- `python_lab/src/dataset.py`
- `python_lab/src/lit_model.py` (LiTConfig: in_channels=9 → 13)
- `python_lab/src/train.py` (in_channels=9 → 13)

**Изменения:**

1. **`_calculate_6_channels_raw` — расширить до 13 каналов:**

Добавить после расчёта основных каналов (после строки 1391):

```python
# ===== НОВЫЕ TEMPORAL DERIVATIVE КАНАЛЫ (Задача 318.4) =====

# ch[9]: ΔImb — скорость изменения Imbalance (first derivative)
# diff по оси времени (dim=0), prepend нулями
delta_imb = torch.diff(imb_ch_raw[:, 0], dim=0, prepend=torch.zeros(1))
delta_imb_ch = delta_imb.unsqueeze(-1).expand(-1, 50)  # (seq_len, 50)

# ch[10]: ΔSpread — скорость изменения спреда
spread_1d = (ask_p_0 - bid_p_0)  # (seq_len,)
delta_spread = torch.diff(spread_1d, dim=0, prepend=torch.zeros(1))
delta_spread_ch = delta_spread.unsqueeze(-1).expand(-1, 50)

# ch[11]: CumOFI — кумулятивный OFI внутри окна (сброс на 0 в начале каждого окна)
if ofi_precomputed is not None:
    ofi_window = ofi_precomputed  # (seq_len,)
    # Сдвигаем чтобы OFI[0] = 0 (normalise to window start)
    ofi_cum = ofi_window - ofi_window[0]
else:
    # Fallback: используем кумулятивную сумму OFI
    ofi_cum = torch.cumsum(o_raw[:, 0], dim=0)  # (seq_len,)
    ofi_cum = ofi_cum - ofi_cum[0]
cumofi_ch = ofi_cum.unsqueeze(-1).expand(-1, 50)

# ch[12]: ImbAccel — ускорение Imbalance (second derivative)
accel_imb = torch.diff(delta_imb, dim=0, prepend=torch.zeros(1))
accel_imb_ch = accel_imb.unsqueeze(-1).expand(-1, 50)
```

И добавить в return:
```python
return price_ch_raw, vol_ch_raw, imb_ch_raw, ofi_raw, vib_ch_raw, \
       ret_10_ch_raw, ret_50_ch_raw, ret_100_ch_raw, spread, \
       delta_imb_ch, delta_spread_ch, cumofi_ch, accel_imb_ch
```

2. **`_process_sample` — обновить распаковку и нормализацию (строка 1483-1497):**

```python
# Распаковка 13 каналов
p_raw, v_raw, i_raw, o_raw, vi_raw, ret10_raw, ret50_raw, ret100_raw, sp_raw, \
    di_raw, ds_raw, co_raw, ai_raw = self._calculate_6_channels_raw(
        ask_p, ask_v, bid_p, bid_v, vib_raw, pr_raw, ofi_precomputed=ofi_precomp
    )

# Нормализация (13 каналов)
price_ch = self.normalize_channel(p_raw, channel_idx=0)
vol_ch = self.normalize_channel(v_raw, channel_idx=1)
imb_ch = self.normalize_channel(i_raw, channel_idx=2)
ofi_ch = self.normalize_channel(o_raw, channel_idx=3)
vib_ch = self.normalize_channel(vi_raw, channel_idx=4)
ret10_ch = self.normalize_channel(ret10_raw, channel_idx=5)
ret50_ch = self.normalize_channel(ret50_raw, channel_idx=6)
ret100_ch = self.normalize_channel(ret100_raw, channel_idx=7)
spread_ch = self.normalize_channel(sp_raw, channel_idx=8)
delta_imb_ch = self.normalize_channel(di_raw, channel_idx=9)
delta_spread_ch = self.normalize_channel(ds_raw, channel_idx=10)
cumofi_ch = self.normalize_channel(co_raw, channel_idx=11)
accel_imb_ch = self.normalize_channel(ai_raw, channel_idx=12)

x_final = torch.stack([
    price_ch, vol_ch, imb_ch, ofi_ch, vib_ch,
    ret10_ch, ret50_ch, ret100_ch, spread_ch,
    delta_imb_ch, delta_spread_ch, cumofi_ch, accel_imb_ch
], dim=1)  # (seq_len, 13, 50)
```

3. **`_compute_channels_for_normalization` (строка 1436-1439) — расширить:**

Распаковать 13 каналов и собрать DataFrame с 13*50 = 650 столбцами:
```python
p_raw, v_raw, i_raw, o_raw, vi_raw, ret10_raw, ret50_raw, ret100_raw, sp_raw, \
    di_raw, ds_raw, co_raw, ai_raw = self._calculate_6_channels_raw(
        ask_p, ask_v, bid_p, bid_v, vib_raw, pr_raw, ofi_precomputed=ofi_precomp
    )
channels = torch.cat([
    p_raw, v_raw, i_raw, o_raw, vi_raw, ret10_raw, ret50_raw, ret100_raw, sp_raw,
    di_raw, ds_raw, co_raw, ai_raw
], dim=1).numpy()
return pl.DataFrame(channels, schema=[f"feat_{i}" for i in range(650)])
```

4. **`lit_model.py` — LiTConfig (строка 21):**

```python
# БЫЛО:
in_channels: int = 9

# СТАЛО:
in_channels: int = 13  # Задача 318: MicropriceDev, Vol, Imb, OFI, VIB, Ret_10, Ret_50, Ret_100, Spread, DeltaImb, DeltaSpread, CumOFI, ImbAccel
```

5. **`train.py` (строка 1551):**

```python
# БЫЛО:
in_channels = 9

# СТАЛО:
in_channels = 13  # Задача 318: +DeltaImb, DeltaSpread, CumOFI, ImbAccel
```

6. **`train.py` — print statement (строка 1554):**

```python
print(f"Total input channels: {in_channels} (MicropriceDev, Vol, Imb, OFI, VIB, Ret_10, Ret_50, Ret_100, Spread, DeltaImb, DeltaSpread, CumOFI, ImbAccel)")
```

7. **`lit_model.py` — main block тест (строка 548):**

Обновить тест с 9 каналами на 13 каналов:
```python
model_13ch = LiTModel(seq_len=100, in_channels=13, activation='silu')
dummy_input_13ch = torch.randn(8, 100, 13, 50)
output_13ch = model_13ch(dummy_input_13ch)
```

**Верификация:**
- `in_channels=13` согласован между `LiTConfig`, `train.py` и actual output
- Тестовый прогон модели с `(batch, 100, 13, 50)` без ошибок
- Нормализация всех 13 каналов корректна (нет NaN)

---

### Подзадача 318.6: Увеличить seq_len с 100 до 200 (2:1 к горизонту)

**Обоснование:**
- Текущее seq_len=100 = horizon=100 → ratio 1:1 (минимальный порог)
- Успешные крипт-модели (Wang 2025) используют seq_len >> horizon (10:1 и выше)
- seq_len=200 даёт модели 20 секунд контекста для определения рыночного режима перед предсказанием на 10 сек вперёд
- 200 тиков × 13 каналов × 50 уровней = 1.3M значений на сэмпл — помещается в память GPU

**Файлы:**
- `python_lab/src/dataset.py` (LOBDataset default seq_len)
- `python_lab/src/lit_model.py` (LiTConfig seq_len, time_pos_emb size)
- `python_lab/src/train.py` (default seq_len=100 → 200)

**Изменения:**

1. **`lit_model.py` — LiTConfig (строка 20):**
```python
# БЫЛО:
seq_len: int = 100

# СТАЛО:
seq_len: int = 200  # Задача 318.6: 2:1 ratio к горизонту (100 тиков)
```

2. **`lit_model.py` — LOBPatching (строка 32):**
```python
# time_pos_emb создаётся динамически по seq_len, менять не нужно — автоматически
# Проверить: self.time_pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model))
```

3. **`train.py` (аргументы CLI):**
```python
# БЫЛО (default в argparse):
parser.add_argument("--seq_len", type=int, default=100)

# СТАЛО:
parser.add_argument("--seq_len", type=int, default=200)
```

4. **Пересоздать нормализатор** — при seq_len=200 датасет выдаёт другие окна, norm_params.json нужно пересчитать (запуск `normalizer.fit()` с новым seq_len).

**Верификация:**
- Модель принимает вход `(batch, 200, 13, 50)` без ошибок
- Нормализатор обучен на 200-tick окнах
- Память GPU < 8GB при batch_size=64

**Ожидаемый эффект:**
- Больше контекста → лучше определение рыночного режима
- Снижение шума в индикаторах (усреднение за более длинное окно)
- MCC может вырасти дополнительно на 0.01-0.03

**Риск:**
- Увеличение размера входа в 2x → увеличение времени обучения на ~30-50%
- Может потребоваться больше эпох для сходимости

---

### Подзадача 318.5: Запуск обучения и сравнение метрик

**Действия:**

1. Запустить обучение на тех же данных что и Задача 317
2. Сравнить метрики:
   - MCC (ожидаемый: рост с ~0 к 0.05-0.1)
   - Directional Accuracy (ожидаемый: рост с 10.6% к 15-20%)
   - Class distribution (ожидаемый: Flat 63% может снизиться до 55-60%)
   - Validation Loss (ожидаемый: снижение на 5-10%)
3. Диагностика: вывести статистики новых каналов (mean, std, min, max) после нормализации

**Файл:** `python_lab/src/train.py` — без изменений, просто запуск

---

## Порядок выполнения

```
318.1 (save update_id)
  → 318.2 (rewrite OFI, depends on 318.1)
  → 318.3 (fix MicropriceDev, independent)
  → 318.4 (temporal derivatives, depends on 318.2)
  → 318.6 (increase seq_len to 200, independent)
  → 318.5 (run training, depends on all above)
```

## Ожидаемый эффект по подзадачам

| Подзадача | Ожидаемый эффект | Риск |
|-----------|-----------------|------|
| 318.1 | last_update_id доступен для OFI | Низкий — просто сохранение колонки |
| 318.2 | OFI реагирует на обновления стакана, а не на diff(imb) | Средний — новый алгоритм OFI может быть неоптимален для crypto |
| 318.3 | MicropriceDev сигнал в 2 раза сильнее | Низкий — математически верно |
| 318.4 | Модель видит velocity/acceleration индикаторов | Средний — 4 новых канала = 44% рост входного размера |
| 318.6 | Больше контекста (20 сек вместо 10), лучшее определение режима | Средний — рост времени обучения на ~30-50% |
| 318.5 | Рост MCC и Directional Accuracy | Зависит от качества предыдущих подзадач |

## Список файлов к изменению

1. `python_lab/src/dataset.py` — основные изменения (318.1-318.4)
2. `python_lab/src/lit_model.py` — in_channels 9→13, seq_len 100→200
3. `python_lab/src/train.py` — in_channels 9→13, seq_len 100→200

===============
# ОБЩАЯ ИНФОМРАЦИЯ О ТОМ, КАКОЙ HORIZON И seq_len в будущем лучше ставить для Скальпинга(у нас не HFT)
Что говорят исследования
                                                                         _   u          e       p  :
  Ratio seq_len к horizon

  Из литературы складывается чёткая картина:

  ┌─────────────────────────┬─────────────┬────────────┬───────────┬───────────────────────────┐
  │         Модель          │   Seq_len   │  Horizon   │   Ratio   │         Результат         │
  ├─────────────────────────┼─────────────┼────────────┼───────────┼───────────────────────────┤
  │ DeepLOB (NASDAQ)        │ 100 тиков   │ 10 тиков   │ 10:1      │ F1 = 83.4% (лучший)       │
  ├─────────────────────────┼─────────────┼────────────┼───────────┼───────────────────────────┤
  │ DeepLOB (NASDAQ)        │ 100         │ 20         │ 5:1       │ F1 = 72.8%                │
  ├─────────────────────────┼─────────────┼────────────┼───────────┼───────────────────────────┤
  │ DeepLOB (NASDAQ)        │ 100         │ 50         │ 2:1       │ F1 = 80.4%                │
  ├─────────────────────────┼─────────────┼────────────┼───────────┼───────────────────────────┤
  │ LiT (Binance, 2025)     │ ~320        │ 0.3-1.0    │ ~3:10     │ F1 ~64-68%                │
  │                         │ patched     │ сек        │           │                           │
  ├─────────────────────────┼─────────────┼────────────┼───────────┼───────────────────────────┤
  │ TLOB (Bitcoin, 2025)    │ 100         │ 10-100     │ 10:1 →    │ Лучше на длинных          │
  │                         │             │            │ 1:1       │ горизонтах                │
  ├─────────────────────────┼─────────────┼────────────┼───────────┼───────────────────────────┤
  │ Jha et al. (Coinbase    │ 100         │ 2 сек      │ ~50:1     │ 71% accuracy              │
  │ BTC)                    │             │            │           │                           │
  └─────────────────────────┴─────────────┴────────────┴───────────┴───────────────────────────┘

  Rule of thumb: Ratio 2:1 до 10:1 — это золотой диапазон. Глубже 10:1 на зашумлённых криптоданных
  прирост обычно не даёт.

  ---
  Почему не нужно брать seq_len=1000 для horizon=100

  Твой интуитивный вывод правильный. Вот почему:

  1. Шум съедает длинный контекст

  Wang (2025, arxiv:2506.05764) показал на данных Bybit, что простые модели с хорошей предобработкой
   (Kalman/Savitzky-Golay фильтры) побеждают сложные нейросети без фильтрации. Для крипты контекст
  >200-300 тиков содержит больше шума, чем сигнала.

  2. Модели видят лучше пространство, чем время

  LiT-статья (Frontiers, 2025) показала: узкие временные патчи (W=4) работают лучше широких (W=16)
  для длинных горизонтов. Критичнее видеть больше уровней стакана, чем больше истории по времени. У
  тебя 50 уровней — это хорошо.

  3. Криптоданные — не акции NASDAQ

  Briola et al. (2025) подтверждает: на зашумлённых данных оптимальная длина контекста короче, чем
  на чистых традиционных рынках. Модели теряют точность при "глубоком" контексте (>200-300 тиков) —
  distant levels = noise features.

  4. Мемкоины имеют эпизодическую ликвидность

  CoinAPI (2025): ликвидность мемкоинов "кластерится и исчезает". Order-book imbalance velocity —
  ключевой сигнал, но окно меньше 300 мс. Алгоритмы, предполагающие стабильный стакан, "ловят
  призраков".

  ---
  Что делать с горизонтом: 10 сек vs 1 сек

  Твоё замечание про практическую пользу 10-секундного горизонта обосновано, но картина nuanced:

  10-секундный горизонт (100 тиков) — плюсы:
  - Можно планировать сделки, есть время на исполнение
  - На mid-cap/альткоинах средняя ликвидность стабильна 5-10 сек
  - Меньше конкуренция с高频 ботами (<10мс реакция)

  1-секундный горизонт (10 тиков) — плюсы:
  - Точнее (F1 выше на 5-10% по DeepLOB)
  - Меньше шума, предсказуемость выше
  - Больше торговых возможностей (в 10x больше сигналов)

  Но: для мемкоинов 1-секундный горизонт — это скальпинг, который проигрывает HFT-ботам с
  латентностью <10мс. Твоя модель inference на GPU с几十мс латентности — ты не победишь их в этом.

  10-секундный горизонт — стратегически вернее для твоего случая:
  - Ты не конкурируешь с HFT
  - Можно открывать/закрывать позиции осмысленно
  - Больше "макро"-сигналов в стакане

  ---
  Рекомендация для neirobot-lit

  ┌────────────┬────────────────┬───────────────────┬──────────────────────────────────────────┐
  │  Параметр  │    Текущее     │   Рекомендация    │               Обоснование                │
  ├────────────┼────────────────┼───────────────────┼──────────────────────────────────────────┤
  │ Horizon    │ 100 тиков (10  │ 100 тиков —       │ Оптимально для твоего профиля (не HFT)   │
  │            │ сек)           │ оставить          │                                          │
  ├────────────┼────────────────┼───────────────────┼──────────────────────────────────────────┤
  │ Seq_len    │ 200 (задача    │ 200-300 —         │ Ratio 2:1 — 3:1, разумный максимум для   │
  │            │ 318.6)         │ оставить          │ крипты                                   │
  ├────────────┼────────────────┼───────────────────┼──────────────────────────────────────────┤
  │ Ratio      │ 2:1            │ 2:1 — 3:1         │ Больше 5:1 на noisy данных мемкоинов не  │
  │            │                │                   │ даст прироста                            │
  ├────────────┼────────────────┼───────────────────┼──────────────────────────────────────────┤
  │ Optuna     │ задача 055     │ Диапазон 100-300  │ Эмпирически найти оптимум для            │
  │ поиск      │                │                   │ конкретного актива                       │
  └────────────┴────────────────┴───────────────────┴──────────────────────────────────────────┘

  Главный lever для качества — не seq_len, а:
  1. Предобработка данных (фильтрация, сглаживание)
  2. Качество индикаторов (OFI из задачи 318 — правильный шаг)
  3. Правильная нормализация
  4. Калибровка вероятностей (ECE)