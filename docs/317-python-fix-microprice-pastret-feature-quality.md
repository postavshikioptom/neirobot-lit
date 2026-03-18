# Задача 317: Исправление качества входных фичей MicropriceDev и PastRet - ЗАВЕРШЕНО

**Категория:** Обучение Python
**Статус:** ПЛАН
**Дата:** 18.мар.2026

---

## Проблема

Directional Accuracy модели крайне низкий: DA=5.9% при пороге 0.1%, DA=35.6% при пороге 0.01%. Две причины:

1. **MicropriceDev (канал 0):** Формула `(microprice - mid) / mid` даёт значения ~0.0015 с STD=0.0001. После clamp [-5,5] сигнал практически равен нулю. Кроме того, канал 0 пропущен из нормализации (`channel_idx == 0` skip).
2. **PastRet (канал 5):** Три горизонта (10, 50, 100 тиков) усредняются в одно скалярное значение, теряя информацию о мульти-таймфреймовом моментуме.

---

## Подзадача 1: Исправить формулу MicropriceDev

**Файл:** `python_lab/src/dataset.py`

**Метод:** `_calculate_6_channels_raw()`, строки 1323-1328

**Было:**
```python
microprice = (bid_p_0 * ask_v_0 + ask_p_0 * bid_v_0) / (ask_v_0 + bid_v_0 + eps)
microprice_dev = microprice  # уже относительное отклонение (mid = 0)
price_ch_raw = microprice_dev.unsqueeze(-1).expand(-1, 50)
```

**Стать:**
```python
microprice = (bid_p_0 * ask_v_0 + ask_p_0 * bid_v_0) / (ask_v_0 + bid_v_0 + eps)
spread_width = ask_p_0 - bid_p_0  # ширина спреда в относительных ценах
microprice_dev = microprice / (spread_width + eps)  # нормируем на ширину спреда
price_ch_raw = microprice_dev.unsqueeze(-1).expand(-1, 50)
```

**Обоснование:** Деление на spread_width переводит значения в единицы "спредных ширин" с диапазоном ~[-0.5, 0.5], что даёт Conv1d значимый сигнал.

**Верификация:**
- Запустить `_calculate_6_channels_raw()` на тестовых данных
- Проверить что `price_ch_raw` имеет std > 0.01 (а не 0.0001)
- Проверить что значения лежат в разумном диапазоне (не NaN, не inf)

---

## Подзадача 2: Убрать skip нормализации для канала 0

**Файл:** `python_lab/src/dataset.py`

**Метод:** `normalize_channel()`, строка 1259

**Было:**
```python
if self.normalizer is None or channel_idx == 0:
    return channel_data
```

**Стать:**
```python
if self.normalizer is None:
    return channel_data
```

Также обновить докстринг метода (строки 1255-1258), убрать упоминание что channel 0 пропускается.

**Верификация:**
- Проверить что `normalize_channel(p_raw, channel_idx=0)` теперь вызывает код zscore/robust нормализации
- Проверить что нормализованные значения имеют mean~0, std~1

---

## Подзадача 3: Расширить past_ret_cache до 3 лагов

**Файл:** `python_lab/src/dataset.py`

**Метод:** `_init_memory_mode()`, строки 839-844

**Было:**
```python
log_p = np.log(np.maximum(raw_prices, 1e-9))
self.past_ret_cache = np.zeros(len(log_p), dtype=np.float32)
lag = 100
if len(log_p) > lag:
    self.past_ret_cache[lag:] = (log_p[lag:] - log_p[:-lag]).astype(np.float32)
```

**Стать:**
```python
log_p = np.log(np.maximum(raw_prices, 1e-9))
self.past_ret_cache = {}  # словарь: lag -> массив
for lag in [10, 50, 100]:
    ret = np.zeros(len(log_p), dtype=np.float32)
    if len(log_p) > lag:
        ret[lag:] = (log_p[lag:] - log_p[:-lag]).astype(np.float32)
    self.past_ret_cache[lag] = ret
```

**Верификация:**
- Проверить `self.past_ret_cache.keys() == {10, 50, 100}`
- Проверить что массивы имеют длину `len(log_p)`
- Проверить что для индекса > 100 значения трёх лагов различаются (ret_10 > ret_50 > ret_100 по абсолютной величине)

---

## Подзадача 4: Расширить _calculate_6_channels_raw до 9 каналов

**Файл:** `python_lab/src/dataset.py`

**Метод:** `_calculate_6_channels_raw()`, строки 1349-1376

**Было (1 канал PastRet):**
```python
# ch[5]: Past Returns — среднее из лог-возвратов на лагах [10, 50, 100]
if pr_raw is None:
    ...
    r_val = (ret_10 + ret_50 + ret_100) / 3.0
    pr_ch_raw = r_val.unsqueeze(0).unsqueeze(-1).expand(seq_len, 50)
else:
    pr_ch_raw = pr_raw.unsqueeze(-1).repeat(1, 50) if pr_raw.ndim == 1 else pr_raw
```

**Стать (3 отдельных канала):**
```python
# ch[5]: Ret_10 (short-term momentum)
if pr_raw is None:
    seq_len = ask_p.shape[0]
    mid_approx = ask_p[:, 0] + 1.0
    if seq_len >= 10:
        ret_10 = torch.log(torch.clamp(mid_approx[-1], min=eps)) - \
                 torch.log(torch.clamp(mid_approx[-10], min=eps))
    else:
        ret_10 = torch.tensor(0.0, device=ask_p.device)
    ret_10_ch_raw = ret_10.unsqueeze(0).unsqueeze(-1).expand(seq_len, 50)
else:
    # pr_raw приходит словарь {10: ..., 50: ..., 100: ...} или массив (seq_len, 3)
    ret_10_ch_raw = pr_raw[:, 0].unsqueeze(-1).expand(-1, 50) if pr_raw.ndim > 1 else pr_raw.unsqueeze(-1).expand(seq_len, 50)

# ch[6]: Ret_50 (medium-term momentum)
if pr_raw is None:
    if seq_len >= 50:
        ret_50 = torch.log(torch.clamp(mid_approx[-1], min=eps)) - \
                 torch.log(torch.clamp(mid_approx[-50], min=eps))
    else:
        ret_50 = torch.tensor(0.0, device=ask_p.device)
    ret_50_ch_raw = ret_50.unsqueeze(0).unsqueeze(-1).expand(seq_len, 50)
else:
    ret_50_ch_raw = pr_raw[:, 1].unsqueeze(-1).expand(-1, 50)

# ch[7]: Ret_100 (long-term trend)
if pr_raw is None:
    if seq_len >= 100:
        ret_100 = torch.log(torch.clamp(mid_approx[-1], min=eps)) - \
                  torch.log(torch.clamp(mid_approx[-100], min=eps))
    else:
        ret_100 = torch.tensor(0.0, device=ask_p.device)
    ret_100_ch_raw = ret_100.unsqueeze(0).unsqueeze(-1).expand(seq_len, 50)
else:
    ret_100_ch_raw = pr_raw[:, 2].unsqueeze(-1).expand(-1, 50)
```

**Изменить возвращаемое значение** с 7 на 9 тензоров:
```python
return price_ch_raw, vol_ch_raw, imb_ch_raw, ofi_raw, vib_ch_raw, ret_10_ch_raw, ret_50_ch_raw, ret_100_ch_raw, spread
```

**Обновить докстринг** (строки 1298-1309) — описать 9 каналов.

**Верификация:**
- Проверить что возвращается 9 тензоров формы (seq_len, 50)
- Проверить что ret_10, ret_50, ret_100 имеют разные значения (не идентичны)

---

## Подзадача 5: Обновить _process_sample для 9 каналов

**Файл:** `python_lab/src/dataset.py`

**Метод:** `_process_sample()`, строки 1458-1516

### 5a. Подготовка pr_raw как словаря/массива с 3 лагами

**Было (строки 1458-1464):**
```python
pr_raw = None
if idx is not None and hasattr(self, 'past_ret_cache') and self.past_ret_cache is not None:
    r_seq = self.past_ret_cache[idx : idx + self.seq_len]
    pr_raw = torch.from_numpy((r_seq[:, -1] if r_seq.ndim > 1 else r_seq).copy()).float()
elif self.past_ret_indices:
    pr_raw = torch.from_numpy(x_raw[:, self.past_ret_indices[-1]].copy()).float()
```

**Стать:**
```python
pr_raw = None
if idx is not None and hasattr(self, 'past_ret_cache') and self.past_ret_cache is not None:
    # past_ret_cache теперь словарь {10: arr, 50: arr, 100: arr}
    r10 = self.past_ret_cache[10][idx : idx + self.seq_len]
    r50 = self.past_ret_cache[50][idx : idx + self.seq_len]
    r100 = self.past_ret_cache[100][idx : idx + self.seq_len]
    pr_raw = np.stack([r10, r50, r100], axis=1)  # (seq_len, 3)
    pr_raw = torch.from_numpy(pr_raw).float()
elif self.past_ret_indices:
    # fallback для non-memory режима — берём первый доступный лаг
    pr_raw = torch.from_numpy(x_raw[:, self.past_ret_indices[0]].copy()).float().unsqueeze(-1)
    # pad до 3 колонок нулями
    pr_raw = torch.cat([pr_raw, torch.zeros_like(pr_raw), torch.zeros_like(pr_raw)], dim=-1)
```

### 5b. Распаковка 9 каналов

**Было (строка 1466-1467):**
```python
p_raw, v_raw, i_raw, o_raw, vi_raw, pr_raw, sp_raw = self._calculate_6_channels_raw(...)
```

**Стать:**
```python
p_raw, v_raw, i_raw, o_raw, vi_raw, ret10_raw, ret50_raw, ret100_raw, sp_raw = self._calculate_6_channels_raw(...)
```

### 5c. Нормализация и сборка 9 каналов

**Было (строки 1471-1480):**
```python
price_ch = self.normalize_channel(p_raw, channel_idx=0)
vol_ch = self.normalize_channel(v_raw, channel_idx=1)
imb_ch = self.normalize_channel(i_raw, channel_idx=2)
ofi_ch = self.normalize_channel(o_raw, channel_idx=3)
vib_ch = self.normalize_channel(vi_raw, channel_idx=4)
pr_ch = self.normalize_channel(pr_raw, channel_idx=5)
spread_ch = self.normalize_channel(sp_raw, channel_idx=6)

x_final = torch.stack([price_ch, vol_ch, imb_ch, ofi_ch, vib_ch, pr_ch, spread_ch], dim=1)
```

**Стать:**
```python
price_ch = self.normalize_channel(p_raw, channel_idx=0)
vol_ch = self.normalize_channel(v_raw, channel_idx=1)
imb_ch = self.normalize_channel(i_raw, channel_idx=2)
ofi_ch = self.normalize_channel(o_raw, channel_idx=3)
vib_ch = self.normalize_channel(vi_raw, channel_idx=4)
ret10_ch = self.normalize_channel(ret10_raw, channel_idx=5)
ret50_ch = self.normalize_channel(ret50_raw, channel_idx=6)
ret100_ch = self.normalize_channel(ret100_raw, channel_idx=7)
spread_ch = self.normalize_channel(sp_raw, channel_idx=8)

x_final = torch.stack([price_ch, vol_ch, imb_ch, ofi_ch, vib_ch, ret10_ch, ret50_ch, ret100_ch, spread_ch], dim=1)
```

### 5d. Обновить диагностику

**Было (строки 1486-1500):**
```python
channels_names = ["MicropriceDev", "Vol", "Imb", "OFI", "VIB", "PastRet", "Spread"]
```

**Стать:**
```python
channels_names = ["MicropriceDev", "Vol", "Imb", "OFI", "VIB", "Ret_10", "Ret_50", "Ret_100", "Spread"]
```

**Верификация:**
- Проверить что `x_final.shape == (seq_len, 9, 50)`
- Проверить что все 9 каналов имеют ненулевой std

---

## Подзадача 6: Обновить _compute_channels_for_normalization для 9 каналов

**Файл:** `python_lab/src/dataset.py`

**Метод:** `_compute_channels_for_normalization()`, строки 1412-1429

### 6a. Обновить чтение past_ret_cache

**Было (строки 1412-1415):**
```python
pr_raw = None
if hasattr(self, 'past_ret_cache') and self.past_ret_cache is not None:
    pr_val = self.past_ret_cache[data]
    pr_raw = torch.from_numpy(pr_val[:, -1] if pr_val.ndim > 1 else pr_val).float()
```

**Стать:**
```python
pr_raw = None
if hasattr(self, 'past_ret_cache') and self.past_ret_cache is not None:
    r10 = self.past_ret_cache[10][data]
    r50 = self.past_ret_cache[50][data]
    r100 = self.past_ret_cache[100][data]
    pr_raw = np.stack([r10, r50, r100], axis=1)  # (N, 3)
    pr_raw = torch.from_numpy(pr_raw).float()
```

### 6b. Обновить return

**Было (строки 1425-1429):**
```python
p_raw, v_raw, i_raw, o_raw, vi_raw, pr_raw, sp_raw = self._calculate_6_channels_raw(...)
channels = torch.cat([p_raw, v_raw, i_raw, o_raw, vi_raw, pr_raw, sp_raw], dim=1).numpy()
return pl.DataFrame(channels, schema=[f"feat_{i}" for i in range(350)])
```

**Стать:**
```python
p_raw, v_raw, i_raw, o_raw, vi_raw, ret10_raw, ret50_raw, ret100_raw, sp_raw = self._calculate_6_channels_raw(...)
channels = torch.cat([p_raw, v_raw, i_raw, o_raw, vi_raw, ret10_raw, ret50_raw, ret100_raw, sp_raw], dim=1).numpy()
return pl.DataFrame(channels, schema=[f"feat_{i}" for i in range(450)])
```

**Верификация:**
- Проверить что возвращается DataFrame с 450 колонками (9 * 50)
- Проверить что нормализатор корректно обучается на 9 каналах

---

## Подзадача 7: Обновить in_channels в модели и train.py

### 7a. LiTConfig

**Файл:** `python_lab/src/lit_model.py`, строка 21

**Было:**
```python
in_channels: int = 7  # Задача 316: MicropriceDev, Vol, Imb, OFI, VIB, PastRet, Spread
```

**Стать:**
```python
in_channels: int = 9  # Задача 317: MicropriceDev, Vol, Imb, OFI, VIB, Ret_10, Ret_50, Ret_100, Spread
```

### 7b. train.py — hardcoded in_channels

**Файл:** `python_lab/src/train.py`, строка 1551

**Было:**
```python
in_channels = 7
print(f"Total input channels: {in_channels} (MicropriceDev, Vol, Imb, OFI, VIB, PastRet, Spread)")
```

**Стать:**
```python
in_channels = 9
print(f"Total input channels: {in_channels} (MicropriceDev, Vol, Imb, OFI, VIB, Ret_10, Ret_50, Ret_100, Spread)")
```

### 7c. Dummy input для warmup

**Файл:** `python_lab/src/train.py`, строка 2998 — здесь уже используется переменная `in_channels`, поэтому менять не нужно. Но убедиться что она = 9 после изменения выше.

**Верификация:**
- Проверить что `LiTConfig().in_channels == 9`
- Проверить что модель принимает вход `(batch, seq_len, 9, 50)` без ошибок

---

## Подзадача 8: Финальная верификация и запуск обучения

1. **Shape check:** Запустить один батч через dataset и модель:
   ```python
   # Ожидается: x_final.shape == (100, 9, 50)
   # Ожидается: model(dummy_9ch_input) без ошибок
   ```

2. **Статистика каналов:** Запустить `_process_sample` на нескольких индексах и убедиться:
   - Channel 0 (MicropriceDev): std > 0.01 (не ~0.0001)
   - Channel 5, 6, 7 (Ret_10, Ret_50, Ret_100): значения различаются между каналами
   - Все каналы после нормализации: mean ~ 0, std ~ 1

3. **Запуск обучения:**
   ```bash
   cd python_lab/src
   python train.py --symbol BTCUSDT --data_mode memory --epochs 30 --threshold 0.0005
   ```

4. **Сравнение метрик с предыдущим запуском (Задача 316):**
   - Directional Accuracy должен значительно вырасти
   - Ожидаемый DA при 0.1% threshold: >20% (с текущих 5.9%)
   - Проверить MCC, классовое распределение предсказаний

---

## Сводка изменений

| Файл | Строка | Изменение |
|------|--------|-----------|
| `dataset.py` | 1323-1328 | MicropriceDev /= spread_width |
| `dataset.py` | 1259 | Убрать `or channel_idx == 0` |
| `dataset.py` | 839-844 | past_ret_cache -> dict с 3 лагами |
| `dataset.py` | 1349-1376 | 3 отдельных канала возвратов |
| `dataset.py` | 1386 | return 9 тензоров |
| `dataset.py` | 1458-1480 | _process_sample -> 9 каналов |
| `dataset.py` | 1486-1500 | Обновить диагностику |
| `dataset.py` | 1412-1429 | _compute_channels_for_norm -> 9 каналов |
| `lit_model.py` | 21 | in_channels = 9 |
| `train.py` | 1551-1554 | in_channels = 9 |

---

## Порядок выполнения

Выполнять строго по порядку, каждую подзадачу коммитить отдельно:

1. Подзадача 1 -> коммит "317-1: Fix MicropriceDev formula"
2. Подзадача 2 -> коммит "317-2: Remove channel 0 normalize skip"
3. Подзадача 3 -> коммит "317-3: Expand past_ret_cache to 3 lags"
4. Подзадача 4 -> коммит "317-4: Expand channels to 9 in _calculate_6_channels_raw"
5. Подзадача 5 -> коммит "317-5: Update _process_sample for 9 channels"
6. Подзадача 6 -> коммит "317-6: Update _compute_channels_for_normalization"
7. Подзадача 7 -> коммит "317-7: Update in_channels to 9"
8. Подзадача 8 -> запуск обучения, сравнение метрик


====================
# АНАЛИЗ ПРОБЛЕМЫ:
Диагноз: Почему плохие показатели?

  После анализа кода и логов обнаружены две корневые причины:

  ---
  Причина 1: Канал MicropriceDev (Channel 0) — практические нули

  В _calculate_6_channels_raw() строка 1327:
  microprice_dev = microprice  # уже относительное отклонение (mid = 0)
  Формула (microprice - mid) / mid даёт значения ~0.0015 с STD=0.0001. После clamp [-5,5] канал выглядит так:
  min=0.0013, max=0.0017, std=0.0001.

  Это означает, что Conv1d в LOBPatching получает практически пустой сигнал от канала 0. Кроме того, канал 0 пропущен из
   нормализации (строка 1259: if ... or channel_idx == 0: return).

  Fix: (microprice - mid) / spread_width → range ~[-0.5, 0.5] в единицах спредных ширин. Это стандартный подход из
  литературы (Cont et al. 2014).

  ---
  Причина 2: PastRet (Channel 5) — усреднение убирает информацию

  В _calculate_6_channels_raw() строки 1356-1375: три разных лага (10, 50, 100) вычисляются, но потом усредняются в одно
   число: r_val = (ret_10 + ret_50 + ret_100) / 3.0. Это уничтожает multi-scale информацию. DeepLOB/TransLOB/TLOB
  используют отдельные каналы для каждого лага.

  Fix: Заменить 1 канал PastRet на 3 отдельных: Ret_10, Ret_50, Ret_100.

  ---
  О Threshold: он работает КОРРЕКТНО

  Ваше наблюдение насчёт threshold неверно толкуется. В labels.py строка 92-94:
  .when(col(return_col) > threshold_expr).then(1)   # Up
  .when(col(return_col) < -threshold_expr).then(2)   # Down
  .otherwise(0)                                       # Flat
  Чем выше threshold → тем больше Flat, как и должно быть. Логирование в output.txt показывает предсказания модели, а не
   распределение меток. При threshold=0.1% модель предсказывает Up часто, но почти всегда ошибается (DA=5.9%), потому
  что не может выучить directional signal из-за плохих фичей (Причины 1+2).

  ---
  План: Задача 317 — Исправление качества входных фичей

  Одна задача: улучшить информативность каналов MicropriceDev и PastRet.

  Подзадача 1: Исправить формулу MicropriceDev

  - Файл: python_lab/src/dataset.py, _calculate_6_channels_raw(), строка 1327
  - Было: microprice_dev = microprice (относительно mid, ~0.0015)
  - Стало: microprice_dev = microprice / (spread_width + eps) (относительно спреда, ~0.25)
  - Верификация: STD канала 0 > 0.01 (не 0.0001)

  Подзадача 2: Убрать skip нормализации для канала 0

  - Файл: python_lab/src/dataset.py, normalize_channel(), строка 1259
  - Было: if ... or channel_idx == 0: return channel_data
  - Стало: if ...: return channel_data (убрать or channel_idx == 0)
  - Верификация: Канал 0 получает z-score/robust нормализацию

  Подзадача 3: Расширить past_ret_cache до 3 лагов

  - Файл: python_lab/src/dataset.py, _init_memory_mode(), строки 839-844
  - Было: Один массив self.past_ret_cache с лагом 100
  - Стало: Словарь self.past_ret_cache = {10: arr, 50: arr, 100: arr}
  - Верификация: Значения трёх лагов различаются для одного индекса

  Подзадача 4: Расширить _calculate_6_channels_raw до 9 каналов

  - Файл: python_lab/src/dataset.py, _calculate_6_channels_raw()
  - Было: 7 каналов (PastRet = 1 усреднённый)
  - Стало: 9 каналов (Ret_10, Ret_50, Ret_100 отдельно)
  - Верификация: Возврат 9 тензоров формы (seq_len, 50)

  Подзадача 5: Обновить _process_sample для 9 каналов

  - Файл: python_lab/src/dataset.py, _process_sample()
  - Обновить чтение past_ret_cache, распаковку каналов, нормализацию, сборку x_final, диагностику
  - Верификация: x_final.shape == (seq_len, 9, 50), все каналы ненулевые

  Подзадача 6: Обновить _compute_channels_for_normalization

  - Файл: python_lab/src/dataset.py
  - Прочитать 3 лага из словаря past_ret_cache, вернуть 450 колонок (9×50)
  - Верификация: Нормализатор обучается на 9 каналах

  Подзадача 7: Обновить in_channels в модели и train.py

  - Файлы: lit_model.py строка 21, train.py строка 1551
  - in_channels = 7 → 9
  - Верификация: Модель принимает вход (batch, seq_len, 9, 50)

  Подзадача 8: Финальная верификация и запуск обучения

  1. Shape check: 9 каналов проходят через dataset → модель без ошибок
  2. Статистика: Channel 0 std > 0.01, Channels 5-7 различаются
  3. Запуск: python train.py --symbol BTCUSDT --epochs 30 --threshold 0.0005
  4. Сравнение: Ожидаемый DA при 0.1%: >20% (с текущих 5.9%)
  
 ===============================
 # ОБЩАЯ ИНФОМРАЦИЯ ДЛЯ ПОЛЬЗОВАТЕЛЯ
 Другой вопрос, видел в рассуждениях в будущем ты хочешь добавить
 трешолд на основание волатильности или уже добавил? И как это вообще можно считать? Я уже делал в другом боте считали
барьеры на основание спреда, волатильности - все это закончилось ужасно. ничего из этого за 5 сек посчитать было
невозможно. И в итоге просто трешолд всегда становился один и тот же автоматический, потому что ниего за 5 сек не могло
 считать никакая волатильность. А спред вообще оказалось, что не значит высокую волатильность. Он может быть на топовых
 монетах всегда низкий, но при этом волатильность высокая. А на мусорных монетах может быть то низкий то очень большой,
 при этом волатильность слабая просто потому что это мусорные монеты и может залетать один два крупных покупателя и
шатать монету. Объясни это ,исспользуй searcher для объяснения. только коротко объясни
 
 ---------
 # ОТВЕТ ПРО ДИНАМИЧЕСКИЙ TRESHOLD НА ОСНОВАНИЕ ВОЛАТИЛЬНОСТИ - ЕГО МЫ НЕ ДОБАВЛЯЕМ
 Я не добавляю адаптивный threshold в план. Статический 0.05% — правильный подход, вы и сами к нему пришли в задаче
  309.

  Почему адаптивные трешолды не работают на 5 сек:

  1. Волатильность на 100 тиков — величина почти постоянная. Rolling_std за 1-10 сек на BTCUSDT ≈ константа. Она
  меняется за минуты/часы, а не за тики. Поэтому rolling_std * K всегда даёт одно и то же — адаптивность деградирует в
  статику.
  2. Спред ≠ волатильность. Спред состоит из двух частей: adverse-selection (пропорционален волатильности) +
  order-processing/inventory (фиксирован, задаётся тик-сайзом биржи). На топовых монетах спред всегда мал (биржа
  жёсткая), но волатильность может быть высокой. На мусорных — спред скачет из-за ликвидности, а волатильность слабая.
  Вы измеряете не то, что нужно.
  3. SNR на 5 сек слишком низкий. Шум (bid-ask bounce, queue jumping) и сигнал (направленное движение) неразличимы на
  этом горизонте (Roll 1984 model). Любой адаптивный порог просто отслеживает уровень шума, а не "опасность рынка".

  Вывод: static threshold per coin, подобранный на ~55-65% Flat — оптимальный подход. Confidence gating при инференсе
  (отсекать слабые сигналы) — вот где адаптивность имеет смысл, а не при разметке.
 
----------
# ЗАЧЕМ НУЖЕН Spread в каналах
Spread в каналах нужен по двум причинам:

  1. Единственная фича, которая измеряет стоимость торговли. Imbalance и OFI говорят модели куда пойдёт цена. Spread
  говорит — стоит ли туда идти. Узкий спред + сильный OFI = сигнал actionable. Широкий спред + сильный OFI = съедят
  слэджем.

  2. Естественный деноминатор для нормализации. Именно spread позволяет переводить абсолютные отклонения цен в
  безразмерные единицы. Ваша же задача 317 это демонстрирует: (microprice - mid) / spread вместо (microprice - mid) /
  mid. Без spread microprice_dev = 0.0015 ± 0.0001 (мёртвый канал), со spread = [-0.5, 0.5] (живой сигнал).

  Spread кодирует uncertainty маркетмейкера — адверс селекшн компонента пропорциональна волатильности, поэтому spread
  это косвенный proxy нестабильности цены, который дополняет чисто объёмные фичи (imbalance, volume).