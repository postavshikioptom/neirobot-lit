# Задача 315: Исправить функции расчета каналов и их нормализации в dataset.py VOL, OFI.

Продолжаю анализ и формирую итоговый план.

Отлично! Вижу что уже были попытки исправить OFI, но основная проблема с двойной нормализацией объемов не была решена. Теперь сформирую финальный план.

Теперь сформирую итоговый план на русском:

---

# ПЛАН ИСПРАВЛЕНИЯ: Задача 315 - Устранение двойной нормализации в пайплайне LOB данных - ЗАВЕРШЕНО

## 🎯 ГЛАВНАЯ ПРОБЛЕМА

**Критическая ошибка "Double Normalization"** в пайплайне обработки данных:

1. **features.py** применяет предобработку:
   - Цены: `(price - mid) / mid` → относительное отклонение
   - Объемы: `log(1 + volume)` → логарифмирование

2. **dataset.py** использует эти УЖЕ преобразованные данные как сырые для расчета каналов:
   - `vol_ch = ask_v + bid_v` → складывает ЛОГАРИФМЫ (математически неверно!)
   - `imbalance = (bid_v - ask_v) / (bid_v + ask_v)` → использует ЛОГАРИФМЫ для дисбаланса

3. **Результат**: Полная потеря сигнала в данных

## 📊 СИМПТОМЫ ПРОБЛЕМЫ (из логов)

```
Channel 1 (Vol): min=5.0000, max=5.0000, mean=5.0000  ← ВСЕ ЗНАЧЕНИЯ ОДИНАКОВЫЕ!
Channel 3 (OFI): все значения = 0.0                   ← ПОТЕРЯ СИГНАЛА!

Hit Rate Up: 6%                                        ← КАТАСТРОФА
Hit Rate Down: 8%                                      ← КАТАСТРОФА
Directional Accuracy: 5.86%                            ← ХУЖЕ СЛУЧАЙНОГО
```

## ✅ РЕШЕНИЕ: Восстановление сырых значений перед расчетом каналов

### ГЛАВНАЯ ЗАДАЧА: Исправить функции расчета каналов в dataset.py

#### Подзадача 1: Исправить расчет Volume Channel (Channel 1)

**Текущий код (НЕПРАВИЛЬНО):**
```python
vol_ch_raw = ask_v + bid_v  # Складывает логарифмы!
```

**Правильный код:**
```python
# Восстанавливаем сырые объемы из логарифмов
ask_v_raw = torch.exp(ask_v) - 1.0  # Обратная операция к log(1+v)
bid_v_raw = torch.exp(bid_v) - 1.0

# Суммируем СЫРЫЕ объемы
vol_sum = ask_v_raw + bid_v_raw

# Применяем логарифм к сумме
vol_ch_raw = torch.log1p(vol_sum)  # log(1 + sum)
```

**Обоснование:** 
- `log(a) + log(b) = log(a*b)` ≠ `log(a+b)`
- Нужно сначала восстановить сырые значения, потом суммировать, потом логарифмировать

---

#### Подзадача 2: Исправить расчет Imbalance Channel (Channel 2)

**Текущий код (НЕПРАВИЛЬНО):**
```python
denom = bid_v + ask_v + 1e-8
imb_ch_raw = (bid_v - ask_v) / denom  # Использует логарифмы!
```

**Правильный код:**
```python
# Восстанавливаем сырые объемы
ask_v_raw = torch.exp(ask_v) - 1.0
bid_v_raw = torch.exp(bid_v) - 1.0

# Вычисляем дисбаланс из СЫРЫХ объемов
denom = bid_v_raw + ask_v_raw + 1e-8
imb_ch_raw = (bid_v_raw - ask_v_raw) / denom
```

**Обоснование:**
- Формула OBI из литературы: `(V_bid - V_ask) / (V_bid + V_ask)`
- Требует СЫРЫЕ объемы, не логарифмы

---

#### Подзадача 3: Исправить расчет OFI Channel (Channel 3)

**Текущий код (НЕПРАВИЛЬНО):**
```python
denom_fb = bid_v + ask_v + 1e-8
imb_fb = (bid_v - ask_v) / denom_fb  # Логарифмы!
ofi_raw = torch.diff(imb_fb, dim=0, prepend=imb_fb[:1])
```

**Правильный код:**
```python
# Восстанавливаем сырые объемы
ask_v_raw = torch.exp(ask_v) - 1.0
bid_v_raw = torch.exp(bid_v) - 1.0

# Вычисляем imbalance из СЫРЫХ объемов
denom_fb = bid_v_raw + ask_v_raw + 1e-8
imb_fb = (bid_v_raw - ask_v_raw) / denom_fb

# Delta OFI - изменение дисбаланса по времени
ofi_raw = torch.diff(imb_fb, dim=0, prepend=imb_fb[:1])
```

---

#### Подзадача 4: Синхронизировать _compute_channels_for_normalization

**Место:** `dataset.py`, функция `_compute_channels_for_normalization`

**Что исправить:**
```python
# БЫЛО (неправильно):
vol_ch = ask_v + bid_v
imb_ch = (bid_v - ask_v) / (bid_v + ask_v + 1e-8)

# ДОЛЖНО БЫТЬ (правильно):
# Восстанавливаем сырые объемы
ask_v_raw = np.exp(ask_v) - 1.0
bid_v_raw = np.exp(bid_v) - 1.0

# Вычисляем каналы из сырых объемов
vol_sum = ask_v_raw + bid_v_raw
vol_ch = np.log1p(vol_sum)

denom = bid_v_raw + ask_v_raw + 1e-8
imb_ch = (bid_v_raw - ask_v_raw) / denom
```

**Обоснование:**
- Нормализатор должен обучаться на тех же данных, что используются в _process_sample
- Иначе статистики (mean, std) будут неправильными

---

#### Подзадача 5: Обновить compute_delta_imbalance

**Место:** `dataset.py`, функция `compute_delta_imbalance`

**Что исправить:**
```python
def compute_delta_imbalance(bid_v: np.ndarray, ask_v: np.ndarray) -> np.ndarray:
    """
    Delta Imbalance из СЫРЫХ объемов (не логарифмов).
    """
    # Восстанавливаем сырые объемы из логарифмов
    bid_v_raw = np.exp(bid_v) - 1.0
    ask_v_raw = np.exp(ask_v) - 1.0
    
    # Вычисляем дисбаланс из сырых объемов
    denom = bid_v_raw + ask_v_raw + 1e-7
    imbalance = (bid_v_raw - ask_v_raw) / denom
    
    # Разница между тиками
    delta_imb = np.diff(imbalance, axis=0, prepend=imbalance[:1])
    
    return delta_imb.astype(np.float32)
```

---

#### Подзадача 6: Добавить диагностику для проверки исправления

**Место:** `dataset.py`, функция `_process_sample`

**Что добавить:**
```python
# После расчета каналов, перед нормализацией
if idx is not None and idx == 0:
    print(f"[ДИАГНОСТИКА] Статистика каналов ДО нормализации:")
    print(f"  Channel 0 (Price): min={price_ch_raw.min():.4f}, max={price_ch_raw.max():.4f}, mean={price_ch_raw.mean():.4f}")
    print(f"  Channel 1 (Vol): min={vol_ch_raw.min():.4f}, max={vol_ch_raw.max():.4f}, mean={vol_ch_raw.mean():.4f}")
    print(f"  Channel 2 (Imb): min={imb_ch_raw.min():.4f}, max={imb_ch_raw.max():.4f}, mean={imb_ch_raw.mean():.4f}")
    print(f"  Channel 3 (OFI): min={ofi_raw.min():.4f}, max={ofi_raw.max():.4f}, mean={ofi_raw.mean():.4f}")
```

**Ожидаемый результат после исправления:**
- Channel 1 (Vol): разнообразные значения, НЕ константа 5.0
- Channel 3 (OFI): значения в диапазоне [-1, 1], НЕ все нули

---

## 📋 ПОРЯДОК ВЫПОЛНЕНИЯ

1. ✅ **Подзадача 1**: Исправить Volume Channel в `_process_sample`
2. ✅ **Подзадача 2**: Исправить Imbalance Channel в `_process_sample`
3. ✅ **Подзадача 3**: Исправить OFI Channel в `_process_sample`
4. ✅ **Подзадача 4**: Синхронизировать `_compute_channels_for_normalization`
5. ✅ **Подзадача 5**: Обновить `compute_delta_imbalance`
6. ✅ **Подзадача 6**: Добавить диагностику

---


## 🔍 ТЕХНИЧЕСКИЕ ДЕТАЛИ

### Математическое обоснование

**Проблема с логарифмами:**
```
log(a) + log(b) = log(a × b)  ≠  log(a + b)
```

**Правильная последовательность:**
1. Восстановить: `v_raw = exp(log_v) - 1`
2. Агрегировать: `sum = v1_raw + v2_raw`
3. Нормализовать: `result = log(1 + sum)`

### Ссылки на литературу

Из найденных статей (TransLOB, DeepLOB):
- Нормализация применяется ПОСЛЕ извлечения признаков
- Order Book Imbalance: `OBI = (V_bid - V_ask) / (V_bid + V_ask)`
- Используются СЫРЫЕ объемы для расчета метрик

---

## ⚠️ ВАЖНЫЕ ЗАМЕЧАНИЯ

1. **НЕ ТРОГАТЬ features.py** - там всё правильно, это предобработка сырых данных
2. **НЕ ТРОГАТЬ normalization.py** - там всё правильно, это финальная нормализация каналов
3. **ИСПРАВЛЯТЬ ТОЛЬКО dataset.py** - восстановление сырых значений перед расчетом каналов
4. **СИНХРОНИЗАЦИЯ** - обе функции (_process_sample и _compute_channels_for_normalization) должны использовать одинаковую логику

---

## 📝 ФАЙЛЫ ДЛЯ ИЗМЕНЕНИЯ

- ✅ `python_lab/src/dataset.py` - основной файл для исправления
  - Функция `_process_sample` (строки ~1450-1580)
  - Функция `_compute_channels_for_normalization` (строки ~1388-1449)
  - Функция `compute_delta_imbalance` (строки ~206-230)

---

Этот план решает ОДНУ конкретную, но критическую проблему - двойную нормализацию, которая полностью уничтожает сигнал в данных. После исправления модель сможет обучаться на правильных данных и показатели должны значительно улучшиться.

========================================
# Задача 315-2: Исправление Volume Channel (константа 5.0)  - ЗАВЕРШЕНО

## 🎯 ПРОБЛЕМА

После исправлений в Задаче 315 каналы имеют неправильную статистику:

```
Channel 1 (Vol): min=5.0000, max=5.0000, mean=5.0000  ← КОНСТАНТА!
Channel 3 (OFI): std=2.8575                            ← Должен быть ~1.0
Channel 2 (Imb): std=0.6738                            ← Должен быть ~1.0
```

## 🔍 КОРНЕВАЯ ПРИЧИНА

**Volume Channel стал константой из-за неправильной агрегации:**

1. В `features.py` объемы логарифмируются: `log(1 + volume)`
2. В `dataset.py` происходило:
   - Восстановление: `exp(v) - 1` → получаем сырые объемы
   - **СУММИРОВАНИЕ 50 уровней**: `vol_sum = ask_v_raw + bid_v_raw` 
   - Логарифмирование: `log1p(vol_sum)`

3. **Проблема**: Если все исходные объемы примерно одинаковые (например, 3.5), то:
   - После суммирования 50 уровней: `50 × 3.5 = 175`
   - После `log1p(175)` ≈ `5.17`
   - Результат: **константа для всех тиков!**

## ✅ РЕШЕНИЕ

### Изменение 1: Volume Channel - СРЕДНЕЕ вместо СУММЫ

**Было (НЕПРАВИЛЬНО):**
```python
vol_sum = ask_v_raw + bid_v_raw  # (seq_len, 50)
vol_ch_raw = torch.log1p(vol_sum)  # Суммирование убивает вариативность
```

**Стало (ПРАВИЛЬНО):**
```python
vol_mean = (ask_v_raw + bid_v_raw) / 2.0  # (seq_len, 50)
vol_ch_raw = torch.log1p(vol_mean)  # Среднее сохраняет вариативность
```

**Обоснование:**
- Суммирование 50 уровней убивает вариативность между уровнями
- Среднее сохраняет различия между уровнями
- Каждый уровень LOB теперь вносит свой вклад в канал

### Изменение 2: Детальная диагностика

Добавлена диагностика на каждом этапе обработки:
1. Исходные данные (логарифмы из features.py)
2. Восстановленные сырые объемы
3. Volume channel после агрегации
4. Imbalance channel
5. OFI calculation

### Изменение 3: Синхронизация функций

Обе функции используют одинаковую логику:
- `_process_sample` - для обработки данных во время обучения
- `_compute_channels_for_normalization` - для расчета параметров нормализации

## 📝 ИЗМЕНЕННЫЕ ФАЙЛЫ

- `python_lab/src/dataset.py`:
  - Функция `_process_sample` (строки ~1500-1520)
  - Функция `_compute_channels_for_normalization` (строки ~1470-1480)

================================
# ПРОБЛЕМЫ ПОСЛЕ ЗАДАЧИ 315 и 315-2

✅ Задача 315-3: Найдена корневая причина!
Я проанализировал логи диагностики и нашёл проблему!

🔍 Корневая причина:
Volume Channel ДО нормализации:

vol_ch_raw: min=7.42, max=11.87, mean=9.78, std=0.87  ← НОРМАЛЬНО!
Volume Channel ПОСЛЕ нормализации:

Channel 1 (Vol): min=5.0, max=5.0, mean=5.0  ← КОНСТАНТА!
Проблема: Восстановление через exp(v)-1 создаёт ОГРОМНЫЕ числа (до 144567!). Параметры нормализации не справляются с такими данными, все z-scores получаются >> 5 и клипятся до 5.0.

✅ Решение:
УБРАТЬ восстановление объемов! Использовать логарифмы напрямую:

# БЫЛО (НЕПРАВИЛЬНО):
ask_v_raw = torch.exp(ask_v) - 1.0  # Огромные числа!
bid_v_raw = torch.exp(bid_v) - 1.0
vol_mean = (ask_v_raw + bid_v_raw) / 2.0
vol_ch_raw = torch.log1p(vol_mean)

# СТАЛО (ПРАВИЛЬНО):
vol_ch_raw = (ask_v + bid_v) / 2.0  # Среднее логарифмов
Обоснование:

Среднее логарифмов = геометрическое среднее (валидный признак для ML)
Диапазон [2.4, 11.9] удобен для нормализации
Не нужно восстанавливать, чтобы снова логарифмировать!
============================

# ЗАДАЧА 315-3 ПЛАН РЕАЛИЗАЦИИ:
### Шаг 1: Добавить диагностику параметров нормализации

Добавить вывод параметров (mean, std) для Channel 1 в `_process_sample`:

```python
# После normalize_channel
if idx is not None and 100 <= idx <= 105:
    print(f"\n[ЗАДАЧА 315-3] ПАРАМЕТРЫ НОРМАЛИЗАЦИИ Channel 1:")
    # Извлекаем параметры для первых 5 уровней
    for level in range(5):
        feat_idx = 1 * 50 + level  # Channel 1, level
        param_key = f"feat_{feat_idx}"
        if self.normalizer and self.normalizer.params:
            params = self.normalizer.params.get(param_key, {})
            mean_val = params.get("mean", 0.0)
            std_val = params.get("std", 1.0)
            print(f"  Level {level}: mean={mean_val:.4f}, std={std_val:.4f}")
    
    print(f"\n[ЗАДАЧА 315-3] ПОСЛЕ НОРМАЛИЗАЦИИ Channel 1:")
    print(f"  vol_ch: min={vol_ch.min():.4f}, max={vol_ch.max():.4f}, mean={vol_ch.mean():.4f}, std={vol_ch.std():.4f}")
```

### Шаг 2: Проверить, откуда берутся параметры нормализации

Проверить, вызывается ли `_compute_channels_for_normalization` при инициализации датасета:

```python
# В _init_memory_mode или _init_streaming_mode
print(f"[DEBUG] Computing normalization parameters...")
channels_df = self._compute_channels_for_normalization(sample_indices)
print(f"[DEBUG] Channels DF shape: {channels_df.shape}")
print(f"[DEBUG] Channel 1 (Vol) stats: min={channels_df['feat_50'].min():.4f}, max={channels_df['feat_50'].max():.4f}")
```

### Шаг 3: ГЛАВНОЕ ИСПРАВЛЕНИЕ - Убрать восстановление объемов

**Проблема:** Восстановление через `exp(v)-1` создаёт огромные числа (до 144567!), что усложняет нормализацию.

**Решение:** Использовать логарифмы НАПРЯМУЮ, без восстановления:

```python
# СТАРЫЙ КОД (НЕПРАВИЛЬНО):
ask_v_raw = torch.exp(ask_v) - 1.0
bid_v_raw = torch.exp(bid_v) - 1.0
vol_mean = (ask_v_raw + bid_v_raw) / 2.0
vol_ch_raw = torch.log1p(vol_mean)

# НОВЫЙ КОД (ПРАВИЛЬНО):
# ask_v и bid_v уже логарифмированы в features.py
# Просто берём среднее логарифмов
vol_ch_raw = (ask_v + bid_v) / 2.0  # Среднее логарифмов
```

**Обоснование:**
1. `log(a) + log(b) ≠ log(a+b)`, НО для нашей задачи среднее логарифмов - это валидный признак
2. Логарифмы уже в разумном диапазоне [2.4, 11.9], не нужно восстанавливать
3. Проще, быстрее, стабильнее для нормализации

### Шаг 4: Аналогично исправить Imbalance и OFI

**Imbalance Channel:**
```python
# СТАРЫЙ КОД (восстановление):
ask_v_raw = torch.exp(ask_v) - 1.0
bid_v_raw = torch.exp(bid_v) - 1.0
denom = bid_v_raw + ask_v_raw + 1e-8
imb_ch_raw = (bid_v_raw - ask_v_raw) / denom

# НОВЫЙ КОД (использовать логарифмы):
# Для imbalance нужны СЫРЫЕ объемы, но можно использовать приближение
# Или оставить как есть, если imbalance работает нормально (std=0.54)
```

**OFI Channel:**
```python
# OFI = diff(imbalance)
# Если imbalance правильный, то OFI тоже будет правильным
```

### Шаг 5: Синхронизировать _compute_channels_for_normalization

Применить те же изменения в `_compute_channels_for_normalization`:

```python
# НОВЫЙ КОД:
vol_ch = (ask_v + bid_v) / 2.0  # Среднее логарифмов, без восстановления
```

### Шаг 6: Удалить symlog_transform (если используется)

Если `symlog_transform` применяется где-то в коде, удалить его. Обычная z-score нормализация достаточна.

### Шаг 7: Проверить clipping

Убедиться, что clipping [-5, 5] применяется ПОСЛЕ нормализации:

```python
# В _process_sample
x_final = torch.stack([price_ch, vol_ch, imb_ch, ofi_ch, vib_ch, pr_ch], dim=1)
x_final = torch.clamp(x_final, -5.0, 5.0)  # Clipping ПОСЛЕ нормализации
```

## 📝 ИЗМЕНЕННЫЕ ФАЙЛЫ

### 1. `python_lab/src/dataset.py` - функция `_process_sample`

**Изменение 1: Упростить расчет Volume Channel**
```python
# Строки ~1520-1540
# БЫЛО:
ask_v_raw = torch.exp(ask_v) - 1.0
bid_v_raw = torch.exp(bid_v) - 1.0
vol_mean = (ask_v_raw + bid_v_raw) / 2.0
vol_ch_raw = torch.log1p(vol_mean)

# СТАЛО:
vol_ch_raw = (ask_v + bid_v) / 2.0  # Среднее логарифмов
```

**Изменение 2: Добавить диагностику параметров нормализации**
```python
# После vol_ch = self.normalize_channel(vol_ch_raw, channel_idx=1)
if idx is not None and 100 <= idx <= 105:
    print(f"\n[ЗАДАЧА 315-3] ПАРАМЕТРЫ НОРМАЛИЗАЦИИ Channel 1:")
    for level in range(5):
        feat_idx = 50 + level
        param_key = f"feat_{feat_idx}"
        if self.normalizer and self.normalizer.params:
            params = self.normalizer.params.get(param_key, {})
            print(f"  Level {level}: mean={params.get('mean', 0.0):.4f}, std={params.get('std', 1.0):.4f}")
    print(f"  vol_ch ПОСЛЕ нормализации: min={vol_ch.min():.4f}, max={vol_ch.max():.4f}, mean={vol_ch.mean():.4f}, std={vol_ch.std():.4f}")
```

**Изменение 3: Упростить Imbalance (опционально)**
```python
# Если imbalance работает нормально (std=0.54), можно оставить как есть
# Или упростить, если нужно
```

### 2. `python_lab/src/dataset.py` - функция `_compute_channels_for_normalization`

**Изменение: Синхронизировать с _process_sample**
```python
# Строки ~1470-1475
# БЫЛО:
ask_v_raw = np.exp(ask_v) - 1.0
bid_v_raw = np.exp(bid_v) - 1.0
vol_mean = (ask_v_raw + bid_v_raw) / 2.0
vol_ch = np.log1p(vol_mean)

# СТАЛО:
vol_ch = (ask_v + bid_v) / 2.0  # Среднее логарифмов
```












=================== # ОБЩАЯ ИНФОРМАЦИЯ ==================
# АНАЛИЗ ПРОБЛЕМ
Я нашел критическую ошибку в пайплайне данных: в dataset.py каналы (объем, дисбаланс, цена) рассчитываются неправильно, потому что используются уже предобработанные данные (логи и относительные цены) как если бы они были сырыми. Это приводит к потере сигнала и "каше" в данных.
