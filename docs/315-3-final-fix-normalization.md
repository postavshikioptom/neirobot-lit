# Задача 315-3: Финальное исправление нормализации каналов

## 🎯 ДИАГНОСТИКА ПРОБЛЕМЫ

### Результаты анализа логов

**Исходные данные (логарифмы из features.py):**
```
ask_v: min=2.40, max=11.88, mean=9.44, std=1.41
bid_v: min=4.96, max=11.88, mean=9.59, std=1.13
```

**После восстановления (exp(v)-1):**
```
ask_v_raw: min=10, max=144567, mean=23940, std=26750
bid_v_raw: min=141, max=144306, mean=25270, std=28471
```

**Volume Channel ДО нормализации:**
```
vol_ch_raw: min=7.42, max=11.87, mean=9.78, std=0.87  ← НОРМАЛЬНАЯ ДИСПЕРСИЯ!
```

**Volume Channel ПОСЛЕ нормализации:**
```
Channel 1 (Vol): min=5.0, max=5.0, mean=5.0  ← КОНСТАНТА!
```

### 🔍 КОРНЕВАЯ ПРИЧИНА

**Проблема:** `vol_ch_raw` имеет нормальную дисперсию (std=0.87), но после `normalize_channel` становится константой 5.0!

**Причина:** Параметры нормализации (mean, std) НЕПРАВИЛЬНЫЕ. При применении z-score:
```python
z = (x - mean) / std
```

Если `mean` и `std` не соответствуют реальным данным, то все z-scores получаются >> 5 и клипятся до 5.0.

**Гипотеза:** Normalizer обучается на НЕПРАВИЛЬНЫХ данных или использует СТАРЫЕ параметры из кэша.

## ✅ ПЛАН ИСПРАВЛЕНИЯ

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

## 🎯 ОЖИДАЕМЫЙ РЕЗУЛЬТАТ

После исправлений:

**Volume Channel ДО нормализации:**
```
vol_ch_raw: min=3.7, max=11.9, mean=9.5, std=1.3  ← Среднее логарифмов
```

**Volume Channel ПОСЛЕ нормализации:**
```
Channel 1 (Vol): min=-2.5, max=2.0, mean=0.0, std=1.0  ← Нормальное распределение!
```

**Все каналы ПОСЛЕ нормализации:**
```
Channel 0 (Price): mean≈0, std≈1
Channel 1 (Vol): mean≈0, std≈1  ← ИСПРАВЛЕНО!
Channel 2 (Imb): mean≈0, std≈1
Channel 3 (OFI): mean≈0, std≈1  ← Должно улучшиться
Channel 4 (VIB): mean≈0, std≈1
Channel 5 (PastRet): mean≈0, std≈1
```

## 🔗 МАТЕМАТИЧЕСКОЕ ОБОСНОВАНИЕ

### Почему среднее логарифмов - это правильно?

**Старый подход (НЕПРАВИЛЬНО):**
```
1. features.py: v → log(1+v)
2. dataset.py: log(1+v) → exp-1 → v (восстановление)
3. dataset.py: v → sum → log(1+sum)
```
Проблема: Огромные числа после восстановления (до 144567!)

**Новый подход (ПРАВИЛЬНО):**
```
1. features.py: v → log(1+v)
2. dataset.py: (log(1+v_ask) + log(1+v_bid)) / 2
```

**Интерпретация:**
- Среднее логарифмов = log(√(v_ask × v_bid)) ≈ геометрическое среднее
- Это валидный признак для ML, показывает "типичный" объем на уровне
- Диапазон [2.4, 11.9] удобен для нормализации

### Почему не нужно восстанавливать?

1. Логарифмы уже нормализованы в features.py
2. Восстановление создаёт огромные числа, которые сложно нормализовать
3. Для ML важна ОТНОСИТЕЛЬНАЯ информация, а не абсолютные значения
4. Среднее логарифмов сохраняет вариативность между уровнями

## 📊 СЛЕДУЮЩИЕ ШАГИ

1. ✅ Упростить расчет Volume Channel (убрать восстановление)
2. ✅ Синхронизировать _compute_channels_for_normalization
3. ✅ Добавить диагностику параметров нормализации
4. ⏳ Запустить обучение и проверить результаты
5. ⏳ Убедиться, что все каналы имеют mean≈0, std≈1
6. ⏳ Проверить метрики обучения (Directional Accuracy, Hit Rate)

## 🔗 СВЯЗАННЫЕ ЗАДАЧИ

- Задача 315: Первая попытка (восстановление сырых объемов)
- Задача 315-2: Вторая попытка (среднее вместо суммы)
- Задача 315-3: Третья попытка (убрать восстановление, использовать логарифмы напрямую)
