# Задача 315: Исправить функции расчета каналов и их нормализации в dataset.py VOL, OFI.

Продолжаю анализ и формирую итоговый план.

Отлично! Вижу что уже были попытки исправить OFI, но основная проблема с двойной нормализацией объемов не была решена. Теперь сформирую финальный план.

Теперь сформирую итоговый план на русском:

---

# ПЛАН ИСПРАВЛЕНИЯ: Задача 315 - Устранение двойной нормализации в пайплайне LOB данных

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




=================== # ОБЩАЯ ИНФОРМАЦИЯ ==================
# АНАЛИЗ ПРОБЛЕМ
Я нашел критическую ошибку в пайплайне данных: в dataset.py каналы (объем, дисбаланс, цена) рассчитываются неправильно, потому что используются уже предобработанные данные (логи и относительные цены) как если бы они были сырыми. Это приводит к потере сигнала и "каше" в данных.
