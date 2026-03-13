# План оптимизации скорости обучения (эпоха 25 мин → 5 мин)

## Диагностика проблемы

### Текущее состояние (из логов)
- Эпоха 0: 23m 31s
- Эпоха 1: 22m 24s
- Цель: ~5 минут на эпоху
- Датасет: 370,680 семплов
- Batch size: 64
- Количество батчей: ~5792

### Главная причина замедления

В файле [`python_lab/src/dataset.py`](python_lab/src/dataset.py:1248) метод `normalize_channel` делает нормализацию в цикле `for` по 50 уровням:

```python
def normalize_channel(self, channel_data: torch.Tensor, channel_idx: int) -> torch.Tensor:
    for level in range(n_levels):  # <-- 50 уровней!
        feat_idx = channel_idx * n_levels + level
        param_key = f"feat_{feat_idx}"
        # нормализация для каждого уровня отдельно
```

Этот метод вызывается 6 раз (для каждого канала: Price, Vol, Imb, OFI, VIB, PastRet) и для КАЖДОГО семпла!

**Итого: 6 каналов × 50 уровней = 300 операций нормализации на каждый семпл!**

Для 370,680 семплов это создаёт огромную нагрузку на CPU.

---

## План оптимизации (одна конкретная задача)

### Задача: Векторизовать normalize_channel

**Файл для изменения:** `python_lab/src/dataset.py`

**Текущий код (строки 1248-1279):**
```python
def normalize_channel(self, channel_data: torch.Tensor, channel_idx: int) -> torch.Tensor:
    if self.normalizer is None:
        return channel_data
    
    seq_len, n_levels = channel_data.shape
    normalized = torch.zeros_like(channel_data)
    
    for level in range(n_levels):  # <-- ПРОБЛЕМА
        feat_idx = channel_idx * n_levels + level
        param_key = f"feat_{feat_idx}"
        
        if self.normalizer.scaler_type == "zscore":
            mean = self.normalizer.params.get(param_key, {}).get("mean", 0.0)
            std = self.normalizer.params.get(param_key, {}).get("std", 1.0)
            normalized[:, level] = (channel_data[:, level] - mean) / (std + 1e-8)
        
        elif self.normalizer.scaler_type == "robust":
            median = self.normalizer.params.get(param_key, {}).get("median", 0.0)
            iqr = self.normalizer.params.get(param_key, {}).get("iqr", 1.0)
            normalized[:, level] = (channel_data[:, level] - median) / (iqr + 1e-8)
    
    return normalized
```

**Оптимизированный код (векторизованный):**
```python
def normalize_channel(self, channel_data: torch.Tensor, channel_idx: int) -> torch.Tensor:
    if self.normalizer is None:
        return channel_data
    
    n_levels = channel_data.shape[1]
    start_feat_idx = channel_idx * n_levels
    
    # Векторизованное извлечение параметров для всех уровней сразу
    if self.normalizer.scaler_type == "zscore":
        means = []
        stds = []
        for level in range(n_levels):
            feat_idx = start_feat_idx + level
            param_key = f"feat_{feat_idx}"
            means.append(self.normalizer.params.get(param_key, {}).get("mean", 0.0))
            stds.append(self.normalizer.params.get(param_key, {}).get("std", 1.0))
        
        mean_tensor = torch.tensor(means, device=channel_data.device, dtype=channel_data.dtype)
        std_tensor = torch.tensor(stds, device=channel_data.device, dtype=channel_data.dtype)
        return (channel_data - mean_tensor) / (std_tensor + 1e-8)
    
    elif self.normalizer.scaler_type == "robust":
        medians = []
        iqrs = []
        for level in range(n_levels):
            feat_idx = start_feat_idx + level
            param_key = f"feat_{feat_idx}"
            medians.append(self.normalizer.params.get(param_key, {}).get("median", 0.0))
            iqrs.append(self.normalizer.params.get(param_key, {}).get("iqr", 1.0))
        
        median_tensor = torch.tensor(medians, device=channel_data.device, dtype=channel_data.dtype)
        iqr_tensor = torch.tensor(iqrs, device=channel_data.device, dtype=channel_data.dtype)
        return (channel_data - median_tensor) / (iqr_tensor + 1e-8)
    
    return channel_data
```

---

## ОШИБКА В НОРМАЛИЗАЦИИ ДАННЫХ (КРИТИЧЕСКАЯ!)

### Проблема: Channel 2 (Imb) и Channel 3 (OFI) дублируют друг друга

**Симптомы (из логов):**
```
Channel 2 (Imb): min=-8.9519, max=8.6557, mean=0.0045, std=0.9640
Channel 3 (OFI): min=-8.9519, max=8.6557, mean=0.0045, std=0.9640
```

**Причина в коде:**

1. [`compute_static_imbalance()`](python_lab/src/dataset.py:206) (строка 206-232):
   ```python
   imbalance = (bid_v - ask_v) / denom  # Формула для OFI (Channel 3)
   ```

2. [`_process_sample()`](python_lab/src/dataset.py:1371) (строка 1371):
   ```python
   imb_ch_raw = (bid_v - ask_v) / denom  # Формула для Imb (Channel 2)
   ```

**Это одна и та же формула!** OFI должен быть ДРУГИМ - это должен быть кумулятивный (cumulative) дисбаланс или динамический поток ордеров, а не статический.

### Решение:

Для Channel 3 (OFI) нужно использовать ДРУГУЮ формулу - кумулятивный OFI:
```python
def compute_cumulative_ofi(bid_v, ask_v):
    """Кумулятивный OFI - сумма дисбалансов по всем уровням"""
    ofi = np.zeros_like(bid_v)
    for i in range(bid_v.shape[1]):
        denom = bid_v[:, i] + ask_v[:, i] + 1e-8
        ofi[:, i] = (bid_v[:, i] - ask_v[:, i]) / denom
    # Кумулятивная сумма
    return np.cumsum(ofi, axis=1)
```

ИЛИ - если это не критично для модели - можно просто отключить один из дублирующихся каналов (убрать OFI или Imb).

---

## ОЖИДАЕМЫЙ РЕЗУЛЬТАТ

- Текущее время эпохи: ~23 минуты
- Ожидаемое время после оптимизации: ~3-5 минут
- Ускорение: примерно в 5-7 раз

---

## ДОПОЛНИТЕЛЬНО: Использование GPU

### Вопрос: Используется ли GPU?

**Да, GPU используется правильно!**

В [`train.py:1139-1141`](python_lab/src/train.py:1139):
```python
accelerator="auto",
devices=1,
precision="16-mixed" if torch.cuda.is_available() else 32,
```

Данные переносятся на GPU:
```python
x = batch[0].to(model.device)
y = batch[1].to(model.device)
```

### НО: Heavy preprocessing на CPU!

**Что происходит на CPU (в _process_sample, строки 1341-1431):**
1. Создание torch.Tensor из numpy: `torch.from_numpy(x_raw[:, ...])`
2. **Нормализация (normalize_channel)** - 300 операций на sample - ГЛАВНАЯ ПРОБЛЕМА
3. Аугментация (symmetric flip, volume jitter)

**Что происходит на GPU:**
- Модель (LiTModel) - forward + backward pass
- Loss computation

### Вывод:
- GPU работает, но данные приходят на него уже "тяжёлыми" - после CPU preprocessing
- Аугментация на CPU - это меньшая проблема (тензоры уже созданы в torch)
- **Главная проблема - normalize_channel с циклом for по 50 уровням**

---

## Дополнительные потенциальные оптимизации (после основной)

1. **Предварительная нормализация** - нормализовать все данные при инициализации dataset, а не при каждом __getitem__
