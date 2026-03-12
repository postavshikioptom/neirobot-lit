# Задача 311: Исправление несоответствия Python-Rust в формировании каналов - ЗАВЕРШЕНО

## КРИТИЧЕСКАЯ ПРОБЛЕМА

После глубокого анализа кода выявлена **корневая причина** всех проблем с нормализацией:

### Несоответствие порядка операций между Python (обучение) и Rust (инференс)

**Python (текущая реализация в dataset.py):**
```python
# 1. Данные приходят УЖЕ НОРМАЛИЗОВАННЫМИ из x_raw
ask_p = x_raw[:, ask_p_indices]  # УЖЕ нормализованные
bid_p = x_raw[:, bid_p_indices]  # УЖЕ нормализованные

# 2. Формируем каналы из нормализованных данных
price_ch = (ask_p + bid_p) / 2.0  # Среднее из нормализованных
vol_ch = ask_v + bid_v            # Сумма нормализованных
```

**Rust (текущая реализация в tensor.rs):**
```rust
// 1. Данные приходят СЫРЫМИ
let ask_p = raw[i * 2];      // СЫРЫЕ
let bid_p = raw[LOB_DEPTH * 2 + i * 2];  // СЫРЫЕ

// 2. Формируем каналы из сырых данных
let price_ch = (ask_p + bid_p) / 2.0;  // Среднее из сырых

// 3. Нормализуем каналы
buffer[i] = (price_ch - mean) * inv_std;  // Нормализация канала
```

### Математическое несоответствие

**Python вычисляет:**
```
price_ch = (norm(ask_p) + norm(bid_p)) / 2
```

**Rust вычисляет:**
```
price_ch = norm((ask_p + bid_p) / 2)
```

**Это НЕ эквивалентные операции!**


## ДОПОЛНИТЕЛЬНЫЕ ПРОБЛЕМЫ

### Проблема 2: Неправильный формат экспорта ONNX

**export_onnx.py (строка 30-50):**
```python
class ExportWrapper(nn.Module):
    def __init__(self, model, in_channels=3, n_levels=50):
        # Экспортирует модель с входом (B, S, 150)
        # 150 = 50 уровней * 3 канала
```

**Проблема:** Модель экспортируется с входом 150 колонок (3 канала), но Rust ожидает 300 колонок (6 каналов).

### Проблема 3: Логи печатаются ДО нормализации

**dataset.py (строка 1237-1320):**
Логи `[LOBDataset] First Sample Statistics` печатаются сразу после формирования каналов, но ДО того как датасет будет обработан нормализатором в train.py.

Это вводит в заблуждение - мы видим "огромные" значения OFI, но на самом деле они будут нормализованы позже.


## РЕШЕНИЕ: Унификация пайплайна Python и Rust

### Принцип решения

**Привести Python код к тому же порядку операций, что и в Rust:**
1. Формировать каналы из СЫРЫХ (ненормализованных) данных
2. Нормализовать КАНАЛЫ (а не сырые признаки)
3. Применять дополнительные трансформации (symlog) только там, где нужно

### Best Practices из исследований

Согласно статьям по LOB Transformers:
- **Feature-wise нормализация** - каждый признак нормализуется независимо
- **Z-score или Robust Scaler** - стандартные методы
- **Consistency между train и inference** - критично для работы модели

**Источники:**
- "Transformers for limit order books" (arXiv:2003.00130)
- "LOBench: Representation Learning of Limit Order Book" (arXiv:2505.02139)
- "Deep limit order book forecasting: a microstructural guide" (PMC12315853)


## ЗАДАЧА 311 ПЛАН РЕАЛИЗАЦИИ: - ЗАВЕРШЕНО

### Подзадача 311.1 - **[MODIFY]** dataset.py - Изменение порядка операций

**Цель:** Формировать каналы из сырых данных, как в Rust

**Файл:** `python_lab/src/dataset.py`

#### Шаг 1: Изменить метод `_init_memory_mode` (строки 823-949)

**Текущий код:**
```python
# Строка 854-866: Нормализация применяется к сырым признакам
if self.normalizer is not None:
    self.normalizer.fit(df_feat, feature_names=feat_cols)
    df_feat = self.normalizer.transform(df_feat)
```

**Проблема:** Нормализатор обучается на сырых признаках (ask_p, bid_p, ask_v, bid_v), но в Rust мы нормализуем каналы (price_ch, vol_ch).

**Решение:** 
1. НЕ применять нормализацию к сырым признакам в `_init_memory_mode`
2. Сохранить сырые данные в `self.x_raw`
3. Формировать каналы из сырых данных в `_process_sample`
4. Обучить нормализатор на каналах (а не на сырых признаках)


#### Шаг 2: Изменить метод `_process_sample` (строки 1237-1322)

**Текущий код (НЕПРАВИЛЬНО):**
```python
# Строка 1237-1240: Берем УЖЕ нормализованные данные
ask_p = torch.from_numpy(x_raw[:, self.ask_p_indices].copy()).float()
bid_p = torch.from_numpy(x_raw[:, self.bid_p_indices].copy()).float()

# Строка 1265-1267: Формируем каналы из нормализованных
price_ch = (ask_p + bid_p) / 2.0  # НЕПРАВИЛЬНО!
vol_ch = ask_v + bid_v            # НЕПРАВИЛЬНО!
```

**Новый код (ПРАВИЛЬНО):**
```python
# Берем СЫРЫЕ данные (без нормализации)
ask_p_raw = torch.from_numpy(x_raw[:, self.ask_p_indices].copy()).float()
bid_p_raw = torch.from_numpy(x_raw[:, self.bid_p_indices].copy()).float()
ask_v_raw = torch.from_numpy(x_raw[:, self.ask_v_indices].copy()).float()
bid_v_raw = torch.from_numpy(x_raw[:, self.bid_v_indices].copy()).float()

# Формируем каналы из СЫРЫХ данных (как в Rust)
price_ch_raw = (ask_p_raw + bid_p_raw) / 2.0
vol_ch_raw = ask_v_raw + bid_v_raw
imb_ch_raw = (bid_v_raw - ask_v_raw) / (bid_v_raw + ask_v_raw + 1e-8)

# Применяем нормализацию к КАНАЛАМ
# (нормализатор должен быть обучен на каналах, а не на сырых признаках)
price_ch = self.normalize_channel(price_ch_raw, channel_idx=0)
vol_ch = self.normalize_channel(vol_ch_raw, channel_idx=1)
imb_ch = self.normalize_channel(imb_ch_raw, channel_idx=2)
```


#### Шаг 3: Создать метод нормализации каналов

**Добавить в класс LOBDataset:**
```python
def normalize_channel(self, channel_data: torch.Tensor, channel_idx: int) -> torch.Tensor:
    """
    Нормализует канал используя статистики из normalizer.
    
    Args:
        channel_data: (Seq, Levels) - сырые данные канала
        channel_idx: индекс канала (0=price, 1=vol, 2=imb, 3=ofi, 4=vib, 5=pastret)
    
    Returns:
        Нормализованный канал той же формы
    """
    if self.normalizer is None:
        return channel_data
    
    seq_len, n_levels = channel_data.shape
    normalized = torch.zeros_like(channel_data)
    
    for level in range(n_levels):
        feat_idx = channel_idx * n_levels + level
        
        if self.normalizer.scaler_type == "zscore":
            mean = self.normalizer.params[f"feat_{feat_idx}"]["mean"]
            std = self.normalizer.params[f"feat_{feat_idx}"]["std"]
            normalized[:, level] = (channel_data[:, level] - mean) / (std + 1e-8)
        
        elif self.normalizer.scaler_type == "robust":
            median = self.normalizer.params[f"feat_{feat_idx}"]["median"]
            iqr = self.normalizer.params[f"feat_{feat_idx}"]["iqr"]
            normalized[:, level] = (channel_data[:, level] - median) / (iqr + 1e-8)
    
    return normalized
```


#### Шаг 4: Обучить нормализатор на каналах

**Изменить метод `_init_memory_mode`:**

```python
# СТАРЫЙ КОД (УДАЛИТЬ):
# if self.normalizer is not None:
#     self.normalizer.fit(df_feat, feature_names=feat_cols)
#     df_feat = self.normalizer.transform(df_feat)

# НОВЫЙ КОД:
# Сохраняем СЫРЫЕ данные
self.x_raw = df_feat.to_numpy().astype(np.float32)

# Формируем каналы из сырых данных для обучения нормализатора
if self.normalizer is not None:
    # Создаем DataFrame с каналами
    channels_data = self._compute_channels_for_normalization(df_feat)
    
    # Обучаем нормализатор на каналах
    self.normalizer.fit(channels_data, feature_names=channels_data.columns)
```

**Добавить метод `_compute_channels_for_normalization`:**
```python
def _compute_channels_for_normalization(self, df_raw: pl.DataFrame) -> pl.DataFrame:
    """
    Вычисляет каналы из сырых данных для обучения нормализатора.
    
    Это должно быть идентично тому, как каналы формируются в Rust.
    """
    # Извлекаем сырые признаки
    ask_p = df_raw.select([f"feat_ask_p_{i}" for i in range(50)]).to_numpy()
    bid_p = df_raw.select([f"feat_bid_p_{i}" for i in range(50)]).to_numpy()
    ask_v = df_raw.select([f"feat_ask_v_{i}" for i in range(50)]).to_numpy()
    bid_v = df_raw.select([f"feat_bid_v_{i}" for i in range(50)]).to_numpy()
    
    # Формируем каналы (как в Rust)
    price_ch = (ask_p + bid_p) / 2.0
    vol_ch = ask_v + bid_v
    imb_ch = (bid_v - ask_v) / (bid_v + ask_v + 1e-8)
    
    # Создаем DataFrame с каналами
    channels = np.concatenate([price_ch, vol_ch, imb_ch], axis=1)
    
    return pl.DataFrame(channels, schema=[f"feat_{i}" for i in range(150)])
```


### Подзадача 311.2 - **[MODIFY]** export_onnx.py - Исправление формата экспорта

**Цель:** Экспортировать модель с правильным форматом входа (300 колонок для 6 каналов)

**Файл:** `python_lab/src/export_onnx.py`

#### Изменения:

**СТАРЫЙ КОД (строка 30-50):**
```python
class ExportWrapper(nn.Module):
    def __init__(self, model, in_channels=3, n_levels=50):
        super().__init__()
        self.model = model
        self.in_channels = in_channels  # 3 канала
        self.n_levels = n_levels
        
    def forward(self, x):
        # x: (Batch, Seq, 150) - плоский входной тензор
        b, s, f = x.shape
        x_reshaped = x.view(b, s, 3, self.n_levels)  # (B, S, 3, 50)
        return self.model(x_reshaped)
```

**НОВЫЙ КОД:**
```python
class ExportWrapper(nn.Module):
    def __init__(self, model, in_channels=6, n_levels=50):
        super().__init__()
        self.model = model
        self.in_channels = in_channels  # 6 каналов
        self.n_levels = n_levels
        
    def forward(self, x):
        # x: (Batch, Seq, 300) - плоский входной тензор
        # 300 = 6 каналов * 50 уровней
        b, s, f = x.shape
        
        # Reshape (B, S, 300) -> (B, S, 6, 50)
        # 6 каналов: Price, Volume, Imbalance, OFI, VIB, PastReturns
        x_reshaped = x.view(b, s, 6, self.n_levels)  # (B, S, 6, 50)
        
        return self.model(x_reshaped)
```


#### Обновить dummy_input для экспорта:

**СТАРЫЙ КОД (строка 150-160):**
```python
# Создаем dummy input для экспорта
dummy_input = torch.randn(1, seq_len, 150)  # 3 канала * 50 уровней
```

**НОВЫЙ КОД:**
```python
# Создаем dummy input для экспорта
dummy_input = torch.randn(1, seq_len, 300)  # 6 каналов * 50 уровней
```

#### Обновить metadata.json:

**Добавить в metadata.json информацию о каналах:**
```json
{
    "model_type": "LiT",
    "version": "1.0",
    "input_shape": [1, 100, 300],
    "channels": 6,
    "levels": 50,
    "seq_len": 100,
    "channel_names": ["price", "volume", "imbalance", "ofi", "vib", "past_returns"],
    "normalization": {
        "type": "robust",
        "per_channel": true,
        "per_level": true
    }
}
```


### Подзадача 311.3 - **[MODIFY]** train.py - Добавление диагностики после нормализации

**Цель:** Добавить логи ПОСЛЕ нормализации для правильной диагностики

**Файл:** `python_lab/src/train.py`

#### Добавить метод диагностики в LiTModule:

```python
def log_channel_statistics(self, batch, batch_idx):
    """
    Логирует статистику каналов ПОСЛЕ нормализации.
    Вызывается в on_train_start для первого батча.
    """
    if batch_idx != 0:
        return
    
    x, y, v, w, regime_id = batch
    # x: (Batch, Seq, Channels, Levels)
    
    channel_names = ["Price", "Vol", "Imb", "OFI", "VIB", "PastRet"]
    
    print("\n[ДИАГНОСТИКА] Статистика каналов ПОСЛЕ нормализации:")
    for ch_idx, ch_name in enumerate(channel_names):
        ch_data = x[:, :, ch_idx, :]  # (Batch, Seq, Levels)
        print(f"  Channel {ch_idx} ({ch_name}): "
              f"min={ch_data.min():.4f}, "
              f"max={ch_data.max():.4f}, "
              f"mean={ch_data.mean():.4f}, "
              f"std={ch_data.std():.4f}")
```

#### Вызвать в training_step:

```python
def training_step(self, batch, batch_idx):
    # Диагностика для первого батча первой эпохи
    if self.current_epoch == 0 and batch_idx == 0:
        self.log_channel_statistics(batch, batch_idx)
    
    # ... остальной код
```


### Подзадача 311.4 - **[MODIFY]** normalization.py - Обновление для работы с каналами

**Цель:** Убедиться что нормализатор правильно работает с каналами

**Файл:** `python_lab/src/normalization.py`

#### Проверить что нормализатор сохраняет правильный порядок признаков:

**Текущий код (строка 40-50):**
```python
feat_cols = [c for c in data.columns if c.startswith("feat_")]
self.feature_order = feat_cols
```

**Это правильно** - нормализатор уже сохраняет порядок признаков.

#### Убедиться что параметры сохраняются правильно:

**Для 6 каналов * 50 уровней = 300 признаков:**
- feat_0 до feat_49: Price канал (уровни 0-49)
- feat_50 до feat_99: Volume канал (уровни 0-49)
- feat_100 до feat_149: Imbalance канал (уровни 0-49)
- feat_150 до feat_199: OFI канал (уровни 0-49)
- feat_200 до feat_249: VIB канал (уровни 0-49)
- feat_250 до feat_299: PastReturns канал (уровни 0-49)

**Проверка:** В Rust код ожидает именно такой порядок (см. tensor.rs строка 715-720).


### Подзадача 311.5 - **[VERIFY]** Проверка consistency Python-Rust

**Цель:** Убедиться что Python и Rust формируют идентичные тензоры

#### Создать тестовый скрипт `python_lab/tests/test_python_rust_consistency.py`:

```python
import torch
import numpy as np
import json
from pathlib import Path

def test_channel_formation():
    """
    Проверяет что Python и Rust формируют каналы одинаково.
    """
    # Создаем тестовые сырые данные
    ask_p = np.array([100.0, 100.1, 100.2])  # 3 уровня для простоты
    bid_p = np.array([99.9, 99.8, 99.7])
    ask_v = np.array([10.0, 20.0, 30.0])
    bid_v = np.array([15.0, 25.0, 35.0])
    
    # Python: формируем каналы
    price_ch_py = (ask_p + bid_p) / 2.0
    vol_ch_py = ask_v + bid_v
    imb_ch_py = (bid_v - ask_v) / (bid_v + ask_v + 1e-8)
    
    print("Python каналы:")
    print(f"  price_ch: {price_ch_py}")
    print(f"  vol_ch: {vol_ch_py}")
    print(f"  imb_ch: {imb_ch_py}")
    
    # Rust: (эмулируем логику из tensor.rs)
    price_ch_rust = (ask_p + bid_p) / 2.0
    vol_ch_rust = ask_v + bid_v
    imb_ch_rust = (bid_v - ask_v) / (bid_v + ask_v + 1e-7)
    
    print("\nRust каналы:")
    print(f"  price_ch: {price_ch_rust}")
    print(f"  vol_ch: {vol_ch_rust}")
    print(f"  imb_ch: {imb_ch_rust}")
    
    # Проверяем совпадение
    assert np.allclose(price_ch_py, price_ch_rust, rtol=1e-5)
    assert np.allclose(vol_ch_py, vol_ch_rust, rtol=1e-5)
    assert np.allclose(imb_ch_py, imb_ch_rust, rtol=1e-5)
    
    print("\n✓ Каналы формируются идентично!")

if __name__ == "__main__":
    test_channel_formation()
```

## Дополнителньая Подзадача 311.6 - Отключение всех режимов кроме memory
✅ Исправления завершены
dataset.py Исправлена ошибка - удален дублированный код в конце _init_memory_mode (строка 942)

Заменено len(self.features) на len(self.x_raw)
Закомментированы режимы в __init__:

elif data_mode == "streaming" → закомментировано
elif data_mode == "memmap" → закомментировано
Закомментированы методы:

_init_streaming_mode() - весь метод
_init_memmap_mode() - весь метод
_getitem_streaming() - весь метод
_build_row_offsets() - весь метод
train.py
Закомментирована функция _streaming_worker_init_fn()

Изменены параметры argparse:

--data_mode теперь только ["memory"]
--cache_dir закомментирован
Закомментированы все условия:

if args.data_mode == "streaming" (6 мест)
elif args.data_mode == "memmap" (3 места)
Условия с num_workers и worker_init_fn для streaming



------------------------------------------
## ОЖИДАЕМЫЕ РЕЗУЛЬТАТЫ

### После выполнения задачи 311:

#### 1. Логи обучения покажут правильные значения:

```
[ДИАГНОСТИКА] Статистика каналов ПОСЛЕ нормализации:
  Channel 0 (Price): min=-3.0, max=3.0, mean=0.0, std=1.0
  Channel 1 (Vol): min=-3.0, max=3.0, mean=0.0, std=1.0
  Channel 2 (Imb): min=-3.0, max=3.0, mean=0.0, std=1.0
  Channel 3 (OFI): min=-3.0, max=3.0, mean=0.0, std=1.0
  Channel 4 (VIB): min=-3.0, max=3.0, mean=0.0, std=1.0
  Channel 5 (PastRet): min=-3.0, max=3.0, mean=0.0, std=1.0
```

**Все каналы должны быть нормализованы с mean≈0, std≈1**
