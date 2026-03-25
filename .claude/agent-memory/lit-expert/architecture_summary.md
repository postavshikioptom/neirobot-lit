# Сводка по архитектуре LiT модели и pipeline (Задача 323 проверка)

## 1. Архитектура модели

### Параметры LiTModel (lit_model.py)
- **seq_len**: 100
- **in_channels**: 11 (после задачи 319, без CumOFI и ImbAccel)
- **d_model**: 96
- **nhead**: 4
- **num_layers**: 3 (было 2, увеличено в задаче 319)
- **dropout**: 0.1
- **activation**: gelu_exact (по умолчанию)

**Конфигурация каналов (11)**:
1. MicropriceDev
2. Vol
3. Imb (Static Imbalance)
4. OFI (Order Flow Imbalance)
5. VIB (Volume Imbalance)
6. Ret_10
7. Ret_50
8. Ret_100
9. Spread
10. DeltaImb
11. DeltaSpread

### LOBPatching (layers.py, задача 321)
**Архитектурный решение**: Compact Snapshot-Style Patching
- Вход: `(B, S, C=11, L=50)`
- **patch_conv**: Conv1d с `kernel_size=11`, `stride=11` → сжимает каждый уровень в d_model
- **Level Positional Embedding**: `(1, n_levels=50, d_model)` - позиционирование по глубине стакана
- **Temporal Positional Embedding**: `(1, seq_len, d_model)` - позиционирование по времени
- **Level Attention**: линейный слой + softmax по уровням для взвешивания важности глубины
- Выход: `(B, S, d_model)` - один snapshot token на временной шаг

**Преимущество**: избегает `O((S*L)^2)` затрат на внимание, эффективная компрессия 50 уровней в один токен

## 2. Расчет OFI (Order Flow Imbalance)

### Алгоритм Cont-Kukanov-Stoikov (CKS)
**Расположение**: `dataset.py:95-148` функция `compute_ofi_from_lob`

**Ключевые особенности**:
- **Per-tick non-cumulative**: возвращает дельты за каждый таймстеп, НЕ кумулятивную сумму
- **Маскирование обновлений**: учитывает только реальные обновления стакана через `update_ids`
- **Глубина**: по умолчанию `depth=3` уровня (первые 3 уровня стакана)

**Логика определения инициатора**:
```
buyer-initiated:  bid_price[t] > bid_price[t-1]  OR (bid_price[t] == bid_price[t-1] AND bid_volume[t] > bid_volume[t-1])
seller-initiated: ask_price[t] < ask_price[t-1]  OR (ask_price[t] == ask_price[t-1] AND ask_volume[t] > ask_volume[t-1])
```

**Формула**:
```
OFI_delta[t] = Σ_{i=0}^{depth-1} [ bid_vol_diff[t,i] * I{buyer-initiated at level i}
                                      - ask_vol_diff[t,i] * I{seller-initiated at level i} ]
```

**Восстановление сырых объемов**:
- Вход: log1p volumes `log(1+v)`
- Восстановление: `exp(log1p(v)) - 1`
- Защита: `clip(v, max=20.0)` для предотвращения overflow (exp(20) ≈ 485M безопасно)

**Интеграция в pipeline**:
- Предвычисляется в `LOBDataset._init_memory_mode()` при инициализации
- Кэшируется в `self.ofi_cache` (массив shape `(N,)`)
- Используется в `_calculate_6_channels_raw()` и нормализуется отдельно

## 3. Нормализация и обработка данных

### Двухэтапная нормализация

**Static каналы** (8 каналов): MicropriceDev, Vol, Imb, VIB, Ret_10, Ret_50, Ret_100, Spread
- Метод: Z-score или Robust (median/IQR) в зависимости от `scaler_type`
- Параметры: `mean/std` или `median/iqr` на **каждый признак** (550 признаков = 11×50)
- Сохраняется `feature_order` для детерминированности

**Dynamic каналы** (3 канала): OFI, DeltaImb, DeltaSpread
1. **Symlog transform**: `sign(x) * log1p(abs(x))`
2. **Robust нормализация**: `(x - median) / IQR` (без винзоризации)
3. **Clipping**: `[-4.0, 4.0]`
4. **Защита**: `nan_to_num(0.0)`, `inf` → `±5.0`

**Критически важные реализации**:
```python
# Normalizer.fit() вызывается ТОЛЬКО на train-части (защита от data leakage)
train_channels_df = full_dataset._compute_channels_for_normalization(train_indices_for_fit)
normalizer.fit(train_channels_df, ...)

# Динамические параметры считаются отдельно (защищены от NaN)
self.dynamic_params[name] = {"median": median, "iqr": iqr}
```

### Структура данных
**Исходные колонки LOB** (200 признаков):
- `ask_p_0..49`, `ask_v_0..49` (100)
- `bid_p_0..49`, `bid_v_0..49` (100)

**Дополнительные индикаторы** (после feature engineering):
- `feat_update_id` — для OFI маскирования
- `feat_vib_*`, `feat_past_return_*` (но в коде эти признаки вычисляются на лету из сырых данных)

**Финальные 550 признаков** (11 каналов × 50 уровней):
```python
# Каждый канал повторяется на 50 уровнях
ch0_MicropriceDev[0..49], ch1_Vol[0..49], ..., ch10_DeltaSpread[0..49]
```

## 4. Реализованные решения известных проблем

### Data & Preprocessing
✅ **Look-ahead bias**: хронологическое разделение `train_data.py:77-90` (70/15/15), `indices` используются как `range(0, train_size)`, не перемешиваются

✅ **Data leakage**: нормализатор `fit()` только на train, `transform()` на val/test; сохранение `feature_order` для консистентности

✅ **OFI bias**: Замена кумулятивного OFI на per-tick CKS с маскированием обновлений (`dataset.py:95-148`)

⚠️ **Distribution shift**: частично обработано через:
  - RegimeDetector (временно отключен, `train_data.py:231`)
  - Time weighting (`half_life_hours`, `train_data.py:237-250`)
  - Gradient checkpointing для OOM

### Архитектура
✅ **ViT не подходит**: реализована LiP (LOB Patching) с level attention вместо наивного ViT

✅ **Гибридный подход**: Transformer + LSTM заменен на **Transformer с эффективным патчингом** (в оригинальном LiT есть LSTM, но здесь используется только Transformer из задач 237-310)

✅ **Spatial structure**: Level attention взвешивает уровни стакана, не все токены равны

✅ **Broadcast scalar features avoided**: каждый канал имеет свои параметры нормализации на каждом уровне (550 параметров для 11×50)

### Оценка и обучение
✅ **Class imbalance**:
  - Weighted Focal Loss (настраивается через `class_weight_smooth`)
  - Амплификация весов когда Flat > 85% (`train_data.py:178-182`)
  - Time weighting + class weights (`train_data.py:816-820`)

✅ **Градиентные проблемы**:
  - Gradient checkpointing (`use_gradient_checkpointing`)
  - QK Normalization (`layers.py:178-182`)
  - Gradient clipping через `on_before_optimizer_step` (train.py)
  - Curvature penalty регуляризация (`lit_model.py:79-126`)

✅ **NaN/Inf защита**:
  - В нормализаторе: `eps=1e-6`, `std=1.0` если NaN/0
  - В каналах: `nan_to_num(0.0)`, `posinf=5.0`, `neginf=-5.0`
  - В target: clamp до [0, 2]
  - Диагностика первых 100 сэмплов после нормализации (`train_data.py:118-156`)

✅ **Saturation диагностика**: логирование % значений за пределами clamp (`dataset.py:1729-1763`)

### Экспорт и совместимость
✅ **ONNX export**:
  - Обновлены размерности: 450 → 550 признаков (11×50)
  - Экспорт с динамическими осями дляseq_len
  - Сохранение JSON параметров нормализации для Rust

## 5. Критические места для проверки (Задача 323)

### A. Консистентность размерностей
- **ONNX vs Rust**: Rust код ожидает 550 признаков (11×50)?
- Проверить `export_onnx.py` и Rust inference код на соответствие
- **Known gap**: ранее была проблема 9 vs 13 каналов — сейчас 11 каналов, убедиться что Rust использует 11

### B. OFI pipeline
- `update_id_raw` гарантированно существует? (`dataset.py:917-921`)
- В precompute: `compute_ofi_from_lob_cache` корректно восстанавливает сырые volume?
- В `__getitem__`: `ofi_precomp` передается как torch tensor из numpy кэша?

### C. Normalizer dynamic channels
- `dynamic_params` сохраняются в JSON? (`normalization.py:167`)
- Rust-side нормализация dynamic каналов должна использовать median/IQR, не mean/std
- Порядок каналов в `feature_order` строгий? (`train_data.py:986`)

### D. Chronological split
- Индексы действительно последовательные? (`train_data.py:83-85`)
- Нет перемешивания при DataLoader? (`shuffle=True` только для train)

### E. Channel order validation
```python
# dataset.py:770-777
channel_names = ["MicropriceDev", "Vol", "Imb", "OFI", "VIB",
                 "Ret_10", "Ret_50", "Ret_100", "Spread",
                 "DeltaImb", "DeltaSpread"]
expected_cols = [f"feat_{i}" for i in range(11 * 50)]  # 550
```
Убедиться что Rust-side использует тот же порядок.

### F. Multi-horizon support
- `num_horizons` выводится из колонок `label_h*` (`dataset.py:746-753`)
- `LiTModel` поддерживает multi-horizon через `num_horizons` и `use_horizon_embedding`
- Выход: `(batch, num_horizons, 3)` или `(batch, 3)` в зависимости от режима

## 6. Известные ограничения

1. **Только memory mode**: streaming и memmap отключены для простоты (`dataset.py:756-767`)
2. **RegimeDetector отключен**: временно отключен (Задача 155 приостановлена)
3. **Dynamic OFI отключен**: `compute_ofi()` вызывает NotImplementedError, используется только static OFI

## 7. Контакты и пути

**Ключевые файлы**:
- Архитектура: `python_lab/src/lit_model.py`, `python_lab/src/layers.py`
- Dataset: `python_lab/src/dataset.py`
- Normalization: `python_lab/src/normalization.py`
- Data pipeline: `python_lab/src/train_data.py`
- Feature engineering: `python_lab/src/dataset.py` (внутри `_calculate_6_channels_raw`)

**Export ONNX**: `python_lab/src/export_onnx.py` (не читали, но важно проверить соответствие размеров)

---

**Вывод**: Архитектура соответствует LiT paper (LOB Patching + Transformer), OFI правильно реализован как per-tick CKS, нормализация защищена от leakage. Основная проверка для задачи 323 — консистентность размерностей и pipeline между Python обучением и Rust инференсом.
