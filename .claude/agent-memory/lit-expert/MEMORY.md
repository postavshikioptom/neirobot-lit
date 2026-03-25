# Lit-expert Agent Memory

## Архитектурные решения (Задача 319-323)

### Параметры LiTModel
- seq_len: 100
- in_channels: 11 (MicropriceDev, Vol, Imb, OFI, VIB, Ret_10/50/100, Spread, DeltaImb, DeltaSpread)
- d_model: 96, nhead: 4, num_layers: 3
- activation: gelu_exact

### LOBPatching (Задача 321)
- Compact snapshot-style: (B, S, 11, 50) → (B, S, d_model)
- Patch Conv1d с kernel_size=11 (каждый уровень → d_model)
- Level + Temporal positional embeddings
- Level attention для взвешивания глубины стакана
- Избегает O((S*L)^2) затрат

### OFI Pipeline (Задача 318-323)
- Per-tick non-cumulative CKS (Cont-Kukanov-Stoikov)
- Маскирование по update_ids (только при реальных обновлениях)
- Depth=3 уровня
- Восстановление сырых объемов: exp(log1p(v)) - 1 с clamp(max=20)
- Precomputed кэш, используется в каналах

### Нормализация
**Static каналы** (8): Z-score или Robust (median/IQR) на каждый из 550 признаков
**Dynamic каналы** (3: OFI, DeltaImb, DeltaSpread):
  1. Symlog: sign(x) * log1p(abs(x))
  2. Robust: (x - median) / IQR
  3. Clamp: [-4.0, 4.0]

**Защита от leakage**: fit только на train, сохранение feature_order

### Проблемы и решения
✅ Look-ahead bias: хронологический split 70/15/15
✅ OFI cumulative bias: заменен на per-tick CKS
✅ Gradient issues: checkpointing, QK norm, clipping
✅ Class imbalance: weighted loss + amplification if Flat>85%
✅ NaN/Inf: многоуровневая защита, диагностика
✅ Saturation: логирование % выходящих за clamp
✅ Dynamic feature contract (Задача 324):
   - DeltaImb/DeltaSpread переведены на event-consistent кэши
   - Fit normalizer использует полное train распределение
   - Агрегированная диагностика train split
   - Hard guard против плохого scale (saturation >10%, zeros>95%, collapsed)
   - CLI флаг: --allow-bad-dynamic-scale

### Ключевые файлы
- `python_lab/src/lit_model.py` — модель
- `python_lab/src/layers.py` — LOBPatching + CustomTransformer
- `python_lab/src/dataset.py` — OFI, каналы, датасет
- `python_lab/src/normalization.py` — Normalizer
- `python_lab/src/train_data.py` — pipeline

### Проверка для задачи 323
**Консистентность Python↔Rust**:
- ONNX input: 550 (11×50) признаков?
- Rust использует тот же order каналов?
- Dynamic channels нормализуются через symlog+robust (не zscore)?

**OFI pipeline**:
- update_id_raw всегда присутствует?
- compute_ofi_from_lob_cache корректно восстанавливает volumes?
