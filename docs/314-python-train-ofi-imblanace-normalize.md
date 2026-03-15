# Задача 314: Исправление ошибок OFI нормализации и корректировка весов классов. - ЗАВЕРШЕНО

## Описание проблемы

Последний запуск обучения (output.txt) показал катастрофически низкие метрики:
- **MCC ≈ 0** (от -0.01 до 0.03) — модель НЕ различает классы
- **DA ≈ 10%** — хуже случайного угадывания (33%)
- **Precision Up ≈ 9%**, Down ≈ 13% — модель ошибается в ~90% предсказаний
- **Confidence Gap ≈ 0.015** — модель одинаково «уверена» в верных и ошибочных предсказаниях

## Корневые причины

### 🔴 КРИТИЧНО: Channel 3 (OFI) — сломанная нормализация

**Факт из логов:** После нормализации Channel 3 имеет `mean=5.51, std=5.97` (при норме `mean≈0, std≈1`).

**Причина:** В [dataset.py](file:///e:/MAX/PYTHON/NEURAL-BOTS/neirobot-lit/python_lab/src/dataset.py#L1469):
```python
ofi_raw = torch.from_numpy(np.cumsum(imb_seq, axis=1).copy()).float()
```
`cumsum` по axis=1 (50 уровней) создаёт **монотонно нарастающие** значения:
- Уровень 0: `imb[0]` (≈ нормальный, ~[-1, 1])
- Уровень 49: `sum(imb[0..49])` (может быть ≈ ±25)

Нормализатор пытается подогнать каждый уровень отдельно, но cumsum-паттерн **катастрофически ломает распределение**.

**Последствие:** Transformer уделяет ~80% внимания аномальному Channel 3, подавляя сигнал из остальных 5 каналов. Модель "видит" постоянное положительное давление покупки → предсказывает 54% Up в Epoch 1.

### 🟡 СУЩЕСТВЕННО: Отсутствие per-channel clipping

Только Channel 0 (Price) имеет clipping `[-5, 5]`. Остальные каналы не ограничены, что позволяет Channel 3 доминировать.

### 🟡 СУЩЕСТВЕННО: Неэффективная компенсация дисбаланса классов

Распределение: Flat=67%, Up=16.4%, Down=16.8%. При текущей конфигурации `class_weight_smooth=1.0` и `loss_type=focal`, веса классов могут быть недостаточно агрессивными в сочетании с аномальным каналом OFI.

---

## ЗАДАЧА 314 ПЛАН РЕАЛИЗАЦИИ

### Компонент 1: Подготовка признаков (dataset.py)

> [!IMPORTANT]
> Это ГЛАВНОЕ изменение. Без него остальные улучшения не дадут эффекта.

#### [MODIFY] [dataset.py](file:///e:/MAX/PYTHON/NEURAL-BOTS/neirobot-lit/python_lab/src/dataset.py)

**Подзадача 1.1: Заменить cumulative OFI на temporal delta-imbalance**

Строки ~1462-1477. Вместо `cumsum(imb_seq, axis=1)` (по уровням), использовать `diff(imb_seq, axis=0)` (по времени).

```diff
-        # ch[3]: Cumulative OFI (Задача 312.2.2) - кумулятивный дисбаланс потока ордеров
-        if idx is not None and hasattr(self, 'imbalance_cache') and self.imbalance_cache is not None:
-            imb_seq = self.imbalance_cache[idx : idx + self.seq_len]
-            ofi_raw = torch.from_numpy(np.cumsum(imb_seq, axis=1).copy()).float()
+        # ch[3]: Delta OFI (Задача 314.1) - изменение дисбаланса по времени
+        if idx is not None and hasattr(self, 'imbalance_cache') and self.imbalance_cache is not None:
+            imb_seq = self.imbalance_cache[idx : idx + self.seq_len]  # (seq_len, 50)
+            # Вычисляем разницу между тиками (axis=0 = время)
+            delta_imb = np.diff(imb_seq, axis=0, prepend=imb_seq[:1])  # (seq_len, 50)
+            ofi_raw = torch.from_numpy(delta_imb.copy()).float()
```

**Подзадача 1.2: Аналогичное исправление в fallback ветке**

```diff
         else:
-            bid_v_np = bid_v.numpy()
-            ask_v_np = ask_v.numpy()
-            ofi_np = compute_cumulative_ofi(bid_v_np, ask_v_np)
-            ofi_raw = torch.from_numpy(ofi_np).float()
+            # Fallback: delta-imbalance на лету
+            denom_fb = bid_v + ask_v + 1e-8
+            imb_fb = (bid_v - ask_v) / denom_fb  # (seq_len, 50)
+            delta_fb = torch.diff(imb_fb, dim=0, prepend=imb_fb[:1])
+            ofi_raw = delta_fb
```

**Подзадача 1.3: Добавить per-channel clipping для ВСЕХ каналов**

После строки 1513 (`x_final = torch.stack(...)`) добавить:

```python
        # Задача 314.3: Per-channel clipping для стабильности
        x_final = torch.clamp(x_final, -5.0, 5.0)
```

И удалить отдельный `clamp` для Price на строке ~1460 (он станет избыточен).

**Подзадача 1.4: Аналогичное исправление в [_compute_channels_for_normalization](file:///e:/MAX/PYTHON/NEURAL-BOTS/neirobot-lit/python_lab/src/dataset.py#1360-1419)**

Строки ~1406-1407 — там тоже `cumsum` для OFI, нужно заменить на `diff`:

```diff
-            ofi_ch = compute_static_imbalance(bid_v, ask_v)  # (N, 50)
+            # Delta imbalance по строкам (временная ось)
+            raw_imb = compute_static_imbalance(bid_v, ask_v)  # (N, 50)
+            ofi_ch = np.diff(raw_imb, axis=0, prepend=raw_imb[:1])  # (N, 50)
```

---

**Подзадача 1.5: Добавить диагностический warning при аномальных каналах**

В [_process_sample](file:///e:/MAX/PYTHON/NEURAL-BOTS/neirobot-lit/python_lab/src/dataset.py#1420-1542), после формирования `x_final`, добавить проверку (только первые 100 сэмплов):

```python
        # Задача 314.5: Диагностика нормализации
        if idx is not None and idx < 100 and idx % 50 == 0:
            for ch_idx in range(x_final.shape[1]):
                ch_mean = x_final[:, ch_idx, :].mean().item()
                ch_std = x_final[:, ch_idx, :].std().item()
                if abs(ch_mean) > 2.0 or ch_std > 5.0:
                    print(f"[WARNING] Channel {ch_idx} has anomalous stats: mean={ch_mean:.2f}, std={ch_std:.2f}")
```

---

### Компонент 2: Веса классов (train.py)

#### [MODIFY] [train.py](file:///e:/MAX/PYTHON/NEURAL-BOTS/neirobot-lit/python_lab/src/train.py)

**Подзадача 2.1: Inverse frequency class weights**

Найти место, где вычисляются [weights](file:///e:/MAX/PYTHON/NEURAL-BOTS/neirobot-lit/python_lab/src/dataset.py#773-792) (class_weights), и убедиться, что используется формула inverse frequency.

Нужно найти блок с вычислением весов (примерно строки 1700-1800) и проверить формулу:
```python
# Веса = N_total / (K * N_class_i) — inverse frequency
weights = total_samples / (num_classes * class_counts)
```

---

### Компонент 3: Слой патчинга (layers.py)

#### [MODIFY] [layers.py](file:///e:/MAX/PYTHON/NEURAL-BOTS/neirobot-lit/python_lab/src/layers.py)

**Подзадача 3.1: Добавить BatchNorm перед attention pooling**

После `x_patched = self.act(x_patched)` (строка ~90) и перед level_attention, добавить BatchNorm для стабилизации:

```python
        # Задача 314.6: Дополнительная нормализация перед aggegation
        self.pre_attn_norm = nn.LayerNorm(d_model)
        # ... в forward:
        x_patched = self.pre_attn_norm(x_patched)
```

















=================== # ОБЩАЯ ИНФОРМАЦИЯ ==================
# АНАЛИЗ ПРОБЛЕМ

Главная находка: Channel 3 (OFI) имеет mean=5.51, std=5.97 после нормализации из-за cumsum по уровням — это полностью ломает обучение, т.к. Transformer фокусируется на аномальном канале и подавляет сигнал из остальных 5 каналов.

План содержит 6 подзадач в 3 файлах:

Замена cumsum на diff (delta-imbalance по времени)
Исправление fallback ветки OFI
Per-channel clipping [-5, 5] для всех каналов
Исправление _compute_channels_for_normalization
Диагностический warning при аномалиях
LayerNorm в layers.py перед attention pooling