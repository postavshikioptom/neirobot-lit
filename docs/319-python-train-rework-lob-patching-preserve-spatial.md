# Задача 319: Переработка LOB Patching — сохранение пространственной информации стакана

## Анализ: почему модель НЕ обучается

### Текущие показатели (из output.txt, Epoch 2-8)
| Метрика | Значение | Интерпретация |
|---------|----------|---------------|
| MCC | 0.03-0.05 | Практически случайный (0 = random) |
| Macro-F1 | 0.26-0.33 | Модель почти не различает классы |
| Hit Rate Flat | 80% | Модель учится предсказывать Flat (самый частый класс) |
| Hit Rate Up | 11% | Хуже случайного угадывания (33%) |
| Hit Rate Down | 15% | Хуже случайного угадывания (33%) |
| Directional Accuracy | 12.7% | Практически бесполезно |
| Conf Correct < Conf Wrong | -0.045 | Модель УВЕРЕНА в ОШИБКАХ |
| Corr с LOB Imbalance | 0.006 | Модель НЕ учится на данных стакана |
| Correlation with Signal | -0.019 | Обратная корреляция (!) |

### Корневая причина — LOBPatching уничтожает информацию

**Текущая архитектура патчинга** (`layers.py:27-115`):

```
Вход: (B, S, 13 каналов, 50 уровней) = 130,000 значений на сэмпл
  ↓ Транспонируем (B, S, 50, 13)
  ↓ Flatten (B*S, 1, 650)
  ↓ Conv1d(1, d_model=64, kernel=13, stride=13)  ← ЗДЕСЬ ТЕРЯЕТСЯ ВСЯ ИНФОРМАЦИЯ
  ↓ Результат: (B*S, 64, 50) — 50 скаляров по 64 измерения
  ↓ Attention Pooling (Linear 64→1, softmax, weighted sum) ← ВТОРОЕ СЖАТИЕ
  ↓ Результат: (B*S, 64) — ОДИН вектор на таймстеп
  ↓ Всего: (B, S, 64) = 64 числа на весь таймстеп

Итого: 130,000 значений → 64 числа (сжатие в 2000 раз!)
```

**Проблема Conv1d(kernel=13, stride=13)**:
- Сворачивает ВСЕ 13 каналов (MicropriceDev, Vol, Imb, OFI, VIB, Ret_10, Ret_50, Ret_100, Spread, DeltaImb, DeltaSpread, CumOFI, ImbAccel) в ОДИН скаляр на каждый уровень
- Каналы, которые несут РАЗНУЮ информацию (цены vs объемы vs моментум), смешиваются в одно число
- Модель физически не может различить "bid-side loaded" от "ask-side loaded" стакана

**Проблема Attention Pooling**:
- Сжимает 50 уровней в 1 вектор через learned weighted average
- Уничтожает пространственную структуру стакана (какие уровни загружены, где кластеры ликвидности)
- Трансформер НЕ видит отдельные уровни

**Результат**: Трансформер получает 100 токенов (по одному на таймстеп), каждый — сжатое представление всего стакана. Self-attention не может моделировать взаимодействие между уровнями стакана, потому что уровни уже "схлопнуты".

### Дополнительные проблемы

**Проблема 2: Шумные каналы**
- **CumOFI (канал 11)**: mean=-0.95, всегда отрицательный, клиппен [-2.2, -0.4]. После z-score нормализации почти нулевой информативности.
- **ImbAccel (канал 12)**: вторая производная imbalance, std=3.17. Экстремально зашумлён. После clamp [-5,5] до 95% обрезано.
- Оба канала вносят шум в обучение нормализатора и ухудшают SNR.

**Проблема 3: Недостаточная ёмкость модели**
- d_model=64, 2 слоя, ~85K параметров
- Мало для learning сложных паттернов в 13 каналах × 50 уровней

---

## Решение: Per-Level Token Architecture

**Ключевая идея**: каждый уровень стакана — отдельный токен. Трансформер видит 50 токенов на таймстеп и учит attention между уровнями.

**Новая архитектура**:
```
Вход: (B, S, 11 каналов, 50 уровней)
  ↓ Транспонируем (B, S, 50, 11)
  ↓ Linear(11, d_model=96) — проекция каждого уровня
  ↓ (B, S, 50, 96) — 50 токенов на таймстеп, каждый размерности 96
  ↓ Level PE (1, 1, 50, 96) + Temporal PE (1, S, 1, 96)
  ↓ Reshape (B, S*50, 96) = (B, 5000, 96)
  ↓ Transformer Encoder (3 слоя, nhead=6)
  ↓ GAP по всем 5000 токенам: (B, 96)
  ↓ Classifier Head → 3 класса
```

**Почему это работает**:
1. Self-attention на 5000 токенах позволяет моделировать зависимости между уровнями стакана
2. "Bid-side loaded" vs "ask-side loaded" стакан会产生 РАЗНЫЕ attention patterns
3. 5000 токенов вместо 100 — на порядок больше информации для трансформера
4. Flash Attention делает это вычислительно практичным

---

## Подзадачи

### Подзадача 319.1: Очистка шумных каналов в dataset.py

**Файл:** `python_lab/src/dataset.py`

**Проблема:** Канал 11 (CumOFI) и канал 12 (ImbAccel) вносят шум.

**Что менять:**

1. **Метод `_calculate_6_channels_raw`** (строка ~1375):
   - Удалить расчёт `cumofi_ch` (строки ~1488-1496) и `accel_imb_ch` (строки ~1498-1500)
   - Изменить return: убрать `cumofi_ch, accel_imb_ch` (строки ~1502-1504)
   - Теперь возвращается 11 тензоров вместо 13

2. **Метод `_process_sample`** (строка ~1564):
   - Обновить распаковку `_calculate_6_channels_raw` (строки ~1610-1614) — убрать `co_raw, ai_raw`
   - Удалить вызовы `normalize_channel(co_raw, 11)` и `normalize_channel(ai_raw, 12)` (строки ~1628-1629)
   - Обновить `torch.stack` (строки ~1632-1636) — оставить 11 каналов
   - Обновить channel_names диагностики (строка ~1650)

3. **Метод `_compute_channels_for_normalization`** (строка ~1506):
   - Обновить распаковку и конкатенацию (строки ~1553-1562)
   - Изменить schema: `range(650)` → `range(550)` (11 каналов × 50 = 550)

4. **Метод `_init_memory_mode`** (строка ~869):
   - Обновить diagnostic prints для 11 каналов

**Верификация:** Запустить `_calculate_6_channels_raw` на тестовых данных, убедиться что возвращается 11 тензоров формы (seq_len, 50).

---

### Подзадача 319.2: Переписание LOBPatching в layers.py

**Файл:** `python_lab/src/layers.py`

**Текущая проблема:**
- Conv1d(kernel=13, stride=13) сворачивает 13 каналов в 1 число
- Attention pooling сжимает 50 уровней в 1 вектор

**Новая архитектура `LOBPatching.__init__`:**

```python
def __init__(self, seq_len=100, n_levels=50, in_channels=11, d_model=96, activation='gelu_exact'):
    self.d_model = d_model
    self.in_channels = in_channels
    self.n_levels = n_levels
    self.seq_len = seq_len

    # Per-level projection: each level's channels → d_model
    self.level_proj = nn.Linear(in_channels, d_model)
    self.act = get_activation(activation)

    # Level Positional Embedding: (1, 1, 50, d_model)
    self.level_pos_emb = nn.Parameter(torch.randn(1, 1, n_levels, d_model) * 0.02)

    # Temporal Positional Embedding: (1, S, 1, d_model)
    self.time_pos_emb = nn.Parameter(torch.randn(1, seq_len, 1, d_model) * 0.02)

    # Финальная нормализация
    self.norm = nn.LayerNorm(d_model)

    # УДАЛЯЕМ: patch_conv, level_attention, pre_attn_norm, num_features, num_patches
```

**Новая архитектура `LOBPatching.forward`:**

```python
def forward(self, x):
    # x: (B, S, C, L) — C=11, L=50
    b, s, c, l = x.shape

    # (B, S, C, L) → (B, S, L, C) — each level's features together
    x_perm = x.transpose(2, 3).contiguous()  # (B, S, 50, 11)

    # Per-level projection
    x_flat = x_perm.view(b * s, l, c)  # (B*S, 50, 11)
    x_proj = self.level_proj(x_flat)    # (B*S, 50, d_model)
    x_proj = self.act(x_proj)

    # Reshape back: (B, S, 50, d_model)
    x_tokens = x_proj.view(b, s, l, self.d_model)

    # Add positional embeddings (broadcasting)
    x_tokens = x_tokens + self.level_pos_emb + self.time_pos_emb

    # Flatten for transformer: (B, S*50, d_model)
    x_out = x_tokens.view(b, s * l, self.d_model)

    return self.norm(x_out)
```

**Что удалить:**
- `self.patch_conv` (Conv1d)
- `self.level_attention` (Linear)
- `self.pre_attn_norm` (LayerNorm)
- `self.num_features`, `self.num_patches`
- Блок attention pooling в forward

**Верификация:** `dummy = torch.randn(2, 100, 11, 50)` → выход `(2, 5000, d_model)`.

---

### Подзадача 319.3: Адаптация LiTModel в lit_model.py

**Файл:** `python_lab/src/lit_model.py`

**Что менять:**

1. **Константы и конфиг** (строки ~8-33):
   - `DEFAULT_INPUT_CHANNELS = 7` → `11`
   - `LiTConfig.in_channels = 13` → `11`
   - `LiTConfig.seq_len = 200` → `100` (теперь 5000 токенов = 100×50, достаточно)
   - `LiTConfig.d_model = 64` → `96` (больше capacity для level embeddings)
   - `LiTConfig.num_layers = 2` → `3` (больше слоёв для attention между уровнями)

2. **`__init__` модели** (строка ~282):
   - Обновить defaults: `seq_len=100, in_channels=11, d_model=96, num_layers=3`
   - Обновить `nhead` логику: `d_model % nhead == 0` (96 % 6 == 0 ✓)

3. **`forward` метод** (строка ~438):
   - После `x = self.patching(x)` результат: `(B, S*50, d_model)`
   - **Убрать CLS токен** (строки ~471-473) — не нужен при 5000 токенах
   - Убрать добавление PE отдельным шагом (строки ~454-457) — PE теперь внутри patching
   - **Пуллинг** (строка ~490): просто GAP по всем токенам:
     ```python
     pooled = x_trans.mean(dim=1)  # (B, d_model) — GAP по 5000 токенам
     pooled = self.norm(pooled)
     ```
   - Bottleneck → classifier и vol_regressor (строки ~493-531) — без изменений

4. **Тестовый код `__main__`** (строки ~534-582):
   - Обновить для новых размерностей: `(8, 100, 11, 50)`

**Верификация:** Запустить `__main__` тест: `dummy = torch.randn(8, 100, 11, 50)` → выход `(8, 3)` для классификации.

---

### Подзадача 319.4: Обновление train.py

**Файл:** `python_lab/src/train.py`

**Что менять:**

1. **Channel names в diagnostics** (строка ~390):
   - Обновить `channel_names` список: убрать "CumOFI" и "ImbAccel"

2. **Создание модели** (строки ~1183-1194, ~1296-1320):
   - `in_channels` параметр уже передаётся из args — убедиться что default = 11

3. **Argparse defaults** (найти где задаются):
   - `--seq_len` default → 100 (если ещё 200)
   - `--d_model` default → 96 (если ещё 64)
   - `--num_layers` default → 3 (если ещё 2)

4. **ONNX export dummy_input** (estructor ~2998):
   - Shape: `(1, args.seq_len, 11, 50)` — 11 вместо 13

5. **Print-сообщения**: обновить упоминания "13 каналов" → "11 каналов"

**Верификация:** `python train.py --help` — все параметры корректны.

---

### Подзадача 319.5: Обновление Normalizer и экспортных скриптов

**Файлы:** `python_lab/src/normalization.py`, `python_lab/src/export_onnx.py`

**Что менять:**

1. **normalization.py** — существенных изменений не нужно (работает с feat_ колонками динамически)
   - Но убедиться что после удаления 2 каналов нормализатор корректно фитится на 550 feat_ колонках

2. **export_onnx.py** — обновить входной shape:
   - `(1, seq_len, 11, 50)` вместо `(1, seq_len, 13, 50)`

3. **Документация**: обновить `LiTConfig` metadata.json экспорт (размер входа 11×50)

---

### Подзадача 319.6: Запуск обучения и сравнение метрик

**Что делать:**

1. Запустить обучение:
   ```bash
   python -m python_lab.src.train --symbol CAKEUSDT --mode train \
     --seq_len 100 --d_model 96 --nhead 6 --num_layers 3 \
     --epochs 30 --batch_size 64
   ```

2. Сравнить с baseline (задача 318):
   | Метрика | Baseline (318) | Ожидание (319) |
   |---------|----------------|-----------------|
   | MCC | 0.03-0.05 | >0.10 |
   | Dir. Accuracy | 12.7% | >30% |
   | Corr(Imbalance) | 0.006 | >0.05 |
   | Conf Correct > Wrong | -0.045 | >0 |

3. **GPU memory check**: 5000 токенов × 96 = 480K значений/сэмпл. При batch=64: ~30M float32 = 120MB. OK для 8GB+ GPU.

4. **Fallback plan** (если OOM или медленно):
   - Уменьшить seq_len до 50 (2500 токенов)
   - Уменьшить batch_size
   - Добавить gradient checkpointing

5. Записать результаты в лог, обновить `000-tasks_list.md`

---

## Сводка изменений

| Файл | Изменение |
|------|-----------|
| `dataset.py` | Убрать CumOFI и ImbAccel (каналы 11, 12). in_channels 13→11. Schema 650→550. `compute_ofi_from_lob`: убрать cumsum, вернуть per-tick OFI дельты. Обновить все комментарии "13 каналов" → "11". |
| `layers.py` | Conv1d(13,13)+attention pool → Linear(11,d_model). Level+Temporal PE. Выход (B, S*50, d). |
| `lit_model.py` | in_channels 13→11, seq_len 200→100, d_model 64→96, num_layers 2→3. Убрать CLS. GAP по 5000 токенам. Обновить тестовый block и все константы. |
| `train.py` | in_channels 13→11, seq_len→100, d_model→96, num_layers→3. Обновить prints, channel_names, ONNX shape, argparse defaults. |
| `normalization.py` | Без изменений (динамический fit). |
| `export_onnx.py` | in_channels 9→11. Shape 450→550. Обновить metadata и comments (Задача 317→319). |

## Ожидаемый эффект

- Трансформер видит **50 отдельных токенов** (уровней стакана) на каждый таймстеп
- Self-attention моделирует зависимости между уровнями: "bid loaded vs ask loaded"
- **5000 токенов** (100 × 50) вместо 100 — на порядок больше информации
- Удаление шумных каналов улучшает SNR и стабильность нормализации
- Увеличение capacity (d_model=96, 3 слоя) даёт модели больше power для learning

### Подзадача 319.7: Переписать OFI на стандартный Cont-Kukanov-Stoikov (некумулятивный)

**Файл:** `python_lab/src/dataset.py`

**Проблема:** Текущая реализация OFI в `compute_ofi_from_lob` (строка 132) возвращает `np.cumsum(delta)` — кумулятивную сумму. Канал 3 (OFI) использует эту кумулятивную величину. Конкуренты (Kolm et al. 2023, LiT, TLOB, DeepLOB) используют **некумулятивный** event-based OFI или его агрегацию за окно.

**Почему CumOFI неверен для краткосрочного предсказания:**
- Кумулятивная сумма OFI превращает стационарный (анти-коррелированный) процесс в нестационарный с дрейфом
- Hurst exponent OFI < 0.5 (анти-персистентность), кумулятивная сумма создаёт "случайное блуждание"
- Модель видит "падающий/растущий" ряд, который коррелирует с ценой тривиально, но не несёт прогнозной информации о направлении следующего шага
- Kolm et al. (2023) явно используют некумулятивный OFI и находят его лучшим

**Что менять:**

1. **Функция `compute_ofi_from_lob`** (строки 95-132):
   - Убрать `np.cumsum(delta)` — вернуть **дельты** (сырой OFI за каждый таймстеп):
     ```python
     # БЫЛО:
     ofi = np.cumsum(delta).astype(np.float32)
     # СТАЛО:
     ofi = delta.astype(np.float32)  # некумулятивный, per-tick OFI
     ```
   - Обновить docstring: убрать "cumulative running sum", указать "per-tick OFI deltas"

2. **Канал 3 (OFI) в `_calculate_6_channels_raw`** (строки 1423-1427):
   - Сейчас: `ofi_raw = ofi_precomputed.unsqueeze(-1).expand(-1, 50)` — берёт кумулятивный OFI
   - После исправления `compute_ofi_from_lob` это автоматически станет дельтой
   - Fallback `torch.diff(imb_ch_raw)` — оставить как есть (это и есть дельта imbalance)

3. **Удалить канал 11 (CumOFI)** — уже удаляется в Подзадаче 319.1

4. **Диагностика в `_init_memory_mode`** (строка ~869):
   - Обновить print для OFI: указать что это raw (per-tick) OFI, не cumulative

**Ожидаемый формат OFI (Cont-Kukanov-Stoikov, per-tick):**
```
e_n = I{P^B_n >= P^B_{n-1}} * q^B_n
    - I{P^B_n <= P^B_{n-1}} * q^B_{n-1}
    - I{P^A_n <= P^A_{n-1}} * q^A_n
    + I{P^A_n >= P^A_{n-1}} * q^A_{n-1}
```
Текущая реализация `compute_ofi_from_lob` уже считает delta по этой формуле через `np.where` (строки 123-131) — просто убрать `cumsum`.

**Источники:**
- Cont, Kukanov & Stoikov (2014): "The Price Impact of Order Book Events"
- Kolm, Turiel & Westray (2023): "Deep Order Flow Imbalance" — OFI (non-cumulative) best performer
- Li et al. (2021): "Generalized Order Flow Imbalance" (arXiv:2112.02947)

**Верификация:** После изменений `compute_ofi_from_lob` должен возвращать массив с нулевым средним (или близким к 0), std>0, без тренда. Проверить: `ofi.mean() ≈ 0`, `abs(ofi.std()) > 0`, нет тренда (ADF test или визуально).

---

### Подзадача 319.8: Заменить все захардкоженные "13 каналов" → "11 каналов" в коде

**Проблема:** После удаления CumOFI и ImbAccel (Подзадача 319.1) все константы, комментарии и print'ы со значением 13 каналов become неверными. Ошибки в одном месте приведут к shape mismatch и краху при обучении.

**Что искать и менять:**

| Файл | Строка | Было | Стало |
|------|--------|------|-------|
| `lit_model.py` | 21 | `in_channels: int = 13` + comment | `in_channels: int = 11` + comment без CumOFI/ImbAccel |
| `lit_model.py` | 547 | `"Testing with 13 channels"` | `"Testing with 11 channels"` |
| `lit_model.py` | 548 | `LiTModel(..., in_channels=13, ...)` | `LiTModel(..., in_channels=11, ...)` |
| `lit_model.py` | 549 | `torch.randn(8, 200, 13, 50)` | `torch.randn(8, 100, 11, 50)` |
| `dataset.py` | 1377 | docstring `"13 каналов"` | `"11 каналов"` |
| `dataset.py` | 1552 | comment `"13 каналов"` | `"11 каналов"` |
| `dataset.py` | 1610 | comment `"13 каналов"` | `"11 каналов"` |
| `dataset.py` | 1616 | comment `"13 каналов"` | `"11 каналов"` |
| `dataset.py` | 1631 | `shape (Seq, 13, 50)` | `shape (Seq, 11, 50)` |
| `dataset.py` | 1562 | `range(650)` (13×50) | `range(550)` (11×50) |
| `train.py` | 1550 | comment `"13 каналов"` | `"11 каналов"` |
| `train.py` | 1551 | `in_channels = 13` | `in_channels = 11` |
| `train.py` | 1554 | `print("Total input channels: 13...")` | `print("Total input channels: 11...")` |

**Где искать дополнительно:**
- `channel_names` списки (уже обновляются в 319.1 и 319.4, но перепроверить)
- `schema=[f"feat_{i}" for i in range(650)]` → `range(550)` (в `_compute_channels_for_normalization`)
- ONNX dummy input shape `(1, seq_len, in_channels, 50)` — должен автоматически подставиться из args, но перепроверить
- `export_onnx.py` — `in_channels=9` (Задача 317) → `11` (Задача 319), shape `(B, S, 450)` → `(B, S, 550)`

**Верификация:**
```bash
# Поиск оставшихся "13" в контексте каналов
grep -rn "13" python_lab/src/*.py | grep -i "channel\|канал\|in_channels"
# Должен вернуть 0 совпадений (кроме нумерации строк и несвязанных значений)
```

---

## Risks и Mitigations

| Risk | Mitigation |
|------|------------|
| OOM при 5000 токенов | Уменьшить seq_len до 50, batch_size, или добавить gradient checkpointing |
| Медленное обучение | Flash Attention (SDPA уже используется) компенсирует рост токенов |
| MCC не вырастет | Проверить quality of features, рассмотреть additional feature engineering |
| Нестабильная сходимость | Увеличить warmup, уменьшить lr (попробовать 5e-5 вместо 1e-4) |
