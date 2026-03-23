# Задача 321: убрать архитектурный регресс скорости после задачи 319

## Диагностика проблемы

После задачи 319 обучение `python_lab/src/train.py` не зависает, а продолжает идти с крайне низким throughput. `output.txt` показывает нормальный progress по batch, например около `0.26 it/s`, без тотального развала по `NaN/Inf`, но с оценкой порядка `~4 часов на эпоху`. Это значит, что проблема не в hang между `train/validation`, а в том, что сам train loop стал вычислительно слишком дорогим.

Подтвержденные факты по текущему коду:

- В `python_lab/src/layers.py` `LOBPatching` сейчас делает `per-level projection`, добавляет `level_pos_emb` и `time_pos_emb`, затем разворачивает данные в `(B, S*50, d_model)`, добавляет `CLS` и возвращает `1 + S*50` токенов.
- В `python_lab/src/lit_model.py` `forward()` ожидает именно этот формат, расширяет `mask` на все 50 уровней каждого timestep и прогоняет full self-attention по всей последовательности `1 + S*50`, после чего берёт `CLS`.
- До задачи 319 логика была компактной: `LOBPatching` сжимал глубину стакана в snapshot token на timestep через `patch_conv + level_attention` и возвращал `(B, S, d_model)`.
- В `python_lab/src/train.py` по умолчанию выставлен `num_workers=0`, а в train loop добавлено много print-диагностики: `[BATCH ...]`, `[DIAG ...]`, `[VOL_DIAG ...]`, `[VOL_LOSS_DIAG ...]`.

## Главная причина

Главный `root cause` — архитектурный регресс после задачи 319.

Сейчас модель перестала быть "light" в части tokenization: вместо компактного snapshot token на timestep она превращает каждый из 50 уровней стакана в отдельный attention token. При `seq_len=100` это даёт `1 + 100*50 = 5001` токен. Дальше `python_lab/src/lit_model.py` прогоняет по ним обычный full self-attention, у которого стоимость растёт квадратично по длине последовательности. Практически это превращает LiT из компактной LOB-модели в дорогую `O((S*L)^2)`-модель.

Для скальпинга и LOB-задач это архитектурно избыточно: SOTA-подходы обычно не держат все 50 уровней как равноправные `full-attention` токены на всей временной оси, а делают `compress/pool/factorize depth`. Поэтому основной путь исправления должен быть не в `axial-attention` как первом решении, а в возврате к компактному `snapshot-style patching` с текущими 11 каналами.

---

## План исправления

### 321.1. Вернуть компактный `snapshot-style patching` в `python_lab/src/layers.py`

**Файл:** `python_lab/src/layers.py`

**Текущее поведение:**

- `LOBPatching.__init__()` использует `self.level_proj`, `self.level_pos_emb`, `self.time_pos_emb`, `self.cls_token`.
- `LOBPatching.forward()` делает:
  - `(B, S, C, L) -> (B, S, L, C)`
  - `Linear(C -> d_model)` для каждого уровня
  - `view(B, S*L, d_model)`
  - `cat([CLS, tokens], dim=1)`
  - возвращает `(B, 1 + S*50, d_model)`

**Что заменить:**

Вернуть логику, близкую к `pre-319`, но оставить текущие `11` каналов:

1. В `__init__()` убрать:
   - `self.level_proj`
   - `self.cls_token`
   - зависимость patching от возврата `1 + S*50` токенов

2. В `__init__()` вернуть или создать:
   - `self.num_features = in_channels * n_levels`
   - `self.num_patches = n_levels`
   - `self.patch_conv = nn.Conv1d(1, d_model, kernel_size=in_channels, stride=in_channels)`
   - `self.level_pos_emb = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)`
   - `self.time_pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)`
   - `self.level_attention = nn.Linear(d_model, 1)`
   - `self.pre_attn_norm = nn.LayerNorm(d_model)`
   - `self.norm = nn.LayerNorm(d_model)`

3. В `forward()` заменить текущий `per-level token path` на `compact snapshot path`:
   - `(B, S, C, L) -> (B, S, L, C)`
   - `view(B, S, 1, L*C)`
   - `view(B*S, 1, L*C)`
   - `patch_conv` по блокам уровня
   - `permute -> (B, S, L, d_model)`
   - `+ level_pos_emb`
   - `pre_attn_norm`
   - `level_attention + softmax` по dimension уровня
   - `weighted sum` по глубине -> `(B, S, d_model)`
   - `+ time_pos_emb[:, :S, :]`
   - `norm`
   - вернуть `(B, S, d_model)`

**Целевой код-ориентир:**

```python
self.patch_conv = nn.Conv1d(1, d_model, kernel_size=in_channels, stride=in_channels)
self.level_attention = nn.Linear(d_model, 1)
self.pre_attn_norm = nn.LayerNorm(d_model)
```

```python
x_permuted = x.transpose(2, 3).contiguous()          # (B, S, L, C)
x_flat_seq = x_permuted.view(b, s, 1, l * c)        # (B, S, 1, L*C)
x_flat = x_flat_seq.view(b * s, 1, l * c)           # (B*S, 1, L*C)

x_patched = self.patch_conv(x_flat)                 # (B*S, d_model, L)
x_patched = self.act(x_patched).permute(0, 2, 1)    # (B*S, L, d_model)
x_patched = x_patched.view(b, s, self.num_patches, self.d_model)

x_patched = x_patched + self.level_pos_emb
x_patched_norm = self.pre_attn_norm(x_patched)
attn_weights = F.softmax(self.level_attention(x_patched_norm), dim=2)
x_snapshot = (x_patched_norm * attn_weights).sum(dim=2)   # (B, S, d_model)

x_temporal = x_snapshot + self.time_pos_emb[:, :s, :]
return self.norm(x_temporal)
```

**Что должно получиться после правки:**

- `patching` снова возвращает `(B, S, d_model)`, а не `(B, 1 + S*50, d_model)`
- один timestep = один компактный `snapshot token`
- глубина стакана остаётся в модели, но уже не через full attention по всем 50 уровням

---

### 321.2. Адаптировать `python_lab/src/lit_model.py` под `compact snapshot tokens`

**Файл:** `python_lab/src/lit_model.py`

**Текущее поведение:**

- `x = self.patching(x)` возвращает `(B, 1 + S*50, d_model)`
- `mask` разворачивается на `self.n_levels`:
  - `mask.unsqueeze(-1).expand(-1, -1, self.n_levels)`
  - затем `flatten` в `(B, S*50)`
- `transformer` работает по всем `1 + S*50` токенам
- `pooling` делается через `x_trans[:, 0, :]`

**Что заменить:**

1. После правки `layers.py` считать, что:

```python
x = self.patching(x)   # (B, S, d_model)
```

2. Убрать расширение `mask` на 50 уровней.
   Вместо текущей логики:

```python
mask_expanded = mask.unsqueeze(-1).expand(-1, -1, self.n_levels)
mask_flat = mask_expanded.reshape(b, -1)
```

использовать `mask` напрямую:

```python
src_key_padding_mask = mask
```

3. Убрать предположение, что `CLS` приходит из patching.
   Основной рекомендованный вариант: не возвращать `CLS` из patching вообще и не гонять лишний токен через всю модель.

4. `Pooling` заменить на обычный `snapshot-style pooling` по времени:

```python
pooled = self.norm(x_trans.mean(dim=1))
```

5. Если потребуется оставить `CLS`, делать это только на уровне `LiTModel`, а не на уровне `LOBPatching`. Но это optional; основной путь для задачи 321 — максимально упростить и удешевить тракт.

**Целевой код-ориентир:**

```python
x = self.patching(x)  # (B, S, d_model)

if self.regime_embedding is not None and regime_id is not None:
    regime_emb = self.regime_embedding(regime_id).unsqueeze(1).expand(-1, x.shape[1], -1)
    x = torch.cat([x, regime_emb], dim=-1)
    x = self.regime_projection(x)

src_key_padding_mask = mask if mask is not None else None

x_trans = x
for layer in self.transformer_layers:
    x_trans = layer(x_trans, src_key_padding_mask=src_key_padding_mask)

pooled = self.norm(x_trans.mean(dim=1))
```

**Что должно получиться после правки:**

- `transformer` снова работает по `S` токенам, а не по `S*50`
- убирается квадратичная цена attention по глубине стакана
- модель остаётся совместимой с текущими `11` каналами и `multi-task head`

---

### 321.3. Снизить вторичный `runtime overhead` в `python_lab/src/train.py`

**Файл:** `python_lab/src/train.py`

#### 321.3.1. Вернуть рабочие `DataLoader defaults`

**Текущее:**

- `--num_workers` default = `0`
- `--persistent_workers` default = `False`
- `--prefetch_factor` default = `2`

**Что заменить:**

- `--num_workers` default: `4`
- `--persistent_workers` default: `True`
- `--prefetch_factor` оставить `2`

`build_dataloader_kwargs()` уже написан правильно: `persistent_workers` и `prefetch_factor` добавляются только если `num_workers > 0`. Эту логику не менять.

**Конкретная замена:**

```python
# было
parser.add_argument("--num_workers", type=int, default=0, ...)
parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=False, ...)
parser.add_argument("--prefetch_factor", type=int, default=2, ...)

# должно стать
parser.add_argument("--num_workers", type=int, default=4, ...)
parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True, ...)
parser.add_argument("--prefetch_factor", type=int, default=2, ...)
```

#### 321.3.2. Убрать постоянный `debug spam` из `train loop`

**Сейчас в `training_step()`:**

- каждые 100 batch печатаются `[BATCH ...]` и `[DIAG ...]`
- до `batch_idx <= 300` печатаются `[VOL_DIAG ...]` и `[VOL_LOSS_DIAG ...]`
- это полезно для аварийной диагностики, но не должно быть активным по умолчанию в штатном training run

**Что заменить:**

1. Добавить CLI-флаги:

```python
parser.add_argument("--train_batch_log_interval", type=int, default=0, ...)
parser.add_argument("--enable_vol_debug", action=argparse.BooleanOptionalAction, default=False, ...)
```

2. Блок `[BATCH]/[DIAG]` заменить с:

```python
if batch_idx % 100 == 0:
```

на:

```python
train_batch_log_interval = self.hparams.get("train_batch_log_interval", 0)
if train_batch_log_interval > 0 and batch_idx % train_batch_log_interval == 0:
```

3. Блок `[VOL_DIAG]` и `[VOL_LOSS_DIAG]` заменить с безусловно-диагностического режима на `opt-in`:

```python
if self.hparams.get("enable_vol_debug", False) and batch_idx <= 300 and batch_idx % 50 == 0:
```

4. Оставить:
   - `[TRAIN] Epoch started/completed`
   - короткие `validation phase logs` из задачи 320
   - `critical warning` only when error really happens

#### 321.3.3. Перевести полный `gradient finite scan` в `debug-only` режим

**Сейчас в `on_before_optimizer_step()`:**
на каждом optimizer step идёт проход по `self.named_parameters()` и `torch.isfinite(param.grad).all()` для каждого параметра.

Это не главная причина 4 часов на эпоху, но это лишний `per-step overhead` и потенциальные `GPU sync`.

**Что заменить:**

- добавить `--grad_finite_check_interval` default `0`
- запускать полный scan только если `interval > 0` и `self.global_step % interval == 0`
- по умолчанию этот scan выключить

**Целевой код-ориентир:**

```python
grad_check_interval = self.hparams.get("grad_finite_check_interval", 0)
if grad_check_interval <= 0 or self.global_step % grad_check_interval != 0:
    return
```

---

### 321.4. Сделать быстрый `smoke-run` и `A/B` после фикса архитектуры

**Файл:** `python_lab/src/train.py`

Чтобы не ждать полную эпоху, добавить `trainer knobs` для короткого `smoke-run`:

```python
parser.add_argument("--limit_train_batches", type=int, default=0, ...)
parser.add_argument("--limit_val_batches", type=int, default=0, ...)
```

И в `pl.Trainer(...)` пробросить:

```python
limit_train_batches=args.limit_train_batches if args.limit_train_batches > 0 else 1.0,
limit_val_batches=args.limit_val_batches if args.limit_val_batches > 0 else 1.0,
```

**Зачем:**

- быстро проверить throughput после возврата к `compact patching`
- не гонять полную эпоху, если regression всё ещё остался

---

## Что именно должно измениться по результату задачи 321

1. `LOBPatching` больше не создаёт `1 + seq_len*50 attention tokens`.
2. `LiTModel.forward()` больше не разворачивает `mask` на все уровни стакана.
3. `transformer` снова работает по компактной temporal sequence `(B, S, d_model)`.
4. `train.py` по умолчанию снова использует многопроцессную загрузку данных.
5. `runtime debug-логика` перестаёт быть `always-on`.

---

## Верификация

### 1. Shape verification

После правок проверить вручную:

- `LOBPatching(dummy)`:
  - вход: `(2, 100, 11, 50)`
  - выход: `(2, 100, d_model)`

- `LiTModel(dummy)`:
  - вход: `(2, 100, 11, 50)`
  - `logits`: `(2, 3)` или `(2, H, 3)` для `multi-horizon`
  - `vol_pred`: `(2, 1)`

### 2. Smoke-run на 100-200 train batches

Запуск после правок:

```bash
python -m python_lab.src.train \
  --symbol FARTCOINUSDT \
  --epochs 1 \
  --limit_train_batches 200 \
  --limit_val_batches 50 \
  --num_workers 4 \
  --persistent_workers \
  --prefetch_factor 2
```

**Ожидание:**

- training идёт заметно быстрее, чем `~0.26 it/s`
- в первые 100-200 batch не видно оценки `~4 часа на эпоху`
- нет ощущения hang, есть нормальный `batch progress`

### 3. `A/B` без `gradient checkpointing`

Сделать два коротких прогона на одинаковом `smoke-run`:

- Run A: без `--use_gradient_checkpointing`
- Run B: c `--use_gradient_checkpointing`

**Ожидание:**

- обычный режим без `checkpointing` должен быть быстрее
- `checkpointing` оставлять только как аварийный `memory fallback`, а не как дефолтный путь `speed run`

### 4. Проверка логов

После фикса в логе должно быть:

- минимум служебного `spam`
- без постоянных `[VOL_DIAG]` / `[VOL_LOSS_DIAG]` по умолчанию
- только `epoch-level` и `phase-level` сообщения
- `critical warnings` только по реальной ошибке

---

## Optional alternatives (не основной путь задачи 321)

Если после возврата к `compact snapshot-style patching` скорость всё ещё окажется недостаточной, рассматривать уже отдельной задачей, а не внутри 321:

1. `factorized/axial attention`: отдельно по времени и отдельно по `depth`
2. `top-k depth compression`: держать отдельно только первые `N` уровней, остальные предварительно агрегировать
3. `dual-attention path` в стиле `TLOB`: spatial compression + temporal attention
4. `pooled depth groups`: 50 уровней агрегировать в 5-10 `depth bins` до `transformer`

Для задачи 321 это optional. Основной путь — вернуть компактную `snapshot-style` схему, максимально близкую к `pre-319` идее, но оставить текущие `11` каналов.

---

## Подтверждения и ориентиры

- Локальный факт проекта: `output.txt` показывает, что обучение не висит, а идёт с throughput порядка `~0.26 it/s`.
- `PyTorch SDPA`: https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html
- `PyTorch checkpointing overhead`: https://docs.pytorch.org/docs/stable/checkpoint.html
- `LiT paper`: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1616485/full
- `TransLOB code`: https://github.com/jwallbridge/translob
- `TLOB code/paper`: https://github.com/LeonardoBerti00/TLOB
