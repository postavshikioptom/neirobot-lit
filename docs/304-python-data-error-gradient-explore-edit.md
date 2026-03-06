# Задача 304: Исправление ошибки разрыва градиента - неправильный порядок загрузки столбцов данных parquet - теперь загружаем по именам 202 столбца.

# ПЛАН РЕАЛИЗАЦИИ

Признаю аргументы Grok — **совместимость с Rust и кодом аугментации (Symmetric Flip/Jitter) является приоритетом.** Если код аугментации в `dataset.py` уже написан под **interleaved** (чередующийся) порядок, то мой предыдущий план со сгруппированным порядком действительно всё бы сломал.

Я принимаю условия: **Interleaved порядок + выбор по именам.** Это решит проблему "битых" данных и сохранит работоспособность Rust-инференса.


### ФИНАЛЬНЫЙ ПЛАН (Версия 2.0 — Согласованная)

#### 1. Файл: `python_lab/src/features.py` (Группировка по стороне)
Это единственный способ сделать так, чтобы слайсы `0:50` в `dataset.py` работали, а блоки `0:100` (Ask) и `100:200` (Bid) оставались целыми для Symmetric Flip.

```python
# В конце метода transform:
ordered_feat_cols = []
# Блок ASK (индексы 0-99)
ordered_feat_cols.extend([f"feat_ask_p_{i}" for i in range(50)]) # 0-49: только цены
ordered_feat_cols.extend([f"feat_ask_v_{i}" for i in range(50)]) # 50-99: только объемы

# Блок BID (индексы 100-199)
ordered_feat_cols.extend([f"feat_bid_p_{i}" for i in range(50)]) # 100-149: только цены
ordered_feat_cols.extend([f"feat_bid_v_{i}" for i in range(50)]) # 150-199: только объемы

# Все остальное (past_returns, metadata)
meta_cols = ["timestamp_ms", "mid_price", "last_update_id"]
other_cols = [c for c in df.columns if c.startswith("feat_past_ret_")]

# Формируем итоговый DF: Признаки LOB всегда ПЕРВЫЕ 200 колонок
return df.select(
    [pl.col(c) for c in ordered_feat_cols] + 
    [pl.col(c) for c in meta_cols if c in df.columns] +
    [pl.col(c) for c in other_cols]
)
```

#### 2. Файл: `python_lab/src/dataset.py` (Минимальные правки)
Благодаря порядку выше, старый код слайсинга теперь будет получать **чистые** данные (только цены или только объемы).

```python
# В _process_sample (оставляем как было, теперь это будет работать):
lob_flat = x_raw[:, :200]
ask_p, ask_v = lob_flat[:, 0:50], lob_flat[:, 50:100]    # 0-49 теперь только цены Ask!
bid_p, bid_v = lob_flat[:, 100:150], lob_flat[:, 150:200] # 100-149 только цены Bid!

# ВАЖНО: правка в _init_memory_mode / _init_streaming_mode
# Нужно гарантировать, что feat_cols берется именно в этом порядке
self.feat_cols = [f"feat_ask_p_{i}" for i in range(50)] + \
                 [f"feat_ask_v_{i}" for i in range(50)] + \
                 [f"feat_bid_p_{i}" for i in range(50)] + \
                 [f"feat_bid_v_{i}" for i in range(50)]
```

#### 3. Файл: `python_lab/src/train.py` (Защита)
Настаиваю на этих пунктах для предотвращения "тихого" взрыва:
1.  **Gradient Clipping**: В `L.Trainer` добавить `gradient_clip_val=0.5`.
2.  **Sanity Check**:
```python
# Перед trainer.fit()
batch = next(iter(train_loader))
x = batch[0] # [B, 3, 50]
if not torch.isfinite(x).all():
    raise ValueError("NaN detected in input features before training!")
```

### Почему это сработает и ничего не сломает:
1.  **Symmetric Flip**: Он меняет местами блоки `x[:, 0:100]` и `x[:, 100:200]`. В моем плане `0:100` — это весь Ask (P и V), а `100:200` — это весь Bid. Смена блоков местами — это корректный "флип" стакана.
2.  **Volume Jitter**: Он применяется к объемам. Теперь мы точно знаем, что объемы — это индексы `50:100` и `150:200`.
3.  **Rust**: Инференс в Rust (ONNX) просто ожидает 200 чисел. Ему не важно, interleaved они или grouped, главное, чтобы порядок в `features.py` (при обучении) и в Rust (при работе) был **одинаковым**. Мы зафиксируем этот порядок в `ordered_feat_cols`.

**Мой вердикт**: План Grok с interleaved-порядком и старыми слайсами — это гарантированный NaN. Мой план с "группировкой по стороне" — единственный способ спасти текущий код `dataset.py` и обучить модель.





===================
# Задача 304-2 - Исправление ошибок NaN
Я завершил исправления.

lit_model.py: Убрана двойная активация self.bottleneck_act перед классификатором.
train.py: Исправлена формула Multi-Task Loss (добавлен коэффициент 0.5 перед log_var).
layers.py: В LOBPatching добавлена перестановка осей transpose(2, 3) перед view, чтобы свертка правильно объединяла каналы одного уровня.





========================== ОБЩАЯ ИНФОРМАЦИЯ О ПРОБЛЕМЕ (ДИАГНОСТИКА) ===========

# ПРОБЛЕМА ОШИБКА
Epoch 0 (0m 0s) Validation: MCC=0.1317, Macro-F1=0.2866, ECE=0.2344, MCE=0.7388
✓ Reliability diagram saved to: /kaggle/reports/FARTCOINUSDT/reliability_diagram_epoch_0.png
  Vol-MSE=117.543922, Vol-MAE=10.810051

Metrics by Market Regime:
  Regime 0: MCC=0.1317, F1=0.2866, Samples=256
✓ Logged 1000 embeddings to TensorBoard Projector
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/loops/fit_loop.py:317: The number of training batches (3785) is smaller than the logging interval Trainer(log_every_n_steps=10000). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
⚠️  CRITICAL WARNING: NaN or Inf detected in model logits during validation!
   This indicates extreme numerical instability (exploding gradients).
   Metrics and visualizations for this epoch will be unreliable.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


WARNING: CalibrationMetrics.calculate: logits contain NaN or Inf. Skipping calculation.

Epoch 0 (4m 30s) Validation: MCC=0.0000, Macro-F1=0.2723, ECE=0.0000, MCE=0.0000
/kaggle/working/src/utils.py:372: UserWarning: Data has no positive values, and therefore cannot be log-scaled.
  ax2.set_yscale('log')
✓ Reliability diagram saved to: /kaggle/reports/FARTCOINUSDT/reliability_diagram_epoch_0.png

WARNING: plot_pr_curves_tensorboard: NaN in y_pred_probs. Skipping plot pr_curves.
  Vol-MSE=nan, Vol-MAE=nan

Metrics by Market Regime:
  Regime 0: MCC=0.0000, F1=0.2723, Samples=103797
warning: Embedding dir exists, did you set global_step for add_embedding()?
✓ Logged 1000 embeddings to TensorBoard Projector

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
⚠️  CRITICAL WARNING: NaN or Inf detected in model logits during validation!
   This indicates extreme numerical instability (exploding gradients).
   Metrics and visualizations for this epoch will be unreliable.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


WARNING: CalibrationMetrics.calculate: logits contain NaN or Inf. Skipping calculation.

Epoch 1 (4m 12s) Validation: MCC=0.0000, Macro-F1=0.2723, ECE=0.0000, MCE=0.0000
  Vol-MSE=nan, Vol-MAE=nan

Metrics by Market Regime:
  Regime 0: MCC=0.0000, F1=0.2723, Samples=103797
==================================================


# АНАЛИЗ от GROK-1:

**Анализ ошибки (только критические проблемы, без метрик качества):**

Критическая ошибка — **NaN/Inf в logits и vol_pred** появляется уже на валидации после первой эпохи обучения (в on_validation_epoch_end → compute_metrics + CalibrationMetrics). Первая валидация (до обучения) проходит нормально (MCC 0.1317), вторая — полностью ломается. Это классический **exploding gradients** (градиенты взрываются во время первого backward + optimizer.step).

**Точная причина (найдена по коду и логам):**
- В `dataset.py` → `_process_sample` (строки ~350-370) жёстко нарезают `lob_flat = x_raw[:, :200]` так:
  - `ask_p, ask_v = x[:, 0:50], x[:, 50:100]`
  - `bid_p, bid_v = x[:, 100:150], x[:, 150:200]`
- Но в `features.py` (transform) колонки формируются **группированно**:
  - 0-49: feat_ask_p_*
  - 50-99: feat_bid_p_*
  - 100-149: feat_ask_v_*
  - 150-199: feat_bid_v_*
- Итог: `ask_v` получает feat_bid_p (цены!), `bid_p` получает feat_ask_v (объёмы!) → `price_ch`, `vol_ch`, `imb_ch` — полный мусор. Модель получает искажённый вход → после первого обновления весов (даже с lr=1e-4) активации в Transformer + Conv1d в LOBPatching взрываются → NaN в logits/vol.
- Дополнительный триггер: past_returns (3 канала) добавляются **после нормализации** (в _init_memory_mode), их масштаб ~0.001, а остальные каналы ~N(0,1). Это усиливает нестабильность в patching (Conv1d kernel=6 на 300 фичах).
- AMP 16-bit + отсутствие gradient clipping (mode=none) + curvature_reg + input_noise только ускоряют взрыв.
- Проверка на NaN в train.py (строки 1581-1584) срабатывает только на y/vol_target, но не ловит NaN внутри модели.

**В какой задаче допустили ошибку:**
Задача **022-python-lab-feature-engineering.md** (там ввели группированный порядок колонок в price_exprs + vol_exprs). Слайсинг в dataset.py (из задачи 021 + доработки под multi-channel в 091) остался под старый interleaved-порядок (ask_p0, ask_v0, ask_p1…). Несоответствие не заметили при добавлении past_returns (задача 091) и memory-режима (задача 303).

**Что такое таблица «Name ┃ Type ┃ Params ┃ Mode ┃ FLOP» (из логов):**
Это summary от PyTorch Lightning (ModelSummary callback + lightning.utilities.model_summary). Показывает:
- Name/Type — все модули LiTModule (model=LiTModel 118K, criterion, метрики torchmetrics).
- Params — количество обучаемых параметров (118K ок для d_model=64).
- Mode — train/eval (всё в train, нормально).
- FLOPs — оценка операций (здесь 0). Lightning не вычисляет FLOPs автоматически для кастомных моделей (нужен torchinfo, fvcore или ptflops). 0 — не ошибка, просто заглушка. Не влияет на обучение.

**Вывод по логам (пока только ошибки):**
- Всё до создания датасета (загрузка, labels, normalizer, split, weights) — чисто.
- Проблема именно в данных, которые модель получает после _process_sample.
- CUDA/XLA предупреждения (cuFFT/cuDNN уже registered) — стандартный шум Kaggle/Colab, не причина NaN.
- Vol-MSE/MAE = nan — следствие NaN в vol_pred из модели.

============
# Ответ - Claude-1
Проанализировав логи ошибки (NaN в logits), код python_lab/src/dataset.py и python_lab/src/features.py, я подтверждаю наличие критической архитектурной ошибки, которая приводит к взрыву градиентов.

1. Подтверждение критической ошибки (Data Mismatch)
Grok-1 абсолютно прав: в проекте возник рассинхрон между тем, как признаки генерируются и как они слайсятся для подачи в модель.

В features.py (как создаются колонки):

price_cols = [f"ask_p_{i}" for i in range(50)] + [f"bid_p_{i}" for i in range(50)] # 0-99
vol_cols = [f"ask_v_{i}" for i in range(50)] + [f"bid_v_{i}" for i in range(50)]   # 100-199
То есть порядок: [Ask_Prices (50), Bid_Prices (50), Ask_Volumes (50), Bid_Volumes (50)].

В dataset.py (как они нарезаются в _process_sample):

# Строки 873-874
ask_p, ask_v = x[:, 0:50], x[:, 50:100]    # ОШИБКА: 50:100 — это Bid_Prices, а не Ask_Volumes!
bid_p, bid_v = x[:, 100:150], x[:, 150:200] # ОШИБКА: 100:150 — это Ask_Volumes, а не Bid_Prices!
Последствия: Модель вычисляет price_ch (среднюю цену) как среднее между Ask_Price и Bid_Price (тут случайно совпало, что 0-49 и 100-149 — это цены и объемы, но индексы перепутаны). Но vol_ch и imb_ch превращаются в полный хаос (смешиваются цены и объемы). Это приводит к тому, что на вход Transformer подаются неадекватные значения, что вызывает моментальный взрыв активаций и градиентов после первого же шага оптимизатора.

Мои рекомендации и корректировка плана
Я не согласен с тем, что достаточно просто поправить индексы. Проблема глубже — в жестком кодировании (hardcode) магических чисел (0, 50, 100, 150, 200). При добавлении любых новых признаков все снова сломается.

Мои предложения по переписыванию плана:
Устранение Hardcode (Критично): Вместо x[:, 0:50] нужно использовать динамическое определение индексов на основе feat_cols, которые уже передаются в LOBDataset. Необходимо в __init__ класса LOBDataset вычислить маски индексов для ask_p, bid_p, ask_v, bid_v.

Защита от взрыва (Gradient Clipping): Даже с правильными данными, High-Frequency данные могут быть шумными. В train.py необходимо обязательно включить gradient_clip_val=0.5 или 1.0 в Trainer. Сейчас, судя по логам, он либо выключен, либо не помогает из-за мусора на входе.

Валидация данных перед обучением: Добавить в train.py проверку первого батча: если x.abs().max() > 100 (после нормализации), обучение должно падать с понятной ошибкой "Data scale mismatch", а не "NaN detected".

Past Returns Scaling: Grok заметил, что past_returns имеют масштаб ~0.001. Это верно. При конкатенации с нормализованными LOB-данными (где std=1) это создает дисбаланс весов в первом слое. Их нужно нормализовать отдельно (например, через StandardScaler).

С чем я НЕ СОГЛАСЕН в аргументации Grok:
Grok-1 пишет, что "Первая валидация проходит нормально (MCC 0.1317)". Это ошибка вывода. На первой валидации веса модели случайные. Тот факт, что она выдала 0.1317, — это просто случайность или смещение инициализации. Наличие NaN на второй итерации — это 100% следствие optimizer.step() по "мусорным" градиентам от перепутанных колонок.

Предлагаемый План Действий (корректировка 304):
Dataset.py: Переписать _process_sample так, чтобы он не использовал срезы 0:50, а брал индексы из self.feat_cols.
Lit_model.py: Добавить torch.nn.utils.clip_grad_norm_ или убедиться, что Lightning настроен правильно.
Train.py: Уменьшить log_every_n_steps до 100 (как просит лог Lightning), чтобы видеть проблему раньше.
Features.py: Добавить юнит-тест, проверяющий, что FeatureEngineer.transform выдает колонки именно в том порядке, который ожидает Dataset.

----
Однако, я настаиваю: проблема не только в порядке колонок, но и в отсутствии защиты от взрыва (clipping). Даже с правильными данными финансовые временные ряды могут генерировать огромные градиенты.

Вот мой итоговый, скорректированный план для задачи 304, который учитывает критику Grok и требования по синхронизации с Rust-частью:

1. Исправление порядка колонок (Root Cause)
Необходимо привести features.py и dataset.py к единому стандарту, который ожидает модель.

Файл python_lab/src/features.py:
Изменить порядок конкатенации в transform. Сейчас там: [ask_p, bid_p, ask_v, bid_v].
Нужно сделать: [ask_p, ask_v, bid_p, bid_v]. Это соответствует логике dataset.py (строки 873-874), где 0:100 — это Ask, а 100:200 — это Bid.
Аргумент: Это минимальное изменение, которое не ломает существующий data_schema.json.
2. Нормализация Past Returns (Stability)
Я согласен с Grok, что отдельный скалер не нужен, но текущая реализация в dataset.py (строки 881-884) добавляет их после нормализации LOB-данных.

Файл python_lab/src/dataset.py:
Перенести расчет past_returns внутрь FeatureEngineer или убедиться, что Normalizer.fit видит эти колонки.
В _process_sample убрать copy() и from_numpy для каждой выборки, если это возможно, для оптимизации.
3. Принудительная стабилизация градиентов (Defense)
Даже с правильным порядком колонок, при lr=1e-4 и архитектуре LiT (Transformer), возможны выбросы.

Файл python_lab/src/train.py:
В конфигурацию Trainer (PyTorch Lightning) добавить параметр gradient_clip_val=0.5.
Изменить log_every_n_steps=100 (сейчас 10000), чтобы в логах Kaggle/TensorBoard мы видели момент начала деградации MCC.
Добавить в LiTModule.validation_step проверку: если loss.isnan(), немедленно прерывать обучение с сохранением последнего "живого" чекпоинта.
4. Верификация (Check)
Добавить в python_lab/src/train.py (перед началом fit) "Sanity Check":
Прогнать один батч через модель.
Проверить shape и mean/std выходных активаций price_ch, vol_ch, imb_ch. Если std > 10 — значит, данные все еще "мусорные".

========
# Мой вопрос-1
продожди, одно уточнее, ты написал, что колнки должны быть строко в таком то порядке. 
Но они у анс выгружаются в таком порядке: ask_p_0, ask_v_0.... 
Т.е у нас не идет сначала ask_p1-50, потом ask_v-1-50. Нужно , чтобы модель выгружала колонки четко по названию, а не просто по порядку. Чтоб не получилось, что мы вместо ask_p2 выгрузки ask_v1 - эт понятная задача? Или это уже прописано в нашем плане? У нас идут столбцы все - цена, Потмо объем к этой цене
=========

