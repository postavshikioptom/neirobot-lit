# Задача 305: Исправление ошибок: Неправильная логика взвешивания режимов (Regime Weighting) в multi-horizon. Compute_curvature_penalty не использует regime_id для инференса. DistillationLoss не поддерживает label_smoothing в soft loss.Отсутствие инициализации bias для bottleneck слоёв

# ПЛАН РЕАЛИЗАЦИИ 
Grok прав в обнаружении некоторых критических багов, но в ряде моментов его аргументация слаба или не учитывает специфику нашей архитектуры. Ниже мой детальный разбор и **КОНТР-ПЛАН**.

---

## 1. Анализ критических ошибок (Спор с Grok)

### Ошибка №1: Некорректный `MultiHorizonLoss` при `reduction='mean'`
*   **Где:** [./python_lab/src/utils.py:176-181](./python_lab/src/utils.py:176-181)
*   **Вердикт Grok:** Верно. В текущем коде `total_loss += horizon_weights[h] * loss_h.mean()`. Это приводит к двойному усреднению (сначала по батчу внутри горизонта, потом суммирование по горизонтам).
*   **Мой аргумент:** Если веса горизонтов в сумме дают 1, то итоговый лосс будет иметь правильный масштаб, НО это ломает логику `sample_weights` (временное взвешивание), так как веса применяются к `loss_h`, а потом берется `mean()`, что размывает влияние весов конкретных "важных" примеров.
*   **Решение:** Сначала суммировать взвешенные лоссы по горизонтам для каждого примера, а потом применять общее усреднение по батчу.

### Ошибка №2: `compute_curvature_penalty` и `**kwargs`
*   **Где:** [./python_lab/src/lit_model.py:77-115](./python_lab/src/lit_model.py:77-115)
*   **Вердикт Grok:** Верно. В [./python_lab/src/train.py](./python_lab/src/train.py) вызов идет с `regime_id`, который попадает в `**kwargs` функции регуляризации, но внутри функции `model(perturbed_inputs, **kwargs)` может вызвать конфликт, если `regime_id` не ожидается моделью в определенном формате или если он передается повторно.
*   **Мой аргумент:** Проблема глубже. В `LiTModel.forward` ([./python_lab/src/lit_model.py:410](./python_lab/src/lit_model.py:410)) аргумент `regime_id` является именованным. Если в `compute_curvature_penalty` мы передаем его через `**kwargs`, это сработает, НО только если мы уверены, что `inputs` и `regime_id` согласованы по размеру батча.

### Ошибка №3: `DistillationLoss` и `label_smoothing`
*   **Где:** [./python_lab/src/utils.py:425-462](./python_lab/src/utils.py:425-462) (в коде это класс `DistillationLoss`)
*   **Вердикт Grok:** Верно. `DistillationLoss` жестко задает `nn.CrossEntropyLoss()` без учета `label_smoothing`.
*   **Мой аргумент:** Я **НЕ СОГЛАСЕН** с Grok, что это "критическая ошибка". В классической дистилляции (по Хинтону) `hard_loss` часто используется без сглаживания, так как "мягкие" метки от учителя уже выполняют роль регуляризатора. Однако для консистентности с нашим конфигом (Задача 054), поддержку добавить стоит.

---

## 2. Мой ПЛАН ИСПРАВЛЕНИЙ (Скорректированный)

Я предлагаю более системный подход, чем Grok. Мы не просто "патчим" дыры, а приводим лосс-функции к единому стандарту.

### Этап 1: Рефакторинг `MultiHorizonLoss` в `utils.py`
1.  Изменить цикл расчета так, чтобы сохранять тензор лоссов по батчу:
    ```python
    # Внутри forward:
    per_example_loss = torch.zeros(batch_size, device=device)
    for h in range(self.num_horizons):
        loss_h = F.cross_entropy(..., reduction='none')
        if sample_weights is not None:
            loss_h = loss_h * sample_weights
        per_example_loss += self.horizon_weights[h] * loss_h
    
    if self.reduction == 'mean': return per_example_loss.mean()
    # ... и так далее
    ```

### Этап 2: Исправление `DistillationLoss`
1.  Добавить `label_smoothing` в `__init__` и прокинуть его в `ce_loss`.
2.  **Важно:** В `train.py` при инициализации `DistillationLoss` нужно передавать `effective_label_smoothing`, который вычисляется в строке 151. Grok это упустил.

### Этап 3: Стабилизация `Curvature Regularization`
1.  В `lit_model.py` изменить `compute_curvature_penalty`, чтобы она явно принимала `regime_id`.
2.  Добавить проверку: если модель в режиме `eval`, регуляризация не должна считаться (сейчас этого нет, что может замедлить валидацию).

### Этап 4: Инициализация весов
1.  Я **ПОЛНОСТЬЮ СОГЛАСЕН** с Grok по поводу `nn.init.zeros_(self.classifier.bias)`. В текущем коде [./python_lab/src/lit_model.py:368-371](./python_lab/src/lit_model.py:368-371) инициализация bias отсутствует, что для финансовых данных (где Flat класс доминирует) может привести к долгому выходу из локального минимума.

---


=============================
# ВЫПОЛНЕНО В КОДЕ:
Список выполненных изменений:
Рефакторинг MultiHorizonLoss (utils.py):

Переписана логика forward: теперь лоссы суммируются по горизонтам для каждого примера индивидуально (per_example_loss).
Веса sample_weights (включая regime_weighting) применяются к каждому примеру до усреднения по батчу.
Исправлена работа reduction='none', теперь возвращается тензор лоссов по батчу.
Обновление DistillationLoss (utils.py):

Добавлен параметр label_smoothing в конструктор.
nn.CrossEntropyLoss теперь инициализируется с поддержкой сглаживания меток.
Стабилизация compute_curvature_penalty (lit_model.py):

Параметр regime_id стал явным в сигнатуре функции.
Добавлена проверка model.training: расчет штрафа пропускается в режиме eval.
В вызове model внутри функции теперь явно передается regime_id.
Исправление инициализации весов (lit_model.py):

Добавлена явная инициализация bias нулями для слоев class_bottleneck, vol_bottleneck, classifier и vol_regressor.
Обновление логики обучения (train.py):

Regime Weighting: Удалено некорректное усреднение regime_w.mean(). Теперь веса режимов объединяются с временными весами в combined_weights и передаются в MultiHorizonLoss для корректного взвешивания каждого примера.
Distillation: При создании DistillationLoss теперь передается актуальный effective_label_smoothing.
Curvature Penalty: Параметр regime_id передается в функцию регуляризации явно.


Я исправил дополнительные ошибки в реализации дистилляции:

В train.py:

Reduction: Исправлено условие включения reduction='none' для DistillationLoss. Теперь оно учитывает и use_regime_weighting, что предотвращает падение при использовании весов режимов без временных весов.
Multi-Horizon Weights: В training_step добавлена корректная обработка весов для дистилляции. Теперь веса батча (B,) правильно применяются к лоссу (B, H) через unsqueeze(-1), устраняя ошибку несовпадения размерностей.
Horizon Weights Initialization: При создании DistillationLoss теперь передаются horizon_weights, чтобы дистилляция учитывала важность разных горизонтов аналогично основному лоссу.
В utils.py (класс DistillationLoss):

Маскирование: Добавлена поддержка ignore_index=-100. Теперь soft_loss (KL Divergence) маскируется на основе меток, чтобы модель не получала градиенты от учителя для недоступных (маскированных) горизонтов будущего.
Horizon Weighting: В методе forward добавлено применение horizon_weights к лоссу каждого горизонта.
Корректная редукция: Метод mean теперь усредняет лосс только по реально существующим (не маскированным) элементам.


======================
# ПРОБЛЕМЫ


## Резюме

В ходе анализа кода python_lab было изучено:
- Основные модули обучения (`train.py`, `tune.py`)
- Архитектура модели (`lit_model.py`, `layers.py`)
- Обработка данных (`dataset.py`, `features.py`, `labels.py`, `normalization.py`)
- Функции потерь и утилиты (`utils.py`)

**Общее количество файлов:** ~75 Python файлов

**Результат:** Обнаружено **4 грубые ошибки** и несколько потенциальных проблем, требующих внимания.

---

## Обнаруженные грубые ошибки

### 1. Неправильная логика взвешивания режимов (Regime Weighting) в multi-horizon режиме

**Файл:** `src/train.py` (строки 286-289)

**Проблема:**
```python
# Применяем regime weighting если нужно
if self.use_regime_weighting and self.regime_weights is not None and regime_id is not None:
    regime_w = self.regime_weights[regime_id].to(loss_cls.device)
    loss_cls = loss_cls * regime_w.mean()  # Усредняем веса режимов по батчу
```

**Описание:**
Когда включено `use_regime_weighting` для multi-horizon режима, код:
1. Сначала вычисляет `loss_cls` через `MultiHorizonLoss` с учётом `sample_weights`
2. Затем умножает полученный лосс на среднее значение весов режимов по батчу

**Почему это ошибка:**
- `regime_w.mean()` усредняет веса режимов по всему батчу, что **обнуляет индивидуальное взвешивание** для каждого примера
- Это делает regime weighting бесполезным в multi-horizon режиме
- Правильный подход: regime weighting должен применяться **внутри** `MultiHorizonLoss.forward()` для каждого примера отдельно

**Влияние:**
- Regime weighting не работает как ожидается в multi-horizon режиме
- Модель не получает преимущества от учёта разных рыночных режимов

**Как исправить:**
Вариант 1: Передать regime weighting в MultiHorizonLoss:
```python
# В MultiHorizonLoss.__init__ добавить параметр
def __init__(self, num_horizons=3, horizon_weights=None, class_weights=None, 
             label_smoothing=0.0, reduction='mean', use_regime_weighting=False):
    self.use_regime_weighting = use_regime_weighting

# В forward методе MultiHorizonLoss
if sample_weights is not None and self.use_regime_weighting:
    # sample_weights уже содержит regime weighting из train.py
    loss_h = loss_h * sample_weights
elif sample_weights is not None:
    # Только time weighting
    loss_h = loss_h * sample_weights
```

Вариант 2: Убрать умножение на `regime_w.mean()` и убедиться, что regime weighting уже учтён в `sample_weights`.

---


### 2. Compute_curvature_penalty не использует regime_id для инференса

**Файл:** `src/lit_model.py` (строки 77-115) и вызов в `src/train.py` (строки 315-323)

**Проблема:**
```python
def compute_curvature_penalty(model, inputs, outputs, lambda_=1e-4, epsilon=1e-3, **kwargs):
    # ...
    perturbed_outputs = model(perturbed_inputs, **kwargs)  # regime_id передаётся в kwargs
```

**Описание:**
Функция `compute_curvature_penalty` принимает `regime_id` через `**kwargs` и передаёт его при повторном вызове модели. Однако:

1. Если модель использует regime embedding (когда `num_regimes > 0`), regime_id критически важен
2. При вычислении кривизны, regime_id должен быть одинаковым для оригинального и возмущённого инференса
3. Нет явной проверки или документации, что regime_id корректно используется

**Почему это ошибка:**
- Если `regime_id` не передан явно или передан неправильно, regime embedding не будет применён
- Это приведёт к неправильной оценке кривизны модели в контексте разных режимов рынка

**Влияние:**
- Неправильное вычисление регуляризации кривизны
- Регуляризация может быть менее эффективной или даже вредной для разных режимов

**Как исправить:**
```python
def compute_curvature_penalty(model, inputs, outputs, lambda_=1e-4, epsilon=1e-3, regime_id=None, **kwargs):
    """
    Добавить явный параметр regime_id вместо передачи через kwargs
    """
    # ...
    # Если regime_id нужен для модели, используем его явно
    if regime_id is not None:
        perturbed_outputs = model(perturbed_inputs, regime_id=regime_id)
    else:
        perturbed_outputs = model(perturbed_inputs, **kwargs)
```

И обновить вызов в train.py:
```python
reg_loss = compute_curvature_penalty(
    self.model, 
    x, 
    logits, 
    lambda_=self.curvature_lambda,
    regime_id=regime_id  # Явный параметр вместо **kwargs
)
```

---

### 3. DistillationLoss не поддерживает label_smoothing в soft loss

**Файл:** `src/utils.py` (строки 425-462)

**Проблема:**
```python
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.9, temperature=3.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.reduction = reduction
        self.kl_div = nn.KLDivLoss(reduction=reduction)
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)  # Нет label_smoothing!
```

**Описание:**
Knowledge Distillation использует два компонента:
1. **Soft loss:** KL divergence между размягчёнными распределениями teacher и student
2. **Hard loss:** Cross entropy с истинными метками

Проблема в том, что `hard_loss` не поддерживает `label_smoothing`, хотя он объявлен в `LiTModule.__init__` и используется для других режимов обучения.

**Почему это ошибка:**
- При distillation нельзя применить label smoothing к hard loss
- Это делает distillation несовместимым с label smoothing
- Модель, обученная с label smoothing, не может быть корректно дистиллирована

**Влияние:**
- Несовместимость между distillation и обычным обучением с label smoothing
- Потенциальное ухудшение качества при distillation

**Как исправить:**
```python
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.9, temperature=3.0, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.kl_div = nn.KLDivLoss(reduction=reduction)
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction, label_smoothing=label_smoothing)
```

И обновить вызов в train.py:
```python
self.distillation_criterion = DistillationLoss(
    alpha=alpha, 
    temperature=temperature, 
    reduction='none' if use_time_weighting else 'batchmean',
    label_smoothing=effective_label_smoothing  # Добавить этот параметр
)
```

---



### 4. Отсутствие инициализации bias для bottleneck слоёв

**Файл:** `src/lit_model.py` (строки 368-371)

**Проблема:**
```python
# Инициализация параметров
nn.init.trunc_normal_(self.cls_token, std=0.02)
if self.regime_embedding is not None:
    nn.init.trunc_normal_(self.regime_embedding.weight, std=0.02)
nn.init.xavier_uniform_(self.class_bottleneck.weight)
nn.init.xavier_uniform_(self.vol_bottleneck.weight)
nn.init.xavier_uniform_(self.classifier.weight)
nn.init.xavier_uniform_(self.vol_regressor.weight)
```

**Описание:**
Инициализируются веса для:
- `class_bottleneck`
- `vol_bottleneck`
- `classifier`
- `vol_regressor`

Но не инициализируются `bias` для этих слоёв!

По умолчанию PyTorch инициализирует bias нулями, что может быть неоптимально.

**Почему это ошибка:**
- Нулевая инициализация bias может привести к медленному обучению в начале
- Для bottleneck слоёв, особенно в multi-task learning, важно иметь ненулевой bias
- Современные практики (например, в GPT, BERT, ViT) используют инициализацию bias нулями для Linear слоёв в feedforward, но для классификаторов иногда используют другие подходы

**Влияние:**
- Медленная сходимость в начале обучения
- Потенциально худшая производительность в early stages

**Как исправить:**
```python
# Инициализация параметров
nn.init.trunc_normal_(self.cls_token, std=0.02)
if self.regime_embedding is not None:
    nn.init.trunc_normal_(self.regime_embedding.weight, std=0.02)

# Инициализация bottleneck слоёв
nn.init.xavier_uniform_(self.class_bottleneck.weight)
nn.init.zeros_(self.class_bottleneck.bias)  # Явная нулевая инициализация
nn.init.xavier_uniform_(self.vol_bottleneck.weight)
nn.init.zeros_(self.vol_bottleneck.bias)  # Явная нулевая инициализация

# Инициализация голов
nn.init.xavier_uniform_(self.classifier.weight)
nn.init.zeros_(self.classifier.bias)
nn.init.xavier_uniform_(self.vol_regressor.weight)
nn.init.zeros_(self.vol_regressor.bias)
```

---

## Потенциальные проблемы (менее критичные)

### 1. Positional Encoding при нечётном d_model

**Файл:** `src/lit_model.py` (строки 297-307)

**Проблема:**
```python
if d_model % 2 == 1:
    pe[:, 1::2] = torch.cos(position * div_term[:-1])  # Обрезаем div_term
```

При нечётном `d_model` используется `div_term[:-1]`, что создаёт асимметрию между синусными и косинусными компонентами.

**Почему это менее критично:**
- d_model обычно выбирается чётным (64, 128, 256, 512)
- Используется только один раз при создании PE
- Не влияет на градиенты

**Рекомендация:** Использовать только чётные значения d_model в конфигурациях.

---

### 2. MultiHorizonLoss с reduction='none'

**Файл:** `src/utils.py` (строки 184-188)

**Проблема:**
```python
elif self.reduction == 'none':
    # Для 'none' возвращаем лосс по примерам (усредненный по горизонтам)
    return total_loss
```

При `reduction='none'` возвращается усреднённый по горизонтам лосс, что может быть неожиданным.

**Почему это менее критично:**
- `reduction='none'` используется редко (только для sample_weighting)
- В реальном использовании это корректно работает с sample_weights

**Рекомендация:** Добавить ясный комментарий в документацию.

---

### 3. Отсутствие явной обработки NaN/Inf в FocalLoss

**Файл:** `src/utils.py` (строки 74-99)

**Проблема:**
Нет явных проверок на NaN/Inf в вычислениях Focal Loss.

**Почему это менее критично:**
- FocalLoss использует PyTorch операции, которые уже стабильны
- Ce_loss из `F.cross_entropy` уже имеет встроенную защиту
- В реальном использовании не наблюдается проблем

**Рекомендация:** Добавить assert для отладки в development mode.

---

