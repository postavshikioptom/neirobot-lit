# Задача 303: Оптимизация памяти при нормализации и балансировке (Batch + Append-to-Disk)

## Статус: В РАБОТЕ

## 1. Описание проблемы (Memory Explosion)

Задача **127** (оверсэмплинг) привела к критической ошибке: в `train.py` весь тренировочный набор данных материализовался в 3D-тензор `(N, Seq_Len, Features)` перед балансировкой.

**Потребление RAM:**
`560,000 (окон) × 100 (длина) × 203 (фичи) × 4 байта ≈ 45 ГБ`.
Это вызывает зависание (swap) или крах (OOM) на большинстве систем.

**Критические замечания (Grok + User):**
1. **Порядок нормализации:** SMOTE должен работать на **нормализованных** данных. 
2. **Несогласованность батчей:** У каждого батча должен быть один и тот же "целевой ориентир" по количеству классов (глобальная стратегия).
3. **Вариативный размер:** Из-за валидации LOB-последовательностей (задача 127) мы не знаем заранее точное количество сэмплов после SMOTE. 
4. **Отсутствие альтернатив:** План должен содержать один четкий путь реализации.

---

## 2. Финальная стратегия: Batch Processing + Append-to-Disk

Мы реализуем надежный промышленный вариант: **Батчевая обработка с записью в бинарный поток.** Это решает проблему неизвестного заранее размера и экономит RAM.

### Алгоритм:
1. **Fit Normalizer (2D)**: Обучаем нормализатор на сырых строках `(N_train, 203)`.
2. **Global Strategy**: Один раз считаем распределение классов во всей выборке для вычисления единого `sampling_strategy`.
3. **Batch Loop (Append mode)**: 
   - Обрабатываем данные по 50,000 окон (~4 ГБ RAM).
   - Нормализуем батч.
   - Балансируем батч.
   - **Записываем (append)** батч в бинарный файл на диске.
4. **Memmap Dataset**: После цикла подключаем `np.memmap` к созданному файлу для обучения.

---

## 3. Детали реализации

### 3.1 Модификация `python_lab/src/dataset.py`
Обновляем `balance_dataset` для поддержки внешней стратегии:
```python
def balance_dataset(features, labels, method='bgmm', ratio=0.5, sampling_strategy=None):
    # Если стратегия передана извне, используем её, иначе считаем локально (совместимость)
    if sampling_strategy is None:
        counts = np.bincount(labels)
        maj_class = np.argmax(counts)
        target_count = int(counts[maj_class] * ratio)
        sampling_strategy = {1: max(counts[1], target_count), 2: max(counts[2], target_count)}
    # SMOTE/BGMM_SMOTE...
```

### 3.2 Реализация в `python_lab/src/train.py`

**Шаг 1: Глобальный расчет (до условий по балансировке)**
```python
train_indices = train_ds.indices
train_labels_all = full_dataset.labels[train_indices + seq_len - 1]
global_counts = np.bincount(train_labels_all)
# sampling_strategy = {1: target, 2: target} ...
```

**Шаг 2: Реализация развилки "Балансировка или нет"**
```python
if args.balance_method != "none":
    # 1. Сначала ОБУЧАЕМ нормализатор на 2D-сырых данных (~0.5 ГБ)
    normalizer.fit(full_dataset.features[train_indices], feature_names=feat_cols, ...)
    normalizer.save(...)

    # 2. Потоковая запись сбалансированных данных
    total_balanced_samples = 0
    with open(feat_bin_path, 'wb') as f_feat, open(lab_bin_path, 'wb') as f_lab:
        for i in range(0, len(train_indices), BATCH_SIZE):
            batch_indices = train_indices[i : i + BATCH_SIZE]
            
            # А) Сборка 3D батча + Метки батча
            batch_3d = np.stack([full_dataset.features[j : j + seq_len] for j in batch_indices])
            batch_labels = full_dataset.labels[batch_indices + seq_len - 1]
            
            # Б) Нормализация батча (строка 181 normalization.py)
            batch_3d_norm = normalizer.transform(batch_3d)
            
            # В) Балансировка батча с глобальной стратегией
            b_feat, b_lab = balance_dataset(batch_3d_norm, batch_labels, sampling_strategy=sampling_strategy)
            
            # Г) Запись в файл (append)
            f_feat.write(b_feat.tobytes())
            f_lab.write(b_lab.tobytes())
            total_balanced_samples += len(b_lab)

    # 3. Подключаем Memmap к результату
    features_res = np.memmap(feat_bin_path, dtype='float32', mode='r', 
                             shape=(total_balanced_samples, seq_len, n_features))
    labels_res = np.memmap(lab_bin_path, dtype='int64', mode='r', 
                           shape=(total_balanced_samples,))

    # 4. Создаем BalancedTrainDataset
    train_ds = BalancedTrainDataset(features_res, labels_res, full_dataset)
    
else:
    # Случай без балансировки - просто фит нормализатора
    normalizer.fit(full_dataset.features[train_indices], ...)
    normalizer.save(...)
    # train_ds остается оригинальным
```

### 3.3 Класс `BalancedTrainDataset`
Должен быть обновлен, чтобы работать с уже нормализованными данными из memmap и применять решейп/бродкаст каналов (как в строках 1621–1655 текущего `train.py`).

---

## 4. Ожидаемый результат

1. **RAM:** Пиковое потребление снизится с **45 ГБ до ~4-5 ГБ** (один батч в RAM).
2. **Корректность:** SMOTE работает на нормализованных признаках. Глобальная стратегия гарантирует единство классов.
3. **Стабильность:** Подход с `open('wb')` и `.tobytes()` справляется с любым количеством отфильтрованных (invalid) сэмплов.

---

## 5. План действий

1. Обновить `balance_dataset` в `dataset.py`.
2. Реализовать батчевый цикл с записью в файл в `train.py`.
3. Добавить класс `BalancedTrainDataset` для работы с memmap-результатом.
4. Проверить запуск с `--balance_method smote/bgmm` и `none`.

---

## 6. Задача 303-2: Исправление ValueError (Feature Mismatch)

После реализации батчевой обработки возникла ошибка `ValueError` при обучении нормализатора в режиме `--balance_method none`. Это связано с тем, что данные в `full_dataset.features` уже содержат `past_returns` (203 признака), а список `feat_cols` содержит только 200 оригинальных колонок стакана.

### План исправления:

1. **Синхронизация имен признаков**:
   - В `train.py` перед вызовом `normalizer.fit` необходимо динамически расширять список `feat_cols`, если включены расчеты `past_returns`.
   - Если `n_past_returns > 0`, в список `feat_cols` будут добавлены имена для лагов (например, `past_return_10`, `past_return_50`, `past_return_100`).

2. **Сохранение индикаторов**:
   - Все дополнительные индикаторы (доходности и др.) рассчитываются "на лету" из сырого стакана. Мы сохраняем эту логику, так как она важна для качества модели. 
   - Исходные данные в файлах (сырой стакан + timestamp) остаются нетронутыми.

3. **Верификация размерности**:
   - Убедиться, что `train_features_2d.shape[1]` строго равно `len(feat_cols)` перед вызовом `normalizer.fit`.

### Ожидаемый результат:
- Обучение запускается без ошибок в Kaggle/на сервере.
- Размерность признаков консистентна во всех модулях.

---

## 7. Задача 303-3: Исправление NameError 'test_ds' и Split 70/15/15 (v3)

В ходе оптимизации памяти возникла регрессия: хронологическое разделение было заменено на `random_split`, а флаг `is_train = True` был установлен глобально, что привело к "утечке" аугментации в валидацию и тест.

### План исправления:

**ШАГ 1: Класс `TrainSubset` для безопасной аугментации**

Для того чтобы аугментация применялась **только** к тренировочной выборке, мы внедрим специальный класс-обертку в `train.py`:

```python
class TrainSubset(torch.utils.data.Subset):
    def __getitem__(self, idx):
        original_is_train = self.dataset.is_train
        self.dataset.is_train = True
        try:
            result = super().__getitem__(idx)
        finally:
            self.dataset.is_train = original_is_train
        return result
```

**ШАГ 2: Хронологическое разделение 70/15/15**

Мы заменяем `random_split` на строгое разделение по индексам. Это исключает Data Leakage и гарантирует методологическую чистоту тестов.

```python
total_len = len(full_dataset)
train_size = int(0.70 * total_len)
val_size = int(0.15 * total_len)
test_size = total_len - train_size - val_size

train_indices = list(range(0, train_size))
val_indices = list(range(train_size, train_size + val_size))
test_indices = list(range(train_size + val_size, total_len))

from torch.utils.data import Subset
train_ds = TrainSubset(full_dataset, train_indices)
val_ds = Subset(full_dataset, val_indices)
test_ds = Subset(full_dataset, test_indices)

# Верификационный вывод
print(f"\nChronological split verification:")
print(f"  Train: indices {train_indices[0]}-{train_indices[-1]} ({len(train_ds)} samples, {len(train_ds)/total_len*100:.1f}%)")
print(f"  Val:   indices {val_indices[0]}-{val_indices[-1]} ({len(val_ds)} samples, {len(val_ds)/total_len*100:.1f}%)")
print(f"  Test:  indices {test_indices[0]}-{test_indices[-1]} ({len(test_ds)} samples, {len(test_ds)/total_len*100:.1f}%)")
```

### Ожидаемый результат:
- Устранение `NameError` по `test_ds`.
- Корректная и безопасная аугментация (только для Train).
- Точное хронологическое разделение 70/15/15.
