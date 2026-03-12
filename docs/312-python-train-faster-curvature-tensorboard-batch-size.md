# Задача 312: Отключить Curvature Regularization. Оптимизировать частоту TensorBoard. Добавить PyTorch Profiler. Оптимизировать DataLoader. Batch Size 64.

# ЗАДАЧА 312: ПЛАН РЕАЛИЗАЦИИ: оптимизации скорости обучения LiT модели

#### Подзадача 312.1: Отключить Curvature Regularization по умолчанию
**Что делать:** Изменить в `train.py` строку 1289:
```python
# Было:
parser.add_argument("--use_curvature_reg", action=argparse.BooleanOptionalAction, default=True, ...)

# Должно быть:
parser.add_argument("--use_curvature_reg", action=argparse.BooleanOptionalAction, default=False, ...)
```
**Ожидаемый эффект:** Ускорение в ~2 раза (с 25 минут до 12-13 минут)

#### Подзадача 312.2: Оптимизировать частоту TensorBoard визуализаций
**Что делать:** В `train.py` в методе `on_validation_epoch_end`:
- Изменить частоту сохранения Reliability Diagrams с 5 на 20 эпох
- Изменить частоту Confusion Matrix/PR-кривых с 5 на 20 эпох
- Изменить частоту Embeddings с 10 на 30 эпох

**Ожидаемый эффект:** Ускорение validation на 10-15%

#### Подзадача 312.3: Добавить PyTorch Profiler для точного измерения
**Что делать:** Добавить профилирование в `train.py`:
```python
from torch.profiler import profile, ProfilerActivity, schedule

profiler = profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
    record_shapes=True,
    profile_memory=True
)
```

**Ожидаемый эффект:** Точное понимание узких мест

#### Подзадача 312.4: Оптимизировать DataLoader
**Что делать:** В `train.py` при создании DataLoader добавить:
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    num_workers=4,  # Добавить параллельную загрузку
    pin_memory=True,  # Ускорить CPU->GPU передачу
    prefetch_factor=2,  # Предзагрузка батчей
    persistent_workers=True  # Не убивать workers между эпохами
)
```

**Ожидаемый эффект:** Ускорение на 20-30% если есть CPU bottleneck

#### Подзадача 312.5: Протестировать оптимальный batch_size
**Что делать:** Запустить эксперименты с batch_size=64 вместо 128
- Исследования по LOB данным рекомендуют 32-64 для трансформеров
- Меньший batch_size может улучшить генерализацию

**Ожидаемый эффект:** Возможное улучшение метрик модели (MCC, F1)

## 🎯 Ожидаемый результат

**Скорость обучения:** С 25 минут до 5-8 минут на эпоху (ускорение в 3-5 раз)
**Метрики модели:** После отключения curvature reg и оптимизации DataLoader можно будет быстрее итерировать и искать лучшие гиперпараметры для улучшения MCC и F1.

План сохранен в память Supermemory для дальнейшего использования.










=================== # ОБЩАЯ ИНФОРМАЦИЯ ==================
# АНАЛИЗ ПРОБЛЕМ
[ДИАГНОСТИКА] Статистика каналов ПОСЛЕ нормализации:
  Channel 0 (Price): min=-4.9942, max=5.0000, mean=0.1834, std=1.1216
  Channel 1 (Vol): min=-8.0442, max=2.5764, mean=-0.0929, std=0.8320
  Channel 2 (Imb): min=-8.9519, max=8.6557, mean=0.0045, std=0.9640
  Channel 3 (OFI): min=-8.9519, max=8.6557, mean=0.0045, std=0.9640
  Channel 4 (VIB): min=-2.8073, max=3.9537, mean=0.0166, std=0.8111
  Channel 5 (PastRet): min=-4.6731, max=7.5534, mean=0.0285, std=1.1435
  
 
 Дополнителньая проблема, не знаю, что пменяли, но раньше Эпоха была 5 мин, а теперь 25 мин, это огроменное время и я так все 50 эпох никогд не пройду.
 Только не думай, что это из-за режима memory, я всегда memory запускал. Изучи в коде, что в обучение или расчетах изменилось, возможно какойто параметр тысячи разлишний разпересчитываем
 
 Epoch 1 (22m 20s) Validation: MCC=-0.0030, Macro-F1=0.2126, ECE=0.2628, MCE=0.8685
  Vol-MSE=3.304562, Vol-MAE=1.199462

Metrics by Market Regime:
  Regime 0: MCC=-0.0030, F1=0.2126, Samples=55602

Epoch 2 (22m 18s) Validation: MCC=0.0093, Macro-F1=0.2484, ECE=0.2762, MCE=0.9352
  Vol-MSE=2.864979, Vol-MAE=1.109995
  
==================

# АНАЛИЗ Kiro-1

Понял вашу задачу. Я планировщик, который должен проанализировать Python скрипты обучения LiT модели и составить один конкретный план улучшения с подзадачами.


## 🔍 Анализ проблемы

**Основная проблема:** Обучение замедлилось с 5 минут до 25 минут на эпоху (в 5 раз!)

**Причины замедления:**

1. **Curvature Regularization (Задача 238)** - ГЛАВНАЯ ПРИЧИНА
   - Включена по умолчанию в `train.py` (строка 1289: `default=True`)
   - На каждом батче делается **дополнительный forward pass** модели для вычисления штрафа за кривизну
   - Это удваивает количество вызовов модели: основной forward + perturbed forward
   - Код в `lit_model.py:78-120` показывает, что это дорогая операция

2. **Частые TensorBoard визуализации**
   - Каждые 5 эпох: Confusion Matrix, PR-кривые, Reliability Diagrams
   - Каждые 10 эпох: Embeddings для TensorBoard Projector
   - Каждую эпоху: Градиенты

3. **Multi-task learning** с предсказанием волатильности (дополнительные вычисления)

4. **Input noise injection** на каждом батче

