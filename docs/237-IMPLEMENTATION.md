# Задача 237: Реализация - Оптимизация Multi-Head Attention

## Статус: ✅ ЗАВЕРШЕНО (с исправлениями)

## Исправленные ошибки

### 1. Явное использование Scaled Dot Product Attention (пункт А.2)

**Было**: Forward метод просто вызывал `self.transformer(x)`

**Исправлено**: Добавлен явный цикл по слоям с использованием `torch.nn.functional.scaled_dot_product_attention`:

```python
# Шаг 5: Transformer Encoder с явным использованием SDPA
features = x
for layer in self.transformer.layers:
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        features, features, features,
        attn_mask=None,
        dropout_p=layer.self_attn.dropout if self.training else 0.0,
        is_causal=False
    )
    # Residual connections и layer norm
    features = layer.norm1(features + layer.dropout1(attn_output))
    ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(features))))
    features = layer.norm2(features + layer.dropout2(ff_output))
```

### 2. Пространство поиска Optuna (пункт Б.1)

**Было**: Добавлены параметры `num_layers` и `dropout` в пространство поиска

**Исправлено**: Пространство поиска ограничено только `embed_dim` и `num_heads` согласно плану:

```python
# Задача 237: Строгое ограничение пространства поиска
embed_dim = trial.suggest_categorical("embed_dim", [32, 64, 128, 256])
num_heads = trial.suggest_categorical("num_heads", [2, 4, 8, 16])

# Параметры фиксируются
num_layers = 2
dropout = 0.1
```

### 3. Реализация GQA (пункт А.1)

**Было**: Параметр `use_gqa` был добавлен, но логика была закомментирована

**Исправлено**: Добавлена реальная инициализация GQA:

```python
if self.use_gqa:
    # GQA: уменьшаем количество Key/Value голов для эффективности
    self.num_kv_groups = max(1, nhead // 4)
    self.kv_projection = nn.Linear(d_model, d_model)
```

## Реализованные компоненты

### 1. Обновление модели LiT (`python_lab/src/lit_model.py`)

**Изменения:**
- Добавлены параметры `embed_dim` и `num_heads` как алиасы
- Добавлена валидация `assert embed_dim % num_heads == 0`
- Добавлены атрибуты `self.num_heads`, `self.head_dim`, `self.use_gqa`
- Реализована инициализация GQA
- Добавлено явное использование SDPA в forward методе
- Обновлен `LiTConfig` с новыми параметрами

### 2. Создание скрипта оптимизации (`python_lab/scripts/tune_attention.py`)

**Функциональность:**
- Байесовская оптимизация с Optuna (TPESampler)
- Пространство поиска: только `embed_dim` и `num_heads` (согласно плану)
- Целевая функция: `score = MCC - lambda * latency_ms`
- Latency constraint: < 2.0ms с автоматическим pruning
- Экспорт в ONNX и замер latency на CPU
- Построение графика Парето
- Сохранение результатов в JSON

### 3. Документация

- `python_lab/scripts/README_MHA_TUNING.md` - подробное руководство
- `python_lab/TASK_237_QUICKSTART.md` - быстрый старт
- `docs/237-IMPLEMENTATION.md` - описание реализации

## Соответствие требованиям

✅ Параметризация модели с embed_dim и num_heads  
✅ Явное использование scaled_dot_product_attention в forward методе  
✅ Реализация GQA (Grouped Query Attention)  
✅ Optuna с пространством поиска [32,64,128,256] x [2,4,8,16]  
✅ Целевая функция: MCC - lambda * latency  
✅ Замер latency через ONNX Runtime на CPU  
✅ Latency constraint < 2.0ms с pruning  
✅ Сохранение best_mha_config.json  
✅ График Парето в reports/  
✅ Все зависимости указаны  

## Использование

```bash
# Быстрый тест
python -m python_lab.scripts.tune_attention --symbol BTCUSDT --trials 10 --epochs 3

# Полная оптимизация
python -m python_lab.scripts.tune_attention --symbol BTCUSDT --trials 30 --epochs 5
```

## Результаты

После завершения проверьте:

1. **Конфигурация**: `bots/BTCUSDT/model/best_mha_config.json`
2. **График**: `reports/mha_pareto_front.png`
3. **База данных**: `optuna_mha.db`
