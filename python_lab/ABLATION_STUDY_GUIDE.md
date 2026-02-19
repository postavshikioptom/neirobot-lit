# Automated Ablation Studies - Руководство

## Задача 239: Автоматизированные исследования абляции

Система для автоматического тестирования влияния отдельных компонентов (признаков, слоев, голов внимания) на качество модели.

## Цель

Выявить "мертвый вес" в данных и избыточность в архитектуре для оптимизации инференса путем:
- Систематического удаления групп признаков
- Переобучения модели без этих признаков
- Сравнения метрик (MCC, Latency) с базовой моделью

## Установка зависимостей

```bash
pip install -r requirements.txt
```

Основные зависимости:
- `hydra-core>=1.3.2` - конфигурация экспериментов
- `shap>=0.45.0` - анализ важности признаков
- `optuna>=3.6.0` - pruning для ускорения экспериментов
- `pytorch-lightning>=2.2.0` - обучение моделей

## Конфигурация

Файл `ablation_config.yaml` содержит:

### Группы признаков (feature_groups)

```yaml
feature_groups:
  lob_depth_shallow:  # Первые 10 уровней стакана
    - feat_ask_p_0
    - feat_ask_v_0
    # ...
  
  lob_depth_deep:  # Глубокие уровни (10-49)
    - feat_ask_p_10
    # ... (генерируются автоматически)
  
  trade_imb:  # Trade imbalance признаки
    - feat_imb_vol_1s
    - feat_imb_vol_5s
    # ...
  
  past_returns:  # Исторические доходности
    - feat_r_2
    - feat_r_3
    # ...
```

### Варианты архитектуры (arch_variants)

```yaml
arch_variants:
  heads: [2, 4, 8, 16]  # Количество голов внимания
  layers: [1, 2, 3, 4]  # Количество слоев Transformer
  d_model: [64, 128, 256]  # Размерность модели
```

### Параметры обучения

```yaml
training:
  epochs: 10  # Сокращенное обучение для быстрой оценки
  batch_size: 256
  lr: 1e-4
  early_stopping_patience: 3
  use_pruning: true  # Optuna Pruning для ранней остановки
```

## Использование

### Базовый запуск

```bash
python scripts/ablation_study.py \
  --data_path bots/BTCUSDT/data \
  --symbol BTCUSDT \
  --config ablation_config.yaml
```

### Параметры командной строки

- `--config` - путь к конфигурационному файлу (по умолчанию: `python_lab/ablation_config.yaml`)
- `--data_path` - путь к директории с данными (обязательный)
- `--symbol` - торговый символ (обязательный)
- `--output_dir` - директория для результатов (по умолчанию: `python_lab/ablation_results`)
- `--skip_baseline` - пропустить обучение baseline (использовать кэшированные результаты)
- `--skip_shap` - пропустить SHAP анализ

### Пример с пропуском baseline

Если baseline модель уже обучена:

```bash
python scripts/ablation_study.py \
  --data_path bots/BTCUSDT/data \
  --symbol BTCUSDT \
  --skip_baseline
```

## Процесс работы

1. **Baseline обучение**
   - Обучается базовая модель со всеми признаками и максимальными архитектурными параметрами
   - Измеряются метрики: MCC, Latency, Val Loss
   - Результаты сохраняются в `baseline_results.yaml`

2. **SHAP анализ** (опционально)
   - Быстрая предварительная оценка важности признаков
   - Использует GradientExplainer на n_samples примерах
   - Помогает приоритизировать группы для тестирования
   - Результаты сохраняются в `shap_importance.csv`

3. **Architecture Ablation эксперименты**
   - Для каждой комбинации (heads, layers, d_model):
     - Обучается модель с этой архитектурой
     - Измеряются метрики
     - Вычисляется ΔMCC относительно baseline
   - Используется Optuna Pruning для ранней остановки

4. **Feature Ablation эксперименты**
   - Для каждой группы признаков:
     - Исключаются признаки из датасета
     - Модель переобучается (10 эпох с EarlyStopping)
     - Измеряются метрики
     - Вычисляется ΔMCC относительно baseline
   - Используется Optuna Pruning для ранней остановки бесперспективных экспериментов

5. **Генерация отчета**
   - Создается `ablation_report.md` с результатами
   - Группы признаков ранжируются по |ΔMCC|
   - Архитектуры ранжируются по MCC и Latency
   - Выделяются кандидаты на удаление (ΔMCC ≈ 0)
   - Анализ Pareto frontier для компромисса MCC vs Latency

## Структура отчета

Файл `reports/ablation_report.md` содержит:

### Baseline Results
- MCC, Latency, Val Loss базовой модели

### Feature Ablation Results
Таблица с результатами для каждой группы признаков:
- MCC - Matthews Correlation Coefficient
- ΔMCC - изменение относительно baseline
- Latency - задержка инференса в миллисекундах
- ΔLatency - изменение задержки
- Val Loss - валидационная ошибка

### Architecture Ablation Results
Таблица с результатами для каждой комбинации архитектуры:
- Config - название конфигурации (heads, layers, d_model)
- Heads - количество голов внимания
- Layers - количество слоев Transformer
- D_Model - размерность модели
- MCC - качество модели
- ΔMCC - изменение относительно baseline
- Latency - задержка инференса
- ΔLatency - изменение задержки

### Dead Weight Analysis
Список признаков с минимальным влиянием на качество (|ΔMCC| < threshold)

### Architecture Recommendations
- Лучшая архитектура по MCC
- Самая быстрая архитектура
- Pareto frontier (оптимальные компромиссы MCC vs Latency)

### Feature Recommendations
- Самые важные группы признаков
- Кандидаты на удаление

## Интерпретация результатов

### ΔMCC (Delta MCC)

- **ΔMCC < -0.05**: Критически важная группа признаков. Удаление сильно ухудшает качество.
- **-0.05 < ΔMCC < -0.01**: Важная группа. Рекомендуется сохранить.
- **-0.01 < ΔMCC < 0.01**: "Мертвый вес". Можно удалить без потери качества.
- **ΔMCC > 0.01**: Удаление улучшает качество (возможно, переобучение на этих признаках).

### Latency

- Если удаление группы снижает latency без потери MCC - хороший кандидат для оптимизации
- Компромисс между качеством и скоростью

## Модификация датасета

В `src/dataset.py` добавлен параметр `exclude_features`:

```python
dataset = LOBDataset(
    df,
    seq_len=100,
    exclude_features=['feat_ask_p_10', 'feat_ask_v_10', ...]  # Исключаемые признаки
)
```

Это позволяет:
- Тестировать модель без определенных признаков
- Сохранять обратную совместимость (если `exclude_features=None`)
- Работать во всех режимах загрузки (memory, streaming, memmap)

## Расширение конфигурации

### Добавление новых групп признаков

Отредактируйте `ablation_config.yaml`:

```yaml
feature_groups:
  my_custom_group:
    - feat_custom_1
    - feat_custom_2
```

### Тестирование архитектурных вариантов

Для тестирования разного количества голов/слоев нужно модифицировать `run_ablation_experiment()`:

```python
# Вместо фиксированных значений
model = LiTModule(
    n_heads=config['arch_variants']['heads'][i],
    n_layers=config['arch_variants']['layers'][j],
    d_model=config['arch_variants']['d_model'][k]
)
```

## Оптимизация производительности

### Optuna Pruning

Автоматически останавливает бесперспективные эксперименты:
- Мониторит `val_mcc` на каждой эпохе
- Прерывает обучение, если результаты хуже медианы предыдущих trials
- Экономит время на очевидно плохих конфигурациях

### EarlyStopping

Останавливает обучение при плато валидации:
- `patience=3` - ждет 3 эпохи без улучшения
- Предотвращает переобучение
- Ускоряет эксперименты

### Сокращенное обучение

10 эпох достаточно для оценки ΔMCC:
- Модель адаптируется к отсутствию признаков за 5-7 эпох
- Дальнейшее обучение дает минимальный прирост
- Баланс между точностью оценки и скоростью

## Troubleshooting

### Out of Memory

Уменьшите `batch_size` в конфигурации:
```yaml
training:
  batch_size: 128  # Вместо 256
```

### Медленное выполнение

- Используйте `--skip_shap` для пропуска SHAP анализа
- Уменьшите количество групп в конфигурации
- Увеличьте `early_stopping_patience` для более ранней остановки

### Ошибки с признаками

Проверьте, что все признаки в `feature_groups` существуют в датасете:
```python
import polars as pl
df = pl.read_parquet("bots/BTCUSDT/data/BTCUSDT_*.parquet")
feat_cols = [c for c in df.columns if c.startswith("feat_")]
print(feat_cols)
```

## Примеры результатов

### Типичный вывод

```
[Baseline] Results:
  MCC: 0.3245
  Latency: 12.34 ms
  Val Loss: 0.8765

[Ablation] Testing group: lob_depth_shallow
[Ablation] Excluding 40 features
[Ablation] lob_depth_shallow - MCC: 0.2891, Latency: 10.12 ms

[Ablation] Testing group: trade_imb
[Ablation] Excluding 8 features
[Ablation] trade_imb - MCC: 0.3198, Latency: 12.01 ms

[Report] Report saved to python_lab/reports/ablation_report.md
```

### Интерпретация

- `lob_depth_shallow`: ΔMCC = -0.0354 → Важная группа
- `trade_imb`: ΔMCC = -0.0047 → Можно удалить (минимальное влияние)

## Дополнительные ресурсы

- [SHAP Documentation](https://shap.readthedocs.io/)
- [Optuna Pruning Guide](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html)
- [PyTorch Lightning Callbacks](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html)

## Авторы

Задача 239: Automated Ablation Studies
Реализовано в рамках проекта neirobot-lit
