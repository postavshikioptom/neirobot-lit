# TensorBoard Integration Guide (Задача 158)

## Обзор

Интеграция TensorBoard для глубокого анализа обучения модели LiT с упором на производительность и визуализацию микроструктуры стакана (LOB).

## Новые CLI параметры

```bash
python -m python_lab.src.train \
    --symbol BTCUSDT \
    --tb_dir runs/BTCUSDT \              # Директория для TensorBoard логов
    --tb_hist_freq 10 \                  # Частота записи полных гистограмм (эпохи)
    --tb_embedding_samples 1000          # Макс. сэмплов для Projector
```

### Параметры:

- `--tb_dir`: Директория для TensorBoard логов (по умолчанию: `runs/SYMBOL/`)
- `--tb_hist_freq`: Частота записи полных гистограмм активаций (по умолчанию: каждые 10 эпох)
- `--tb_embedding_samples`: Максимальное количество сэмплов для TensorBoard Projector (по умолчанию: 1000)

## Запуск TensorBoard

После начала обучения запустите TensorBoard:

```bash
tensorboard --logdir=runs/BTCUSDT
```

Откройте браузер: http://localhost:6006

## Функции визуализации

### 1. Custom Scalars Layout

Структурированный дашборд с группировкой метрик:

- **Losses**: train_loss, val_loss_cls, val_loss_vol
- **Performance**: val_mcc, val_f1_macro, precision/recall по классам
- **Learning**: learning rate, task weights
- **Calibration**: ECE, MCE

### 2. Мониторинг активаций

Оптимизированный мониторинг через forward hooks:

- **Каждую эпоху**: mean, std, max для слоев Patching и Attention
- **Каждые N эпох**: полные гистограммы (параметр `--tb_hist_freq`)

Слои:
- `activations/patching/*`
- `activations/transformer_layer_*/attention`
- `activations/transformer_layer_*/ffn_*`

### 3. Мониторинг градиентов

Нормы градиентов для каждого именованного параметра:

- `gradients/{param_name}/norm`

Помогает обнаружить vanishing/exploding gradients в глубоких слоях.

### 4. Confusion Matrix

Интерактивная матрица ошибок через `add_figure`:

- Сырая матрица (Raw Counts)
- Нормализованная матрица (Recall)

Обновляется каждые 5 эпох.

### 5. PR-кривые

Precision-Recall кривые для каждого класса:

- Использует `sklearn.metrics.precision_recall_curve`
- Показывает Average Precision для каждого класса
- Обновляется каждые 5 эпох

### 6. TensorBoard Projector

Визуализация embeddings после слоя патчинга:

- Извлекает векторы после LOBPatching
- Ограничивает до `--tb_embedding_samples` (по умолчанию 1000)
- Метаданные: labels, regime_id (если доступен)
- Обновляется каждые 10 эпох

Позволяет увидеть, как модель кластеризует различные состояния стакана.

### 7. HParams

Сравнение гиперпараметров между запусками:

- Логируются в начале обучения: lr, d_model, nhead, num_layers, batch_size, etc.
- Итоговые метрики в конце: best_val_mcc

## Автоматическая очистка логов

Автоматически удаляет старые запуски, оставляя только 50 последних:

```python
cleanup_old_tensorboard_logs(log_dir, max_runs=50)
```

Полезно при тысячах запусков Optuna.

## Производительность

Оптимизации для больших `seq_len`:

1. **Статистика вместо гистограмм**: mean/std/max каждую эпоху, полные гистограммы редко
2. **Ограничение сэмплов**: Projector использует только 1000 сэмплов
3. **Периодическое логирование**: Confusion Matrix, PR-кривые, embeddings обновляются не каждую эпоху

## Пример использования

```bash
# Обучение с TensorBoard визуализацией
python -m python_lab.src.train \
    --symbol BTCUSDT \
    --epochs 100 \
    --batch_size 128 \
    --d_model 128 \
    --nhead 8 \
    --num_layers 4 \
    --tb_dir runs/experiment_001 \
    --tb_hist_freq 5 \
    --tb_embedding_samples 2000

# Запуск TensorBoard
tensorboard --logdir=runs/experiment_001
```

## Структура логов

```
runs/
└── BTCUSDT/
    └── lit_training/
        └── version_0/
            ├── events.out.tfevents.*  # Основные метрики
            ├── hparams.yaml           # Гиперпараметры
            └── checkpoints/           # Чекпоинты модели
```

## Рекомендации

1. **Для быстрого обучения**: используйте `--tb_hist_freq 20` для редких гистограмм
2. **Для детального анализа**: используйте `--tb_hist_freq 5` и `--tb_embedding_samples 2000`
3. **Для Optuna**: автоматическая очистка сохранит только 50 последних запусков
4. **Для LOB анализа**: обратите внимание на Projector - он покажет кластеризацию состояний стакана

## Troubleshooting

### TensorBoard зависает при загрузке Projector

Уменьшите `--tb_embedding_samples`:

```bash
--tb_embedding_samples 500
```

### Слишком много места на диске

Автоматическая очистка удаляет старые запуски. Можно также вручную удалить:

```bash
rm -rf runs/BTCUSDT/lit_training/version_*
```

### Гистограммы замедляют обучение

Увеличьте `--tb_hist_freq`:

```bash
--tb_hist_freq 20  # Каждые 20 эпох вместо 10
```
