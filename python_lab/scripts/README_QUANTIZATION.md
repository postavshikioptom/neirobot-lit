# ONNX Model Quantization (FP32 → INT8)

Руководство по квантованию ONNX моделей для ускорения инференса на CPU.

## Обзор

Квантование преобразует модель из FP32 (32-битные числа с плавающей точкой) в INT8 (8-битные целые числа), что обеспечивает:

- **Ускорение инференса**: 2-4x на CPU с AVX-512/VNNI инструкциями
- **Уменьшение размера модели**: ~4x меньше размер файла
- **Снижение потребления памяти**: ~4x меньше RAM

## Требования

```bash
pip install onnx onnxruntime scikit-learn
```

## Быстрый старт

### 1. Базовое квантование

```bash
python python_lab/scripts/quantize_onnx.py \
    --input bots/CAKEUSDT/model/lit.onnx \
    --output bots/CAKEUSDT/model/lit_int8.onnx \
    --data_path bots/CAKEUSDT/data/raw \
    --symbol CAKEUSDT
```

### 2. С настройкой параметров

```bash
python python_lab/scripts/quantize_onnx.py \
    --input bots/CAKEUSDT/model/lit.onnx \
    --output bots/CAKEUSDT/model/lit_int8.onnx \
    --data_path bots/CAKEUSDT/data/raw \
    --symbol CAKEUSDT \
    --max_mcc_drop 0.01 \
    --n_calibration_samples 1000 \
    --seq_len 100 \
    --n_past_returns 0
```

### 3. Быстрое квантование (без проверки качества)

```bash
python python_lab/scripts/quantize_onnx.py \
    --input bots/CAKEUSDT/model/lit.onnx \
    --output bots/CAKEUSDT/model/lit_int8.onnx \
    --data_path bots/CAKEUSDT/data/raw \
    --symbol CAKEUSDT \
    --skip_quality_check \
    --skip_benchmark
```

## Параметры

### Обязательные параметры

- `--input`: Путь к FP32 ONNX модели
- `--output`: Путь для сохранения INT8 модели
- `--data_path`: Путь к директории с parquet данными
- `--symbol`: Торговый символ (например, BTCUSDT)

### Опциональные параметры

- `--max_mcc_drop`: Максимально допустимое падение MCC (default: 0.02)
  - Меньше значение = строже контроль качества
  - Больше значение = больше узлов будет квантовано
  
- `--n_calibration_samples`: Количество снапшотов для калибровки (default: 1000)
  - Больше = лучше качество, но медленнее квантование
  - Рекомендуется: 500-1000
  
- `--seq_len`: Длина последовательности (default: 100)
- `--n_past_returns`: Количество past returns каналов (default: 0)

- `--skip_quality_check`: Пропустить проверку качества (Auto-Exclude)
- `--skip_benchmark`: Пропустить бенчмарк производительности

## Как это работает

### 1. Статическое квантование (PTQ)

Скрипт использует Post-Training Quantization (PTQ) - квантование после обучения:

1. **Калибровка**: Прогоняет 500-1000 примеров через модель для сбора статистики активаций
2. **Квантование**: Преобразует веса и активации в INT8
3. **Проверка качества**: Сравнивает MCC квантованной модели с оригиналом

### 2. Настройки квантования

- **Activations**: QInt8 (symmetric) - симметричное квантование для активаций
- **Weights**: QUInt8 (asymmetric) с `per_channel=True` - асимметричное поканальное квантование для весов
- **Format**: QDQ (Quantize-DeQuantize) - для лучшей совместимости

### 3. Auto-Exclude механизм

Если падение MCC превышает порог, скрипт автоматически:

1. Определяет проблемные узлы (Softmax, LayerNorm, финальный Linear)
2. Исключает их из квантования
3. Повторяет процесс (до 3 попыток)

Проблемные узлы остаются в FP32, остальные квантуются в INT8.

## Использование квантованной модели

### 1. Обновите config.toml

```toml
# Раскомментируйте для использования INT8 модели
model_path = "bots/CAKEUSDT/model/lit_int8.onnx"

[onnx]
execution_provider = "cpu"  # Рекомендуется для INT8 моделей
intra_threads = 4
```

**ВАЖНО**: INT8 квантованные модели оптимизированы для CPU с VNNI инструкциями.

### Execution Provider для INT8 моделей

- **CPU (рекомендуется)**: Оптимально для INT8 моделей на CPU с AVX-512/VNNI
- **CUDA**: Будет работать, но без преимуществ INT8 (используется FP32)
- **TensorRT**: НЕ рекомендуется для ONNX INT8 моделей
  - TensorRT требует собственного процесса INT8 калибровки
  - Используйте FP32 модель с TensorRT или INT8 модель с CPU

### 2. Запустите бота

```bash
cargo run --release
```

Rust код в `src/ml/onnx.rs` автоматически определит квантованную модель и выведет соответствующие сообщения.

## Бенчмарк

Скрипт автоматически замеряет производительность:

```
BENCHMARK RESULTS
================================================================================
FP32 Model: 2.456 ms ± 0.123 ms
INT8 Model: 0.812 ms ± 0.045 ms
Speedup: 3.02x
================================================================================
```

### Ожидаемое ускорение

- **CPU с AVX-512/VNNI** (Intel Ice Lake+, AMD Zen 4+): 2-4x
- **CPU с AVX2** (старые Intel/AMD): 1.5-2x
- **ARM с dot-product** (Apple M1+, AWS Graviton): 2-3x
- **Старые CPU**: может быть медленнее из-за overhead

## Проверка качества

### Метрика MCC

Matthews Correlation Coefficient (MCC) - сбалансированная метрика для классификации:

- **MCC = 1.0**: Идеальная классификация
- **MCC = 0.0**: Случайная классификация
- **MCC = -1.0**: Полностью неправильная классификация

### Типичные результаты

```
[Quality Check] Baseline MCC: 0.4523
[Quality Check] Quantized MCC: 0.4489
[Quality Check] MCC Drop: 0.0034 (threshold: 0.0200)
[Success] MCC drop within acceptable range!
```

### Если качество упало

1. **Увеличьте количество калибровочных данных**:
   ```bash
   --n_calibration_samples 2000
   ```

2. **Ослабьте порог**:
   ```bash
   --max_mcc_drop 0.03
   ```

3. **Проверьте исключенные узлы** в выводе скрипта

## Troubleshooting

### Ошибка: "No parquet files found"

Убедитесь, что данные существуют:
```bash
ls bots/CAKEUSDT/data/raw/CAKEUSDT_*.parquet
```

### Ошибка: "Input name not found"

Проверьте, что ONNX модель корректна:
```bash
python -c "import onnx; m = onnx.load('bots/CAKEUSDT/model/lit.onnx'); print(m.graph.input[0].name)"
```

### Квантованная модель медленнее FP32

Возможные причины:
1. Старый CPU без VNNI инструкций
2. Слишком много исключенных узлов
3. Overhead от QDQ операций

Решение: используйте FP32 модель или попробуйте GPU (CUDA/TensorRT).

### Предупреждение при использовании TensorRT с INT8 моделью

Если вы видите:
```
[WARN] INT8 quantized model detected with TensorRT provider!
[WARN] TensorRT requires its own INT8 calibration process.
```

Это нормально. TensorRT использует собственный механизм INT8 квантования, отличный от ONNX Runtime.

**Решения**:
1. **Рекомендуется**: Используйте `execution_provider = "cpu"` для INT8 моделей
2. Используйте FP32 модель с TensorRT (позвольте TensorRT выполнить собственное квантование)
3. Выполните INT8 калибровку через TensorRT API (выходит за рамки этой задачи)

## Дополнительные ресурсы

- [ONNX Runtime Quantization Guide](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [Задача №157: Экспорт квантованной модели](../docs/157-python-lab-export-quantized-int8-model.md)
- [Архитектура проекта](../docs/000-architecture.md)

## Примеры для разных символов

### BTCUSDT
```bash
python python_lab/scripts/quantize_onnx.py \
    --input bots/BTCUSDT/model/lit.onnx \
    --output bots/BTCUSDT/model/lit_int8.onnx \
    --data_path bots/BTCUSDT/data/raw \
    --symbol BTCUSDT
```

### FARTCOINUSDT
```bash
python python_lab/scripts/quantize_onnx.py \
    --input bots/FARTCOINUSDT/model/lit.onnx \
    --output bots/FARTCOINUSDT/model/lit_int8.onnx \
    --data_path bots/FARTCOINUSDT/data/raw \
    --symbol FARTCOINUSDT
```

## Автоматизация

Создайте скрипт для квантования всех моделей:

```bash
#!/bin/bash
# quantize_all.sh

for symbol in BTCUSDT CAKEUSDT FARTCOINUSDT; do
    echo "Quantizing $symbol..."
    python python_lab/scripts/quantize_onnx.py \
        --input bots/$symbol/model/lit.onnx \
        --output bots/$symbol/model/lit_int8.onnx \
        --data_path bots/$symbol/data/raw \
        --symbol $symbol \
        --max_mcc_drop 0.02
done
```

## Заключение

Квантование - мощный инструмент для ускорения инференса на CPU. Используйте его для production deployment, когда важна низкая латентность и эффективное использование ресурсов.
