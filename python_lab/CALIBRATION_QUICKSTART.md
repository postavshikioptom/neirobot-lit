# Быстрый старт: Калибровка модели

## Шаг 1: Обучите модель

```bash
python python_lab/scripts/train.py --symbol BTCUSDT --epochs 100
```

## Шаг 2: Калибруйте модель

```bash
python python_lab/calibrate.py \
    --symbol BTCUSDT \
    --checkpoint bots/BTCUSDT/model/checkpoints/lit-epoch=XX-val_mcc=X.XXXX.ckpt
```

Результат:
- Оптимальная температура сохранена в `bots/BTCUSDT/model/metadata.json`
- Графики калибровки в `bots/BTCUSDT/model/reports/`

## Шаг 3: Экспортируйте модель

### Вариант A: Температура в Rust (рекомендуется)

```bash
python python_lab/src/export_onnx.py \
    --input bots/BTCUSDT/model/checkpoints/lit-epoch=XX-val_mcc=X.XXXX.ckpt \
    --output bots/BTCUSDT/model/lit.onnx
```

### Вариант B: Температура в ONNX

```bash
python python_lab/src/export_onnx.py \
    --input bots/BTCUSDT/model/checkpoints/lit-epoch=XX-val_mcc=X.XXXX.ckpt \
    --output bots/BTCUSDT/model/lit.onnx \
    --embed_temperature
```

## Шаг 4: Запустите бота

```bash
cargo run --release -- --symbol BTCUSDT
```

Rust автоматически применит калибровку!

## Проверка результатов

Откройте графики в `bots/BTCUSDT/model/reports/`:
- `reliability_before_calibration.png` - до калибровки
- `reliability_after_calibration.png` - после калибровки

Хорошая калибровка: линия близка к диагонали (y = x)
