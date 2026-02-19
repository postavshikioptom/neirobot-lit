# Команды для тестирования функций активации

## 1. Тест архитектуры модели

Проверка, что модель корректно создается с разными активациями:

```bash
python python_lab/src/lit_model.py
```

Ожидаемый вывод:
```
Testing with 3 channels (baseline) and GELU (exact):
Input shape: torch.Size([8, 100, 3, 50])
Output shape: torch.Size([8, 3])
Total parameters: 123,456

Testing with 6 channels (3 baseline + 3 past returns) and SiLU:
Input shape: torch.Size([8, 100, 6, 50])
Output shape: torch.Size([8, 3])
Total parameters: 234,567

Testing all activation types:
  relu: Output shape torch.Size([8, 3]), Mean: 0.1234
  gelu_exact: Output shape torch.Size([8, 3]), Mean: 0.2345
  gelu_tanh: Output shape torch.Size([8, 3]), Mean: 0.2346
  silu: Output shape torch.Size([8, 3]), Mean: 0.3456
```

## 2. Тест паритета активаций

Проверка, что все активации корректно экспортируются в ONNX:

```bash
python python_lab/scripts/test_activation_parity.py
```

Ожидаемый вывод:
```
ACTIVATION PARITY TEST SUITE
============================================================

Testing Activation: RELU
============================================================
Exporting to ONNX (opset 17)...
Loading ONNX model...

Parity Check Results:
  Max difference: 1.19e-07
  Mean difference: 2.34e-08
  Tolerance: 1.00e-06
✓ Parity test PASSED for relu

============================================================
Testing Gradient Flow: RELU
============================================================

Gradient Statistics:
  Gradient norm: 12.3456
  Gradient mean (abs): 0.001234
✓ Gradient flow test PASSED for relu

[... аналогично для gelu_exact, gelu_tanh, silu ...]

============================================================
FINAL RESULTS
============================================================

Parity Tests:
  relu           : ✓ PASSED
  gelu_exact     : ✓ PASSED
  gelu_tanh      : ✓ PASSED
  silu           : ✓ PASSED

Gradient Flow Tests:
  relu           : ✓ PASSED
  gelu_exact     : ✓ PASSED
  gelu_tanh      : ✓ PASSED
  silu           : ✓ PASSED

============================================================
ALL TESTS PASSED ✓
============================================================
```

## 3. Обучение с разными активациями

### GELU (exact)
```bash
python -m python_lab.scripts.train \
  --symbol CAKEUSDT \
  --activation gelu_exact \
  --epochs 10 \
  --batch_size 128 \
  --seq_len 100
```

### SiLU
```bash
python -m python_lab.scripts.train \
  --symbol CAKEUSDT \
  --activation silu \
  --epochs 10 \
  --batch_size 128 \
  --seq_len 100
```

### GELU (tanh)
```bash
python -m python_lab.scripts.train \
  --symbol CAKEUSDT \
  --activation gelu_tanh \
  --epochs 10 \
  --batch_size 128 \
  --seq_len 100
```

### ReLU
```bash
python -m python_lab.scripts.train \
  --symbol CAKEUSDT \
  --activation relu \
  --epochs 10 \
  --batch_size 128 \
  --seq_len 100
```

Ожидаемый вывод (фрагмент):
```
Initializing model with loss: ce (gamma=N/A), activation: gelu_exact
Starting training...
Epoch 0 Validation: MCC=0.1234, Macro-F1=0.4567, Balanced-Acc=0.5678
Epoch 1 Validation: MCC=0.2345, Macro-F1=0.5678, Balanced-Acc=0.6789
...
```

## 4. Автоматический подбор с Optuna

```bash
python -m python_lab.scripts.tune \
  --symbol CAKEUSDT \
  --trials 10
```

Ожидаемый вывод (фрагмент):
```
Starting HPO for CAKEUSDT (Trials: 10)...
[I 2024-01-01 12:00:00,000] Trial 0 finished with value: 0.3456 and parameters: {'activation': 'gelu_exact', ...}
[I 2024-01-01 12:05:00,000] Trial 1 finished with value: 0.3789 and parameters: {'activation': 'silu', ...}
...
==============================
Best MCC: 0.4523
Best params: {'activation': 'silu', 'd_model': 64, 'nhead': 4, ...}
==============================
```

## 5. Экспорт модели

```bash
python -m python_lab.scripts.export_onnx \
  --input bots/CAKEUSDT/models/checkpoints/lit-epoch=05-val_mcc=0.4523.ckpt \
  --output bots/CAKEUSDT/models/lit.onnx
```

Ожидаемый вывод:
```
Loading model from bots/CAKEUSDT/models/checkpoints/lit-epoch=05-val_mcc=0.4523.ckpt...
Exporting to ONNX (opset 17), seq_len=100, activation=silu...
Metadata saved to bots/CAKEUSDT/models/metadata.json
Verifying ONNX model...
✓ Model exported to bots/CAKEUSDT/models/lit.onnx and verified successfully.
  Max difference: 3.45e-07
```

Проверка metadata.json:
```bash
cat bots/CAKEUSDT/models/metadata.json
```

Ожидаемый вывод:
```json
{
    "model_name": "LiT",
    "seq_len": 100,
    "n_levels": 50,
    "in_channels": 6,
    "past_returns_lags": [10, 50, 100],
    "activation": "silu",
    "onnx_file": "lit.onnx"
}
```

## 6. Просмотр результатов в TensorBoard

```bash
tensorboard --logdir tb_logs
```

Откройте браузер: http://localhost:6006

Сравните метрики для разных активаций:
- `val_mcc` - Основная метрика
- `val_f1_macro` - Macro F1-score
- `val_balanced_acc` - Balanced Accuracy

## Примечания

- Все команды предполагают, что вы находитесь в корневой директории проекта
- Для Windows используйте `python` вместо `python3`
- Для полного обучения увеличьте `--epochs` до 100
- Для Optuna увеличьте `--trials` до 50 для лучших результатов
