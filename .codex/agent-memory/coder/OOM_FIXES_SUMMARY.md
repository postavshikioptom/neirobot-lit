# OOM Memory Optimizations Implementation

Дата: 2026-03-21

## Реализованные фиксы для борьбы с CUDA OOM:

### 1. Mixed Precision (train.py)
- Изменен `precision="32"` на `precision="16-mixed" if torch.cuda.is_available() else 32`
- Сокращает потребление памяти ~2x

### 2. Gradient Accumulation (train.py)
- Добавлен аргумент `--accumulate_grad_batches` (default=1)
- Передан в `pl.Trainer` как `accumulate_grad_batches=args.accumulate_grad_batches`
- Позволяет использовать эффективный batch size больше, чем помещается в память

### 3. Gradient Checkpointing (lit_model.py + train.py)
**В lit_model.py:**
- Добавлен флаг `use_gradient_checkpointing: bool = False` в `LiTConfig` dataclass
- `CustomTransformerEncoderLayer.__init__` теперь принимает `use_gradient_checkpointing` и сохраняет как `self.use_gradient_checkpointing`
- `CustomTransformerEncoderLayer.forward` модифицирован: при `self.use_gradient_checkpointing and self.training` использует `torch.utils.checkpoint.checkpoint` для `_sa_block` и `_ff_block` с `use_reentrant=False`
- Сохранены все `nan_to_num` защиты и residual connections

**В LiTModel:**
- `__init__` принимает `use_gradient_checkpointing=False`
- Сохраняет как `self.use_gradient_checkpointing`
- Передает флаг в каждый `CustomTransformerEncoderLayer` при создании

**В train.py:**
- Добавлен CLI аргумент `--use_gradient_checkpointing` (store_true)
- Передан в `LiTModule` через `model_params` во всех местах создания:
  - trial_model (Optuna)
  - student model (distillation)
  - teacher model (main training)
  - fold_model (cross-validation)

### Важные детали:
- `use_reentrant=False` в checkpoint для стабильности с dropout
- Все residual connections сохранены
- Все `nan_to_num` защиты сохранены
- Backward compatibility: все флаги по умолчанию False
- Mixed precision автоматически включается только при availability CUDA

## Файлы, измененные:
- `E:\MAX\PYTHON\NEURAL-BOTS\neirobot-lit\python_lab\src\train.py`
- `E:\MAX\PYTHON\NEURAL-BOTS\neirobot-lit\python_lab\src\lit_model.py`
