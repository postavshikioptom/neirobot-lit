Задача 098: ML ONNX Dynamic Batching Disable (Фиксация размерностей)
1. Цель
Отключить динамические размерности (Dynamic Axes) при экспорте модели и в конфигурации сессии ort. Это критично для оптимизации TensorRT, так как фиксированный batch_size=1 позволяет исключить накладные расходы на перестроение графа и выделение памяти.

2. Изменения
Python: python_lab/scripts/export_onnx.py
Удаление dynamic_axes: При вызове torch.onnx.export полностью удалить аргумент dynamic_axes.
Явная фиксация: Использовать dummy_input с формой (1, CHANNELS, LEVELS, SEQ_LEN).
Post-export Fix: Если модель была экспортирована с динамическими осями ранее, добавить шаг принудительной фиксации:
import onnx
from onnxruntime.tools import make_dynamic_shape_fixed
# Фиксируем 'batch' в 1
make_dynamic_shape_fixed.make_dynamic_shape_fixed('batch', 1, 'lit.onnx', 'lit_fixed.onnx')
Rust: src/ml/onnx.rs
Настройка SessionBuilder:
Использовать GraphOptimizationLevel::All вместо устаревшего Level3.
Для TensorRT задать профили, где min == opt == max:
let shape = vec![1, channels, levels, seq_len];
let trt_provider = TensorRTExecutionProvider::default()
    .with_profile_min_shape("input", shape.clone())
    .with_profile_opt_shape("input", shape.clone())
    .with_profile_max_shape("input", shape.clone());
Валидация при загрузке:
В методе init_session добавить проверку: let input_dim = self.session.inputs()[0].dimensions().collect::<Vec<_>>();.
Если input_dim[0] != Some(1), выводить tracing::warn!("Dynamic batch size detected, performance might be sub-optimal").
3. Критические требования
Zero-copy: Убедиться, что ndarray::Array4 передается в ort::Value::from_array без лишних аллокаций.
Batch=1: Любые попытки подать тензор с batch > 1 должны приводить к ошибке в Rust-слое до вызова ONNX.