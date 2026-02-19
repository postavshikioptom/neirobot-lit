Задача 099: ML Tensor Safe Math Normalization
1. Цель
Реализовать безопасную нормализацию (z-score) в src/ml/tensor.rs. При низком объеме или "замершем" рынке стандартное отклонение (std) может быть равно 0, что при делении приведет к NaN или Inf. Это "отравит" веса модели и приведет к краху торговой логики.

2. Изменения
Файл: src/ml/tensor.rs
Константа EPSILON: Добавить const EPSILON: f32 = 1e-8;.
Метод safe_normalize:
Реализовать функцию или метод для Array4<f32>, который применяет параметры из norm.json (загруженные в задаче 034).
Вместо прямого деления (x - mean) / std, использовать зажим (clamping) знаменателя.
Пример реализации:
use ndarray::{Array4, s};

pub fn apply_normalization(tensor: &mut Array4<f32>, means: &[f32], stds: &[f32]) {
    // Проходим по каналам (CHANNELS), так как у каждого свои mean/std
    for c in 0..tensor.shape()[1] {
        let m = means[c];
        let s = stds[c].max(1e-8); // Защита от деления на 0
        
        let mut slice = tensor.slice_mut(s![.., c, .., ..]);
        slice.mapv_inplace(|x| (x - m) / s);
    }
}
3. Критические требования
Отсутствие NaN: После выполнения нормализации тензор не должен содержать NaN. Добавить debug_assert!(tensor.iter().all(|x| x.is_finite()));.
Производительность: Использовать mapv_inplace, чтобы не аллоцировать новый тензор при нормализации "на лету" перед инференсом.
Типы данных: Работать строго с f32 (стандарт для ONNX/TensorRT).