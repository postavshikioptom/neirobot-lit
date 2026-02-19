# Задача 097: Реализация Padding и Truncation в TensorBuilder

**Цель**: Реализовать логику подготовки входного тензора фиксированной длины `seq_len` для ONNX-модели. Обеспечить корректное "дополнение нулями" (Left-Padding) при холодном старте и "отсечение" (Truncation) старых данных.

## 1. Изменения в [src/ml/tensor.rs](./src/ml/tensor.rs)

*   **Зависимости**:
    ```rust
    use std::collections::VecDeque;
    use ndarray::{Array4, s, prelude::*};
    use std::sync::{Arc, RwLock};
    // Импорт нормализации из задачи 034
    use crate::ml::normalization::normalize_snapshot; 
    ```

*   **Структура `TensorBuilder`**:
    Параметры `seq_len`, `channels` и `levels` должны инициализироваться из [metadata.json](./docs/056-export-model-metadata.md) при загрузке модели.
```rust
pub struct TensorBuilder {
    pub buffer: VecDeque<Snapshot>, // История снимков
    pub seq_len: usize,             // Из metadata.json
    pub channels: usize,            // Обычно 3 (Price, Vol, Imbalance)
    pub levels: usize,              // Обычно 50
}
```

*   **Логика Truncation (`add_snapshot`)**:
    При добавлении нового снимка через `push_back`, если `buffer.len() > seq_len`, вызывается `pop_front()`. Это обеспечивает скользящее окно (sliding window) актуальных данных.

*   **Логика Padding и сборки (`build_tensor`)**:
    Реализовать **Left-Padding** (заполнение нулями слева), чтобы самые свежие данные всегда находились в правой части тензора (индексы ближе к `seq_len - 1`).

```rust
pub fn build_tensor(&self) -> Array4<f32> {
    // Создаем пустой тензор (Batch=1, Channels, Levels, Time)
    let mut tensor = Array4::<f32>::zeros((1, self.channels, self.levels, self.seq_len));
    
    // Рассчитываем смещение для вставки (если данных < seq_len)
    let offset = self.seq_len.saturating_sub(self.buffer.len());
    
    for (i, snap) in self.buffer.iter().enumerate() {
        // 1. Нормализация данных (Z-score из задачи 034)
        let features: Array2<f32> = normalize_snapshot(snap); 
        
        // 2. Вставка в тензор со смещением offset
        // Используем slice_mut для выбора нужного временного шага
        tensor.slice_mut(s![0, .., .., i + offset])
              .assign(&features.view().into_shape((self.channels, self.levels)).unwrap());
    }
    
    tensor
}
```

## 2. Почему этот план лучше (Аргументы Grok):
*   **Metadata over Config**: Использование `seq_len` из `metadata.json` (задача 056) гарантирует, что Rust-бот всегда подает на вход ровно столько данных, сколько ожидает конкретная скомпилированная модель, предотвращая `RuntimeError` в `ort`.
*   **Right-Alignment (Left-Padding)**: Это стандарт для временных рядов. Модель обучается на "хвостах" последовательностей, и нули слева (Padding) интерпретируются как отсутствие данных в прошлом, не искажая текущий момент (t=0).
*   **Ndarray Efficiency**: Использование `slice_mut` и `assign` позволяет избежать лишних копирований памяти и аллокаций промежуточных векторов внутри цикла.
*   **Normalization Sync**: Прямая ссылка на [src/ml/normalization.rs](./src/ml/normalization.rs) (задача 034) обеспечивает идентичность данных между Python (при обучении) и Rust (при инференсе).

## 3. Критические требования
*   **Precision**: Все расчеты в `f32` (стандарт для ONNX).
*   **Zero-Fill**: Дополнение должно быть строго нулями (после Z-score нормализации `0.0` — это среднее значение, что является наиболее нейтральным входом).
*   **Concurrency**: Если `TensorBuilder` доступен из нескольких потоков (например, чтение данных и инференс), обернуть его в `Arc<RwLock<TensorBuilder>>`.

## 4. Тестирование
*   **Unit Test**: Подать 1 снимок при `seq_len=10`. Проверить, что тензор заполнен нулями в индексах `0..8` и данными в индексе `9`.
*   **Shape Test**: Убедиться, что `build_tensor().shape()` всегда возвращает `[1, channels, levels, seq_len]`.
