# 033 - ML Types Definition
Цель задачи: Определить базовые типы данных для модуля машинного обучения в Rust. Эти структуры станут «контрактом» между результатами работы ONNX-модели и торговой стратегией.

Файлы:

src/ml/mod.rs (создать)
src/ml/types.rs (создать)
Инструкции для Gemini:

src/ml/types.rs: Реализовать перечисление сигналов и структуру выхода инференса.
use serde::{Deserialize, Serialize};

/// Сигналы предсказания модели (согласно задаче 023)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Signal {
    Flat = 0,
    Up = 1,
    Down = 2,
}

impl From<usize> for Signal {
    fn from(v: usize) -> Self {
        match v {
            1 => Signal::Up,
            2 => Signal::Down,
            _ => Signal::Flat,
        }
    }
}

/// Результат инференса модели LiT
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceOutput {
    pub signal: Signal,
    pub probabilities: [f32; 3], // [Flat, Up, Down]
}
src/ml/mod.rs:
pub mod types;
pub use types::*;
Технические требования:

Пути: Использовать папку src/ml/ (согласно архитектуре проекта).
Имена: Строго Signal (enum) и InferenceOutput (struct).
Минимализм: Не добавлять в InferenceOutput метаданные (symbol, timestamp) — они управляются на уровне выше в модуле инференса.
Serde: Добавить Serialize/Deserialize для интеграции с логами и мониторингом.