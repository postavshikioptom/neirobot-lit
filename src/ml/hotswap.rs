//! Hot-Swap Engine для безопасной замены ONNX моделей без остановки бота
//!
//! Задача 228: Автоматизированная дистрибуция и безопасный Hot-Swap моделей
//!
//! Этот модуль реэкспортирует HotSwapEngine из onnx_engine для обратной совместимости.

pub use crate::ml::onnx_engine::HotSwapEngine;
