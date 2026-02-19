# 037 - ML ONNX Inference
Цель задачи: Реализовать метод predict в структуре OnnxEngine. Метод должен преобразовывать входной вектор признаков в тензор ndarray, выполнять инференс через ort, проверять корректность выходных данных и возвращать результат с примененным softmax.

Файлы:

Cargo.toml (обновить)
src/ml/onnx.rs (обновить)
Инструкции для Gemini:

Добавить зависимость в Cargo.toml:
ndarray = "0.15"
Обновить структуру и реализовать метод predict в src/ml/onnx.rs: Убедиться, что OnnxEngine хранит количество признаков (input_features), полученное при загрузке.
use crate::ml::types::{Signal, InferenceOutput};
use ort::Value;
use ndarray::ArrayViewD;
use anyhow::{Result, Context, bail};

impl OnnxEngine {
    /// Выполнение инференса [1, seq_len, input_features] -> InferenceOutput
    pub fn predict(&self, input_data: &[f32]) -> Result<InferenceOutput> {
        // 1. Проверка размера входного буфера
        let expected_len = self.seq_len * self.input_features;
        if input_data.len() != expected_len {
            bail!("Input data length mismatch: expected {}, got {}", expected_len, input_data.len());
        }

        // 2. Создание тензора без копирования через ArrayView
        let input_shape = [1, self.seq_len, self.input_features];
        let input_tensor = Value::from_array(
            self.session.allocator(),
            ArrayViewD::from_shape(&input_shape[..], input_data)
                .context("Failed to create ndarray view")?
        ).context("Failed to create ONNX input tensor")?;

        // 3. Запуск инференса
        let outputs = self.session.run(ort::inputs![input_tensor]?)?;
        let output_tensor = outputs[0].try_extract_tensor::<f32>()
            .context("Failed to extract output tensor")?;
        
        let logits = output_tensor.as_slice()
            .context("Failed to get logits as slice")?;

        // 4. Валидация выхода (должно быть 3 класса: Flat, Up, Down)
        if logits.len() != 3 {
            bail!("Model output size mismatch: expected 3 logits, got {}", logits.len());
        }

        // 5. Softmax и выбор сигнала
        let probabilities = self.softmax(logits);
        let max_idx = probabilities.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        Ok(InferenceOutput {
            signal: Signal::from(max_idx),
            probabilities,
        })
    }

    /// Численно стабильный Softmax
    fn softmax(&self, logits: &[f32]) -> [f32; 3] {
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        
        [exps[0] / sum, exps[1] / sum, exps[2] / sum]
    }
}
Технические требования:

Ndarray: Использовать ArrayViewD для минимизации аллокаций при создании входного тензора.
Безопасность: Обязательно проверять длину logits (должна быть 3) перед доступом к индексам массива.
Softmax: Использовать max_val (центрирование) для предотвращения overflow при вычислении экспонент.
Поля: Убедиться, что OnnxEngine имеет pub input_features: usize (сохраненный в load).