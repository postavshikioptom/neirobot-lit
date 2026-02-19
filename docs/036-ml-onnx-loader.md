# 036 - ML ONNX Loader
Цель задачи: Реализовать структуру OnnxEngine для загрузки модели lit.onnx с использованием библиотеки ort. Необходимо настроить параметры сессии для минимальной задержки (low-latency) и добавить строгую валидацию размерностей входного и выходного тензоров.

Файлы:

src/ml/onnx.rs (создать)
src/ml/mod.rs (обновить)
Инструкции для Gemini:

Добавить зависимость в Cargo.toml:
ort = { version = "2.0", features = ["load-dynamic"] }
src/ml/onnx.rs: Реализовать загрузку модели с явным указанием провайдера (CPU) и ограничением потоков.
use ort::{
    Session, SessionBuilder, GraphOptimizationLevel, 
    LoggingLevel, ExecutionProviderDispatch, CPUExecutionProvider
};
use std::path::Path;
use anyhow::{Result, Context, bail};

pub struct OnnxEngine {
    pub session: Session,
    pub seq_len: usize,
    pub input_features: usize,
}

impl OnnxEngine {
    pub fn load(model_path: &Path, seq_len: usize, input_features: usize) -> Result<Self> {
        // 1. Настройка сессии для Low-Latency
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)? // Минимум потоков для исключения конкуренции CPU
            .with_inter_threads(1)?
            .with_log_level(LoggingLevel::Warning)?
            .with_execution_providers([
                ExecutionProviderDispatch::Cpu(CPUExecutionProvider::default())
            ])?
            .commit_from_file(model_path)
            .context("Failed to load ONNX model")?;

        // 2. Валидация входного тензора [batch, seq_len, features]
        let input0 = &session.inputs[0];
        let shape = input0.input_type.as_tensor_type()
            .context("Input 0 is not a tensor")?.shape.clone();

        // Проверка seq_len (dim 1)
        if let Some(dim) = shape.get(1) {
            if let Some(fixed) = dim.as_fixed() {
                if fixed as usize != seq_len {
                    bail!("Model seq_len mismatch: expected {}, got {}", seq_len, fixed);
                }
            }
        }

        // Проверка features (dim 2)
        if let Some(dim) = shape.get(2) {
            if let Some(fixed) = dim.as_fixed() {
                if fixed as usize != input_features {
                    bail!("Model features mismatch: expected {}, got {}", input_features, fixed);
                }
            }
        }

        // 3. Валидация выходного тензора [batch, 3]
        let output0 = &session.outputs[0];
        let out_shape = output0.output_type.as_tensor_type()
            .context("Output 0 is not a tensor")?.shape.clone();

        if let Some(dim) = out_shape.get(1) {
            if let Some(fixed) = dim.as_fixed() {
                if fixed != 3 {
                    bail!("Model must have 3 output classes (Flat, Up, Down), got {}", fixed);
                }
            }
        }

        Ok(Self { session, seq_len, input_features })
    }
}
Технические требования:

Потоки: Строго intra_threads(1) и inter_threads(1) для предотвращения задержек на переключение контекста.
Провайдер: Явно указать ExecutionProviderDispatch::Cpu.
Логирование: Установить LoggingLevel::Warning.
Валидация: Проверять seq_len и input_features (обычно 200) по Dimension::as_fixed().