# Задача 096: Выбор и настройка Execution Providers для ONNX Runtime

**Цель**: Реализовать механизм гибкого выбора провайдеров исполнения (CPU, CUDA, TensorRT) в Rust-клиенте на уровне конфигурации конкретного бота. Это обеспечит максимальную производительность инференса без пересборки кода.

## 1. Изменения в [src/config/types.rs](./src/config/types.rs)
*   **OnnxConfig**: Создать структуру для настроек инференса.
*   **BotConfig**: Добавить `OnnxConfig` как поле, чтобы каждый бот мог использовать свой тип ускорения.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxConfig {
    #[serde(default = "default_ep")]
    pub execution_provider: String, // "cpu", "cuda", "tensorrt"
    #[serde(default = "default_device_id")]
    pub device_id: i32,             // ID GPU (по умолчанию 0)
    pub intra_threads: Option<usize>, // Для CPU-оптимизации
}

fn default_ep() -> String { "cpu".to_string() }
fn default_device_id() -> i32 { 0 }

pub struct BotConfig {
    // ...
    pub onnx: OnnxConfig,
}
```

## 2. Изменения в [src/ml/onnx.rs](./src/ml/onnx.rs)
*   **Метод `init_session`**: Реализовать логику подключения провайдеров с автоматическим **fallback** на CPU.

```rust
use ort::{
    session::{builder::GraphOptimizationLevel, Session, SessionBuilder},
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider, TensorRTExecutionProvider},
};

pub fn init_session(config: &OnnxConfig, model_path: &str) -> ort::Result<Session> {
    let mut builder = Session::builder()?;
    
    // Настройка оптимизаций
    builder = builder.with_optimization_level(GraphOptimizationLevel::Level3)?;

    match config.execution_provider.as_str() {
        "cuda" => {
            // Пытаемся подключить CUDA, при ошибке — откатываемся на CPU
            if let Err(e) = builder.with_execution_providers([
                CUDAExecutionProvider::default().with_device_id(config.device_id)
            ]) {
                tracing::warn!("Failed to initialize CUDA: {}. Falling back to CPU.", e);
                builder = builder.with_execution_providers([CPUExecutionProvider::default()])?;
            }
        }
        "tensorrt" => {
            if let Err(e) = builder.with_execution_providers([
                TensorRTExecutionProvider::default().with_device_id(config.device_id)
            ]) {
                tracing::warn!("Failed to initialize TensorRT: {}. Falling back to CPU.", e);
                builder = builder.with_execution_providers([CPUExecutionProvider::default()])?;
            }
        }
        _ => {
            builder = builder.with_execution_providers([CPUExecutionProvider::default()])?;
            if let Some(threads) = config.intra_threads {
                builder = builder.with_intra_threads(threads)?;
            }
        }
    }

    builder.with_model_from_file(model_path)
}
```

## 3. Изменения в [Cargo.toml](./Cargo.toml)
*   Использовать актуальную версию `ort` и включить необходимые фичи:
```toml
ort = { version = "2.0.0-rc.6", features = ["ndarray", "cuda", "tensorrt"] }
```

## 4. Почему это важно (Аргументы Grok)
*   **Latency**: Инференс на CUDA может быть в 10 раз быстрее CPU. Для HFT-стратегий задержка более 10мс делает сигнал неактуальным.
*   **Per-Bot Isolation**: Каждый бот (символ) может использовать свой ресурс. Например, BTC на GPU, а низколиквидный альткоин — на CPU.
*   **Resilience**: Автоматический переход на CPU позволяет боту запуститься даже если возникли проблемы с драйверами NVIDIA или версиями CUDA.

## 5. Критические требования и примечания
*   **Environment**: Системные настройки (например, `LD_LIBRARY_PATH` для Linux) должны быть описаны в [docs/096-notes.md](./docs/096-notes.md), а не в коде.
*   **TensorRT SDK**: Убедиться, что для работы TensorRT в системе установлены соответствующие библиотеки (обычно `libnvinfer`).
*   **Logging**: При успешной инициализации выводить в `info!` тип активного провайдера.

## 6. Тестирование
*   **Unit test**: Запуск `init_session` с фиктивным путем к модели и проверка, что ошибка `model not found` возвращается после попытки инициализации EP.
*   **Integration test**: Запуск бота на машине с GPU и проверка логов на отсутствие `warn! Falling back to CPU`.
