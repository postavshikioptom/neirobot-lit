use serde::{Deserialize, Serialize};
use ndarray::Array2;

/// Параметры нормализации модели (mean и std для Z-score, или median и iqr для Robust Scaler)
/// Задача 240: Поддержка разных типов скейлеров
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationParams {
    /// Тип скейлера: "zscore", "robust", "winsor_robust"
    #[serde(default = "default_scaler_type")]
    pub scaler_type: String,
    
    /// Параметры для Z-score нормализации
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub std: Option<Vec<f32>>,
    
    /// Параметры для Robust Scaling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub median: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iqr: Option<Vec<f32>>,
    
    /// Пороги винзоризации для WinsorRobust (Задача 240)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub winsor_limits: Option<Vec<f32>>,
}

fn default_scaler_type() -> String {
    "zscore".to_string()
}

/// Метаданные текущей активной модели (metadata.json)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub onnx_hash: String,
    pub version: String,
    pub normalization: NormalizationParams,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mcc_score: Option<f32>,
    /// Среднее значение энтропии предсказаний на валидационной выборке (задача 224)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline_entropy_mean: Option<f32>,
    /// Стандартное отклонение энтропии на валидационной выборке (задача 224)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline_entropy_std: Option<f32>,
}

/// Запись в реестре моделей (registry.json)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistryEntry {
    pub version_tag: String,
    pub onnx_hash: String,
    pub mcc_score: f32,
    pub created_at: String,
    pub file_path: String,
}

/// Реестр всех версий моделей (registry.json)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistry {
    pub entries: Vec<ModelRegistryEntry>,
}

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

/// Сигнал с временной меткой для замера latency (Задача 201)
/// Содержит направление торговли и монотонный таймер для точного измерения задержки
#[derive(Debug, Clone)]
pub struct SignalWithTimestamp {
    pub signal: Signal,
    pub start_instant: std::time::Instant,
}

/// Результат инференса модели LiT
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceOutput {
    pub signal: Signal,             // Up, Down, Flat
    pub probabilities: Vec<f32>,    // [prob_flat, prob_up, prob_down] - matches Signal enum indices
    /// Матрица формы [Horizons, 3], где 3 — это классы [Flat, Up, Down]
    pub probs: Array2<f32>,
    /// Время получения исходного снепшота стакана (receive_ts) в миллисекундах (Задача 169)
    pub source_timestamp_ms: u64,
    /// Энтропия предсказания H = -Σ p_i * log(p_i) (задача 224)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entropy: Option<f32>,
    /// Флаг обнаружения дрейфа модели (задача 224)
    #[serde(default)]
    pub drift_detected: bool,
}
