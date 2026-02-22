use ort::{
    Session, SessionBuilder, GraphOptimizationLevel, 
    LoggingLevel, inputs, ep, ExecutionMode, Value
};
use ndarray::ArrayView4;
use std::path::Path;
use std::time::Instant;
use std::fs::File;
use std::io::Read;
use anyhow::{Result, Context, bail};
use tracing::{warn, info, error};
use sha2::{Sha256, Digest};
use crate::ml::types::{Signal, InferenceOutput, ModelMetadata};
use crate::monitoring::latency::HOT_PATH_STATS;
use crate::config::types::{OnnxConfig, OnnxExecutionMode, BotConfig};
use crate::ml::tensor::TensorBuffer;
use std::cell::RefCell;

impl From<OnnxExecutionMode> for ExecutionMode {
    fn from(mode: OnnxExecutionMode) -> Self {
        match mode {
            OnnxExecutionMode::Sequential => ExecutionMode::Sequential,
            OnnxExecutionMode::Parallel => ExecutionMode::Parallel,
        }
    }
}

/// Результат инференса с метриками производительности (Задача 169)
#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub output: crate::ml::types::InferenceOutput,
    pub duration_us: u64, // Время выполнения модели в микросекундах
}

/// Вычисляет SHA-256 хэш файла
/// 
/// # Аргументы
/// * `file_path` - Путь к файлу для хэширования
/// 
/// # Возвращает
/// Строку с hex-представлением SHA-256 хэша
fn compute_file_hash(file_path: &Path) -> Result<String> {
    let mut file = File::open(file_path)
        .with_context(|| format!("Failed to open file for hashing: {}", file_path.display()))?;
    
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];
    
    loop {
        let bytes_read = file.read(&mut buffer)
            .with_context(|| format!("Failed to read file: {}", file_path.display()))?;
        
        if bytes_read == 0 {
            break;
        }
        
        hasher.update(&buffer[..bytes_read]);
    }
    
    let hash_bytes = hasher.finalize();
    Ok(format!("{:x}", hash_bytes))
}

/// Вычисляет энтропию распределения вероятностей (задача 224)
/// 
/// Формула: H = -Σ p_i * log(p_i)
/// 
/// # Аргументы
/// * `probs` - Вероятности классов (должны суммироваться в 1.0)
/// 
/// # Возвращает
/// Энтропию в натах (использует натуральный логарифм)
/// 
/// # Примечания
/// - Использует f32 для минимизации нагрузки на FPU
/// - Обрабатывает случай p_i = 0 (0 * log(0) = 0 по определению)
#[inline]
fn calculate_entropy(probs: &[f32]) -> f32 {
    probs.iter()
        .filter(|&&p| p > 0.0)  // Исключаем нулевые вероятности
        .map(|&p| -p * p.ln())
        .sum()
}

/// Обновляет экспоненциальное скользящее среднее (EMA) (задача 224)
/// 
/// Формула: EMA = alpha * new_value + (1 - alpha) * current_ema
/// 
/// # Аргументы
/// * `current_ema` - Текущее значение EMA
/// * `new_value` - Новое значение для добавления
/// * `alpha` - Коэффициент сглаживания (обычно 0.1)
/// 
/// # Возвращает
/// Обновленное значение EMA
#[inline]
fn update_ema(current_ema: f32, new_value: f32, alpha: f32) -> f32 {
    alpha * new_value + (1.0 - alpha) * current_ema
}

/// Инициализирует ONNX сессию с выбранным Execution Provider
pub fn init_session(config: &OnnxConfig, model_path: &Path, symbol: &str, seq_len: usize, input_features: usize) -> Result<Session> {
    let model_name = model_path.to_string_lossy();
    let is_fp16 = model_name.contains("_fp16") || model_name.contains(".fp16");
    let is_int8 = model_name.contains("_int8") || model_name.contains(".int8");

    let mut builder = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::All)?
        .with_log_level(LoggingLevel::Warning)?;

    // Настройка пулов потоков и режима исполнения (Задача №100)
    let mut builder = builder
        .with_execution_mode(config.execution_mode.into())?;
    
    // Расчёт intra_threads: физические ядра / количество_активных_ботов (минимум 1)
    let intra = config.intra_threads.unwrap_or_else(|| {
        let cp_total = num_cpus::get_physical();
        // Получаем количество активных ботов из переменной окружения или используем консервативный дефолт (2)
        let active_bots = std::env::var("NUM_ACTIVE_BOTS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(2); // Консервативный дефолт: 2 бота
        (cp_total / active_bots).max(1)
    });
    builder = builder.with_intra_op_num_threads(intra as i32)?;
    
    // Расчёт inter_threads: 1 для небольших LOB моделей (избегаем переключений контекста)
    let inter = config.inter_threads.unwrap_or(1);
    builder = builder.with_inter_op_num_threads(inter as i32)?;
    
    info!(
        "[ML] ONNX Runtime configured: intra={}, inter={}, mode={:?}, active_bots={}",
        intra, inter, config.execution_mode,
        std::env::var("NUM_ACTIVE_BOTS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(2)
    );

    match config.execution_provider.as_str() {
        "cuda" => {
            info!("Attempting to initialize CUDA execution provider (device_id: {}, fp16: {})", config.device_id, is_fp16);
            // Пытаемся подключить CUDA, при ошибке — откатываемся на CPU
            match builder.with_execution_providers([
                ep::CUDA::default()
                    .with_device_id(config.device_id as i32)
                    .build()
            ]) {
                Ok(b) => {
                    builder = b;
                    if is_fp16 {
                        info!("✓ CUDA execution provider initialized with FP16 acceleration");
                    } else {
                        info!("✓ CUDA execution provider initialized successfully");
                    }
                }
                Err(e) => {
                    warn!("Failed to initialize CUDA: {}. Falling back to CPU.", e);
                    builder = builder.with_execution_providers([ep::CPU::default().build()])?;
                    info!("Using CPU execution provider (CUDA fallback)");
                }
            }
        }
        "tensorrt" => {
            info!("Attempting to initialize TensorRT execution provider (device_id: {}, symbol: {})", config.device_id, symbol);
            
            // ВАЖНО: TensorRT с INT8 квантованными моделями
            // TensorRT требует собственного процесса калибровки для INT8 через TensorRT API.
            // Квантованные ONNX модели (созданные через onnxruntime.quantization) оптимизированы
            // для CPU с VNNI инструкциями, а не для TensorRT.
            // 
            // Для INT8 на TensorRT рекомендуется:
            // 1. Использовать FP32 модель с TensorRT
            // 2. Позволить TensorRT выполнить собственное INT8 квантование
            // 3. Или использовать INT8 модель на CPU (execution_provider = "cpu")
            if is_int8 {
                warn!("INT8 quantized model detected with TensorRT provider!");
                warn!("TensorRT requires its own INT8 calibration process.");
                warn!("For best INT8 performance, use execution_provider = \"cpu\" with VNNI support.");
                warn!("Proceeding with TensorRT FP16 mode...");
            }
            
            let cache_path = format!("bots/{}/model/trt_cache", symbol);
            let shape_str = format!("input:1x{}x{}", seq_len, input_features);

            // Настройка TensorRT с изолированным кэшем и фиксированным профилем
            // Используем FP16 для оптимальной производительности на GPU
            let trt_options = ep::TensorRT::default()
                .with_device_id(config.device_id as i32)
                .with_engine_cache(true)
                .with_engine_cache_path(&cache_path)
                .with_fp16(true)  // FP16 для GPU, INT8 требует отдельной калибровки TensorRT
                .with_builder_optimization_level(3);

            match builder.with_execution_providers([
                trt_options.build(),
                ep::CUDA::default()
                    .with_device_id(config.device_id as i32)
                    .build()
            ]) {
                Ok(b) => {
                    builder = b;
                    info!("✓ TensorRT execution provider initialized with fixed profiles and isolated cache: {}", cache_path);
                }
                Err(e) => {
                    warn!("Failed to initialize TensorRT: {}. Falling back to CPU.", e);
                    builder = builder.with_execution_providers([ep::CPU::default().build()])?;
                    info!("Using CPU execution provider (TensorRT fallback)");
                }
            }
        }
        _ => {
            info!("Using CPU execution provider");
            
            // CPU - оптимальный выбор для INT8 квантованных моделей
            // Современные CPU с AVX-512/VNNI (Intel Ice Lake+, AMD Zen 4+) обеспечивают
            // 2-4x ускорение для INT8 моделей по сравнению с FP32
            if is_int8 {
                info!("INT8 quantized model detected - optimal for CPU with VNNI support");
            }
            
            builder = builder.with_execution_providers([ep::CPU::default().build()])?;
        }
    }

    builder.commit_from_file(model_path)
        .context("Failed to load ONNX model")
}


/// Трекер уверенности инференса для мониторинга здоровья модели (задача 224)
/// 
/// Отслеживает распределение вероятностей и энтропию предсказаний в реальном времени,
/// сравнивая их с базовыми значениями (Baseline) для обнаружения деградации модели.
struct InferenceConfidenceTracker {
    /// EMA энтропии предсказаний
    ema_entropy: f32,
    /// Счетчик инференсов для сэмплирования
    sample_counter: u32,
    /// Среднее значение энтропии из baseline (валидационная выборка)
    baseline_entropy_mean: Option<f32>,
    /// Стандартное отклонение энтропии из baseline
    baseline_entropy_std: Option<f32>,
    /// CSV writer для записи сэмплов
    csv_writer: Option<csv::Writer<File>>,
    /// Коэффициент сглаживания для EMA (обычно 0.1)
    alpha: f32,
}

impl InferenceConfidenceTracker {
    /// Создает новый трекер, загружая baseline из metadata.json
    fn new(model_dir: &Path, config: &BotConfig) -> Result<Self> {
        let metadata_path = model_dir.join("metadata.json");
        
        let mut baseline_entropy_mean = None;
        let mut baseline_entropy_std = None;
        
        // Загружаем baseline из metadata.json если существует
        if metadata_path.exists() {
            let metadata_content = std::fs::read_to_string(&metadata_path)
                .with_context(|| format!("Failed to read metadata.json: {}", metadata_path.display()))?;
            
            let metadata: ModelMetadata = serde_json::from_str(&metadata_content)
                .with_context(|| format!("Failed to parse metadata.json: {}", metadata_path.display()))?;
            
            baseline_entropy_mean = metadata.baseline_entropy_mean;
            baseline_entropy_std = metadata.baseline_entropy_std;
            
            if let (Some(mean), Some(std)) = (baseline_entropy_mean, baseline_entropy_std) {
                info!("Loaded baseline entropy: mean={:.4}, std={:.4}", mean, std);
            } else {
                warn!("Baseline entropy not found in metadata.json. Drift detection will use threshold only.");
            }
        }
        
        // Создаем CSV writer если включено сэмплирование
        let csv_writer = if config.confidence_sample_rate > 0 {
            let csv_path = model_dir.join("confidence_samples.csv");
            let file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&csv_path)
                .with_context(|| format!("Failed to create CSV file: {}", csv_path.display()))?;
            
            let mut writer = csv::Writer::from_writer(file);
            
            // Записываем заголовок если файл пустой
            if csv_path.metadata()?.len() == 0 {
                writer.write_record(&["timestamp_ms", "prob_flat", "prob_up", "prob_down", "entropy", "ema_entropy"])?;
                writer.flush()?;
            }
            
            info!("Confidence sampling enabled: 1 out of {} inferences will be recorded to {}", 
                  config.confidence_sample_rate, csv_path.display());
            Some(writer)
        } else {
            None
        };
        
        Ok(Self {
            ema_entropy: 0.0,
            sample_counter: 0,
            baseline_entropy_mean,
            baseline_entropy_std,
            csv_writer,
            alpha: 0.1, // Стандартный коэффициент сглаживания
        })
    }
    
    /// Записывает инференс и проверяет дрейф модели
    /// 
    /// # Возвращает
    /// Кортеж (entropy, drift_detected)
    fn record_inference(&mut self, probs: &[f32], config: &BotConfig) -> Result<(f32, bool)> {
        // 1. Вычисляем энтропию
        let entropy = calculate_entropy(probs);
        
        // 2. Обновляем EMA
        if self.sample_counter == 0 {
            // Первый инференс - инициализируем EMA
            self.ema_entropy = entropy;
        } else {
            self.ema_entropy = update_ema(self.ema_entropy, entropy, self.alpha);
        }
        
        // 3. Инкрементируем счетчик
        self.sample_counter += 1;
        
        // 4. Сэмплирование: записываем каждый N-й инференс
        if let Some(ref mut writer) = self.csv_writer {
            if self.sample_counter % config.confidence_sample_rate == 0 {
                let timestamp_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64;
                
                writer.write_record(&[
                    timestamp_ms.to_string(),
                    probs[0].to_string(),
                    probs[1].to_string(),
                    probs[2].to_string(),
                    entropy.to_string(),
                    self.ema_entropy.to_string(),
                ])?;
                writer.flush()?;
            }
        }
        
        // 5. Проверка дрейфа (если включена)
        let mut drift_detected = false;
        if config.enable_realtime_drift_check {
            // Проверка по порогу энтропии
            if self.ema_entropy > config.entropy_drift_threshold {
                drift_detected = true;
                warn!(
                    "Model drift detected: EMA entropy {:.4} exceeds threshold {:.4}",
                    self.ema_entropy, config.entropy_drift_threshold
                );
            }
            
            // Дополнительная проверка по baseline (если доступен)
            if let (Some(mean), Some(std)) = (self.baseline_entropy_mean, self.baseline_entropy_std) {
                let z_score = (self.ema_entropy - mean) / std;
                if z_score.abs() > 3.0 {
                    drift_detected = true;
                    warn!(
                        "Model drift detected: EMA entropy {:.4} deviates {:.2} std from baseline (mean={:.4}, std={:.4})",
                        self.ema_entropy, z_score, mean, std
                    );
                }
            }
        }
        
        Ok((entropy, drift_detected))
    }
}

pub struct OnnxEngine {
    pub session: Session,
    pub seq_len: usize,
    pub input_features: usize,
    pub temperature: Option<f32>,
    pub temperature_embedded: bool,
    pub use_regime_embedding: bool,
    pub num_regimes: usize,
    pub input_buffer: TensorBuffer, // Пре-аллоцированный буфер (Задача №197)
    confidence_tracker: Option<RefCell<InferenceConfidenceTracker>>, // Трекер уверенности (задача 224)
    bot_config: Option<BotConfig>, // Конфигурация бота для доступа к порогам (задача 224)
}

impl OnnxEngine {
    /// Загрузка модели с настройкой для Low-Latency и валидацией размерностей
    pub fn load(model_path: &Path, seq_len: usize, input_features: usize, onnx_config: &OnnxConfig, symbol: &str, bot_config: Option<&BotConfig>) -> Result<Self> {
        // 0. Валидация целостности модели (Задача 185)
        let model_dir = model_path.parent()
            .context("Failed to get model directory")?;
        let metadata_path = model_dir.join("metadata.json");
        
        // Загружаем metadata.json
        if !metadata_path.exists() {
            bail!("metadata.json not found at: {}. Model version control requires metadata file.", metadata_path.display());
        }
        
        let metadata_content = std::fs::read_to_string(&metadata_path)
            .with_context(|| format!("Failed to read metadata.json: {}", metadata_path.display()))?;
        
        let metadata: ModelMetadata = serde_json::from_str(&metadata_content)
            .with_context(|| format!("Failed to parse metadata.json: {}", metadata_path.display()))?;
        
        // Вычисляем хэш model.onnx
        info!("Computing SHA-256 hash for model file: {}", model_path.display());
        let computed_hash = compute_file_hash(model_path)?;
        
        // Сравниваем с хэшем из metadata
        if Some(computed_hash.clone()) != metadata.onnx_hash {
            error!(
                "Model integrity check FAILED! Expected hash: {}, computed hash: {}",
                metadata.onnx_hash.as_deref().unwrap_or("unknown"), computed_hash
            );
            bail!(
                "Model file integrity violation detected. The model file may be corrupted or tampered with. \
                Expected hash: {}, computed hash: {}",
                metadata.onnx_hash.as_deref().unwrap_or("unknown"), computed_hash
            );
        }
        
        // Логируем успешную валидацию
        info!("✓ Model integrity check PASSED");
        info!("Model version: {}", metadata.version.as_deref().unwrap_or("unknown"));
        if let Some(mcc) = metadata.mcc_score {
            info!("Model MCC score: {:.4}", mcc);
        }
        info!("Model hash: {}", metadata.onnx_hash.as_deref().unwrap_or("unknown"));
        
        // 1. Инициализация сессии с выбранным Execution Provider
        let session = init_session(onnx_config, model_path, symbol, seq_len, input_features)?;

        // 2. Валидация входного тензора [batch, seq_len, features]
        let input0 = &session.inputs[0];
        let shape = input0.input_type.as_tensor_type()
            .context("Input 0 is not a tensor")?.shape.clone();

        // Проверка batch_size (dim 0) - должен быть фиксирован в 1
        if let Some(dim) = shape.get(0) {
            if let Some(fixed) = dim.as_fixed() {
                if fixed != 1 {
                    bail!("Model batch_size must be 1 for optimal performance, got {}", fixed);
                }
            } else {
                bail!("Dynamic batch size detected. TensorRT requires fixed batch_size=1 for optimal performance. Please re-export the model with fixed batch dimension.");
            }
        }

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

        // 4. Загрузка температуры и regime параметров из metadata.json (если существует)
        let metadata_path = model_path.parent()
            .context("Failed to get model directory")?
            .join("metadata.json");
        
        let mut temperature = None;
        let mut temperature_embedded = false;
        let mut use_regime_embedding = false;
        let mut num_regimes = 0;
        
        if metadata_path.exists() {
            match std::fs::read_to_string(&metadata_path) {
                Ok(content) => {
                    match serde_json::from_str::<serde_json::Value>(&content) {
                        Ok(metadata) => {
                            // Проверяем, встроена ли температура в ONNX граф
                            if let Some(embedded) = metadata.get("temperature_embedded").and_then(|v| v.as_bool()) {
                                temperature_embedded = embedded;
                            }
                            
                            // Загружаем температуру только если она не встроена
                            if !temperature_embedded {
                                if let Some(temp) = metadata.get("temperature").and_then(|v| v.as_f64()) {
                                    temperature = Some(temp as f32);
                                    info!("Loaded temperature from metadata: T = {:.4}", temp);
                                }
                            } else {
                                info!("Temperature scaling is embedded in ONNX graph");
                            }
                            
                            // Загружаем параметры regime embedding
                            if let Some(use_regime) = metadata.get("use_regime_embedding").and_then(|v| v.as_bool()) {
                                use_regime_embedding = use_regime;
                            }
                            
                            if let Some(n_regimes) = metadata.get("num_regimes").and_then(|v| v.as_u64()) {
                                num_regimes = n_regimes as usize;
                            }
                            
                            if use_regime_embedding {
                                info!("Model uses regime embedding with {} regimes", num_regimes);
                            }
                        }
                        Err(e) => warn!("Failed to parse metadata.json: {}", e),
                    }
                }
                Err(e) => warn!("Failed to read metadata.json: {}", e),
            }
        }

        let mut input_buffer = TensorBuffer::new(1, input_features / 50, 50, seq_len);
        // Проверка соответствия размера буфера ожиданиям модели
        if input_buffer.as_slice().len() != seq_len * input_features {
             // Если расчет по каналам не совпал (например, input_features не кратно 50),
             // создаем плоский буфер нужного размера
             input_buffer = TensorBuffer::new(1, 1, 1, seq_len * input_features);
        }

        // 6. Инициализация трекера уверенности (задача 224)
        let confidence_tracker = if let Some(config) = bot_config {
            if config.enable_realtime_drift_check {
                match InferenceConfidenceTracker::new(model_dir, config) {
                    Ok(tracker) => {
                        info!("Confidence tracker initialized successfully");
                        Some(RefCell::new(tracker))
                    }
                    Err(e) => {
                        warn!("Failed to initialize confidence tracker: {}. Continuing without drift detection.", e);
                        None
                    }
                }
            } else {
                info!("Realtime drift check is disabled in config");
                None
            }
        } else {
            None
        };

        let engine = Self { 
            session, 
            seq_len, 
            input_features,
            temperature,
            temperature_embedded,
            use_regime_embedding,
            num_regimes,
            input_buffer,
            confidence_tracker,
            bot_config: bot_config.cloned(),
        };

        // 7. Прогрев модели (Warm-up): 3 холостых прогона для аллокации памяти и JIT
        info!("Starting ONNX model warmup (3 iterations)...");
        let dummy_input = vec![0.0f32; seq_len * input_features];
        let warmup_regime_id = if engine.use_regime_embedding { Some(0) } else { None };
        for i in 0..3 {
            let _ = engine.predict(&dummy_input, warmup_regime_id)?;
            info!("Warmup iteration {}/3 completed", i + 1);
        }
        info!("Model warmup completed successfully.");

        Ok(engine)
    }

    /// Выполняет инференс модели с использованием пре-аллоцированного буфера (Задача №197)
    /// Обеспечивает передачу данных в рантайм ONNX без лишнего копирования (Zero-copy)
    #[inline(always)]
    pub fn predict_with_buffer(&self, regime_id: Option<usize>) -> Result<InferenceResult> {
        let start_build = Instant::now();

        // 1. Создаем ndarray View (Batch=1, Channels, Levels, Seq) напрямую из пре-аллоцированного буфера
        let array = ArrayView4::from_shape(
            self.input_buffer.shape(),
            self.input_buffer.as_slice()
        ).context("Failed to create array view from pre-allocated buffer")?;

        // 2. Валидация regime_id если модель использует regime embedding
        if self.use_regime_embedding {
            match regime_id {
                Some(id) if id >= self.num_regimes => {
                    bail!("Invalid regime_id: {} (model has {} regimes)", id, self.num_regimes);
                }
                None => bail!("Model requires regime_id but none was provided"),
                _ => {}
            }
        }

        // 3. Подготовка входов для ONNX (Zero-copy через ArrayView)
        // ВАЖНО: ort::Value::from_array при передаче ArrayView заимствует память
        let ort_input = if self.use_regime_embedding {
            let regime_id_val = regime_id.unwrap_or(0) as i64;
            // regime_array всё еще аллоцирует малый массив, но это пренебрежимо мало (8 байт)
            let regime_array = ndarray::Array1::from_vec(vec![regime_id_val]);
            inputs![Value::from_array(array)?, Value::from_array(regime_array)?]
        } else {
            // Создаём пустой массив для второго входа, чтобы размер был совместим
            let empty_regime = ndarray::Array1::<i64>::zeros(1);
            inputs![Value::from_array(array)?, Value::from_array(empty_regime)?]
        };
        
        let build_us = start_build.elapsed().as_micros() as u64;
        let start_run = Instant::now();

        // 4. Запуск инференса
        let outputs = self.session.run(ort_input)?;
        let run_us = start_run.elapsed().as_micros() as u64;

        // Обновляем статистику
        HOT_PATH_STATS.record_inference(run_us);
        metrics::histogram!("bot_inference_duration_us").record(run_us as f64);

        if run_us > 15_000 {
            warn!("Slow inference detected: {}us (Model run only)", run_us);
        }
        
        // 5. Извлечение тензора результатов
        let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
        let logits = output_tensor.view();
        let batch_logits = logits.slice(ndarray::s![0, ..]);
        
        if batch_logits.len() != 3 {
            bail!("Model output size mismatch: expected 3, got {}", batch_logits.len());
        }

        let logits_slice = batch_logits.as_slice()
            .context("Failed to convert logits to slice")?;

        // 6. Применение температуры
        let calibrated_logits: Vec<f32> = if !self.temperature_embedded && self.temperature.is_some() {
            let temp = self.temperature.unwrap();
            logits_slice.iter().map(|&x| x / temp).collect()
        } else {
            logits_slice.to_vec()
        };

        // 7. Расчет Softmax
        let probs = self.softmax(&calibrated_logits);

        // 8. Определение сигнала
        let mut max_prob = -1.0;
        let mut signal_idx = 0;
        for (i, &p) in probs.iter().enumerate() {
            if p > max_prob {
                max_prob = p;
                signal_idx = i;
            }
        }

        // 9. Мониторинг уверенности модели (задача 224)
        let mut entropy = None;
        let mut drift_detected = false;
        
        if let (Some(ref tracker), Some(ref config)) = (&self.confidence_tracker, &self.bot_config) {
            match tracker.borrow_mut().record_inference(&probs, config) {
                Ok((ent, drift)) => {
                    entropy = Some(ent);
                    drift_detected = drift;
                }
                Err(e) => {
                    warn!("Failed to record inference confidence: {}", e);
                }
            }
        }

        Ok(InferenceResult {
            output: InferenceOutput {
                signal: Signal::from(signal_idx),
                probabilities: probs.to_vec(),
                probs: ndarray::Array2::from_shape_vec((1, 3), probs.to_vec()).unwrap(),
                entropy,
                drift_detected,
            },
            duration_us: run_us,
        })
    }

    /// Предоставляет мутабельное представление буфера для заполнения данными (Задача №197)
    #[inline(always)]
    pub fn get_input_view_mut(&mut self) -> ndarray::ArrayViewMut4<f32> {
        let shape = self.input_buffer.shape();
        ndarray::ArrayViewMut4::from_shape(shape, self.input_buffer.get_mut()).unwrap()
    }

    /// Выполняет инференс модели. Принимает слайс f32 для Zero-copy.
    /// 
    /// # Аргументы
    /// * `input_data` - Входные данные модели (seq_len * input_features)
    /// * `regime_id` - Опциональный ID режима рынка (если модель использует regime embedding)
    pub fn predict(&self, input_data: &[f32], regime_id: Option<usize>) -> Result<InferenceResult> {
        let start_build = Instant::now();

        // 1. Валидация размера входных данных (должен соответствовать batch=1)
        // Задача 098: Явная проверка batch size
        let expected_size = self.seq_len * self.input_features;
        if input_data.len() != expected_size {
            // Проверяем, не пытается ли пользователь подать batch > 1
            if input_data.len() % expected_size == 0 {
                let batch_size = input_data.len() / expected_size;
                bail!(
                    "Input data size suggests batch > 1: expected {} elements (batch=1, seq_len={}, features={}), got {} elements (batch={}). \
                    Dynamic batching is disabled for TensorRT optimization. Only batch_size=1 is supported.",
                    expected_size, self.seq_len, self.input_features, input_data.len(), batch_size
                );
            } else {
                bail!(
                    "Input data size mismatch: expected {} (batch=1, seq_len={}, features={}), got {}",
                    expected_size, self.seq_len, self.input_features, input_data.len()
                );
            }
        }
        
        // 2. Валидация regime_id если модель использует regime embedding
        if self.use_regime_embedding {
            match regime_id {
                Some(id) if id >= self.num_regimes => {
                    bail!(
                        "Invalid regime_id: {} (model has {} regimes)",
                        id, self.num_regimes
                    );
                }
                None => {
                    bail!("Model requires regime_id but none was provided");
                }
                _ => {}
            }
        }

        // 3. Создаем ndarray View (Batch=1, Seq, Feat) без копирования данных
        let shape = (1, self.seq_len, self.input_features);
        let array = ndarray::ArrayView3::from_shape(shape, input_data)
            .context("Failed to create array view for ONNX inference")?;

        // 4. Подготовка входов для ONNX
        let ort_input = if self.use_regime_embedding {
            let regime_id_val = regime_id.unwrap_or(0) as i64;
            let regime_array = ndarray::Array1::from_vec(vec![regime_id_val]);
            inputs![Value::from_array(array)?, Value::from_array(regime_array)?]
        } else {
            // Создаём пустой массив для второго входа, чтобы размер был совместим
            let empty_regime = ndarray::Array1::<i64>::zeros(1);
            inputs![Value::from_array(array)?, Value::from_array(empty_regime)?]
        };
        
        let build_us = start_build.elapsed().as_micros() as u64;

        let start_run = Instant::now();
        // 5. Запуск инференса
        let outputs = self.session.run(ort_input)?;
        let run_us = start_run.elapsed().as_micros() as u64;

        // Обновляем статистику
        HOT_PATH_STATS.record_inference(run_us);

        // Запись метрики длительности инференса (задача 189)
        metrics::histogram!("bot_inference_duration_us").record(run_us as f64);

        // Порог медленного инференса (15мс)
        if run_us > 15_000 {
            warn!("Slow inference detected: {}us (Model run only)", run_us);
        }
        
        // 6. Извлечение тензора результатов
        let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
        let logits = output_tensor.view();
        
        // Берем результаты для первого (и единственного) элемента батча
        let batch_logits = logits.slice(ndarray::s![0, ..]);
        
        // Валидация размерности выхода
        if batch_logits.len() != 3 {
            bail!("Model output size mismatch: expected 3, got {}", batch_logits.len());
        }

        let logits_slice = batch_logits.as_slice()
            .context("Failed to convert logits to slice")?;

        // 7. Применение температуры (если не встроена в ONNX и задана в metadata)
        let calibrated_logits: Vec<f32> = if !self.temperature_embedded && self.temperature.is_some() {
            let temp = self.temperature.unwrap();
            logits_slice.iter().map(|&x| x / temp).collect()
        } else {
            logits_slice.to_vec()
        };

        // 8. Расчет Softmax
        let probs = self.softmax(&calibrated_logits);

        // 9. Определение сигнала (Argmax)
        let mut max_prob = -1.0;
        let mut signal_idx = 0;
        for (i, &p) in probs.iter().enumerate() {
            if p > max_prob {
                max_prob = p;
                signal_idx = i;
            }
        }

        // 10. Мониторинг уверенности модели (задача 224)
        let mut entropy = None;
        let mut drift_detected = false;
        
        if let (Some(ref tracker), Some(ref config)) = (&self.confidence_tracker, &self.bot_config) {
            match tracker.borrow_mut().record_inference(&probs, config) {
                Ok((ent, drift)) => {
                    entropy = Some(ent);
                    drift_detected = drift;
                }
                Err(e) => {
                    warn!("Failed to record inference confidence: {}", e);
                }
            }
        }

        Ok(InferenceResult {
            output: InferenceOutput {
                signal: Signal::from(signal_idx),
                probabilities: probs.to_vec(),
                probs: ndarray::Array2::from_shape_vec((1, 3), probs.to_vec()).unwrap(),
                entropy,
                drift_detected,
            },
            duration_us: run_us,
        })
    }

    /// Задача 169: Метод для получения результата инференса с метриками производительности
    /// Возвращает InferenceResult с duration_us для отслеживания задержки
    pub fn predict_with_metrics(&self, input_data: &[f32], regime_id: Option<usize>) -> Result<InferenceResult> {
        let start_total = Instant::now();
        let output = self.predict(input_data, regime_id)?;
        let duration_us = start_total.elapsed().as_micros() as u64;
        
        Ok(InferenceResult {
            output: output.output,
            duration_us,
        })
    }

    /// Прогрев модели (warmup) для исключения пиков при старте
    pub fn warmup(&self) -> Result<()> {
        info!("Starting ONNX model warmup (50 iterations)...");
        let input_size = self.seq_len * self.input_features;
        let dummy_data = vec![0.0f32; input_size];
        
        // Для моделей с regime embedding используем regime_id=0
        let regime_id = if self.use_regime_embedding { Some(0) } else { None };
        
        for _ in 0..50 {
            let _ = self.predict(&dummy_data, regime_id)?;
        }
        
        // Сбрасываем статистику после прогрева
        HOT_PATH_STATS.reset();
        info!("Model warmup completed successfully.");
        Ok(())
    }

    /// Приватный метод Softmax для 3-х классов
    #[inline(always)]
    fn softmax(&self, logits: &[f32]) -> [f32; 3] {
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exps: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exps: f32 = exps.iter().sum();
        
        [
            exps[0] / sum_exps,
            exps[1] / sum_exps,
            exps[2] / sum_exps,
        ]
    }
}
