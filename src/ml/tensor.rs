use crate::data::orderbook::OrderBook;
use crate::ml::normalization::Normalizer;
use crate::ml::types::{ModelMetadata, NormalizationParams};
use std::collections::VecDeque;
use anyhow::{Result, Context};
use serde::Deserialize;
use rust_decimal::Decimal;
use ndarray::{Array2, Array4, s};
use std::sync::{Arc, RwLock};
use wide::f32x8;
use aligned_vec::{AVec, ConstAlign};

/// Константа для защиты от деления на 0 при нормализации
const EPSILON: f32 = 1e-8;

/// SIMD-ускоренная нормализация признаков
/// 
/// Применяет z-score нормализацию: (x - mean) * inv_std
/// Использует предварительно вычисленные обратные значения std для замены деления на умножение
/// 
/// # Аргументы
/// * `features` - Мутабельный срез признаков для нормализации
/// * `means` - Средние значения для каждого признака
/// * `inv_stds` - Обратные значения стандартных отклонений (1.0 / std)
/// 
/// # Производительность
/// Использует f32x8 для обработки 8 элементов за раз на архитектурах с AVX/AVX2
/// Автоматически использует скалярный fallback для оставшихся элементов
pub fn normalize_features_simd(features: &mut [f32], means: &[f32], inv_stds: &[f32]) {
    let len = features.len();
    
    // Проверка размеров
    debug_assert_eq!(len, means.len(), "Features and means length mismatch");
    debug_assert_eq!(len, inv_stds.len(), "Features and inv_stds length mismatch");
    
    // SIMD обработка блоками по 8 элементов
    let simd_len = len - (len % 8);
    let mut i = 0;
    
    while i < simd_len {
        // Эффективная загрузка 8 элементов из памяти используя from_slice_unaligned
        let feat_vec = f32x8::from_slice_unaligned(&features[i..i+8]);
        let mean_vec = f32x8::from_slice_unaligned(&means[i..i+8]);
        let inv_std_vec = f32x8::from_slice_unaligned(&inv_stds[i..i+8]);
        
        // Применяем нормализацию: (x - mean) * inv_std
        let normalized = (feat_vec - mean_vec) * inv_std_vec;
        
        // Сохраняем результат обратно
        normalized.write_to_slice_unaligned(&mut features[i..i+8]);
        
        i += 8;
    }
    
    // Скалярный fallback для оставшихся элементов
    for j in i..len {
        features[j] = (features[j] - means[j]) * inv_stds[j];
    }
    
    // Проверка на NaN/Inf после нормализации
    debug_assert!(
        features.iter().all(|x| x.is_finite()),
        "Features contain NaN or Inf after SIMD normalization"
    );
}

/// SIMD-ускоренное клиппинг значений (Задача 240)
/// 
/// Ограничивает значения в диапазоне [low, high]
/// Формула: max(low, min(x, high))
/// 
/// # Аргументы
/// * `features` - Мутабельный срез признаков для клиппинга
/// * `lows` - Нижние границы для каждого признака
/// * `highs` - Верхние границы для каждого признака
pub fn clip_features_simd(features: &mut [f32], lows: &[f32], highs: &[f32]) {
    let len = features.len();
    
    // Проверка размеров
    debug_assert_eq!(len, lows.len(), "Features and lows length mismatch");
    debug_assert_eq!(len, highs.len(), "Features and highs length mismatch");
    
    // SIMD обработка блоками по 8 элементов
    let simd_len = len - (len % 8);
    let mut i = 0;
    
    while i < simd_len {
        let feat_vec = f32x8::from_slice_unaligned(&features[i..i+8]);
        let low_vec = f32x8::from_slice_unaligned(&lows[i..i+8]);
        let high_vec = f32x8::from_slice_unaligned(&highs[i..i+8]);
        
        // Клиппинг: max(low, min(x, high))
        let clipped = feat_vec.max(low_vec).min(high_vec);
        
        clipped.write_to_slice_unaligned(&mut features[i..i+8]);
        
        i += 8;
    }
    
    // Скалярный fallback для оставшихся элементов
    for j in i..len {
        features[j] = features[j].max(lows[j]).min(highs[j]);
    }
}

/// SIMD-ускоренная нормализация временного среза
/// 
/// Применяет нормализацию к непрерывному срезу по времени для одного (канал, уровень)
/// Формула: (x - mean) * inv_std
/// 
/// # Аргументы
/// * `time_slice` - Мутабельный срез по времени (непрерывный в памяти)
/// * `mean` - Среднее значение для этого (канал, уровень)
/// * `inv_std` - Обратное значение стандартного отклонения
pub fn normalize_time_slice_simd(time_slice: &mut [f32], mean: f32, inv_std: f32) {
    let len = time_slice.len();
    
    // Создаем SIMD-векторы с одинаковыми значениями
    let mean_vec = f32x8::splat(mean);
    let inv_std_vec = f32x8::splat(inv_std);
    
    // SIMD обработка блоками по 8 элементов
    let simd_len = len - (len % 8);
    let mut i = 0;
    
    while i < simd_len {
        // Эффективная загрузка 8 элементов из памяти
        let feat_vec = f32x8::from_slice_unaligned(&time_slice[i..i+8]);
        
        // Применяем нормализацию: (x - mean) * inv_std
        let normalized = (feat_vec - mean_vec) * inv_std_vec;
        
        // Сохраняем результат обратно
        normalized.write_to_slice_unaligned(&mut time_slice[i..i+8]);
        
        i += 8;
    }
    
    // Скалярный fallback для оставшихся элементов
    for j in i..len {
        time_slice[j] = (time_slice[j] - mean) * inv_std;
    }
    
    // Проверка на NaN/Inf после нормализации
    debug_assert!(
        time_slice.iter().all(|x| x.is_finite()),
        "Time slice contains NaN or Inf after SIMD normalization"
    );
}

#[derive(Deserialize)]
struct ModelParams {
    seq_len: usize,
    n_levels: usize,
    in_channels: usize,
    past_returns_lags: Option<Vec<usize>>,
}

#[derive(Deserialize)]
struct ModelMetadataInternal {
    model_params: ModelParams,
    normalization: NormalizationParams,
}

/// Буфер для пре-аллокации памяти под входной тензор (Задача №197)
pub struct TensorBuffer {
    data: Vec<f32>,
    shape: [usize; 4],
}

impl TensorBuffer {
    /// Создает новый буфер на основе размеров тензора [batch, channels, levels, seq_len]
    pub fn new(batch: usize, channels: usize, levels: usize, seq_len: usize) -> Self {
        let size = batch * channels * levels * seq_len;
        let mut data = Vec::with_capacity(size);
        data.resize(size, 0.0);
        Self {
            data,
            shape: [batch, channels, levels, seq_len],
        }
    }

    /// Возвращает мутабельный срез для заполнения данными без аллокаций
    pub fn get_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Возвращает неизменяемый срез данных
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Возвращает форму тензора
    pub fn shape(&self) -> [usize; 4] {
        self.shape
    }
}

    /// Применяет безопасную SIMD-ускоренную нормализацию (z-score) к представлению тензора (Задача №197)
pub fn apply_normalization_view(tensor: &mut ndarray::ArrayViewMut4<f32>, means: &[f32], inv_stds: &[f32]) {
    let (_, channels, levels, _) = tensor.dim();
    let feature_dim = channels * levels;
    
    // Проверка размеров параметров
    debug_assert_eq!(means.len(), feature_dim, "Means size mismatch");
    debug_assert_eq!(inv_stds.len(), feature_dim, "Inv_stds size mismatch");
    
    // Проходим по каждому каналу и уровню
    for c in 0..channels {
        for l in 0..levels {
            let idx = c * levels + l; 
            let m = means[idx];
            let inv_s = inv_stds[idx];
            
            let mut time_slice = tensor.slice_mut(s![0, c, l, ..]);
            
            if let Some(slice_mut) = time_slice.as_slice_mut() {
                normalize_time_slice_simd(slice_mut, m, inv_s);
            } else {
                time_slice.mapv_inplace(|x| (x - m) * inv_s);
            }
        }
    }
}

/// Применяет безопасную SIMD-ускоренную нормализацию (z-score) к тензору
pub fn apply_normalization(tensor: &mut Array4<f32>, means: &[f32], inv_stds: &[f32]) {
    apply_normalization_view(&mut tensor.view_mut(), means, inv_stds);
}

/// Тип для представления одного снимка данных стакана (Задача 097)
/// Содержит 150 нормализованных признаков: 3 канала * 50 уровней
pub type Snapshot = Vec<f32>;

/// Структура для построения входного тензора фиксированной длины для ONNX-модели (Задача 097)
/// 
/// Реализует логику Left-Padding (заполнение нулями слева) и Truncation (отсечение старых данных)
/// для обеспечения скользящего окна актуальных данных размером seq_len.
/// 
/// # Конкурентность (Задача 097)
/// Если TensorBuilder доступен из нескольких потоков (например, чтение данных и инференс),
/// рекомендуется обернуть его в Arc<RwLock<TensorBuilder>>:
/// 
/// ```rust
/// let builder = TensorBuilder::from_metadata("metadata.json")?;
/// let builder = Arc::new(RwLock::new(builder));
/// 
/// // В потоке 1: добавление данных
/// let mut b = builder.write().unwrap();
/// b.process_snapshot(&ob)?;
/// 
/// // В потоке 2: построение тензора
/// let b = builder.read().unwrap();
/// let tensor = b.build_tensor();
/// ```
pub struct TensorBuilder {
    pub buffer: VecDeque<Snapshot>,         // История снимков (Задача 097)
    pub seq_len: usize,                    // Размер скользящего окна (из metadata.json)
    pub channels: usize,                   // Количество каналов (обычно 3: Price, Vol, Imbalance)
    pub levels: usize,                     // Количество уровней стакана (обычно 50)
    normalizer: Normalizer,
    mid_price_history: VecDeque<Decimal>,  // Задача 091: История средних цен для расчета log-returns
    past_returns_lags: Vec<usize>,         // Задача 091: Лаги для расчета Past Returns [10, 50, 100]
}

impl TensorBuilder {
    pub fn new(normalizer: Normalizer, seq_len: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(seq_len),
            seq_len,
            channels: 3,  // По умолчанию 3 канала (Price, Vol, Imbalance)
            levels: 50,   // По умолчанию 50 уровней
            normalizer,
            mid_price_history: VecDeque::with_capacity(seq_len),  // Задача 091
            past_returns_lags: vec![10, 50, 100],  // Задача 091: Лаги по умолчанию
        }
    }

    /// Создает TensorBuilder, загружая параметры из файла метаданных
    pub fn from_metadata(metadata_path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(metadata_path)
            .with_context(|| format!("Failed to read metadata file: {}", metadata_path))?;
        let metadata: ModelMetadataInternal = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse metadata JSON at {}", metadata_path))?;
        
        let seq_len = metadata.model_params.seq_len;
        
        // Задача 240: Выбор типа нормализации на основе scaler_type
        let normalizer = match metadata.normalization.scaler_type.as_str() {
            "winsor_robust" => {
                let winsor_limits = metadata.normalization.winsor_limits
                    .ok_or_else(|| anyhow::anyhow!("Winsor limits missing for winsor_robust scaler"))?;
                let medians = metadata.normalization.median
                    .ok_or_else(|| anyhow::anyhow!("Median parameters missing for winsor_robust scaler"))?;
                let iqrs = metadata.normalization.iqr
                    .ok_or_else(|| anyhow::anyhow!("IQR parameters missing for winsor_robust scaler"))?;
                
                // winsor_limits должен содержать [low_0, high_0, low_1, high_1, ...]
                let mut winsor_low = Vec::new();
                let mut winsor_high = Vec::new();
                for i in (0..winsor_limits.len()).step_by(2) {
                    if i + 1 < winsor_limits.len() {
                        winsor_low.push(winsor_limits[i]);
                        winsor_high.push(winsor_limits[i + 1]);
                    }
                }
                
                Normalizer::new_winsor_robust(winsor_low, winsor_high, medians, iqrs)
            }
            "robust" => {
                let medians = metadata.normalization.median
                    .ok_or_else(|| anyhow::anyhow!("Median parameters missing for robust scaler"))?;
                let iqrs = metadata.normalization.iqr
                    .ok_or_else(|| anyhow::anyhow!("IQR parameters missing for robust scaler"))?;
                Normalizer::new_robust(medians, iqrs)
            }
            "zscore" | _ => {
                let means = metadata.normalization.mean
                    .ok_or_else(|| anyhow::anyhow!("Mean parameters missing for zscore scaler"))?;
                let stds = metadata.normalization.std
                    .ok_or_else(|| anyhow::anyhow!("Std parameters missing for zscore scaler"))?;
                Normalizer::new(means, stds)
            }
        };

        tracing::info!(
            "Initialized TensorBuilder with seq_len: {}, scaler_type: {}", 
            seq_len,
            metadata.normalization.scaler_type
        );

        // Задача 091: Загружаем лаги для Past Returns
        let past_returns_lags = metadata.model_params.past_returns_lags
            .unwrap_or_else(|| vec![10, 50, 100]);
        
        let mut builder = Self::new(normalizer, seq_len);
        builder.past_returns_lags = past_returns_lags.clone();
        
        // Задача 097: Загружаем channels и levels из metadata
        builder.channels = metadata.model_params.in_channels;
        builder.levels = metadata.model_params.n_levels;
        
        tracing::info!(
            "Loaded past_returns_lags: {:?}, channels: {}, levels: {}", 
            past_returns_lags,
            builder.channels,
            builder.levels
        );
        
        Ok(builder)
    }

    /// Добавляет новый снимок в историю с автоматическим truncation (скользящее окно)
    /// При превышении seq_len удаляет самый старый снимок (Задача 097)
    fn add_snapshot(&mut self, snapshot: Snapshot) {
        // Truncation: если буфер полон, удаляем самый старый элемент
        if self.buffer.len() >= self.seq_len {
            self.buffer.pop_front();
        }
        self.buffer.push_back(snapshot);
    }

    /// Строит входной тензор фиксированной длины с Left-Padding (Задача 097)
    /// 
    /// Создает Array4<f32> формы (1, channels, levels, seq_len) с нулями слева
    /// и свежими данными справа. Это обеспечивает, что самые актуальные данные
    /// всегда находятся в правой части тензора (индексы ближе к seq_len - 1).
    /// 
    /// # Логика Left-Padding
    /// - Если buffer содержит N снимков, где N < seq_len:
    ///   - Первые (seq_len - N) временных шагов заполняются нулями
    ///   - Последние N временных шагов содержат данные из buffer
    /// - Если buffer содержит seq_len снимков:
    ///   - Все временные шаги содержат данные (нет padding)
    /// 
    /// # Возвращаемое значение
    /// Array4<f32> с формой (1, channels, levels, seq_len)
    pub fn build_tensor(&self) -> Array4<f32> {
        // Создаем пустой тензор (Batch=1, Channels, Levels, Time)
        let mut tensor = Array4::<f32>::zeros((1, self.channels, self.levels, self.seq_len));
        
        // Рассчитываем смещение для вставки (если данных < seq_len)
        // offset показывает, с какого временного шага начинать вставку данных
        let offset = self.seq_len.saturating_sub(self.buffer.len());
        
        // Вставляем каждый снимок в тензор со смещением
        for (i, snap) in self.buffer.iter().enumerate() {
            // Проверяем размер снимка
            if snap.len() != self.channels * self.levels {
                tracing::warn!(
                    "Snapshot size mismatch: expected {}, got {}",
                    self.channels * self.levels,
                    snap.len()
                );
                continue;
            }
            
            // Преобразуем плоский вектор в Array2 (channels, levels)
            // Структура snap: [channel_0_level_0, ..., channel_0_level_49, 
            //                   channel_1_level_0, ..., channel_1_level_49,
            //                   channel_2_level_0, ..., channel_2_level_49]
            if let Ok(features) = Array2::from_shape_vec(
                (self.channels, self.levels),
                snap.clone()
            ) {
                // Вставляем в тензор со смещением offset
                // tensor.slice_mut(s![0, .., .., i + offset]) выбирает временной шаг (i + offset)
                tensor.slice_mut(s![0, .., .., i + offset])
                    .assign(&features);
            } else {
                tracing::warn!(
                    "Failed to reshape snapshot into ({}, {})",
                    self.channels,
                    self.levels
                );
            }
        }
        
        tensor
    }

    /// Задача 091: Рассчитывает log-returns для заданного лага
    /// Формула: ln(current_mid) - ln(old_mid)
    /// Если буфер еще не заполнен до нужного лага, возвращает 0.0
    fn calculate_log_returns(&self, lag: usize) -> f32 {
        if self.mid_price_history.len() < lag + 1 {
            // Буфер еще не заполнен, возвращаем 0.0 (нейтральный сигнал)
            return 0.0;
        }
        
        let current_idx = self.mid_price_history.len() - 1;
        let old_idx = current_idx - lag;
        
        if let (Some(current_mid), Some(old_mid)) = (
            self.mid_price_history.get(current_idx),
            self.mid_price_history.get(old_idx),
        ) {
            if *old_mid > Decimal::ZERO {
                // Конвертируем в f64 для расчета логарифма
                let current_f64 = current_mid.to_f64().unwrap_or(0.0);
                let old_f64 = old_mid.to_f64().unwrap_or(0.0);
                
                if current_f64 > 0.0 && old_f64 > 0.0 {
                    let log_return = (current_f64.ln() - old_f64.ln()) as f32;
                    return log_return;
                }
            }
        }
        
        0.0
    }

    /// Добавляет новый снимок стакана в историю и возвращает полный тензор
    /// 
    /// Согласно плану 035:
    /// 1. Извлекаем топ-50 уровней (200 признаков: 50 Ask [p,v] + 50 Bid [p,v])
    /// 2. Feature Engineering: (p-mid)/mid для цен, ln(1+v) для объемов
    /// 3. Нормализация (Z-score)
    /// 4. Валидация (NaN/Inf check) ПОСЛЕ нормализации
    /// 5. Обновляем скользящее окно
    /// 6. Возвращаем Some(Vec<f32>) только если окно заполнено полностью
    pub fn process_snapshot(&mut self, ob: &crate::data::orderbook::OrderBookSnapshot) -> Result<Option<Vec<f32>>> {
        let mid = ob.get_mid_price() as f32;
        let mid_dec = ob.get_mid_price_dec();  // Задача 091: Получаем mid_price для расчета log-returns
        
        if mid <= 0.0 {
            return Ok(None); // Недостаточно данных в стакане
        }

        // 1. Извлекаем топ-50 уровней (200 фич)
        let raw = ob.get_flat_snapshot();
        if raw.len() != 200 {
            anyhow::bail!("Expected 200 raw features, got {}", raw.len());
        }

        // TODO: Зеркальная реализация 053. Канал 2: (Vbid - Vask) / (Vbid + Vask + 1e-7).
        // Консистентность с Python dataset.py должна быть 100%.
        // Структура: [ask_p_0..49, ask_v_0..49, bid_p_0..49, bid_v_0..49]
        // Выход: [price_0..49, vol_0..49, imb_0..49] (3 канала * 50 уровней = 150 фич)

        // 2. Разделяем на компоненты согласно плану 053
        // ВАЖНО: Формат данных из get_flat_snapshot() - чередующийся (interleaved):
        // Индексы 0-99: [ask_p_0, ask_v_0, ask_p_1, ask_v_1, ..., ask_p_49, ask_v_49]
        // Индексы 100-199: [bid_p_0, bid_v_0, bid_p_1, bid_v_1, ..., bid_p_49, bid_v_49]
        let mut ask_p = vec![0.0; 50];
        let mut ask_v = vec![0.0; 50];
        let mut bid_p = vec![0.0; 50];
        let mut bid_v = vec![0.0; 50];
        
        for i in 0..50 {
            ask_p[i] = raw[i * 2];           // Индексы: 0, 2, 4, ..., 98
            ask_v[i] = raw[i * 2 + 1];       // Индексы: 1, 3, 5, ..., 99
            bid_p[i] = raw[100 + i * 2];     // Индексы: 100, 102, ..., 198
            bid_v[i] = raw[100 + i * 2 + 1]; // Индексы: 101, 103, ..., 199
        }

        // 3. Вычисляем 3 канала согласно плану 053
        let mut features = vec![0.0; 150]; // 3 канала * 50 уровней
        
        for i in 0..50 {
            // Канал 0: Normalized Price (среднее отклонение)
            let price_ch = (ask_p[i] + bid_p[i]) / 2.0;
            features[i] = price_ch;
            
            // Канал 1: Log Volume
            let vol_ch = ask_v[i] + bid_v[i];
            features[50 + i] = vol_ch;
            
            // Канал 2: Static Level Imbalance
            // Формула: (Vbid - Vask) / (Vbid + Vask + eps)
            let imb_ch = (bid_v[i] - ask_v[i]) / (bid_v[i] + ask_v[i] + 1e-7);
            features[100 + i] = imb_ch;
        }

        // 4. Нормализация (Z-score)
        self.normalizer.normalize(&mut features);

        // 5. Валидация (NaN/Inf check) ПОСЛЕ нормализации
        for val in &features {
            if !val.is_finite() {
                anyhow::bail!("Invalid feature value detected (NaN or Inf) after normalization");
            }
        }

        // 6. Обновляем скользящее окно
        self.add_snapshot(features);
        
        // Задача 091: Обновляем mid_price_history для расчета log-returns
        if self.mid_price_history.len() >= self.past_returns_lags.iter().max().copied().unwrap_or(100) {
            self.mid_price_history.pop_front();
        }
        self.mid_price_history.push_back(mid_dec);

        // 7. Возвращаем тензор только если окно заполнено
        if self.buffer.len() == self.seq_len {
            // Плоский вектор: [snapshot_0 (150), snapshot_1 (150), ..., snapshot_N (150)]
            let mut flattened: Vec<f32> = self.buffer.iter().flatten().cloned().collect();
            
            // Задача 091: Добавляем каналы Past Returns
            if !self.past_returns_lags.is_empty() {
                for lag in &self.past_returns_lags {
                    let log_return = self.calculate_log_returns(*lag);
                    // Broadcast на 50 уровней
                    for _ in 0..50 {
                        flattened.push(log_return);
                    }
                }
            }
            
            Ok(Some(flattened))
        } else {
            Ok(None)
        }
    }

    /// Zero-copy версия process_snapshot (Задача 078.3)
    /// Пишет нормализованные фичи напрямую в переданный буфер без промежуточных Vec
    /// 
    /// Требование: Без создания промежуточных Vec для ask_p, ask_v, bid_p, bid_v, features
    /// Нормализация применяется на лету в цикле
    pub fn process_snapshot_to_buffer(
        &mut self,
        ob: &crate::data::orderbook::OrderBookSnapshot,
        buffer: &mut [f32],
    ) -> Result<Option<Vec<f32>>> {
        // Проверяем размер буфера
        if buffer.len() != 150 {
            anyhow::bail!("Buffer size must be 150, got {}", buffer.len());
        }

        let mid = ob.get_mid_price() as f32;
        let mid_dec = ob.get_mid_price_dec();  // Задача 091
        
        if mid <= 0.0 {
            return Ok(None); // Недостаточно данных в стакане
        }

        // Получаем raw данные (это единственный Vec, который нельзя избежать)
        let raw = ob.get_flat_snapshot();
        if raw.len() != 200 {
            anyhow::bail!("Expected 200 raw features, got {}", raw.len());
        }

        // Zero-copy: Вычисляем 3 канала на лету и пишем напрямую в буфер
        // Структура raw: [ask_p_0, ask_v_0, ask_p_1, ask_v_1, ..., ask_p_49, ask_v_49,
        //                 bid_p_0, bid_v_0, bid_p_1, bid_v_1, ..., bid_p_49, bid_v_49]
        
        for i in 0..50 {
            // Извлекаем компоненты напрямую из raw без промежуточных Vec
            let ask_p = raw[i * 2];
            let ask_v = raw[i * 2 + 1];
            let bid_p = raw[100 + i * 2];
            let bid_v = raw[100 + i * 2 + 1];

            // Канал 0: Normalized Price
            let price_ch = (ask_p + bid_p) / 2.0;
            
            // Канал 1: Log Volume
            let vol_ch = ask_v + bid_v;
            
            // Канал 2: Static Level Imbalance
            let imb_ch = (bid_v - ask_v) / (bid_v + ask_v + 1e-7);

            // Применяем нормализацию на лету в зависимости от типа скейлера
            match self.normalizer.scaler_type {
                crate::ml::normalization::ScalerType::ZScore => {
                    buffer[i] = (price_ch - self.normalizer.means[i]) * self.normalizer.inv_stds[i];
                    buffer[50 + i] = (vol_ch - self.normalizer.means[50 + i]) * self.normalizer.inv_stds[50 + i];
                    buffer[100 + i] = (imb_ch - self.normalizer.means[100 + i]) * self.normalizer.inv_stds[100 + i];
                }
                crate::ml::normalization::ScalerType::Robust => {
                    buffer[i] = (price_ch - self.normalizer.medians[i]) * self.normalizer.inv_iqrs[i];
                    buffer[50 + i] = (vol_ch - self.normalizer.medians[50 + i]) * self.normalizer.inv_iqrs[50 + i];
                    buffer[100 + i] = (imb_ch - self.normalizer.medians[100 + i]) * self.normalizer.inv_iqrs[100 + i];
                }
                crate::ml::normalization::ScalerType::WinsorRobust => {
                    // 1. Клиппинг (винзоризация)
                    let price_clipped = price_ch.clamp(self.normalizer.winsor_low[i], self.normalizer.winsor_high[i]);
                    let vol_clipped = vol_ch.clamp(self.normalizer.winsor_low[50 + i], self.normalizer.winsor_high[50 + i]);
                    let imb_clipped = imb_ch.clamp(self.normalizer.winsor_low[100 + i], self.normalizer.winsor_high[100 + i]);
                    
                    // 2. Robust масштабирование
                    buffer[i] = (price_clipped - self.normalizer.medians[i]) * self.normalizer.inv_iqrs[i];
                    buffer[50 + i] = (vol_clipped - self.normalizer.medians[50 + i]) * self.normalizer.inv_iqrs[50 + i];
                    buffer[100 + i] = (imb_clipped - self.normalizer.medians[100 + i]) * self.normalizer.inv_iqrs[100 + i];
                }
            }
        }

        // Валидация (NaN/Inf check) ПОСЛЕ нормализации
        for val in buffer.iter() {
            if !val.is_finite() {
                anyhow::bail!("Invalid feature value detected (NaN or Inf) after normalization");
            }
        }

        // Добавляем в историю (копируем 150 f32 = 600 байт, это приемлемо)
        let snapshot: Vec<f32> = buffer.to_vec();
        self.add_snapshot(snapshot);
        
        // Задача 091: Обновляем mid_price_history для расчета log-returns
        if self.mid_price_history.len() >= self.past_returns_lags.iter().max().copied().unwrap_or(100) {
            self.mid_price_history.pop_front();
        }
        self.mid_price_history.push_back(mid_dec);

        // Возвращаем тензор только если окно заполнено
        if self.buffer.len() == self.seq_len {
            let mut flattened: Vec<f32> = self.buffer.iter().flatten().cloned().collect();
            
            // Задача 091: Добавляем каналы Past Returns
            if !self.past_returns_lags.is_empty() {
                for lag in &self.past_returns_lags {
                    let log_return = self.calculate_log_returns(*lag);
                    // Broadcast на 50 уровней
                    for _ in 0..50 {
                        flattened.push(log_return);
                    }
                }
            }
            
            Ok(Some(flattened))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::orderbook::OrderBook;
    use crate::data::types::{OrderBookUpdate, PriceLevel};
    use smallvec::SmallVec;

    /// Unit Test: Проверяем, что process_snapshot возвращает None пока окно не заполнено
    #[test]
    fn test_returns_none_until_window_full() {
        let mean = vec![0.0; 150];
        let std = vec![1.0; 150];
        let normalizer = Normalizer::new(mean, std);
        
        let seq_len = 3;
        let mut builder = TensorBuilder::new(normalizer, seq_len);
        
        // Создаем тестовый OrderBook
        let mut ob = OrderBook::new("BTCUSDT");
        let mut bids = SmallVec::new();
        let mut asks = SmallVec::new();
        
        // Добавляем 50 уровней
        for i in 0..50 {
            bids.push(PriceLevel { 
                price: 50000.0 - (i as f64) * 0.1, 
                size: 1.0 + (i as f64) * 0.01 
            });
            asks.push(PriceLevel { 
                price: 50001.0 + (i as f64) * 0.1, 
                size: 1.0 + (i as f64) * 0.01 
            });
        }
        
        let update = OrderBookUpdate {
            symbol: "BTCUSDT".to_string(),
            timestamp_ms: 1000,
            last_update_id: 1,
            is_snapshot: true,
            bids,
            asks,
            checksum: None,
        };
        
        ob.apply_update(&update);
        
        // Первый снимок - должен вернуть None
        let result = builder.process_snapshot(&ob).unwrap();
        assert!(result.is_none(), "Expected None for first snapshot");
        
        // Второй снимок - должен вернуть None
        let result = builder.process_snapshot(&ob).unwrap();
        assert!(result.is_none(), "Expected None for second snapshot");
        
        // Третий снимок - должен вернуть Some
        let result = builder.process_snapshot(&ob).unwrap();
        assert!(result.is_some(), "Expected Some for third snapshot");
        
        let tensor = result.unwrap();
        // Проверяем размер: seq_len * 150 (3 канала * 50 уровней)
        assert_eq!(tensor.len(), seq_len * 150);
    }

    /// Test: Проверяем, что возвращается плоский вектор размером seq_len * 150
    #[test]
    fn test_returns_flat_vector() {
        let mean = vec![0.0; 150];
        let std = vec![1.0; 150];
        let normalizer = Normalizer::new(mean, std);
        
        let seq_len = 2;
        let mut builder = TensorBuilder::new(normalizer, seq_len);
        
        // Создаем тестовый OrderBook
        let mut ob = OrderBook::new("ETHUSDT");
        let mut bids = SmallVec::new();
        let mut asks = SmallVec::new();
        
        for i in 0..50 {
            bids.push(PriceLevel { 
                price: 3000.0 - (i as f64) * 0.1, 
                size: 2.0 
            });
            asks.push(PriceLevel { 
                price: 3001.0 + (i as f64) * 0.1, 
                size: 2.0 
            });
        }
        
        let update = OrderBookUpdate {
            symbol: "ETHUSDT".to_string(),
            timestamp_ms: 2000,
            last_update_id: 2,
            is_snapshot: true,
            bids,
            asks,
            checksum: None,
        };
        
        ob.apply_update(&update);
        
        // Добавляем 2 снимка
        builder.process_snapshot(&ob).unwrap();
        let result = builder.process_snapshot(&ob).unwrap();
        
        assert!(result.is_some());
        let tensor = result.unwrap();
        
        // Проверяем размер: seq_len * 150 (3 канала * 50 уровней)
        assert_eq!(tensor.len(), seq_len * 150, "Expected flat vector of size {}", seq_len * 150);
    }

    /// Test: Проверяем, что валидация работает ПОСЛЕ нормализации
    #[test]
    fn test_validation_after_normalization() {
        let mean = vec![0.0; 150];
        let std = vec![1.0; 150];
        let normalizer = Normalizer::new(mean, std);
        
        let seq_len = 1;
        let mut builder = TensorBuilder::new(normalizer, seq_len);
        
        // Создаем OrderBook с нулевыми объемами (будут ln(1+0) = 0)
        let mut ob = OrderBook::new("BTCUSDT");
        let mut bids = SmallVec::new();
        let mut asks = SmallVec::new();
        
        for i in 0..50 {
            bids.push(PriceLevel { 
                price: 50000.0 - (i as f64) * 0.1, 
                size: 0.0 // Нулевой объем
            });
            asks.push(PriceLevel { 
                price: 50001.0 + (i as f64) * 0.1, 
                size: 0.0 // Нулевой объем
            });
        }
        
        let update = OrderBookUpdate {
            symbol: "BTCUSDT".to_string(),
            timestamp_ms: 1000,
            last_update_id: 1,
            is_snapshot: true,
            bids,
            asks,
            checksum: None,
        };
        
        ob.apply_update(&update);
        
        // Должно вернуть Some (валидация прошла)
        let result = builder.process_snapshot(&ob).unwrap();
        assert!(result.is_some());
    }
}



/// Структура для хранения истории снапшотов для расчета признаков режима
pub struct RegimeFeatureCalculator {
    /// История временных меток (в миллисекундах)
    timestamps: VecDeque<i64>,
    /// История mid_price
    mid_prices: VecDeque<f64>,
    /// История ask_p_0
    ask_prices: VecDeque<f64>,
    /// История ask_v_0
    ask_volumes: VecDeque<f64>,
    /// История bid_p_0
    bid_prices: VecDeque<f64>,
    /// История bid_v_0
    bid_volumes: VecDeque<f64>,
    /// Размер окна для расчета признаков
    window: usize,
}

impl RegimeFeatureCalculator {
    /// Создает новый калькулятор с заданным размером окна
    pub fn new(window: usize) -> Self {
        Self {
            timestamps: VecDeque::with_capacity(window),
            mid_prices: VecDeque::with_capacity(window),
            ask_prices: VecDeque::with_capacity(window),
            ask_volumes: VecDeque::with_capacity(window),
            bid_prices: VecDeque::with_capacity(window),
            bid_volumes: VecDeque::with_capacity(window),
            window,
        }
    }
    
    /// Добавляет новый снапшот в историю
    pub fn push(&mut self, ob: &OrderBook, timestamp_ms: i64) {
        // Вычисляем mid_price
        let best_bid = ob.bids.first().map(|l| l.price).unwrap_or(Decimal::ZERO);
        let best_ask = ob.asks.first().map(|l| l.price).unwrap_or(Decimal::ZERO);
        let mid_price = ((best_bid + best_ask) / Decimal::TWO).to_f64().unwrap_or(0.0);
        
        // Извлекаем данные первого уровня
        let ask_p_0 = best_ask.to_f64().unwrap_or(0.0);
        let ask_v_0 = ob.asks.first().map(|l| l.quantity.to_f64().unwrap_or(0.0)).unwrap_or(0.0);
        let bid_p_0 = best_bid.to_f64().unwrap_or(0.0);
        let bid_v_0 = ob.bids.first().map(|l| l.quantity.to_f64().unwrap_or(0.0)).unwrap_or(0.0);
        
        // Добавляем в историю
        self.timestamps.push_back(timestamp_ms);
        self.mid_prices.push_back(mid_price);
        self.ask_prices.push_back(ask_p_0);
        self.ask_volumes.push_back(ask_v_0);
        self.bid_prices.push_back(bid_p_0);
        self.bid_volumes.push_back(bid_v_0);
        
        // Ограничиваем размер окна
        if self.timestamps.len() > self.window {
            self.timestamps.pop_front();
            self.mid_prices.pop_front();
            self.ask_prices.pop_front();
            self.ask_volumes.pop_front();
            self.bid_prices.pop_front();
            self.bid_volumes.pop_front();
        }
    }
    
    /// Вычисляет интенсивность (количество событий в окне)
    fn compute_intensity(&self) -> f64 {
        self.timestamps.len() as f64
    }
    
    /// Вычисляет волатильность (log std mid_price)
    fn compute_volatility(&self) -> f64 {
        if self.mid_prices.len() < 2 {
            return 0.0;
        }
        
        // Вычисляем стандартное отклонение цен
        let mean: f64 = self.mid_prices.iter().sum::<f64>() / self.mid_prices.len() as f64;
        let variance: f64 = self.mid_prices
            .iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f64>() / self.mid_prices.len() as f64;
        
        let std = variance.sqrt();
        
        // Логарифм волатильности
        (std + 1e-8).ln()
    }
    
    /// Вычисляет Z-score спреда
    fn compute_spread_zscore(&self) -> f64 {
        if self.ask_prices.len() < 2 {
            return 0.0;
        }
        
        // Вычисляем спреды
        let spreads: Vec<f64> = self.ask_prices
            .iter()
            .zip(&self.bid_prices)
            .map(|(ask, bid)| ask - bid)
            .collect();
        
        // Вычисляем среднее и std спредов
        let mean: f64 = spreads.iter().sum::<f64>() / spreads.len() as f64;
        let variance: f64 = spreads
            .iter()
            .map(|s| (s - mean).powi(2))
            .sum::<f64>() / spreads.len() as f64;
        
        let std = variance.sqrt();
        
        // Z-score текущего спреда
        let current_spread = spreads.last().copied().unwrap_or(0.0);
        (current_spread - mean) / (std + 1e-8)
    }
    
    /// Вычисляет Order Flow Imbalance (OFI)
    fn compute_ofi(&self) -> f64 {
        if self.ask_prices.len() < 2 {
            return 0.0;
        }
        
        let mut ofi_sum = 0.0;
        
        for i in 1..self.ask_prices.len() {
            // Изменение объема на bid стороне
            let delta_bid = if self.bid_prices[i] >= self.bid_prices[i - 1] {
                if self.bid_prices[i] == self.bid_prices[i - 1] {
                    self.bid_volumes[i] - self.bid_volumes[i - 1]
                } else {
                    self.bid_volumes[i]
                }
            } else {
                -self.bid_volumes[i - 1]
            };
            
            // Изменение объема на ask стороне
            let delta_ask = if self.ask_prices[i] <= self.ask_prices[i - 1] {
                if self.ask_prices[i] == self.ask_prices[i - 1] {
                    self.ask_volumes[i] - self.ask_volumes[i - 1]
                } else {
                    self.ask_volumes[i]
                }
            } else {
                -self.ask_volumes[i - 1]
            };
            
            // OFI = delta_bid - delta_ask
            ofi_sum += delta_bid - delta_ask;
        }
        
        ofi_sum
    }
    
    /// Вычисляет все признаки режима
    /// 
    /// Возвращает вектор из 4 признаков: [intensity, volatility, spread_zscore, ofi]
    pub fn compute_features(&self) -> Option<Vec<f64>> {
        if self.timestamps.len() < 2 {
            return None;
        }
        
        let intensity = self.compute_intensity();
        let volatility = self.compute_volatility();
        let spread_zscore = self.compute_spread_zscore();
        let ofi = self.compute_ofi();
        
        Some(vec![intensity, volatility, spread_zscore, ofi])
    }
    
    /// Проверяет, достаточно ли данных для расчета признаков
    pub fn is_ready(&self) -> bool {
        self.timestamps.len() >= self.window.min(100) // Минимум 100 событий
    }
}

#[cfg(test)]
mod regime_tests {
    use super::*;
    use rust_decimal_macros::dec;
    
    #[test]
    fn test_regime_feature_calculator() {
        let mut calc = RegimeFeatureCalculator::new(1000);
        
        // Создаем тестовый orderbook
        let mut ob = OrderBook::new("BTCUSDT".to_string());
        ob.update_bid(dec!(100.0), dec!(1.0));
        ob.update_ask(dec!(101.0), dec!(1.0));
        
        // Добавляем несколько снапшотов
        for i in 0..150 {
            calc.push(&ob, i * 1000);
            
            // Изменяем цены для создания волатильности
            let price_change = (i as f64 * 0.1).sin();
            ob.update_bid(dec!(100.0) + Decimal::from_f64_retain(price_change).unwrap(), dec!(1.0));
            ob.update_ask(dec!(101.0) + Decimal::from_f64_retain(price_change).unwrap(), dec!(1.0));
        }
        
        assert!(calc.is_ready());
        
        let features = calc.compute_features().unwrap();
        assert_eq!(features.len(), 4);
        
        // Проверяем, что признаки не NaN
        assert!(features.iter().all(|f| f.is_finite()));
        
        // Intensity должна быть равна количеству событий
        assert_eq!(features[0], 150.0);
    }
}

#[cfg(test)]
mod simd_tests {
    use super::*;
    
    /// Test SIMD Normalization: Проверяем корректность SIMD-нормализации
    #[test]
    fn test_simd_normalization() {
        // Создаем тестовые данные: 100 элементов
        let mut features = vec![100.0; 100];
        let means = vec![50.0; 100];
        let stds = vec![10.0; 100];
        
        // Вычисляем inv_stds
        let inv_stds: Vec<f32> = stds.iter().map(|&s| 1.0 / s.max(EPSILON)).collect();
        
        // Применяем SIMD-нормализацию
        normalize_features_simd(&mut features, &means, &inv_stds);
        
        // Проверяем результат: (100.0 - 50.0) * (1.0 / 10.0) = 5.0
        for &val in &features {
            assert!((val - 5.0).abs() < 1e-6, "Expected 5.0, got {}", val);
        }
    }
    
    /// Test SIMD with Non-Aligned Size: Проверяем работу с размером не кратным 8
    #[test]
    fn test_simd_non_aligned_size() {
        // 13 элементов (не кратно 8)
        let mut features = vec![42.0; 13];
        let means = vec![10.0; 13];
        let stds = vec![5.0; 13];
        
        let inv_stds: Vec<f32> = stds.iter().map(|&s| 1.0 / s).collect();
        
        normalize_features_simd(&mut features, &means, &inv_stds);
        
        // Проверяем результат: (42.0 - 10.0) * (1.0 / 5.0) = 6.4
        for &val in &features {
            assert!((val - 6.4).abs() < 1e-6, "Expected 6.4, got {}", val);
        }
    }
    
    /// Test SIMD with Zero Std: Проверяем защиту от деления на 0
    #[test]
    fn test_simd_zero_std() {
        let mut features = vec![100.0; 16];
        let means = vec![50.0; 16];
        let stds = vec![0.0; 16]; // Все std=0
        
        // inv_stds должны использовать EPSILON
        let inv_stds: Vec<f32> = stds.iter().map(|&s| 1.0 / s.max(EPSILON)).collect();
        
        normalize_features_simd(&mut features, &means, &inv_stds);
        
        // Проверяем, что все значения конечны
        assert!(features.iter().all(|x| x.is_finite()));
        
        // Все значения должны быть очень большими: (100.0 - 50.0) / EPSILON
        for &val in &features {
            assert!(val > 1e6, "Expected large value, got {}", val);
        }
    }
    
    /// Test SIMD Performance Comparison: Сравниваем SIMD с скалярной версией
    #[test]
    fn test_simd_correctness() {
        use std::time::Instant;
        
        // Большой массив для тестирования
        let size = 10000;
        let mut features_simd = vec![123.456; size];
        let mut features_scalar = features_simd.clone();
        
        let means: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let stds: Vec<f32> = (0..size).map(|i| 1.0 + (i as f32) * 0.01).collect();
        let inv_stds: Vec<f32> = stds.iter().map(|&s| 1.0 / s).collect();
        
        // SIMD версия
        let start = Instant::now();
        normalize_features_simd(&mut features_simd, &means, &inv_stds);
        let simd_duration = start.elapsed();
        
        // Скалярная версия
        let start = Instant::now();
        for i in 0..size {
            features_scalar[i] = (features_scalar[i] - means[i]) * inv_stds[i];
        }
        let scalar_duration = start.elapsed();
        
        // Проверяем, что результаты идентичны
        for i in 0..size {
            assert!(
                (features_simd[i] - features_scalar[i]).abs() < 1e-5,
                "Mismatch at index {}: SIMD={}, Scalar={}",
                i, features_simd[i], features_scalar[i]
            );
        }
        
        println!("SIMD: {:?}, Scalar: {:?}, Speedup: {:.2}x", 
                 simd_duration, scalar_duration, 
                 scalar_duration.as_nanos() as f64 / simd_duration.as_nanos() as f64);
    }
}

#[cfg(test)]
mod time_slice_simd_tests {
    use super::*;
    
    /// Test Time Slice SIMD Normalization: Проверяем корректность нормализации временного среза
    #[test]
    fn test_time_slice_simd_normalization() {
        // Создаем временной срез (10 временных шагов)
        let mut time_slice = vec![100.0; 10];
        let mean = 50.0;
        let std = 10.0;
        let inv_std = 1.0 / std;
        
        // Применяем SIMD-нормализацию
        normalize_time_slice_simd(&mut time_slice, mean, inv_std);
        
        // Проверяем результат: (100.0 - 50.0) * (1.0 / 10.0) = 5.0
        for &val in &time_slice {
            assert!((val - 5.0).abs() < 1e-6, "Expected 5.0, got {}", val);
        }
    }
    
    /// Test Time Slice with Non-Aligned Size: Проверяем работу с размером не кратным 8
    #[test]
    fn test_time_slice_non_aligned_size() {
        // 13 временных шагов (не кратно 8)
        let mut time_slice = vec![42.0; 13];
        let mean = 10.0;
        let std = 5.0;
        let inv_std = 1.0 / std;
        
        normalize_time_slice_simd(&mut time_slice, mean, inv_std);
        
        // Проверяем результат: (42.0 - 10.0) * (1.0 / 5.0) = 6.4
        for &val in &time_slice {
            assert!((val - 6.4).abs() < 1e-6, "Expected 6.4, got {}", val);
        }
    }
    
    /// Test Time Slice with Zero Std: Проверяем защиту от деления на 0
    #[test]
    fn test_time_slice_zero_std() {
        let mut time_slice = vec![100.0; 16];
        let mean = 50.0;
        let inv_std = 1.0 / EPSILON; // Защита от деления на 0
        
        normalize_time_slice_simd(&mut time_slice, mean, inv_std);
        
        // Проверяем, что все значения конечны
        assert!(time_slice.iter().all(|x| x.is_finite()));
        
        // Все значения должны быть очень большими: (100.0 - 50.0) / EPSILON
        for &val in &time_slice {
            assert!(val > 1e6, "Expected large value, got {}", val);
        }
    }
    
    /// Test Time Slice Performance: Сравниваем SIMD с скалярной версией
    #[test]
    fn test_time_slice_simd_correctness() {
        use std::time::Instant;
        
        // Большой временной срез
        let size = 10000;
        let mut time_slice_simd = vec![123.456; size];
        let mut time_slice_scalar = time_slice_simd.clone();
        
        let mean = 50.0;
        let std = 10.0;
        let inv_std = 1.0 / std;
        
        // SIMD версия
        let start = Instant::now();
        normalize_time_slice_simd(&mut time_slice_simd, mean, inv_std);
        let simd_duration = start.elapsed();
        
        // Скалярная версия
        let start = Instant::now();
        for i in 0..size {
            time_slice_scalar[i] = (time_slice_scalar[i] - mean) * inv_std;
        }
        let scalar_duration = start.elapsed();
        
        // Проверяем, что результаты идентичны
        for i in 0..size {
            assert!(
                (time_slice_simd[i] - time_slice_scalar[i]).abs() < 1e-5,
                "Mismatch at index {}: SIMD={}, Scalar={}",
                i, time_slice_simd[i], time_slice_scalar[i]
            );
        }
        
        println!("Time Slice SIMD: {:?}, Scalar: {:?}, Speedup: {:.2}x", 
                 simd_duration, scalar_duration, 
                 scalar_duration.as_nanos() as f64 / simd_duration.as_nanos() as f64);
    }

    /// Test: Проверяем build_tensor() с Left-Padding (Задача 097)
    #[test]
    fn test_build_tensor_with_left_padding() {
        let mean = vec![0.0; 150];
        let std = vec![1.0; 150];
        let normalizer = Normalizer::new(mean, std);
        
        let seq_len = 10;
        let mut builder = TensorBuilder::new(normalizer, seq_len);
        builder.channels = 3;
        builder.levels = 50;
        
        // Добавляем только 1 снимок (вместо 10)
        let snapshot = vec![1.0; 150]; // 3 канала * 50 уровней
        builder.buffer.push_back(snapshot);
        
        // Строим тензор
        let tensor = builder.build_tensor();
        
        // Проверяем форму
        assert_eq!(tensor.shape(), &[1, 3, 50, 10]);
        
        // Проверяем Left-Padding: первые 9 временных шагов должны быть нулями
        for t in 0..9 {
            for c in 0..3 {
                for l in 0..50 {
                    assert_eq!(tensor[[0, c, l, t]], 0.0, 
                        "Expected zero at time step {}, channel {}, level {}", t, c, l);
                }
            }
        }
        
        // Проверяем, что последний временной шаг содержит данные (1.0)
        for c in 0..3 {
            for l in 0..50 {
                assert_eq!(tensor[[0, c, l, 9]], 1.0, 
                    "Expected 1.0 at time step 9, channel {}, level {}", c, l);
            }
        }
    }

    /// Test: Проверяем build_tensor() без padding (буфер полный)
    #[test]
    fn test_build_tensor_without_padding() {
        let mean = vec![0.0; 150];
        let std = vec![1.0; 150];
        let normalizer = Normalizer::new(mean, std);
        
        let seq_len = 3;
        let mut builder = TensorBuilder::new(normalizer, seq_len);
        builder.channels = 3;
        builder.levels = 50;
        
        // Добавляем 3 снимка (полный буфер)
        for i in 0..3 {
            let snapshot = vec![(i as f32) + 1.0; 150];
            builder.buffer.push_back(snapshot);
        }
        
        // Строим тензор
        let tensor = builder.build_tensor();
        
        // Проверяем форму
        assert_eq!(tensor.shape(), &[1, 3, 50, 3]);
        
        // Проверяем, что все временные шаги содержат данные (без padding)
        for t in 0..3 {
            for c in 0..3 {
                for l in 0..50 {
                    let expected = (t as f32) + 1.0;
                    assert_eq!(tensor[[0, c, l, t]], expected, 
                        "Expected {} at time step {}, channel {}, level {}", expected, t, c, l);
                }
            }
        }
    }
}
