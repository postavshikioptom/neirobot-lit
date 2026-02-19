use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use anyhow::{Result, Context, bail};
use aligned_vec::{AVec, ConstAlign};

#[derive(Debug, Deserialize)]
struct NormParams {
    mean: f32,
    std: f32,
}

#[derive(Debug, Deserialize)]
struct DataSchema {
    columns: Vec<String>,
}

/// Константа для защиты от деления на 0 при нормализации
const EPSILON: f32 = 1e-8;

/// Тип нормализации (Задача 240)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalerType {
    ZScore,
    Robust,
    WinsorRobust,  // Винзоризация + Robust Scaling
}

#[derive(Debug)]
pub struct Normalizer {
    pub scaler_type: ScalerType,
    
    // Параметры для Z-score
    pub means: AVec<f32, ConstAlign<32>>,
    pub stds: AVec<f32, ConstAlign<32>>,
    pub inv_stds: AVec<f32, ConstAlign<32>>, // Предварительно вычисленные обратные значения
    
    // Параметры для Robust Scaling (Задача 240)
    pub medians: AVec<f32, ConstAlign<32>>,
    pub iqrs: AVec<f32, ConstAlign<32>>,
    pub inv_iqrs: AVec<f32, ConstAlign<32>>, // Предварительно вычисленные обратные значения IQR
    
    // Параметры для WinsorRobust (Задача 240)
    pub winsor_low: AVec<f32, ConstAlign<32>>,
    pub winsor_high: AVec<f32, ConstAlign<32>>,
}

impl Normalizer {
    /// Создает нормализатор Z-score из векторов средних и отклонений
    /// Автоматически вычисляет обратные значения std для SIMD-оптимизации
    pub fn new(means: Vec<f32>, stds: Vec<f32>) -> Self {
        // Создаем выровненные векторы
        let mut aligned_means = AVec::<f32, ConstAlign<32>>::with_capacity(32, means.len());
        let mut aligned_stds = AVec::<f32, ConstAlign<32>>::with_capacity(32, stds.len());
        let mut aligned_inv_stds = AVec::<f32, ConstAlign<32>>::with_capacity(32, stds.len());
        
        // Копируем данные и вычисляем обратные значения
        for (m, s) in means.iter().zip(stds.iter()) {
            aligned_means.push(*m);
            let safe_std = s.max(EPSILON); // Защита от деления на 0
            aligned_stds.push(safe_std);
            aligned_inv_stds.push(1.0 / safe_std);
        }
        
        // Создаем пустые векторы для robust параметров
        let empty_vec = AVec::<f32, ConstAlign<32>>::new(32);
        
        Self { 
            scaler_type: ScalerType::ZScore,
            means: aligned_means,
            stds: aligned_stds,
            inv_stds: aligned_inv_stds,
            medians: empty_vec.clone(),
            iqrs: empty_vec.clone(),
            inv_iqrs: empty_vec.clone(),
            winsor_low: empty_vec.clone(),
            winsor_high: empty_vec,
        }
    }
    
    /// Создает нормализатор Robust Scaler из векторов медиан и IQR (Задача 240)
    /// Автоматически вычисляет обратные значения IQR для SIMD-оптимизации
    pub fn new_robust(medians: Vec<f32>, iqrs: Vec<f32>) -> Self {
        // Создаем выровненные векторы
        let mut aligned_medians = AVec::<f32, ConstAlign<32>>::with_capacity(32, medians.len());
        let mut aligned_iqrs = AVec::<f32, ConstAlign<32>>::with_capacity(32, iqrs.len());
        let mut aligned_inv_iqrs = AVec::<f32, ConstAlign<32>>::with_capacity(32, iqrs.len());
        
        // Копируем данные и вычисляем обратные значения
        for (med, iqr) in medians.iter().zip(iqrs.iter()) {
            aligned_medians.push(*med);
            let safe_iqr = iqr.max(EPSILON); // Защита от деления на 0
            aligned_iqrs.push(safe_iqr);
            aligned_inv_iqrs.push(1.0 / safe_iqr);
        }
        
        // Создаем пустые векторы для zscore параметров
        let empty_vec = AVec::<f32, ConstAlign<32>>::new(32);
        
        Self { 
            scaler_type: ScalerType::Robust,
            means: empty_vec.clone(),
            stds: empty_vec.clone(),
            inv_stds: empty_vec.clone(),
            medians: aligned_medians,
            iqrs: aligned_iqrs,
            inv_iqrs: aligned_inv_iqrs,
            winsor_low: empty_vec.clone(),
            winsor_high: empty_vec,
        }
    }
    
    /// Создает нормализатор WinsorRobust (винзоризация + robust scaling) (Задача 240)
    pub fn new_winsor_robust(winsor_low: Vec<f32>, winsor_high: Vec<f32>, medians: Vec<f32>, iqrs: Vec<f32>) -> Self {
        // Создаем выровненные векторы для винзоризации
        let mut aligned_winsor_low = AVec::<f32, ConstAlign<32>>::with_capacity(32, winsor_low.len());
        let mut aligned_winsor_high = AVec::<f32, ConstAlign<32>>::with_capacity(32, winsor_high.len());
        aligned_winsor_low.extend_from_slice(&winsor_low);
        aligned_winsor_high.extend_from_slice(&winsor_high);
        
        // Создаем выровненные векторы для robust scaling
        let mut aligned_medians = AVec::<f32, ConstAlign<32>>::with_capacity(32, medians.len());
        let mut aligned_iqrs = AVec::<f32, ConstAlign<32>>::with_capacity(32, iqrs.len());
        let mut aligned_inv_iqrs = AVec::<f32, ConstAlign<32>>::with_capacity(32, iqrs.len());
        
        for (med, iqr) in medians.iter().zip(iqrs.iter()) {
            aligned_medians.push(*med);
            let safe_iqr = iqr.max(EPSILON);
            aligned_iqrs.push(safe_iqr);
            aligned_inv_iqrs.push(1.0 / safe_iqr);
        }
        
        let empty_vec = AVec::<f32, ConstAlign<32>>::new(32);
        
        Self {
            scaler_type: ScalerType::WinsorRobust,
            means: empty_vec.clone(),
            stds: empty_vec.clone(),
            inv_stds: empty_vec.clone(),
            medians: aligned_medians,
            iqrs: aligned_iqrs,
            inv_iqrs: aligned_inv_iqrs,
            winsor_low: aligned_winsor_low,
            winsor_high: aligned_winsor_high,
        }
    }

    /// Загрузка параметров на основе схемы и файла нормализации (устарело, используйте metadata.json)
    pub fn load(schema_path: &Path, norm_path: &Path) -> Result<Self> {
        // 1. Загружаем схему для определения порядка колонок
        let schema_str = std::fs::read_to_string(schema_path)
            .context("Failed to read schema file")?;
        let schema: DataSchema = serde_json::from_str(&schema_str)?;

        // 2. Загружаем параметры нормализации
        let norm_str = std::fs::read_to_string(norm_path)
            .context("Failed to read norm file")?;
        let norm_map: HashMap<String, NormParams> = serde_json::from_str(&norm_str)?;

        let mut means = Vec::with_capacity(200);
        let mut stds = Vec::with_capacity(200);

        // 3. Мапим параметры строго по порядку из схемы (исключая timestamp_ms и last_update_id)
        for col in schema.columns.iter() {
            if col == "timestamp_ms" || col == "last_update_id" {
                continue;
            }

            let params = norm_map.get(col)
                .with_context(|| format!("Column {} missing in norm.json", col))?;
            
            means.push(params.mean);
            stds.push(params.std.max(EPSILON)); // Защита от деления на 0
        }

        if means.len() != 200 {
            bail!("Expected 200 features, found {}", means.len());
        }

        Ok(Self::new(means, stds))
    }

    /// Применяет нормализацию к срезу данных с использованием SIMD
    /// Выбирает алгоритм в зависимости от типа скейлера (Задача 240)
    pub fn normalize(&self, features: &mut [f32]) {
        match self.scaler_type {
            ScalerType::ZScore => {
                // (x - mean) * inv_std
                crate::ml::tensor::normalize_features_simd(features, &self.means, &self.inv_stds);
            }
            ScalerType::Robust => {
                // (x - median) * inv_iqr
                crate::ml::tensor::normalize_features_simd(features, &self.medians, &self.inv_iqrs);
            }
            ScalerType::WinsorRobust => {
                // 1. Клиппинг (винзоризация)
                crate::ml::tensor::clip_features_simd(features, &self.winsor_low, &self.winsor_high);
                // 2. Robust масштабирование
                crate::ml::tensor::normalize_features_simd(features, &self.medians, &self.inv_iqrs);
            }
        }
    }
}
