# 034 - ML Tensor Normalization
Цель задачи: Реализовать структуру Normalizer для загрузки параметров Z-score из norm.json и их применения к входному вектору признаков. Для обеспечения корректного порядка признаков (согласно Python-обучению) необходимо использовать docs/data_schema.json как эталон последовательности колонок.

Файлы:

src/ml/normalization.rs (создать)
src/ml/mod.rs (обновить)
Инструкции для Gemini:

src/ml/normalization.rs: Реализовать загрузку параметров с проверкой соответствия схеме данных.
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use anyhow::{Result, Context, bail};

#[derive(Debug, Deserialize)]
struct NormParams {
    mean: f32,
    std: f32,
}

#[derive(Debug, Deserialize)]
struct DataSchema {
    columns: Vec<String>,
}

#[derive(Debug)]
pub struct Normalizer {
    pub means: Vec<f32>,
    pub stds: Vec<f32>,
}

impl Normalizer {
    /// Загрузка параметров на основе схемы и файла нормализации
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
            stds.push(params.std.max(1e-6)); // Защита от деления на 0
        }

        if means.len() != 200 {
            bail!("Expected 200 features, found {}", means.len());
        }

        Ok(Self { means, stds })
    }

    /// Применяет (x - mean) / std к срезу данных
    pub fn normalize(&self, features: &mut [f32]) {
        for (i, val) in features.iter_mut().enumerate() {
            if let (Some(&m), Some(&s)) = (self.means.get(i), self.stds.get(i)) {
                *val = (*val - m) / s;
            }
        }
    }
}
src/ml/mod.rs:
pub mod types;
pub mod normalization;

pub use types::*;
pub use normalization::*;
Технические требования:

Порядок колонок: Использовать docs/data_schema.json для получения списка колонок. Пропускать метаданные (timestamp_ms, last_update_id), оставляя только 200 признаков.
Безопасность: Обязательная проверка на деление на ноль через max(1e-6).
Ошибки: Использовать anyhow::bail! если количество найденных параметров не равно 200 или колонка из схемы отсутствует в norm.json.
Типы: Все вычисления и параметры в f32.