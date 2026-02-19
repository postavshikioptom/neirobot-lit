# 035 - ML Tensor Builder
Цель задачи: Реализовать структуру TensorBuilder, которая превращает последовательность снимков стакана (OrderBook) в многомерный тензор признаков (f32), готовый для подачи в Transformer-модель. Модуль должен поддерживать скользящее окно из последних seq_len снимков, выполнять расчет признаков (relative price, log volume), нормализацию и проверку на валидность (NaN/Inf).

Файлы:

src/ml/tensor.rs (создать)
src/ml/mod.rs (обновить)
Инструкции для Gemini:

src/ml/tensor.rs: Создать TensorBuilder с внутренним буфером для хранения истории признаков.
use crate::data::order_book::OrderBook;
use crate::ml::normalization::Normalizer;
use std::collections::VecDeque;
use anyhow::{Result, Context};

pub struct TensorBuilder {
    normalizer: Normalizer,
    history: VecDeque<Vec<f32>>,
    seq_len: usize,
}

impl TensorBuilder {
    pub fn new(normalizer: Normalizer, seq_len: usize) -> Self {
        Self {
            normalizer,
            history: VecDeque::with_capacity(seq_len),
            seq_len,
        }
    }

    /// Добавляет новый снимок стакана в историю и возвращает полный тензор
    pub fn process_snapshot(&mut self, ob: &OrderBook) -> Result<Option<Vec<f32>>> {
        let mid = ob.get_mid_price() as f32;
        
        // 1. Извлекаем топ-50 уровней (200 фич)
        let mut features = ob.get_top_n(50); 
        if features.len() != 200 {
            anyhow::bail!("Expected 200 features from OrderBook, got {}", features.len());
        }

        // 2. Feature Engineering (как в задаче 022)
        for i in (0..features.len()).step_by(2) {
            let price = features[i];
            let vol = features[i+1];

            // Relative Price: (p - mid) / mid
            features[i] = if mid > 0.0 { (price - mid) / mid } else { 0.0 };
            // Log Volume: ln(1 + vol)
            features[i+1] = (1.0 + vol).ln();
        }

        // 3. Normalization (Z-score)
        self.normalizer.normalize(&mut features);

        // 4. Валидация (NaN/Inf check)
        for val in &features {
            if !val.is_finite() {
                anyhow::bail!("Invalid feature value detected (NaN or Inf)");
            }
        }

        // 5. Обновляем скользящее окно
        if self.history.len() >= self.seq_len {
            self.history.pop_front();
        }
        self.history.push_back(features);

        // Возвращаем тензор только если окно заполнено
        if self.history.len() == self.seq_len {
            // Плоский вектор: [snapshot_0 (200), snapshot_1 (200), ..., snapshot_N]
            let flattened: Vec<f32> = self.history.iter().flatten().cloned().collect();
            Ok(Some(flattened))
        } else {
            Ok(None)
        }
    }
}
src/ml/mod.rs:
pub mod types;
pub mod normalization;
pub mod tensor;

pub use types::*;
pub use normalization::*;
pub use tensor::*;
Технические требования:

Скользящее окно: Поддерживать seq_len (обычно 100) последних состояний через VecDeque.
Feature Engineering: Строгое соответствие Python-логике (relative price + log volume).
Валидация: Использовать f32::is_finite() для каждой фичи. При обнаружении NaN/Inf — bail!.
Типы: Выход — плоский Vec<f32> размером seq_len * 200.
Защита: Если mid == 0.0, относительная цена должна быть 0.0.