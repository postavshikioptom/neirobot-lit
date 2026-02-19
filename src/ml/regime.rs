/// Market Regime Detection через Hidden Markov Model (HMM)
/// 
/// Реализует предсказание режима рынка на основе параметров HMM,
/// обученных в Python (hmmlearn).

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Конфигурация HMM для определения режимов рынка
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeConfig {
    /// Количество скрытых состояний (режимов)
    pub n_components: usize,
    /// Тип ковариационной матрицы
    pub covariance_type: String,
    /// Начальные вероятности состояний
    pub startprob: Vec<f64>,
    /// Матрица переходов между состояниями
    pub transmat: Vec<Vec<f64>>,
    /// Средние значения признаков для каждого состояния
    pub means: Vec<Vec<f64>>,
    /// Ковариационные матрицы для каждого состояния
    pub covars: Vec<Vec<f64>>,
    /// Средние значения признаков для нормализации
    pub feature_means: Vec<f64>,
    /// Стандартные отклонения признаков для нормализации
    pub feature_stds: Vec<f64>,
}

/// Предсказатель режимов рынка на основе HMM
pub struct RegimePredictor {
    config: RegimeConfig,
}

impl RegimePredictor {
    /// Загружает конфигурацию HMM из JSON файла
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .context("Failed to read regime config file")?;
        
        let config: RegimeConfig = serde_json::from_str(&content)
            .context("Failed to parse regime config JSON")?;
        
        Ok(Self { config })
    }
    
    /// Нормализует признаки (Z-score normalization)
    fn normalize_features(&self, features: &[f64]) -> Vec<f64> {
        features
            .iter()
            .zip(&self.config.feature_means)
            .zip(&self.config.feature_stds)
            .map(|((f, mean), std)| (f - mean) / std)
            .collect()
    }
    
    /// Вычисляет логарифм вероятности наблюдения для Gaussian распределения
    /// 
    /// log P(x | μ, σ²) = -0.5 * [log(2π) + log(σ²) + ((x - μ) / σ)²]
    fn log_gaussian_prob(&self, x: &[f64], mean: &[f64], covar: &[f64]) -> f64 {
        let n = x.len();
        let mut log_prob = -0.5 * (n as f64) * (2.0 * std::f64::consts::PI).ln();
        
        for i in 0..n {
            let variance = covar[i];
            let diff = x[i] - mean[i];
            
            log_prob -= 0.5 * variance.ln();
            log_prob -= 0.5 * (diff * diff) / variance;
        }
        
        log_prob
    }
    
    /// Предсказывает текущий режим рынка на основе признаков
    /// 
    /// Использует упрощенный алгоритм: выбирает состояние с максимальной
    /// вероятностью наблюдения (без учета последовательности).
    /// 
    /// Для полноценного HMM inference нужен алгоритм Витерби,
    /// но для online inference достаточно текущего наблюдения.
    pub fn predict_state(&self, features: &[f64]) -> Result<usize> {
        if features.len() != self.config.feature_means.len() {
            anyhow::bail!(
                "Feature dimension mismatch: expected {}, got {}",
                self.config.feature_means.len(),
                features.len()
            );
        }
        
        // Нормализуем признаки
        let normalized = self.normalize_features(features);
        
        // Вычисляем log-вероятность для каждого состояния
        let mut log_probs = Vec::with_capacity(self.config.n_components);
        
        for i in 0..self.config.n_components {
            let log_prior = self.config.startprob[i].ln();
            let log_likelihood = self.log_gaussian_prob(
                &normalized,
                &self.config.means[i],
                &self.config.covars[i],
            );
            
            log_probs.push(log_prior + log_likelihood);
        }
        
        // Выбираем состояние с максимальной вероятностью
        let best_state = log_probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        
        Ok(best_state)
    }
    
    /// Возвращает количество режимов
    pub fn num_regimes(&self) -> usize {
        self.config.n_components
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_normalize_features() {
        let config = RegimeConfig {
            n_components: 2,
            covariance_type: "diag".to_string(),
            startprob: vec![0.5, 0.5],
            transmat: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            means: vec![vec![0.0, 0.0], vec![1.0, 1.0]],
            covars: vec![vec![1.0, 1.0], vec![1.0, 1.0]],
            feature_means: vec![0.5, 0.5],
            feature_stds: vec![0.5, 0.5],
        };
        
        let predictor = RegimePredictor { config };
        let features = vec![1.0, 1.0];
        let normalized = predictor.normalize_features(&features);
        
        assert_eq!(normalized.len(), 2);
        assert!((normalized[0] - 1.0).abs() < 1e-6);
        assert!((normalized[1] - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_predict_state() {
        let config = RegimeConfig {
            n_components: 2,
            covariance_type: "diag".to_string(),
            startprob: vec![0.5, 0.5],
            transmat: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            means: vec![vec![0.0, 0.0], vec![1.0, 1.0]],
            covars: vec![vec![1.0, 1.0], vec![1.0, 1.0]],
            feature_means: vec![0.0, 0.0],
            feature_stds: vec![1.0, 1.0],
        };
        
        let predictor = RegimePredictor { config };
        
        // Признаки близкие к состоянию 0
        let features = vec![0.0, 0.0];
        let state = predictor.predict_state(&features).unwrap();
        assert_eq!(state, 0);
        
        // Признаки близкие к состоянию 1
        let features = vec![1.0, 1.0];
        let state = predictor.predict_state(&features).unwrap();
        assert_eq!(state, 1);
    }
}
