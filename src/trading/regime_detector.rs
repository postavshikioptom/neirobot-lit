use crate::config::types::RegimeId;
use crate::data::orderbook::OrderBook;
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::Path;
use tracing::{info, debug, warn};

/// Конфигурация режимов рынка (загружается из regime_config.json)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RegimeConfig {
    /// Центроиды для каждого режима: [intensity, volatility, spread_z, ofi]
    pub centroids: HashMap<String, Vec<f64>>,
    /// Размер окна для расчета признаков (по умолчанию 100)
    #[serde(default = "default_window")]
    pub window: usize,
}

fn default_window() -> usize { 100 }

/// Детектор режимов рынка с механизмом гистерезиса
pub struct RegimeDetector {
    config: RegimeConfig,
    current_regime: RegimeId,
    candidate_regime: Option<RegimeId>,
    confirmation_count: usize,
    confirmation_threshold: usize,
    
    // Буферы для онлайн-расчета признаков
    mid_history: VecDeque<f64>,
    spread_history: VecDeque<f64>,
    bid_price_history: VecDeque<f64>,
    bid_vol_history: VecDeque<f64>,
    ask_price_history: VecDeque<f64>,
    ask_vol_history: VecDeque<f64>,
    timestamp_history: VecDeque<u64>,
}

impl RegimeDetector {
    /// Создает новый детектор и загружает конфигурацию из файла
    pub fn new<P: AsRef<Path>>(config_path: P, confirmation_threshold: usize) -> Result<Self> {
        let config_str = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read regime config from {:?}", config_path.as_ref()))?;
        
        let config: RegimeConfig = serde_json::from_str(&config_str)
            .with_context(|| "Failed to parse regime_config.json")?;
        
        info!("Loaded regime config with {} centroids, window={}", config.centroids.len(), config.window);
        
        Ok(Self {
            config,
            current_regime: RegimeId::Unknown,
            candidate_regime: None,
            confirmation_count: 0,
            confirmation_threshold,
            mid_history: VecDeque::new(),
            spread_history: VecDeque::new(),
            bid_price_history: VecDeque::new(),
            bid_vol_history: VecDeque::new(),
            ask_price_history: VecDeque::new(),
            ask_vol_history: VecDeque::new(),
            timestamp_history: VecDeque::new(),
        })
    }
    
    /// Обновляет буферы данными из нового снапшота
    pub fn update(&mut self, orderbook: &OrderBook, timestamp_ms: u64) {
        let window = self.config.window;
        
        // Получаем данные из стакана
        let (best_bid, bid_vol, best_ask, ask_vol) = orderbook.get_best_bid_ask_with_vol();
        
        if best_bid == 0.0 || best_ask == 0.0 {
            warn!("Invalid orderbook data: bid={}, ask={}", best_bid, best_ask);
            return;
        }
        
        let mid = (best_bid + best_ask) / 2.0;
        let spread = best_ask - best_bid;
        
        // Добавляем в буферы
        self.mid_history.push_back(mid);
        self.spread_history.push_back(spread);
        self.bid_price_history.push_back(best_bid);
        self.bid_vol_history.push_back(bid_vol);
        self.ask_price_history.push_back(best_ask);
        self.ask_vol_history.push_back(ask_vol);
        self.timestamp_history.push_back(timestamp_ms);
        
        // Ограничиваем размер буферов
        if self.mid_history.len() > window + 1 {
            self.mid_history.pop_front();
            self.spread_history.pop_front();
            self.bid_price_history.pop_front();
            self.bid_vol_history.pop_front();
            self.ask_price_history.pop_front();
            self.ask_vol_history.pop_front();
            self.timestamp_history.pop_front();
        }
    }
    
    /// Вычисляет признаки для текущего состояния рынка
    /// Возвращает: [intensity, volatility, spread_z, ofi]
    fn compute_features(&self) -> Option<[f64; 4]> {
        let _window = self.config.window;
        
        // Нужно минимум данных для расчета
        if self.mid_history.len() < 2 {
            return None;
        }
        
        // 1. Intensity: количество обновлений в окне (нормализованное)
        let intensity = self.mid_history.len() as f64;
        
        // 2. Volatility: log(std(mid_price))
        let volatility = self.compute_volatility();
        
        // 3. Spread Z-Score: (spread - mean) / std
        let spread_z = self.compute_spread_zscore();
        
        // 4. OFI: Order Flow Imbalance
        let ofi = self.compute_ofi();
        
        Some([intensity, volatility, spread_z, ofi])
    }
    
    /// Расчет волатильности: log(std(mid_price))
    fn compute_volatility(&self) -> f64 {
        if self.mid_history.len() < 2 {
            return 0.0;
        }
        
        let n = self.mid_history.len() as f64;
        let sum: f64 = self.mid_history.iter().sum();
        let mean = sum / n;
        
        let variance: f64 = self.mid_history.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / n;
        
        let std_dev = variance.sqrt();
        
        // log(std + epsilon) для стабилизации
        (std_dev + 1e-8).ln()
    }
    
    /// Расчет Z-score спреда
    fn compute_spread_zscore(&self) -> f64 {
        if self.spread_history.len() < 2 {
            return 0.0;
        }
        
        let n = self.spread_history.len() as f64;
        let sum: f64 = self.spread_history.iter().sum();
        let mean = sum / n;
        
        let variance: f64 = self.spread_history.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / n;
        
        let std_dev = variance.sqrt();
        
        // Текущий спред
        let current_spread = *self.spread_history.back().unwrap_or(&0.0);
        
        // Z-score = (current - mean) / std
        if std_dev > 1e-8 {
            (current_spread - mean) / std_dev
        } else {
            0.0
        }
    }
    
    /// Расчет Order Flow Imbalance (OFI)
    /// Реализация согласно python_lab/src/dataset.py
    fn compute_ofi(&self) -> f64 {
        if self.bid_price_history.len() < 2 {
            return 0.0;
        }
        
        let mut ofi_sum = 0.0;
        let n = self.bid_price_history.len();
        
        for i in 1..n {
            let bid_price_prev = self.bid_price_history[i-1];
            let bid_price_curr = self.bid_price_history[i];
            let bid_vol_prev = self.bid_vol_history[i-1];
            let bid_vol_curr = self.bid_vol_history[i];
            
            let ask_price_prev = self.ask_price_history[i-1];
            let ask_price_curr = self.ask_price_history[i];
            let ask_vol_prev = self.ask_vol_history[i-1];
            let ask_vol_curr = self.ask_vol_history[i];
            
            // Delta bid
            let delta_bid = if bid_price_curr >= bid_price_prev {
                if (bid_price_curr - bid_price_prev).abs() < 1e-8 {
                    bid_vol_curr - bid_vol_prev
                } else {
                    bid_vol_curr
                }
            } else {
                -bid_vol_prev
            };
            
            // Delta ask
            let delta_ask = if ask_price_curr <= ask_price_prev {
                if (ask_price_curr - ask_price_prev).abs() < 1e-8 {
                    ask_vol_curr - ask_vol_prev
                } else {
                    ask_vol_curr
                }
            } else {
                -ask_vol_prev
            };
            
            // OFI = delta_bid - delta_ask
            ofi_sum += delta_bid - delta_ask;
        }
        
        ofi_sum
    }
    
    /// Классифицирует текущий режим рынка с гистерезисом
    /// Возвращает текущий подтвержденный режим
    pub fn detect(&mut self) -> RegimeId {
        // Вычисляем признаки
        let features = match self.compute_features() {
            Some(f) => f,
            None => {
                debug!("Not enough data for regime detection");
                return self.current_regime;
            }
        };
        
        // Находим ближайший центроид (Euclidean distance)
        let mut min_distance = f64::MAX;
        let mut closest_regime = RegimeId::Unknown;
        
        for (regime_name, centroid) in &self.config.centroids {
            if centroid.len() != 4 {
                warn!("Invalid centroid for regime {}: expected 4 features, got {}", regime_name, centroid.len());
                continue;
            }
            
            let distance: f64 = features.iter()
                .zip(centroid.iter())
                .map(|(f, c)| (f - c).powi(2))
                .sum::<f64>()
                .sqrt();
            
            if distance < min_distance {
                min_distance = distance;
                closest_regime = match regime_name.as_str() {
                    "Quiet" => RegimeId::Quiet,
                    "Trend" => RegimeId::Trend,
                    "Volatile" => RegimeId::Volatile,
                    _ => RegimeId::Unknown,
                };
            }
        }
        
        debug!(
            "Regime detection: features={:?}, closest={:?}, distance={:.3}",
            features, closest_regime, min_distance
        );
        
        // Механизм гистерезиса
        if closest_regime == self.current_regime {
            // Режим не изменился - сбрасываем кандидата
            self.candidate_regime = None;
            self.confirmation_count = 0;
        } else {
            // Новый режим отличается от текущего
            if Some(closest_regime) == self.candidate_regime {
                // Кандидат подтверждается
                self.confirmation_count += 1;
                
                if self.confirmation_count >= self.confirmation_threshold {
                    // Достаточно подтверждений - переключаем режим
                    info!(
                        "Regime changed: {:?} -> {:?} (confirmed {} times)",
                        self.current_regime, closest_regime, self.confirmation_count
                    );
                    self.current_regime = closest_regime;
                    self.candidate_regime = None;
                    self.confirmation_count = 0;
                }
            } else {
                // Новый кандидат - начинаем отсчет
                self.candidate_regime = Some(closest_regime);
                self.confirmation_count = 1;
                debug!(
                    "New regime candidate: {:?} (1/{})",
                    closest_regime, self.confirmation_threshold
                );
            }
        }
        
        self.current_regime
    }
    
    /// Возвращает текущий подтвержденный режим
    pub fn current_regime(&self) -> RegimeId {
        self.current_regime
    }
}
