use crate::config::AdversarialConfig;
use crate::data::types::{OrderBookUpdateOwned, PublicTrade, Side};
use std::collections::VecDeque;
use smallvec::SmallVec;

/// Структура для отслеживания жизненного цикла уровня (для Spoofing detection)
#[derive(Debug, Clone)]
struct LevelSnapshot {
    price: f64,
    size: f64,
    timestamp_ms: u64,
    filled: f64, // Сколько было исполнено
}

/// Структура для хранения информации о корзине VPIN
#[derive(Debug, Clone)]
struct VPINBucket {
    buy_volume: f64,
    sell_volume: f64,
}

/// Детектор адверсариальной активности
pub struct AdversarialDetector {
    config: AdversarialConfig,
    
    // VPIN компоненты
    vpin_buckets: VecDeque<VPINBucket>,
    current_bucket: VPINBucket,
    current_bucket_volume: f64,
    last_trade_price: Option<f64>,
    
    // Layering компоненты - используем SmallVec для топ-10 уровней
    last_bid_levels: SmallVec<[f64; 10]>,
    last_ask_levels: SmallVec<[f64; 10]>,
    
    // Spoofing компоненты - используем BTreeMap для лучшей производительности
    level_history: std::collections::BTreeMap<String, LevelSnapshot>,
    avg_level_volume: f64,
    level_volume_count: usize,
}

impl AdversarialDetector {
    /// Создать новый детектор
    pub fn new(config: AdversarialConfig) -> Self {
        Self {
            config,
            vpin_buckets: VecDeque::with_capacity(config.vpin_buckets_count),
            current_bucket: VPINBucket {
                buy_volume: 0.0,
                sell_volume: 0.0,
            },
            current_bucket_volume: 0.0,
            last_trade_price: None,
            last_bid_levels: SmallVec::new(),
            last_ask_levels: SmallVec::new(),
            level_history: std::collections::BTreeMap::new(),
            avg_level_volume: 0.0,
            level_volume_count: 0,
        }
    }

    /// Классифицировать объем через Tick Test
    /// Возвращает Side::Buy если цена выше или равна последней, иначе Side::Sell
    fn classify_trade_side(&self, trade_price: f64) -> Side {
        match self.last_trade_price {
            None => Side::Buy, // По умолчанию Buy для первой сделки
            Some(last_price) => {
                if trade_price >= last_price {
                    Side::Buy
                } else {
                    Side::Sell
                }
            }
        }
    }

    /// Обновить VPIN с новой сделкой
    fn update_vpin(&mut self, trade: &PublicTrade) {
        let trade_price = trade.price.to_f64().unwrap_or(0.0);
        let trade_volume = trade.amount.to_f64().unwrap_or(0.0);

        // Классифицировать сделку
        let side = self.classify_trade_side(trade_price);

        // Добавить объем в текущую корзину
        match side {
            Side::Buy => self.current_bucket.buy_volume += trade_volume,
            Side::Sell => self.current_bucket.sell_volume += trade_volume,
        }

        self.current_bucket_volume += trade_volume;
        self.last_trade_price = Some(trade_price);

        // Если корзина заполнена, переместить в историю
        if self.current_bucket_volume >= self.config.vpin_volume_threshold {
            self.vpin_buckets
                .push_back(self.current_bucket.clone());

            // Удалить старые корзины, если превышен лимит
            if self.vpin_buckets.len() > self.config.vpin_buckets_count {
                self.vpin_buckets.pop_front();
            }

            // Сбросить текущую корзину
            self.current_bucket = VPINBucket {
                buy_volume: 0.0,
                sell_volume: 0.0,
            };
            self.current_bucket_volume = 0.0;
        }
    }

    /// Рассчитать VPIN с использованием SIMD-оптимизированного суммирования
    fn calculate_vpin(&self) -> f64 {
        if self.vpin_buckets.is_empty() {
            return 0.0;
        }

        // Используем итератор для эффективного суммирования
        let sum_imbalance: f64 = self.vpin_buckets
            .iter()
            .map(|bucket| (bucket.buy_volume - bucket.sell_volume).abs())
            .sum();

        let n = self.vpin_buckets.len() as f64;
        sum_imbalance / (n * self.config.vpin_volume_threshold)
    }

    /// Проверить Layering (слоистость)
    fn check_layering(&mut self, update: &OrderBookUpdateOwned) -> bool {
        // Собрать цены топ-10 уровней на покупку в SmallVec
        let mut bid_prices: SmallVec<[f64; 10]> = SmallVec::new();
        for level in update.bids.iter().take(10) {
            bid_prices.push(level.price);
        }

        // Собрать цены топ-10 уровней на продажу в SmallVec
        let mut ask_prices: SmallVec<[f64; 10]> = SmallVec::new();
        for level in update.asks.iter().take(10) {
            ask_prices.push(level.price);
        }

        // Рассчитать ценовые зазоры для bid
        let bid_deltas = self.calculate_price_deltas(&bid_prices);
        let ask_deltas = self.calculate_price_deltas(&ask_prices);

        // Рассчитать std_dev
        let bid_std = self.calculate_std_dev(&bid_deltas);
        let ask_std = self.calculate_std_dev(&ask_deltas);

        // Сохранить для следующей итерации
        self.last_bid_levels = bid_prices;
        self.last_ask_levels = ask_prices;

        // Если std_dev меньше порога, это признак Layering
        let is_layering = bid_std < self.config.layering_std_threshold
            || ask_std < self.config.layering_std_threshold;

        is_layering
    }

    /// Рассчитать ценовые зазоры между уровнями
    fn calculate_price_deltas(&self, prices: &[f64]) -> SmallVec<[f64; 10]> {
        let mut deltas: SmallVec<[f64; 10]> = SmallVec::new();
        for i in 1..prices.len() {
            deltas.push((prices[i] - prices[i - 1]).abs());
        }
        deltas
    }

    /// Рассчитать стандартное отклонение
    fn calculate_std_dev(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / values.len() as f64;

        variance.sqrt()
    }

    /// Проверить Spoofing (спуфинг) с инкрементальными обновлениями
    fn check_spoofing(&mut self, update: &OrderBookUpdateOwned) -> bool {
        // Инкрементальное обновление среднего объема
        let all_levels = update.bids.len() + update.asks.len();
        if all_levels > 0 {
            let total_volume: f64 = update
                .bids
                .iter()
                .chain(update.asks.iter())
                .map(|level| level.size)
                .sum();

            let avg_volume = total_volume / all_levels as f64;

            // Инкрементальное обновление скользящего среднего (Welford's algorithm)
            self.level_volume_count += 1;
            let delta = avg_volume - self.avg_level_volume;
            self.avg_level_volume += delta / self.level_volume_count as f64;
        }

        // Проверить текущие уровни на аномально крупные ордеры
        let mut spoofing_detected = false;

        // Проверить bid уровни
        for level in &update.bids {
            let key = format!("bid_{}", level.price);
            if level.size > self.avg_level_volume * self.config.spoofing_min_vol_multiple {
                // Это аномально крупный ордер
                self.level_history.insert(
                    key,
                    LevelSnapshot {
                        price: level.price,
                        size: level.size,
                        timestamp_ms: update.timestamp_ms,
                        filled: 0.0,
                    },
                );
            }
        }

        // Проверить ask уровни
        for level in &update.asks {
            let key = format!("ask_{}", level.price);
            if level.size > self.avg_level_volume * self.config.spoofing_min_vol_multiple {
                // Это аномально крупный ордер
                self.level_history.insert(
                    key,
                    LevelSnapshot {
                        price: level.price,
                        size: level.size,
                        timestamp_ms: update.timestamp_ms,
                        filled: 0.0,
                    },
                );
            }
        }

        // Проверить, исчезли ли аномально крупные ордеры без исполнения
        let current_bid_prices: std::collections::HashSet<_> =
            update.bids.iter().map(|l| l.price).collect();
        let current_ask_prices: std::collections::HashSet<_> =
            update.asks.iter().map(|l| l.price).collect();

        let mut to_remove = Vec::new();
        for (key, snapshot) in self.level_history.iter() {
            let is_bid = key.starts_with("bid_");
            let current_prices = if is_bid {
                &current_bid_prices
            } else {
                &current_ask_prices
            };

            // Если уровень исчез и не было исполнения, это спуфинг
            if !current_prices.contains(&snapshot.price) && snapshot.filled == 0.0 {
                spoofing_detected = true;
                to_remove.push(key.clone());
            }
        }

        // Удалить обнаруженные спуфинги
        for key in to_remove {
            self.level_history.remove(&key);
        }

        spoofing_detected
    }

    /// Основной метод: обновить состояние и проверить на токсичность
    pub fn update_and_check(
        &mut self,
        orderbook: &OrderBookUpdateOwned,
        trades: &[PublicTrade],
    ) -> bool {
        // Обновить VPIN с новыми сделками
        for trade in trades {
            self.update_vpin(trade);
        }

        // Рассчитать VPIN
        let vpin = self.calculate_vpin();

        // Проверить Layering
        let is_layering = self.check_layering(orderbook);

        // Проверить Spoofing
        let is_spoofing = self.check_spoofing(orderbook);

        // Поток считается токсичным, если:
        // 1. VPIN высокий (> 0.5 - признак информированной торговли)
        // 2. Обнаружен Layering
        // 3. Обнаружен Spoofing
        let is_toxic = vpin > 0.5 || is_layering || is_spoofing;

        if is_toxic {
            log::warn!(
                "[Adversarial] Toxic flow detected: VPIN={:.4}, Layering={}, Spoofing={}",
                vpin,
                is_layering,
                is_spoofing
            );
        }

        is_toxic
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal::Decimal;

    #[test]
    fn test_detector_creation() {
        let config = AdversarialConfig::default();
        let detector = AdversarialDetector::new(config);
        assert_eq!(detector.vpin_buckets.len(), 0);
    }

    #[test]
    fn test_tick_test_classification() {
        let config = AdversarialConfig::default();
        let detector = AdversarialDetector::new(config);

        // Первая сделка - по умолчанию Buy
        assert_eq!(detector.classify_trade_side(100.0), Side::Buy);
    }

    #[test]
    fn test_price_deltas_calculation() {
        let config = AdversarialConfig::default();
        let detector = AdversarialDetector::new(config);

        let prices = vec![100.0, 100.5, 101.0, 101.5];
        let deltas = detector.calculate_price_deltas(&prices);

        assert_eq!(deltas.len(), 3);
        assert!((deltas[0] - 0.5).abs() < 0.001);
        assert!((deltas[1] - 0.5).abs() < 0.001);
        assert!((deltas[2] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_std_dev_calculation() {
        let config = AdversarialConfig::default();
        let detector = AdversarialDetector::new(config);

        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let std_dev = detector.calculate_std_dev(&values);

        // Стандартное отклонение для [1,2,3,4,5] должно быть ~1.414
        assert!((std_dev - 1.414).abs() < 0.01);
    }
}
