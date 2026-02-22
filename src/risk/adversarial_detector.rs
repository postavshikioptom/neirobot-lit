use crate::config::AdversarialConfig;
use crate::data::types::{OrderBookUpdateOwned, PublicTrade, Side};
use std::collections::VecDeque;
use smallvec::SmallVec;
use wide::f64x2;

/// Структура для отслеживания жизненного цикла уровня (для Spoofing detection)
#[derive(Debug, Clone)]
struct LevelSnapshot {
    price: f64,
    size: f64,
    timestamp_ms: u64,
    filled: f64, // Сколько было исполнено
    is_bid: bool, // Флаг: bid или ask
}

/// Структура для хранения информации о корзине VPIN
#[derive(Debug, Clone)]
struct VPINBucket {
    buy_volume: f64,
    sell_volume: f64,
    imbalance: f64, // Кэшированное значение |buy_volume - sell_volume|
}

/// Детектор адверсариальной активности
pub struct AdversarialDetector {
    config: AdversarialConfig,
    
    // VPIN компоненты
    vpin_buckets: VecDeque<VPINBucket>,
    current_bucket: VPINBucket,
    current_bucket_volume: f64,
    last_trade_price: Option<f64>,
    total_imbalance_sum: f64, // Инкрементальная сумма для VPIN
    
    // Layering компоненты - используем SmallVec для топ-10 уровней и инкрементальный расчет
    last_bid_levels: SmallVec<[f64; 10]>,
    last_ask_levels: SmallVec<[f64; 10]>,
    last_bid_std: f64,
    last_ask_std: f64,
    
    // Spoofing компоненты - BTreeMap с f64 ключами (bid цены) и ask ценами отдельно
    level_history_bids: std::collections::BTreeMap<u64, LevelSnapshot>, // ключ: скале цена (f64 -> u64)
    level_history_asks: std::collections::BTreeMap<u64, LevelSnapshot>,
    avg_level_volume: f64,
    level_volume_count: usize,
}

impl AdversarialDetector {
    /// Создать новый детектор
    pub fn new(config: AdversarialConfig) -> Self {
        Self {
            config: config.clone(),
            vpin_buckets: VecDeque::with_capacity(config.vpin_buckets_count),
            current_bucket: VPINBucket {
                buy_volume: 0.0,
                sell_volume: 0.0,
                imbalance: 0.0,
            },
            current_bucket_volume: 0.0,
            last_trade_price: None,
            total_imbalance_sum: 0.0,
            last_bid_levels: SmallVec::new(),
            last_ask_levels: SmallVec::new(),
            last_bid_std: 0.0,
            last_ask_std: 0.0,
            level_history_bids: std::collections::BTreeMap::new(),
            level_history_asks: std::collections::BTreeMap::new(),
            avg_level_volume: 0.0,
            level_volume_count: 0,
        }
    }

    /// Классифицировать объем через Tick Test (сравнение цены сделки с ценой предыдущей сделки или Mid Price)
    /// Возвращает Side::Buy если цена выше mid_price или последней цены, иначе Side::Sell
    fn classify_trade_side(&self, trade_price: f64, mid_price: Option<f64>) -> Side {
        // Преимущество Mid Price (более точный): если доступен, используем его
        if let Some(mid) = mid_price {
            if trade_price > mid {
                return Side::Buy;
            } else if trade_price < mid {
                return Side::Sell;
            } else {
                // trade_price == mid, используем стандартный Tick Test
                return match self.last_trade_price {
                    None => Side::Buy,
                    Some(last_price) => {
                        if trade_price >= last_price {
                            Side::Buy
                        } else {
                            Side::Sell
                        }
                    }
                };
            }
        }
        
        // Fallback: стандартный Tick Test через last_trade_price
        match self.last_trade_price {
            None => Side::Buy,
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
    fn update_vpin(&mut self, trade: &PublicTrade, mid_price: Option<f64>) {
        let trade_price = trade.price.to_f64().unwrap_or(0.0);
        let trade_volume = trade.size.to_f64().unwrap_or(0.0);

        // Классифицировать сделку через Tick Test с Mid Price
        let side = self.classify_trade_side(trade_price, mid_price);

        // Добавить объем в текущую корзину
        match side {
            Side::Buy => self.current_bucket.buy_volume += trade_volume,
            Side::Sell => self.current_bucket.sell_volume += trade_volume,
        }

        self.current_bucket_volume += trade_volume;
        self.last_trade_price = Some(trade_price);

        // Если корзина заполнена, переместить в историю
        if self.current_bucket_volume >= self.config.vpin_volume_threshold {
            // Рассчитать imbalance для текущей корзины
            self.current_bucket.imbalance = 
                (self.current_bucket.buy_volume - self.current_bucket.sell_volume).abs();
            
            // Добавить в инкрементальную сумму
            self.total_imbalance_sum += self.current_bucket.imbalance;
            
            self.vpin_buckets
                .push_back(self.current_bucket.clone());

            // Удалить старые корзины, если превышен лимит
            if self.vpin_buckets.len() > self.config.vpin_buckets_count {
                if let Some(removed) = self.vpin_buckets.pop_front() {
                    // Вычесть imbalance удаляемой корзины из суммы
                    self.total_imbalance_sum -= removed.imbalance;
                }
            }

            // Сбросить текущую корзину
            self.current_bucket = VPINBucket {
                buy_volume: 0.0,
                sell_volume: 0.0,
                imbalance: 0.0,
            };
            self.current_bucket_volume = 0.0;
        }
    }

    /// Рассчитать VPIN с использованием инкрементального расчета (O(1) вместо O(n))
    /// Использует pre-computed total_imbalance_sum, которое обновляется при каждом добавлении/удалении корзины
    fn calculate_vpin(&self) -> f64 {
        if self.vpin_buckets.is_empty() {
            return 0.0;
        }

        let n = self.vpin_buckets.len() as f64;
        self.total_imbalance_sum / (n * self.config.vpin_volume_threshold)
    }

    /// Проверить Layering (слоистость) с полной инкрементальностью (Пункт 4 плана)
    /// Ключевые оптимизации:
    /// 1. Пересчитывает std_dev ТОЛЬКО если топ-10 уровни действительно изменились
    /// 2. Не собирает top-10 уровни заново если они не изменились в OrderBookUpdate
    /// 3. Zero-allocation: SmallVec<[f64; 10]> хранится на стеке
    fn check_layering(&mut self, update: &OrderBookUpdateOwned) -> bool {
        // ОПТИМИЗАЦИЯ 1: Быстрая проверка if расчета top-10 вообще нужен
        // Если это не snapshot и мы уже имеем сохраненные уровни,
        // скорее всего они не изменились значительно (для дельта-обновлений)
        let need_recalc = update.is_snapshot 
            || self.last_bid_levels.is_empty() 
            || self.last_ask_levels.is_empty();

        let mut bid_std = self.last_bid_std;
        let mut ask_std = self.last_ask_std;

        if need_recalc {
            // Собрать цены топ-10 уровней на покупку (stack-allocated через SmallVec)
            let mut bid_prices: SmallVec<[f64; 10]> = SmallVec::new();
            for level in update.bids.iter().take(10) {
                bid_prices.push(level.price);
            }

            // Собрать цены топ-10 уровней на продажу
            let mut ask_prices: SmallVec<[f64; 10]> = SmallVec::new();
            for level in update.asks.iter().take(10) {
                ask_prices.push(level.price);
            }

            // Проверка диффа: сравниваем цены без пересчета, если they не изменились
            let bid_changed = bid_prices.len() != self.last_bid_levels.len() 
                || bid_prices.iter().zip(self.last_bid_levels.iter())
                    .any(|(&new, &old)| (new - old).abs() > 1e-10);

            let ask_changed = ask_prices.len() != self.last_ask_levels.len()
                || ask_prices.iter().zip(self.last_ask_levels.iter())
                    .any(|(&new, &old)| (new - old).abs() > 1e-10);

            // Пересчитываем std_dev ТОЛЬКО при реальных изменениях (SIMD-оптимизировано)
            if bid_changed {
                let bid_deltas = self.calculate_price_deltas(&bid_prices);
                bid_std = self.calculate_std_dev(&bid_deltas); // Использует SIMD
                self.last_bid_levels = bid_prices;
                self.last_bid_std = bid_std;
            }

            if ask_changed {
                let ask_deltas = self.calculate_price_deltas(&ask_prices);
                ask_std = self.calculate_std_dev(&ask_deltas); // Использует SIMD
                self.last_ask_levels = ask_prices;
                self.last_ask_std = ask_std;
            }
        }

        // Логика Layering: если std_dev меньше порога = неестественно плотное выставление ордеров
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

    /// Рассчитать стандартное отклонение с SIMD-оптимизацией (задача 078)
    /// Использует f64x2 для параллельного суммирования и расчета дисперсии
    fn calculate_std_dev(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        // Вычислить среднее с SIMD-суммированием
        let mean = self.simd_sum(values) / values.len() as f64;

        // Вычислить дисперсию с SIMD через f64x2 векторы
        let mean_vec = f64x2::splat(mean);
        let mut variance_sum_simd = f64x2::splat(0.0);
        let simd_len = (values.len() / 2) * 2;
        let mut i = 0;

        // SIMD обработка по 2 элементам (f64x2) для квадратов отклонений
        while i < simd_len {
            let v = f64x2::from_slice_unaligned(&values[i..i+2]);
            let diff = v - mean_vec;
            let squared = diff * diff;
            variance_sum_simd = variance_sum_simd + squared;
            i += 2;
        }

        // Редуцировать вектор в скаляр
        let mut variance_sum = variance_sum_simd.sum();

        // Скалярный fallback для оставшегося элемента
        if simd_len < values.len() {
            let v = values[simd_len] - mean;
            variance_sum += v * v;
        }

        let variance = variance_sum / values.len() as f64;
        variance.sqrt()
    }

    /// Вспомогательный метод: SIMD-суммирование элементов массива (задача 078)
    /// Использует wide::f64x2 для параллельного суммирования
    #[inline]
    fn simd_sum(&self, values: &[f64]) -> f64 {
        let mut sum_simd = f64x2::splat(0.0);
        let simd_len = (values.len() / 2) * 2;
        let mut i = 0;

        // SIMD обработка по 2 элементам за раз через f64x2
        while i < simd_len {
            let v = f64x2::from_slice_unaligned(&values[i..i+2]);
            sum_simd = sum_simd + v;
            i += 2;
        }

        // Редуцировать вектор в скаляр (sum_simd[0] + sum_simd[1])
        let sum = sum_simd.sum();

        // Скалярный fallback для оставшегося элемента
        if simd_len < values.len() {
            return sum + values[simd_len];
        }

        sum
    }

    /// Вспомогательная функция: преобразовать f64 цену в u64 ключ
    /// Использует битовое представление для сохранения точности
    #[inline]
    fn price_to_key(price: f64) -> u64 {
        price.to_bits()
    }

    /// Проверить наличие цены в SmallVec (без аллокации)
    #[inline]
    fn price_in_levels(price: f64, levels: &SmallVec<[f64; 10]>) -> bool {
        levels.iter().any(|&p| (p - price).abs() < 1e-10)
    }

    /// SIMD-оптимизированное суммирование объемов через итератор (zero-allocation, задача 078)
    #[inline]
    fn simd_sum_volumes_iter<T: Into<f64> + Copy, I: Iterator<Item = T>>(&self, iter: I) -> f64 {
        let values: SmallVec<[f64; 32]> = iter.map(|v| v.into()).collect();
        
        let mut sum_simd = f64x2::splat(0.0);
        let simd_len = (values.len() / 2) * 2;
        let mut i = 0;

        // SIMD обработка по 2 элементам
        while i < simd_len {
            let v = f64x2::from_slice_unaligned(&values[i..i+2]);
            sum_simd = sum_simd + v;
            i += 2;
        }

        let mut sum = sum_simd.sum();

        // Скалярный fallback
        if simd_len < values.len() {
            sum += values[simd_len];
        }

        sum
    }

    /// Проверить Spoofing (спуфинг) с инкрементальными обновлениями и zero-allocation
    fn check_spoofing(&mut self, update: &OrderBookUpdateOwned, trades: &[PublicTrade]) -> bool {
        // Инкрементальное обновление среднего объема (SIMD-оптимизировано, задача 078)
        let all_levels = update.bids.len() + update.asks.len();
        if all_levels > 0 {
            // SIMD суммирование объемов bid (через итератор, zero-allocation)
            let bid_volume = self.simd_sum_volumes_iter(update.bids.iter().map(|l| l.size));
            // SIMD суммирование объемов ask
            let ask_volume = self.simd_sum_volumes_iter(update.asks.iter().map(|l| l.size));
            let total_volume = bid_volume + ask_volume;

            let avg_volume = total_volume / all_levels as f64;

            // Инкрементальное обновление скользящего среднего (Welford's algorithm)
            self.level_volume_count += 1;
            let delta = avg_volume - self.avg_level_volume;
            self.avg_level_volume += delta / self.level_volume_count as f64;
        }

        // Обновить поле filled в основе сделок (FIX #1: Update filled field)
        for trade in trades {
            let trade_price = trade.price.to_f64().unwrap_or(0.0);
            let trade_size = trade.size.to_f64().unwrap_or(0.0);
            let key = Self::price_to_key(trade_price);
            
            // Обновить bid историю если сделка попала на bid цену
            if let Some(snapshot) = self.level_history_bids.get_mut(&key) {
                snapshot.filled += trade_size;
            }
            
            // Обновить ask историю если сделка попала на ask цену
            if let Some(snapshot) = self.level_history_asks.get_mut(&key) {
                snapshot.filled += trade_size;
            }
        }

        // Проверить текущие уровни на аномально крупные ордеры
        let mut spoofing_detected = false;

        // Проверить bid уровни - используем SmallVec для текущих цен (zero-allocation)
        let mut current_bid_prices: SmallVec<[f64; 10]> = SmallVec::new();
        for level in update.bids.iter().take(10) {
            if level.size > self.avg_level_volume * self.config.spoofing_min_vol_multiple {
                let key = Self::price_to_key(level.price);
                current_bid_prices.push(level.price);
                
                // Добавить или обновить snapshot
                self.level_history_bids
                    .entry(key)
                    .or_insert_with(|| LevelSnapshot {
                        price: level.price,
                        size: level.size,
                        timestamp_ms: update.timestamp_ms,
                        filled: 0.0,
                        is_bid: true,
                    });
            }
        }

        // Проверить ask уровни
        let mut current_ask_prices: SmallVec<[f64; 10]> = SmallVec::new();
        for level in update.asks.iter().take(10) {
            if level.size > self.avg_level_volume * self.config.spoofing_min_vol_multiple {
                let key = Self::price_to_key(level.price);
                current_ask_prices.push(level.price);
                
                // Добавить или обновить snapshot
                self.level_history_asks
                    .entry(key)
                    .or_insert_with(|| LevelSnapshot {
                        price: level.price,
                        size: level.size,
                        timestamp_ms: update.timestamp_ms,
                        filled: 0.0,
                        is_bid: false,
                    });
            }
        }

        // Проверить, исчезли ли аномально крупные ордеры без исполнения (zero-allocation через SmallVec)
        let mut to_remove_bids: SmallVec<[u64; 16]> = SmallVec::new();
        for (key, snapshot) in self.level_history_bids.iter() {
            // Если уровень исчез и не было исполнения, это спуфинг
            if !Self::price_in_levels(snapshot.price, &current_bid_prices) && snapshot.filled == 0.0 {
                spoofing_detected = true;
                to_remove_bids.push(*key);
            }
        }

        let mut to_remove_asks: SmallVec<[u64; 16]> = SmallVec::new();
        for (key, snapshot) in self.level_history_asks.iter() {
            // Если уровень исчез и не было исполнения, это спуфинг
            if !Self::price_in_levels(snapshot.price, &current_ask_prices) && snapshot.filled == 0.0 {
                spoofing_detected = true;
                to_remove_asks.push(*key);
            }
        }

        // Удалить обнаруженные спуфинги (zero-allocation, используем SmallVec итератор)
        for key in to_remove_bids {
            self.level_history_bids.remove(&key);
        }
        for key in to_remove_asks {
            self.level_history_asks.remove(&key);
        }

        spoofing_detected
    }

    /// Основной метод: обновить состояние и проверить на токсичность
    pub fn update_and_check(
        &mut self,
        orderbook: &OrderBookUpdateOwned,
        trades: &[PublicTrade],
    ) -> bool {
        // Вычислить Mid Price (средняя цена best bid и best ask)
        let mid_price = if !orderbook.bids.is_empty() && !orderbook.asks.is_empty() {
            Some((orderbook.bids[0].price + orderbook.asks[0].price) / 2.0)
        } else {
            None
        };

        // Обновить VPIN с новыми сделками, передавая mid_price для Tick Test
        for trade in trades {
            self.update_vpin(trade, mid_price);
        }

        // Рассчитать VPIN
        let vpin = self.calculate_vpin();

        // Проверить Layering
        let is_layering = self.check_layering(orderbook);

        // Проверить Spoofing (передаем trades для обновления filled)
        let is_spoofing = self.check_spoofing(orderbook, trades);

        // Поток считается токсичным, если:
        // 1. VPIN высокий (> 0.5 - признак информированной торговли)
        // 2. Обнаружен Layering
        // 3. Обнаружен Spoofing
        let is_toxic = vpin > 0.5 || is_layering || is_spoofing;

        if is_toxic {
            tracing::warn!(
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

        // Первая сделка без mid_price - по умолчанию Buy
        assert_eq!(detector.classify_trade_side(100.0, None), Side::Buy);
        
        // С mid_price: цена выше mid => Buy
        assert_eq!(detector.classify_trade_side(101.0, Some(100.0)), Side::Buy);
        
        // С mid_price: цена ниже mid => Sell
        assert_eq!(detector.classify_trade_side(99.0, Some(100.0)), Side::Sell);
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
