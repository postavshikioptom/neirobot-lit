use rust_decimal::Decimal;
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};
use smallvec::SmallVec;
use std::fmt::Write;
use std::time::Instant;
use std::sync::Arc;
use arc_swap::ArcSwap;
use tracing::warn;
use crate::data::types::OrderBookUpdateArc;
use crate::monitoring::latency::PROC_LATENCY;
use serde::{Serialize, Deserialize};
use circular_buffer::CircularBuffer;

pub const LOB_DEPTH: usize = 50;

/// Снимок стакана для периодического дампа (задача 132)
/// Соответствует глобальной схеме данных (задача 012): 50 уровней
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookSnapshot {
    pub timestamp_ms: i64,
    pub last_update_id: u64,
    pub symbol: String,
    pub bids: Vec<(f64, f64)>, // (price, volume)
    pub asks: Vec<(f64, f64)>, // (price, volume)
    pub checksum: u32, // Задача 191: контрольная сумма для lock-free доступа
    pub mark_price: f64, // Задача 233: маркированная цена для проверки отклонения
    pub volatility_bps: f64, // Задача 191: волатильность в базисных пунктах
    pub spread_bps: f64, // Задача 191: спред в базисных пунктах
}

impl OrderBookSnapshot {
    /// Задача 191: Методы для lock-free доступа к данным снапшота
    
    #[inline(always)]
    pub fn get_mid_price_dec(&self) -> Decimal {
        let best_bid = self.bids.first().map(|(p, _)| Decimal::from_f64(*p).unwrap_or_default());
        let best_ask = self.asks.first().map(|(p, _)| Decimal::from_f64(*p).unwrap_or_default());

        match (best_bid, best_ask) {
            (Some(bid), Some(ask)) => (bid + ask) / Decimal::from(2),
            _ => Decimal::ZERO,
        }
    }

    #[inline(always)]
    pub fn get_mid_price(&self) -> f64 {
        let best_bid = self.bids.first().map(|(p, _)| *p);
        let best_ask = self.asks.first().map(|(p, _)| *p);

        match (best_bid, best_ask) {
            (Some(bid), Some(ask)) => (bid + ask) / 2.0,
            _ => 0.0,
        }
    }

    #[inline(always)]
    pub fn get_best_bid_ask(&self) -> (f64, f64) {
        let best_bid = self.bids.first().map(|(p, _)| *p).unwrap_or(0.0);
        let best_ask = self.asks.first().map(|(p, _)| *p).unwrap_or(0.0);
        (best_bid, best_ask)
    }

    #[inline(always)]
    pub fn get_best_bid_ask_with_vol(&self) -> (f64, f64, f64, f64) {
        let (best_bid, bid_vol) = self.bids.first().map(|(p, v)| (*p, *v)).unwrap_or((0.0, 0.0));
        let (best_ask, ask_vol) = self.asks.first().map(|(p, v)| (*p, *v)).unwrap_or((0.0, 0.0));
        (best_bid, bid_vol, best_ask, ask_vol)
    }

    /// Возвращает плоский снимок стакана для ML (200 значений)
    /// Формат: [asks(p,v) * 50, bids(p,v) * 50]
    pub fn get_flat_snapshot(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(LOB_DEPTH * 4);
        
        // Asks (50 уровней)
        for i in 0..LOB_DEPTH {
            if let Some((p, v)) = self.asks.get(i) {
                result.push(*p as f32);
                result.push(*v as f32);
            } else {
                result.push(0.0);
                result.push(0.0);
            }
        }
        
        // Bids (50 уровней)
        for i in 0..LOB_DEPTH {
            if let Some((p, v)) = self.bids.get(i) {
                result.push(*p as f32);
                result.push(*v as f32);
            } else {
                result.push(0.0);
                result.push(0.0);
            }
        }
        
        result
    }

    /// Расчет дисбаланса стакана (Order Book Imbalance)
    /// Возвращает значение в диапазоне [-1, 1]
    #[inline(always)]
    pub fn calculate_imbalance(&self, depth: usize) -> f64 {
        let d = depth.min(self.bids.len().min(self.asks.len()));
        if d == 0 {
            return 0.0;
        }

        let bid_vol: f64 = self.bids.iter().take(d).map(|(_, v)| v).sum();
        let ask_vol: f64 = self.asks.iter().take(d).map(|(_, v)| v).sum();
        
        let total_vol = bid_vol + ask_vol;
        if total_vol == 0.0 {
            return 0.0;
        }

        (bid_vol - ask_vol) / total_vol
    }

    /// Возвращает объем на лучшем уровне в USD
    pub fn get_volume_at_best(&self, side: crate::data::types::Side) -> f64 {
        use crate::data::types::Side;
        
        let levels = match side {
            Side::Buy => &self.asks,
            Side::Sell => &self.bids,
        };

        if let Some((price, volume)) = levels.first() {
            price * volume
        } else {
            0.0
        }
    }

    /// Рассчитывает VWAP для исполнения (упрощенная версия без кешей)
    pub fn get_execution_vwap(&self, side: crate::data::types::Side, size: f64) -> Option<f64> {
        use crate::data::types::Side;
        
        let levels = match side {
            Side::Buy => &self.asks,
            Side::Sell => &self.bids,
        };

        if levels.is_empty() || size <= 0.0 {
            return None;
        }

        let mut cum_vol = 0.0;
        let mut cum_pv = 0.0;

        for (price, volume) in levels.iter() {
            if cum_vol >= size {
                break;
            }
            
            let remaining = size - cum_vol;
            let vol_to_use = volume.min(remaining);
            
            cum_vol += vol_to_use;
            cum_pv += price * vol_to_use;
        }

        if cum_vol < size {
            return None; // Недостаточно ликвидности
        }

        Some(cum_pv / cum_vol)
    }

    /// Рассчитывает средневзвешенную цену исполнения (VWAP) и возвращает отклонение от best_price в базисных пунктах (bps).
    /// Используется для оценки проскальзывания при исполнении ордера заданного размера.
    #[inline(always)]
    pub fn calculate_vwap_impact(&self, side: crate::data::types::Side, target_size_quote: f64) -> f64 {
        use crate::data::types::Side;
        
        if target_size_quote <= 0.0 {
            return 0.0;
        }

        let best_price = match side {
            Side::Buy => self.asks.first().map(|(p, _)| *p).unwrap_or(0.0),
            Side::Sell => self.bids.first().map(|(p, _)| *p).unwrap_or(0.0),
        };

        if best_price <= 0.0 {
            return 0.0;
        }

        // Переведем объем из USD в базовую валюту через best_price для оценки
        let size_base = target_size_quote / best_price;
        
        match self.get_execution_vwap(side, size_base) {
            Some(vwap) => {
                ((vwap / best_price) - 1.0).abs() * 10000.0
            },
            None => 1000.0, // Высокое проскальзывание если ликвидности совсем нет
        }
    }

    /// Возвращает объем на указанном уровне asks
    #[inline(always)]
    pub fn get_ask_volume_at_level(&self, level: usize) -> f64 {
        self.asks.get(level).map(|(_, v)| *v).unwrap_or(0.0)
    }

    /// Возвращает объем на указанном уровне bids
    #[inline(always)]
    pub fn get_bid_volume_at_level(&self, level: usize) -> f64 {
        self.bids.get(level).map(|(_, v)| *v).unwrap_or(0.0)
    }
}

pub struct OrderBook {
    pub symbol: String,
    // Уровни (Price, Vol). Резервируем 64 элемента на стеке.
    bids: SmallVec<[(Decimal, Decimal); 64]>, // Descending (high to low)
    asks: SmallVec<[(Decimal, Decimal); 64]>, // Ascending (low to high)
    pub last_update_id: u64,
    pub timestamp_ms: u64,
    dirty: bool, // Флаг изменений для дампа
    // Задача 177: Цена последней сделки для проверки отклонения
    pub last_trade_price: Option<f64>,
    // Задача 233: Маркированная цена для проверки отклонения от Mark Price
    pub mark_price: f64,
    // Кеши для быстрого расчета VWAP (задача 168)
    cum_vol_bids: Vec<f64>,      // Кумулятивный объем для Bids
    cum_price_vol_bids: Vec<f64>, // Кумулятивная сумма (Price * Volume) для Bids
    cum_vol_asks: Vec<f64>,      // Кумулятивный объем для Asks
    cum_price_vol_asks: Vec<f64>, // Кумулятивная сумма (Price * Volume) для Asks
    // Задача 191: Lock-free доступ к снапшоту стакана
    pub current_snapshot: ArcSwap<OrderBookSnapshot>,
    // Задача 210: Адаптивные пороги отмены - алгоритм Велфорда для расчета волатильности
    price_buffer: CircularBuffer<500, f64>,  // Кольцевой буфер для mid_price
    volatility_count: usize,                  // Количество элементов для расчета
    volatility_mean: f64,                     // Текущее среднее (алгоритм Велфорда)
    volatility_m2: f64,                       // Сумма квадратов отклонений (для дисперсии)
}

impl OrderBook {
    pub fn new(symbol: &str) -> Self {
        let mut bids = SmallVec::new();
        bids.reserve(LOB_DEPTH * 2);
        let mut asks = SmallVec::new();
        asks.reserve(LOB_DEPTH * 2);
        
        // Инициализируем пустой снапшот для lock-free доступа (задача 191)
        let initial_snapshot = OrderBookSnapshot {
            timestamp_ms: 0,
            last_update_id: 0,
            symbol: symbol.to_string(),
            bids: Vec::new(),
            asks: Vec::new(),
            checksum: 0,
            mark_price: 0.0,
            spread_bps: 0.0,
            volatility_bps: 0.0,
        };
        
        Self {
            symbol: symbol.to_string(),
            bids,
            asks,
            last_update_id: 0,
            timestamp_ms: 0,
            dirty: false,
            last_trade_price: None,
            mark_price: 0.0,
            cum_vol_bids: Vec::with_capacity(LOB_DEPTH),
            cum_price_vol_bids: Vec::with_capacity(LOB_DEPTH),
            cum_vol_asks: Vec::with_capacity(LOB_DEPTH),
            cum_price_vol_asks: Vec::with_capacity(LOB_DEPTH),
            current_snapshot: ArcSwap::from_pointee(initial_snapshot),
            // Задача 210: Инициализация полей для расчета волатильности
            price_buffer: CircularBuffer::new(),
            volatility_count: 0,
            volatility_mean: 0.0,
            volatility_m2: 0.0,
        }
    }

    /// Создает снимок стакана для дампа (задача 132)
    /// Использует f64 для максимальной производительности при записи
    /// Соответствует глобальной схеме данных (задача 012): 50 уровней
    pub fn take_snapshot(&self) -> OrderBookSnapshot {
        OrderBookSnapshot {
            // Задача 132: Используем системное время через helpers::unix_ms() для минимизации оверхеда в асинхронном цикле
            timestamp_ms: crate::utils::helpers::unix_ms() as i64,
            last_update_id: self.last_update_id,
            symbol: self.symbol.clone(),
            // Копируем все 50 уровней согласно глобальной схеме (задача 012)
            bids: self.bids.iter()
                .take(LOB_DEPTH)
                .map(|(p, v)| (p.to_f64().unwrap_or(0.0), v.to_f64().unwrap_or(0.0)))
                .collect(),
            asks: self.asks.iter()
                .take(LOB_DEPTH)
                .map(|(p, v)| (p.to_f64().unwrap_or(0.0), v.to_f64().unwrap_or(0.0)))
                .collect(),
            checksum: self.calculate_checksum(), // Задача 191: добавляем контрольную сумму
            mark_price: self.mark_price, // Задача 233: сохраняем маркированную цену
            volatility_bps: self.get_volatility_bps(), // Задача 191: волатильность
            spread_bps: self.get_spread_bps(), // Задача 191: спред
        }
    }

    /// Проверяет, был ли стакан изменен с последнего дампа
    #[inline(always)]
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Сбрасывает флаг изменений после дампа
    #[inline(always)]
    pub fn reset_dirty(&mut self) {
        self.dirty = false;
    }

    /// Проверяет, пуст ли стакан
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.bids.is_empty() && self.asks.is_empty()
    }

    /// Вычисляет контрольную сумму стакана по алгоритму Bybit V5 (25 levels)
    /// Оптимизировано: используется единый буфер String для минимизации аллокаций.
    /// Задача 180: Расширенная проверка целостности данных
    pub fn calculate_checksum(&self) -> u32 {
        use crc32fast::Hasher;
        
        let mut hasher = Hasher::new();
        let mut buffer = String::with_capacity(1024);

        // 1. Топ-25 Bids (уже отсортированы по убыванию, от лучшей цены к худшей)
        for (price, qty) in self.bids.iter().take(25) {
            let _ = write!(buffer, "{}:{}|", price.normalize(), qty.normalize());
        }

        // 2. Топ-25 Asks (отсортированы по возрастанию, от лучшей цены к худшей)
        for (price, qty) in self.asks.iter().take(25) {
            let _ = write!(buffer, "{}:{}|", price.normalize(), qty.normalize());
        }

        // 3. Удаляем последний символ | перед хешированием
        if buffer.ends_with('|') {
            buffer.pop();
        }

        hasher.update(buffer.as_bytes());
        hasher.finalize()
    }

    /// Проверяет контрольную сумму стакана (Задача 180)
    pub fn verify_checksum(&self, expected: u32) -> bool {
        let calculated = self.calculate_checksum();
        calculated == expected
    }

    /// Очищает стакан (Задача 180)
    pub fn clear(&mut self) {
        self.bids.clear();
        self.asks.clear();
        self.last_update_id = 0;
        self.update_cumulative_caches();
    }

    /// Сбрасывает состояние и записывает новые данные из снапшота
    pub fn reset_with_snapshot(&mut self, update: &OrderBookUpdateArc) {
        self.bids.clear();
        self.asks.clear();
        self.last_update_id = update.last_update_id;
        self.timestamp_ms = update.timestamp_ms;

        for level in &update.bids {
            let p = Decimal::from_f64(level.price).unwrap_or(Decimal::ZERO);
            let v = Decimal::from_f64(level.size).unwrap_or(Decimal::ZERO);
            if v > Decimal::ZERO {
                self.bids.push((p, v));
            }
        }
        self.bids.sort_by(|a, b| b.0.cmp(&a.0));

        for level in &update.asks {
            let p = Decimal::from_f64(level.price).unwrap_or(Decimal::ZERO);
            let v = Decimal::from_f64(level.size).unwrap_or(Decimal::ZERO);
            if v > Decimal::ZERO {
                self.asks.push((p, v));
            }
        }
        self.asks.sort_by(|a, b| a.0.cmp(&b.0));
        
        self.bids.truncate(LOB_DEPTH);
        self.asks.truncate(LOB_DEPTH);
        
        // Обновляем кеши для быстрого расчета VWAP (задача 168)
        self.update_cumulative_caches();
        
        // Задача 191: Обновляем lock-free снапшот после сброса
        // Используем оптимизированное обновление с Arc::make_mut
        self.update_snapshot_optimized();
    }

    /// Применяет обновление
    #[inline(always)]
    pub fn apply_update(&mut self, update: &OrderBookUpdateArc) {
        let start = Instant::now();

        if update.is_snapshot {
            self.reset_with_snapshot(update);
            self.dirty = true; // Устанавливаем флаг изменений
            // Снапшот уже обновлен в reset_with_snapshot
            PROC_LATENCY.update(start.elapsed().as_micros() as u64);
            return;
        } else {
            if self.last_update_id == 0 {
                warn!("[{}] Received delta before first snapshot. Skipping.", self.symbol);
                return;
            }
            if update.last_update_id <= self.last_update_id {
                warn!("[{}] Out-of-order update for {}: {} <= {}. Skipping.", 
                      self.symbol, self.symbol, update.last_update_id, self.last_update_id);
                return;
            }
        }

        self.last_update_id = update.last_update_id;
        self.timestamp_ms = update.timestamp_ms;
        self.dirty = true; // Устанавливаем флаг изменений

        // Применение Bids (descending)
        for level in &update.bids {
            let p = Decimal::from_f64(level.price).unwrap_or(Decimal::ZERO);
            let v = Decimal::from_f64(level.size).unwrap_or(Decimal::ZERO);
            
            match self.bids.binary_search_by(|&(price, _)| p.cmp(&price).reverse()) {
                Ok(idx) => {
                    if v == Decimal::ZERO {
                        self.bids.remove(idx);
                    } else {
                        self.bids[idx].1 = v;
                    }
                }
                Err(idx) => {
                    if v > Decimal::ZERO {
                        self.bids.insert(idx, (p, v));
                    }
                }
            }
        }

        // Применение Asks (ascending)
        for level in &update.asks {
            let p = Decimal::from_f64(level.price).unwrap_or(Decimal::ZERO);
            let v = Decimal::from_f64(level.size).unwrap_or(Decimal::ZERO);

            match self.asks.binary_search_by(|&(price, _)| price.cmp(&p)) {
                Ok(idx) => {
                    if v == Decimal::ZERO {
                        self.asks.remove(idx);
                    } else {
                        self.asks[idx].1 = v;
                    }
                }
                Err(idx) => {
                    if v > Decimal::ZERO {
                        self.asks.insert(idx, (p, v));
                    }
                }
            }
        }

        self.bids.truncate(LOB_DEPTH);
        self.asks.truncate(LOB_DEPTH);

        // Обновляем кеши для быстрого расчета VWAP (задача 168)
        self.update_cumulative_caches();

        // Задача 210: Обновляем волатильность при изменении mid_price
        let mid_price = self.get_mid_price();
        if mid_price > 0.0 {
            self.update_volatility(mid_price);
        }

        // Задача 191: Обновляем lock-free снапшот после применения изменений
        // Используем Arc::make_mut для оптимизации аллокаций
        self.update_snapshot_optimized();

        PROC_LATENCY.update(start.elapsed().as_micros() as u64);
    }

    /// Задача 191: Оптимизированное обновление снапшота с переиспользованием Arc
    /// Если на текущем Arc только одна ссылка, переиспользуем его через Arc::make_mut
    /// Иначе создаем новый Arc
    fn update_snapshot_optimized(&mut self) {
        // Загружаем текущий Arc и пытаемся получить мутабельный доступ
        let mut current_arc = self.current_snapshot.load_full();
        
        // Если на Arc только одна ссылка (Arc::strong_count == 1), можем переиспользовать
        if Arc::strong_count(&current_arc) == 1 {
            // Переиспользуем существующий Arc через Arc::make_mut
            let snapshot_mut = Arc::make_mut(&mut current_arc);
            
            // Обновляем поля существующего снапшота
            snapshot_mut.timestamp_ms = self.timestamp_ms as i64;
            snapshot_mut.last_update_id = self.last_update_id;
            
            // Обновляем bids и asks
            snapshot_mut.bids = self.bids.iter()
                .take(LOB_DEPTH)
                .map(|(p, v)| (p.to_f64().unwrap_or(0.0), v.to_f64().unwrap_or(0.0)))
                .collect();
            snapshot_mut.asks = self.asks.iter()
                .take(LOB_DEPTH)
                .map(|(p, v)| (p.to_f64().unwrap_or(0.0), v.to_f64().unwrap_or(0.0)))
                .collect();
            
            // Обновляем контрольную сумму
            snapshot_mut.checksum = self.calculate_checksum();
            
            // Задача 191: обновляем метрики волатильности и спреда
            snapshot_mut.volatility_bps = self.get_volatility_bps();
            snapshot_mut.spread_bps = self.get_spread_bps();
            
            // Сохраняем переиспользованный Arc обратно
            self.current_snapshot.store(current_arc);
        } else {
            // На Arc есть другие ссылки, создаем новый
            let new_snapshot = self.take_snapshot();
            self.current_snapshot.store(Arc::new(new_snapshot));
        }
    }

    #[inline(always)]
    pub fn get_mid_price_dec(&self) -> Decimal {
        let best_bid = self.bids.first().map(|(p, _)| *p);
        let best_ask = self.asks.first().map(|(p, _)| *p);

        match (best_bid, best_ask) {
            (Some(bid), Some(ask)) => (bid + ask) / Decimal::from(2),
            _ => Decimal::ZERO,
        }
    }

    #[inline(always)]
    pub fn get_mid_price(&self) -> f64 {
        let best_bid = self.bids.first().map(|(p, _)| p.to_f64().unwrap_or(0.0));
        let best_ask = self.asks.first().map(|(p, _)| p.to_f64().unwrap_or(0.0));

        match (best_bid, best_ask) {
            (Some(bid), Some(ask)) => (bid + ask) / 2.0,
            _ => 0.0,
        }
    }

    #[inline(always)]
    pub fn get_spread(&self) -> f64 {
        let best_bid = self.bids.first().map(|(p, _)| p.to_f64().unwrap_or(0.0));
        let best_ask = self.asks.first().map(|(p, _)| p.to_f64().unwrap_or(0.0));

        match (best_bid, best_ask) {
            (Some(bid), Some(ask)) => ask - bid,
            _ => 0.0,
        }
    }

    #[inline(always)]
    pub fn get_best_bid_ask_with_vol(&self) -> (f64, f64, f64, f64) {
        let bid = self.bids.first().map(|(p, v): &(Decimal, Decimal)| (p.to_f64().unwrap_or(0.0), v.to_f64().unwrap_or(0.0))).unwrap_or((0.0, 0.0));
        let ask = self.asks.first().map(|(p, v): &(Decimal, Decimal)| (p.to_f64().unwrap_or(0.0), v.to_f64().unwrap_or(0.0))).unwrap_or((0.0, 0.0));
        (bid.0, bid.1, ask.0, ask.1)
    }

    #[inline(always)]
    pub fn get_best_bid_ask(&self) -> (f64, f64) {
        let (bid, _, ask, _) = self.get_best_bid_ask_with_vol();
        (bid, ask)
    }

    /// Заполняет переданный буфер топ N уровнями стакана.
    /// Оптимизировано: Zero-copy, запись напрямую в слайс.
    /// Порядок: Asks (p_0, v_0, ...), затем Bids (p_0, v_0, ...).
    #[inline(always)]
    pub fn fill_flat_buffer(&self, n: usize, buffer: &mut [f32]) {
        let mut offset = 0;

        // Asks (best to worst)
        for i in 0..n {
            if let Some((p, v)) = self.asks.get(i) {
                buffer[offset] = p.to_f32().unwrap_or(0.0);
                buffer[offset + 1] = v.to_f32().unwrap_or(0.0);
            } else {
                buffer[offset] = 0.0;
                buffer[offset + 1] = 0.0;
            }
            offset += 2;
        }

        // Bids (best to worst)
        for i in 0..n {
            if let Some((p, v)) = self.bids.get(i) {
                buffer[offset] = p.to_f32().unwrap_or(0.0);
                buffer[offset + 1] = v.to_f32().unwrap_or(0.0);
            } else {
                buffer[offset] = 0.0;
                buffer[offset + 1] = 0.0;
            }
            offset += 2;
        }
    }

    pub fn get_flat_snapshot(&self) -> Vec<f32> {
        let mut buffer = vec![0.0f32; LOB_DEPTH * 4];
        self.fill_flat_buffer(LOB_DEPTH, &mut buffer);
        buffer
    }

    /// Расчет дисбаланса стакана (Order Book Imbalance)
    /// Возвращает значение в диапазоне [-1, 1]:
    /// - Положительное значение: доминируют покупатели (больше bid volume)
    /// - Отрицательное значение: доминируют продавцы (больше ask volume)
    #[inline(always)]
    pub fn calculate_imbalance(&self, depth: usize) -> f64 {
        // Защита от выхода за пределы стакана
        let d = depth.min(self.bids.len().min(self.asks.len()));
        if d == 0 {
            return 0.0;
        }

        let bid_vol = self.bids.iter().take(d)
            .fold(Decimal::ZERO, |acc, (_, v)| acc + v);
        let ask_vol = self.asks.iter().take(d)
            .fold(Decimal::ZERO, |acc, (_, v)| acc + v);
        
        let total_vol = bid_vol + ask_vol;
        if total_vol.is_zero() {
            return 0.0;
        }

        // Конвертируем в f64 для вычисления коэффициента
        let b = bid_vol.to_f64().unwrap_or(0.0);
        let a = ask_vol.to_f64().unwrap_or(0.0);
        
        (b - a) / (b + a)
    }

    /// Рассчитывает средневзвешенную цену исполнения (VWAP) и возвращает отклонение от best_price в базисных пунктах (bps).
    /// Используется для оценки проскальзывания при исполнении ордера заданного размера.
    /// 
    /// # Параметры
    /// - `side`: Сторона ордера (Buy = покупка по asks, Sell = продажа по bids)
    /// - `target_size_quote`: Целевой объем в USD (quote currency)
    /// 
    /// # Возвращает
    /// Отклонение VWAP от best_price в базисных пунктах (bps). 
    /// Положительное значение означает проскальзывание против нас.
    pub fn calculate_vwap_impact(&self, side: crate::data::types::Side, target_size_quote: f64) -> f64 {
        use crate::data::types::Side;
        
        if target_size_quote <= 0.0 {
            return 0.0;
        }

        // Выбираем уровни в зависимости от стороны
        let levels = match side {
            Side::Buy => &self.asks,  // Покупаем по asks
            Side::Sell => &self.bids, // Продаем по bids
        };

        if levels.is_empty() {
            return f64::MAX; // Нет ликвидности
        }

        // Получаем best_price
        let best_price = levels[0].0.to_f64().unwrap_or(0.0);
        if best_price <= 0.0 {
            return f64::MAX;
        }

        // Рассчитываем VWAP через оптимизированный метод с бинарным поиском (задача 168)
        let size_base = target_size_quote / best_price;
        let vwap = match self.get_execution_vwap(side, size_base) {
            Some(v) => v,
            None => return f64::MAX, // Недостаточно ликвидности
        };
        
        // Отклонение в базисных пунктах (bps)
        // Для Buy: если VWAP > best_price, то проскальзывание положительное
        // Для Sell: если VWAP < best_price, то проскальзывание положительное
        let impact = match side {
            Side::Buy => (vwap - best_price) / best_price * 10000.0,
            Side::Sell => (best_price - vwap) / best_price * 10000.0,
        };

        impact.max(0.0) // Возвращаем только положительное проскальзывание
    }

    /// Возвращает объем на лучшем уровне (Top of Book) в USD.
    /// 
    /// # Параметры
    /// - `side`: Сторона ордера (Buy = смотрим на asks, Sell = смотрим на bids)
    /// 
    /// # Возвращает
    /// Объем в USD на лучшем уровне. Возвращает 0.0 если стакан пуст.
    pub fn get_volume_at_best(&self, side: crate::data::types::Side) -> f64 {
        use crate::data::types::Side;
        
        let levels = match side {
            Side::Buy => &self.asks,  // Покупаем по asks
            Side::Sell => &self.bids, // Продаем по bids
        };

        if let Some((price, volume)) = levels.first() {
            let p = price.to_f64().unwrap_or(0.0);
            let v = volume.to_f64().unwrap_or(0.0);
            p * v
        } else {
            0.0
        }
    }

    /// Обновляет кеши кумулятивных объемов и цена*объем (задача 168)
    /// Вызывается после каждого обновления стакана
    fn update_cumulative_caches(&mut self) {
        // Обновляем кеши для Bids
        self.cum_vol_bids.clear();
        self.cum_price_vol_bids.clear();
        let mut cum_vol = 0.0;
        let mut cum_pv = 0.0;
        
        for (price, volume) in &self.bids {
            let p = price.to_f64().unwrap_or(0.0);
            let v = volume.to_f64().unwrap_or(0.0);
            cum_vol += v;
            cum_pv += p * v;
            self.cum_vol_bids.push(cum_vol);
            self.cum_price_vol_bids.push(cum_pv);
        }

        // Обновляем кеши для Asks
        self.cum_vol_asks.clear();
        self.cum_price_vol_asks.clear();
        cum_vol = 0.0;
        cum_pv = 0.0;
        
        for (price, volume) in &self.asks {
            let p = price.to_f64().unwrap_or(0.0);
            let v = volume.to_f64().unwrap_or(0.0);
            cum_vol += v;
            cum_pv += p * v;
            self.cum_vol_asks.push(cum_vol);
            self.cum_price_vol_asks.push(cum_pv);
        }
    }

    /// Рассчитывает VWAP (Volume-Weighted Average Price) для исполнения (задача 168)
    /// Использует бинарный поиск по кумулятивным объемам для O(log N) сложности
    /// 
    /// # Параметры
    /// - `side`: Сторона ордера (Buy = смотрим на asks, Sell = смотрим на bids)
    /// - `size`: Требуемый объем для исполнения
    /// 
    /// # Возвращает
    /// Some(vwap) если достаточно ликвидности, None если размер больше доступного
    pub fn get_execution_vwap(&self, side: crate::data::types::Side, size: f64) -> Option<f64> {
        use crate::data::types::Side;
        
        let (cum_vols, cum_pvs, _levels) = match side {
            Side::Buy => (&self.cum_vol_asks, &self.cum_price_vol_asks, &self.asks),
            Side::Sell => (&self.cum_vol_bids, &self.cum_price_vol_bids, &self.bids),
        };

        if cum_vols.is_empty() {
            return None;
        }

        // Проверяем, достаточно ли ликвидности
        if let Some(&total_vol) = cum_vols.last() {
            if size > total_vol {
                return None; // Недостаточно ликвидности
            }
        } else {
            return None;
        }

        // Бинарный поиск для нахождения уровня, где накопленный объем >= size
        let idx = cum_vols.binary_search_by(|&v| {
            if v < size {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        }).unwrap_or_else(|i| i);

        if idx >= cum_pvs.len() {
            return None;
        }

        // Вычисляем VWAP по формуле: VWAP = cum_price_vol / cum_vol
        let vwap = cum_pvs[idx] / cum_vols[idx];
        Some(vwap)
    }

    /// Получить объем на конкретной цене (задача 203)
    /// Использует бинарный поиск по отсортированному SmallVec для O(log n) производительности
    pub fn get_volume_at_price(&self, price: f64, side: crate::data::types::Side) -> f64 {
        use crate::data::types::Side;
        
        let levels = match side {
            Side::Buy => &self.asks,  // Для покупки смотрим на asks
            Side::Sell => &self.bids, // Для продажи смотрим на bids
        };

        let price_dec = Decimal::from_f64(price).unwrap_or(Decimal::ZERO);
        let tolerance = Decimal::from_f64(1e-8).unwrap_or(Decimal::ZERO);
        
        // Бинарный поиск по отсортированному стакану
        match levels.binary_search_by(|(level_price, _)| {
            let diff = *level_price - price_dec;
            if diff.abs() < tolerance {
                std::cmp::Ordering::Equal
            } else if diff < Decimal::ZERO {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        }) {
            Ok(idx) => levels[idx].1.to_f64().unwrap_or(0.0),
            Err(_) => 0.0,
        }
    }

    /// Задача 210: Обновление волатильности с использованием алгоритма Велфорда
    /// Вызывается при каждом обновлении mid_price
    /// Использует f64 для всех вычислений для избежания накопления ошибки округления
    /// 
    /// Кольцевой буфер: 500 последних цен (достаточно для расчета волатильности)
    /// Алгоритм Велфорда: O(1) на каждый тик, без пересчета всех данных
    /// Инверсный метод: корректно удаляет старые значения при переполнении буфера
    pub fn update_volatility(&mut self, mid_price: f64) {
        if mid_price <= 0.0 || !mid_price.is_finite() {
            return; // Игнорируем некорректные цены
        }

        // Если буфер полон, нужно удалить старое значение перед добавлением нового
        let old_value = if self.price_buffer.len() == self.price_buffer.capacity() {
            // Получаем самое старое значение (первое в буфере)
            self.price_buffer.front().copied()
        } else {
            None
        };

        // Добавляем новое значение в буфер (автоматически вытесняет старое при переполнении)
        self.price_buffer.push_back(mid_price);

        // Если был удален старый элемент, применяем инверсный метод Велфорда
        if let Some(old_val) = old_value {
            if self.volatility_count > 1 {
                // Инверсный метод Велфорда для удаления старого значения
                let old_delta = old_val - self.volatility_mean;
                self.volatility_count -= 1;
                let new_mean = (self.volatility_mean * (self.volatility_count + 1) as f64 - old_val) 
                    / self.volatility_count as f64;
                let new_delta = old_val - new_mean;
                self.volatility_m2 -= old_delta * new_delta;
                self.volatility_mean = new_mean;
            }
        }

        // Прямой метод Велфорда для добавления нового значения
        self.volatility_count += 1;
        let delta = mid_price - self.volatility_mean;
        self.volatility_mean += delta / self.volatility_count as f64;
        let delta2 = mid_price - self.volatility_mean;
        self.volatility_m2 += delta * delta2;
    }

    /// Задача 210: Получение текущей волатильности в базисных пунктах (bps)
    /// Возвращает стандартное отклонение относительно среднего в bps
    /// Формула: (std_dev / mean) * 10000
    #[inline(always)]
    pub fn get_volatility_bps(&self) -> f64 {
        if self.volatility_count < 2 || self.volatility_mean <= 0.0 {
            return 0.0;
        }

        // Вычисляем дисперсию: variance = M2 / (n - 1)
        let variance = self.volatility_m2 / (self.volatility_count - 1) as f64;
        
        // Стандартное отклонение
        let std_dev = variance.sqrt();
        
        // Волатильность в базисных пунктах
        (std_dev / self.volatility_mean) * 10000.0
    }

    /// Задача 210: Получение текущего спреда в базисных пунктах (bps)
    /// Формула: ((ask - bid) / mid_price) * 10000
    #[inline(always)]
    pub fn get_spread_bps(&self) -> f64 {
        let (bid, ask) = self.get_best_bid_ask();
        if bid <= 0.0 || ask <= 0.0 {
            return 0.0;
        }
        
        let mid = (bid + ask) / 2.0;
        if mid <= 0.0 {
            return 0.0;
        }
        
        ((ask - bid) / mid) * 10000.0
    }

    /// Задача 210: Получение статуса волатильности с логированием
    /// Возвращает кортеж (volatility_bps, buffer_fill_percent, volatility_level)
    /// Используется для мониторинга и логирования
    pub fn get_volatility_status(&self) -> (f64, f64, &'static str) {
        let volatility_bps = self.get_volatility_bps();
        let buffer_fill_percent = (self.volatility_count as f64 / self.price_buffer.capacity() as f64) * 100.0;
        
        let volatility_level = if volatility_bps < 200.0 {
            "LOW"
        } else if volatility_bps < 500.0 {
            "MEDIUM"
        } else if volatility_bps < 1000.0 {
            "HIGH"
        } else {
            "EXTREME"
        };
        
        (volatility_bps, buffer_fill_percent, volatility_level)
    }

    /// Задача 233: Получить маркированную цену
    #[inline(always)]
    pub fn get_mark_price(&self) -> f64 {
        self.mark_price
    }

    /// Задача 233: Установить маркированную цену
    #[inline(always)]
    pub fn set_mark_price(&mut self, price: f64) {
        self.mark_price = price;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::types::PriceLevel;

    #[test]
    fn test_orderbook_snapshot() {
        let mut ob = OrderBook::new("BTCUSDT");
        let mut bids = SmallVec::new();
        bids.push(PriceLevel { price: 50000.0, size: 1.0 });
        let update = OrderBookUpdateArc {
            symbol: Arc::from("BTCUSDT"),
            timestamp_ms: 1000,
            last_update_id: 10,
            is_snapshot: true,
            bids,
            asks: SmallVec::from_buf([PriceLevel { price: 50001.0, size: 2.0 }; 1]),
            checksum: None,
        };
        ob.apply_update(&update);
        
        assert_eq!(ob.last_update_id, 10);
        let flat = ob.get_flat_snapshot();
        assert_eq!(flat[0], 50001.0); // Best Ask Price
        assert_eq!(flat[1], 2.0);     // Best Ask Size
        assert_eq!(flat[100], 50000.0); // Best Bid Price
        assert_eq!(flat[101], 1.0);     // Best Bid Size
    }

    #[test]
    fn test_calculate_checksum() {
        let mut ob = OrderBook::new("BTCUSDT");
        let mut bids = SmallVec::new();
        bids.push(PriceLevel { price: 50000.0, size: 1.5 });
        ob.apply_update(&OrderBookUpdateArc {
            symbol: Arc::from("BTCUSDT"),
            timestamp_ms: 1000,
            last_update_id: 1,
            is_snapshot: true,
            bids,
            asks: SmallVec::from_buf([PriceLevel { price: 50001.0, size: 2.5 }; 1]),
            checksum: None,
        });

        let cs = ob.calculate_checksum();
        
        // Вычисляем ожидаемый CRC32 от строки в формате Bybit V5
        // Формат: price:size|price:size|...
        // Bids сначала (50000:1.5), потом Asks (50001:2.5)
        let expected_string = "50000:1.5|50001:2.5";
        let expected_cs = crc32fast::hash(expected_string.as_bytes());
        
        assert_eq!(cs, expected_cs, "Checksum must match Bybit V5 CRC32 IEEE algorithm");
    }

    #[test]
    fn test_lock_free_concurrent_access() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::thread;
        use std::time::Duration;
        
        // Создаем OrderBook и оборачиваем в Arc для многопоточного доступа
        let ob = Arc::new(std::sync::Mutex::new(OrderBook::new("BTCUSDT")));
        
        // Инициализируем начальный снапшот
        {
            let mut ob_guard = ob.lock().unwrap();
            let mut bids = SmallVec::new();
            bids.push(PriceLevel { price: 50000.0, size: 1.0 });
            ob_guard.apply_update(&OrderBookUpdateArc {
                symbol: Arc::from("BTCUSDT"),
                timestamp_ms: 1000,
                last_update_id: 1,
                is_snapshot: true,
                bids,
                asks: SmallVec::from_buf([PriceLevel { price: 50001.0, size: 2.0 }; 1]),
                checksum: None,
            });
        }
        
        // Флаг для остановки потоков
        let stop_flag = Arc::new(AtomicBool::new(false));
        let mut handles = vec![];
        
        // Поток записи (имитация WebSocket) - постоянно обновляет снапшот
        let ob_write = Arc::clone(&ob);
        let stop_write = Arc::clone(&stop_flag);
        let write_handle = thread::spawn(move || {
            let mut update_id = 2u64;
            while !stop_write.load(Ordering::Relaxed) {
                let mut ob_guard = ob_write.lock().unwrap();
                
                // Обновляем цены
                let new_bid_price = 50000.0 + (update_id as f64 * 0.1) % 10.0;
                let new_ask_price = 50001.0 + (update_id as f64 * 0.1) % 10.0;
                
                let mut bids = SmallVec::new();
                bids.push(PriceLevel { price: new_bid_price, size: 1.0 + (update_id as f64 % 5.0) });
                
                let mut asks = SmallVec::new();
                asks.push(PriceLevel { price: new_ask_price, size: 2.0 + (update_id as f64 % 5.0) });
                
                ob_guard.apply_update(&OrderBookUpdateArc {
                    symbol: Arc::from("BTCUSDT"),
                    timestamp_ms: 1000 + update_id,
                    last_update_id: update_id,
                    is_snapshot: false,
                    bids,
                    asks,
                    checksum: None,
                });
                
                update_id += 1;
                drop(ob_guard);
                thread::sleep(Duration::from_micros(100));
            }
        });
        handles.push(write_handle);
        
        // Несколько потоков чтения (имитация Strategy) - постоянно читают снапшот
        for i in 0..5 {
            let ob_read = Arc::clone(&ob);
            let stop_read = Arc::clone(&stop_flag);
            let read_handle = thread::spawn(move || {
                let mut read_count = 0;
                while !stop_read.load(Ordering::Relaxed) {
                    let ob_guard = ob_read.lock().unwrap();
                    
                    // Загружаем снапшот (lock-free операция)
                    let snap = ob_guard.current_snapshot.load();
                    
                    // Проверяем консистентность данных
                    let mid = snap.get_mid_price();
                    let (bid, ask) = snap.get_best_bid_ask();
                    let (bid2, bid_vol, ask2, ask_vol) = snap.get_best_bid_ask_with_vol();
                    let flat = snap.get_flat_snapshot();
                    
                    // Проверяем, что данные консистентны
                    assert!(mid > 0.0, "Reader {} got invalid mid price", i);
                    assert_eq!(bid, bid2, "Reader {} got inconsistent bid", i);
                    assert_eq!(ask, ask2, "Reader {} got inconsistent ask", i);
                    assert_eq!(flat.len(), LOB_DEPTH * 4, "Reader {} got invalid flat snapshot length", i);
                    
                    // Проверяем, что снапшот имеет правильную структуру
                    assert!(snap.bids.len() > 0, "Reader {} got empty bids", i);
                    assert!(snap.asks.len() > 0, "Reader {} got empty asks", i);
                    assert_eq!(snap.bids.len(), snap.asks.len(), "Reader {} got mismatched bid/ask lengths", i);
                    
                    read_count += 1;
                    drop(ob_guard);
                    thread::sleep(Duration::from_micros(50));
                }
                read_count
            });
            handles.push(read_handle);
        }
        
        // Даем потокам время на выполнение
        thread::sleep(Duration::from_millis(500));
        
        // Останавливаем все потоки
        stop_flag.store(true, Ordering::Relaxed);
        
        // Ждем завершения всех потоков
        let mut total_reads = 0;
        for (idx, handle) in handles.into_iter().enumerate() {
            if idx == 0 {
                // Поток записи
                handle.join().expect("Write thread panicked");
            } else {
                // Потоки чтения
                let read_count = handle.join().expect("Read thread panicked");
                total_reads += read_count;
                println!("Reader {} completed {} reads", idx - 1, read_count);
            }
        }
        
        println!("Total reads across all readers: {}", total_reads);
        
        // Проверяем, что данные остались консистентными после всех операций
        let ob_guard = ob.lock().unwrap();
        let final_snap = ob_guard.current_snapshot.load();
        assert!(final_snap.get_mid_price() > 0.0, "Final snapshot has invalid mid price");
        assert_eq!(final_snap.bids.len(), final_snap.asks.len(), "Final snapshot has mismatched bid/ask lengths");
        assert!(final_snap.bids.len() > 0, "Final snapshot has empty bids");
        assert!(final_snap.asks.len() > 0, "Final snapshot has empty asks");
    }
}
