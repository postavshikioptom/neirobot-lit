# 013 - Data Orderbook Struct

Цель задачи: Реализовать структуру OrderBook в Rust, которая отвечает за поддержание актуального состояния стакана (LOB). Она должна использовать BTreeMap для автоматической сортировки цен, корректно обрабатывать дельты (обновления) и снапшоты от Bybit, а также подготавливать данные для записи в Parquet согласно схеме из задачи 012.

Файлы для изменения/создания:

src/data/orderbook.rs
src/data/mod.rs
Инструкции для Gemini:

src/data/orderbook.rs: Реализовать структуру с использованием BTreeMap и OrderedFloat.

use std::collections::BTreeMap;
use ordered_float::OrderedFloat;
use tracing::warn;
use crate::data::types::{OrderBookUpdate, PriceLevel};

pub const LOB_DEPTH: usize = 50;

pub struct OrderBook {
    pub symbol: String,
    bids: BTreeMap<OrderedFloat<f64>, f64>, // Key: Price, Value: Size
    asks: BTreeMap<OrderedFloat<f64>, f64>,
    pub last_update_id: u64,
    pub timestamp_ms: u64,
}

impl OrderBook {
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            last_update_id: 0,
            timestamp_ms: 0,
        }
    }

    /// Применяет обновление. Принимает ссылку для эффективности.
    pub fn apply_update(&mut self, update: &OrderBookUpdate) {
        // Базовая валидация последовательности (только для дельт)
        if !update.is_snapshot && update.last_update_id <= self.last_update_id {
            warn!("[{}] Out-of-order update: {} <= {}", self.symbol, update.last_update_id, self.last_update_id);
            return;
        }

        if update.is_snapshot {
            self.bids.clear();
            self.asks.clear();
        }

        self.last_update_id = update.last_update_id;
        self.timestamp_ms = update.timestamp_ms;

        // Обновление Bids (покупка)
        for level in &update.bids {
            if level.size == 0.0 {
                self.bids.remove(&OrderedFloat(level.price));
            } else {
                self.bids.insert(OrderedFloat(level.price), level.size);
            }
        }

        // Обновление Asks (продажа)
        for level in &update.asks {
            if level.size == 0.0 {
                self.asks.remove(&OrderedFloat(level.price));
            } else {
                self.asks.insert(OrderedFloat(level.price), level.size);
            }
        }
        
        // После применения обновлений обрезка (truncate) не требуется, 
        // так как мы берем top-50 только при запросе снимка.
    }

    /// Возвращает плоский вектор уровней для Parquet (p_0, v_0, ..., p_49, v_49)
    /// Сначала Asks, затем Bids. Добавляет padding (0.0), если уровней < 50.
    pub fn get_flat_snapshot(&self) -> Vec<f32> {
        let mut data = Vec::with_capacity(LOB_DEPTH * 4); // 50 asks (p,v) + 50 bids (p,v)

        // Asks: от лучшего (min price) к худшему
        let mut ask_count = 0;
        for (&price, &size) in self.asks.iter().take(LOB_DEPTH) {
            data.push(price.0 as f32);
            data.push(size as f32);
            ask_count += 1;
        }
        // Padding для Asks
        for _ in ask_count..LOB_DEPTH {
            data.push(0.0);
            data.push(0.0);
        }

        // Bids: от лучшего (max price) к худшему
        let mut bid_count = 0;
        for (&price, &size) in self.bids.iter().rev().take(LOB_DEPTH) {
            data.push(price.0 as f32);
            data.push(size as f32);
            bid_count += 1;
        }
        // Padding для Bids
        for _ in bid_count..LOB_DEPTH {
            data.push(0.0);
            data.push(0.0);
        }

        data
    }
}
Технические требования:

Типы: Использовать f32 в get_flat_snapshot согласно схеме данных.
Сортировка: Аски — iter() (по возрастанию), Биды — iter().rev() (по убыванию).
Padding: Если стакан пуст или содержит меньше 50 уровней, заполнять недостающие места нулями (0.0).
Эффективность: Метод apply_update должен быть максимально быстрым. Валидация last_update_id — только warn!.