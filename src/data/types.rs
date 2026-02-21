use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use serde_with::{serde_as, DisplayFromStr};

/// Один уровень в стакане (цена + объем)
#[serde_as]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PriceLevel {
    #[serde_as(as = "DisplayFromStr")]
    pub price: f64,
    #[serde_as(as = "DisplayFromStr")]
    pub size: f64, // объем (volume)
}

/// Обновление или полный снапшот стакана от Bybit (borrowed версия для zero-copy парсинга)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookUpdate<'a> {
    #[serde(borrow)]
    pub symbol: &'a str,
    pub timestamp_ms: u64,      // ts события от биржи
    pub last_update_id: u64,    // 'u' или 'seqNum' для проверки последовательности
    pub is_snapshot: bool,      // true для полного снимка, false для дельты (update)
    pub bids: SmallVec<[PriceLevel; 50]>,  // Лимиты на покупку (до 50 уровней на стеке)
    pub asks: SmallVec<[PriceLevel; 50]>,  // Лимиты на продажу (до 50 уровней на стеке)
    pub checksum: Option<u32>,   // Контрольная сумма от биржи (Bybit cs)
}

impl<'a> OrderBookUpdate<'a> {
    /// Конвертирует borrowed версию в owned для хранения
    pub fn to_owned(&self) -> OrderBookUpdateOwned {
        OrderBookUpdateOwned {
            symbol: self.symbol.to_string(),
            timestamp_ms: self.timestamp_ms,
            last_update_id: self.last_update_id,
            is_snapshot: self.is_snapshot,
            bids: self.bids.clone(),
            asks: self.asks.clone(),
            checksum: self.checksum,
        }
    }
}

/// Owned версия OrderBookUpdate для хранения в буферах и каналах
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookUpdateOwned {
    pub symbol: String,
    pub timestamp_ms: u64,
    pub last_update_id: u64,
    pub is_snapshot: bool,
    pub bids: SmallVec<[PriceLevel; 50]>,
    pub asks: SmallVec<[PriceLevel; 50]>,
    pub checksum: Option<u32>,
}

/// Публичная сделка от биржи (для расчета VWAP/TWAP) - borrowed версия для zero-copy парсинга
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicTrade {
    pub price: rust_decimal::Decimal,
    pub size: rust_decimal::Decimal,
    pub side: Side,       // Buy/Sell
    pub timestamp: i64,   // Unix MS
}

impl PublicTrade {
    /// Конвертирует borrowed версию в owned для хранения
    pub fn to_owned(&self) -> PublicTradeOwned {
        PublicTradeOwned {
            price: self.price,
            size: self.size,
            side: self.side,
            timestamp: self.timestamp,
        }
    }
}

/// Owned версия PublicTrade для хранения
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicTradeOwned {
    pub price: rust_decimal::Decimal,
    pub size: rust_decimal::Decimal,
    pub side: Side,
    pub timestamp: i64,
}

/// Сторона сделки
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    Buy,
    Sell,
}


/// Enum для передачи разных типов данных из WebSocket
#[derive(Debug, Clone)]
pub enum WsData {
    OrderBook(OrderBookUpdateOwned),
    Trades(Vec<PublicTradeOwned>),
    Ticker(TickerOwned),
    MarkPrice(String, f64), // (symbol, mark_price) - Задача 233
}


/// Информация о тикере (текущая цена, объемы, ставка финансирования) - borrowed версия
#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker<'a> {
    #[serde(borrow)]
    pub symbol: &'a str,
    #[serde_as(as = "DisplayFromStr")]
    pub last_price: f64,           // Последняя цена
    #[serde_as(as = "DisplayFromStr")]
    pub bid: f64,                  // Лучшая цена покупки
    #[serde_as(as = "DisplayFromStr")]
    pub ask: f64,                  // Лучшая цена продажи
    #[serde_as(as = "DisplayFromStr")]
    pub bid_size: f64,             // Объем на лучшей цене покупки
    #[serde_as(as = "DisplayFromStr")]
    pub ask_size: f64,             // Объем на лучшей цене продажи
    #[serde_as(as = "DisplayFromStr")]
    pub volume_24h: f64,           // Объем за 24 часа
    #[serde_as(as = "DisplayFromStr")]
    pub turnover_24h: f64,         // Оборот за 24 часа
    #[serde_as(as = "DisplayFromStr")]
    pub funding_rate: f64,         // Текущая ставка финансирования (например, 0.0005 = 0.05%)
    pub next_funding_time: u64,    // Время следующего клиринга фандинга (Unix MS)
    pub timestamp_ms: u64,         // Время получения данных
}

impl<'a> Ticker<'a> {
    /// Конвертирует borrowed версию в owned для хранения
    pub fn to_owned(&self) -> TickerOwned {
        TickerOwned {
            symbol: self.symbol.to_string(),
            last_price: self.last_price,
            bid: self.bid,
            ask: self.ask,
            bid_size: self.bid_size,
            ask_size: self.ask_size,
            volume_24h: self.volume_24h,
            turnover_24h: self.turnover_24h,
            funding_rate: self.funding_rate,
            next_funding_time: self.next_funding_time,
            timestamp_ms: self.timestamp_ms,
        }
    }
}

/// Owned версия Ticker для хранения
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickerOwned {
    pub symbol: String,
    pub last_price: f64,
    pub bid: f64,
    pub ask: f64,
    pub bid_size: f64,
    pub ask_size: f64,
    pub volume_24h: f64,
    pub turnover_24h: f64,
    pub funding_rate: f64,
    pub next_funding_time: u64,
    pub timestamp_ms: u64,
}
