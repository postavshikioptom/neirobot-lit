use serde::Deserialize;
use tracing::debug;
use crate::data::types::PriceLevel;
use crate::trading::types::{OrderUpdate, OrderStatus};
use anyhow::{Result, Context};
use rust_decimal::Decimal;
use std::str::FromStr;

/// Быстрый парсинг f64 с использованием fast_float
#[inline]
fn parse_f64_fast(s: &str) -> Result<f64> {
    fast_float::parse(s).context("Failed to parse f64")
}

#[derive(Debug, Deserialize)]
pub struct BybitOrderbookMsg<'a> {
    #[serde(rename = "type", borrow)]
    pub msg_type: &'a str,
    #[serde(borrow)]
    pub topic: &'a str,
    pub ts: u64,
    #[serde(borrow)]
    pub data: BybitData<'a>,
}

/// Уровень стакана с заимствованными строками (кортеж для десериализации массива)
#[derive(Debug, Deserialize)]
pub struct PriceLevelData<'a>(#[serde(borrow)] &'a str, #[serde(borrow)] &'a str);

#[derive(Debug, Deserialize)]
pub struct BybitData<'a> {
    #[serde(borrow)]
    pub s: &'a str,      // symbol
    pub b: Vec<PriceLevelData<'a>>, // bids [price, size]
    pub a: Vec<PriceLevelData<'a>>, // asks [price, size]
    pub u: u64,         // update id (u)
    pub seq: Option<u64>, // sequence (seq)
    pub cs: Option<u64>,  // checksum (cs)
}

#[derive(Debug, Deserialize)]
pub struct BybitPrivateMsg {
    pub topic: String,
    pub data: serde_json::Value,
    pub ts: u64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BybitOrderData {
    pub symbol: String,
    pub order_id: String,
    pub order_link_id: String,
    pub order_status: String,
    pub cum_exec_qty: String,
    pub avg_price: String,
    pub cum_exec_fee: String,
    pub price: Option<String>,  // Цена ордера (может измениться при amendment)
    pub qty: Option<String>,    // Объем ордера (может измениться при amendment)
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BybitExecutionData {
    pub symbol: String,
    pub order_id: String,
    pub order_link_id: String,
    pub exec_qty: String,
    pub exec_price: String,
    pub exec_fee: String,
    pub cum_exec_qty: String,
    pub cum_exec_fee: String,
    pub exec_type: String, // "Trade", "AdlTrade", etc.
    pub last_liquidity_ind: String, // "AddedLiquidity" (Maker), "RemovedLiquidity" (Taker)
}

/// Парсит сырое JSON-сообщение от Bybit в список OrderUpdate.
pub fn parse_private_msg(json_str: &str) -> Result<Vec<OrderUpdate>> {
    let msg: BybitPrivateMsg = serde_json::from_str(json_str)
        .context("Failed to deserialize Bybit private message")?;

    let mut updates = Vec::new();

    if msg.topic == "order" {
        let data_list: Vec<BybitOrderData> = serde_json::from_value(msg.data)?;
        for d in data_list {
            updates.push(OrderUpdate {
                order_link_id: d.order_link_id,
                order_id: d.order_id,
                status: OrderStatus::from_bybit_status(&d.order_status),
                cum_exec_qty: Decimal::from_str(&d.cum_exec_qty).unwrap_or_default(),
                exec_price: Decimal::from_str(&d.avg_price).ok(),
                exec_fee: Decimal::from_str(&d.cum_exec_fee).ok(),
                is_maker: None, // В топике order нет информации о ликвидности
                reason: None,
                timestamp: msg.ts,
                new_price: d.price.and_then(|p| Decimal::from_str(&p).ok()),
                new_qty: d.qty.and_then(|q| Decimal::from_str(&q).ok()),
            });
        }
    } else if msg.topic == "execution" {
        let data_list: Vec<BybitExecutionData> = serde_json::from_value(msg.data)?;
        for d in data_list {
            if d.exec_type != "Trade" { continue; }
            
            updates.push(OrderUpdate {
                order_link_id: d.order_link_id,
                order_id: d.order_id,
                status: OrderStatus::PartiallyFilled, // Исполнение всегда означает как минимум частичное заполнение
                cum_exec_qty: Decimal::from_str(&d.cum_exec_qty).unwrap_or_default(), 
                exec_price: Decimal::from_str(&d.exec_price).ok(),
                exec_fee: Decimal::from_str(&d.exec_fee).ok(),
                is_maker: Some(d.last_liquidity_ind == "AddedLiquidity"),
                reason: None,
                timestamp: msg.ts,
                new_price: None,  // В топике execution нет информации о цене/объеме ордера
                new_qty: None,
            });
        }
    }

    Ok(updates)
}

/// Парсит сырое JSON-сообщение от Bybit в OrderBookUpdate.
/// Возвращает None, если сообщение не относится к стакану (например, ответ на ping).
pub fn parse_orderbook_msg<'a>(json_str: &'a str) -> Result<Option<crate::data::types::OrderBookUpdate<'a>>> {
    // 1. Быстрая фильтрация: если это не сообщение стакана, пропускаем
    if !json_str.contains("orderbook") {
        return Ok(None);
    }

    // Используем from_slice для работы напрямую с байтами (zero-copy)
    let msg: BybitOrderbookMsg<'a> = serde_json::from_slice(json_str.as_bytes())
        .context("Failed to deserialize Bybit orderbook message")?;

    // 2. Конвертируем уровни стакана из заимствованных строк в f64
    let parse_levels = |levels: &Vec<PriceLevelData>| -> Result<Vec<PriceLevel>> {
        levels.iter()
            .map(|level| {
                Ok(PriceLevel {
                    price: parse_f64_fast(level.0)?,
                    size: parse_f64_fast(level.1)?,
                })
            })
            .collect()
    };

    debug!("[{}] Parsing LOB message: ts={}, u={}, type={}", msg.data.s, msg.ts, msg.data.u, msg.msg_type);
    
    Ok(Some(crate::data::types::OrderBookUpdate {
        symbol: msg.data.s,
        timestamp_ms: msg.ts,
        last_update_id: msg.data.u,
        is_snapshot: msg.msg_type == "snapshot",
        bids: parse_levels(&msg.data.b)?.into(),
        asks: parse_levels(&msg.data.a)?.into(),
        checksum: msg.data.cs.map(|c| c as u32),
    }))
}

use crate::data::types::Side;

#[derive(Debug, Deserialize)]
pub struct BybitPublicTradeMsg<'a> {
    #[serde(borrow)]
    pub topic: &'a str,
    #[serde(rename = "type", borrow)]
    pub msg_type: &'a str,
    pub ts: u64,
    pub data: Vec<BybitPublicTradeData<'a>>,
}

#[derive(Debug, Deserialize)]
pub struct BybitPublicTradeData<'a> {
    #[serde(rename = "T")]
    pub timestamp: i64,  // Timestamp сделки
    #[serde(rename = "s", borrow)]
    pub symbol: &'a str,
    #[serde(rename = "S", borrow)]
    pub side: &'a str,    // "Buy" или "Sell"
    #[serde(rename = "v", borrow)]
    pub size: &'a str,    // Объем
    #[serde(rename = "p", borrow)]
    pub price: &'a str,   // Цена
    #[serde(rename = "i", borrow)]
    pub trade_id: &'a str, // ID сделки
}

/// Парсит сырое JSON-сообщение от Bybit в список PublicTrade.
/// Возвращает None, если сообщение не относится к публичным сделкам.
pub fn parse_public_trade_msg<'a>(json_str: &'a str) -> Result<Option<Vec<crate::data::types::PublicTrade<'a>>>> {
    // Быстрая фильтрация: если это не сообщение publicTrade, пропускаем
    if !json_str.contains("publicTrade") {
        return Ok(None);
    }

    // Используем from_slice для работы напрямую с байтами (zero-copy)
    let msg: BybitPublicTradeMsg<'a> = serde_json::from_slice(json_str.as_bytes())
        .context("Failed to deserialize Bybit publicTrade message")?;

    let mut trades = Vec::new();
    for data in msg.data {
        let side = match data.side {
            "Buy" => Side::Buy,
            "Sell" => Side::Sell,
            _ => continue, // Пропускаем неизвестные стороны
        };

        trades.push(crate::data::types::PublicTrade {
            symbol: data.symbol,
            price: parse_f64_fast(data.price)?,
            size: parse_f64_fast(data.size)?,
            side,
            timestamp: data.timestamp,
        });
    }

    Ok(Some(trades))
}

/// Структура для парсинга сообщения о тикере от Bybit
#[derive(Debug, Deserialize)]
pub struct BybitTickerMsg<'a> {
    #[serde(borrow)]
    pub topic: &'a str,
    #[serde(rename = "type", borrow)]
    pub msg_type: &'a str,
    pub ts: u64,
    #[serde(borrow)]
    pub data: BybitTickerData<'a>,
}

/// Данные тикера от Bybit (V5)
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BybitTickerData<'a> {
    #[serde(borrow)]
    pub symbol: &'a str,
    pub last_price: Option<&'a str>,
    pub bid1_price: Option<&'a str>,
    pub bid1_size: Option<&'a str>,
    pub ask1_price: Option<&'a str>,
    pub ask1_size: Option<&'a str>,
    pub volume24h: Option<&'a str>,
    pub turnover24h: Option<&'a str>,
    pub funding_rate: Option<&'a str>,
    pub next_funding_time: Option<&'a str>, // В V5 может приходить строкой
    pub mark_price: Option<&'a str>, // Задача 233: Маркированная цена внутри тикера
}

/// Структура для парсинга сообщения о маркированной цене от Bybit (Задача 233)
#[derive(Debug, Deserialize)]
pub struct BybitMarkPriceMsg<'a> {
    #[serde(borrow)]
    pub topic: &'a str,
    #[serde(rename = "type", borrow)]
    pub msg_type: &'a str,
    pub ts: u64,
    #[serde(borrow)]
    pub data: BybitMarkPriceData<'a>,
}

/// Данные маркированной цены от Bybit (Задача 233)
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BybitMarkPriceData<'a> {
    #[serde(borrow)]
    pub symbol: &'a str,
    pub mark_price: &'a str,
}

/// Парсит сырое JSON-сообщение от Bybit в Ticker.
/// Возвращает None, если сообщение не относится к тикерам.
pub fn parse_ticker_msg<'a>(json_str: &'a str) -> Result<Option<crate::data::types::Ticker<'a>>> {
    // Быстрая фильтрация: если это не сообщение tickers, пропускаем
    if !json_str.contains("tickers") {
        return Ok(None);
    }

    // Используем from_slice для работы напрямую с байтами (zero-copy)
    let msg: BybitTickerMsg<'a> = serde_json::from_slice(json_str.as_bytes())
        .context("Failed to deserialize Bybit ticker message")?;

    let ticker = crate::data::types::Ticker {
        symbol: msg.data.symbol,
        last_price: msg.data.last_price.and_then(|p| parse_f64_fast(p).ok()).unwrap_or(0.0),
        bid: msg.data.bid1_price.and_then(|p| parse_f64_fast(p).ok()).unwrap_or(0.0),
        ask: msg.data.ask1_price.and_then(|p| parse_f64_fast(p).ok()).unwrap_or(0.0),
        bid_size: msg.data.bid1_size.and_then(|p| parse_f64_fast(p).ok()).unwrap_or(0.0),
        ask_size: msg.data.ask1_size.and_then(|p| parse_f64_fast(p).ok()).unwrap_or(0.0),
        volume_24h: msg.data.volume24h.and_then(|p| parse_f64_fast(p).ok()).unwrap_or(0.0),
        turnover_24h: msg.data.turnover24h.and_then(|p| parse_f64_fast(p).ok()).unwrap_or(0.0),
        funding_rate: msg.data.funding_rate.and_then(|p| parse_f64_fast(p).ok()).unwrap_or(0.0),
        next_funding_time: msg.data.next_funding_time.and_then(|t| t.parse::<u64>().ok()).unwrap_or(0),
        timestamp_ms: msg.ts,
        mark_price: msg.data.mark_price.and_then(|p| parse_f64_fast(p).ok()),
    };

    Ok(Some(ticker))
}

/// Парсит сырое JSON-сообщение от Bybit в маркированную цену (Задача 233).
/// Возвращает None, если сообщение не относится к markPrice.
pub fn parse_mark_price_msg<'a>(json_str: &'a str) -> Result<Option<(&'a str, f64)>> {
    // Быстрая фильтрация: если это не сообщение markPrice, пропускаем
    if !json_str.contains("markPrice") {
        return Ok(None);
    }

    // Используем from_slice для работы напрямую с байтами (zero-copy)
    let msg: BybitMarkPriceMsg<'a> = serde_json::from_slice(json_str.as_bytes())
        .context("Failed to deserialize Bybit markPrice message")?;

    let mark_price = parse_f64_fast(msg.data.mark_price)?;
    Ok(Some((msg.data.symbol, mark_price)))
}
