use serde::Deserialize;
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
pub fn parse_orderbook_msg(json_str: &str) -> Result<Option<crate::data::types::OrderBookUpdateOwned>> {
    // 1. Быстрая фильтрация: если это не сообщение стакана, пропускаем
    if !json_str.contains("orderbook") {
        return Ok(None);
    }

    // Используем from_slice для работы напрямую с байтами (zero-copy)
    let msg: BybitOrderbookMsg = serde_json::from_slice(json_str.as_bytes())
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

    Ok(Some(crate::data::types::OrderBookUpdateOwned {
        symbol: msg.data.s.to_string(),
        timestamp_ms: msg.ts,
        last_update_id: msg.data.u,
        is_snapshot: msg.msg_type == "snapshot",
        bids: parse_levels(&msg.data.b)?.into(),
        asks: parse_levels(&msg.data.a)?.into(),
        checksum: msg.data.cs.map(|c| c as u32),
    }))
}

use crate::data::types::{PublicTradeOwned, Side};

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
pub fn parse_public_trade_msg(json_str: &str) -> Result<Option<Vec<PublicTradeOwned>>> {
    // Быстрая фильтрация: если это не сообщение publicTrade, пропускаем
    if !json_str.contains("publicTrade") {
        return Ok(None);
    }

    // Используем from_slice для работы напрямую с байтами (zero-copy)
    let msg: BybitPublicTradeMsg = serde_json::from_slice(json_str.as_bytes())
        .context("Failed to deserialize Bybit publicTrade message")?;

    let mut trades = Vec::new();
    for data in msg.data {
        let side = match data.side {
            "Buy" => Side::Buy,
            "Sell" => Side::Sell,
            _ => continue, // Пропускаем неизвестные стороны
        };

        trades.push(PublicTradeOwned {
            price: Decimal::from_str(data.price).context("Failed to parse price")?,
            size: Decimal::from_str(data.size).context("Failed to parse size")?,
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

/// Данные тикера от Bybit
#[derive(Debug, Deserialize)]
pub struct BybitTickerData<'a> {
    #[serde(rename = "s", borrow)]
    pub symbol: &'a str,
    #[serde(rename = "lp", borrow)]
    pub last_price: &'a str,
    #[serde(rename = "bp", borrow)]
    pub bid_price: &'a str,
    #[serde(rename = "bv", borrow)]
    pub bid_size: &'a str,
    #[serde(rename = "ap", borrow)]
    pub ask_price: &'a str,
    #[serde(rename = "av", borrow)]
    pub ask_size: &'a str,
    #[serde(rename = "v24h", borrow)]
    pub volume_24h: &'a str,
    #[serde(rename = "t24h", borrow)]
    pub turnover_24h: &'a str,
    #[serde(rename = "fr", borrow)]
    pub funding_rate: &'a str,
    #[serde(rename = "nft")]
    pub next_funding_time: u64,
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
pub struct BybitMarkPriceData<'a> {
    #[serde(rename = "s", borrow)]
    pub symbol: &'a str,
    #[serde(rename = "p", borrow)]
    pub mark_price: &'a str,
}

/// Парсит сырое JSON-сообщение от Bybit в Ticker.
/// Возвращает None, если сообщение не относится к тикерам.
pub fn parse_ticker_msg(json_str: &str) -> Result<Option<crate::data::types::TickerOwned>> {
    // Быстрая фильтрация: если это не сообщение tickers, пропускаем
    if !json_str.contains("tickers") {
        return Ok(None);
    }

    // Используем from_slice для работы напрямую с байтами (zero-copy)
    let msg: BybitTickerMsg = serde_json::from_slice(json_str.as_bytes())
        .context("Failed to deserialize Bybit ticker message")?;

    let ticker = crate::data::types::TickerOwned {
        symbol: msg.data.symbol.to_string(),
        last_price: parse_f64_fast(msg.data.last_price)?,
        bid: parse_f64_fast(msg.data.bid_price)?,
        ask: parse_f64_fast(msg.data.ask_price)?,
        bid_size: parse_f64_fast(msg.data.bid_size)?,
        ask_size: parse_f64_fast(msg.data.ask_size)?,
        volume_24h: parse_f64_fast(msg.data.volume_24h)?,
        turnover_24h: parse_f64_fast(msg.data.turnover_24h)?,
        funding_rate: parse_f64_fast(msg.data.funding_rate)?,
        next_funding_time: msg.data.next_funding_time,
        timestamp_ms: msg.ts,
    };

    Ok(Some(ticker))
}

/// Парсит сырое JSON-сообщение от Bybit в маркированную цену (Задача 233).
/// Возвращает None, если сообщение не относится к markPrice.
pub fn parse_mark_price_msg(json_str: &str) -> Result<Option<(String, f64)>> {
    // Быстрая фильтрация: если это не сообщение markPrice, пропускаем
    if !json_str.contains("markPrice") {
        return Ok(None);
    }

    // Используем from_slice для работы напрямую с байтами (zero-copy)
    let msg: BybitMarkPriceMsg = serde_json::from_slice(json_str.as_bytes())
        .context("Failed to deserialize Bybit markPrice message")?;

    let mark_price = parse_f64_fast(msg.data.mark_price)?;
    Ok(Some((msg.data.symbol.to_string(), mark_price)))
}
