# 019 - Data WebSocket Parser

Цель задачи: Реализовать модуль src/data/parser.rs для трансформации сырых JSON-сообщений от Bybit в типизированные структуры OrderBookUpdate. Модуль должен фильтровать сервисные сообщения биржи, корректно парсить строковые значения цен в f64 и подготавливать данные для обновления OrderBook.

Файлы для изменения/создания:

src/data/mod.rs (обновить)
src/data/parser.rs (создать)
src/data/websocket.rs (обновить для использования парсера)
Инструкции для Gemini:

src/data/parser.rs: Определить DTO-структуры, соответствующие формату Bybit V5 Public Orderbook.

use serde::Deserialize;
use crate::data::types::{OrderBookUpdate, PriceLevel};
use anyhow::{Result, Context};

#[derive(Debug, Deserialize)]
pub struct BybitOrderbookMsg {
    #[serde(rename = "type")]
    pub msg_type: String,
    pub topic: String,
    pub ts: u64,
    pub data: BybitData,
}

#[derive(Debug, Deserialize)]
pub struct BybitData {
    pub s: String,      // symbol
    pub b: Vec<Vec<String>>, // bids [price, size]
    pub a: Vec<Vec<String>>, // asks [price, size]
    pub u: u64,         // update id (u)
    pub seq: Option<u64>, // sequence (seq) - бывает в снапшотах
    pub cs: Option<u64>,  // checksum (cs) - для будущих проверок
}

pub fn parse_orderbook_msg(json_str: &str) -> Result<Option<OrderBookUpdate>> {
    // 1. Быстрая фильтрация: если это не сообщение стакана, пропускаем
    if !json_str.contains("orderbook") {
        return Ok(None);
    }

    let msg: BybitOrderbookMsg = serde_json::from_str(json_str)
        .context("Failed to deserialize Bybit orderbook message")?;

    // 2. Конвертация строк в f64
    let parse_levels = |levels: &Vec<Vec<String>>| -> Result<Vec<PriceLevel>> {
        levels.iter()
            .map(|pair| {
                Ok(PriceLevel {
                    price: pair[0].parse::<f64>().context("Price parse error")?,
                    size: pair[1].parse::<f64>().context("Size parse error")?,
                })
            })
            .collect()
    };

    Ok(Some(OrderBookUpdate {
        symbol: msg.data.s,
        timestamp_ms: msg.ts,
        last_update_id: msg.data.u,
        is_snapshot: msg.msg_type == "snapshot",
        bids: parse_levels(&msg.data.b)?,
        asks: parse_levels(&msg.data.a)?,
    }))
}
src/data/websocket.rs: Обновить цикл обработки сообщений, используя новую функцию парсинга.

// Внутри цикла ws.next()
if let Some(Ok(Message::Text(text))) = msg {
    match crate::data::parser::parse_orderbook_msg(&text) {
        Ok(Some(update)) => {
            // Здесь в будущем будет вызов apply_update или отправка дальше
            info!("Parsed update for {}, ID: {}", update.symbol, update.last_update_id);
        }
        Ok(None) => {} // Пропускаем сервисные сообщения
        Err(e) => warn!("Parser error: {}", e), // Логируем, но не падаем
    }
}
Технические требования:

Изоляция: Весь код парсинга вынесен в parser.rs.
Отказоустойчивость: Ошибки парсинга конкретного сообщения не должны приводить к разрыву WebSocket-соединения. Используйте warn! и продолжайте цикл.
DTO: Сохранить поля seq и cs как Option, они понадобятся для задач по валидации данных (049).
Почему это важно: Bybit присылает данные в текстовом формате для обеспечения точности (чтобы избежать проблем с floating point в JSON). Парсинг их на раннем этапе в Rust позволяет нам работать с числами во всей остальной системе, что на порядки быстрее.