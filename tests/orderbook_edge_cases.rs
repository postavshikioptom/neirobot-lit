mod common;

use common::ws_mock::WsMockServer;
use neirobot_lit::data::websocket::{BybitWsClient, ReconnectSignal};
use neirobot_lit::data::types::{OrderBookUpdate, WsData};
use neirobot_lit::config::types::{ExchangeConfig, WebSocketConfig};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use serde_json::json;
use tokio::time::{timeout, Duration};

fn create_test_config(ws_url: String) -> ExchangeConfig {
    ExchangeConfig {
        name: "bybit".to_string(),
        websocket: WebSocketConfig {
            public_url: ws_url.clone(),
            private_ws_url: ws_url,
            max_attempts: Some(3),
            warn_rtt_ms: 100,
            chaos: None,
        },
        ws_ping_interval_secs: 10,
        ws_pong_timeout_secs: 20,
        ws_retry_initial_ms: 10,
        ws_retry_max_ms: 100,
        ws_retry_multiplier: 2.0,
        ws_retry_jitter: 0.1,
        ..Default::default()
    }
}

#[tokio::test]
async fn test_crossed_orderbook_handling() {
    let server = WsMockServer::new().await;
    let config = create_test_config(server.url());
    
    let (tx, mut rx) = mpsc::channel::<WsData>(100);
    let client = BybitWsClient::new(config, "BTCUSDT".to_string());
    let token = CancellationToken::new();
    let (_reconnect_tx, reconnect_rx) = mpsc::channel(1);
    
    let _client_handle = tokio::spawn(async move {
        client.run(tx, reconnect_rx, token).await
    });
    
    tokio::time::sleep(Duration::from_millis(200)).await;
    
    // Отправляем Crossed Orderbook: Best Bid (50100) > Best Ask (50000)
    let crossed_snapshot = json!({
        "topic": "orderbook.50.BTCUSDT",
        "type": "snapshot",
        "ts": 1000,
        "data": {
            "s": "BTCUSDT",
            "b": [["50100.0", "1.0"]],
            "a": [["50000.0", "1.0"]],
            "u": 100,
            "seq": 1
        }
    });
    
    server.send_json(crossed_snapshot).await.unwrap();
    
    // Ожидаем, что клиент все же передаст данные
    let data = timeout(Duration::from_secs(1), rx.recv()).await.unwrap().unwrap();
    if let WsData::OrderBook(update) = data {
        assert_eq!(update.bids[0].price, 50100.0);
        assert_eq!(update.asks[0].price, 50000.0);
        assert!(update.bids[0].price > update.asks[0].price, "Should receive crossed prices as sent");
    } else {
        panic!("Expected OrderBook data");
    }
}

#[tokio::test]
async fn test_empty_orderbook_handling() {
    let server = WsMockServer::new().await;
    let config = create_test_config(server.url());
    
    let (tx, mut rx) = mpsc::channel::<WsData>(100);
    let client = BybitWsClient::new(config, "BTCUSDT".to_string());
    let token = CancellationToken::new();
    let (_reconnect_tx, reconnect_rx) = mpsc::channel(1);
    
    let _client_handle = tokio::spawn(async move {
        client.run(tx, reconnect_rx, token).await
    });
    
    tokio::time::sleep(Duration::from_millis(200)).await;
    
    // Отправляем стакан с пустой стороной Bids
    let empty_bids = json!({
        "topic": "orderbook.50.BTCUSDT",
        "type": "snapshot",
        "ts": 1000,
        "data": {
            "s": "BTCUSDT",
            "b": [],
            "a": [["50000.0", "1.0"]],
            "u": 100,
            "seq": 1
        }
    });
    
    server.send_json(empty_bids).await.unwrap();
    
    let data = timeout(Duration::from_secs(1), rx.recv()).await.unwrap().unwrap();
    if let WsData::OrderBook(update) = data {
        assert!(update.bids.is_empty());
        assert_eq!(update.asks[0].price, 50000.0);
    } else {
        panic!("Expected OrderBook data");
    }
}
