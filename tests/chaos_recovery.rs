mod common;

use common::ws_mock::WsMockServer;
use neirobot_lit::data::websocket::{BybitWsClient, ReconnectSignal};
use neirobot_lit::data::types::{OrderBookUpdate, WsData};
use neirobot_lit::config::types::{ExchangeConfig, WebSocketConfig};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use serde_json::json;
use tokio::time::{timeout, Duration, Instant};

fn create_test_config(ws_url: String) -> ExchangeConfig {
    ExchangeConfig {
        name: "bybit".to_string(),
        websocket: WebSocketConfig {
            public_url: ws_url.clone(),
            private_ws_url: ws_url,
            max_attempts: Some(5),
            warn_rtt_ms: 100,
            chaos: None,
        },
        ws_ping_interval_secs: 1,
        ws_pong_timeout_secs: 2,
        ws_retry_initial_ms: 10,
        ws_retry_max_ms: 100,
        ws_retry_multiplier: 2.0,
        ws_retry_jitter: 0.1,
        ..Default::default()
    }
}

#[tokio::test]
async fn test_network_delay_recovery() {
    // Включаем фичу chaos через переменные окружения, если это возможно в тестах
    // Но в коде BybitWsClient inject_chaos вызывается только под #[cfg(feature = "chaos")]
    
    let server = WsMockServer::new().await;
    let config = create_test_config(server.url());
    
    let (tx, mut rx) = mpsc::channel::<WsData>(100);
    let client = BybitWsClient::new(config, "BTCUSDT".to_string());
    let token = CancellationToken::new();
    let token_clone = token.clone();
    let (_reconnect_tx, reconnect_rx) = mpsc::channel(1);
    
    let _client_handle = tokio::spawn(async move {
        client.run(tx, reconnect_rx, token_clone).await
    });
    
    // Ждем подключения
    tokio::time::sleep(Duration::from_millis(200)).await;
    
    // Отправляем снимок
    let snapshot = json!({
        "topic": "orderbook.50.BTCUSDT",
        "type": "snapshot",
        "ts": 1000,
        "data": {
            "s": "BTCUSDT",
            "b": [["50000.0", "1.0"]],
            "a": [["50100.0", "1.0"]],
            "u": 100,
            "seq": 1
        }
    });
    server.send_json(snapshot).await.unwrap();
    
    // Проверяем получение снимка
    let data = timeout(Duration::from_secs(2), rx.recv()).await.unwrap().unwrap();
    if let WsData::OrderBook(update) = data {
        assert_eq!(update.last_update_id, 100);
    } else {
        panic!("Expected OrderBook data");
    }
    
    // Имитируем задержку в коде клиента (через chaos feature)
    // Если тест запущен без --features chaos, задержки не будет
    let start = Instant::now();
    
    let delta = json!({
        "topic": "orderbook.50.BTCUSDT",
        "type": "delta",
        "ts": 1100,
        "data": {
            "s": "BTCUSDT",
            "b": [["50001.0", "1.1"]],
            "a": [["50101.0", "1.1"]],
            "u": 101,
            "seq": 2
        }
    });
    server.send_json(delta).await.unwrap();
    
    let data = timeout(Duration::from_secs(5), rx.recv()).await.unwrap().unwrap();
    if let WsData::OrderBook(update) = data {
        assert_eq!(update.last_update_id, 101);
    }
    
    let elapsed = start.elapsed();
    println!("Elapsed for delta with potential chaos: {:?}", elapsed);
    
    token.cancel();
}

#[tokio::test]
async fn test_packet_loss_reconnect() {
    let server = WsMockServer::new().await;
    let config = create_test_config(server.url());
    
    let (tx, mut rx) = mpsc::channel::<WsData>(100);
    let client = BybitWsClient::new(config, "BTCUSDT".to_string());
    let token = CancellationToken::new();
    let token_clone = token.clone();
    let (_reconnect_tx, reconnect_rx) = mpsc::channel(1);
    
    let _client_handle = tokio::spawn(async move {
        client.run(tx, reconnect_rx, token_clone).await
    });
    
    tokio::time::sleep(Duration::from_millis(200)).await;
    
    // 1. Отправляем снимок u=100
    server.send_json(json!({
        "topic": "orderbook.50.BTCUSDT",
        "type": "snapshot",
        "ts": 1000,
        "data": {
            "s": "BTCUSDT",
            "b": [["50000.0", "1.0"]],
            "a": [["50100.0", "1.0"]],
            "u": 100,
            "seq": 1
        }
    })).await.unwrap();
    
    let _ = timeout(Duration::from_secs(1), rx.recv()).await.unwrap().unwrap();
    
    // 2. Отправляем дельту u=101
    server.send_json(json!({
        "topic": "orderbook.50.BTCUSDT",
        "type": "delta",
        "ts": 1100,
        "data": { "s": "BTCUSDT", "b": [], "a": [], "u": 101, "seq": 2 }
    })).await.unwrap();
    
    let data = timeout(Duration::from_secs(1), rx.recv()).await.unwrap().unwrap();
    if let WsData::OrderBook(update) = data {
        assert_eq!(update.last_update_id, 101);
    }
    
    // 3. Имитируем потерю пакета: отправляем u=103 вместо u=102
    // Клиент должен увидеть гэп (101 -> 103) и инициировать реконнект
    server.send_json(json!({
        "topic": "orderbook.50.BTCUSDT",
        "type": "delta",
        "ts": 1200,
        "data": { "s": "BTCUSDT", "b": [], "a": [], "u": 103, "seq": 3 }
    })).await.unwrap();
    
    // После обнаружения гэпа клиент должен переподключиться и снова получить данные
    // Ждем, пока клиент сделает реконнект и подпишется заново
    tokio::time::sleep(Duration::from_millis(500)).await;
    
    // Отправляем новый снимок после реконнекта
    server.send_json(json!({
        "topic": "orderbook.50.BTCUSDT",
        "type": "snapshot",
        "ts": 2000,
        "data": {
            "s": "BTCUSDT",
            "b": [["50000.0", "2.0"]],
            "a": [["50100.0", "2.0"]],
            "u": 200,
            "seq": 10
        }
    })).await.unwrap();
    
    let data = timeout(Duration::from_secs(5), rx.recv()).await.unwrap().unwrap();
    if let WsData::OrderBook(update) = data {
        assert_eq!(update.last_update_id, 200);
    }
    
    token.cancel();
}
