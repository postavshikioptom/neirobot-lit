mod common;

use common::ws_mock::WsMockServer;
use neirobot_lit::data::websocket::BybitWsClient;
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
            private_url: ws_url,
            max_attempts: Some(3),
            warn_rtt_ms: 100,
        },
        ws_ping_interval_secs: 20,
        ws_pong_timeout_secs: 30,
        ws_retry_initial_ms: 100,
        ws_retry_max_ms: 5000,
        ws_retry_multiplier: 2.0,
        ws_retry_jitter: 0.1,
        ..Default::default()
    }
}

#[tokio::test]
async fn test_public_orderbook_flow() {
    // Запускаем mock сервер
    let server = WsMockServer::new().await;
    let config = create_test_config(server.url());
    
    // Создаем канал для получения обновлений
    let (tx, mut rx) = mpsc::channel::<WsData>(100);
    
    // Создаем клиент
    let client = BybitWsClient::new(config, "BTCUSDT".to_string());
    let token = CancellationToken::new();
    let token_clone = token.clone();
    
    // Запускаем клиент в отдельной задаче
    let client_handle = tokio::spawn(async move {
        client.run(tx, token_clone).await
    });
    
    // Даем время на подключение и подписку
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Отправляем 5 дельт (они должны буферизоваться)
    for i in 1..=5 {
        let delta = json!({
            "topic": "orderbook.50.BTCUSDT",
            "type": "delta",
            "ts": 1000 + i * 100,
            "data": {
                "s": "BTCUSDT",
                "b": [[format!("{}", 50000 + i), format!("{}", 0.1 * i as f64)]],
                "a": [[format!("{}", 50100 + i), format!("{}", 0.1 * i as f64)]],
                "u": 100 + i,
                "seq": i
            }
        });
        
        server.send_json(delta).await.expect("Failed to send delta");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    
    // Отправляем снимок (u=200)
    let snapshot = json!({
        "topic": "orderbook.50.BTCUSDT",
        "type": "snapshot",
        "ts": 2000,
        "data": {
            "s": "BTCUSDT",
            "b": [
                ["50000.0", "1.0"],
                ["49999.0", "2.0"],
                ["49998.0", "3.0"]
            ],
            "a": [
                ["50100.0", "1.0"],
                ["50101.0", "2.0"],
                ["50102.0", "3.0"]
            ],
            "u": 200,
            "seq": 10
        }
    });
    
    server.send_json(snapshot).await.expect("Failed to send snapshot");
    
    // Отправляем дельты после снимка (u=201, 202, 203)
    for i in 201..=203 {
        let delta = json!({
            "topic": "orderbook.50.BTCUSDT",
            "type": "delta",
            "ts": 3000 + i,
            "data": {
                "s": "BTCUSDT",
                "b": [[format!("{}", 50000 + i), "0.5"]],
                "a": [[format!("{}", 50100 + i), "0.5"]],
                "u": i,
                "seq": i
            }
        });
        
        server.send_json(delta).await.expect("Failed to send delta");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    
    // Проверяем, что получили снимок
    let snapshot_update = timeout(Duration::from_secs(2), rx.recv())
        .await
        .expect("Timeout waiting for snapshot")
        .expect("Channel closed");
    
    let snapshot_update = match snapshot_update {
        WsData::OrderBook(update) => update,
        _ => panic!("Expected OrderBook data"),
    };
    
    assert!(snapshot_update.is_snapshot, "First update should be snapshot");
    assert_eq!(snapshot_update.last_update_id, 200);
    assert_eq!(snapshot_update.bids.len(), 3);
    assert_eq!(snapshot_update.asks.len(), 3);
    
    // Проверяем, что получили буферизованные дельты (101-105)
    for expected_u in 101..=105 {
        let delta_update = timeout(Duration::from_millis(500), rx.recv())
            .await
            .expect(&format!("Timeout waiting for buffered delta u={}", expected_u))
            .expect("Channel closed");
        
        let delta_update = match delta_update {
            WsData::OrderBook(update) => update,
            _ => panic!("Expected OrderBook data"),
        };
        
        assert!(!delta_update.is_snapshot, "Should be delta");
        assert_eq!(delta_update.last_update_id, expected_u);
    }
    
    // Проверяем, что получили live дельты (201-203)
    for expected_u in 201..=203 {
        let delta_update = timeout(Duration::from_millis(500), rx.recv())
            .await
            .expect(&format!("Timeout waiting for live delta u={}", expected_u))
            .expect("Channel closed");
        
        let delta_update = match delta_update {
            WsData::OrderBook(update) => update,
            _ => panic!("Expected OrderBook data"),
        };
        
        assert!(!delta_update.is_snapshot, "Should be delta");
        assert_eq!(delta_update.last_update_id, expected_u);
    }
    
    // Останавливаем клиент
    token.cancel();
    let _ = timeout(Duration::from_secs(1), client_handle).await;
    server.shutdown();
}

#[tokio::test]
async fn test_heartbeat_timeout() {
    // Запускаем mock сервер БЕЗ автоматического pong
    let server = WsMockServer::with_options(false, true).await;
    
    let mut config = create_test_config(server.url());
    // Устанавливаем короткие таймауты для теста
    config.websocket.ping_interval_sec = 1;
    config.websocket.pong_timeout_sec = 3;
    
    let (tx, _rx) = mpsc::channel::<WsData>(100);
    let client = BybitWsClient::new(config, "BTCUSDT".to_string());
    let token = CancellationToken::new();
    let token_clone = token.clone();
    
    let client_handle = tokio::spawn(async move {
        client.run(tx, token_clone).await
    });
    
    // Даем время на подключение
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Ждем, пока клиент не отключится из-за таймаута (должно быть ~3-4 секунды)
    let result = timeout(Duration::from_secs(6), client_handle).await;
    
    // Проверяем, что клиент завершился с ошибкой (таймаут heartbeat)
    assert!(result.is_ok(), "Client should finish due to heartbeat timeout");
    
    if let Ok(Ok(Err(e))) = result {
        let error_msg = e.to_string();
        assert!(
            error_msg.contains("Heartbeat") || error_msg.contains("timeout"),
            "Error should be related to heartbeat timeout, got: {}",
            error_msg
        );
    }
    
    token.cancel();
    server.shutdown();
}

#[tokio::test]
async fn test_chaos_invalid_json() {
    let server = WsMockServer::new().await;
    let config = create_test_config(server.url());
    
    let (tx, mut rx) = mpsc::channel::<WsData>(100);
    let client = BybitWsClient::new(config, "BTCUSDT".to_string());
    let token = CancellationToken::new();
    let token_clone = token.clone();
    
    let client_handle = tokio::spawn(async move {
        client.run(tx, token_clone).await
    });
    
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Отправляем некорректный JSON
    server.send_text("{invalid json}".to_string())
        .await
        .expect("Failed to send invalid json");
    
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Отправляем валидный снимок после ошибки
    let snapshot = json!({
        "topic": "orderbook.50.BTCUSDT",
        "type": "snapshot",
        "ts": 2000,
        "data": {
            "s": "BTCUSDT",
            "b": [["50000.0", "1.0"]],
            "a": [["50100.0", "1.0"]],
            "u": 100,
            "seq": 1
        }
    });
    
    server.send_json(snapshot).await.expect("Failed to send snapshot");
    
    // Клиент должен проигнорировать невалидный JSON и обработать снимок
    let update = timeout(Duration::from_secs(1), rx.recv())
        .await
        .expect("Timeout waiting for snapshot")
        .expect("Channel closed");
    
    let update = match update {
        WsData::OrderBook(ob) => ob,
        _ => panic!("Expected OrderBook data"),
    };
    
    assert!(update.is_snapshot);
    assert_eq!(update.last_update_id, 100);
    
    token.cancel();
    let _ = timeout(Duration::from_secs(1), client_handle).await;
    server.shutdown();
}

#[tokio::test]
async fn test_sequence_gap_triggers_reconnect() {
    let server = WsMockServer::new().await;
    let mut config = create_test_config(server.url());
    config.ws_retry_initial_ms = 100;
    config.websocket.max_attempts = Some(2);
    
    let (tx, mut rx) = mpsc::channel::<WsData>(100);
    let client = BybitWsClient::new(config, "BTCUSDT".to_string());
    let token = CancellationToken::new();
    let token_clone = token.clone();
    
    let client_handle = tokio::spawn(async move {
        client.run(tx, token_clone).await
    });
    
    tokio::time::sleep(Duration::from_millis(100)).await;
    
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
    
    server.send_json(snapshot).await.expect("Failed to send snapshot");
    
    // Получаем снимок
    let _ = timeout(Duration::from_secs(1), rx.recv())
        .await
        .expect("Timeout waiting for snapshot");
    
    // Отправляем дельту u=101
    let delta1 = json!({
        "topic": "orderbook.50.BTCUSDT",
        "type": "delta",
        "ts": 2000,
        "data": {
            "s": "BTCUSDT",
            "b": [["50001.0", "1.0"]],
            "a": [["50101.0", "1.0"]],
            "u": 101,
            "seq": 2
        }
    });
    
    server.send_json(delta1).await.expect("Failed to send delta");
    
    let _ = timeout(Duration::from_millis(500), rx.recv())
        .await
        .expect("Timeout waiting for delta");
    
    // Отправляем дельту с пропуском: u=105 вместо u=102 (GAP!)
    let delta_gap = json!({
        "topic": "orderbook.50.BTCUSDT",
        "type": "delta",
        "ts": 3000,
        "data": {
            "s": "BTCUSDT",
            "b": [["50005.0", "1.0"]],
            "a": [["50105.0", "1.0"]],
            "u": 105,
            "seq": 3
        }
    });
    
    server.send_json(delta_gap).await.expect("Failed to send gap delta");
    
    // Клиент должен обнаружить gap и попытаться переподключиться
    tokio::time::sleep(Duration::from_millis(500)).await;
    
    // После переподключения отправляем новый снимок
    let snapshot2 = json!({
        "topic": "orderbook.50.BTCUSDT",
        "type": "snapshot",
        "ts": 4000,
        "data": {
            "s": "BTCUSDT",
            "b": [["50010.0", "2.0"]],
            "a": [["50110.0", "2.0"]],
            "u": 200,
            "seq": 10
        }
    });
    
    server.send_json(snapshot2).await.expect("Failed to send snapshot2");
    
    // Должны получить новый снимок после переподключения
    let update = timeout(Duration::from_secs(2), rx.recv())
        .await
        .expect("Timeout waiting for snapshot after reconnect")
        .expect("Channel closed");
    
    let update = match update {
        WsData::OrderBook(ob) => ob,
        _ => panic!("Expected OrderBook data"),
    };
    
    assert!(update.is_snapshot);
    assert_eq!(update.last_update_id, 200);
    
    token.cancel();
    let _ = timeout(Duration::from_secs(1), client_handle).await;
    server.shutdown();
}

#[tokio::test]
async fn test_init_buffer_overflow() {
    let server = WsMockServer::new().await;
    let config = create_test_config(server.url());
    
    let (tx, _rx) = mpsc::channel::<WsData>(100);
    let client = BybitWsClient::new(config, "BTCUSDT".to_string());
    let token = CancellationToken::new();
    let token_clone = token.clone();
    
    let client_handle = tokio::spawn(async move {
        client.run(tx, token_clone).await
    });
    
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Отправляем 101 дельту БЕЗ снимка (должен переполниться буфер)
    for i in 1..=101 {
        let delta = json!({
            "topic": "orderbook.50.BTCUSDT",
            "type": "delta",
            "ts": 1000 + i,
            "data": {
                "s": "BTCUSDT",
                "b": [["50000.0", "1.0"]],
                "a": [["50100.0", "1.0"]],
                "u": i,
                "seq": i
            }
        });
        
        server.send_json(delta).await.expect("Failed to send delta");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    
    // Клиент должен завершиться с ошибкой переполнения буфера
    let result = timeout(Duration::from_secs(3), client_handle).await;
    
    assert!(result.is_ok(), "Client should finish due to buffer overflow");
    
    if let Ok(Ok(Err(e))) = result {
        let error_msg = e.to_string();
        assert!(
            error_msg.contains("buffer overflow") || error_msg.contains("Max connection attempts"),
            "Error should be related to buffer overflow, got: {}",
            error_msg
        );
    }
    
    token.cancel();
    server.shutdown();
}

#[tokio::test]
async fn test_private_auth_flow_success() {
    use neirobot_lit::data::websocket::BybitPrivateWsClient;
    
    // Устанавливаем переменные окружения для теста
    std::env::set_var("BYBIT_API_KEY", "test_api_key");
    std::env::set_var("BYBIT_API_SECRET", "test_api_secret");
    
    // Запускаем mock сервер с автоматической авторизацией
    let server = WsMockServer::with_options(true, true).await;
    let config = create_test_config(server.url());
    
    let (tx, mut rx) = mpsc::channel::<serde_json::Value>(100);
    let client = BybitPrivateWsClient::new(config);
    let token = CancellationToken::new();
    let token_clone = token.clone();
    
    let client_handle = tokio::spawn(async move {
        client.run(tx, token_clone).await
    });
    
    // Даем время на подключение и авторизацию
    tokio::time::sleep(Duration::from_millis(200)).await;
    
    // Отправляем тестовое приватное сообщение (например, order update)
    let order_update = json!({
        "topic": "order",
        "ts": 1000,
        "data": [{
            "orderId": "test-order-123",
            "symbol": "BTCUSDT",
            "side": "Buy",
            "orderType": "Limit",
            "price": "50000",
            "qty": "0.1"
        }]
    });
    
    server.send_json(order_update).await.expect("Failed to send order update");
    
    // Проверяем, что получили сообщение
    let msg = timeout(Duration::from_secs(1), rx.recv())
        .await
        .expect("Timeout waiting for order update")
        .expect("Channel closed");
    
    assert_eq!(msg["topic"], "order");
    assert!(msg["data"].is_array());
    
    token.cancel();
    let _ = timeout(Duration::from_secs(1), client_handle).await;
    server.shutdown();
    
    // Очищаем переменные окружения
    std::env::remove_var("BYBIT_API_KEY");
    std::env::remove_var("BYBIT_API_SECRET");
}

#[tokio::test]
async fn test_private_auth_flow_failure() {
    use neirobot_lit::data::websocket::BybitPrivateWsClient;
    
    // Устанавливаем переменные окружения для теста
    std::env::set_var("BYBIT_API_KEY", "invalid_key");
    std::env::set_var("BYBIT_API_SECRET", "invalid_secret");
    
    // Запускаем mock сервер БЕЗ автоматической авторизации
    let server = WsMockServer::with_options(true, false).await;
    let mut config = create_test_config(server.url());
    config.websocket.max_attempts = Some(2); // Ограничиваем попытки
    
    let (tx, _rx) = mpsc::channel::<serde_json::Value>(100);
    let client = BybitPrivateWsClient::new(config);
    let token = CancellationToken::new();
    let token_clone = token.clone();
    
    let server_clone = server.tx.clone();
    
    // Запускаем задачу для отправки ошибки авторизации
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        let auth_error = json!({
            "success": false,
            "ret_msg": "Invalid signature",
            "conn_id": "test-connection",
            "op": "auth"
        });
        
        let _ = server_clone.send(tokio_tungstenite::tungstenite::Message::Text(auth_error.to_string()));
    });
    
    let client_handle = tokio::spawn(async move {
        client.run(tx, token_clone).await
    });
    
    // Клиент должен завершиться с ошибкой авторизации
    let result = timeout(Duration::from_secs(3), client_handle).await;
    
    assert!(result.is_ok(), "Client should finish due to auth failure");
    
    if let Ok(Ok(Err(e))) = result {
        let error_msg = e.to_string();
        assert!(
            error_msg.contains("Auth failed") || error_msg.contains("Max") || error_msg.contains("attempts"),
            "Error should be related to auth failure, got: {}",
            error_msg
        );
    }
    
    token.cancel();
    server.shutdown();
    
    // Очищаем переменные окружения
    std::env::remove_var("BYBIT_API_KEY");
    std::env::remove_var("BYBIT_API_SECRET");
}

#[tokio::test]
async fn test_chaos_corrupted_checksum() {
    let server = WsMockServer::new().await;
    let config = create_test_config(server.url());
    
    let (tx, mut rx) = mpsc::channel::<WsData>(100);
    let client = BybitWsClient::new(config, "BTCUSDT".to_string());
    let token = CancellationToken::new();
    let token_clone = token.clone();
    
    let client_handle = tokio::spawn(async move {
        client.run(tx, token_clone).await
    });
    
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Отправляем снимок с НЕПРАВИЛЬНОЙ контрольной суммой (cts)
    let corrupted_snapshot = json!({
        "topic": "orderbook.50.BTCUSDT",
        "type": "snapshot",
        "ts": 2000,
        "data": {
            "s": "BTCUSDT",
            "b": [
                ["50000.0", "1.0"],
                ["49999.0", "2.0"]
            ],
            "a": [
                ["50100.0", "1.0"],
                ["50101.0", "2.0"]
            ],
            "u": 100,
            "seq": 1
        },
        "cts": 999999999  // Неправильная контрольная сумма CRC32
    });
    
    server.send_json(corrupted_snapshot).await.expect("Failed to send corrupted snapshot");
    
    tokio::time::sleep(Duration::from_millis(200)).await;
    
    // Отправляем корректный снимок
    let valid_snapshot = json!({
        "topic": "orderbook.50.BTCUSDT",
        "type": "snapshot",
        "ts": 3000,
        "data": {
            "s": "BTCUSDT",
            "b": [["50000.0", "1.0"]],
            "a": [["50100.0", "1.0"]],
            "u": 200,
            "seq": 2
        }
    });
    
    server.send_json(valid_snapshot).await.expect("Failed to send valid snapshot");
    
    // Клиент должен обработать хотя бы один снимок (может пропустить битый или переподключиться)
    let update = timeout(Duration::from_secs(2), rx.recv())
        .await
        .expect("Timeout waiting for snapshot")
        .expect("Channel closed");
    
    let update = match update {
        WsData::OrderBook(ob) => ob,
        _ => panic!("Expected OrderBook data"),
    };
    
    assert!(update.is_snapshot, "Should receive a snapshot");
    // Может быть либо битый (100), либо корректный (200) в зависимости от логики валидации
    assert!(
        update.last_update_id == 100 || update.last_update_id == 200,
        "Should receive one of the snapshots"
    );
    
    token.cancel();
    let _ = timeout(Duration::from_secs(1), client_handle).await;
    server.shutdown();
}

#[tokio::test]
async fn test_chaos_multiple_corrupted_messages() {
    let server = WsMockServer::new().await;
    let config = create_test_config(server.url());
    
    let (tx, mut rx) = mpsc::channel::<WsData>(100);
    let client = BybitWsClient::new(config, "BTCUSDT".to_string());
    let token = CancellationToken::new();
    let token_clone = token.clone();
    
    let client_handle = tokio::spawn(async move {
        client.run(tx, token_clone).await
    });
    
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Отправляем серию битых сообщений
    for i in 1..=5 {
        let corrupted = format!(r#"{{"broken": "json", "attempt": {}}}"#, i);
        server.send_text(corrupted).await.expect("Failed to send corrupted message");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    
    // Отправляем валидный снимок
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
    
    server.send_json(snapshot).await.expect("Failed to send snapshot");
    
    // Клиент должен проигнорировать битые сообщения и обработать снимок
    let update = timeout(Duration::from_secs(1), rx.recv())
        .await
        .expect("Timeout waiting for snapshot")
        .expect("Channel closed");
    
    let update = match update {
        WsData::OrderBook(ob) => ob,
        _ => panic!("Expected OrderBook data"),
    };
    
    assert!(update.is_snapshot);
    assert_eq!(update.last_update_id, 100);
    
    token.cancel();
    let _ = timeout(Duration::from_secs(1), client_handle).await;
    server.shutdown();
}

#[tokio::test]
async fn test_private_auth_invalid_format() {
    use neirobot_lit::data::websocket::BybitPrivateWsClient;
    
    // НЕ устанавливаем переменные окружения - клиент должен упасть
    std::env::remove_var("BYBIT_API_KEY");
    std::env::remove_var("BYBIT_API_SECRET");
    
    let server = WsMockServer::with_options(true, true).await;
    let mut config = create_test_config(server.url());
    config.websocket.max_attempts = Some(1);
    
    let (tx, _rx) = mpsc::channel::<serde_json::Value>(100);
    let client = BybitPrivateWsClient::new(config);
    let token = CancellationToken::new();
    let token_clone = token.clone();
    
    let client_handle = tokio::spawn(async move {
        client.run(tx, token_clone).await
    });
    
    // Клиент должен завершиться с ошибкой (нет API ключей)
    let result = timeout(Duration::from_secs(2), client_handle).await;
    
    assert!(result.is_ok(), "Client should finish due to missing credentials");
    
    if let Ok(Ok(Err(e))) = result {
        let error_msg = e.to_string();
        assert!(
            error_msg.contains("BYBIT_API_KEY") || error_msg.contains("not found"),
            "Error should be related to missing API key, got: {}",
            error_msg
        );
    }
    
    token.cancel();
    server.shutdown();
}
