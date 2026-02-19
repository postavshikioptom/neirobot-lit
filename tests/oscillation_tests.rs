mod common;

use neirobot_lit::ml::types::{Signal, InferenceOutput};
use neirobot_lit::trading::types::{OrderSide, OrderState};
use neirobot_lit::data::orderbook::OrderBook;
use neirobot_lit::config::types::ExchangeConfig;
use neirobot_lit::trading::rest_client::{BybitRestClientTrait, AmendOrderResult};
use rust_decimal::Decimal;
use ndarray::Array2;
use anyhow::Result;
use async_trait::async_trait;
use serde::Serialize;
use serde::de::DeserializeOwned;
use common::*;

struct MockRestClient;

#[async_trait]
impl BybitRestClientTrait for MockRestClient {
    async fn post<T: Serialize + Send + Sync, R: DeserializeOwned + Send>(
        &self,
        _endpoint: &str,
        _body: &T,
    ) -> Result<R> {
        let val = serde_json::json!({
            "orderId": "test_order_1",
            "orderLinkId": "test_link_1"
        });
        Ok(serde_json::from_value(val)?)
    }
    
    async fn get_signed<R: DeserializeOwned + Send>(
        &self,
        _endpoint: &str,
        _params: &str,
    ) -> Result<R> {
        unimplemented!()
    }
    
    async fn amend_order<T: Serialize + Send + Sync>(
        &self,
        _body: &T,
    ) -> Result<AmendOrderResult> {
        unimplemented!()
    }

    async fn get_equity_with_retry(&self, _retries: u32) -> Result<Decimal> {
        Ok(Decimal::from(10000))
    }
    
    async fn get_position(
        &self,
        _category: &str,
        _symbol: &str,
        _position_idx: i32,
    ) -> Result<Option<neirobot_lit::trading::rest_client::PositionInfo>> {
        Ok(None)
    }
    
    async fn get_open_orders(
        &self,
        _category: &str,
        _symbol: &str,
    ) -> Result<Vec<neirobot_lit::trading::types::OrderInfo>> {
        Ok(vec![])
    }
}

#[tokio::test]
async fn test_rapid_signal_flip_oscillation_protection() {
    let symbol = "BTCUSDT";
    // Устанавливаем min_flip_interval_ms = 500 для теста
    let mut execution = setup_test_engine(symbol, Some(1.0));
    execution.bot_config.min_flip_interval_ms = 500;
    
    let rest_client = MockRestClient;
    let exchange_config = ExchangeConfig::default();
    
    let price = 50000.0;
    let best_bid = Decimal::from_f64(49990.0).unwrap();
    let best_ask = Decimal::from_f64(50010.0).unwrap();
    let mut orderbook = OrderBook::new(symbol);
    let snapshot = create_mock_snapshot(symbol, price);
    orderbook.apply_update(&snapshot);

    // 1. Посылаем сигнал UP
    let probs_up = Array2::from_shape_vec((1, 3), vec![0.1, 0.8, 0.1]).unwrap();
    let output_up = InferenceOutput { 
        probs: probs_up,
        signal: Signal::Up,
        probabilities: vec![0.1, 0.8, 0.1],
        source_timestamp_ms: neirobot_lit::utils::helpers::unix_ms(),
    };
    
    execution.on_inference_output(
        output_up, 
        price, 
        best_bid, Decimal::from(100), 
        best_ask, Decimal::from(100),
        &orderbook,
        &snapshot,
        &rest_client,
        &exchange_config
    ).await.unwrap();
    
    assert!(execution.order_manager.get_active_count() > 0, "Should have placed UP order");
    let first_flip_ts = execution.last_flip_ts;
    assert!(first_flip_ts > 0);

    // Имитируем исполнение ордера, чтобы позиция стала Long
    let order_link_id = execution.order_manager.get_active_orders().keys().next().unwrap().clone();
    let fill = neirobot_lit::trading::types::FillEvent {
        symbol: symbol.to_string(),
        side: OrderSide::Buy,
        exec_qty: Decimal::from_f64(0.01).unwrap(),
        exec_price: best_bid,
        exec_id: "fill_1".to_string(),
        order_id: "ord_1".to_string(),
        order_link_id: Some(order_link_id.clone()),
        timestamp: 1000,
        is_maker: true,
        exec_fee: Decimal::ZERO,
    };
    execution.position_manager.update_from_fill(fill);
    execution.order_manager.update_order(&order_link_id, None, OrderState::Filled, None);
    
    assert!(execution.position_manager.get_position().qty > Decimal::ZERO);
    assert_eq!(execution.order_manager.get_active_count(), 0);

    // 2. Сразу посылаем сигнал DOWN (меньше чем через 500мс)
    let probs_down = Array2::from_shape_vec((1, 3), vec![0.1, 0.1, 0.8]).unwrap();
    let output_down = InferenceOutput { 
        probs: probs_down,
        signal: Signal::Down,
        probabilities: vec![0.1, 0.1, 0.8],
        source_timestamp_ms: neirobot_lit::utils::helpers::unix_ms(),
    };
    
    execution.on_inference_output(
        output_down, 
        price, 
        best_bid, Decimal::from(100), 
        best_ask, Decimal::from(100),
        &orderbook,
        &snapshot,
        &rest_client,
        &exchange_config
    ).await.unwrap();
    
    // Проверяем, что сигнал DOWN был подавлен
    assert_eq!(execution.order_manager.get_active_count(), 0, "DOWN order should be suppressed due to oscillation");
    assert_eq!(execution.last_flip_ts, first_flip_ts, "last_flip_ts should not be updated");

    // 3. Ждем > 500мс и посылаем DOWN снова
    tokio::time::sleep(tokio::time::Duration::from_millis(600)).await;
    
    let probs_down_2 = Array2::from_shape_vec((1, 3), vec![0.1, 0.1, 0.8]).unwrap();
    let output_down_2 = InferenceOutput { 
        probs: probs_down_2,
        signal: Signal::Down,
        probabilities: vec![0.1, 0.1, 0.8],
        source_timestamp_ms: neirobot_lit::utils::helpers::unix_ms(),
    };
    
    execution.on_inference_output(
        output_down_2, 
        price, 
        best_bid, Decimal::from(100), 
        best_ask, Decimal::from(100),
        &orderbook,
        &snapshot,
        &rest_client,
        &exchange_config
    ).await.unwrap();
    
    assert!(execution.order_manager.get_active_count() > 0, "DOWN order should be placed after timeout");
    assert!(execution.last_flip_ts > first_flip_ts, "last_flip_ts should be updated");
}
