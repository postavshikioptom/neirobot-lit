pub mod ws_mock;

use neirobot_lit::data::orderbook::OrderBook;
use neirobot_lit::data::types::{OrderBookUpdate, PriceLevel};
use neirobot_lit::trading::{ExecutionEngine, RiskManager, types::MarketInfo};
use neirobot_lit::config::types::{RiskConfig, BotConfig, ExchangeConfig};
use neirobot_lit::ml::types::{Signal, InferenceOutput};
use neirobot_lit::trading::types::{OrderSide, OrderUpdate, OrderStatus};
use neirobot_lit::trading::rest_client::BybitRestClientTrait;
use std::path::PathBuf;
use rust_decimal::Decimal;
use rust_decimal::prelude::FromPrimitive;
use smallvec::SmallVec;
use ndarray::Array2;
use anyhow::Result;
use async_trait::async_trait;
use serde::{Serialize, de::DeserializeOwned};

pub fn setup_test_engine(symbol: &str, max_pos: Option<f64>) -> ExecutionEngine {
    let risk_config = RiskConfig {
        max_position_size: max_pos.map(|v| Decimal::from_f64(v).unwrap()),
        max_drawdown_pct: Some(Decimal::from_f64(0.1).unwrap()),
        max_open_orders: Some(5),
        ..Default::default()
    };

    let bot_config = BotConfig {
        symbol: symbol.to_string(),
        max_position_size: max_pos.map(|v| Decimal::from_f64(v).unwrap()),
        ..Default::default()
    };

    let market_info = MarketInfo {
        qty_step: Decimal::from_f64(0.01).unwrap(),
        min_order_qty: Decimal::from_f64(0.01).unwrap(),
        max_order_qty: Decimal::from_f64(100.0).unwrap(),
        tick_size: Decimal::from_f64(0.5).unwrap(),
    };

    let risk_manager = RiskManager::new(risk_config, Decimal::from_f64(10000.0).unwrap());
    let (tx, _) = tokio::sync::mpsc::channel(10);
    let state_path = PathBuf::from(format!("/tmp/test_state_{}.json", symbol));
    ExecutionEngine::new(
        symbol.to_string(),
        risk_manager,
        bot_config,
        market_info,
        tx,
        state_path,
    )
}

pub fn create_mock_snapshot(symbol: &str, price: f64) -> OrderBookUpdate {
    let mut bids = SmallVec::new();
    bids.push(PriceLevel { price: price - 0.5, size: 1.0 });
    
    let mut asks = SmallVec::new();
    asks.push(PriceLevel { price: price + 0.5, size: 1.0 });
    
    OrderBookUpdate {
        symbol: symbol.to_string(),
        timestamp_ms: 1000,
        last_update_id: 100,
        is_snapshot: true,
        bids,
        asks,
        checksum: None,
    }
}

/// Тестовая инфраструктура для интеграционного тестирования (Задача 149)
pub struct BotTestHarness {
    pub engine: ExecutionEngine,
    pub orderbook: OrderBook,
    pub exchange_config: ExchangeConfig,
    pub last_snapshot: OrderBookUpdate,
}

impl BotTestHarness {
    pub fn new(symbol: &str, max_slice_size: f64) -> Self {
        let mut engine = setup_test_engine(symbol, Some(10.0));
        engine.bot_config.max_slice_size = max_slice_size;
        
        let mut orderbook = OrderBook::new(symbol);
        let snapshot = create_mock_snapshot(symbol, 50000.0);
        orderbook.apply_update(&snapshot);
        
        Self {
            engine,
            orderbook,
            exchange_config: ExchangeConfig::default(),
            last_snapshot: snapshot,
        }
    }
    
    /// Подача сигнала в движок
    pub async fn inject_signal(
        &mut self,
        signal_side: crate::ml::types::SignalSide,
        prob: f32,
        rest_client: &impl BybitRestClientTrait,
    ) -> Result<()> {
        let probs = match signal_side {
            crate::ml::types::SignalSide::Up => Array2::from_shape_vec((1, 3), vec![0.0, prob, 1.0 - prob]).unwrap(),
            crate::ml::types::SignalSide::Down => Array2::from_shape_vec((1, 3), vec![0.0, 1.0 - prob, prob]).unwrap(),
            crate::ml::types::SignalSide::Flat => Array2::from_shape_vec((1, 3), vec![prob, (1.0 - prob) / 2.0, (1.0 - prob) / 2.0]).unwrap(),
        };
        
        let result = crate::ml::onnx::InferenceResult {
            output: InferenceOutput { 
                probs,
                signal: Signal::new(signal_side, crate::utils::helpers::unix_ms()),
                probabilities: vec![0.0, 0.0, 0.0], // Заглушка
                entropy: None,
                drift_detected: false,
            },
            duration_us: 1000,
        };
        
        let best_bid = Decimal::from_f64(49990.0).unwrap();
        let best_ask = Decimal::from_f64(50010.0).unwrap();
        
        self.engine.on_inference_output(
            result,
            50000.0,
            best_bid,
            Decimal::from(100),
            best_ask,
            Decimal::from(100),
            &self.orderbook,
            &self.last_snapshot,
            rest_client,
            &self.exchange_config,
        ).await
    }
    
    /// Эмуляция исполнения ордера
    pub async fn emulate_fill(
        &mut self,
        link_id: &str,
        qty: Decimal,
        rest_client: &impl BybitRestClientTrait,
    ) -> Result<()> {
        let order = self.engine.order_manager.get_by_client_id(link_id)
            .ok_or_else(|| anyhow::anyhow!("Order not found: {}", link_id))?;
        
        let side = order.side;
        let price = Decimal::from_f64(order.price).unwrap_or(Decimal::ZERO);
        
        let update = OrderUpdate {
            symbol: self.engine.symbol.clone(),
            order_id: format!("ord_{}", link_id),
            order_link_id: link_id.to_string(),
            status: OrderStatus::Filled,
            side,
            price,
            qty,
            cum_exec_qty: qty,
            cum_exec_value: qty * price,
            cum_exec_fee: Decimal::ZERO,
            timestamp: neirobot_lit::utils::timestamp_ms(),
            reject_reason: None,
        };
        
        let best_bid = Decimal::from_f64(49990.0).unwrap();
        let best_ask = Decimal::from_f64(50010.0).unwrap();
        
        self.engine.handle_order_update(
            update,
            rest_client,
            &self.exchange_config,
            best_bid,
            best_ask,
        ).await
    }
    
    /// Ожидание появления нового ордера в менеджере
    pub async fn wait_for_order(&self, timeout_ms: u64) -> Option<String> {
        let start = std::time::Instant::now();
        
        while start.elapsed().as_millis() < timeout_ms as u128 {
            if let Some(link_id) = self.engine.order_manager.get_active_orders().keys().next() {
                return Some(link_id.clone());
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }
        
        None
    }
    
    /// Получить количество активных ордеров
    pub fn active_orders_count(&self) -> usize {
        self.engine.order_manager.get_active_count()
    }
    
    /// Получить текущую позицию
    pub fn get_position_qty(&self) -> Decimal {
        self.engine.position_manager.get_position().qty
    }
}

/// Mock REST клиент для тестов
pub struct MockRestClient;

#[async_trait]
impl BybitRestClientTrait for MockRestClient {
    async fn post<T: Serialize + Send + Sync, R: DeserializeOwned + Send>(
        &self,
        _endpoint: &str,
        _body: &T,
    ) -> Result<R> {
        let val = serde_json::json!({
            "orderId": "test_order_1",
            "orderLinkId": format!("link_{}", neirobot_lit::utils::timestamp_ms())
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
    ) -> Result<neirobot_lit::trading::rest_client::AmendOrderResult> {
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
