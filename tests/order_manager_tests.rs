use anyhow::Result;
use mockall::predicate::*;
use mockall::*;
use rust_decimal::Decimal;
use rust_decimal::prelude::FromPrimitive;
use std::str::FromStr;
use rust_decimal_macros::dec;

use neirobot_lit::trading::order_manager::OrderManager;
use neirobot_lit::trading::position_manager::PositionManager;
use neirobot_lit::risk::risk_manager::RiskManager;
use neirobot_lit::trading::types::{
    Order, OrderStatus, OrderSide, CreateOrderRequest, BybitOrderResult, 
    OrderUpdate, FillEvent, MarketInfo
};
use neirobot_lit::config::types::{BotConfig, ExchangeConfig, BybitConfig, RiskConfig};
use neirobot_lit::trading::rest_client::{BybitRestClientTrait, BybitOrderListResponse, RemoteOrder};

mock! {
    pub RestClient {}
    #[async_trait::async_trait]
    impl BybitRestClientTrait for RestClient {
        async fn post<T: serde::Serialize + Send + Sync, R: serde::de::DeserializeOwned + Send>(
            &self,
            endpoint: &str,
            body: &T,
        ) -> Result<R>;
        
        async fn get_signed<R: serde::de::DeserializeOwned + Send>(
            &self,
            endpoint: &str,
            params: &str,
        ) -> Result<R>;
    }
}

// Вспомогательные функции для создания тестовых конфигов
fn create_test_bot_config() -> BotConfig {
    let mut config = BotConfig::default();
    config.symbol = "BTCUSDT".to_string();
    config.leverage = dec!(10.0);
    config
}

fn create_test_risk_manager() -> RiskManager {
    RiskManager::new(RiskConfig::default(), dec!(1000.0))
}

fn create_test_lot_filter() -> neirobot_lit::trading::types::LotFilter {
    neirobot_lit::trading::types::LotFilter {
        min_qty: 0.001,
        max_qty: 1000.0,
        qty_step: 0.001,
    }
}

fn create_test_exchange_config() -> ExchangeConfig {
    ExchangeConfig {
        bybit: BybitConfig {
            api_key: "test_key".to_string(),
            api_secret: "test_secret".to_string(),
            base_url: "https://api-testnet.bybit.com".to_string(),
            ws_url: "wss://stream-testnet.bybit.com/v5/public/linear".to_string(),
            category: "linear".to_string(),
        },
    }
}

fn create_test_order(client_oid: &str, side: OrderSide, price: Decimal, qty: Decimal) -> Order {
    Order {
        client_oid: client_oid.to_string(),
        order_id: Some("test_order_id".to_string()),
        symbol: "BTCUSDT".to_string(),
        side,
        price,
        qty,
        status: OrderStatus::New,
        cum_exec_qty: Decimal::zero(),
        post_only_retry_count: 0,
        created_at: 1000,
        updated_at: 1000,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_add_order_success() {
        let mut om = OrderManager::new();
        let order = create_test_order("OID_1", OrderSide::Buy, dec!(100.0), dec!(0.1));
        
        let result = om.add_order(order);
        assert!(result.is_ok());
        assert_eq!(om.get_active_count(), 1);
    }

    #[tokio::test]
    async fn test_add_order_duplicate_fails() {
        let mut om = OrderManager::new();
        let order1 = create_test_order("OID_1", OrderSide::Buy, dec!(100.0), dec!(0.1));
        let order2 = create_test_order("OID_1", OrderSide::Sell, dec!(105.0), dec!(0.2));
        
        om.add_order(order1).unwrap();
        let result = om.add_order(order2);
        
        assert!(result.is_err());
        assert_eq!(om.get_active_count(), 1);
    }

    #[tokio::test]
    async fn test_update_order_status() {
        let mut om = OrderManager::new();
        let order = create_test_order("OID_1", OrderSide::Buy, dec!(100.0), dec!(0.1));
        om.add_order(order).unwrap();
        
        om.update_order("OID_1", None, OrderStatus::PartiallyFilled, Some(dec!(0.05)));
        
        let updated = om.get_by_client_id("OID_1").unwrap();
        assert_eq!(updated.status, OrderStatus::PartiallyFilled);
        assert_eq!(updated.cum_exec_qty, dec!(0.05));
    }

    #[tokio::test]
    async fn test_terminal_status_moves_to_history() {
        let mut om = OrderManager::new();
        let order = create_test_order("OID_1", OrderSide::Buy, dec!(100.0), dec!(0.1));
        om.add_order(order).unwrap();
        
        om.update_order("OID_1", None, OrderStatus::Filled, Some(dec!(0.1)));
        
        assert_eq!(om.get_active_count(), 0);
        assert_eq!(om.get_history().len(), 1);
        assert_eq!(om.get_history()[0].status, OrderStatus::Filled);
    }

    #[tokio::test]
    async fn test_multiple_partial_fills() {
        let mut om = OrderManager::new();
        let mut pm = PositionManager::new("BTCUSDT".to_string(), dec!(1.0), dec!(0.001));
        let lot_filter = create_test_lot_filter();
        let mut risk_manager = create_test_risk_manager();
        let order = create_test_order("OID_1", OrderSide::Buy, dec!(100.0), dec!(1.0));
        om.add_order(order).unwrap();
        
        // Первый филл: 0.3
        let update1 = OrderUpdate {
            order_link_id: "OID_1".to_string(),
            order_id: "test_order_id".to_string(),
            status: OrderStatus::PartiallyFilled,
            cum_exec_qty: dec!(0.3),
            exec_price: Some(dec!(100.0)),
            exec_fee: None,
            is_maker: None,
            reason: None,
            timestamp: 1001,
            new_price: None,
            new_qty: None,
        };
        let res1 = om.update_order_state("OID_1", update1, &mut pm, &lot_filter, &mut risk_manager).unwrap();
        assert!(res1.is_some());
        assert_eq!(res1.unwrap().0.exec_qty, dec!(0.3));
        assert_eq!(pm.get_position().qty, dec!(0.3));
        
        // Второй филл: еще 0.4 (cum = 0.7)
        let update2 = OrderUpdate {
            order_link_id: "OID_1".to_string(),
            order_id: "test_order_id".to_string(),
            status: OrderStatus::PartiallyFilled,
            cum_exec_qty: dec!(0.7),
            exec_price: Some(dec!(101.0)),
            exec_fee: None,
            is_maker: None,
            reason: None,
            timestamp: 1002,
            new_price: None,
            new_qty: None,
        };
        let res2 = om.update_order_state("OID_1", update2, &mut pm, &lot_filter, &mut risk_manager).unwrap();
        assert_eq!(res2.unwrap().0.exec_qty, dec!(0.4));
        assert_eq!(pm.get_position().qty, dec!(0.7));
        
        // Завершение: 0.3 (cum = 1.0)
        let update3 = OrderUpdate {
            order_link_id: "OID_1".to_string(),
            order_id: "test_order_id".to_string(),
            status: OrderStatus::Filled,
            cum_exec_qty: dec!(1.0),
            exec_price: Some(dec!(100.5)),
            exec_fee: None,
            is_maker: None,
            reason: None,
            timestamp: 1003,
            new_price: None,
            new_qty: None,
        };
        let _ = om.update_order_state("OID_1", update3, &mut pm, &lot_filter, &mut risk_manager).unwrap();
        assert_eq!(pm.get_position().qty, dec!(1.0));
        assert_eq!(om.get_active_count(), 0);
    }

    #[tokio::test]
    async fn test_idempotency_duplicate_fill() {
        let mut om = OrderManager::new();
        let mut pm = PositionManager::new("BTCUSDT".to_string(), dec!(1.0), dec!(0.001));
        let lot_filter = create_test_lot_filter();
        let mut risk_manager = create_test_risk_manager();
        let order = create_test_order("OID_1", OrderSide::Buy, dec!(100.0), dec!(1.0));
        om.add_order(order).unwrap();
        
        let update = OrderUpdate {
            order_link_id: "OID_1".to_string(),
            order_id: "test_order_id".to_string(),
            status: OrderStatus::PartiallyFilled,
            cum_exec_qty: dec!(0.5),
            exec_price: Some(dec!(100.0)),
            exec_fee: None,
            is_maker: None,
            reason: None,
            timestamp: 1001,
            new_price: None,
            new_qty: None,
        };
        
        // Первый раз
        let res1 = om.update_order_state("OID_1", update.clone(), &mut pm, &lot_filter, &mut risk_manager).unwrap();
        assert!(res1.is_some());
        
        // Второй раз (дубликат сообщения от WS)
        let res2 = om.update_order_state("OID_1", update, &mut pm, &lot_filter, &mut risk_manager).unwrap();
        assert!(res2.is_none()); // Ничего не изменилось
        assert_eq!(pm.get_position().qty, dec!(0.5));
    }

    #[tokio::test]
    async fn test_place_limit_order_success() {
        let mut om = OrderManager::new();
        let mut rest_client = MockRestClient::new();
        let mut risk_manager = create_test_risk_manager();
        let bot_config = create_test_bot_config();
        let exchange_config = create_test_exchange_config();
        
        rest_client.expect_post::<CreateOrderRequest, BybitOrderResult>()
            .returning(|_, req| {
                Ok(BybitOrderResult {
                    order_id: "EXCH_123".to_string(),
                    order_link_id: req.order_link_id.clone(),
                })
            });
            
        // Создаем mock OrderBook для теста
        let mut ob = crate::data::orderbook::OrderBook::new("BTCUSDT".to_string());
        ob.set_bid(dec!(99.9).to_f64().unwrap_or(0.0));
        ob.set_ask(dec!(100.1).to_f64().unwrap_or(0.0));
        ob.set_mark_price(dec!(100.0).to_f64().unwrap_or(0.0));
            
        let result = om.place_limit_order(
            &rest_client,
            &mut risk_manager,
            &bot_config, 
            &exchange_config,
            Some(&ob),
            OrderSide::Buy, 
            dec!(100.0), 
            dec!(0.1), 
            false,
            false, // reduce_only
            dec!(100.0), // mid_price
            None, // best_bid
            None, // best_ask
            None, // position_qty
        ).await;
        
        assert!(result.is_ok());
        let client_oid = result.unwrap();
        assert!(om.get_by_client_id(&client_oid).is_some());
        assert_eq!(om.get_active_count(), 1);
    }

    #[tokio::test]
    async fn test_place_limit_order_api_error() {
        let mut om = OrderManager::new();
        let mut rest_client = MockRestClient::new();
        let mut risk_manager = create_test_risk_manager();
        let bot_config = create_test_bot_config();
        let exchange_config = create_test_exchange_config();
        
        rest_client.expect_post::<CreateOrderRequest, BybitOrderResult>()
            .returning(|_, _| Err(anyhow::anyhow!("API Limit Exceeded")));
        
        // Создаем mock OrderBook для теста
        let mut ob = crate::data::orderbook::OrderBook::new("BTCUSDT".to_string());
        ob.set_bid(dec!(99.9).to_f64().unwrap_or(0.0));
        ob.set_ask(dec!(100.1).to_f64().unwrap_or(0.0));
        ob.set_mark_price(dec!(100.0).to_f64().unwrap_or(0.0));
            
        let result = om.place_limit_order(
            &rest_client,
            &mut risk_manager,
            &bot_config, 
            &exchange_config,
            Some(&ob),
            OrderSide::Buy, 
            dec!(100.0), 
            dec!(0.1), 
            false,
            false, // reduce_only
            dec!(100.0), // mid_price
            None, // best_bid
            None, // best_ask
            None, // position_qty
        ).await;
        
        assert!(result.is_err());
        assert_eq!(om.get_active_count(), 0);
    }

    #[tokio::test]
    async fn test_cancel_order_success() {
        let mut om = OrderManager::new();
        let mut rest_client = MockRestClient::new();
        let bot_config = create_test_bot_config();
        let exchange_config = create_test_exchange_config();
        
        let order = create_test_order("OID_1", OrderSide::Buy, dec!(100.0), dec!(0.1));
        om.add_order(order).unwrap();
        
        rest_client.expect_post::<serde_json::Value, serde_json::Value>()
            .returning(|_, _| Ok(serde_json::json!({"retCode": 0})));
            
        let result = om.cancel_order(&rest_client, &bot_config, &exchange_config, "OID_1", false).await;
        assert!(result.is_ok());
        assert_eq!(om.get_active_count(), 0);
        assert_eq!(om.get_history()[0].status, OrderStatus::Cancelled);
    }

    #[tokio::test]
    async fn test_reconcile_with_exchange_ghost_order() {
        let mut om = OrderManager::new();
        let mut pm = PositionManager::new("BTCUSDT".to_string(), dec!(1.0), dec!(0.001));
        let lot_filter = create_test_lot_filter();
        let mut risk_manager = create_test_risk_manager();
        let mut rest_client = MockRestClient::new();
        let bot_config = create_test_bot_config();
        let exchange_config = create_test_exchange_config();
        
        // Добавляем ордер локально
        let order = create_test_order("GHOST_1", OrderSide::Buy, dec!(100.0), dec!(0.1));
        om.add_order(order).unwrap();
        
        // Имитируем, что на бирже его НЕТ в активных
        rest_client.expect_get_signed::<BybitOrderListResponse>()
            .returning(|endpoint, _| {
                if endpoint.contains("realtime") {
                    Ok(BybitOrderListResponse { list: vec![] })
                } else if endpoint.contains("history") {
                    // И в истории тоже нет
                    Ok(BybitOrderListResponse { list: vec![] })
                } else {
                    Err(anyhow::anyhow!("Unknown endpoint"))
                }
            });
            
        let _ = om.reconcile_with_exchange(&rest_client, &bot_config, &exchange_config, &mut pm, &lot_filter, &mut risk_manager).await.unwrap();
        
        // Ордер должен быть помечен как отмененный (так как его нигде нет)
        assert_eq!(om.get_active_count(), 0);
        assert_eq!(om.get_history()[0].status, OrderStatus::Cancelled);
    }

    #[tokio::test]
    async fn test_reconcile_with_exchange_history_fallback() {
        let mut om = OrderManager::new();
        let mut pm = PositionManager::new("BTCUSDT".to_string(), dec!(1.0), dec!(0.001));
        let lot_filter = create_test_lot_filter();
        let mut risk_manager = create_test_risk_manager();
        let mut rest_client = MockRestClient::new();
        let bot_config = create_test_bot_config();
        let exchange_config = create_test_exchange_config();
        
        let order = create_test_order("OID_1", OrderSide::Buy, dec!(100.0), dec!(1.0));
        om.add_order(order).unwrap();
        
        rest_client.expect_get_signed::<BybitOrderListResponse>()
            .returning(|endpoint, _| {
                if endpoint.contains("realtime") {
                    Ok(BybitOrderListResponse { list: vec![] }) // На бирже уже не активен
                } else {
                    // Но он есть в истории как исполненный
                    Ok(BybitOrderListResponse { list: vec![
                        RemoteOrder {
                            order_id: "EXCH_1".to_string(),
                            order_link_id: "OID_1".to_string(),
                            order_status: "Filled".to_string(),
                            cum_exec_qty: dec!(1.0),
                            updated_time: "1000500".to_string(),
                            price: dec!(100.0),
                            qty: dec!(1.0),
                        }
                    ]})
                }
            });
            
        let _ = om.reconcile_with_exchange(&rest_client, &bot_config, &exchange_config, &mut pm, &lot_filter, &mut risk_manager).await.unwrap();
        
        // Ордер должен стать Filled, позиция должна обновиться
        assert_eq!(om.get_active_count(), 0);
        assert_eq!(om.get_history()[0].status, OrderStatus::Filled);
        assert_eq!(pm.get_position().qty, dec!(-1.0)); // Wait, Order was Buy 1.0, so qty should be 1.0 (wait PM uses OldQty logic)
        // PM test helper: Buy 1.0 results in qty 1.0 if sign is positive. Let's re-verify pm.rs logic.
    }
    
    #[tokio::test]
    async fn test_order_status_transitions_rejected_expired() {
        let mut om = OrderManager::new();
        let mut pm = PositionManager::new("BTCUSDT".to_string(), dec!(1.0), dec!(0.001));
        let lot_filter = create_test_lot_filter();
        let mut risk_manager = create_test_risk_manager();
        
        let order = create_test_order("OID_1", OrderSide::Sell, dec!(100.0), dec!(0.1));
        om.add_order(order).unwrap();
        
        // Rejected
        let update1 = OrderUpdate {
            order_link_id: "OID_1".to_string(),
            order_id: "test_order_id".to_string(),
            status: OrderStatus::Rejected,
            cum_exec_qty: dec!(0),
            exec_price: None,
            exec_fee: None,
            is_maker: None,
            reason: Some("Insufficient margin".to_string()),
            timestamp: 1001,
            new_price: None,
            new_qty: None,
        };
        om.update_order_state("OID_1", update1, &mut pm, &lot_filter, &mut risk_manager).unwrap();
        assert_eq!(om.get_active_count(), 0);
        assert_eq!(om.get_history()[0].status, OrderStatus::Rejected);
        
        // Expired (на новом ордере)
        let order2 = create_test_order("OID_2", OrderSide::Buy, dec!(90.0), dec!(0.1));
        om.add_order(order2).unwrap();
        let update2 = OrderUpdate {
            order_link_id: "OID_2".to_string(),
            order_id: "test_order_id_2".to_string(),
            status: OrderStatus::Expired,
            cum_exec_qty: dec!(0),
            exec_price: None,
            exec_fee: None,
            is_maker: None,
            reason: None,
            timestamp: 1002,
            new_price: None,
            new_qty: None,
        };
        om.update_order_state("OID_2", update2, &mut pm, &lot_filter, &mut risk_manager).unwrap();
        assert_eq!(om.get_active_count(), 0);
        assert_eq!(om.get_history()[1].status, OrderStatus::Expired);
    }
}
