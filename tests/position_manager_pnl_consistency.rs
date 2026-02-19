// Задача 166: Тесты консистентности PnL при частичном закрытии позиций

use neirobot_lit::trading::position_manager::PositionManager;
use neirobot_lit::trading::types::{OrderSide, FillEvent, MarketInfo};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

/// Создает MarketInfo для тестов
fn create_test_market_info() -> MarketInfo {
    MarketInfo {
        qty_step: dec!(0.01),
        min_order_qty: dec!(0.01),
        max_order_qty: dec!(1000.0),
        tick_size: dec!(0.1),
    }
}

/// Тест: Открытие Long позиции -> Частичное закрытие 50% -> Полное закрытие
#[test]
fn test_long_position_partial_close_pnl_consistency() {
    let market_info = create_test_market_info();
    let mut pm = PositionManager::new(
        "BTCUSDT".to_string(),
        dec!(10.0), // leverage
        market_info.qty_step,
    );

    // 1. Открытие Long позиции: Buy 1.0 @ 100.0
    let open_fill = FillEvent {
        symbol: "BTCUSDT".to_string(),
        side: OrderSide::Buy,
        exec_qty: dec!(1.0),
        exec_price: dec!(100.0),
        exec_fee: dec!(0.06), // 0.06% taker fee на 100 USDT = 0.06 USDT
        is_maker: false,
        exec_id: "exec_1".to_string(),
        order_id: "order_1".to_string(),
        order_link_id: Some("link_1".to_string()),
        timestamp: 1000,
    };

    let (pnl_1, closed_1) = pm.update_from_fill(open_fill);
    assert!(pnl_1.is_none(), "Opening position should not generate PnL");
    assert!(!closed_1, "Position should not be closed after opening");

    let position = pm.get_position();
    assert_eq!(position.qty, dec!(1.0), "Position size should be 1.0");
    assert_eq!(position.avg_price, dec!(100.0), "Average price should be 100.0");
    assert_eq!(position.side, OrderSide::Buy, "Position side should be Buy (Long)");
    assert_eq!(position.initial_size, 1.0, "Initial size should be 1.0");
    assert_eq!(position.realized_pnl, dec!(-0.06), "Realized PnL should be -0.06 (fee)");

    // 2. Частичное закрытие 50%: Sell 0.5 @ 110.0 (профит 10 USDT на 0.5 BTC = 5 USDT)
    let partial_close_fill = FillEvent {
        symbol: "BTCUSDT".to_string(),
        side: OrderSide::Sell,
        exec_qty: dec!(0.5),
        exec_price: dec!(110.0),
        exec_fee: dec!(0.033), // 0.06% taker fee на 55 USDT = 0.033 USDT
        is_maker: false,
        exec_id: "exec_2".to_string(),
        order_id: "order_2".to_string(),
        order_link_id: Some("link_2".to_string()),
        timestamp: 2000,
    };

    let (pnl_2, closed_2) = pm.update_from_fill(partial_close_fill);
    assert!(pnl_2.is_some(), "Partial close should generate PnL");
    assert!(!closed_2, "Position should not be fully closed after partial close");

    let partial_pnl = pnl_2.unwrap();
    // PnL = (110 - 100) * 0.5 = 5.0 USDT
    assert_eq!(partial_pnl, dec!(5.0), "Partial close PnL should be 5.0 USDT");

    let position = pm.get_position();
    assert_eq!(position.qty, dec!(0.5), "Position size should be 0.5 after partial close");
    assert_eq!(position.avg_price, dec!(100.0), "Average price should remain 100.0");
    // Realized PnL = -0.06 (open fee) + 5.0 (partial profit) - 0.033 (close fee) = 4.907
    assert_eq!(position.realized_pnl, dec!(4.907), "Realized PnL should be 4.907 USDT");

    // 3. Полное закрытие оставшихся 50%: Sell 0.5 @ 120.0 (профит 20 USDT на 0.5 BTC = 10 USDT)
    let full_close_fill = FillEvent {
        symbol: "BTCUSDT".to_string(),
        side: OrderSide::Sell,
        exec_qty: dec!(0.5),
        exec_price: dec!(120.0),
        exec_fee: dec!(0.036), // 0.06% taker fee на 60 USDT = 0.036 USDT
        is_maker: false,
        exec_id: "exec_3".to_string(),
        order_id: "order_3".to_string(),
        order_link_id: Some("link_3".to_string()),
        timestamp: 3000,
    };

    let (pnl_3, closed_3) = pm.update_from_fill(full_close_fill);
    assert!(pnl_3.is_some(), "Full close should generate PnL");
    assert!(closed_3, "Position should be fully closed");

    let final_pnl = pnl_3.unwrap();
    // PnL = (120 - 100) * 0.5 = 10.0 USDT
    assert_eq!(final_pnl, dec!(10.0), "Final close PnL should be 10.0 USDT");

    let position = pm.get_position();
    assert_eq!(position.qty, dec!(0.0), "Position size should be 0.0 after full close");
    assert_eq!(position.avg_price, dec!(0.0), "Average price should be 0.0 after full close");
    // Total Realized PnL = -0.06 (open) + 5.0 (partial) - 0.033 (partial fee) + 10.0 (final) - 0.036 (final fee) = 14.871
    assert_eq!(position.realized_pnl, dec!(14.871), "Total realized PnL should be 14.871 USDT");
    assert_eq!(position.initial_size, 0.0, "Initial size should be reset to 0.0");
    assert!(position.completed_tp_stages.is_empty(), "TP stages should be cleared");
}

/// Тест: Открытие Short позиции -> Частичное закрытие 50% -> Полное закрытие
#[test]
fn test_short_position_partial_close_pnl_consistency() {
    let market_info = create_test_market_info();
    let mut pm = PositionManager::new(
        "BTCUSDT".to_string(),
        dec!(10.0), // leverage
        market_info.qty_step,
    );

    // 1. Открытие Short позиции: Sell 1.0 @ 100.0
    let open_fill = FillEvent {
        symbol: "BTCUSDT".to_string(),
        side: OrderSide::Sell,
        exec_qty: dec!(1.0),
        exec_price: dec!(100.0),
        exec_fee: dec!(0.06), // 0.06% taker fee на 100 USDT = 0.06 USDT
        is_maker: false,
        exec_id: "exec_1".to_string(),
        order_id: "order_1".to_string(),
        order_link_id: Some("link_1".to_string()),
        timestamp: 1000,
    };

    let (pnl_1, closed_1) = pm.update_from_fill(open_fill);
    assert!(pnl_1.is_none(), "Opening position should not generate PnL");
    assert!(!closed_1, "Position should not be closed after opening");

    let position = pm.get_position();
    assert_eq!(position.qty, dec!(-1.0), "Position size should be -1.0 (Short)");
    assert_eq!(position.avg_price, dec!(100.0), "Average price should be 100.0");
    assert_eq!(position.side, OrderSide::Sell, "Position side should be Sell (Short)");
    assert_eq!(position.initial_size, 1.0, "Initial size should be 1.0");
    assert_eq!(position.realized_pnl, dec!(-0.06), "Realized PnL should be -0.06 (fee)");

    // 2. Частичное закрытие 50%: Buy 0.5 @ 90.0 (профит 10 USDT на 0.5 BTC = 5 USDT)
    let partial_close_fill = FillEvent {
        symbol: "BTCUSDT".to_string(),
        side: OrderSide::Buy,
        exec_qty: dec!(0.5),
        exec_price: dec!(90.0),
        exec_fee: dec!(0.027), // 0.06% taker fee на 45 USDT = 0.027 USDT
        is_maker: false,
        exec_id: "exec_2".to_string(),
        order_id: "order_2".to_string(),
        order_link_id: Some("link_2".to_string()),
        timestamp: 2000,
    };

    let (pnl_2, closed_2) = pm.update_from_fill(partial_close_fill);
    assert!(pnl_2.is_some(), "Partial close should generate PnL");
    assert!(!closed_2, "Position should not be fully closed after partial close");

    let partial_pnl = pnl_2.unwrap();
    // PnL = (100 - 90) * 0.5 = 5.0 USDT (Short профит при падении цены)
    assert_eq!(partial_pnl, dec!(5.0), "Partial close PnL should be 5.0 USDT");

    let position = pm.get_position();
    assert_eq!(position.qty, dec!(-0.5), "Position size should be -0.5 after partial close");
    assert_eq!(position.avg_price, dec!(100.0), "Average price should remain 100.0");
    // Realized PnL = -0.06 (open fee) + 5.0 (partial profit) - 0.027 (close fee) = 4.913
    assert_eq!(position.realized_pnl, dec!(4.913), "Realized PnL should be 4.913 USDT");

    // 3. Полное закрытие оставшихся 50%: Buy 0.5 @ 80.0 (профит 20 USDT на 0.5 BTC = 10 USDT)
    let full_close_fill = FillEvent {
        symbol: "BTCUSDT".to_string(),
        side: OrderSide::Buy,
        exec_qty: dec!(0.5),
        exec_price: dec!(80.0),
        exec_fee: dec!(0.024), // 0.06% taker fee на 40 USDT = 0.024 USDT
        is_maker: false,
        exec_id: "exec_3".to_string(),
        order_id: "order_3".to_string(),
        order_link_id: Some("link_3".to_string()),
        timestamp: 3000,
    };

    let (pnl_3, closed_3) = pm.update_from_fill(full_close_fill);
    assert!(pnl_3.is_some(), "Full close should generate PnL");
    assert!(closed_3, "Position should be fully closed");

    let final_pnl = pnl_3.unwrap();
    // PnL = (100 - 80) * 0.5 = 10.0 USDT
    assert_eq!(final_pnl, dec!(10.0), "Final close PnL should be 10.0 USDT");

    let position = pm.get_position();
    assert_eq!(position.qty, dec!(0.0), "Position size should be 0.0 after full close");
    assert_eq!(position.avg_price, dec!(0.0), "Average price should be 0.0 after full close");
    // Total Realized PnL = -0.06 (open) + 5.0 (partial) - 0.027 (partial fee) + 10.0 (final) - 0.024 (final fee) = 14.889
    assert_eq!(position.realized_pnl, dec!(14.889), "Total realized PnL should be 14.889 USDT");
    assert_eq!(position.initial_size, 0.0, "Initial size should be reset to 0.0");
    assert!(position.completed_tp_stages.is_empty(), "TP stages should be cleared");
}
