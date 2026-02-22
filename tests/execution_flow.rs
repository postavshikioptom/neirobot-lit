use neirobot_lit::ml::types::{Signal, InferenceOutput};
use neirobot_lit::trading::types::{OrderSide, FillEvent};
use neirobot_lit::data::orderbook::OrderBook;
use rust_decimal::Decimal;
use tokio_stream::iter;
use ndarray::Array2;

mod common;
use common::*;

#[tokio::test]
async fn test_execution_flow_success_buy() {
    let symbol = "BTCUSDT";
    let mut execution = setup_test_engine(symbol, Some(1.0));
    
    // В данном тесте мы проверяем непосредственно логику ExecutionEngine
    // так как run_bot_loop требует OnnxEngine, который сложно мокать без трейтов.
    // Но мы используем общие хелперы из common.
    
    let price = 50000.0;
    // Создаем матрицу [1, 3] для одного горизонта
    let probs = Array2::from_shape_vec((1, 3), vec![0.1, 0.1, 0.8]).unwrap();
    let output = InferenceOutput { 
        probs,
        signal: Signal::Down,
        probabilities: vec![0.1, 0.1, 0.8],
        source_timestamp_ms: neirobot_lit::utils::helpers::unix_ms(),
    };

    let best_bid = Decimal::from_f64(49990.0).unwrap();
    let best_ask = Decimal::from_f64(50010.0).unwrap();
    
    // Создаем OrderBook для теста
    let mut orderbook = OrderBook::new(symbol);
    let snapshot = create_mock_snapshot(symbol, price);
    orderbook.apply_update(&snapshot);
    
    let result = execution.on_inference_output(
        output, 
        price, 
        best_bid, 
        Decimal::from(100), 
        best_ask, 
        Decimal::from(100),
        &orderbook,
        &snapshot
    );
    
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_execution_flow_risk_block() {
    let symbol = "BTCUSDT";
    let mut execution = setup_test_engine(symbol, Some(0.0)); // Лимит 0
    
    let price = 50000.0;
    // Создаем матрицу [1, 3] для одного горизонта
    let probs = Array2::from_shape_vec((1, 3), vec![0.1, 0.1, 0.8]).unwrap();
    let output = InferenceOutput { 
        probs,
        signal: Signal::Down,
        probabilities: vec![0.1, 0.1, 0.8],
        source_timestamp_ms: neirobot_lit::utils::helpers::unix_ms(),
    };

    let best_bid = Decimal::from_f64(49990.0).unwrap();
    let best_ask = Decimal::from_f64(50010.0).unwrap();
    
    // Создаем OrderBook для теста
    let mut orderbook = OrderBook::new(symbol);
    let snapshot = create_mock_snapshot(symbol, price);
    orderbook.apply_update(&snapshot);
    
    let result = execution.on_inference_output(
        output, 
        price, 
        best_bid, 
        Decimal::from(100), 
        best_ask, 
        Decimal::from(100),
        &orderbook,
        &snapshot
    );
    
    assert!(result.is_err());
}

#[tokio::test]
async fn test_execution_flow_close_on_flat() {
    let symbol = "BTCUSDT";
    let mut execution = setup_test_engine(symbol, Some(1.0));
    
    // Имитируем открытую позицию
    let fill = FillEvent {
        symbol: symbol.to_string(),
        side: OrderSide::Buy,
        exec_qty: Decimal::from_f64(0.1).unwrap(),
        exec_price: Decimal::from_f64(50000.0).unwrap(),
        exec_id: "test_fill_1".to_string(),
        order_id: "order_1".to_string(),
        order_link_id: None,
        timestamp: 1000,
        is_maker: true,
        exec_fee: Decimal::ZERO,
    };
    execution.position_manager.update_from_fill(fill);
    
    let price = 50000.0;
    // Создаем матрицу [1, 3] для одного горизонта - сигнал Flat
    let probs = Array2::from_shape_vec((1, 3), vec![0.9, 0.05, 0.05]).unwrap();
    let output = InferenceOutput { 
        probs,
        signal: Signal::Flat,
        probabilities: vec![0.9, 0.05, 0.05],
        source_timestamp_ms: neirobot_lit::utils::helpers::unix_ms(),
    };

    let best_bid = Decimal::from_f64(49990.0).unwrap();
    let best_ask = Decimal::from_f64(50010.0).unwrap();
    
    // Создаем OrderBook для теста
    let mut orderbook = OrderBook::new(symbol);
    let snapshot = create_mock_snapshot(symbol, price);
    orderbook.apply_update(&snapshot);
    
    let result = execution.on_inference_output(
        output, 
        price, 
        best_bid, 
        Decimal::from(100), 
        best_ask, 
        Decimal::from(100),
        &orderbook,
        &snapshot
    );
    
    assert!(result.is_ok());
}


#[tokio::test]
async fn test_trailing_stop_loss_activation_and_update() {
    // Тест-кейс для проверки цепочки событий TSL:
    // 1. Entry на 100.0
    // 2. Mid растет до 102.0 (активация при 200 bps)
    // 3. Mid растет до 110.0 (TSL подтягивается к ~108.9 при дистанции 100 bps)
    // 4. Mid падает до 108.5 -> Триггер на закрытие
    
    let symbol = "BTCUSDT";
    let mut execution = setup_test_engine(symbol, Some(1.0));
    
    // Конфигурируем TSL параметры
    execution.bot_config.trailing_stop.tsl_mode = neirobot_lit::config::types::TSLMode::Bot;
    execution.bot_config.trailing_stop.tsl_activation_bps = 200;  // 2% для активации
    execution.bot_config.trailing_stop.tsl_distance_bps = 100;    // 1% отступа
    execution.bot_config.trailing_stop.tsl_step_bps = 10;         // 0.1% минимальный шаг
    
    // Шаг 1: Открываем позицию на 100.0
    let entry_price = 100.0;
    let fill = FillEvent {
        symbol: symbol.to_string(),
        side: OrderSide::Buy,
        exec_qty: Decimal::from_f64(1.0).unwrap(),
        exec_price: Decimal::from_f64(entry_price).unwrap(),
        exec_id: "test_fill_1".to_string(),
        order_id: "order_1".to_string(),
        order_link_id: None,
        timestamp: 1000,
        is_maker: true,
        exec_fee: Decimal::ZERO,
    };
    execution.position_manager.update_from_fill(fill);
    
    let position = execution.position_manager.get_position();
    assert_eq!(position.qty, Decimal::from_f64(1.0).unwrap());
    assert_eq!(position.avg_price, Decimal::from_f64(entry_price).unwrap());
    assert!(!position.tsl_active);
    assert_eq!(position.extreme_water_mark, entry_price);
    assert_eq!(position.current_stop_loss, entry_price);
    
    // Шаг 2: Mid растет до 102.0 (активация при 200 bps)
    let mid_price_2 = 102.0;
    execution.update_tsl(mid_price_2);
    
    let position = execution.position_manager.get_position();
    // Profit = (102 - 100) / 100 * 10000 = 200 bps -> активация
    assert!(position.tsl_active);
    assert_eq!(position.extreme_water_mark, mid_price_2);
    // new_sl = 102 * (1 - 100/10000) = 102 * 0.99 = 100.98
    assert!((position.current_stop_loss - 100.98).abs() < 0.01);
    
    // Шаг 3: Mid растет до 110.0 (TSL подтягивается)
    let mid_price_3 = 110.0;
    execution.update_tsl(mid_price_3);
    
    let position = execution.position_manager.get_position();
    assert!(position.tsl_active);
    assert_eq!(position.extreme_water_mark, mid_price_3);
    // new_sl = 110 * (1 - 100/10000) = 110 * 0.99 = 108.9
    assert!((position.current_stop_loss - 108.9).abs() < 0.01);
    
    // Шаг 4: Mid падает до 108.5 -> Триггер на закрытие
    let mid_price_4 = 108.5;
    // Проверяем, что цена пересекла стоп
    let position = execution.position_manager.get_position();
    let should_close = mid_price_4 <= position.current_stop_loss;
    assert!(should_close, "Price {} should trigger stop loss at {}", mid_price_4, position.current_stop_loss);
}

#[tokio::test]
async fn test_trailing_stop_loss_short_position() {
    // Тест для Short позиции
    let symbol = "BTCUSDT";
    let mut execution = setup_test_engine(symbol, Some(1.0));
    
    execution.bot_config.trailing_stop.tsl_mode = neirobot_lit::config::types::TSLMode::Bot;
    execution.bot_config.trailing_stop.tsl_activation_bps = 200;
    execution.bot_config.trailing_stop.tsl_distance_bps = 100;
    execution.bot_config.trailing_stop.tsl_step_bps = 10;
    
    // Открываем Short позицию на 100.0
    let entry_price = 100.0;
    let fill = FillEvent {
        symbol: symbol.to_string(),
        side: OrderSide::Sell,
        exec_qty: Decimal::from_f64(1.0).unwrap(),
        exec_price: Decimal::from_f64(entry_price).unwrap(),
        exec_id: "test_fill_1".to_string(),
        order_id: "order_1".to_string(),
        order_link_id: None,
        timestamp: 1000,
        is_maker: true,
        exec_fee: Decimal::ZERO,
    };
    execution.position_manager.update_from_fill(fill);
    
    let position = execution.position_manager.get_position();
    assert_eq!(position.qty, Decimal::from_f64(-1.0).unwrap());
    assert_eq!(position.extreme_water_mark, entry_price);
    
    // Mid падает до 98.0 (активация при 200 bps)
    let mid_price_2 = 98.0;
    execution.update_tsl(mid_price_2);
    
    let position = execution.position_manager.get_position();
    // Profit = (100 - 98) / 100 * 10000 = 200 bps -> активация
    assert!(position.tsl_active);
    assert_eq!(position.extreme_water_mark, mid_price_2);
    // new_sl = 98 * (1 + 100/10000) = 98 * 1.01 = 98.98
    assert!((position.current_stop_loss - 98.98).abs() < 0.01);
    
    // Mid падает до 90.0 (TSL подтягивается)
    let mid_price_3 = 90.0;
    execution.update_tsl(mid_price_3);
    
    let position = execution.position_manager.get_position();
    assert_eq!(position.extreme_water_mark, mid_price_3);
    // new_sl = 90 * (1 + 100/10000) = 90 * 1.01 = 90.9
    assert!((position.current_stop_loss - 90.9).abs() < 0.01);
    
    // Mid растет до 91.0 -> Триггер на закрытие
    let mid_price_4 = 91.0;
    let position = execution.position_manager.get_position();
    let should_close = mid_price_4 >= position.current_stop_loss;
    assert!(should_close, "Price {} should trigger stop loss at {}", mid_price_4, position.current_stop_loss);
}

#[tokio::test]
async fn test_trailing_stop_loss_no_activation_without_profit() {
    // Тест: TSL не активируется без достаточного профита
    let symbol = "BTCUSDT";
    let mut execution = setup_test_engine(symbol, Some(1.0));
    
    execution.bot_config.trailing_stop.tsl_mode = neirobot_lit::config::types::TSLMode::Bot;
    execution.bot_config.trailing_stop.tsl_activation_bps = 200;
    execution.bot_config.trailing_stop.tsl_distance_bps = 100;
    execution.bot_config.trailing_stop.tsl_step_bps = 10;
    
    // Открываем позицию на 100.0
    let entry_price = 100.0;
    let fill = FillEvent {
        symbol: symbol.to_string(),
        side: OrderSide::Buy,
        exec_qty: Decimal::from_f64(1.0).unwrap(),
        exec_price: Decimal::from_f64(entry_price).unwrap(),
        exec_id: "test_fill_1".to_string(),
        order_id: "order_1".to_string(),
        order_link_id: None,
        timestamp: 1000,
        is_maker: true,
        exec_fee: Decimal::ZERO,
    };
    execution.position_manager.update_from_fill(fill);
    
    // Mid растет только до 101.0 (100 bps < 200 bps)
    let mid_price = 101.0;
    execution.update_tsl(mid_price);
    
    let position = execution.position_manager.get_position();
    // TSL не должен активироваться
    assert!(!position.tsl_active);
    // extreme_water_mark и current_stop_loss не должны обновляться
    assert_eq!(position.extreme_water_mark, entry_price);
    assert_eq!(position.current_stop_loss, entry_price);
}


#[tokio::test]
async fn test_funding_rate_filter_blocks_adverse_entry() {
    // Тест: Фильтр по фандингу блокирует вход при adverse направлении и высокой ставке
    let symbol = "BTCUSDT";
    let mut execution = setup_test_engine(symbol, Some(1.0));
    
    // Конфигурируем фильтр по фандингу
    execution.bot_config.funding_filter.max_funding_rate_bps = 30;  // 30 bps = 0.03%
    execution.bot_config.funding_filter.avoid_settlement_window_ms = 300_000;  // 5 minutes
    execution.bot_config.funding_filter.min_confidence_to_ignore_funding = 0.75;  // 75%
    
    // Устанавливаем высокий funding rate (50 bps = 0.05%)
    let funding_rate = 0.0005;  // 50 bps
    let next_funding_time = neirobot_lit::utils::helpers::unix_ms() + 60_000;  // 1 minute from now
    execution.update_funding_info(funding_rate, next_funding_time);
    
    // Пытаемся открыть Long позицию (Buy) при positive funding rate (adverse)
    // Confidence = 0.7 (ниже порога 0.75)
    let confidence = 0.7;
    let should_block = execution.should_block_by_funding_rate(OrderSide::Buy, confidence);
    
    assert!(should_block, "Entry should be blocked: adverse direction + high rate + low confidence");
}

#[tokio::test]
async fn test_funding_rate_filter_allows_high_confidence_entry() {
    // Тест: Фильтр по фандингу разрешает вход при высокой уверенности
    let symbol = "BTCUSDT";
    let mut execution = setup_test_engine(symbol, Some(1.0));
    
    execution.bot_config.funding_filter.max_funding_rate_bps = 30;
    execution.bot_config.funding_filter.avoid_settlement_window_ms = 300_000;
    execution.bot_config.funding_filter.min_confidence_to_ignore_funding = 0.75;
    
    // Высокий funding rate
    let funding_rate = 0.0005;  // 50 bps
    let next_funding_time = neirobot_lit::utils::helpers::unix_ms() + 60_000;
    execution.update_funding_info(funding_rate, next_funding_time);
    
    // Пытаемся открыть Long позицию с высокой уверенностью (0.8 >= 0.75)
    let confidence = 0.8;
    let should_block = execution.should_block_by_funding_rate(OrderSide::Buy, confidence);
    
    assert!(!should_block, "Entry should be allowed: high confidence overrides funding filter");
}

#[tokio::test]
async fn test_funding_rate_filter_allows_favorable_direction() {
    // Тест: Фильтр по фандингу разрешает вход при благоприятном направлении
    let symbol = "BTCUSDT";
    let mut execution = setup_test_engine(symbol, Some(1.0));
    
    execution.bot_config.funding_filter.max_funding_rate_bps = 30;
    execution.bot_config.funding_filter.avoid_settlement_window_ms = 300_000;
    execution.bot_config.funding_filter.min_confidence_to_ignore_funding = 0.75;
    
    // Высокий positive funding rate
    let funding_rate = 0.0005;  // 50 bps
    let next_funding_time = neirobot_lit::utils::helpers::unix_ms() + 60_000;
    execution.update_funding_info(funding_rate, next_funding_time);
    
    // Пытаемся открыть Short позицию (Sell) при positive funding rate (благоприятно)
    // Confidence = 0.5 (низкая)
    let confidence = 0.5;
    let should_block = execution.should_block_by_funding_rate(OrderSide::Sell, confidence);
    
    assert!(!should_block, "Entry should be allowed: favorable direction (short when funding positive)");
}

#[tokio::test]
async fn test_funding_rate_filter_allows_low_rate() {
    // Тест: Фильтр по фандингу разрешает вход при низкой ставке
    let symbol = "BTCUSDT";
    let mut execution = setup_test_engine(symbol, Some(1.0));
    
    execution.bot_config.funding_filter.max_funding_rate_bps = 30;
    execution.bot_config.funding_filter.avoid_settlement_window_ms = 300_000;
    execution.bot_config.funding_filter.min_confidence_to_ignore_funding = 0.75;
    
    // Низкий funding rate (10 bps < 30 bps)
    let funding_rate = 0.0001;  // 10 bps
    let next_funding_time = neirobot_lit::utils::helpers::unix_ms() + 60_000;
    execution.update_funding_info(funding_rate, next_funding_time);
    
    // Пытаемся открыть Long позицию (Buy) при positive funding rate (adverse)
    // Но ставка низкая, поэтому должно быть разрешено
    let confidence = 0.5;
    let should_block = execution.should_block_by_funding_rate(OrderSide::Buy, confidence);
    
    assert!(!should_block, "Entry should be allowed: funding rate within acceptable range");
}

#[tokio::test]
async fn test_funding_rate_filter_blocks_near_settlement() {
    // Тест: Фильтр по фандингу блокирует вход близко к времени клиринга
    let symbol = "BTCUSDT";
    let mut execution = setup_test_engine(symbol, Some(1.0));
    
    execution.bot_config.funding_filter.max_funding_rate_bps = 30;
    execution.bot_config.funding_filter.avoid_settlement_window_ms = 300_000;  // 5 minutes
    execution.bot_config.funding_filter.min_confidence_to_ignore_funding = 0.75;
    
    // Высокий funding rate
    let funding_rate = 0.0005;  // 50 bps
    // Клиринг через 1 минуту (в окне 5 минут)
    let next_funding_time = neirobot_lit::utils::helpers::unix_ms() + 60_000;
    execution.update_funding_info(funding_rate, next_funding_time);
    
    // Пытаемся открыть Long позицию с низкой уверенностью
    let confidence = 0.5;
    let should_block = execution.should_block_by_funding_rate(OrderSide::Buy, confidence);
    
    assert!(should_block, "Entry should be blocked: near settlement window + low confidence");
}

#[tokio::test]
async fn test_funding_rate_filter_allows_far_from_settlement() {
    // Тест: Фильтр по фандингу разрешает вход далеко от времени клиринга
    let symbol = "BTCUSDT";
    let mut execution = setup_test_engine(symbol, Some(1.0));
    
    execution.bot_config.funding_filter.max_funding_rate_bps = 30;
    execution.bot_config.funding_filter.avoid_settlement_window_ms = 300_000;  // 5 minutes
    execution.bot_config.funding_filter.min_confidence_to_ignore_funding = 0.75;
    
    // Высокий funding rate
    let funding_rate = 0.0005;  // 50 bps
    // Клиринг через 1 час (далеко от окна 5 минут)
    let next_funding_time = neirobot_lit::utils::helpers::unix_ms() + 3_600_000;
    execution.update_funding_info(funding_rate, next_funding_time);
    
    // Пытаемся открыть Long позицию с низкой уверенностью
    let confidence = 0.5;
    let should_block = execution.should_block_by_funding_rate(OrderSide::Buy, confidence);
    
    assert!(!should_block, "Entry should be allowed: far from settlement window");
}

#[tokio::test]
async fn test_tsl_extreme_tracking_with_small_steps() {
    // Задача 167: Проверка исправления бага отслеживания extreme_water_mark
    // Даже если цена выросла меньше чем на tsl_step_bps, extreme_water_mark должен обновиться
    
    let symbol = "BTCUSDT";
    let mut execution = setup_test_engine(symbol, Some(1.0));
    
    execution.bot_config.trailing_stop.tsl_mode = neirobot_lit::config::types::TSLMode::Bot;
    execution.bot_config.trailing_stop.tsl_activation_bps = 100; // 1%
    execution.bot_config.trailing_stop.tsl_distance_bps = 50;   // 0.5%
    execution.bot_config.trailing_stop.tsl_step_bps = 20;       // 0.2% шаг обновления SL
    
    // 1. Открываем покупку на 100.0
    let fill = FillEvent {
        symbol: symbol.to_string(),
        side: OrderSide::Buy,
        exec_qty: Decimal::from_f64(1.0).unwrap(),
        exec_price: Decimal::from_f64(100.0).unwrap(),
        exec_id: "test_fill_1".to_string(),
        order_id: "order_1".to_string(),
        order_link_id: None,
        timestamp: 1000,
        is_maker: true,
        exec_fee: Decimal::ZERO,
    };
    execution.position_manager.update_from_fill(fill);
    
    // 2. Активируем TSL (Mid = 101.1, > 1% профита)
    execution.update_tsl(101.1);
    let pos = execution.position_manager.get_position();
    assert!(pos.tsl_active);
    assert_eq!(pos.extreme_water_mark, 101.1);
    // SL = 101.1 * 0.995 = 100.5945
    let initial_sl = pos.current_stop_loss;
    
    // 3. Малое движение вверх (Mid = 101.2, рост +0.1% < 0.2% шага)
    // extreme_water_mark должен стать 101.2, но SL не должен измениться
    execution.update_tsl(101.2);
    let pos = execution.position_manager.get_position();
    assert_eq!(pos.extreme_water_mark, 101.2, "Extreme water mark must update even on small moves");
    assert_eq!(pos.current_stop_loss, initial_sl, "Stop loss should NOT update if move < step_bps");
    
    // 4. Еще одно малое движение (Mid = 101.35, суммарно от initial_sl рост достаточный)
    // 101.35 * 0.995 = 100.84325
    // 100.84325 - 100.5945 = 0.24875. 
    // Относительно 100.0 это 0.248% > 0.2% шага. SL должен обновиться.
    execution.update_tsl(101.35);
    let pos = execution.position_manager.get_position();
    assert_eq!(pos.extreme_water_mark, 101.35);
    assert!(pos.current_stop_loss > initial_sl, "Stop loss should update now");
}
