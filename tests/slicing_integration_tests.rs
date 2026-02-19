mod common;

use common::{BotTestHarness, MockRestClient};
use neirobot_lit::ml::types::Signal;
use rust_decimal::Decimal;
use rust_decimal::prelude::FromPrimitive;

/// Тест 1: Simple Slicing
/// Проверяет разбиение крупного ордера на слайсы по max_slice_size
#[tokio::test]
async fn test_simple_slicing() {
    let symbol = "BTCUSDT";
    let max_slice_size = 1.0;
    
    let mut harness = BotTestHarness::new(symbol, max_slice_size);
    let rest_client = MockRestClient;
    
    // Устанавливаем параметры для генерации ордера на 3.5
    // Нужно настроить initial_balance и другие параметры так, чтобы calculate_order_size вернул 3.5
    harness.engine.bot_config.initial_balance = Decimal::from_f64(175000.0).unwrap(); // 3.5 * 50000
    harness.engine.bot_config.leverage = Decimal::ONE;
    harness.engine.bot_config.buffer_pct = Decimal::ZERO;
    harness.engine.bot_config.taker_fee_bps = Decimal::ZERO;
    
    // Подаем сигнал на покупку с высокой вероятностью
    harness.inject_signal(Signal::Up, 0.9, &rest_client).await.unwrap();
    
    // Проверяем, что выставлен первый ордер на 1.0
    assert_eq!(harness.active_orders_count(), 1, "Should have 1 active order");
    
    let first_order_id = harness.wait_for_order(1000).await
        .expect("First order should be placed");
    
    let first_order = harness.engine.order_manager.get_by_client_id(&first_order_id)
        .expect("First order should exist");
    
    assert_eq!(
        Decimal::from_f64(first_order.qty).unwrap(),
        Decimal::from_f64(max_slice_size).unwrap(),
        "First slice should be max_slice_size"
    );
    
    // Эмулируем исполнение первого ордера
    harness.emulate_fill(&first_order_id, Decimal::from_f64(max_slice_size).unwrap(), &rest_client).await.unwrap();
    
    // Ждем появления второго ордера (с учетом rate limiting)
    let second_order_id = harness.wait_for_order(2000).await
        .expect("Second order should be placed");
    
    assert_eq!(harness.active_orders_count(), 1, "Should have second slice order");
    
    let second_order = harness.engine.order_manager.get_by_client_id(&second_order_id)
        .expect("Second order should exist");
    
    assert_eq!(
        Decimal::from_f64(second_order.qty).unwrap(),
        Decimal::from_f64(max_slice_size).unwrap(),
        "Second slice should be max_slice_size"
    );
    
    // Эмулируем исполнение второго ордера
    harness.emulate_fill(&second_order_id, Decimal::from_f64(max_slice_size).unwrap(), &rest_client).await.unwrap();
    
    // Ждем появления третьего ордера
    let third_order_id = harness.wait_for_order(2000).await
        .expect("Third order should be placed");
    
    assert_eq!(harness.active_orders_count(), 1, "Should have third slice order");
    
    let third_order = harness.engine.order_manager.get_by_client_id(&third_order_id)
        .expect("Third order should exist");
    
    assert_eq!(
        Decimal::from_f64(third_order.qty).unwrap(),
        Decimal::from_f64(max_slice_size).unwrap(),
        "Third slice should be max_slice_size"
    );
    
    // Эмулируем исполнение третьего ордера
    harness.emulate_fill(&third_order_id, Decimal::from_f64(max_slice_size).unwrap(), &rest_client).await.unwrap();
    
    // Ждем появления четвертого (последнего) ордера на 0.5
    let fourth_order_id = harness.wait_for_order(2000).await
        .expect("Fourth order should be placed");
    
    assert_eq!(harness.active_orders_count(), 1, "Should have fourth (last) slice order");
    
    let fourth_order = harness.engine.order_manager.get_by_client_id(&fourth_order_id)
        .expect("Fourth order should exist");
    
    assert_eq!(
        Decimal::from_f64(fourth_order.qty).unwrap(),
        Decimal::from_f64(0.5).unwrap(),
        "Last slice should be 0.5 (remainder)"
    );
    
    // Эмулируем исполнение последнего ордера
    harness.emulate_fill(&fourth_order_id, Decimal::from_f64(0.5).unwrap(), &rest_client).await.unwrap();
    
    // Проверяем, что больше нет активных ордеров
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    assert_eq!(harness.active_orders_count(), 0, "Should have no more orders");
    
    // Проверяем итоговую позицию
    let final_position = harness.get_position_qty();
    assert_eq!(
        final_position,
        Decimal::from_f64(3.5).unwrap(),
        "Final position should be 3.5"
    );
}

/// Тест 2: Risk Block during Slicing
/// Проверяет остановку нарезки при блокировке риск-менеджером
#[tokio::test]
async fn test_risk_block_during_slicing() {
    let symbol = "BTCUSDT";
    let max_slice_size = 1.0;
    
    let mut harness = BotTestHarness::new(symbol, max_slice_size);
    let rest_client = MockRestClient;
    
    // Устанавливаем жесткий лимит позиции на 1.5
    harness.engine.bot_config.max_position_size = Some(Decimal::from_f64(1.5).unwrap());
    harness.engine.risk_manager = neirobot_lit::risk::RiskManager::new(
        neirobot_lit::config::types::RiskConfig {
            max_position_size: Some(Decimal::from_f64(1.5).unwrap()),
            ..Default::default()
        },
        Decimal::from_f64(10000.0).unwrap()
    );
    
    // Настраиваем для генерации ордера на 3.0
    harness.engine.bot_config.initial_balance = Decimal::from_f64(150000.0).unwrap();
    harness.engine.bot_config.leverage = Decimal::ONE;
    harness.engine.bot_config.buffer_pct = Decimal::ZERO;
    harness.engine.bot_config.taker_fee_bps = Decimal::ZERO;
    
    // Подаем сигнал на покупку
    harness.inject_signal(Signal::Up, 0.9, &rest_client).await.unwrap();
    
    // Первый слайс должен быть выставлен
    assert_eq!(harness.active_orders_count(), 1, "Should have first slice");
    
    let first_order_id = harness.wait_for_order(1000).await
        .expect("First order should be placed");
    
    // Исполняем первый слайс (1.0)
    harness.emulate_fill(&first_order_id, Decimal::from_f64(1.0).unwrap(), &rest_client).await.unwrap();
    
    // Ждем появления второго ордера (с учетом rate limiting)
    let second_order_result = harness.wait_for_order(2000).await;
    
    // Второй слайс должен быть выставлен, но урезан до 0.5 из-за лимита
    // (текущая позиция 1.0, лимит 1.5, осталось 0.5)
    assert!(second_order_result.is_some(), "Should have second slice");
    
    let second_order_id = second_order_result.unwrap();
    let second_order = harness.engine.order_manager.get_by_client_id(&second_order_id)
        .expect("Second order should exist");
    
    // Проверяем, что второй слайс урезан до 0.5
    assert!(
        Decimal::from_f64(second_order.qty).unwrap() <= Decimal::from_f64(0.5).unwrap(),
        "Second slice should be trimmed to fit max_position_size limit"
    );
    
    // Исполняем второй слайс
    harness.emulate_fill(&second_order_id, Decimal::from_f64(second_order.qty).unwrap(), &rest_client).await.unwrap();
    
    // Третий слайс НЕ должен быть выставлен, так как достигнут лимит
    let third_order_result = harness.wait_for_order(2000).await;
    assert!(third_order_result.is_none(), "Should have no more orders due to risk limit");
    
    // Проверяем, что позиция не превышает лимит
    let final_position = harness.get_position_qty();
    assert!(
        final_position <= Decimal::from_f64(1.5).unwrap(),
        "Position should not exceed max_position_size limit"
    );
}

/// Тест 3: Rate Limiter Safety
/// Проверяет соблюдение пауз между выставлением слайсов
#[tokio::test]
async fn test_rate_limiter_safety() {
    let symbol = "BTCUSDT";
    let max_slice_size = 1.0;
    
    let mut harness = BotTestHarness::new(symbol, max_slice_size);
    let rest_client = MockRestClient;
    
    // Настраиваем для генерации ордера на 2.5
    harness.engine.bot_config.initial_balance = Decimal::from_f64(125000.0).unwrap();
    harness.engine.bot_config.leverage = Decimal::ONE;
    harness.engine.bot_config.buffer_pct = Decimal::ZERO;
    harness.engine.bot_config.taker_fee_bps = Decimal::ZERO;
    
    // Подаем сигнал на покупку
    let start_time = std::time::Instant::now();
    harness.inject_signal(Signal::Up, 0.9, &rest_client).await.unwrap();
    
    // Первый слайс
    let first_order_id = harness.wait_for_order(1000).await
        .expect("First order should be placed");
    let first_order_time = start_time.elapsed();
    
    harness.emulate_fill(&first_order_id, Decimal::from_f64(1.0).unwrap(), &rest_client).await.unwrap();
    
    // Второй слайс
    let second_order_id = harness.wait_for_order(2000).await
        .expect("Second order should be placed");
    let second_order_time = start_time.elapsed();
    
    // Проверяем, что между слайсами прошло время (минимальная задержка)
    let time_between_slices = second_order_time - first_order_time;
    assert!(
        time_between_slices.as_millis() >= 50,
        "Should have delay between slices (got {}ms)",
        time_between_slices.as_millis()
    );
    
    harness.emulate_fill(&second_order_id, Decimal::from_f64(1.0).unwrap(), &rest_client).await.unwrap();
    
    // Третий (последний) слайс
    let third_order_id = harness.wait_for_order(2000).await
        .expect("Third order should be placed");
    let third_order_time = start_time.elapsed();
    
    // Проверяем задержку перед третьим слайсом
    let time_between_slices_2 = third_order_time - second_order_time;
    assert!(
        time_between_slices_2.as_millis() >= 50,
        "Should have delay between slices (got {}ms)",
        time_between_slices_2.as_millis()
    );
    
    harness.emulate_fill(&third_order_id, Decimal::from_f64(0.5).unwrap(), &rest_client).await.unwrap();
    
    // Проверяем итоговую позицию
    let final_position = harness.get_position_qty();
    assert_eq!(
        final_position,
        Decimal::from_f64(2.5).unwrap(),
        "Final position should be 2.5"
    );
}
