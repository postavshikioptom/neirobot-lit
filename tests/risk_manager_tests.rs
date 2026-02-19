use neirobot_lit::config::types::RiskConfig;
use neirobot_lit::risk::risk_manager::RiskManager;
use neirobot_lit::trading::types::OrderSide;
use neirobot_lit::trading::position_manager::Position;
use rust_decimal::Decimal;
use rust_decimal::prelude::FromPrimitive;
use chrono::{Utc, Duration};

fn dec(val: f64) -> Decimal {
    Decimal::from_f64(val).unwrap()
}

fn create_test_risk_config() -> RiskConfig {
    RiskConfig {
        drawdown_stop_pct: 5.0,
        max_orders_per_minute: 30,
        max_open_orders: Some(10),
        max_position_size: Some(dec(10.0)),
        max_notional_usd: Some(dec(10000.0)),
        max_margin_usd: Some(dec(5000.0)),
        max_daily_drawdown_usd: Some(dec(500.0)),
        max_daily_drawdown_pct: Some(dec(10.0)),
        auto_reset_at_midnight: true,
        max_drawdown_pct: Some(dec(0.20)),
        max_spread_bps: Some(10),
        max_price_deviation_pct: Some(dec(0.01)),
    }
}

fn create_empty_position() -> Position {
    Position {
        symbol: "BTCUSDT".to_string(),
        qty: Decimal::ZERO,
        avg_price: Decimal::ZERO,
        leverage: dec(10.0),
        unrealized_pnl: Decimal::ZERO,
        realized_pnl: Decimal::ZERO,
        unrealized_pnl_pct: Decimal::ZERO,
        mark_pnl: Decimal::ZERO,
        updated_at: 0,
    }
}

// ============================================================================
// Тесты Spread BPS Filter (Задача 073)
// ============================================================================

#[test]
#[tracing_test::traced_test]
fn test_spread_bps_calculation_and_blocking() {
    // Формула: bps = ((ask - bid) / mid) * 10000
    // bid=100.0, ask=100.1 -> spread=0.1, mid=100.05, bps = (0.1 / 100.05) * 10000 ≈ 9.995 bps
    let mut config = create_test_risk_config();
    config.max_spread_bps = Some(5); // Лимит 5 bps
    let rm = RiskManager::new(config, dec(1000.0));

    let bid = dec(100.0);
    let ask = dec(100.1);
    let bid_vol = dec(10.0);
    let ask_vol = dec(10.0);

    let result = rm.check_spread_gate(bid, bid_vol, ask, ask_vol);
    
    // Спред ~10 bps > 5 bps лимит -> должна быть блокировка
    assert_eq!(result, false, "Spread exceeding limit should block");
    
    // Проверяем, что в логах есть сообщение о широком спреде
    assert!(logs_contain("Spread too wide"));
}

#[test]
#[tracing_test::traced_test]
fn test_spread_gate_with_zero_mid_price() {
    // Edge Case: mid = 0 (пустой стакан) -> Должен вернуть false без паники
    let config = create_test_risk_config();
    let rm = RiskManager::new(config, dec(1000.0));

    let bid = dec(0.0);
    let ask = dec(0.0);
    let bid_vol = dec(10.0);
    let ask_vol = dec(10.0);

    let result = rm.check_spread_gate(bid, bid_vol, ask, ask_vol);
    
    // Должен вернуть false без паники (div by zero guard)
    assert_eq!(result, false, "Zero mid price should return false without panic");
    assert!(logs_contain("Invalid prices detected"));
}

#[test]
fn test_spread_gate_config_none() {
    // Проверка, что при Option::None в конфиге гейт всегда возвращает true
    let mut config = create_test_risk_config();
    config.max_spread_bps = None;
    let rm = RiskManager::new(config, dec(1000.0));

    let bid = dec(100.0);
    let ask = dec(200.0); // Огромный спред
    let bid_vol = dec(10.0);
    let ask_vol = dec(10.0);

    let result = rm.check_spread_gate(bid, bid_vol, ask, ask_vol);
    
    // При None гейт должен пропускать
    assert_eq!(result, true, "Spread gate with None config should always pass");
}

// ============================================================================
// Тесты Drawdown & Peak Logic (Задача 072)
// ============================================================================

#[test]
#[tracing_test::traced_test]
fn test_drawdown_peak_logic() {
    // Peak: Вход 1000 -> 1100 (пик 1100). Просадка до 950 (DD = 150).
    let mut config = create_test_risk_config();
    config.max_daily_drawdown_usd = Some(dec(150.0));
    let mut rm = RiskManager::new(config, dec(1000.0));

    // Начальный эквити = 1000, PnL = 0
    rm.update_equity(dec(0.0));
    assert_eq!(rm.peak_daily_equity, dec(1000.0));

    // Рост до 1100 (PnL = +100)
    rm.update_equity(dec(100.0));
    assert_eq!(rm.peak_daily_equity, dec(1100.0));

    // Просадка до 950 (PnL = -50)
    // DD = 1100 - 950 = 150 USD (на лимите)
    let result = rm.check_drawdown(dec(-50.0));
    
    // Должна сработать блокировка
    assert!(result.is_err(), "Drawdown at limit should trigger block");
    assert!(rm.is_blocked, "RiskManager should be blocked");
    assert!(logs_contain("HARD STOP TRIGGERED"));
}

#[test]
fn test_drawdown_reset_logic() {
    // Reset: Вручную установить last_reset_date на вчерашнюю дату.
    // Проверить, что check_drawdown сбрасывает is_blocked.
    let mut config = create_test_risk_config();
    config.auto_reset_at_midnight = true;
    config.max_daily_drawdown_usd = Some(dec(100.0));
    let mut rm = RiskManager::new(config, dec(1000.0));

    // Блокируем из-за просадки
    rm.update_equity(dec(100.0)); // Peak = 1100
    let _ = rm.check_drawdown(dec(-50.0)); // DD = 150, блокировка
    assert!(rm.is_blocked);

    // Устанавливаем дату на вчера
    let yesterday = Utc::now().date_naive() - Duration::days(1);
    rm.last_reset_date = Some(yesterday);

    // Вызываем check_global_risk, который должен сбросить блокировку
    let result = rm.check_global_risk(dec(-50.0));
    
    // После сброса блокировка должна быть снята
    assert!(!rm.is_blocked, "Block should be reset after midnight");
    assert!(result.is_ok(), "Global risk check should pass after reset");
}

// ============================================================================
// Тесты Max Position & Orders
// ============================================================================

#[test]
#[tracing_test::traced_test]
fn test_max_orders_limit_blocking() {
    // Проверка блокировки при count == limit
    let mut config = create_test_risk_config();
    config.max_open_orders = Some(5);
    let rm = RiskManager::new(config, dec(1000.0));

    // До лимита - OK
    assert!(rm.check_orders_limit_gate(4));
    
    // На лимите - блокировка
    assert!(!rm.check_orders_limit_gate(5));
    assert!(logs_contain("MAX ORDERS REACHED"));
    
    // Выше лимита - блокировка
    assert!(!rm.check_orders_limit_gate(6));
}

#[test]
fn test_max_position_size_blocking() {
    let mut config = create_test_risk_config();
    config.max_position_size = Some(dec(5.0));
    let rm = RiskManager::new(config, dec(1000.0));

    let mut pos = create_empty_position();
    pos.qty = dec(3.0); // Текущая позиция 3 BTC

    // Попытка купить еще 3 BTC -> projected = 6 BTC > 5 BTC лимит
    let result = rm.validate_order(
        OrderSide::Buy,
        dec(3.0),
        &pos,
        0,
        dec(50000.0),
    );

    assert!(result.is_err(), "Should block when exceeding max position size");
    assert!(result.unwrap_err().to_string().contains("MaxPositionSize"));
}

#[test]
fn test_max_notional_blocking() {
    let mut config = create_test_risk_config();
    config.max_notional_usd = Some(dec(10000.0));
    let rm = RiskManager::new(config, dec(1000.0));

    let pos = create_empty_position();
    let mid_price = dec(50000.0);

    // Попытка купить 0.5 BTC по 50000 USD = 25000 USD > 10000 лимит
    let result = rm.validate_order(
        OrderSide::Buy,
        dec(0.5),
        &pos,
        0,
        mid_price,
    );

    assert!(result.is_err(), "Should block when exceeding max notional");
    assert!(result.unwrap_err().to_string().contains("MaxNotionalExceeded"));
}

// ============================================================================
// Тесты Price Deviation & Tick (Задача 075)
// ============================================================================

#[test]
#[tracing_test::traced_test]
fn test_price_deviation_blocking() {
    // Deviation: mid=100.0, order=101.1, limit=0.01 (1%) -> Блокировка (1.1% > 1%)
    let mut config = create_test_risk_config();
    config.max_price_deviation_pct = Some(dec(0.01)); // 1%
    let rm = RiskManager::new(config, dec(1000.0));

    let mid = dec(100.0);
    let order_price = dec(101.1);
    let tick = dec(0.01);

    let result = rm.validate_order_price(order_price, mid, tick);
    
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), false, "Should block when deviation exceeds limit");
    assert!(logs_contain("Price deviation too high"));
}

#[test]
fn test_price_tick_precision_blocking() {
    // Precision: price=10.0005, tick=0.01 -> Блокировка (не кратно тику)
    let config = create_test_risk_config();
    let rm = RiskManager::new(config, dec(1000.0));

    let mid = dec(10.0);
    let order_price = dec(10.0005);
    let tick = dec(0.01);

    let result = rm.validate_order_price(order_price, mid, tick);
    
    assert!(result.is_err(), "Should block when price is not multiple of tick");
    assert!(result.unwrap_err().to_string().contains("not a multiple of tick_size"));
}

#[test]
fn test_price_deviation_config_none() {
    // Config Flexibility: при Option::None гейт всегда возвращает true
    let mut config = create_test_risk_config();
    config.max_price_deviation_pct = None;
    let rm = RiskManager::new(config, dec(1000.0));

    let mid = dec(100.0);
    let order_price = dec(200.0); // Огромное отклонение
    let tick = dec(0.01);

    let result = rm.validate_order_price(order_price, mid, tick);
    
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), true, "Should pass when deviation check is disabled");
}

// ============================================================================
// Тест Fail-Fast (цепочка проверок)
// ============================================================================

#[test]
#[tracing_test::traced_test]
fn test_fail_fast_gate_chain() {
    // Если первый гейт заблокировал, остальные не должны вызываться
    let mut config = create_test_risk_config();
    config.max_daily_drawdown_usd = Some(dec(50.0));
    let mut rm = RiskManager::new(config, dec(1000.0));

    // Создаем просадку, которая заблокирует бота
    rm.update_equity(dec(100.0)); // Peak = 1100
    let _ = rm.check_drawdown(dec(-60.0)); // DD = 160 > 50, блокировка
    assert!(rm.is_blocked);

    let pos = create_empty_position();
    
    // Попытка проверить ордер - должна сразу упасть на глобальной проверке
    let result = rm.check_order_gate(
        OrderSide::Buy,
        dec(0.1),
        dec(50000.0),
        &pos,
        0,
        dec(-60.0),
        dec(50000.0),
        dec(0.01), // tick_size (Задача 176)
    );

    assert!(result.is_err(), "Should fail fast on global risk check");
    assert!(result.unwrap_err().to_string().contains("Trading is blocked"));
    
    // Проверяем, что в логах есть сообщение о блокировке
    assert!(logs_contain("Trading is blocked"));
}

// ============================================================================
// Тест Reduce-Only Logic
// ============================================================================

#[test]
fn test_reduce_only_always_allowed() {
    // Закрытие/уменьшение позиции всегда разрешено
    let mut config = create_test_risk_config();
    config.max_position_size = Some(dec(1.0)); // Очень маленький лимит
    let rm = RiskManager::new(config, dec(1000.0));

    let mut pos = create_empty_position();
    pos.qty = dec(5.0); // Текущая позиция 5 BTC (больше лимита)

    // Продажа 2 BTC -> projected = 3 BTC (уменьшение)
    let result = rm.validate_order(
        OrderSide::Sell,
        dec(2.0),
        &pos,
        0,
        dec(50000.0),
    );

    // Должно быть разрешено, несмотря на превышение лимита
    assert!(result.is_ok(), "Reduce-only orders should always be allowed");
}

// Примечание: функция logs_contain автоматически инжектируется
// макросом #[traced_test] и не требует определения

// ============================================================================
// Тесты Max Margin
// ============================================================================

#[test]
fn test_max_margin_blocking() {
    let mut config = create_test_risk_config();
    config.max_margin_usd = Some(dec(1000.0));
    let rm = RiskManager::new(config, dec(10000.0));

    let pos = create_empty_position();
    let mid_price = dec(50000.0);

    // Попытка купить 0.5 BTC по 50000 USD = 25000 USD notional
    // При leverage 10x: margin = 25000 / 10 = 2500 USD > 1000 лимит
    let result = rm.validate_order(
        OrderSide::Buy,
        dec(0.5),
        &pos,
        0,
        mid_price,
    );

    assert!(result.is_err(), "Should block when exceeding max margin");
    assert!(result.unwrap_err().to_string().contains("MaxMarginExceeded"));
}

// ============================================================================
// Тесты для проверки корректности расчета BPS
// ============================================================================

#[test]
fn test_spread_bps_exact_calculation() {
    // Проверка точности расчета BPS
    let mut config = create_test_risk_config();
    config.max_spread_bps = Some(10);
    let rm = RiskManager::new(config, dec(1000.0));

    // bid=100.0, ask=100.1
    // spread = 0.1, mid = 100.05
    // bps = (0.1 / 100.05) * 10000 = 9.995 bps (< 10 bps)
    let bid = dec(100.0);
    let ask = dec(100.1);
    let bid_vol = dec(10.0);
    let ask_vol = dec(10.0);

    let result = rm.check_spread_gate(bid, bid_vol, ask, ask_vol);
    
    // Должно пройти, так как 9.995 < 10
    assert_eq!(result, true, "Spread just below limit should pass");
}

#[test]
#[tracing_test::traced_test]
fn test_spread_bps_at_80_percent_threshold() {
    // Проверка логирования при приближении к лимиту (80%)
    let mut config = create_test_risk_config();
    config.max_spread_bps = Some(100);
    let rm = RiskManager::new(config, dec(1000.0));

    // bid=100.0, ask=100.85
    // spread = 0.85, mid = 100.425
    // bps = (0.85 / 100.425) * 10000 ≈ 84.6 bps (> 80% от 100)
    let bid = dec(100.0);
    let ask = dec(100.85);
    let bid_vol = dec(10.0);
    let ask_vol = dec(10.0);

    let result = rm.check_spread_gate(bid, bid_vol, ask, ask_vol);
    
    assert_eq!(result, true, "Should pass but log warning");
    assert!(logs_contain("Spread nearing limit"));
}

// ============================================================================
// Тесты для проверки глобальной просадки (Cumulative Drawdown)
// ============================================================================

#[test]
#[tracing_test::traced_test]
fn test_cumulative_drawdown_blocking() {
    let mut config = create_test_risk_config();
    config.max_drawdown_pct = Some(dec(0.20)); // 20% от пика
    let mut rm = RiskManager::new(config, dec(1000.0));

    // Рост до 2000 (PnL = +1000)
    rm.update_equity(dec(1000.0));
    assert_eq!(rm.peak_equity, dec(2000.0));

    // Просадка до 1500 (PnL = +500)
    // DD = (2000 - 1500) / 2000 = 25% > 20% лимит
    let result = rm.check_global_risk(dec(500.0));
    
    assert!(result.is_err(), "Should block on cumulative drawdown");
    assert!(rm.is_blocked, "Should set is_blocked flag");
    assert!(logs_contain("HARD STOP: Cumulative drawdown exceeded"));
}

// ============================================================================
// Тесты для проверки минимального notional
// ============================================================================

#[test]
fn test_min_notional_blocking() {
    let config = create_test_risk_config();
    let mut rm = RiskManager::new(config, dec(1000.0));

    let pos = create_empty_position();
    
    // Попытка создать ордер с notional < 5 USD
    // qty=0.0001, price=10.0 -> notional = 0.001 USD < 5 USD
    let result = rm.check_order_gate(
        OrderSide::Buy,
        dec(0.0001),
        dec(10.0),
        &pos,
        0,
        dec(0.0),
        dec(10.0),
        dec(0.01), // tick_size (Задача 176)
    );

    assert!(result.is_err(), "Should block orders below min notional");
    assert!(result.unwrap_err().to_string().contains("Order value too small"));
}

// ============================================================================
// Тесты для проверки дневной просадки в процентах
// ============================================================================

#[test]
#[tracing_test::traced_test]
fn test_daily_drawdown_percentage_blocking() {
    let mut config = create_test_risk_config();
    config.max_daily_drawdown_pct = Some(dec(10.0)); // 10%
    config.max_daily_drawdown_usd = None; // Отключаем USD лимит
    let mut rm = RiskManager::new(config, dec(1000.0));

    // Рост до 1100 (PnL = +100)
    rm.update_equity(dec(100.0));
    assert_eq!(rm.peak_daily_equity, dec(1100.0));

    // Просадка до 980 (PnL = -20)
    // DD = (1100 - 980) / 1100 * 100 = 10.9% > 10%
    let result = rm.check_drawdown(dec(-20.0));
    
    assert!(result.is_err(), "Should block on daily drawdown percentage");
    assert!(rm.is_blocked);
    assert!(logs_contain("Daily DD % limit reached"));
}

// ============================================================================
// Тесты для проверки инвертированного спреда
// ============================================================================

#[test]
#[tracing_test::traced_test]
fn test_inverted_spread_blocking() {
    let config = create_test_risk_config();
    let rm = RiskManager::new(config, dec(1000.0));

    // ask < bid (инвертированный спред)
    let bid = dec(100.0);
    let ask = dec(99.0);
    let bid_vol = dec(10.0);
    let ask_vol = dec(10.0);

    let result = rm.check_spread_gate(bid, bid_vol, ask, ask_vol);
    
    assert_eq!(result, false, "Inverted spread should be blocked");
    assert!(logs_contain("Inverted spread detected"));
}

// ============================================================================
// Тесты для проверки нулевых/отрицательных цен
// ============================================================================

#[test]
#[tracing_test::traced_test]
fn test_negative_prices_blocking() {
    let config = create_test_risk_config();
    let rm = RiskManager::new(config, dec(1000.0));

    let bid = dec(-10.0);
    let ask = dec(10.0);
    let bid_vol = dec(10.0);
    let ask_vol = dec(10.0);

    let result = rm.check_spread_gate(bid, bid_vol, ask, ask_vol);
    
    assert_eq!(result, false, "Negative prices should be blocked");
    assert!(logs_contain("Invalid prices detected"));
}

// ============================================================================
// Тесты для проверки комбинации лимитов
// ============================================================================

#[test]
fn test_multiple_limits_first_violation_wins() {
    // Проверка, что при нарушении нескольких лимитов возвращается первая ошибка
    let mut config = create_test_risk_config();
    config.max_position_size = Some(dec(1.0));
    config.max_notional_usd = Some(dec(10000.0));
    let rm = RiskManager::new(config, dec(10000.0));

    let pos = create_empty_position();
    let mid_price = dec(50000.0);

    // Попытка купить 5 BTC (нарушает оба лимита)
    let result = rm.validate_order(
        OrderSide::Buy,
        dec(5.0),
        &pos,
        0,
        mid_price,
    );

    assert!(result.is_err());
    // Должна вернуться ошибка MaxPositionSize (проверяется первой)
    assert!(result.unwrap_err().to_string().contains("MaxPositionSize"));
}

// ============================================================================
// Тесты для проверки Short позиций
// ============================================================================

#[test]
fn test_short_position_reduce_only() {
    let mut config = create_test_risk_config();
    config.max_position_size = Some(dec(1.0));
    let rm = RiskManager::new(config, dec(10000.0));

    let mut pos = create_empty_position();
    pos.qty = dec(-5.0); // Short позиция 5 BTC

    // Покупка 2 BTC -> projected = -3 BTC (уменьшение short)
    let result = rm.validate_order(
        OrderSide::Buy,
        dec(2.0),
        &pos,
        0,
        dec(50000.0),
    );

    // Должно быть разрешено (reduce-only)
    assert!(result.is_ok(), "Reducing short position should be allowed");
}

// ============================================================================
// Тесты для проверки tick size с нулевым значением
// ============================================================================

#[test]
fn test_price_validation_with_zero_tick() {
    let config = create_test_risk_config();
    let rm = RiskManager::new(config, dec(1000.0));

    let mid = dec(100.0);
    let order_price = dec(101.0);
    let tick = Decimal::ZERO;

    // При tick=0 проверка кратности должна пропускаться
    let result = rm.validate_order_price(order_price, mid, tick);
    
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), true, "Should pass with zero tick size");
}

// Примечание: функция logs_contain автоматически инжектируется
// макросом #[traced_test] и не требует определения
