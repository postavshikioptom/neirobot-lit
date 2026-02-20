use anyhow::Result;
use neirobot_lit::config::loader::{load_full_config, validate_full_config};
use neirobot_lit::config::types::*;
use rust_decimal::Decimal;
use rust_decimal::prelude::FromPrimitive;
use std::fs;
use std::io::Write;
use tempfile::TempDir;

fn dec(val: f64) -> Decimal {
    Decimal::from_f64(val).unwrap()
}

/// Вспомогательная функция для создания временной директории с конфигами
fn setup_test_configs(
    global_content: &str,
    exchange_content: &str,
    bot_content: &str,
) -> Result<(TempDir, std::path::PathBuf)> {
    let temp_dir = TempDir::new()?;
    let root_path = temp_dir.path();

    // Создаем global.toml
    let global_path = root_path.join("global.toml");
    let mut global_file = fs::File::create(&global_path)?;
    global_file.write_all(global_content.as_bytes())?;

    // Создаем exchange.toml
    let exchange_path = root_path.join("exchange.toml");
    let mut exchange_file = fs::File::create(&exchange_path)?;
    exchange_file.write_all(exchange_content.as_bytes())?;

    // Создаем bot config в подпапке
    let bot_dir = root_path.join("bots").join("TESTBOT");
    fs::create_dir_all(&bot_dir)?;
    let bot_config_path = bot_dir.join("config.toml");
    let mut bot_file = fs::File::create(&bot_config_path)?;
    bot_file.write_all(bot_content.as_bytes())?;

    // Создаем фиктивный файл модели
    let model_dir = bot_dir.join("model");
    fs::create_dir_all(&model_dir)?;
    let model_path = model_dir.join("test.onnx");
    fs::File::create(&model_path)?;

    Ok((temp_dir, bot_config_path))
}

// ============================================================================
// Тест 1: Ручное слияние (Manual Merge)
// ============================================================================

#[test]
fn test_manual_merge_hierarchy() {
    let global_content = r#"
[general]
env = "test"
timezone = "UTC"

[logging]
level = "debug"
format = "json"
file_name = "test.log"
rotate = true
max_size_mb = 50

[trading_defaults]
max_latency_ms = 500
retry_interval_ms = 1000
max_retries = 5
snapshot_interval_ms = 100

[risk_defaults]
drawdown_stop_pct = 5.0
max_orders_per_minute = 30
max_spread_bps = 50
"#;

    let exchange_content = r#"
[bybit]
category = "linear"
api_key_path = ".env"

[websocket]
public_url = "wss://stream.bybit.com/v5/public/linear"
private_ws_url = "wss://stream.bybit.com/v5/public/linear"
warn_rtt_ms = 500
max_subscriptions_per_connection = 30
base_delay_ms = 1000
max_delay_ms = 60000
verify_checksum = false

[rest]
base_url = "https://api-demo.bybit.com"
request_timeout_sec = 10

[rate_limits]
order_rate = 20
private_rate = 50

ws_ping_interval_secs = 20
ws_pong_timeout_secs = 30
ws_retry_initial_ms = 1000
ws_retry_max_ms = 60000
ws_retry_multiplier = 2.0
ws_retry_jitter = 0.1
rest_retry_initial_ms = 100
rest_retry_max_ms = 5000
rest_retry_multiplier = 2.0
rest_retry_jitter = 0.1
rest_max_retries = 3
"#;

    let bot_content = r#"
symbol = "TESTUSDT"
model_path = "bots/TESTBOT/model/test.onnx"
seq_len = 10
features_dim = 200
initial_balance = 1000.0
close_on_flat = false
threshold_buy = 0.6
threshold_sell = 0.4
threshold_flat = 0.3
position_idx = 0
position_sync_interval_secs = 60
buffer_pct = 0.01
leverage = 10.0
taker_fee_bps = 6.0
post_only = true
post_only_retry_limit = 3
limit_timeout_ms = 10000
emergency_close_on_exit = false

[risk]
max_spread_bps = 10
"#;

    let (temp_dir, bot_config_path) = setup_test_configs(global_content, exchange_content, bot_content)
        .expect("Failed to setup test configs");

    let result = load_full_config(temp_dir.path(), &bot_config_path);
    assert!(result.is_ok(), "Config loading should succeed");

    let config = result.unwrap();

    // Проверяем, что max_spread_bps взялся из bot config (10), а не из global (50)
    assert_eq!(config.risk.max_spread_bps, Some(10), "Bot config should override global");

    // Проверяем, что ping_interval_sec унаследовался от websocket config
    assert_eq!(config.exchange.websocket.ping_interval_sec, 20, "Should inherit from websocket config");
}

// ============================================================================
// Тест 2: Обработка Option-полей
// ============================================================================

#[test]
fn test_option_fields_inheritance() {
    let global_content = r#"
[general]
env = "test"

[logging]
level = "info"

[trading_defaults]
max_latency_ms = 500

[risk_defaults]
drawdown_stop_pct = 5.0
max_orders_per_minute = 30
max_position_size = 100.0
max_notional_usd = 50000.0
"#;

    let exchange_content = r#"
[bybit]
category = "linear"
api_key_path = ".env"

[websocket]
public_url = "wss://test.com"
private_url = "wss://test.com"
max_subscriptions_per_connection = 30
base_delay_ms = 1000
max_delay_ms = 60000
verify_checksum = false

[rest]
base_url = "https://api.test.com"
request_timeout_sec = 10

[rate_limits]
order_rate = 20
private_rate = 50
"#;

    let bot_content = r#"
symbol = "TESTUSDT"
model_path = "bots/TESTBOT/model/test.onnx"
threshold_buy = 0.6
threshold_sell = 0.4
"#;

    let (temp_dir, bot_config_path) = setup_test_configs(global_content, exchange_content, bot_content)
        .expect("Failed to setup test configs");

    let result = load_full_config(temp_dir.path(), &bot_config_path);
    assert!(result.is_ok(), "Config loading should succeed");

    let config = result.unwrap();

    // Проверяем, что max_position_size унаследовался от global
    assert_eq!(
        config.risk.max_position_size,
        Some(dec(100.0)),
        "Should inherit max_position_size from global"
    );

    // Проверяем, что max_notional_usd унаследовался от global
    assert_eq!(
        config.risk.max_notional_usd,
        Some(dec(50000.0)),
        "Should inherit max_notional_usd from global"
    );
}

// ============================================================================
// Тест 3: Валидация бизнес-логики - отрицательные значения
// ============================================================================

#[test]
fn test_validation_negative_max_position_size() {
    let global_content = r#"
[general]
env = "test"

[logging]
level = "info"

[trading_defaults]
max_latency_ms = 500

[risk_defaults]
drawdown_stop_pct = 5.0
max_orders_per_minute = 30
"#;

    let exchange_content = r#"
[bybit]
category = "linear"
api_key_path = ".env"

[websocket]
public_url = "wss://test.com"
private_url = "wss://test.com"
max_subscriptions_per_connection = 30
base_delay_ms = 1000
max_delay_ms = 60000
verify_checksum = false

[rest]
base_url = "https://api.test.com"
request_timeout_sec = 10

[rate_limits]
order_rate = 20
private_rate = 50
"#;

    let bot_content = r#"
symbol = "TESTUSDT"
model_path = "bots/TESTBOT/model/test.onnx"
threshold_buy = 0.6
threshold_sell = 0.4
max_position_size = -5.0
"#;

    let (temp_dir, bot_config_path) = setup_test_configs(global_content, exchange_content, bot_content)
        .expect("Failed to setup test configs");

    let result = load_full_config(temp_dir.path(), &bot_config_path);
    
    // Должна быть ошибка валидации
    assert!(result.is_err(), "Should fail validation with negative max_position_size");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("max_position_size must be positive"),
        "Error message should mention max_position_size validation"
    );
}

// ============================================================================
// Тест 4: Валидация - несуществующий путь к модели
// ============================================================================

#[test]
fn test_validation_nonexistent_model_path() {
    let global_content = r#"
[general]
env = "test"

[logging]
level = "info"

[trading_defaults]
max_latency_ms = 500

[risk_defaults]
drawdown_stop_pct = 5.0
max_orders_per_minute = 30
"#;

    let exchange_content = r#"
[bybit]
category = "linear"
api_key_path = ".env"

[websocket]
public_url = "wss://test.com"
private_url = "wss://test.com"
max_subscriptions_per_connection = 30
base_delay_ms = 1000
max_delay_ms = 60000
verify_checksum = false

[rest]
base_url = "https://api.test.com"
request_timeout_sec = 10

[rate_limits]
order_rate = 20
private_rate = 50
"#;

    let bot_content = r#"
symbol = "TESTUSDT"
model_path = "/nonexistent/path/to/model.onnx"
threshold_buy = 0.6
threshold_sell = 0.4
"#;

    let (temp_dir, bot_config_path) = setup_test_configs(global_content, exchange_content, bot_content)
        .expect("Failed to setup test configs");

    let result = load_full_config(temp_dir.path(), &bot_config_path);
    
    // Должна быть ошибка валидации
    assert!(result.is_err(), "Should fail validation with nonexistent model path");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("model file not found"),
        "Error message should mention model file not found"
    );
}

// ============================================================================
// Тест 5: Валидация - некорректные пороги (threshold_buy <= threshold_sell)
// ============================================================================

#[test]
fn test_validation_invalid_thresholds() {
    let global_content = r#"
[general]
env = "test"

[logging]
level = "info"

[trading_defaults]
max_latency_ms = 500

[risk_defaults]
drawdown_stop_pct = 5.0
max_orders_per_minute = 30
"#;

    let exchange_content = r#"
[bybit]
category = "linear"
api_key_path = ".env"

[websocket]
public_url = "wss://test.com"
private_url = "wss://test.com"
max_subscriptions_per_connection = 30
base_delay_ms = 1000
max_delay_ms = 60000
verify_checksum = false

[rest]
base_url = "https://api.test.com"
request_timeout_sec = 10

[rate_limits]
order_rate = 20
private_rate = 50
"#;

    let bot_content = r#"
symbol = "TESTUSDT"
model_path = "bots/TESTBOT/model/test.onnx"
threshold_buy = 0.4
threshold_sell = 0.6
"#;

    let (temp_dir, bot_config_path) = setup_test_configs(global_content, exchange_content, bot_content)
        .expect("Failed to setup test configs");

    let result = load_full_config(temp_dir.path(), &bot_config_path);
    
    // Должна быть ошибка валидации
    assert!(result.is_err(), "Should fail validation when threshold_buy <= threshold_sell");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("threshold_buy") && err_msg.contains("must be greater than"),
        "Error message should mention threshold validation"
    );
}

// ============================================================================
// Тест 6: Валидация - threshold_flat вне диапазона
// ============================================================================

#[test]
fn test_validation_threshold_flat_out_of_range() {
    let global_content = r#"
[general]
env = "test"

[logging]
level = "info"

[trading_defaults]
max_latency_ms = 500

[risk_defaults]
drawdown_stop_pct = 5.0
max_orders_per_minute = 30
"#;

    let exchange_content = r#"
[bybit]
category = "linear"
api_key_path = ".env"

[websocket]
public_url = "wss://test.com"
private_url = "wss://test.com"
max_subscriptions_per_connection = 30
base_delay_ms = 1000
max_delay_ms = 60000
verify_checksum = false

[rest]
base_url = "https://api.test.com"
request_timeout_sec = 10

[rate_limits]
order_rate = 20
private_rate = 50
"#;

    let bot_content = r#"
symbol = "TESTUSDT"
model_path = "bots/TESTBOT/model/test.onnx"
threshold_buy = 0.6
threshold_sell = 0.4
threshold_flat = 1.5
"#;

    let (temp_dir, bot_config_path) = setup_test_configs(global_content, exchange_content, bot_content)
        .expect("Failed to setup test configs");

    let result = load_full_config(temp_dir.path(), &bot_config_path);
    
    // Должна быть ошибка валидации
    assert!(result.is_err(), "Should fail validation when threshold_flat > 1.0");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("threshold_flat") && err_msg.contains("between 0.0 and 1.0"),
        "Error message should mention threshold_flat range validation"
    );
}

// ============================================================================
// Тест 7: Синтаксические ошибки в TOML
// ============================================================================

#[test]
fn test_invalid_toml_syntax() {
    let global_content = r#"
[general]
env = "test"

[logging]
level = "info"

[trading_defaults]
max_latency_ms = 500

[risk_defaults]
drawdown_stop_pct = 5.0
max_orders_per_minute = 30
"#;

    let exchange_content = r#"
[bybit]
category = "linear"
api_key_path = ".env"

[websocket]
public_url = "wss://test.com"
private_url = "wss://test.com"
max_subscriptions_per_connection = 30
base_delay_ms = 1000
max_delay_ms = 60000
verify_checksum = false

[rest]
base_url = "https://api.test.com"
request_timeout_sec = 10

[rate_limits]
order_rate = 20
private_rate = 50
"#;

    // Невалидный TOML - пропущены кавычки
    let bot_content = r#"
symbol = TESTUSDT
model_path = "bots/TESTBOT/model/test.onnx"
threshold_buy = 0.6
threshold_sell = 0.4
"#;

    let (temp_dir, bot_config_path) = setup_test_configs(global_content, exchange_content, bot_content)
        .expect("Failed to setup test configs");

    let result = load_full_config(temp_dir.path(), &bot_config_path);
    
    // Должна быть ошибка парсинга TOML
    assert!(result.is_err(), "Should fail to parse invalid TOML");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("Failed to parse TOML") || err_msg.contains("expected"),
        "Error message should mention TOML parsing error"
    );
}

// ============================================================================
// Тест 8: Пустой символ
// ============================================================================

#[test]
fn test_validation_empty_symbol() {
    let global_content = r#"
[general]
env = "test"

[logging]
level = "info"

[trading_defaults]
max_latency_ms = 500

[risk_defaults]
drawdown_stop_pct = 5.0
max_orders_per_minute = 30
"#;

    let exchange_content = r#"
[bybit]
category = "linear"
api_key_path = ".env"

[websocket]
public_url = "wss://test.com"
private_url = "wss://test.com"
max_subscriptions_per_connection = 30
base_delay_ms = 1000
max_delay_ms = 60000
verify_checksum = false

[rest]
base_url = "https://api.test.com"
request_timeout_sec = 10

[rate_limits]
order_rate = 20
private_rate = 50
"#;

    let bot_content = r#"
symbol = ""
model_path = "bots/TESTBOT/model/test.onnx"
threshold_buy = 0.6
threshold_sell = 0.4
"#;

    let (temp_dir, bot_config_path) = setup_test_configs(global_content, exchange_content, bot_content)
        .expect("Failed to setup test configs");

    let result = load_full_config(temp_dir.path(), &bot_config_path);
    
    // Должна быть ошибка валидации
    assert!(result.is_err(), "Should fail validation with empty symbol");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("symbol") && err_msg.contains("empty"),
        "Error message should mention empty symbol"
    );
}

// ============================================================================
// Тест 9: Проверка переопределения полей на всех уровнях
// ============================================================================

#[test]
fn test_three_level_override() {
    let global_content = r#"
[general]
env = "test"

[logging]
level = "info"

[trading_defaults]
max_latency_ms = 500

[risk_defaults]
drawdown_stop_pct = 5.0
max_orders_per_minute = 30
max_spread_bps = 50
max_position_size = 100.0
"#;

    let exchange_content = r#"
[bybit]
category = "linear"
api_key_path = ".env"

[websocket]
public_url = "wss://test.com"
private_url = "wss://test.com"
max_subscriptions_per_connection = 30
base_delay_ms = 1000
max_delay_ms = 60000
verify_checksum = false

[rest]
base_url = "https://api.test.com"
request_timeout_sec = 10

[rate_limits]
order_rate = 20
private_rate = 50

ws_ping_interval_secs = 25
"#;

    let bot_content = r#"
symbol = "TESTUSDT"
model_path = "bots/TESTBOT/model/test.onnx"
threshold_buy = 0.6
threshold_sell = 0.4
max_position_size = 50.0

[risk]
max_spread_bps = 10
"#;

    let (temp_dir, bot_config_path) = setup_test_configs(global_content, exchange_content, bot_content)
        .expect("Failed to setup test configs");

    let result = load_full_config(temp_dir.path(), &bot_config_path);
    assert!(result.is_ok(), "Config loading should succeed");

    let config = result.unwrap();

    // max_spread_bps: bot (10) > exchange (нет) > global (50) = 10
    assert_eq!(config.risk.max_spread_bps, Some(10));

    // max_position_size: bot (50.0) > global (100.0) = 50.0
    assert_eq!(config.risk.max_position_size, Some(dec(50.0)));

    // ping_interval_sec: websocket (25) > default (20) = 25
    assert_eq!(config.exchange.websocket.ping_interval_sec, 25);
}

// ============================================================================
// Тест 10: Проверка значений по умолчанию
// ============================================================================

#[test]
fn test_default_values() {
    let global_content = r#"
[general]
env = "test"

[logging]
level = "info"

[trading_defaults]
max_latency_ms = 500

[risk_defaults]
drawdown_stop_pct = 5.0
max_orders_per_minute = 30
"#;

    let exchange_content = r#"
[bybit]
category = "linear"
api_key_path = ".env"

[websocket]
public_url = "wss://test.com"
private_url = "wss://test.com"
max_subscriptions_per_connection = 30
base_delay_ms = 1000
max_delay_ms = 60000
verify_checksum = false

[rest]
base_url = "https://api.test.com"
request_timeout_sec = 10

[rate_limits]
order_rate = 20
private_rate = 50
"#;

    let bot_content = r#"
symbol = "TESTUSDT"
model_path = "bots/TESTBOT/model/test.onnx"
threshold_buy = 0.6
threshold_sell = 0.4
"#;

    let (temp_dir, bot_config_path) = setup_test_configs(global_content, exchange_content, bot_content)
        .expect("Failed to setup test configs");

    let result = load_full_config(temp_dir.path(), &bot_config_path);
    assert!(result.is_ok(), "Config loading should succeed");

    let config = result.unwrap();

    // Проверяем значения по умолчанию
    assert_eq!(config.bot.seq_len, 10, "Default seq_len should be 10");
    assert_eq!(config.bot.features_dim, 200, "Default features_dim should be 200");
    assert_eq!(config.bot.leverage, dec(1.0), "Default leverage should be 1.0");
    assert_eq!(config.bot.post_only, true, "Default post_only should be true");
}


// ============================================================================
// Тест 11: Валидация формата символа
// ============================================================================

#[test]
fn test_validation_symbol_format_with_dash() {
    let global_content = r#"
[general]
env = "test"

[logging]
level = "info"

[trading_defaults]
max_latency_ms = 500

[risk_defaults]
drawdown_stop_pct = 5.0
max_orders_per_minute = 30
"#;

    let exchange_content = r#"
[bybit]
category = "linear"
api_key_path = ".env"

[websocket]
public_url = "wss://test.com"
private_url = "wss://test.com"
max_subscriptions_per_connection = 30
base_delay_ms = 1000
max_delay_ms = 60000
verify_checksum = false

[rest]
base_url = "https://api.test.com"
request_timeout_sec = 10

[rate_limits]
order_rate = 20
private_rate = 50
"#;

    let bot_content = r#"
symbol = "cake-usdt"
model_path = "bots/TESTBOT/model/test.onnx"
threshold_buy = 0.6
threshold_sell = 0.4
"#;

    let (temp_dir, bot_config_path) = setup_test_configs(global_content, exchange_content, bot_content)
        .expect("Failed to setup test configs");

    let result = load_full_config(temp_dir.path(), &bot_config_path);
    
    // Должна быть ошибка валидации формата
    assert!(result.is_err(), "Should fail validation with invalid symbol format");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("invalid format") || err_msg.contains("uppercase"),
        "Error message should mention symbol format validation"
    );
}

#[test]
fn test_validation_symbol_format_lowercase() {
    let global_content = r#"
[general]
env = "test"

[logging]
level = "info"

[trading_defaults]
max_latency_ms = 500

[risk_defaults]
drawdown_stop_pct = 5.0
max_orders_per_minute = 30
"#;

    let exchange_content = r#"
[bybit]
category = "linear"
api_key_path = ".env"

[websocket]
public_url = "wss://test.com"
private_url = "wss://test.com"
max_subscriptions_per_connection = 30
base_delay_ms = 1000
max_delay_ms = 60000
verify_checksum = false

[rest]
base_url = "https://api.test.com"
request_timeout_sec = 10

[rate_limits]
order_rate = 20
private_rate = 50
"#;

    let bot_content = r#"
symbol = "cakeusdt"
model_path = "bots/TESTBOT/model/test.onnx"
threshold_buy = 0.6
threshold_sell = 0.4
"#;

    let (temp_dir, bot_config_path) = setup_test_configs(global_content, exchange_content, bot_content)
        .expect("Failed to setup test configs");

    let result = load_full_config(temp_dir.path(), &bot_config_path);
    
    // Должна быть ошибка валидации (символ должен быть в верхнем регистре)
    assert!(result.is_err(), "Should fail validation with lowercase symbol");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("uppercase"),
        "Error message should mention uppercase requirement"
    );
}


// ============================================================================
// Тест 12: Валидация max_open_orders = 0
// ============================================================================

#[test]
fn test_validation_zero_max_open_orders() {
    let global_content = r#"
[general]
env = "test"

[logging]
level = "info"

[trading_defaults]
max_latency_ms = 500

[risk_defaults]
drawdown_stop_pct = 5.0
max_orders_per_minute = 30
max_open_orders = 0
"#;

    let exchange_content = r#"
[bybit]
category = "linear"
api_key_path = ".env"

[websocket]
public_url = "wss://test.com"
private_url = "wss://test.com"
max_subscriptions_per_connection = 30
base_delay_ms = 1000
max_delay_ms = 60000
verify_checksum = false

[rest]
base_url = "https://api.test.com"
request_timeout_sec = 10

[rate_limits]
order_rate = 20
private_rate = 50
"#;

    let bot_content = r#"
symbol = "TESTUSDT"
model_path = "bots/TESTBOT/model/test.onnx"
threshold_buy = 0.6
threshold_sell = 0.4
"#;

    let (temp_dir, bot_config_path) = setup_test_configs(global_content, exchange_content, bot_content)
        .expect("Failed to setup test configs");

    let result = load_full_config(temp_dir.path(), &bot_config_path);
    
    // Должна быть ошибка валидации
    assert!(result.is_err(), "Should fail validation with max_open_orders = 0");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("max_open_orders must be > 0"),
        "Error message should mention max_open_orders validation"
    );
}

// ============================================================================
// Тест 13: Валидация max_spread_bps = 0
// ============================================================================

#[test]
fn test_validation_zero_max_spread() {
    let global_content = r#"
[general]
env = "test"

[logging]
level = "info"

[trading_defaults]
max_latency_ms = 500

[risk_defaults]
drawdown_stop_pct = 5.0
max_orders_per_minute = 30
"#;

    let exchange_content = r#"
[bybit]
category = "linear"
api_key_path = ".env"

[websocket]
public_url = "wss://test.com"
private_url = "wss://test.com"
max_subscriptions_per_connection = 30
base_delay_ms = 1000
max_delay_ms = 60000
verify_checksum = false

[rest]
base_url = "https://api.test.com"
request_timeout_sec = 10

[rate_limits]
order_rate = 20
private_rate = 50
"#;

    let bot_content = r#"
symbol = "TESTUSDT"
model_path = "bots/TESTBOT/model/test.onnx"
threshold_buy = 0.6
threshold_sell = 0.4

[risk]
max_spread_bps = 0
"#;

    let (temp_dir, bot_config_path) = setup_test_configs(global_content, exchange_content, bot_content)
        .expect("Failed to setup test configs");

    let result = load_full_config(temp_dir.path(), &bot_config_path);
    
    // Должна быть ошибка валидации
    assert!(result.is_err(), "Should fail validation with max_spread_bps = 0");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("max_spread_bps must be > 0"),
        "Error message should mention max_spread_bps validation"
    );
}

// ============================================================================
// Тест 14: Environment Overrides (Secrets)
// ============================================================================

#[test]
fn test_environment_overrides_secrets() {
    use neirobot_lit::config::loader::load_secrets;

    // Устанавливаем переменные окружения
    std::env::set_var("BYBIT_API_KEY", "test_key_from_env");
    std::env::set_var("BYBIT_API_SECRET", "test_secret_from_env");

    let result = load_secrets();
    assert!(result.is_ok(), "Should successfully load secrets from environment");

    let (key, secret) = result.unwrap();
    assert_eq!(key, "test_key_from_env", "API key should come from environment");
    assert_eq!(secret, "test_secret_from_env", "API secret should come from environment");

    // Очищаем переменные окружения после теста
    std::env::remove_var("BYBIT_API_KEY");
    std::env::remove_var("BYBIT_API_SECRET");
}

#[test]
fn test_secrets_missing_env_vars() {
    use neirobot_lit::config::loader::load_secrets;

    // Убеждаемся, что переменные окружения не установлены
    std::env::remove_var("BYBIT_API_KEY");
    std::env::remove_var("BYBIT_API_SECRET");

    let result = load_secrets();
    assert!(result.is_err(), "Should fail when environment variables are missing");
    
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("BYBIT_API_KEY not found"),
        "Error message should mention missing BYBIT_API_KEY"
    );
}

// ============================================================================
// Тест 15: Default Traits - отсутствие global.toml
// ============================================================================

#[test]
fn test_missing_global_config_error() {
    use neirobot_lit::config::loader::load_full_config;
    
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let root_path = temp_dir.path();

    // Создаем только exchange.toml (без global.toml)
    let exchange_content = r#"
[bybit]
category = "linear"
api_key_path = ".env"

[websocket]
public_url = "wss://test.com"
private_url = "wss://test.com"
max_subscriptions_per_connection = 30
base_delay_ms = 1000
max_delay_ms = 60000
verify_checksum = false

[rest]
base_url = "https://api.test.com"
request_timeout_sec = 10

[rate_limits]
order_rate = 20
private_rate = 50
"#;

    let exchange_path = root_path.join("exchange.toml");
    let mut exchange_file = fs::File::create(&exchange_path).expect("Failed to create exchange.toml");
    exchange_file.write_all(exchange_content.as_bytes()).expect("Failed to write exchange.toml");

    // Создаем bot config
    let bot_dir = root_path.join("bots").join("TESTBOT");
    fs::create_dir_all(&bot_dir).expect("Failed to create bot dir");
    
    let bot_content = r#"
symbol = "TESTUSDT"
model_path = "bots/TESTBOT/model/test.onnx"
threshold_buy = 0.6
threshold_sell = 0.4
"#;

    let bot_config_path = bot_dir.join("config.toml");
    let mut bot_file = fs::File::create(&bot_config_path).expect("Failed to create bot config");
    bot_file.write_all(bot_content.as_bytes()).expect("Failed to write bot config");

    // Создаем фиктивный файл модели
    let model_dir = bot_dir.join("model");
    fs::create_dir_all(&model_dir).expect("Failed to create model dir");
    let model_path = model_dir.join("test.onnx");
    fs::File::create(&model_path).expect("Failed to create model file");

    let result = load_full_config(root_path, &bot_config_path);
    
    // Должна быть ошибка, так как global.toml отсутствует
    assert!(result.is_err(), "Should fail when global.toml is missing");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("global.toml") || err_msg.contains("Failed to read config file"),
        "Error message should mention missing global.toml"
    );
}

// ============================================================================
// Тест 16: Default Traits - отсутствие exchange.toml
// ============================================================================

#[test]
fn test_missing_exchange_config_error() {
    use neirobot_lit::config::loader::load_full_config;
    
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let root_path = temp_dir.path();

    // Создаем только global.toml (без exchange.toml)
    let global_content = r#"
[general]
env = "test"

[logging]
level = "info"

[trading_defaults]
max_latency_ms = 500

[risk_defaults]
drawdown_stop_pct = 5.0
max_orders_per_minute = 30
"#;

    let global_path = root_path.join("global.toml");
    let mut global_file = fs::File::create(&global_path).expect("Failed to create global.toml");
    global_file.write_all(global_content.as_bytes()).expect("Failed to write global.toml");

    // Создаем bot config
    let bot_dir = root_path.join("bots").join("TESTBOT");
    fs::create_dir_all(&bot_dir).expect("Failed to create bot dir");
    
    let bot_content = r#"
symbol = "TESTUSDT"
model_path = "bots/TESTBOT/model/test.onnx"
threshold_buy = 0.6
threshold_sell = 0.4
"#;

    let bot_config_path = bot_dir.join("config.toml");
    let mut bot_file = fs::File::create(&bot_config_path).expect("Failed to create bot config");
    bot_file.write_all(bot_content.as_bytes()).expect("Failed to write bot config");

    // Создаем фиктивный файл модели
    let model_dir = bot_dir.join("model");
    fs::create_dir_all(&model_dir).expect("Failed to create model dir");
    let model_path = model_dir.join("test.onnx");
    fs::File::create(&model_path).expect("Failed to create model file");

    let result = load_full_config(root_path, &bot_config_path);
    
    // Должна быть ошибка, так как exchange.toml отсутствует
    assert!(result.is_err(), "Should fail when exchange.toml is missing");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("exchange.toml") || err_msg.contains("Failed to read config file"),
        "Error message should mention missing exchange.toml"
    );
}
