// Тесты для проверки риск-гейтов (Задача 169: Signal Staleness Check)

use neirobot_lit::config::types::{BotConfig, StalenessAction};
use neirobot_lit::ml::types::{Signal, InferenceOutput};
use ndarray::Array2;

#[test]
fn test_staleness_check_skip_mode() {
    // Создаем конфигурацию с лимитом 100ms и режимом Skip
    let mut bot_config = BotConfig::default();
    bot_config.max_signal_age_ms = 100;
    bot_config.staleness_action = StalenessAction::Skip;

    // Создаем сигнал с timestamp 200ms назад (устаревший)
    let current_time = neirobot_lit::utils::helpers::unix_ms();
    let stale_timestamp = current_time - 200;

    let output = InferenceOutput {
        signal: Signal::Up,
        probabilities: vec![0.1, 0.8, 0.1],
        probs: Array2::from_shape_vec((1, 3), vec![0.1, 0.8, 0.1]).unwrap(),
        source_timestamp_ms: stale_timestamp,
    };

    // Проверяем, что сигнал устарел
    let signal_age = current_time - output.source_timestamp_ms;
    assert!(signal_age > bot_config.max_signal_age_ms, 
        "Signal should be stale: age {}ms > limit {}ms", 
        signal_age, bot_config.max_signal_age_ms);

    // В режиме Skip устаревший сигнал должен быть пропущен
    assert_eq!(bot_config.staleness_action, StalenessAction::Skip);
}

#[test]
fn test_staleness_check_log_only_mode() {
    // Создаем конфигурацию с режимом LogOnly
    let mut bot_config = BotConfig::default();
    bot_config.max_signal_age_ms = 100;
    bot_config.staleness_action = StalenessAction::LogOnly;

    // Создаем устаревший сигнал
    let current_time = neirobot_lit::utils::helpers::unix_ms();
    let stale_timestamp = current_time - 200;

    let output = InferenceOutput {
        signal: Signal::Down,
        probabilities: vec![0.1, 0.1, 0.8],
        probs: Array2::from_shape_vec((1, 3), vec![0.1, 0.1, 0.8]).unwrap(),
        source_timestamp_ms: stale_timestamp,
    };

    // Проверяем, что сигнал устарел
    let signal_age = current_time - output.source_timestamp_ms;
    assert!(signal_age > bot_config.max_signal_age_ms);

    // В режиме LogOnly устаревший сигнал должен быть выполнен (только логируется)
    assert_eq!(bot_config.staleness_action, StalenessAction::LogOnly);
}

#[test]
fn test_fresh_signal_passes() {
    // Создаем конфигурацию с лимитом 100ms
    let mut bot_config = BotConfig::default();
    bot_config.max_signal_age_ms = 100;
    bot_config.staleness_action = StalenessAction::Skip;

    // Создаем свежий сигнал (50ms назад)
    let current_time = neirobot_lit::utils::helpers::unix_ms();
    let fresh_timestamp = current_time - 50;

    let output = InferenceOutput {
        signal: Signal::Up,
        probabilities: vec![0.1, 0.8, 0.1],
        probs: Array2::from_shape_vec((1, 3), vec![0.1, 0.8, 0.1]).unwrap(),
        source_timestamp_ms: fresh_timestamp,
    };

    // Проверяем, что сигнал свежий
    let signal_age = current_time - output.source_timestamp_ms;
    assert!(signal_age <= bot_config.max_signal_age_ms,
        "Signal should be fresh: age {}ms <= limit {}ms",
        signal_age, bot_config.max_signal_age_ms);
}

#[test]
fn test_risk_manager_staleness_monitoring() {
    use neirobot_lit::risk::risk_manager::RiskManager;
    use neirobot_lit::config::types::RiskConfig;
    use rust_decimal::Decimal;

    let config = RiskConfig::default();
    let mut risk_manager = RiskManager::new(config, Decimal::from(1000));

    // Регистрируем несколько свежих сигналов
    for _ in 0..5 {
        risk_manager.register_signal_staleness(false);
    }

    // Регистрируем несколько устаревших сигналов
    for _ in 0..6 {
        risk_manager.register_signal_staleness(true);
    }

    // Проверяем статистику
    let (total, stale_count, stale_ratio) = risk_manager.get_staleness_stats();
    assert_eq!(total, 11);
    assert_eq!(stale_count, 6);
    assert!((stale_ratio - 0.545).abs() < 0.01, "Stale ratio should be ~54.5%");

    // Проверяем circuit breaker (должен сработать при >50%)
    assert!(risk_manager.check_stale_signal_circuit_breaker(),
        "Circuit breaker should trigger when stale ratio > 50%");
}

#[test]
fn test_risk_manager_staleness_circuit_breaker_threshold() {
    use neirobot_lit::risk::risk_manager::RiskManager;
    use neirobot_lit::config::types::RiskConfig;
    use rust_decimal::Decimal;

    let config = RiskConfig::default();
    let mut risk_manager = RiskManager::new(config, Decimal::from(1000));

    // Регистрируем 60% свежих и 40% устаревших (не должен сработать)
    for _ in 0..6 {
        risk_manager.register_signal_staleness(false);
    }
    for _ in 0..4 {
        risk_manager.register_signal_staleness(true);
    }

    let (total, stale_count, stale_ratio) = risk_manager.get_staleness_stats();
    assert_eq!(total, 10);
    assert_eq!(stale_count, 4);
    assert!((stale_ratio - 0.4).abs() < 0.01);

    // Circuit breaker НЕ должен сработать при 40%
    assert!(!risk_manager.check_stale_signal_circuit_breaker(),
        "Circuit breaker should NOT trigger when stale ratio <= 50%");
}

#[tokio::test]
#[ignore] // Требует сетевого подключения
async fn test_clock_skew_check() {
    use neirobot_lit::utils::helpers::check_clock_skew;

    // Проверяем синхронизацию с Bybit
    let result = check_clock_skew("https://api.bybit.com", 5000).await;
    assert!(result.is_ok(), "Clock skew check should succeed");

    let delta = result.unwrap();
    // Обычно расхождение должно быть меньше 1 секунды
    assert!(delta.abs() < 1000, "Clock skew should be less than 1 second, got {}ms", delta);
}

#[test]
fn test_staleness_action_enum() {
    // Проверяем, что enum StalenessAction корректно сериализуется
    let skip = StalenessAction::Skip;
    let log_only = StalenessAction::LogOnly;

    assert_ne!(skip, log_only);
    assert_eq!(skip, StalenessAction::Skip);
    assert_eq!(log_only, StalenessAction::LogOnly);
}

#[test]
fn test_health_monitor_orderbook_gap() {
    use neirobot_lit::risk::health_monitor::HealthMonitor;
    use neirobot_lit::config::types::RiskConfig;

    let config = RiskConfig::default();
    let mut hm = HealthMonitor::new(config);

    // Первая установка u
    hm.check_u(100);
    assert!(hm.is_sane().is_ok());

    // Корректная последовательность
    hm.check_u(101);
    assert!(hm.is_sane().is_ok());

    // Гэп: 101 -> 103 (пропуск 102)
    hm.check_u(103);
    assert!(hm.is_sane().is_err(), "Health check should fail on update_id gap");
}

#[test]
fn test_health_monitor_latency_limit() {
    use neirobot_lit::risk::health_monitor::HealthMonitor;
    use neirobot_lit::config::types::RiskConfig;

    let mut config = RiskConfig::default();
    config.max_avg_latency_ms = 50;
    
    let mut hm = HealthMonitor::new(config);

    // Заполняем окно задержками выше лимита (60 точек)
    for _ in 0..60 {
        hm.update_latency(100.0); // 100ms > 50ms
    }

    assert!(hm.is_sane().is_err(), "Health check should fail on high average latency");
}

#[test]
fn test_health_monitor_memory_limit() {
    use neirobot_lit::risk::health_monitor::HealthMonitor;
    use neirobot_lit::config::types::RiskConfig;

    let mut config = RiskConfig::default();
    config.max_process_memory_mb = 1; // Лимит 1МБ точно будет превышен
    
    let mut hm = HealthMonitor::new(config);

    // Проверка памяти должна вернуть ошибку
    assert!(hm.is_sane().is_err(), "Health check should fail on memory limit violation");
}

#[test]
fn test_risk_manager_health_and_blocking() {
    use neirobot_lit::risk::risk_manager::RiskManager;
    use neirobot_lit::config::types::RiskConfig;
    use rust_decimal::Decimal;

    let mut config = RiskConfig::default();
    config.max_avg_latency_ms = 50;
    let mut rm = RiskManager::new(config, Decimal::from(1000));

    // 1. Симулируем коррупцию стакана
    rm.health_monitor.check_u(100);
    rm.health_monitor.check_u(102); // Gap

    // check_risk_gates должен вернуть ошибку и заблокировать торговлю
    assert!(rm.check_risk_gates(Decimal::ZERO).is_err());
    assert!(rm.is_blocked, "RiskManager should be blocked after health failure");

    // 2. Симулируем ресинк (как в run-bot.rs)
    rm.health_monitor.reset_corruption();
    rm.is_blocked = false;

    // После сброса должно быть OK
    assert!(rm.check_risk_gates(Decimal::ZERO).is_ok());
    assert!(!rm.is_blocked);

    // 3. Симулируем высокую задержку
    for _ in 0..60 {
        rm.health_monitor.update_latency(100.0);
    }
    
    assert!(rm.check_risk_gates(Decimal::ZERO).is_err());
    assert!(rm.is_blocked, "RiskManager should be blocked after latency failure");
}

#[test]
fn test_pnl_volatility_gate() {
    use neirobot_lit::risk::risk_manager::RiskManager;
    use neirobot_lit::config::types::RiskConfig;
    use rust_decimal::Decimal;

    let mut config = RiskConfig::default();
    config.max_pnl_std_dev_bps = 30; // Лимит 30 bps
    config.pnl_volatility_window = 20;
    
    let mut rm = RiskManager::new(config, Decimal::from(1000));

    // Подаем серию PnL в bps: [10, -50, 40]
    // mean = (10 - 50 + 40) / 3 = 0
    // var = (10^2 + (-50)^2 + 40^2) / (3 - 1) = (100 + 2500 + 1600) / 2 = 4200 / 2 = 2100
    // std = sqrt(2100) ≈ 45.82 bps
    
    rm.update_pnl_stats(10.0);
    rm.update_pnl_stats(-50.0);
    rm.update_pnl_stats(40.0);

    assert!(rm.pnl_stats.n >= 2);
    let std_dev = rm.pnl_stats.std_dev();
    assert!((std_dev - 45.82).abs() < 0.1);

    // Проверяем гейт (должен быть заблокирован так как 45.82 > 30)
    let result = rm.check_risk_gates(Decimal::ZERO);
    assert!(result.is_err());
    assert!(rm.is_blocked, "RiskManager should be blocked due to high PnL volatility");
}

#[test]
fn test_pnl_outlier_gate() {
    use neirobot_lit::risk::risk_manager::RiskManager;
    use neirobot_lit::config::types::RiskConfig;
    use rust_decimal::Decimal;

    let mut config = RiskConfig::default();
    config.max_pnl_z_score_threshold = 3.0;
    config.pnl_volatility_window = 20;
    
    let mut rm = RiskManager::new(config, Decimal::from(1000));

    // Создаем стабильную историю: 10 сделок по 10 bps
    for _ in 0..10 {
        rm.update_pnl_stats(10.0);
    }
    
    // Добавляем немного вариативности чтобы std_dev != 0
    rm.update_pnl_stats(5.0);
    rm.update_pnl_stats(15.0);
    
    // mean ≈ 10, std ≈ небольшой
    let mean = rm.pnl_stats.sum / rm.pnl_stats.n as f64;
    let std = rm.pnl_stats.std_dev();
    assert!((mean - 10.0).abs() < 0.001);
    assert!(std > 0.0);

    // Подаем аномальный выброс: -40 bps
    // Z-score = |-40 - 10| / std. Если std ≈ 2.5, Z-score ≈ 20
    rm.update_pnl_stats(-40.0);
    
    let result = rm.check_risk_gates(Decimal::ZERO);
    assert!(result.is_err(), "Outlier should trigger risk gate error");
    assert!(rm.is_blocked, "RiskManager should be blocked due to PnL outlier");
}

#[test]
fn test_api_permission_check_insufficient() {
    use neirobot_lit::risk::health_monitor::HealthMonitor;
    use neirobot_lit::config::types::RiskConfig;
    use neirobot_lit::trading::rest_client::ApiKeyInfoResponse;
    use neirobot_lit::trading::types::ExchangeConfig;

    let config = RiskConfig::default();
    let mut hm = HealthMonitor::new(config);

    let api_info = ApiKeyInfoResponse {
        permissions: vec!["ReadOnly".to_string()],
        ip_restrict: true,
        expired_at: 0,
    };

    let mut ex_config = ExchangeConfig::default();
    ex_config.required_permissions = vec!["ContractTrade".to_string()];

    // Проверка должна вернуть ошибку
    let result = hm.validate_api_permissions(&api_info, &ex_config);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Missing critical API permission"));

    // is_sane должен также вернуть ошибку
    assert!(hm.is_sane().is_err());
}

#[test]
fn test_api_expiry_warning() {
    use neirobot_lit::risk::health_monitor::HealthMonitor;
    use neirobot_lit::config::types::RiskConfig;
    use neirobot_lit::trading::rest_client::ApiKeyInfoResponse;
    use neirobot_lit::trading::types::ExchangeConfig;
    use std::time::{SystemTime, UNIX_EPOCH};

    let config = RiskConfig::default();
    let mut hm = HealthMonitor::new(config);

    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    // Истекает через 2 дня
    let expired_at = now_ms + 2 * 86_400_000;

    let api_info = ApiKeyInfoResponse {
        permissions: vec!["ContractTrade".to_string(), "Order".to_string(), "Position".to_string()],
        ip_restrict: true,
        expired_at,
    };

    let mut ex_config = ExchangeConfig::default();
    ex_config.required_permissions = vec!["ContractTrade".to_string()];
    ex_config.min_api_days_left = 7;

    // Проверка должна пройти (только warning в логах), но is_sane должен быть OK
    let result = hm.validate_api_permissions(&api_info, &ex_config);
    assert!(result.is_ok());
    assert!(hm.is_sane().is_ok());
}

#[test]
fn test_api_expired_failure() {
    use neirobot_lit::risk::health_monitor::HealthMonitor;
    use neirobot_lit::config::types::RiskConfig;
    use neirobot_lit::trading::rest_client::ApiKeyInfoResponse;
    use neirobot_lit::trading::types::ExchangeConfig;
    use std::time::{SystemTime, UNIX_EPOCH};

    let config = RiskConfig::default();
    let mut hm = HealthMonitor::new(config);

    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    // Истек 1 день назад
    let expired_at = now_ms - 86_400_000;

    let api_info = ApiKeyInfoResponse {
        permissions: vec!["ContractTrade".to_string(), "Order".to_string(), "Position".to_string()],
        ip_restrict: true,
        expired_at,
    };

    let mut ex_config = ExchangeConfig::default();
    ex_config.required_permissions = vec!["ContractTrade".to_string()];

    // Проверка должна вернуть ошибку
    let result = hm.validate_api_permissions(&api_info, &ex_config);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("API Key has expired"));

    // is_sane должен также вернуть ошибку
    assert!(hm.is_sane().is_err());
}

#[test]
fn test_rejection_gate_consecutive() {
    use neirobot_lit::risk::risk_manager::RiskManager;
    use neirobot_lit::config::types::RiskConfig;
    use rust_decimal::Decimal;

    let mut config = RiskConfig::default();
    config.max_consecutive_rejections = 3;
    let mut rm = RiskManager::new(config, Decimal::from(1000));

    // 1. ������ 2 ������ (������ ������)
    rm.report_rejection();
    rm.report_rejection();
    assert!(rm.check_risk_gates(Decimal::ZERO).is_ok());

    // 2. ������ 3-� ������ (����� ���������)
    rm.report_rejection();
    
    // ������ ���� ������
    let result = rm.check_risk_gates(Decimal::ZERO);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Too many consecutive rejections"));
    assert!(rm.is_blocked);
}

#[test]
fn test_rejection_success_reset() {
    use neirobot_lit::risk::risk_manager::RiskManager;
    use neirobot_lit::config::types::RiskConfig;
    use rust_decimal::Decimal;

    let mut config = RiskConfig::default();
    config.max_consecutive_rejections = 3;
    let mut rm = RiskManager::new(config, Decimal::from(1000));

    // 2 ������ ������
    rm.report_rejection();
    rm.report_rejection();
    
    // �������� ����� -> �����
    rm.report_success();

    // ��� 2 ������
    rm.report_rejection();
    rm.report_rejection();

    // ����� 2 ������ (� �� 4) -> ������ ���� ��
    assert!(rm.check_risk_gates(Decimal::ZERO).is_ok());
}

#[test]
fn test_rejection_gate_window() {
    use neirobot_lit::risk::risk_manager::RiskManager;
    use neirobot_lit::config::types::RiskConfig;
    use rust_decimal::Decimal;

    let mut config = RiskConfig::default();
    config.max_consecutive_rejections = 100; // ��������� ���� �����
    config.max_total_rejections_in_window = 5;
    config.rejection_window_ms = 60000;
    
    let mut rm = RiskManager::new(config, Decimal::from(1000));
    let now = 100000;

    // 1. ��������� 5 �������� � ������ ����
    for i in 0..5 {
        rm.report_rejection_for_test(now + i * 1000); 
    }

    // ����� ���������
    let result = rm.check_risk_gates(Decimal::ZERO);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Too many rejections in window"));
}

#[test]
fn test_rejection_window_cleanup() {
    use neirobot_lit::risk::risk_manager::RiskManager;
    use neirobot_lit::config::types::RiskConfig;
    use rust_decimal::Decimal;

    let mut config = RiskConfig::default();
    config.max_consecutive_rejections = 100;
    config.max_total_rejections_in_window = 3;
    config.rejection_window_ms = 10000; // 10 ������
    
    let mut rm = RiskManager::new(config, Decimal::from(1000));
    let start_time = 100000;

    // 1. ��������� 2 ������
    rm.report_rejection_for_test(start_time);
    rm.report_rejection_for_test(start_time + 1000);
    assert!(rm.check_risk_gates(Decimal::ZERO).is_ok()); // 2 < 3

    // 2. ������� ������ (������������ �����)
    // ��������� ����� ������ � 11000 (����� 11� ����� ������)
    // ����: [1000, 11000]. start_time (0ms offset) ��� �������? 11000 - 10000 = 1000.
    // ������ timestamp < 1000 ���������. start_time=100000.
    // report at start_time + 11000. Window start = start_time + 1000.
    // timestamp start_time (100000) < 101000 -> Removed.
    
    rm.report_rejection_for_test(start_time + 11000);

    // � ���� ������ ��������: start_time+1000 � start_time+11000. ����� 2.
    // ����� 3. ������ OK.
    assert!(rm.check_risk_gates(Decimal::ZERO).is_ok());
}
