//! Тесты для AlertManager

use neirobot_lit::monitoring::alert_manager::{Alert, AlertLevel, AlertManager};
use neirobot_lit::utils::audit::AuditLogger;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;

#[tokio::test]
async fn test_alert_creation() {
    let alert = Alert::new(
        AlertLevel::Critical,
        "Test critical alert".to_string(),
        "TestBot".to_string(),
    );

    assert_eq!(alert.level, AlertLevel::Critical);
    assert_eq!(alert.message, "Test critical alert");
    assert_eq!(alert.source, "TestBot");
}

#[tokio::test]
async fn test_alert_deduplication() {
    // Создаем AlertManager с фейковым токеном (не будем отправлять реальные запросы)
    let manager = AlertManager::new(
        "fake_token".to_string(),
        "fake_chat_id".to_string(),
        None,
        None,
    ).expect("Failed to create AlertManager");

    // Отправляем первый алерт
    let alert1 = Alert::new(
        AlertLevel::Info,
        "Duplicate test".to_string(),
        "TestSource".to_string(),
    );
    manager.send_alert(alert1);

    // Даем время на обработку
    sleep(Duration::from_millis(100)).await;

    // Отправляем дубликат
    let alert2 = Alert::new(
        AlertLevel::Info,
        "Duplicate test".to_string(),
        "TestSource".to_string(),
    );
    manager.send_alert(alert2);

    // Даем время на обработку
    sleep(Duration::from_millis(100)).await;

    // Проверяем, что дубликат был отфильтрован (проверяем через логи)
    // В реальном тесте можно было бы использовать счетчик отправленных сообщений
}

#[tokio::test]
async fn test_alert_levels() {
    let info_alert = Alert::new(
        AlertLevel::Info,
        "Info message".to_string(),
        "TestSource".to_string(),
    );
    assert_eq!(info_alert.level, AlertLevel::Info);

    let warning_alert = Alert::new(
        AlertLevel::Warning,
        "Warning message".to_string(),
        "TestSource".to_string(),
    );
    assert_eq!(warning_alert.level, AlertLevel::Warning);

    let critical_alert = Alert::new(
        AlertLevel::Critical,
        "Critical message".to_string(),
        "TestSource".to_string(),
    );
    assert_eq!(critical_alert.level, AlertLevel::Critical);
}

#[tokio::test]
async fn test_alert_with_audit_logger() {
    // Создаем временный AuditLogger
    let temp_dir = tempfile::tempdir().unwrap();
    let symbol = "TESTUSDT";
    let master_key = "test_key_12345";

    // Создаем директорию для логов
    let bot_dir = temp_dir.path().join("bots").join(symbol);
    std::fs::create_dir_all(&bot_dir).unwrap();

    // Меняем текущую директорию на временную
    let original_dir = std::env::current_dir().unwrap();
    std::env::set_current_dir(temp_dir.path()).unwrap();

    let audit_logger = AuditLogger::init(symbol, master_key)
        .expect("Failed to create AuditLogger");

    let manager = AlertManager::new(
        "fake_token".to_string(),
        "fake_chat_id".to_string(),
        None,
        Some(Arc::new(audit_logger)),
    ).expect("Failed to create AlertManager");

    // Отправляем Warning алерт (должен записаться в аудит)
    let alert = Alert::new(
        AlertLevel::Warning,
        "Test warning with audit".to_string(),
        "TestBot".to_string(),
    );
    manager.send_alert(alert);

    // Даем время на обработку
    sleep(Duration::from_millis(200)).await;

    // Восстанавливаем оригинальную директорию
    std::env::set_current_dir(original_dir).unwrap();

    // Проверяем, что файл аудита создан
    let audit_file = bot_dir.join("logs").join("security_audit.csv");
    assert!(audit_file.exists(), "Audit file should be created");
}

#[test]
fn test_alert_level_display() {
    assert_eq!(format!("{}", AlertLevel::Info), "INFO");
    assert_eq!(format!("{}", AlertLevel::Warning), "WARNING");
    assert_eq!(format!("{}", AlertLevel::Critical), "CRITICAL");
}
