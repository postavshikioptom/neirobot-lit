//! Тесты для системы защищенного аудита (Задача 217)

use neirobot_lit::utils::AuditLogger;
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

#[test]
fn test_audit_logger_initialization() {
    let temp_dir = TempDir::new().unwrap();
    let symbol = "TESTUSDT";
    let master_key = "test_master_key_12345";

    // Создаем временную директорию для теста
    let bot_dir = temp_dir.path().join("bots").join(symbol);
    fs::create_dir_all(&bot_dir).unwrap();

    // Инициализируем логгер
    let logger = AuditLogger::init(symbol, master_key);
    assert!(logger.is_ok(), "Failed to initialize AuditLogger");

    // Проверяем, что файл создан
    let audit_file = bot_dir.join("logs").join("security_audit.csv");
    assert!(audit_file.exists(), "Audit log file was not created");
}

#[test]
fn test_audit_logger_log_event() {
    let temp_dir = TempDir::new().unwrap();
    let symbol = "TESTUSDT";
    let master_key = "test_master_key_12345";

    let bot_dir = temp_dir.path().join("bots").join(symbol);
    fs::create_dir_all(&bot_dir).unwrap();

    let logger = AuditLogger::init(symbol, master_key).unwrap();

    // Логируем событие
    let result = logger.log_event(
        "System",
        "TEST_ACTION",
        "SUCCESS",
        "old_val",
        "new_val",
    );

    assert!(result.is_ok(), "Failed to log event");

    // Проверяем, что данные записаны в файл
    let audit_file = bot_dir.join("logs").join("security_audit.csv");
    let content = fs::read_to_string(&audit_file).unwrap();
    assert!(content.contains("TEST_ACTION"), "Event not found in audit log");
    assert!(content.contains("SUCCESS"), "Status not found in audit log");
}

#[test]
fn test_audit_logger_hash_chain() {
    let temp_dir = TempDir::new().unwrap();
    let symbol = "TESTUSDT";
    let master_key = "test_master_key_12345";

    let bot_dir = temp_dir.path().join("bots").join(symbol);
    fs::create_dir_all(&bot_dir).unwrap();

    let logger = AuditLogger::init(symbol, master_key).unwrap();

    // Логируем несколько событий
    logger.log_event("System", "ACTION1", "SUCCESS", "old1", "new1").unwrap();
    logger.log_event("System", "ACTION2", "SUCCESS", "old2", "new2").unwrap();
    logger.log_event("System", "ACTION3", "SUCCESS", "old3", "new3").unwrap();

    // Проверяем, что все события записаны
    let audit_file = bot_dir.join("logs").join("security_audit.csv");
    let content = fs::read_to_string(&audit_file).unwrap();
    
    assert!(content.contains("ACTION1"), "ACTION1 not found");
    assert!(content.contains("ACTION2"), "ACTION2 not found");
    assert!(content.contains("ACTION3"), "ACTION3 not found");

    // Проверяем, что хеши присутствуют
    let lines: Vec<&str> = content.lines().collect();
    assert!(lines.len() >= 4, "Expected at least 4 lines (header + 3 events)");
    
    // Каждая строка должна содержать хеш (64 символа hex)
    for line in lines.iter().skip(1) {
        let parts: Vec<&str> = line.split(',').collect();
        assert!(parts.len() >= 7, "Expected at least 7 fields in CSV");
        let hash = parts[6].trim_matches('"');
        assert_eq!(hash.len(), 64, "Hash should be 64 characters (SHA256 hex)");
    }
}

#[test]
fn test_audit_logger_config_decryption_logging() {
    let temp_dir = TempDir::new().unwrap();
    let symbol = "TESTUSDT";
    let master_key = "test_master_key_12345";

    let bot_dir = temp_dir.path().join("bots").join(symbol);
    fs::create_dir_all(&bot_dir).unwrap();

    let logger = AuditLogger::init(symbol, master_key).unwrap();

    // Логируем успешную расшифровку
    let result = logger.log_config_decryption(true, None);
    assert!(result.is_ok());

    // Логируем неудачную расшифровку
    let result = logger.log_config_decryption(false, Some("Wrong password"));
    assert!(result.is_ok());

    // Проверяем, что события записаны
    let audit_file = bot_dir.join("logs").join("security_audit.csv");
    let content = fs::read_to_string(&audit_file).unwrap();
    
    assert!(content.contains("CONFIG_DECRYPTION"), "CONFIG_DECRYPTION not found");
    assert!(content.contains("SUCCESS"), "SUCCESS status not found");
    assert!(content.contains("FAILURE"), "FAILURE status not found");
}

#[test]
fn test_audit_logger_risk_gate_logging() {
    let temp_dir = TempDir::new().unwrap();
    let symbol = "TESTUSDT";
    let master_key = "test_master_key_12345";

    let bot_dir = temp_dir.path().join("bots").join(symbol);
    fs::create_dir_all(&bot_dir).unwrap();

    let logger = AuditLogger::init(symbol, master_key).unwrap();

    // Логируем срабатывание защитного гейта
    let result = logger.log_risk_gate("PRICE_DEVIATION", true, "Price deviation 5% > 2%");
    assert!(result.is_ok());

    // Логируем нормальное состояние гейта
    let result = logger.log_risk_gate("MAX_DAILY_LOSS", false, "Daily loss 0.5% < 2%");
    assert!(result.is_ok());

    // Проверяем, что события записаны
    let audit_file = bot_dir.join("logs").join("security_audit.csv");
    let content = fs::read_to_string(&audit_file).unwrap();
    
    assert!(content.contains("RISK_GATE_PRICE_DEVIATION"), "RISK_GATE_PRICE_DEVIATION not found");
    assert!(content.contains("RISK_GATE_MAX_DAILY_LOSS"), "RISK_GATE_MAX_DAILY_LOSS not found");
    assert!(content.contains("TRIGGERED"), "TRIGGERED status not found");
    assert!(content.contains("OK"), "OK status not found");
}

#[test]
fn test_audit_logger_recovery_from_file() {
    let temp_dir = TempDir::new().unwrap();
    let symbol = "TESTUSDT";
    let master_key = "test_master_key_12345";

    let bot_dir = temp_dir.path().join("bots").join(symbol);
    fs::create_dir_all(&bot_dir).unwrap();

    // Первая инициализация
    {
        let logger = AuditLogger::init(symbol, master_key).unwrap();
        logger.log_event("System", "ACTION1", "SUCCESS", "old1", "new1").unwrap();
    }

    // Вторая инициализация (должна восстановить последний хеш)
    {
        let logger = AuditLogger::init(symbol, master_key).unwrap();
        logger.log_event("System", "ACTION2", "SUCCESS", "old2", "new2").unwrap();
    }

    // Проверяем, что оба события записаны
    let audit_file = bot_dir.join("logs").join("security_audit.csv");
    let content = fs::read_to_string(&audit_file).unwrap();
    
    assert!(content.contains("ACTION1"), "ACTION1 not found");
    assert!(content.contains("ACTION2"), "ACTION2 not found");

    // Проверяем, что хеши разные (цепочка работает)
    let lines: Vec<&str> = content.lines().collect();
    assert!(lines.len() >= 3, "Expected at least 3 lines (header + 2 events)");
    
    let hash1 = lines[1].split(',').last().unwrap().trim_matches('"');
    let hash2 = lines[2].split(',').last().unwrap().trim_matches('"');
    assert_ne!(hash1, hash2, "Hashes should be different (chain should work)");
}
