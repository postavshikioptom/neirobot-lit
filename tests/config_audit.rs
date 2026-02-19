// Тесты для проверки версионирования и аудита конфигураций (Задача 184)

use neirobot_lit::config::loader::{
    compute_config_hash, compute_config_hash_from_file, generate_config_diff, 
    log_config_changes, backup_config
};
use std::fs;
use std::path::Path;
use tempfile::TempDir;

#[test]
fn test_compute_config_hash() {
    // Тест вычисления SHA-256 хэша конфигурации
    let config_content = r#"
[general]
symbol = "BTCUSDT"
mode = "live"
"#;
    
    let hash = compute_config_hash(config_content);
    
    // Проверяем, что хэш не пустой и имеет правильный формат (64 символа для SHA-256)
    assert!(!hash.is_empty(), "Hash should not be empty");
    assert_eq!(hash.len(), 64, "SHA-256 hash should be 64 characters long");
    
    // Проверяем, что одинаковое содержимое дает одинаковый хэш
    let hash2 = compute_config_hash(config_content);
    assert_eq!(hash, hash2, "Same content should produce same hash");
    
    // Проверяем, что разное содержимое дает разный хэш
    let different_content = r#"
[general]
symbol = "ETHUSDT"
mode = "live"
"#;
    let hash3 = compute_config_hash(different_content);
    assert_ne!(hash, hash3, "Different content should produce different hash");
}

#[test]
fn test_compute_config_hash_from_file() {
    // Тест вычисления хэша из файла
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("config.toml");
    
    let config_content = r#"
[general]
symbol = "BTCUSDT"
"#;
    
    fs::write(&config_path, config_content).unwrap();
    
    let hash = compute_config_hash_from_file(&config_path).unwrap();
    let expected_hash = compute_config_hash(config_content);
    
    assert_eq!(hash, expected_hash, "Hash from file should match hash from content");
}

#[test]
fn test_generate_config_diff() {
    // Тест генерации diff между двумя версиями конфигурации
    let old_config = r#"[general]
symbol = "BTCUSDT"
mode = "live"
max_position_size = 100.0

[trading]
maker_offset = 1.0
"#;
    
    let new_config = r#"[general]
symbol = "BTCUSDT"
mode = "live"
max_position_size = 150.0

[trading]
maker_offset = 1.5
"#;
    
    let diff = generate_config_diff(old_config, new_config);
    
    // Проверяем, что diff содержит информацию об изменениях
    assert!(!diff.is_empty(), "Diff should not be empty");
    assert!(diff.contains("100.0") || diff.contains("150.0"), "Diff should contain changed values");
    assert!(diff.contains("1.0") || diff.contains("1.5"), "Diff should contain changed offset");
}

#[test]
fn test_generate_config_diff_no_changes() {
    // Тест diff когда нет изменений
    let config = r#"[general]
symbol = "BTCUSDT"
"#;
    
    let diff = generate_config_diff(config, config);
    
    // Diff может быть пустым или содержать только контекстные строки
    // Главное, что нет строк с + или -
    assert!(!diff.contains("+[general]"), "Diff should not have additions for unchanged content");
}

#[test]
fn test_backup_config() {
    // Тест создания резервной копии конфигурации
    let temp_dir = TempDir::new().unwrap();
    let bot_dir = temp_dir.path().join("BTCUSDT");
    fs::create_dir_all(&bot_dir).unwrap();
    
    let config_path = bot_dir.join("config.toml");
    let config_content = r#"
[general]
symbol = "BTCUSDT"
"#;
    
    fs::write(&config_path, config_content).unwrap();
    
    // Создаем резервную копию
    let backup_path = backup_config(&config_path).unwrap();
    
    // Проверяем, что резервная копия создана
    assert!(backup_path.exists(), "Backup file should exist");
    
    // Проверяем, что содержимое резервной копии совпадает с оригиналом
    let backup_content = fs::read_to_string(&backup_path).unwrap();
    assert_eq!(backup_content, config_content, "Backup content should match original");
    
    // Проверяем, что резервная копия находится в config_history
    assert!(backup_path.to_string_lossy().contains("config_history"), 
        "Backup should be in config_history directory");
}

#[test]
fn test_backup_config_creates_history_dir() {
    // Тест что backup_config создает директорию config_history если её нет
    let temp_dir = TempDir::new().unwrap();
    let bot_dir = temp_dir.path().join("ETHUSDT");
    fs::create_dir_all(&bot_dir).unwrap();
    
    let config_path = bot_dir.join("config.toml");
    fs::write(&config_path, "[general]\nsymbol = \"ETHUSDT\"").unwrap();
    
    // Проверяем, что config_history не существует
    let history_dir = bot_dir.join("config_history");
    assert!(!history_dir.exists(), "config_history should not exist initially");
    
    // Создаем резервную копию
    let backup_path = backup_config(&config_path).unwrap();
    
    // Проверяем, что config_history теперь существует
    assert!(history_dir.exists(), "config_history directory should be created");
    assert!(backup_path.exists(), "Backup file should exist");
}

#[test]
fn test_multiple_backups() {
    // Тест создания нескольких резервных копий с разными временными метками
    let temp_dir = TempDir::new().unwrap();
    let bot_dir = temp_dir.path().join("BTCUSDT");
    fs::create_dir_all(&bot_dir).unwrap();
    
    let config_path = bot_dir.join("config.toml");
    
    // Создаем первую резервную копию
    fs::write(&config_path, "[general]\nsymbol = \"BTCUSDT\"\nversion = 1").unwrap();
    let backup1 = backup_config(&config_path).unwrap();
    
    // Небольшая задержка для разных временных меток
    std::thread::sleep(std::time::Duration::from_millis(100));
    
    // Обновляем конфигурацию и создаем вторую резервную копию
    fs::write(&config_path, "[general]\nsymbol = \"BTCUSDT\"\nversion = 2").unwrap();
    let backup2 = backup_config(&config_path).unwrap();
    
    // Проверяем, что обе резервные копии существуют и разные
    assert!(backup1.exists(), "First backup should exist");
    assert!(backup2.exists(), "Second backup should exist");
    assert_ne!(backup1, backup2, "Backups should have different names (timestamps)");
    
    // Проверяем содержимое
    let content1 = fs::read_to_string(&backup1).unwrap();
    let content2 = fs::read_to_string(&backup2).unwrap();
    assert!(content1.contains("version = 1"), "First backup should have version 1");
    assert!(content2.contains("version = 2"), "Second backup should have version 2");
}

#[test]
fn test_config_hash_consistency() {
    // Тест что хэш конфигурации консистентен при повторных вычислениях
    let config = r#"
[general]
symbol = "BTCUSDT"
mode = "live"

[trading]
maker_offset = 1.0
taker_offset = 1.5

[risk]
max_position_size = 100.0
max_drawdown_pct = 0.05
"#;
    
    let hash1 = compute_config_hash(config);
    let hash2 = compute_config_hash(config);
    let hash3 = compute_config_hash(config);
    
    assert_eq!(hash1, hash2, "Hash should be consistent");
    assert_eq!(hash2, hash3, "Hash should be consistent");
}


// ============================================================================
// Задача 184: Тестирование SIGHUP обработки (Config Reload on Signal)
// ============================================================================

#[test]
fn test_config_hash_change_detection() {
    // Тест что изменение конфигурации правильно обнаруживается через хэш
    let old_config = r#"
[general]
symbol = "BTCUSDT"
max_position_size = 100.0
"#;
    
    let new_config = r#"
[general]
symbol = "BTCUSDT"
max_position_size = 150.0
"#;
    
    let old_hash = compute_config_hash(old_config);
    let new_hash = compute_config_hash(new_config);
    
    // Хэши должны быть разными
    assert_ne!(old_hash, new_hash, "Different configs should have different hashes");
    
    // Проверяем что хэши консистентны
    let old_hash2 = compute_config_hash(old_config);
    let new_hash2 = compute_config_hash(new_config);
    
    assert_eq!(old_hash, old_hash2, "Same config should always produce same hash");
    assert_eq!(new_hash, new_hash2, "Same config should always produce same hash");
}

#[test]
fn test_sighup_mock_config_reload() {
    // Mock-тест для SIGHUP обработки
    // Симулируем сценарий: конфиг изменился, нужно обнаружить и залогировать
    
    let temp_dir = tempfile::TempDir::new().unwrap();
    let config_path = temp_dir.path().join("config.toml");
    
    // Исходная конфигурация
    let initial_config = r#"
[general]
symbol = "BTCUSDT"
mode = "live"
max_position_size = 100.0

[trading]
maker_offset = 1.0
"#;
    
    std::fs::write(&config_path, initial_config).unwrap();
    
    // Вычисляем начальный хэш
    let initial_hash = compute_config_hash_from_file(&config_path).unwrap();
    
    // Симулируем SIGHUP: конфиг изменился
    let updated_config = r#"
[general]
symbol = "BTCUSDT"
mode = "live"
max_position_size = 150.0

[trading]
maker_offset = 1.5
"#;
    
    std::fs::write(&config_path, updated_config).unwrap();
    
    // Читаем новый конфиг и вычисляем хэш
    let new_content = std::fs::read_to_string(&config_path).unwrap();
    let new_hash = compute_config_hash(&new_content);
    
    // Проверяем что хэш изменился
    assert_ne!(initial_hash, new_hash, "Config hash should change after update");
    
    // Проверяем что diff содержит информацию об изменениях
    let diff = generate_config_diff(initial_config, updated_config);
    assert!(!diff.is_empty(), "Diff should not be empty");
    assert!(diff.contains("100.0") || diff.contains("150.0"), "Diff should contain changed values");
}

#[test]
fn test_config_reload_sequence() {
    // Тест последовательности перезагрузок конфигурации
    let temp_dir = tempfile::TempDir::new().unwrap();
    let config_path = temp_dir.path().join("config.toml");
    
    // Версия 1
    let config_v1 = "[general]\nversion = 1\n";
    std::fs::write(&config_path, config_v1).unwrap();
    let hash_v1 = compute_config_hash_from_file(&config_path).unwrap();
    
    // Версия 2
    let config_v2 = "[general]\nversion = 2\n";
    std::fs::write(&config_path, config_v2).unwrap();
    let hash_v2 = compute_config_hash_from_file(&config_path).unwrap();
    
    // Версия 3
    let config_v3 = "[general]\nversion = 3\n";
    std::fs::write(&config_path, config_v3).unwrap();
    let hash_v3 = compute_config_hash_from_file(&config_path).unwrap();
    
    // Все хэши должны быть разными
    assert_ne!(hash_v1, hash_v2, "Version 1 and 2 should have different hashes");
    assert_ne!(hash_v2, hash_v3, "Version 2 and 3 should have different hashes");
    assert_ne!(hash_v1, hash_v3, "Version 1 and 3 should have different hashes");
}

#[test]
fn test_config_backup_permissions() {
    // Тест что резервные копии создаются с правильными правами доступа
    let temp_dir = tempfile::TempDir::new().unwrap();
    let bot_dir = temp_dir.path().join("BTCUSDT");
    std::fs::create_dir_all(&bot_dir).unwrap();
    
    let config_path = bot_dir.join("config.toml");
    std::fs::write(&config_path, "[general]\nsymbol = \"BTCUSDT\"\n").unwrap();
    
    // Создаем резервную копию
    let backup_path = backup_config(&config_path).unwrap();
    
    // Проверяем что резервная копия существует
    assert!(backup_path.exists(), "Backup file should exist");
    
    // Проверяем права доступа (на Unix системах)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let metadata = std::fs::metadata(&backup_path).unwrap();
        let permissions = metadata.permissions();
        let mode = permissions.mode();
        
        // Проверяем что права 0o600 (rw-------)
        assert_eq!(mode & 0o777, 0o600, "Backup should have 600 permissions");
    }
}

#[test]
fn test_config_diff_multiline() {
    // Тест diff для многострочных изменений
    let old_config = r#"[section1]
key1 = "value1"
key2 = "value2"

[section2]
key3 = "value3"
"#;
    
    let new_config = r#"[section1]
key1 = "value1_modified"
key2 = "value2"

[section2]
key3 = "value3"
key4 = "value4_new"
"#;
    
    let diff = generate_config_diff(old_config, new_config);
    
    // Проверяем что diff содержит информацию об изменениях
    assert!(!diff.is_empty(), "Diff should not be empty");
    assert!(diff.contains("value1") || diff.contains("modified"), "Diff should show modified values");
}
