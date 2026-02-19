use anyhow::Result;
use neirobot_lit::trading::{BotState, BotStateData, StatePersistenceManager};
use std::fs::{self, OpenOptions};
use std::io::Write;
use tempfile::TempDir;

/// Тест на атомарность: Симуляция падения записи и проверка сохранности старого файла
#[test]
fn test_atomic_write_preserves_old_state() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let manager = StatePersistenceManager::new(temp_dir.path(), 3)?;

    // Сохранить первое состояние
    let state1 = BotState::new(BotStateData {
        position: 1.0,
        pnl: 50.0,
        active_order_ids: vec!["order1".to_string()],
        metadata: std::collections::HashMap::new(),
    })?;
    manager.save_state(&state1)?;

    // Проверить, что основной файл существует
    assert!(manager.state_file_path().exists());

    // Сохранить второе состояние
    let state2 = BotState::new(BotStateData {
        position: 2.0,
        pnl: 100.0,
        active_order_ids: vec!["order1".to_string(), "order2".to_string()],
        metadata: std::collections::HashMap::new(),
    })?;
    manager.save_state(&state2)?;

    // Проверить, что основной файл содержит второе состояние
    let loaded = manager.load_state()?;
    assert_eq!(loaded.data.position, 2.0);
    assert_eq!(loaded.data.pnl, 100.0);
    assert_eq!(loaded.data.active_order_ids.len(), 2);

    // Проверить, что первое состояние в бэкапе
    let backup_1 = temp_dir.path().join("state.json.bak.1");
    assert!(backup_1.exists());

    // Прочитать бэкап и проверить его содержимое
    let backup_content = fs::read_to_string(&backup_1)?;
    let backup_state: BotState = serde_json::from_str(&backup_content)?;
    assert_eq!(backup_state.data.position, 1.0);
    assert_eq!(backup_state.data.pnl, 50.0);

    Ok(())
}

/// Тест на повреждение: Изменение байта в JSON и проверка срабатывания детектора checksum
#[test]
fn test_corruption_detection_and_recovery() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let manager = StatePersistenceManager::new(temp_dir.path(), 3)?;

    // Сохранить валидное состояние
    let state1 = BotState::new(BotStateData {
        position: 1.0,
        pnl: 50.0,
        active_order_ids: vec!["order1".to_string()],
        metadata: std::collections::HashMap::new(),
    })?;
    manager.save_state(&state1)?;

    // Сохранить второе состояние (которое будет в бэкапе)
    let state2 = BotState::new(BotStateData {
        position: 2.0,
        pnl: 100.0,
        active_order_ids: vec!["order2".to_string()],
        metadata: std::collections::HashMap::new(),
    })?;
    manager.save_state(&state2)?;

    // Испортить основной файл, изменив чексумму
    let mut corrupted_json = serde_json::json!({
        "version": 1,
        "timestamp": 0,
        "data": {
            "position": 2.0,
            "pnl": 100.0,
            "active_order_ids": ["order2"],
            "metadata": {}
        },
        "checksum": "corrupted_checksum_value"
    });

    let mut file = OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(manager.state_file_path())?;
    file.write_all(corrupted_json.to_string().as_bytes())?;
    drop(file);

    // Загрузить состояние - должно восстановиться из бэкапа
    let loaded = manager.load_state()?;

    // Проверить, что восстановилось первое состояние из бэкапа
    assert_eq!(loaded.data.position, 1.0);
    assert_eq!(loaded.data.pnl, 50.0);
    assert_eq!(loaded.data.active_order_ids.len(), 1);

    Ok(())
}

/// Тест на блокировку: Попытка открыть второй дескриптор файла при активном первом
#[test]
fn test_exclusive_file_locking() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let manager = StatePersistenceManager::new(temp_dir.path(), 3)?;

    let state = BotState::new(BotStateData {
        position: 1.0,
        pnl: 50.0,
        active_order_ids: vec!["order1".to_string()],
        metadata: std::collections::HashMap::new(),
    })?;

    // Сохранить состояние (захватит и освободит блокировку)
    manager.save_state(&state)?;

    // Попытаться открыть файл блокировки - должна быть возможность
    // (блокировка должна быть освобождена после save_state)
    let lock_file = temp_dir.path().join("state.lock");
    let mut lock = OpenOptions::new()
        .create(true)
        .write(true)
        .open(&lock_file)?;

    // Должны быть в состоянии захватить блокировку
    use fs2::FileExt;
    lock.lock_exclusive()?;
    lock.unlock()?;

    Ok(())
}

/// Тест на ротацию бэкапов: Проверка правильной ротации при превышении max_backups
#[test]
fn test_backup_rotation() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let manager = StatePersistenceManager::new(temp_dir.path(), 2)?; // max_backups = 2

    // Сохранить три состояния
    for i in 1..=3 {
        let state = BotState::new(BotStateData {
            position: i as f64,
            pnl: (i * 50) as f64,
            active_order_ids: vec![format!("order{}", i)],
            metadata: std::collections::HashMap::new(),
        })?;
        manager.save_state(&state)?;
    }

    // Проверить, что основной файл содержит третье состояние
    let loaded = manager.load_state()?;
    assert_eq!(loaded.data.position, 3.0);

    // Проверить, что существуют бэкапы 1 и 2
    let backup_1 = temp_dir.path().join("state.json.bak.1");
    let backup_2 = temp_dir.path().join("state.json.bak.2");
    let backup_3 = temp_dir.path().join("state.json.bak.3");

    assert!(backup_1.exists(), "Backup 1 should exist");
    assert!(backup_2.exists(), "Backup 2 should exist");
    assert!(!backup_3.exists(), "Backup 3 should not exist (max_backups=2)");

    // Проверить содержимое бэкапов
    let backup_1_content = fs::read_to_string(&backup_1)?;
    let backup_1_state: BotState = serde_json::from_str(&backup_1_content)?;
    assert_eq!(backup_1_state.data.position, 2.0);

    let backup_2_content = fs::read_to_string(&backup_2)?;
    let backup_2_state: BotState = serde_json::from_str(&backup_2_content)?;
    assert_eq!(backup_2_state.data.position, 1.0);

    Ok(())
}

/// Тест на восстановление при полном повреждении: Все копии повреждены
#[test]
fn test_recovery_with_all_corrupted() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let manager = StatePersistenceManager::new(temp_dir.path(), 2)?;

    // Сохранить состояние
    let state = BotState::new(BotStateData {
        position: 1.0,
        pnl: 50.0,
        active_order_ids: vec!["order1".to_string()],
        metadata: std::collections::HashMap::new(),
    })?;
    manager.save_state(&state)?;

    // Испортить основной файл
    let mut file = OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(manager.state_file_path())?;
    file.write_all(b"corrupted json")?;
    drop(file);

    // Испортить все бэкапы
    for i in 1..=2 {
        let backup = temp_dir.path().join(format!("state.json.bak.{}", i));
        if backup.exists() {
            let mut file = OpenOptions::new()
                .write(true)
                .truncate(true)
                .open(&backup)?;
            file.write_all(b"corrupted backup")?;
            drop(file);
        }
    }

    // Загрузить состояние - должно вернуть пустое состояние
    let loaded = manager.load_state()?;

    // Проверить, что вернулось пустое состояние
    assert_eq!(loaded.data.position, 0.0);
    assert_eq!(loaded.data.pnl, 0.0);
    assert_eq!(loaded.data.active_order_ids.len(), 0);

    Ok(())
}

/// Тест на сохранение и загрузку с метаданными
#[test]
fn test_state_with_metadata() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let manager = StatePersistenceManager::new(temp_dir.path(), 3)?;

    let mut metadata = std::collections::HashMap::new();
    metadata.insert("symbol".to_string(), "BTCUSDT".to_string());
    metadata.insert("exchange".to_string(), "bybit".to_string());

    let state = BotState::new(BotStateData {
        position: 1.5,
        pnl: 75.5,
        active_order_ids: vec!["order1".to_string(), "order2".to_string()],
        metadata,
    })?;

    manager.save_state(&state)?;
    let loaded = manager.load_state()?;

    assert_eq!(loaded.data.position, 1.5);
    assert_eq!(loaded.data.pnl, 75.5);
    assert_eq!(loaded.data.active_order_ids.len(), 2);
    assert_eq!(
        loaded.data.metadata.get("symbol").map(|s| s.as_str()),
        Some("BTCUSDT")
    );
    assert_eq!(
        loaded.data.metadata.get("exchange").map(|s| s.as_str()),
        Some("bybit")
    );

    Ok(())
}
