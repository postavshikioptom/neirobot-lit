use anyhow::{anyhow, Result};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use tracing::{error, info, warn};

// Импортируем BotState из types.rs (pub type BotState = RiskState)
use crate::trading::types::BotState;

/// Структура для хранения данных состояния бота (для StatePersistenceManager)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BotStateData {
    /// Текущая позиция (в контрактах или базовых единицах)
    pub position: f64,
    /// Нереализованный PnL
    pub pnl: f64,
    /// Список активных ID ордеров
    pub active_order_ids: Vec<String>,
    /// Дополнительные метаданные
    pub metadata: std::collections::HashMap<String, String>,
}

impl Default for BotStateData {
    fn default() -> Self {
        Self {
            position: 0.0,
            pnl: 0.0,
            active_order_ids: Vec::new(),
            metadata: std::collections::HashMap::new(),
        }
    }
}

/// Полная структура состояния с версией, временем и чексуммой (для StatePersistenceManager)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentState {
    /// Версия схемы для миграций
    pub version: u32,
    /// Время сохранения (Unix timestamp в миллисекундах)
    pub timestamp: u64,
    /// Основные данные состояния
    pub data: BotStateData,
    /// SHA256 чексумма поля data для проверки целостности
    pub checksum: String,
}

impl PersistentState {
    /// Создать новое состояние с вычислением чексуммы
    pub fn new(data: BotStateData) -> Result<Self> {
        let checksum = Self::compute_checksum(&data)?;
        Ok(Self {
            version: 1,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_millis() as u64,
            data,
            checksum,
        })
    }

    /// Вычислить SHA256 чексумму данных состояния
    fn compute_checksum(data: &BotStateData) -> Result<String> {
        let json_str = serde_json::to_string(data)?;
        let mut hasher = Sha256::new();
        hasher.update(json_str.as_bytes());
        let result = hasher.finalize();
        Ok(format!("{:x}", result))
    }

    /// Проверить целостность состояния
    pub fn verify_checksum(&self) -> Result<bool> {
        let computed = Self::compute_checksum(&self.data)?;
        Ok(computed == self.checksum)
    }
}

/// Менеджер персистентности состояния
pub struct StatePersistenceManager {
    state_dir: PathBuf,
    state_file: PathBuf,
    max_backups: u32,
}

impl StatePersistenceManager {
    /// Создать новый менеджер персистентности
    pub fn new(state_dir: impl AsRef<Path>, max_backups: u32) -> Result<Self> {
        let state_dir = state_dir.as_ref().to_path_buf();
        fs::create_dir_all(&state_dir)?;

        let state_file = state_dir.join("state.json");

        Ok(Self {
            state_dir,
            state_file,
            max_backups,
        })
    }

    /// Сохранить состояние атомарно с ротацией бэкапов
    pub fn save_state(&self, state: &PersistentState) -> Result<()> {
        // Захватить эксклюзивную блокировку на файл состояния
        let lock_file = self.state_dir.join("state.lock");
        let lock = OpenOptions::new()
            .create(true)
            .write(true)
            .open(&lock_file)?;

        lock.lock_exclusive()?;

        let result = self._save_state_locked(state);

        // Освободить блокировку
        let _ = lock.unlock();

        result
    }

    /// Внутренняя функция сохранения (вызывается с захватанной блокировкой)
    fn _save_state_locked(&self, state: &PersistentState) -> Result<()> {
        // Выполнить ротацию бэкапов перед записью нового состояния
        self._rotate_backups()?;

        // Переместить текущий state.json в state.json.bak.1
        if self.state_file.exists() {
            let backup_1 = self.state_dir.join("state.json.bak.1");
            fs::rename(&self.state_file, &backup_1)?;
        }

        // Использовать tempfile для атомарной записи
        let temp_file = tempfile::NamedTempFile::new_in(&self.state_dir)?;
        let temp_path = temp_file.path().to_path_buf();

        // Записать JSON в временный файл
        let json_str = serde_json::to_string_pretty(state)?;
        let mut file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&temp_path)?;
        file.write_all(json_str.as_bytes())?;
        file.sync_all()?;
        drop(file);

        // Атомарно переместить временный файл в state.json
        fs::rename(&temp_path, &self.state_file)?;

        info!(
            "State saved successfully: position={}, pnl={}, orders={}",
            state.data.position,
            state.data.pnl,
            state.data.active_order_ids.len()
        );

        Ok(())
    }

    /// Выполнить ротацию бэкапов
    fn _rotate_backups(&self) -> Result<()> {
        // Ротировать существующие бэкапы в обратном порядке
        for i in (1..self.max_backups).rev() {
            let old_backup = self.state_dir.join(format!("state.json.bak.{}", i));
            let new_backup = self.state_dir.join(format!("state.json.bak.{}", i + 1));

            if old_backup.exists() {
                if new_backup.exists() {
                    fs::remove_file(&new_backup)?;
                }
                fs::rename(&old_backup, &new_backup)?;
            }
        }

        // Удалить самый старый бэкап, если превышен лимит
        let oldest_backup = self
            .state_dir
            .join(format!("state.json.bak.{}", self.max_backups + 1));
        if oldest_backup.exists() {
            fs::remove_file(&oldest_backup)?;
        }

        Ok(())
    }

    /// Загрузить состояние с блокировкой и проверкой целостности
    pub fn load_state(&self) -> Result<PersistentState> {
        // Захватить эксклюзивную блокировку на файл состояния
        let lock_file = self.state_dir.join("state.lock");
        let lock = OpenOptions::new()
            .create(true)
            .write(true)
            .open(&lock_file)?;

        lock.lock_exclusive()?;

        let result = self._load_state_locked();

        // Освободить блокировку
        let _ = lock.unlock();

        result
    }

    /// Внутренняя функция загрузки (вызывается с захватанной блокировкой)
    fn _load_state_locked(&self) -> Result<PersistentState> {
        // Попытаться загрузить основной файл состояния
        if self.state_file.exists() {
            match self._load_and_verify_state(&self.state_file) {
                Ok(state) => {
                    info!("State loaded successfully from main file");
                    return Ok(state);
                }
                Err(e) => {
                    warn!(
                        "Failed to load main state file: {}. Trying backups...",
                        e
                    );
                }
            }
        }

        // Попытаться загрузить последний валидный бэкап
        for i in 1..=self.max_backups {
            let backup_file = self.state_dir.join(format!("state.json.bak.{}", i));
            if backup_file.exists() {
                match self._load_and_verify_state(&backup_file) {
                    Ok(state) => {
                        warn!(
                            "State recovered from backup {}: position={}, pnl={}",
                            i, state.data.position, state.data.pnl
                        );
                        return Ok(state);
                    }
                    Err(e) => {
                        warn!("Backup {} is corrupted: {}", i, e);
                    }
                }
            }
        }

        // Все копии повреждены или отсутствуют
        if self.state_file.exists() {
            error!("All state files (main and backups) are corrupted. Manual synchronization required.");
            return Err(anyhow!("All state files corrupted"));
        } else {
            info!("No state files found. Initializing empty state.");
            Ok(PersistentState::new(BotStateData::default())?)
        }
    }

    /// Загрузить и проверить целостность файла состояния
    fn _load_and_verify_state(&self, path: &Path) -> Result<PersistentState> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        let state: PersistentState = serde_json::from_str(&contents)?;

        // Проверить целостность
        if !state.verify_checksum()? {
            return Err(anyhow!(
                "State checksum verification failed for {}",
                path.display()
            ));
        }

        Ok(state)
    }

    /// Получить путь к файлу состояния
    pub fn state_file_path(&self) -> &Path {
        &self.state_file
    }

    /// Получить путь к директории состояния
    pub fn state_dir_path(&self) -> &Path {
        &self.state_dir
    }

    /// Удалить все файлы состояния (для тестирования)
    #[cfg(test)]
    pub fn clear_all(&self) -> Result<()> {
        if self.state_file.exists() {
            fs::remove_file(&self.state_file)?;
        }
        for i in 1..=self.max_backups {
            let backup = self.state_dir.join(format!("state.json.bak.{}", i));
            if backup.exists() {
                fs::remove_file(&backup)?;
            }
        }
        Ok(())
    }
}

// ============================================================================
// Функции для работы с BotState из types.rs (Задача 107)
// ============================================================================

/// Сохранить состояние бота в файл атомарно
/// 
/// Использует паттерн write-to-temp + rename для обеспечения атомарности.
/// Это предотвращает повреждение файла при внезапном отключении питания.
pub fn save_state(state: &BotState, path: &Path) -> Result<()> {
    let tmp_path = path.with_extension("tmp");
    
    // Создаем директории, если их нет
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| anyhow!("Failed to create directories: {}", e))?;
    }
    
    let file = File::create(&tmp_path).map_err(|e| anyhow!("Failed to create tmp file: {}", e))?;
    
    // Пишем красиво для дебага
    serde_json::to_writer_pretty(file, state)?;
    
    // Атомарный перенос
    fs::rename(tmp_path, path).map_err(|e| anyhow!("Failed to rename state file: {}", e))?;
    Ok(())
}

/// Загрузить состояние бота из файла
/// 
/// Если файл не существует, возвращает состояние по умолчанию.
pub fn load_state(path: &Path) -> Result<BotState> {
    if !path.exists() {
        return Ok(BotState::default());
    }
    let file = File::open(path)?;
    let state = serde_json::from_reader(file)?;
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_persistent_state_checksum_verification() -> Result<()> {
        let data = BotStateData {
            position: 1.5,
            pnl: 100.0,
            active_order_ids: vec!["order1".to_string()],
            metadata: std::collections::HashMap::new(),
        };

        let state = PersistentState::new(data)?;
        assert!(state.verify_checksum()?);

        Ok(())
    }

    #[test]
    fn test_persistent_state_checksum_corruption_detection() -> Result<()> {
        let data = BotStateData {
            position: 1.5,
            pnl: 100.0,
            active_order_ids: vec!["order1".to_string()],
            metadata: std::collections::HashMap::new(),
        };

        let mut state = PersistentState::new(data)?;
        // Испортить чексумму
        state.checksum = "corrupted_checksum".to_string();
        assert!(!state.verify_checksum()?);

        Ok(())
    }

    #[test]
    fn test_atomic_write_and_rotation() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let manager = StatePersistenceManager::new(temp_dir.path(), 3)?;

        // Сохранить первое состояние
        let state1 = PersistentState::new(BotStateData {
            position: 1.0,
            pnl: 50.0,
            active_order_ids: vec!["order1".to_string()],
            metadata: std::collections::HashMap::new(),
        })?;
        manager.save_state(&state1)?;

        // Сохранить второе состояние
        let state2 = PersistentState::new(BotStateData {
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

        // Проверить, что первое состояние в бэкапе
        let backup_1 = temp_dir.path().join("state.json.bak.1");
        assert!(backup_1.exists());

        Ok(())
    }

    #[test]
    fn test_corruption_recovery() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let manager = StatePersistenceManager::new(temp_dir.path(), 3)?;

        // Сохранить валидное состояние
        let state1 = PersistentState::new(BotStateData {
            position: 1.0,
            pnl: 50.0,
            active_order_ids: vec!["order1".to_string()],
            metadata: std::collections::HashMap::new(),
        })?;
        manager.save_state(&state1)?;

        // Сохранить второе состояние
        let state2 = PersistentState::new(BotStateData {
            position: 2.0,
            pnl: 100.0,
            active_order_ids: vec!["order2".to_string()],
            metadata: std::collections::HashMap::new(),
        })?;
        manager.save_state(&state2)?;

        // Испортить основной файл
        let corrupted_json = serde_json::json!({
            "version": 1,
            "timestamp": 0,
            "data": {
                "position": 2.0,
                "pnl": 100.0,
                "active_order_ids": ["order2"],
                "metadata": {}
            },
            "checksum": "wrong_checksum"
        });
        let mut file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(manager.state_file_path())?;
        file.write_all(corrupted_json.to_string().as_bytes())?;
        drop(file);

        // Загрузить состояние - должно восстановиться из бэкапа
        let loaded = manager.load_state()?;
        assert_eq!(loaded.data.position, 1.0);
        assert_eq!(loaded.data.pnl, 50.0);

        Ok(())
    }

    #[test]
    fn test_exclusive_locking() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let manager = StatePersistenceManager::new(temp_dir.path(), 3)?;

        let state = PersistentState::new(BotStateData {
            position: 1.0,
            pnl: 50.0,
            active_order_ids: vec!["order1".to_string()],
            metadata: std::collections::HashMap::new(),
        })?;

        // Сохранить состояние (захватит блокировку)
        manager.save_state(&state)?;

        // Попытаться открыть файл блокировки - должна быть возможность
        // (блокировка должна быть освобождена после save_state)
        let lock_file = temp_dir.path().join("state.lock");
        let mut lock = OpenOptions::new()
            .create(true)
            .write(true)
            .open(&lock_file)?;

        // Должны быть в состоянии захватить блокировку
        lock.lock_exclusive()?;
        lock.unlock()?;

        Ok(())
    }

    // Тесты для функций save_state/load_state (Задача 107)
    #[test]
    fn test_save_and_load_bot_state() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let state_path = temp_dir.path().join("bot_state.json");

        // Создаем состояние
        let state = BotState {
            symbol: "BTCUSDT".to_string(),
            position_size: rust_decimal::Decimal::from_f64(1.5).unwrap(),
            avg_price: rust_decimal::Decimal::from_f64(50000.0).unwrap(),
            cumulative_pnl: rust_decimal::Decimal::from_f64(100.0).unwrap(),
            active_orders: std::collections::HashMap::new(),
            day_start_pnl: rust_decimal::Decimal::ZERO,
            last_pnl_reset_ts: 0,
            recent_trade_timestamps: vec![],
            last_update_ts: 1234567890,
            loss_streak: 0,
            last_loss_timestamp_ms: 0,
            pending_slice_qty: None,
            pending_slice_side: None,
            pending_slice_signal: None,
            pending_slice_probs: None,
        };

        // Сохраняем
        save_state(&state, &state_path)?;
        assert!(state_path.exists());

        // Загружаем
        let loaded = load_state(&state_path)?;
        assert_eq!(loaded.symbol, "BTCUSDT");
        assert_eq!(loaded.position_size, rust_decimal::Decimal::from_f64(1.5).unwrap());

        Ok(())
    }

    #[test]
    fn test_load_state_nonexistent() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let state_path = temp_dir.path().join("nonexistent.json");

        // Загружаем несуществующий файл - должно вернуть Default
        let loaded = load_state(&state_path)?;
        assert_eq!(loaded.symbol, ""); // Default для String
        assert!(loaded.position_size.is_zero());

        Ok(())
    }

    #[test]
    fn test_save_state_atomic() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let state_path = temp_dir.path().join("atomic_state.json");

        let state = BotState::default();
        save_state(&state, &state_path)?;

        // Проверяем, что tmp файл не остался
        let tmp_path = state_path.with_extension("tmp");
        assert!(!tmp_path.exists());
        assert!(state_path.exists());

        Ok(())
    }
}
