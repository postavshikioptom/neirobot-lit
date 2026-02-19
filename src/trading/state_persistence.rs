use anyhow::{anyhow, Result};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use tracing::{error, info, warn};

/// Структура для хранения данных состояния бота
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

/// Полная структура состояния с версией, временем и чексуммой
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BotState {
    /// Версия схемы для миграций
    pub version: u32,
    /// Время сохранения (Unix timestamp в миллисекундах)
    pub timestamp: u64,
    /// Основные данные состояния
    pub data: BotStateData,
    /// SHA256 чексумма поля data для проверки целостности
    pub checksum: String,
}

impl BotState {
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
    pub fn save_state(&self, state: &BotState) -> Result<()> {
        // Захватить эксклюзивную блокировку на файл состояния
        let lock_file = self.state_dir.join("state.lock");
        let mut lock = OpenOptions::new()
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
    fn _save_state_locked(&self, state: &BotState) -> Result<()> {
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
    pub fn load_state(&self) -> Result<BotState> {
        // Захватить эксклюзивную блокировку на файл состояния
        let lock_file = self.state_dir.join("state.lock");
        let mut lock = OpenOptions::new()
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
    fn _load_state_locked(&self) -> Result<BotState> {
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
        error!("All state files are corrupted or missing. Initializing empty state.");
        Ok(BotState::new(BotStateData::default())?)
    }

    /// Загрузить и проверить целостность файла состояния
    fn _load_and_verify_state(&self, path: &Path) -> Result<BotState> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        let state: BotState = serde_json::from_str(&contents)?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_state_checksum_verification() -> Result<()> {
        let data = BotStateData {
            position: 1.5,
            pnl: 100.0,
            active_order_ids: vec!["order1".to_string()],
            metadata: std::collections::HashMap::new(),
        };

        let state = BotState::new(data)?;
        assert!(state.verify_checksum()?);

        Ok(())
    }

    #[test]
    fn test_state_checksum_corruption_detection() -> Result<()> {
        let data = BotStateData {
            position: 1.5,
            pnl: 100.0,
            active_order_ids: vec!["order1".to_string()],
            metadata: std::collections::HashMap::new(),
        };

        let mut state = BotState::new(data)?;
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
        let state1 = BotState::new(BotStateData {
            position: 1.0,
            pnl: 50.0,
            active_order_ids: vec!["order1".to_string()],
            metadata: std::collections::HashMap::new(),
        })?;
        manager.save_state(&state1)?;

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
        let state1 = BotState::new(BotStateData {
            position: 1.0,
            pnl: 50.0,
            active_order_ids: vec!["order1".to_string()],
            metadata: std::collections::HashMap::new(),
        })?;
        manager.save_state(&state1)?;

        // Сохранить второе состояние
        let state2 = BotState::new(BotStateData {
            position: 2.0,
            pnl: 100.0,
            active_order_ids: vec!["order2".to_string()],
            metadata: std::collections::HashMap::new(),
        })?;
        manager.save_state(&state2)?;

        // Испортить основной файл
        let mut corrupted_json = serde_json::json!({
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

        let state = BotState::new(BotStateData {
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
}
