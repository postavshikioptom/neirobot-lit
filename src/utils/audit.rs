//! Модуль защищенного аудита критических событий
//! 
//! Реализует систему логирования с криптографической цепочкой HMAC-SHA256.
//! Каждая запись связана с предыдущей через хеш, образуя неизменяемую цепочку.

use anyhow::{Context, Result};
use chrono::Utc;
use csv::Writer;
use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tracing::info;

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

type HmacSha256 = Hmac<Sha256>;

/// Структура для логирования аудита с криптографической цепочкой
#[derive(Clone)]
pub struct AuditLogger {
    inner: Arc<Mutex<AuditLoggerInner>>,
}

struct AuditLoggerInner {
    file: File,
    last_hash: Vec<u8>,
    master_key: Vec<u8>,
    file_path: PathBuf,
}

/// Запись аудита
#[derive(serde::Serialize, Debug, Clone)]
pub struct AuditRecord {
    pub timestamp: String,
    pub actor: String,
    pub action: String,
    pub status: String,
    pub old_value: String,
    pub new_value: String,
    pub hash: String,
}

impl AuditLogger {
    /// Инициализирует AuditLogger для символа
    /// 
    /// # Аргументы
    /// * `symbol` - Символ торговой пары (например, "BTCUSDT")
    /// * `master_key` - Мастер-ключ для HMAC (из задачи 216)
    pub fn init(symbol: &str, master_key: &str) -> Result<Self> {
        let log_dir = PathBuf::from(format!("bots/{}/logs", symbol));
        
        // Создаем директорию, если её нет
        std::fs::create_dir_all(&log_dir)
            .context("Failed to create logs directory")?;

        let file_path = log_dir.join("security_audit.csv");
        
        // Открываем файл в режиме append
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)
            .context("Failed to open audit log file")?;

        // Устанавливаем права доступа 600 (только владелец) на Unix
        #[cfg(unix)]
        {
            let permissions = std::fs::Permissions::from_mode(0o600);
            std::fs::set_permissions(&file_path, permissions)
                .context("Failed to set file permissions")?;
        }

        // Восстанавливаем последний хеш из файла
        let last_hash = Self::recover_last_hash(&file_path, master_key)?;

        info!("AuditLogger initialized for symbol: {}", symbol);

        Ok(AuditLogger {
            inner: Arc::new(Mutex::new(AuditLoggerInner {
                file,
                last_hash,
                master_key: master_key.as_bytes().to_vec(),
                file_path,
            })),
        })
    }

    /// Экранирует значение для CSV (предотвращение CSV-инъекций)
    fn escape_csv_value(value: &str) -> String {
        if value.contains(',') || value.contains('"') || value.contains('\n') {
            format!("\"{}\"", value.replace('"', "\"\""))
        } else {
            value.to_string()
        }
    }

    /// Восстанавливает последний хеш из файла
    fn recover_last_hash(file_path: &PathBuf, master_key: &str) -> Result<Vec<u8>> {
        // Проверяем, существует ли файл и не пуст ли он
        if !file_path.exists() {
            // Генезис-хеш: HMAC-SHA256 от master_key
            let mut mac = HmacSha256::new_from_slice(master_key.as_bytes())
                .context("Failed to create HMAC")?;
            mac.update(b"GENESIS");
            return Ok(mac.finalize().into_bytes().to_vec());
        }

        let file = File::open(file_path)
            .context("Failed to open audit log file for recovery")?;
        
        let reader = BufReader::new(file);
        let mut last_line = String::new();
        
        // Читаем последнюю строку
        for line in reader.lines() {
            let line = line.context("Failed to read line")?;
            if !line.is_empty() && !line.starts_with("timestamp") {
                last_line = line;
            }
        }

        if last_line.is_empty() {
            // Файл пуст или содержит только заголовок
            let mut mac = HmacSha256::new_from_slice(master_key.as_bytes())
                .context("Failed to create HMAC")?;
            mac.update(b"GENESIS");
            return Ok(mac.finalize().into_bytes().to_vec());
        }

        // Парсим последнюю строку через CSV reader для правильной обработки экранированных значений
        let mut reader = csv::Reader::from_reader(last_line.as_bytes());
        if let Some(record) = reader.records().next() {
            if let Ok(record) = record {
                if record.len() >= 7 {
                    let hash_str = record.get(6).unwrap_or("").trim_matches('"');
                    if let Ok(hash) = hex::decode(hash_str) {
                        return Ok(hash);
                    }
                }
            }
        }

        // Если не можем парсить, используем генезис-хеш
        let mut mac = HmacSha256::new_from_slice(master_key.as_bytes())
            .context("Failed to create HMAC")?;
        mac.update(b"GENESIS");
        Ok(mac.finalize().into_bytes().to_vec())
    }

    /// Логирует событие в аудит
    /// 
    /// # Аргументы
    /// * `actor` - Субъект действия ("System" или "Startup")
    /// * `action` - Описание действия
    /// * `status` - Статус ("SUCCESS" или "FAILURE")
    /// * `old_value` - Старое значение
    /// * `new_value` - Новое значение
    pub fn log_event(
        &self,
        actor: &str,
        action: &str,
        status: &str,
        old_value: &str,
        new_value: &str,
    ) -> Result<()> {
        let mut inner = self.inner.lock().unwrap();

        let timestamp = Utc::now().to_rfc3339();
        
        // Экранируем значения для предотвращения CSV-инъекций
        let actor_escaped = Self::escape_csv_value(actor);
        let action_escaped = Self::escape_csv_value(action);
        let status_escaped = Self::escape_csv_value(status);
        let old_value_escaped = Self::escape_csv_value(old_value);
        let new_value_escaped = Self::escape_csv_value(new_value);
        
        // Формируем строку для хеширования: prev_hash + row_data
        let row_data = format!(
            "{},{},{},{},{},{}",
            timestamp, actor_escaped, action_escaped, status_escaped, old_value_escaped, new_value_escaped
        );

        // Вычисляем HMAC-SHA256: HMAC(key: master_key, data: prev_hash + row_data)
        let mut mac = HmacSha256::new_from_slice(&inner.master_key)
            .context("Failed to create HMAC")?;
        
        mac.update(&inner.last_hash);
        mac.update(row_data.as_bytes());
        
        let current_hash = mac.finalize().into_bytes().to_vec();
        let hash_hex = hex::encode(&current_hash);

        // Создаем запись
        let record = AuditRecord {
            timestamp,
            actor: actor.to_string(),
            action: action.to_string(),
            status: status.to_string(),
            old_value: old_value.to_string(),
            new_value: new_value.to_string(),
            hash: hash_hex,
        };

        // Записываем в CSV
        {
            let mut wtr = Writer::from_writer(&mut inner.file);
            wtr.serialize(&record)
                .context("Failed to serialize audit record")?;
            wtr.flush()
                .context("Failed to flush CSV writer")?;
        } // Writer уничтожается здесь, освобождая заимствование

        // Выполняем fsync для гарантии записи на диск
        inner.file.sync_all()
            .context("Failed to sync file")?;

        // Обновляем последний хеш
        inner.last_hash = current_hash;

        info!(
            "Audit event logged: actor={}, action={}, status={}",
            actor, action, status
        );

        Ok(())
    }

    /// Логирует попытку расшифровки конфига
    pub fn log_config_decryption(&self, success: bool, error_msg: Option<&str>) -> Result<()> {
        let status = if success { "SUCCESS" } else { "FAILURE" };
        let error_detail = error_msg.unwrap_or("N/A");
        
        self.log_event(
            "System",
            "CONFIG_DECRYPTION",
            status,
            "",
            error_detail,
        )
    }

    /// Логирует срабатывание защитного гейта
    pub fn log_risk_gate(&self, gate_name: &str, triggered: bool, details: &str) -> Result<()> {
        let status = if triggered { "TRIGGERED" } else { "OK" };
        
        self.log_event(
            "System",
            &format!("RISK_GATE_{}", gate_name),
            status,
            "",
            details,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_audit_logger_init() {
        let temp_dir = TempDir::new().unwrap();
        let symbol = "TESTUSDT";
        let master_key = "test_master_key_12345";

        // Создаем временную директорию для теста
        let bot_dir = temp_dir.path().join("bots").join(symbol);
        std::fs::create_dir_all(&bot_dir).unwrap();

        // Инициализируем логгер
        let logger = AuditLogger::init(symbol, master_key);
        assert!(logger.is_ok());
    }

    #[test]
    fn test_log_event() {
        let temp_dir = TempDir::new().unwrap();
        let symbol = "TESTUSDT";
        let master_key = "test_master_key_12345";

        let bot_dir = temp_dir.path().join("bots").join(symbol);
        std::fs::create_dir_all(&bot_dir).unwrap();

        let logger = AuditLogger::init(symbol, master_key).unwrap();

        // Логируем событие
        let result = logger.log_event(
            "System",
            "TEST_ACTION",
            "SUCCESS",
            "old_val",
            "new_val",
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_hash_chain() {
        let temp_dir = TempDir::new().unwrap();
        let symbol = "TESTUSDT";
        let master_key = "test_master_key_12345";

        let bot_dir = temp_dir.path().join("bots").join(symbol);
        std::fs::create_dir_all(&bot_dir).unwrap();

        let logger = AuditLogger::init(symbol, master_key).unwrap();

        // Логируем несколько событий
        logger.log_event("System", "ACTION1", "SUCCESS", "old1", "new1").unwrap();
        logger.log_event("System", "ACTION2", "SUCCESS", "old2", "new2").unwrap();

        // Проверяем, что хеши разные
        let inner1 = logger.inner.lock().unwrap();
        let hash1 = inner1.last_hash.clone();
        drop(inner1);

        logger.log_event("System", "ACTION3", "SUCCESS", "old3", "new3").unwrap();
        
        let inner2 = logger.inner.lock().unwrap();
        let hash2 = inner2.last_hash.clone();

        assert_ne!(hash1, hash2);
    }
}
