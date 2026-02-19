use anyhow::{Result, Context};
use std::path::Path;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

/// Получить текущий Unix timestamp в секундах
fn get_unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Получить путь к файлу heartbeat для бота
fn get_heartbeat_path(bot_path: &Path, symbol: &str) -> std::path::PathBuf {
    bot_path.join("state").join("liveness.heartbeat")
}

/// Записать текущий timestamp в файл heartbeat
/// Файл создается с правами доступа 640 (rw-r-----)
pub fn write_heartbeat(bot_path: &Path, symbol: &str) -> Result<()> {
    let state_dir = bot_path.join("state");
    
    // Создаем директорию state если её нет
    if !state_dir.exists() {
        fs::create_dir_all(&state_dir)
            .context("Failed to create state directory")?;
    }
    
    let heartbeat_path = get_heartbeat_path(bot_path, symbol);
    let timestamp = get_unix_timestamp();
    
    // Записываем timestamp в файл
    fs::write(&heartbeat_path, timestamp.to_string())
        .context("Failed to write heartbeat file")?;
    
    // Устанавливаем права доступа 640 (rw-r-----)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = fs::Permissions::from_mode(0o640);
        fs::set_permissions(&heartbeat_path, perms)
            .context("Failed to set heartbeat file permissions")?;
    }
    
    Ok(())
}

/// Проверить возраст файла heartbeat
/// Возвращает true если файл существует и не старше max_age_secs
pub fn check_heartbeat_age(bot_path: &Path, symbol: &str, max_age_secs: u64) -> Result<bool> {
    let heartbeat_path = get_heartbeat_path(bot_path, symbol);
    
    if !heartbeat_path.exists() {
        return Ok(false);
    }
    
    let content = fs::read_to_string(&heartbeat_path)
        .context("Failed to read heartbeat file")?;
    
    let last_heartbeat: u64 = content.trim().parse()
        .context("Failed to parse heartbeat timestamp")?;
    
    let current_time = get_unix_timestamp();
    let age = current_time.saturating_sub(last_heartbeat);
    
    Ok(age <= max_age_secs)
}

/// Получить возраст файла heartbeat в секундах
pub fn get_heartbeat_age(bot_path: &Path, symbol: &str) -> Result<u64> {
    let heartbeat_path = get_heartbeat_path(bot_path, symbol);
    
    if !heartbeat_path.exists() {
        return Ok(u64::MAX);
    }
    
    let content = fs::read_to_string(&heartbeat_path)
        .context("Failed to read heartbeat file")?;
    
    let last_heartbeat: u64 = content.trim().parse()
        .context("Failed to parse heartbeat timestamp")?;
    
    let current_time = get_unix_timestamp();
    let age = current_time.saturating_sub(last_heartbeat);
    
    Ok(age)
}

/// Инициализировать heartbeat при старте бота
/// Проверяет наличие старого heartbeat и логирует recovery если нужно
pub fn initialize_heartbeat(bot_path: &Path, symbol: &str) -> Result<()> {
    let heartbeat_path = get_heartbeat_path(bot_path, symbol);
    
    // Проверяем наличие старого heartbeat
    if heartbeat_path.exists() {
        match get_heartbeat_age(bot_path, symbol) {
            Ok(age) if age > 60 => {
                // Heartbeat старше 1 минуты - логируем recovery
                tracing::warn!(
                    "[Liveness] Recovery after crash detected: heartbeat age = {} seconds",
                    age
                );
            }
            Ok(_) => {
                // Heartbeat свежий - нормальный перезапуск
                tracing::debug!("[Liveness] Normal restart detected");
            }
            Err(e) => {
                tracing::warn!("[Liveness] Failed to check heartbeat age: {}", e);
            }
        }
    }
    
    // Создаем/обновляем heartbeat файл
    write_heartbeat(bot_path, symbol)?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_write_and_read_heartbeat() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let bot_path = temp_dir.path();
        
        write_heartbeat(bot_path, "BTCUSDT")?;
        
        let age = get_heartbeat_age(bot_path, "BTCUSDT")?;
        assert!(age < 5, "Heartbeat age should be less than 5 seconds");
        
        Ok(())
    }

    #[test]
    fn test_check_heartbeat_age() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let bot_path = temp_dir.path();
        
        write_heartbeat(bot_path, "BTCUSDT")?;
        
        // Heartbeat должен быть свежим
        assert!(check_heartbeat_age(bot_path, "BTCUSDT", 60)?);
        
        // Heartbeat не должен быть старше 0 секунд (очень строгий лимит)
        assert!(!check_heartbeat_age(bot_path, "BTCUSDT", 0)?);
        
        Ok(())
    }

    #[test]
    fn test_initialize_heartbeat() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let bot_path = temp_dir.path();
        
        initialize_heartbeat(bot_path, "BTCUSDT")?;
        
        let age = get_heartbeat_age(bot_path, "BTCUSDT")?;
        assert!(age < 5);
        
        Ok(())
    }
}
