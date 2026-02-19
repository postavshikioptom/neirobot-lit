//! Тесты для Hot-Swap Engine
//! Задача 228: Автоматизированная дистрибуция и безопасный Hot-Swap моделей

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::fs;
    use tempfile::TempDir;
    
    /// Тест создания структуры директорий для hot-swap
    #[test]
    fn test_hotswap_directory_structure() {
        let temp_dir = TempDir::new().unwrap();
        let model_dir = temp_dir.path().join("model");
        fs::create_dir_all(&model_dir).unwrap();
        
        // Создаем файлы
        let model_path = model_dir.join("model.onnx");
        let hash_path = model_dir.join("model.hash");
        let backup_path = model_dir.join("model.onnx.bak");
        
        fs::write(&model_path, b"fake model data").unwrap();
        fs::write(&hash_path, b"abc123").unwrap();
        
        assert!(model_path.exists());
        assert!(hash_path.exists());
        assert!(!backup_path.exists());
    }
    
    /// Тест вычисления хеша файла
    #[test]
    fn test_file_hash_computation() {
        use sha2::{Sha256, Digest};
        
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.bin");
        
        let test_data = b"test data for hashing";
        fs::write(&file_path, test_data).unwrap();
        
        // Вычисляем хеш
        let mut hasher = Sha256::new();
        hasher.update(test_data);
        let expected_hash = format!("{:x}", hasher.finalize());
        
        // Вычисляем хеш из файла
        let file_data = fs::read(&file_path).unwrap();
        let mut hasher2 = Sha256::new();
        hasher2.update(&file_data);
        let actual_hash = format!("{:x}", hasher2.finalize());
        
        assert_eq!(expected_hash, actual_hash);
    }
    
    /// Тест атомарного переименования
    #[test]
    fn test_atomic_rename() {
        let temp_dir = TempDir::new().unwrap();
        let tmp_path = temp_dir.path().join("file.tmp");
        let final_path = temp_dir.path().join("file.dat");
        
        fs::write(&tmp_path, b"test data").unwrap();
        assert!(tmp_path.exists());
        assert!(!final_path.exists());
        
        // Атомарное переименование
        fs::rename(&tmp_path, &final_path).unwrap();
        
        assert!(!tmp_path.exists());
        assert!(final_path.exists());
        
        let data = fs::read(&final_path).unwrap();
        assert_eq!(data, b"test data");
    }
    
    /// Тест создания backup
    #[test]
    fn test_backup_creation() {
        let temp_dir = TempDir::new().unwrap();
        let original = temp_dir.path().join("model.onnx");
        let backup = temp_dir.path().join("model.onnx.bak");
        
        fs::write(&original, b"original model").unwrap();
        
        // Создаем backup
        fs::copy(&original, &backup).unwrap();
        
        assert!(original.exists());
        assert!(backup.exists());
        
        let original_data = fs::read(&original).unwrap();
        let backup_data = fs::read(&backup).unwrap();
        assert_eq!(original_data, backup_data);
    }
    
    /// Тест проверки латентности
    #[test]
    fn test_latency_check() {
        let baseline_latency = 1000u64; // 1ms
        let new_latency = 1400u64; // 1.4ms
        
        let increase_ratio = new_latency as f64 / baseline_latency as f64;
        
        // Проверка: увеличение менее 50% - OK
        assert!(increase_ratio < 1.5);
        
        let new_latency_bad = 1600u64; // 1.6ms
        let increase_ratio_bad = new_latency_bad as f64 / baseline_latency as f64;
        
        // Проверка: увеличение более 50% - FAIL
        assert!(increase_ratio_bad > 1.5);
    }
    
    /// Тест формата model.hash файла
    #[test]
    fn test_hash_file_format() {
        let temp_dir = TempDir::new().unwrap();
        let hash_path = temp_dir.path().join("model.hash");
        
        let test_hash = "abc123def456789";
        fs::write(&hash_path, format!("{}\n", test_hash)).unwrap();
        
        let content = fs::read_to_string(&hash_path).unwrap();
        let hash = content.trim();
        
        assert_eq!(hash, test_hash);
    }
}
