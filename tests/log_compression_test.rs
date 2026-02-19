use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

#[test]
fn test_log_compression_reduces_file_size() {
    // Arrange: Создаем тестовый файл с логами
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let log_file = temp_dir.path().join("test.log");
    
    // Генерируем тестовые данные (повторяющиеся строки для хорошего сжатия)
    let test_data = "2024-02-14T10:00:00Z INFO Starting bot\n".repeat(1000);
    fs::write(&log_file, &test_data).expect("Failed to write test log");
    
    let original_size = fs::metadata(&log_file)
        .expect("Failed to get metadata")
        .len();
    
    // Act: Сжимаем файл
    let compressed_data = zstd::encode_all(test_data.as_bytes(), 1)
        .expect("Failed to compress");
    
    let compressed_file = temp_dir.path().join("test.log.zst");
    fs::write(&compressed_file, &compressed_data)
        .expect("Failed to write compressed file");
    
    let compressed_size = fs::metadata(&compressed_file)
        .expect("Failed to get compressed metadata")
        .len();
    
    // Assert: Проверяем, что размер уменьшился
    assert!(
        compressed_size < original_size,
        "Compressed size ({}) should be less than original ({})",
        compressed_size,
        original_size
    );
    
    // Проверяем коэффициент сжатия (должен быть значительным для повторяющихся данных)
    let compression_ratio = compressed_size as f64 / original_size as f64;
    assert!(
        compression_ratio < 0.5,
        "Compression ratio should be less than 50%, got {:.1}%",
        compression_ratio * 100.0
    );
    
    // Cleanup: Удаляем оригинальный файл
    fs::remove_file(&log_file).expect("Failed to remove original file");
    
    // Проверяем, что оригинальный файл удален
    assert!(!log_file.exists(), "Original file should be deleted");
    assert!(compressed_file.exists(), "Compressed file should exist");
}

#[test]
fn test_log_decompression_restores_original() {
    // Arrange: Создаем тестовые данные
    let test_data = "2024-02-14T10:00:00Z ERROR Critical error occurred\n".repeat(100);
    
    // Act: Сжимаем и распаковываем
    let compressed = zstd::encode_all(test_data.as_bytes(), 1)
        .expect("Failed to compress");
    
    let decompressed = zstd::decode_all(&compressed[..])
        .expect("Failed to decompress");
    
    let decompressed_str = String::from_utf8(decompressed)
        .expect("Failed to convert to string");
    
    // Assert: Проверяем, что данные совпадают
    assert_eq!(
        test_data, decompressed_str,
        "Decompressed data should match original"
    );
}

#[test]
fn test_compression_with_different_levels() {
    let test_data = "Log entry\n".repeat(500);
    
    // Тестируем разные уровни сжатия
    for level in 1..=3 {
        let compressed = zstd::encode_all(test_data.as_bytes(), level)
            .expect(&format!("Failed to compress with level {}", level));
        
        // Проверяем, что сжатие работает
        assert!(
            compressed.len() < test_data.len(),
            "Level {}: Compressed size should be less than original",
            level
        );
        
        // Проверяем, что можно распаковать
        let decompressed = zstd::decode_all(&compressed[..])
            .expect(&format!("Failed to decompress level {}", level));
        
        assert_eq!(
            test_data.as_bytes(),
            &decompressed[..],
            "Level {}: Decompressed data should match original",
            level
        );
    }
}

#[test]
fn test_empty_file_compression() {
    // Arrange: Пустой файл
    let test_data = "";
    
    // Act: Сжимаем пустой файл
    let compressed = zstd::encode_all(test_data.as_bytes(), 1)
        .expect("Failed to compress empty file");
    
    // Assert: Проверяем, что можно распаковать
    let decompressed = zstd::decode_all(&compressed[..])
        .expect("Failed to decompress empty file");
    
    assert_eq!(
        test_data.as_bytes(),
        &decompressed[..],
        "Empty file should decompress correctly"
    );
}
