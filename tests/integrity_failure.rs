//! Тест проверки целостности модели (Задача 185)
//! Проверяет, что OnnxEngine корректно обнаруживает несоответствие хеша модели

use std::fs;
use tempfile::tempdir;
use neirobot_lit::ml::onnx::OnnxEngine;
use neirobot_lit::config::types::OnnxConfig;

#[test]
fn test_integrity_failure() {
    let dir = tempdir().unwrap();
    let model_path = dir.path().join("model.onnx");
    let metadata_path = dir.path().join("metadata.json");
    
    // Создаем фейковый файл модели
    fs::write(&model_path, "fake onnx content").unwrap();
    
    // Создаем metadata.json с НЕПРАВИЛЬНЫМ хешем
    fs::write(&metadata_path, r#"{
        "metadata_version": "1.0",
        "git_hash": "abc",
        "export_timestamp": "2024",
        "model_name": "test",
        "onnx_hash": "WRONG_HASH_EXPECTED",
        "version": "1.0",
        "model_params": {
            "architecture": "LiT",
            "seq_len": 10,
            "n_levels": 5,
            "in_channels": 4,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "feature_order": [],
            "output_classes": 3,
            "label_map": {},
            "precision": "fp32",
            "quantized": false,
            "onnx_opset": 17
        },
        "normalization": {
            "scaler_type": "zscore"
        }
    }"#).unwrap();
    
    // Создаем конфигурацию ONNX
    let config = OnnxConfig::default();
    
    // Пытаемся загрузить модель - должно провалиться
    let result = OnnxEngine::load(&model_path, 10, 200, &config, "TEST", None);
    
    // Проверяем, что загрузка провалилась
    assert!(result.is_err(), "Expected model loading to fail due to hash mismatch");
    
    // Проверяем сообщение об ошибке
    let err_msg = result.err().unwrap().to_string();
    assert!(err_msg.contains("Model file integrity violation detected"), 
            "Error message should mention integrity violation");
    assert!(err_msg.contains("Expected hash: WRONG_HASH_EXPECTED"), 
            "Error message should show expected hash");
}
