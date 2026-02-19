/// Тест паритета ONNX моделей с разными функциями активации.
/// Проверяет, что модели с GELU (exact/tanh), SiLU и ReLU корректно загружаются
/// и дают предсказуемые результаты в ONNX Runtime.
use ort::{Session, SessionBuilder, Value};
use ndarray::{Array4, ArrayD};
use std::path::Path;

/// Создает тестовый входной тензор (batch=1, seq=100, channels=6, levels=50)
fn create_test_input() -> Array4<f32> {
    // Создаем детерминированный тензор для воспроизводимости
    let mut arr = Array4::<f32>::zeros((1, 100, 6, 50));
    
    // Заполняем известными значениями для проверки
    for b in 0..1 {
        for s in 0..100 {
            for c in 0..6 {
                for l in 0..50 {
                    // Простая формула для генерации тестовых данных
                    let val = ((s + c * 10 + l) as f32) * 0.01;
                    arr[[b, s, c, l]] = val;
                }
            }
        }
    }
    
    arr
}

/// Загружает ONNX модель и делает предсказание
fn run_inference(model_path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // Инициализация ONNX Runtime
    let session = SessionBuilder::new()?
        .with_intra_threads(1)?
        .commit_from_file(model_path)?;
    
    // Подготовка входа
    let input_tensor = create_test_input();
    let input_value = Value::from_array(input_tensor)?;
    
    // Запуск инференса
    let outputs = session.run(ort::inputs!["input" => input_value]?)?;
    
    // Извлечение результата
    let output: ArrayD<f32> = outputs["output"].try_extract_tensor()?.into_owned();
    let output_vec: Vec<f32> = output.iter().copied().collect();
    
    Ok(output_vec)
}

#[test]
#[ignore] // Игнорируем по умолчанию, так как требуются ONNX модели
fn test_onnx_activation_consistency() {
    // Пути к ONNX моделям с разными активациями
    // Эти модели должны быть предварительно экспортированы из Python
    let models = vec![
        ("gelu_exact", "tests/fixtures/models/lit_gelu_exact.onnx"),
        ("gelu_tanh", "tests/fixtures/models/lit_gelu_tanh.onnx"),
        ("silu", "tests/fixtures/models/lit_silu.onnx"),
        ("relu", "tests/fixtures/models/lit_relu.onnx"),
    ];
    
    let mut results = Vec::new();
    
    for (activation_name, model_path) in &models {
        let path = Path::new(model_path);
        
        if !path.exists() {
            eprintln!("⚠️  Model not found: {} (skipping)", model_path);
            continue;
        }
        
        println!("Testing activation: {}", activation_name);
        
        match run_inference(path) {
            Ok(output) => {
                println!("  ✓ Inference successful");
                println!("  Output shape: {}", output.len());
                println!("  Output sample: [{:.4}, {:.4}, {:.4}]", 
                         output[0], output[1], output[2]);
                
                results.push((*activation_name, output));
            }
            Err(e) => {
                panic!("❌ Inference failed for {}: {}", activation_name, e);
            }
        }
    }
    
    // Проверяем, что все модели дали результаты одинаковой формы
    if results.len() > 1 {
        let expected_len = results[0].1.len();
        for (name, output) in &results {
            assert_eq!(
                output.len(), 
                expected_len,
                "Output length mismatch for {}: expected {}, got {}",
                name, expected_len, output.len()
            );
        }
        println!("\n✓ All activations produce consistent output shapes");
    }
    
    // Проверяем, что результаты различаются (разные активации должны давать разные выходы)
    if results.len() > 1 {
        let first_output = &results[0].1;
        let mut all_same = true;
        
        for (name, output) in results.iter().skip(1) {
            let max_diff = first_output.iter()
                .zip(output.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            
            println!("Max difference between {} and {}: {:.6}", 
                     results[0].0, name, max_diff);
            
            if max_diff > 1e-4 {
                all_same = false;
            }
        }
        
        assert!(!all_same, "All activations produce identical outputs (suspicious)");
        println!("✓ Different activations produce different outputs (as expected)");
    }
}

#[test]
#[ignore] // Игнорируем по умолчанию
fn test_onnx_model_metadata() {
    // Проверяем, что metadata.json содержит информацию об активации
    let metadata_paths = vec![
        "tests/fixtures/models/metadata_gelu_exact.json",
        "tests/fixtures/models/metadata_silu.json",
    ];
    
    for path_str in metadata_paths {
        let path = Path::new(path_str);
        
        if !path.exists() {
            eprintln!("⚠️  Metadata not found: {} (skipping)", path_str);
            continue;
        }
        
        let content = std::fs::read_to_string(path)
            .expect("Failed to read metadata");
        
        let metadata: serde_json::Value = serde_json::from_str(&content)
            .expect("Failed to parse metadata JSON");
        
        // Проверяем наличие поля activation
        assert!(
            metadata.get("activation").is_some(),
            "Metadata missing 'activation' field in {}",
            path_str
        );
        
        let activation = metadata["activation"].as_str()
            .expect("activation field must be a string");
        
        println!("✓ Metadata {}: activation = {}", path_str, activation);
        
        // Проверяем, что это один из поддерживаемых типов
        assert!(
            ["relu", "gelu_exact", "gelu_tanh", "silu"].contains(&activation),
            "Unknown activation type: {}",
            activation
        );
    }
}

#[cfg(test)]
mod activation_parity_instructions {
    /// Инструкции по подготовке тестовых моделей:
    /// 
    /// 1. Обучите модели с разными активациями:
    ///    ```bash
    ///    python -m python_lab.scripts.train --symbol BTCUSDT --activation gelu_exact --epochs 1
    ///    python -m python_lab.scripts.train --symbol BTCUSDT --activation silu --epochs 1
    ///    ```
    /// 
    /// 2. Экспортируйте модели в ONNX:
    ///    ```bash
    ///    python -m python_lab.scripts.export_onnx \
    ///      --input bots/BTCUSDT/models/checkpoints/best.ckpt \
    ///      --output tests/fixtures/models/lit_gelu_exact.onnx
    ///    ```
    /// 
    /// 3. Скопируйте metadata.json:
    ///    ```bash
    ///    cp bots/BTCUSDT/models/metadata.json \
    ///       tests/fixtures/models/metadata_gelu_exact.json
    ///    ```
    /// 
    /// 4. Запустите тесты:
    ///    ```bash
    ///    cargo test --test onnx_activation_parity -- --ignored --nocapture
    ///    ```
}
