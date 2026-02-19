use approx::assert_abs_diff_eq;
use neirobot_lit::ml::{TensorBuilder, Normalizer};
use neirobot_lit::data::orderbook::OrderBookSnapshot;
use std::path::Path;

/// Хелпер для создания OrderBookSnapshot из JSON фикстуры
fn create_snapshot_from_json(val: &serde_json::Value) -> OrderBookSnapshot {
    let asks: Vec<(f64, f64)> = serde_json::from_value(val["asks"].clone())
        .expect("Failed to parse asks");
    let bids: Vec<(f64, f64)> = serde_json::from_value(val["bids"].clone())
        .expect("Failed to parse bids");
    
    OrderBookSnapshot {
        timestamp_ms: 1000,
        last_update_id: 1,
        symbol: "BTCUSDT".to_string(),
        asks,
        bids,
        checksum: 0,
        mark_price: 0.0,
    }
}

#[test]
fn test_tensor_sequential_consistency() {
    // 1. Setup - Загружаем реальные параметры как в плане
    let schema_path = Path::new("docs/data_schema.json");
    let norm_path = Path::new("tests/fixtures/norm.json");
    let normalizer = Normalizer::load(schema_path, norm_path)
        .expect("Failed to load normalizer");
    
    let seq_len = 3;
    let mut builder = TensorBuilder::new(normalizer, seq_len);

    // 2. Загрузка тестовых данных из плана
    let fixtures_str = std::fs::read_to_string("tests/fixtures/tensor_test_case.json")
        .expect("Failed to read tensor_test_case.json");
    let fixtures: serde_json::Value = serde_json::from_str(&fixtures_str)
        .expect("Failed to parse tensor_test_case.json");
    
    let mut final_result: Option<Vec<f32>> = None;

    // 3. Прогоняем снимки
    for i in 0..seq_len {
        let snapshot_json = &fixtures["snapshots"][i];
        let snapshot = create_snapshot_from_json(snapshot_json);
        
        let res = builder.process_snapshot(&snapshot)
            .expect("Builder error");
        
        if i == seq_len - 1 {
            final_result = res;
        } else {
            assert!(res.is_none(), "Buffer filled too early at step {}", i);
        }
    }

    // 4. Сравнение с эталоном из Python (точность 1e-6)
    let result = final_result.expect("Buffer should be full");
    let expected: Vec<f32> = serde_json::from_value(fixtures["expected_tensor"].clone())
        .expect("Failed to parse expected_tensor");
    
    assert_eq!(result.len(), expected.len(), "Tensor length mismatch");
    
    for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert_abs_diff_eq!(r, e, epsilon = 1e-6, 
            "Mismatch at index {}: got {}, expected {}", i, r, e);
    }
}
