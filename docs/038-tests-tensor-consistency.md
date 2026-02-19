# 038 - Tests Tensor Consistency
Цель задачи: Гарантировать полную идентичность подготовки тензора в Rust и Python. Тест должен проверять не только финальный вектор признаков, но и работу скользящего окна (seq_len), логику Feature Engineering и нормализацию на нескольких последовательных снимках.

Файлы:

Cargo.toml (обновить)
tests/tensor_consistency.rs (создать)
Инструкции для Gemini:

Добавить зависимость в Cargo.toml:
[dev-dependencies]
approx = "0.5"
serde_json = "1.0"
tests/tensor_consistency.rs: Реализовать интеграционный тест, сравнивающий вывод Rust с эталоном из Python.
use approx::assert_abs_diff_eq;
use crate::ml::{TensorBuilder, Normalizer};
use crate::data::order_book::OrderBook;
use std::path::Path;

/// Хелпер для создания OrderBook из упрощенного JSON (цены и объемы)
fn create_mock_ob(symbol: &str, asks: Vec<(f64, f64)>, bids: Vec<(f64, f64)>) -> OrderBook {
    let mut ob = OrderBook::new(symbol);
    // Имитируем поступление данных через apply_update или прямую вставку
    for (p, v) in asks { ob.asks.insert(p.into(), v); }
    for (p, v) in bids { ob.bids.insert(p.into(), v); }
    ob
}

#[test]
fn test_tensor_sequential_consistency() {
    // 1. Setup
    let schema_path = Path::new("docs/data_schema.json");
    let norm_path = Path::new("tests/fixtures/norm.json");
    let normalizer = Normalizer::load(schema_path, norm_path).expect("Failed to load normalizer");
    
    let seq_len = 3; // Тестируем последовательность из 3 шагов
    let mut builder = TensorBuilder::new(normalizer, seq_len);

    // 2. Загрузка тестовых данных (3 последовательных снимка + 1 ожидаемый тензор)
    let fixtures_str = std::fs::read_to_string("tests/fixtures/tensor_test_case.json").unwrap();
    let fixtures: serde_json::Value = serde_json::from_str(&fixtures_str).unwrap();

    let mut final_result: Option<Vec<f32>> = None;

    // 3. Прогоняем снимки через builder
    for i in 0..seq_len {
        let snapshot = &fixtures["snapshots"][i];
        let ob = create_mock_ob(
            "BTCUSDT",
            serde_json::from_value(snapshot["asks"].clone()).unwrap(),
            serde_json::from_value(snapshot["bids"].clone()).unwrap(),
        );
        
        // Должен возвращать None, пока не заполнено окно seq_len
        let res = builder.process_snapshot(&ob).expect("Builder error");
        if i == seq_len - 1 {
            final_result = res;
        } else {
            assert!(res.is_none(), "Buffer filled too early at step {}", i);
        }
    }

    // 4. Сравнение с эталоном из Python
    let result = final_result.expect("Buffer should be full");
    let expected: Vec<f32> = serde_json::from_value(fixtures["expected_tensor"].clone()).unwrap();

    assert_eq!(result.len(), expected.len(), "Tensor length mismatch");
    
    // Используем approx для сравнения f32 с точностью 1e-6
    for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert_abs_diff_eq!(r, e, epsilon = 1e-6);
    }
}
Технические требования:

Точность: Использовать approx с epsilon = 1e-6.
Последовательность: Тестировать seq_len > 1, чтобы убедиться в правильности работы VecDeque и flatten (порядок: старые снимки в начале, новые в конце).
Данные: Тестовый файл fixtures/tensor_test_case.json должен содержать крайние случаи: нулевые объемы (log(1+0)) и ситуации, когда mid_price может быть некорректным.
Подход: Не использовать serde_json напрямую на OrderBook, если структура не готова — использовать хелпер для сборки.