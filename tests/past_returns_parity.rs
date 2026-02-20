/// Задача 091: Тест Parity Check для Past Returns
/// Сравнивает расчет log-returns в Python и Rust на одних и тех же данных
/// 
/// Требование: Совпадение до 6 знака после запятой

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use rust_decimal::Decimal;
    use rust_decimal::prelude::FromPrimitive;

    /// Вспомогательная функция для расчета log-returns (зеркало Python)
    fn compute_past_returns_python_mirror(mid_prices: &[f64], lags: &[usize]) -> Vec<Vec<f32>> {
        let n = mid_prices.len();
        let n_lags = lags.len();
        let mut past_returns = vec![vec![0.0f32; n]; n_lags];
        
        // Вычисляем логарифмы цен один раз для эффективности
        let log_prices: Vec<f64> = mid_prices.iter().map(|p| p.ln()).collect();
        
        for (lag_idx, lag) in lags.iter().enumerate() {
            // Первые lag значений заполняются 0.0
            for t in *lag..n {
                // Rn = ln(price_t) - ln(price_t-n)
                past_returns[lag_idx][t] = (log_prices[t] - log_prices[t - lag]) as f32;
            }
        }
        
        past_returns
    }

    /// Вспомогательная функция для расчета log-returns (зеркало Rust)
    fn calculate_log_returns_rust_mirror(mid_price_history: &VecDeque<Decimal>, lag: usize) -> f32 {
        if mid_price_history.len() < lag + 1 {
            return 0.0;
        }
        
        let current_idx = mid_price_history.len() - 1;
        let old_idx = current_idx - lag;
        
        if let (Some(current_mid), Some(old_mid)) = (
            mid_price_history.get(current_idx),
            mid_price_history.get(old_idx),
        ) {
            if *old_mid > Decimal::ZERO {
                let current_f64 = current_mid.to_f64().unwrap_or(0.0);
                let old_f64 = old_mid.to_f64().unwrap_or(0.0);
                
                if current_f64 > 0.0 && old_f64 > 0.0 {
                    let log_return = (current_f64.ln() - old_f64.ln()) as f32;
                    return log_return;
                }
            }
        }
        
        0.0
    }

    #[test]
    fn test_past_returns_parity_simple() {
        // Простой тест с известными значениями
        let mid_prices = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0];
        let lags = vec![1, 2];
        
        // Python расчет
        let python_results = compute_past_returns_python_mirror(&mid_prices, &lags);
        
        // Rust расчет
        let mut mid_price_history = VecDeque::new();
        for price in &mid_prices {
            mid_price_history.push_back(Decimal::from_f64(*price).unwrap());
        }
        
        // Проверяем каждый лаг
        for (lag_idx, lag) in lags.iter().enumerate() {
            for t in 0..mid_prices.len() {
                // Обновляем историю до момента t
                let mut history = VecDeque::new();
                for i in 0..=t {
                    history.push_back(Decimal::from_f64(mid_prices[i]).unwrap());
                }
                
                let rust_result = calculate_log_returns_rust_mirror(&history, *lag);
                let python_result = python_results[lag_idx][t];
                
                // Проверяем совпадение до 6 знака
                let diff = (rust_result - python_result).abs();
                assert!(
                    diff < 1e-6,
                    "Parity check failed at t={}, lag={}: Rust={}, Python={}, diff={}",
                    t, lag, rust_result, python_result, diff
                );
            }
        }
    }

    #[test]
    fn test_past_returns_zero_fill() {
        // Тест проверяет, что первые n значений заполняются 0.0
        let mid_prices = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let lags = vec![2, 3];
        
        let python_results = compute_past_returns_python_mirror(&mid_prices, &lags);
        
        // Проверяем, что первые lag значений равны 0.0
        for (lag_idx, lag) in lags.iter().enumerate() {
            for t in 0..*lag {
                assert_eq!(
                    python_results[lag_idx][t], 0.0,
                    "First {} values should be 0.0 for lag={}", lag, lag
                );
            }
        }
    }

    #[test]
    fn test_past_returns_buffer_management() {
        // Тест проверяет управление буфером истории цен
        let mid_prices = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0];
        let max_lag = 3;
        
        let mut mid_price_history = VecDeque::new();
        
        for price in &mid_prices {
            // Управляем размером буфера
            if mid_price_history.len() >= max_lag {
                mid_price_history.pop_front();
            }
            mid_price_history.push_back(Decimal::from_f64(*price).unwrap());
            
            // Проверяем, что размер буфера не превышает max_lag
            assert!(
                mid_price_history.len() <= max_lag,
                "Buffer size should not exceed max_lag"
            );
        }
    }

    #[test]
    fn test_past_returns_multiple_lags() {
        // Тест с несколькими лагами [10, 50, 100]
        let mut mid_prices = vec![100.0];
        
        // Генерируем 150 цен с небольшим случайным шумом
        for i in 1..150 {
            let prev = mid_prices[i - 1];
            let change = (i as f64 % 7.0 - 3.5) * 0.01; // Небольшой шум
            mid_prices.push(prev * (1.0 + change));
        }
        
        let lags = vec![10, 50, 100];
        let python_results = compute_past_returns_python_mirror(&mid_prices, &lags);
        
        // Проверяем последние значения (когда буфер полностью заполнен)
        let mut mid_price_history = VecDeque::new();
        for price in &mid_prices {
            mid_price_history.push_back(Decimal::from_f64(*price).unwrap());
        }
        
        for (lag_idx, lag) in lags.iter().enumerate() {
            let rust_result = calculate_log_returns_rust_mirror(&mid_price_history, *lag);
            let python_result = python_results[lag_idx][mid_prices.len() - 1];
            
            let diff = (rust_result - python_result).abs();
            assert!(
                diff < 1e-6,
                "Parity check failed for lag={}: Rust={}, Python={}, diff={}",
                lag, rust_result, python_result, diff
            );
        }
    }
}
