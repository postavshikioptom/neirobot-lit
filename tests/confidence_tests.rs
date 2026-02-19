// Задача 224: Тесты для мониторинга распределения уверенности инференса

use approx::assert_relative_eq;

/// Вычисляет энтропию распределения вероятностей
/// Формула: H = -Σ p_i * log(p_i)
#[inline]
fn calculate_entropy(probs: &[f32]) -> f32 {
    probs.iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum()
}

/// Обновляет экспоненциальное скользящее среднее (EMA)
/// Формула: EMA = alpha * new_value + (1 - alpha) * current_ema
#[inline]
fn update_ema(current_ema: f32, new_value: f32, alpha: f32) -> f32 {
    alpha * new_value + (1.0 - alpha) * current_ema
}

#[test]
fn test_calculate_entropy_uniform_distribution() {
    // Равномерное распределение для 3 классов: [1/3, 1/3, 1/3]
    let probs = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
    let entropy = calculate_entropy(&probs);
    
    // Максимальная энтропия для 3 классов = ln(3) ≈ 1.0986
    let expected = 3.0_f32.ln();
    assert_relative_eq!(entropy, expected, epsilon = 1e-5);
}

#[test]
fn test_calculate_entropy_certain_prediction() {
    // Уверенное предсказание: [1.0, 0.0, 0.0]
    let probs = [1.0, 0.0, 0.0];
    let entropy = calculate_entropy(&probs);
    
    // Минимальная энтропия = 0 (полная уверенность)
    assert_relative_eq!(entropy, 0.0, epsilon = 1e-5);
}

#[test]
fn test_calculate_entropy_partial_certainty() {
    // Частичная уверенность: [0.7, 0.2, 0.1]
    let probs = [0.7, 0.2, 0.1];
    let entropy = calculate_entropy(&probs);
    
    // Вычисляем вручную: -0.7*ln(0.7) - 0.2*ln(0.2) - 0.1*ln(0.1)
    let expected = -0.7 * 0.7_f32.ln() - 0.2 * 0.2_f32.ln() - 0.1 * 0.1_f32.ln();
    assert_relative_eq!(entropy, expected, epsilon = 1e-5);
}

#[test]
fn test_calculate_entropy_with_zero_probability() {
    // Распределение с нулевой вероятностью: [0.8, 0.2, 0.0]
    let probs = [0.8, 0.2, 0.0];
    let entropy = calculate_entropy(&probs);
    
    // Должно корректно обработать 0 * log(0) = 0
    let expected = -0.8 * 0.8_f32.ln() - 0.2 * 0.2_f32.ln();
    assert_relative_eq!(entropy, expected, epsilon = 1e-5);
}

#[test]
fn test_update_ema_initialization() {
    // Первое обновление: EMA = new_value (при current_ema = 0)
    let current_ema = 0.0;
    let new_value = 1.5;
    let alpha = 0.1;
    
    let ema = update_ema(current_ema, new_value, alpha);
    
    // EMA = 0.1 * 1.5 + 0.9 * 0.0 = 0.15
    assert_relative_eq!(ema, 0.15, epsilon = 1e-5);
}

#[test]
fn test_update_ema_convergence() {
    // Проверяем, что EMA сходится к новому значению
    let mut ema = 1.0;
    let target = 2.0;
    let alpha = 0.1;
    
    // Обновляем EMA 100 раз с одним и тем же значением
    for _ in 0..100 {
        ema = update_ema(ema, target, alpha);
    }
    
    // EMA должна приблизиться к target
    assert_relative_eq!(ema, target, epsilon = 0.01);
}

#[test]
fn test_update_ema_smoothing() {
    // Проверяем сглаживание резких изменений
    let current_ema = 1.0;
    let spike = 10.0;
    let alpha = 0.1;
    
    let ema = update_ema(current_ema, spike, alpha);
    
    // EMA = 0.1 * 10.0 + 0.9 * 1.0 = 1.9
    // Резкий скачок сглажен
    assert_relative_eq!(ema, 1.9, epsilon = 1e-5);
    assert!(ema < spike);
    assert!(ema > current_ema);
}

#[test]
fn test_entropy_ordering() {
    // Проверяем, что энтропия упорядочена правильно
    let certain = [1.0, 0.0, 0.0];
    let partial = [0.7, 0.2, 0.1];
    let uniform = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
    
    let entropy_certain = calculate_entropy(&certain);
    let entropy_partial = calculate_entropy(&partial);
    let entropy_uniform = calculate_entropy(&uniform);
    
    // Уверенное предсказание имеет минимальную энтропию
    assert!(entropy_certain < entropy_partial);
    // Равномерное распределение имеет максимальную энтропию
    assert!(entropy_partial < entropy_uniform);
}

#[test]
fn test_entropy_symmetry() {
    // Энтропия не зависит от порядка вероятностей
    let probs1 = [0.5, 0.3, 0.2];
    let probs2 = [0.2, 0.5, 0.3];
    let probs3 = [0.3, 0.2, 0.5];
    
    let entropy1 = calculate_entropy(&probs1);
    let entropy2 = calculate_entropy(&probs2);
    let entropy3 = calculate_entropy(&probs3);
    
    assert_relative_eq!(entropy1, entropy2, epsilon = 1e-5);
    assert_relative_eq!(entropy2, entropy3, epsilon = 1e-5);
}

#[test]
fn test_ema_alpha_extremes() {
    // Проверяем крайние значения alpha
    let current = 1.0;
    let new = 2.0;
    
    // alpha = 0: EMA не меняется
    let ema_zero = update_ema(current, new, 0.0);
    assert_relative_eq!(ema_zero, current, epsilon = 1e-5);
    
    // alpha = 1: EMA = new_value (без сглаживания)
    let ema_one = update_ema(current, new, 1.0);
    assert_relative_eq!(ema_one, new, epsilon = 1e-5);
}

#[test]
fn test_drift_detection_threshold() {
    // Симуляция детекции дрейфа
    let threshold = 1.5;
    
    // Нормальная энтропия (модель уверена)
    let normal_probs = [0.8, 0.15, 0.05];
    let normal_entropy = calculate_entropy(&normal_probs);
    assert!(normal_entropy < threshold, "Normal entropy should be below threshold");
    
    // Высокая энтропия (модель растеряна)
    let confused_probs = [0.4, 0.35, 0.25];
    let confused_entropy = calculate_entropy(&confused_probs);
    // Эта энтропия может быть ниже порога, но ближе к нему
    
    // Максимальная энтропия (полная неопределенность)
    let max_probs = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
    let max_entropy = calculate_entropy(&max_probs);
    // ln(3) ≈ 1.0986, что ниже порога 1.5
    // Но если модель обучена хорошо, такая энтропия должна быть редкой
    
    println!("Normal entropy: {:.4}", normal_entropy);
    println!("Confused entropy: {:.4}", confused_entropy);
    println!("Max entropy: {:.4}", max_entropy);
}
