use std::time::{SystemTime, UNIX_EPOCH};
use anyhow::Context;

#[inline(always)]
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_millis() as u64
}

/// Алиас для now_ms() для совместимости с документацией
#[inline(always)]
pub fn unix_ms() -> u64 {
    now_ms()
}

#[inline(always)]
pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_secs()
}

use std::collections::VecDeque;
use rust_decimal::Decimal;
use crate::data::types::{PublicTrade, Side};

/// Структура для расчета скользящих статистик цен (VWAP и Time-Weighted TWAP)
pub struct RollingPriceStats {
    window_ms: i64,
    max_trades: usize,
    trades: VecDeque<PublicTrade>,
    
    // Для VWAP (Volume-Weighted Average Price)
    sum_pv: Decimal,      // Sum(Price * Amount)
    sum_vol: Decimal,     // Sum(Amount)
    
    // Для Time-Weighted TWAP (Time-Weighted Average Price)
    sum_pw: Decimal,      // Sum(Price * TimeDelta)
    last_ts: i64,
    total_time_ms: i64,
}

impl RollingPriceStats {
    /// Создает новый экземпляр RollingPriceStats
    pub fn new(window_ms: i64, max_trades: usize) -> Self {
        Self {
            window_ms,
            max_trades,
            trades: VecDeque::new(),
            sum_pv: Decimal::ZERO,
            sum_vol: Decimal::ZERO,
            sum_pw: Decimal::ZERO,
            last_ts: 0,
            total_time_ms: 0,
        }
    }

    /// Обновляет статистику новой сделкой
    pub fn update(&mut self, trade: PublicTrade) {
        // 1. Расчет TWAP интеграла (Time-Weighted)
        if self.last_ts > 0 {
            let delta = (trade.timestamp - self.last_ts).max(0);
            if let Some(last_trade) = self.trades.back() {
                self.sum_pw += last_trade.price * Decimal::from(delta);
                self.total_time_ms += delta;
            }
        }
        self.last_ts = trade.timestamp;

        // 2. Добавление новой сделки (VWAP)
        self.sum_pv += trade.price * trade.size;
        self.sum_vol += trade.size;
        self.trades.push_back(trade);

        // 3. Очистка старых данных (Sliding Window)
        let cutoff = self.last_ts - self.window_ms;
        while self.trades.len() > 1 && (self.trades[0].timestamp < cutoff || self.trades.len() > self.max_trades) {
            let old = self.trades.pop_front().unwrap();
            
            // Вычитаем из VWAP
            self.sum_pv -= old.price * old.size;
            self.sum_vol -= old.size;
            
            // Для TWAP: вычитаем дельту первой сделки
            // Дельта = (следующая_сделка.timestamp - старая_сделка.timestamp) * старая_сделка.price
            if let Some(next_trade) = self.trades.front() {
                let delta = (next_trade.timestamp - old.timestamp).max(0);
                let pw_delta = old.price * Decimal::from(delta);
                self.sum_pw -= pw_delta;
                self.total_time_ms -= delta;
            }
        }
    }

    /// Возвращает VWAP (Volume-Weighted Average Price)
    /// Опционально фильтрует по стороне сделки (Buy/Sell)
    pub fn get_vwap(&self, side_filter: Option<Side>) -> Decimal {
        if let Some(side) = side_filter {
            let (s_pv, s_vol) = self.trades.iter()
                .filter(|t| t.side == side)
                .fold((Decimal::ZERO, Decimal::ZERO), |acc, t| {
                    (acc.0 + t.price * t.size, acc.1 + t.size)
                });
            if s_vol.is_zero() { 
                Decimal::ZERO 
            } else { 
                s_pv / s_vol 
            }
        } else {
            if self.sum_vol.is_zero() { 
                Decimal::ZERO 
            } else { 
                self.sum_pv / self.sum_vol 
            }
        }
    }

    /// Возвращает Time-Weighted TWAP (Time-Weighted Average Price)
    pub fn get_twap(&self) -> Decimal {
        if self.total_time_ms == 0 { 
            // Если нет временных данных, возвращаем цену последней сделки
            self.trades.back().map(|t| t.price).unwrap_or(Decimal::ZERO)
        } else {
            self.sum_pw / Decimal::from(self.total_time_ms)
        }
    }

    /// Возвращает количество сделок в буфере
    pub fn len(&self) -> usize {
        self.trades.len()
    }

    /// Проверяет, пуст ли буфер
    pub fn is_empty(&self) -> bool {
        self.trades.is_empty()
    }
}

// ============================================================================
// Математические хелперы для работы с лотами биржи (Задача 137)
// ============================================================================

/// Округление вниз до шага лота
/// 
/// Используется для приведения объема ордера к допустимому значению согласно qty_step биржи.
/// Например, если qty_step = 0.01, то 1.234 -> 1.23
pub fn round_down_to_step(qty: f64, step: f64) -> f64 {
    if step <= 0.0 { 
        return qty; 
    }
    (qty / step).floor() * step
}

/// Проверка, является ли объем "пылью" (меньше минимального лота)
/// 
/// Учитывает микро-погрешность float (epsilon) для корректного сравнения.
/// Если qty < min_qty (с учетом погрешности), возвращает true.
pub fn is_dust(qty: f64, min_qty: f64) -> bool {
    qty < (min_qty - 1e-10)
}

/// Ограничение объема в диапазоне [min_qty, max_qty] с округлением до шага
/// 
/// Применяет clamp к диапазону, затем округляет вниз до qty_step.
/// Используется для валидации объема перед отправкой ордера на биржу.
pub fn clamp_qty(qty: f64, min_qty: f64, max_qty: f64, step: f64) -> f64 {
    let clamped = qty.max(min_qty).min(max_qty);
    round_down_to_step(clamped, step)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_rolling_price_stats_vwap() {
        let mut stats = RollingPriceStats::new(60000, 100);
        
        // Добавляем несколько сделок
        stats.update(PublicTrade {
            price: dec!(100.0),
            size: dec!(10.0),
            side: Side::Buy,
            timestamp: 1000,
        });
        
        stats.update(PublicTrade {
            price: dec!(110.0),
            size: dec!(20.0),
            side: Side::Sell,
            timestamp: 2000,
        });
        
        // VWAP = (100*10 + 110*20) / (10 + 20) = 3200 / 30 = 106.666...
        let vwap = stats.get_vwap(None);
        assert!((vwap - dec!(106.666666666666666666666666667)).abs() < dec!(0.001));
    }

    #[test]
    fn test_rolling_price_stats_twap() {
        let mut stats = RollingPriceStats::new(60000, 100);
        
        stats.update(PublicTrade {
            price: dec!(100.0),
            size: dec!(10.0),
            side: Side::Buy,
            timestamp: 1000,
        });
        
        stats.update(PublicTrade {
            price: dec!(110.0),
            size: dec!(20.0),
            side: Side::Sell,
            timestamp: 3000,
        });
        
        // TWAP = (100 * 2000) / 2000 = 100
        let twap = stats.get_twap();
        assert_eq!(twap, dec!(100.0));
    }

    #[test]
    fn test_rolling_price_stats_sliding_window() {
        let mut stats = RollingPriceStats::new(5000, 100); // 5 секунд окно
        
        stats.update(PublicTrade {
            price: dec!(100.0),
            size: dec!(10.0),
            side: Side::Buy,
            timestamp: 1000,
        });
        
        stats.update(PublicTrade {
            price: dec!(110.0),
            size: dec!(20.0),
            side: Side::Sell,
            timestamp: 7000, // Старая сделка должна быть удалена
        });
        
        // Только вторая сделка должна остаться
        assert_eq!(stats.len(), 1);
        let vwap = stats.get_vwap(None);
        assert_eq!(vwap, dec!(110.0));
    }

    // ============================================================================
    // Тесты для математических хелперов (Задача 137)
    // ============================================================================

    #[test]
    fn test_round_down_to_step_basic() {
        // Шаг 0.01
        assert_eq!(round_down_to_step(1.234, 0.01), 1.23);
        assert_eq!(round_down_to_step(1.239, 0.01), 1.23);
        
        // Шаг 1.0
        assert_eq!(round_down_to_step(10.7, 1.0), 10.0);
        assert_eq!(round_down_to_step(10.1, 1.0), 10.0);
        
        // Шаг 10.0
        assert_eq!(round_down_to_step(123.0, 10.0), 120.0);
        assert_eq!(round_down_to_step(129.9, 10.0), 120.0);
    }

    #[test]
    fn test_round_down_to_step_edge_cases() {
        // Нулевой или отрицательный шаг - возвращаем исходное значение
        assert_eq!(round_down_to_step(1.234, 0.0), 1.234);
        assert_eq!(round_down_to_step(1.234, -0.01), 1.234);
        
        // Точное совпадение с шагом
        assert_eq!(round_down_to_step(1.23, 0.01), 1.23);
        assert_eq!(round_down_to_step(10.0, 1.0), 10.0);
    }

    #[test]
    fn test_is_dust() {
        // Явная пыль
        assert!(is_dust(0.009, 0.01));
        assert!(is_dust(0.0, 0.01));
        
        // Не пыль
        assert!(!is_dust(0.01, 0.01));
        assert!(!is_dust(0.011, 0.01));
        assert!(!is_dust(1.0, 0.01));
        
        // Граничные случаи с учетом epsilon
        assert!(!is_dust(0.01 - 1e-11, 0.01)); // Меньше epsilon - не пыль
        assert!(is_dust(0.01 - 1e-9, 0.01));   // Больше epsilon - пыль
    }

    #[test]
    fn test_clamp_qty() {
        // Нормальный случай: значение в диапазоне
        let result = clamp_qty(1.234, 0.01, 100.0, 0.01);
        assert_eq!(result, 1.23);
        
        // Значение меньше min_qty
        let result = clamp_qty(0.005, 0.01, 100.0, 0.01);
        assert_eq!(result, 0.01);
        
        // Значение больше max_qty
        let result = clamp_qty(150.0, 0.01, 100.0, 0.01);
        assert_eq!(result, 100.0);
        
        // Шаг 1.0
        let result = clamp_qty(10.7, 1.0, 100.0, 1.0);
        assert_eq!(result, 10.0);
        
        // Шаг 10.0
        let result = clamp_qty(123.0, 10.0, 1000.0, 10.0);
        assert_eq!(result, 120.0);
    }

    #[test]
    fn test_clamp_qty_with_different_steps() {
        // Шаг 0.001 (для высоколиквидных пар)
        let result = clamp_qty(1.2345, 0.001, 100.0, 0.001);
        assert_eq!(result, 1.234);
        
        // Шаг 0.1 (для средних пар)
        let result = clamp_qty(12.34, 0.1, 100.0, 0.1);
        assert_eq!(result, 12.3);
        
        // Шаг 100.0 (для щиткоинов)
        let result = clamp_qty(1234.0, 100.0, 10000.0, 100.0);
        assert_eq!(result, 1200.0);
    }
}

// ============================================================================
// Проверка синхронизации времени с биржей (Задача 169)
// ============================================================================

/// Проверяет расхождение локального времени с временем биржи Bybit
/// 
/// # Параметры
/// - `base_url`: URL REST API биржи (например, "https://api.bybit.com")
/// - `max_skew_ms`: Максимально допустимое расхождение в миллисекундах
/// 
/// # Возвращает
/// - `Ok(delta_ms)`: Расхождение времени (local_ms - server_ms)
/// - `Err`: Ошибка при запросе к API
/// 
/// # Пример
/// ```no_run
/// use neirobot_lit::utils::helpers::check_clock_skew;
/// 
/// let delta = check_clock_skew("https://api.bybit.com", 5000).await?;
/// if delta.abs() > 5000 {
///     eprintln!("CRITICAL: Clock skew detected: {}ms", delta);
/// }
/// ```
pub async fn check_clock_skew(base_url: &str, max_skew_ms: i64) -> anyhow::Result<i64> {
    use reqwest::Client;
    use serde_json::Value;
    
    let client = Client::new();
    let url = format!("{}/v5/market/time", base_url);
    
    // Запрашиваем время биржи
    let response = client
        .get(&url)
        .send()
        .await
        .context("Failed to request server time from Bybit")?;
    
    let local_ms = unix_ms() as i64;
    
    let json: Value = response
        .json()
        .await
        .context("Failed to parse server time response")?;
    
    // Извлекаем время сервера из ответа
    let server_ms = json["time"]
        .as_i64()
        .context("Missing 'time' field in server response")?;
    
    let delta = local_ms - server_ms;
    
    if delta.abs() > max_skew_ms {
        tracing::error!(
            "CRITICAL CLOCK SKEW: Local time differs from Bybit by {}ms (limit: {}ms)",
            delta,
            max_skew_ms
        );
    } else {
        tracing::debug!("Clock skew check passed: delta = {}ms", delta);
    }
    
    Ok(delta)
}

#[cfg(test)]
mod clock_skew_tests {
    use super::*;
    
    #[tokio::test]
    #[ignore] // Требует сетевого подключения
    async fn test_check_clock_skew_real() {
        let result = check_clock_skew("https://api.bybit.com", 5000).await;
        assert!(result.is_ok());
        
        let delta = result.unwrap();
        // Обычно расхождение должно быть меньше 1 секунды
        assert!(delta.abs() < 1000, "Clock skew too large: {}ms", delta);
    }
}

/// Применяет экспоненциальный backoff с джиттером для обработки ошибок rate limit
/// 
/// # Аргументы
/// * `attempt` - номер попытки (0-индексированный)
/// * `base_ms` - базовая задержка в миллисекундах
/// 
/// # Поведение
/// - Вычисляет wait_ms = (base_ms * 2^attempt) + jitter
/// - Добавляет случайный джиттер (0..100 мс) для предотвращения "thundering herd"
/// - Ограничивает максимальное время ожидания 60 секундами
pub async fn apply_backoff(attempt: u32, base_ms: u64) {
    use rand::Rng;
    
    let mut rng = rand::thread_rng();
    let jitter = rng.gen_range(0..100);
    
    // Экспоненциальный расчет: base_ms * 2^attempt
    let exponential_wait = base_ms.saturating_mul(2u64.saturating_pow(attempt));
    let wait_ms = exponential_wait.saturating_add(jitter);
    
    // Ограничиваем максимальное время ожидания 60 секундами
    let final_wait = std::cmp::min(wait_ms, 60_000);
    
    tracing::warn!(
        attempt = attempt,
        base_ms = base_ms,
        jitter = jitter,
        final_wait_ms = final_wait,
        "Applying exponential backoff with jitter"
    );
    
    tokio::time::sleep(std::time::Duration::from_millis(final_wait)).await;
}
