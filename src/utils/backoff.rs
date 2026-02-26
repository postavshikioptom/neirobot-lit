use std::time::Duration;

/// Структура для управления экспоненциальной задержкой с джиттером
pub struct ExponentialBackoff {
    initial_delay: Duration,
    current_delay: Duration,
    max_delay: Duration,
    multiplier: f64,
    jitter: f64,
}

impl ExponentialBackoff {
    /// Создает новый экземпляр ExponentialBackoff
    ///
    /// # Аргументы
    /// * `initial` - начальная задержка
    /// * `max` - максимальная задержка
    /// * `multiplier` - множитель для экспоненциального роста (обычно 2.0)
    /// * `jitter` - коэффициент джиттера (0.0 - 1.0)
    pub fn new(initial: Duration, max: Duration, multiplier: f64, jitter: f64) -> Self {
        Self {
            initial_delay: initial,
            current_delay: initial,
            max_delay: max,
            multiplier,
            jitter,
        }
    }

    /// Возвращает задержку для следующей попытки и обновляет состояние
    ///
    /// Использует Equal Jitter стратегию для симметричного случайного отклонения
    /// вокруг базовой задержки.
    pub fn next_delay(&mut self) -> Duration {
        let mut rng = rand::rng();
        
        // 1. Применяем джиттер к текущей базе (Equal Jitter)
        // Формула: 1.0 + jitter * (rand * 2 - 1) дает значение от (1 - jitter) до (1 + jitter)
        let jitter_factor = self.jitter * (rng.r#gen::<f64>() * 2.0 - 1.0);
        let actual_delay = self.current_delay.mul_f64(1.0 + jitter_factor);

        // 2. Рассчитываем следующую базу для экспоненциального роста
        let next_base = self.current_delay.mul_f64(self.multiplier);
        self.current_delay = next_base.min(self.max_delay);

        // 3. Ограничиваем итоговую задержку максимальным значением
        actual_delay.min(self.max_delay)
    }

    /// Сбрасывает состояние backoff к начальной задержке
    pub fn reset(&mut self) {
        self.current_delay = self.initial_delay;
    }

    /// Возвращает текущую базовую задержку (без джиттера)
    pub fn current_base_delay(&self) -> Duration {
        self.current_delay
    }

    /// Возвращает максимальную задержку
    pub fn max_delay(&self) -> Duration {
        self.max_delay
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_growth() {
        let mut backoff = ExponentialBackoff::new(
            Duration::from_millis(1000),
            Duration::from_millis(60000),
            2.0,
            0.0, // без джиттера для предсказуемости
        );

        // Первая задержка должна быть ~1000ms
        let delay1 = backoff.next_delay();
        assert_eq!(delay1, Duration::from_millis(1000));

        // Вторая задержка должна быть ~2000ms
        let delay2 = backoff.next_delay();
        assert_eq!(delay2, Duration::from_millis(2000));

        // Третья задержка должна быть ~4000ms
        let delay3 = backoff.next_delay();
        assert_eq!(delay3, Duration::from_millis(4000));
    }

    #[test]
    fn test_max_delay_capping() {
        let mut backoff = ExponentialBackoff::new(
            Duration::from_millis(1000),
            Duration::from_millis(10000),
            2.0,
            0.0,
        );

        // Пропускаем несколько итераций
        for _ in 0..10 {
            backoff.next_delay();
        }

        // Задержка не должна превышать max_delay
        let delay = backoff.next_delay();
        assert!(delay <= Duration::from_millis(10000));
    }

    #[test]
    fn test_reset() {
        let mut backoff = ExponentialBackoff::new(
            Duration::from_millis(1000),
            Duration::from_millis(60000),
            2.0,
            0.0,
        );

        // Несколько итераций
        backoff.next_delay();
        backoff.next_delay();
        backoff.next_delay();

        // После reset первая задержка должна быть снова ~1000ms
        backoff.reset();
        let delay = backoff.next_delay();
        assert_eq!(delay, Duration::from_millis(1000));
    }

    #[test]
    fn test_jitter_range() {
        let mut backoff = ExponentialBackoff::new(
            Duration::from_millis(1000),
            Duration::from_millis(60000),
            2.0,
            0.1, // 10% джиттер
        );

        // Проверяем, что задержка находится в ожидаемом диапазоне
        // С джиттером 0.1, задержка должна быть в диапазоне [900ms, 1100ms]
        for _ in 0..100 {
            let delay = backoff.next_delay();
            // Первая итерация: базовая задержка 1000ms
            assert!(delay >= Duration::from_millis(900));
            assert!(delay <= Duration::from_millis(1100));
            backoff.reset();
        }
    }
}
