Задача 085: Реализация Exponential Backoff (хелпер для повторов)
Цель: Создать универсальный инструмент для реализации экспоненциальной задержки с джиттером (jitter) для управления повторными попытками (retries) при подключении к WebSocket и запросах к REST API.

1. Изменения в Cargo.toml
Добавить зависимость: rand = "0.8".
2. Изменения в ./src/config/types.rs
ExchangeConfig: Сделать параметры Backoff настраиваемыми.
pub struct ExchangeConfig {
    // ...
    pub ws_retry_initial_ms: u64,    // Начальная задержка (напр. 1000)
    pub ws_retry_max_ms: u64,        // Максимальная задержка (напр. 60000)
    pub ws_retry_multiplier: f64,    // Множитель (напр. 2.0)
    pub ws_retry_jitter: f64,        // Коэффициент джиттера (0.0 - 1.0)
}
3. Создание ./src/utils/backoff.rs
Структура ExponentialBackoff:
pub struct ExponentialBackoff {
    initial_delay: Duration,
    current_delay: Duration,
    max_delay: Duration,
    multiplier: f64,
    jitter: f64,
}

impl ExponentialBackoff {
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
    pub fn next_delay(&mut self) -> Duration {
        use rand::Rng;
        
        // 1. Применяем джиттер к текущей базе (Equal Jitter)
        let mut rng = rand::thread_rng();
        let jitter_factor = self.jitter * (rng.gen::<f64>() * 2.0 - 1.0); // от -jitter до +jitter
        let actual_delay = self.current_delay.mul_f64(1.0 + jitter_factor);

        // 2. Рассчитываем следующую базу для экспоненциального роста
        let next_base = self.current_delay.mul_f64(self.multiplier);
        self.current_delay = next_base.min(self.max_delay);

        actual_delay.min(self.max_delay)
    }

    pub fn reset(&mut self) {
        self.current_delay = self.initial_delay;
    }
}
4. Почему этот план лучше (Аргументы Grok):
Safe Multiplication: Использование mul_f64 для Duration (вместо ручного умножения as_secs_f64) предотвращает ошибки точности и паники при переполнении.
Equal Jitter: Формула 1.0 + jitter * (rand * 2 - 1) обеспечивает симметричное случайное отклонение вокруг базовой задержки. Это лучше размывает нагрузку на сервер при массовых реконнектах.
Initial Delay Persistence: Хранение initial_delay позволяет корректно сбрасывать состояние объекта методом reset() после успешного соединения, не создавая структуру заново.
Configurable: Параметры вынесены в ExchangeConfig, что позволяет настраивать агрессивность реконнекта под конкретную биржу (Bybit может быть чувствителен к частоте попыток).
5. Критические требования
Precision: Использовать std::time::Duration.
Clamping: Итоговая задержка обязательно ограничивается max_delay через метод .min().
Randomness: Использовать rand::thread_rng() для минимизации накладных расходов.
6. Тестирование
Unit test: Проверить последовательность: 1.0s -> ~2.0s -> ~4.0s (с учетом джиттера).
Reset test: Убедиться, что после reset() первая задержка снова близка к initial_delay.
Cap test: Проверить, что задержка никогда не превышает max_delay даже при 100+ попытках.