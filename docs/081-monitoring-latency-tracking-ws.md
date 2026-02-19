Задача 081: Мониторинг задержек (E2E & Processing Latency)
Цель: Реализовать систему высокоточного измерения задержек: от момента генерации события на бирже до завершения обработки внутри бота.

1. Новая структура ./src/utils/monitoring.rs
Создать вспомогательную структуру для агрегации статистики без блокировок.
pub struct LatencyStats {
    pub min_ms: u64,
    pub max_ms: u64,
    pub sum_ms: u64,
    pub count: u64,
}
// Методы update(&mut self, val) и reset()
2. Изменения в ./src/data/websocket.rs (E2E Latency)
Логика при парсинге (depth message):
Извлечь ts (Exchange Timestamp) из JSON Bybit V5.
Зафиксировать local_recv_ms = chrono::Utc::now().timestamp_millis().
Вычислить E2E Latency: e2e = local_recv_ms - exchange_ts.
Clock Drift Check: Если e2e отрицательный или аномально мал (< 5ms), логировать warn!("Local clock drift detected or ultra-low latency: {}ms", e2e).
Обновить глобальную статистику e2e_stats.
3. Изменения в ./src/data/orderbook.rs (Internal Latency)
Замер скорости обработки (apply_update):
Использовать std::time::Instant для микросекундной точности.
let start = Instant::now();
Выполнить apply_update_batch (задача 078).
let elapsed = start.elapsed().as_micros();
Обновить статистику proc_stats.
4. Логирование и отчетность
В основном цикле бота (./src/trading/execution.rs) или в отдельном tokio::task:
Использовать tokio::time::interval(Duration::from_secs(60)).
Каждую минуту выводить агрегированный лог: INFO: [Latency 60s] E2E: avg={:.1}ms, max={}ms | Proc: avg={}us, max={}us.
После вывода вызывать reset() для статистики.
5. Почему это важно (Аргументы Grok):
Instant vs Chrono: Chrono (ms) подходит для сравнения с метками биржи (Unix Epoch), но для замера внутренней скорости apply_update (которая должна быть < 100мкс) необходим Instant::now() с микросекундным разрешением.
Clock Drift: Если часы сервера спешат относительно Bybit, e2e будет некорректным. Важно детектировать это на раннем этапе.
Periodic Stats: Логирование каждые 60 секунд дает более репрезентативную картину, чем лог на каждые 1000 сообщений, так как нагрузка на рынке меняется во времени (периоды затишья vs волатильность).
6. Критические требования
Performance: Измерения должны быть максимально "дешевыми". Использовать атомарные типы (AtomicU64) для sum/count, если статистика обновляется из разных потоков.
Accuracy: Замер local_recv_ms делать сразу после выхода из ws_stream.next(), до парсинга JSON, чтобы включить задержку десериализации в E2E.
7. Тестирование
Unit Test: Проверить расчеты в LatencyStats (min/max/avg).
Integration Test: Убедиться, что при задержке обработки (через thread::sleep) мониторинг корректно отображает рост max_proc_latency.