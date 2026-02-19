# Задача 133: Прецизионный мониторинг задержек через HdrHistogram (v2.0)

## 1. Контекст и цели
Нам нужно измерять задержки в микросекундах (`μs`) с высокой точностью. Мы будем отслеживать три ключевых показателя:
1.  **Network**: `recv_ts - exchange_ts` (лаг сети/биржи).
2.  **Processing**: Время от `parser` до `execution` внутри нашего кода.
3.  **Total**: Суммарная задержка «событие -> реакция».

## 2. Реализация в `src/monitoring/latency.rs`
Используем крейт `hdrhistogram` для сжатого хранения статистики и `parking_lot::Mutex` для производительной синхронизации между потоками.

```rust
// В [./src/monitoring/latency.rs](./src/monitoring/latency.rs)
use hdrhistogram::Histogram;
use parking_lot::Mutex;
use std::sync::Arc;

pub struct LatencyMonitor {
    network: Mutex<Histogram<u64>>,
    processing: Mutex<Histogram<u64>>,
    total: Mutex<Histogram<u64>>,
}

impl LatencyMonitor {
    pub fn new() -> Self {
        // Настройка: до 1сек (1_000_000 мкс) с точностью 3 значащих цифры
        let factory = || Histogram::<u64>::new_with_max(1_000_000, 3).unwrap();
        Self {
            network: Mutex::new(factory()),
            processing: Mutex::new(factory()),
            total: Mutex::new(factory()),
        }
    }

    pub fn record_network(&self, micros: u64) {
        self.network.lock().record(micros).ok();
    }

    pub fn record_processing(&self, micros: u64) {
        self.processing.lock().record(micros).ok();
    }

    pub fn record_total(&self, micros: u64) {
        self.total.lock().record(micros).ok();
    }

    pub fn print_report(&self) {
        let net = self.network.lock();
        let proc = self.processing.lock();
        let tot = self.total.lock();

        tracing::info!(
            "Latency Report (μs) | P50: {} | P90: {} | P99: {} | P99.9: {} | Count: {}",
            tot.value_at_quantile(0.5),
            tot.value_at_quantile(0.9),
            tot.value_at_quantile(0.99),
            tot.value_at_quantile(0.999),
            tot.len()
        );
        // Аналогично для net и proc
    }
}
```

## 3. Спорные моменты и корректировки (Grok + Zencoder)

*   **HdrHistogram vs Custom**: Согласен с Grok. Самодельные бакеты (buckets) дают большую погрешность на «хвостах». `HdrHistogram` — стандарт индустрии, он позволяет хранить данные в сжатом виде и выдает честные P99.9.
*   **Microseconds (μs)**: Это закон. Миллисекунды слишком грубы для анализа производительности ONNX-инференса и парсинга.
*   **Sync**: Используем `parking_lot::Mutex`. Он быстрее стандартного `std::sync::Mutex` и не подвержен зависаниям (poisoning).
*   **Reporting**: В [./src/config/types.rs](./src/config/types.rs) добавляем `latency_report_interval_sec` (по умолчанию 60). Бот будет выводить отчет в лог каждую минуту.
*   **Network Skew**: При записи `network_latency` (через `unix_ms() - exchange_ts`) добавляем `saturating_sub`, чтобы избежать паники при отрицательном результате из-за рассинхрона часов (clock skew).

## 4. Интеграция в `run-bot.rs`
```rust
// В [./src/bin/run-bot.rs](./src/bin/run-bot.rs)
let start_ts = Instant::now();
let recv_ms = helpers::unix_ms();

// ... обработка ...

let proc_micros = start_ts.elapsed().as_micros() as u64;
let net_micros = (recv_ms.saturating_sub(exchange_ts) * 1000) as u64;

monitor.record_processing(proc_micros);
monitor.record_network(net_micros);
monitor.record_total(proc_micros + net_micros);
```

## 5. Инструкции для Gemini (Coder AI):
1.  **Cargo.toml**: Добавить `hdrhistogram = "7.5"` и `parking_lot = "0.12"`.
2.  **[./src/monitoring/latency.rs](./src/monitoring/latency.rs)**: Реализовать `LatencyMonitor`.
3.  **[./src/bin/run-bot.rs](./src/bin/run-bot.rs)**: Добавить замеры и периодический вызов `print_report()`.
4.  **[./src/config/types.rs](./src/config/types.rs)**: Добавить параметры интервала отчета.

**Результат**: Мы получаем профессиональный инструмент для анализа производительности, который четко покажет, где мы теряем микросекунды — на сети или в коде.
