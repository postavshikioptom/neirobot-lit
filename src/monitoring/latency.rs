use hdrhistogram::Histogram;
use parking_lot::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use lazy_static::lazy_static;

/// Прецизионный мониторинг задержек через HdrHistogram
/// Отслеживает три ключевых показателя в микросекундах (μs):
/// 1. Network: recv_ts - exchange_ts (лаг сети/биржи)
/// 2. Processing: Время от parser до execution внутри нашего кода
/// 3. Total: Суммарная задержка «событие -> реакция»
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

    /// Записывает задержку сети в микросекундах
    pub fn record_network(&self, micros: u64) {
        self.network.lock().record(micros).ok();
    }

    /// Записывает задержку обработки в микросекундах
    pub fn record_processing(&self, micros: u64) {
        self.processing.lock().record(micros).ok();
    }

    /// Записывает общую задержку в микросекундах
    pub fn record_total(&self, micros: u64) {
        self.total.lock().record(micros).ok();
    }

    /// Выводит отчет о задержках с перцентилями
    pub fn print_report(&self) {
        let net = self.network.lock();
        let proc = self.processing.lock();
        let tot = self.total.lock();

        tracing::info!(
            "[Latency Report - Total] P50: {}μs | P90: {}μs | P99: {}μs | P99.9: {}μs | Count: {}",
            tot.value_at_quantile(0.5),
            tot.value_at_quantile(0.9),
            tot.value_at_quantile(0.99),
            tot.value_at_quantile(0.999),
            tot.len()
        );

        tracing::info!(
            "[Latency Report - Network] P50: {}μs | P90: {}μs | P99: {}μs | P99.9: {}μs | Count: {}",
            net.value_at_quantile(0.5),
            net.value_at_quantile(0.9),
            net.value_at_quantile(0.99),
            net.value_at_quantile(0.999),
            net.len()
        );

        tracing::info!(
            "[Latency Report - Processing] P50: {}μs | P90: {}μs | P99: {}μs | P99.9: {}μs | Count: {}",
            proc.value_at_quantile(0.5),
            proc.value_at_quantile(0.9),
            proc.value_at_quantile(0.99),
            proc.value_at_quantile(0.999),
            proc.len()
        );
    }

    /// Сбрасывает все гистограммы
    pub fn reset(&self) {
        self.network.lock().clear();
        self.processing.lock().clear();
        self.total.lock().clear();
    }
}

impl Default for LatencyMonitor {
    fn default() -> Self {
        Self::new()
    }
}

// --- Старые структуры для обратной совместимости ---

pub struct LatencyStats {
    pub min: AtomicU64,
    pub max: AtomicU64,
    pub sum: AtomicU64,
    pub count: AtomicU64,
}

impl LatencyStats {
    pub fn new() -> Self {
        Self {
            min: AtomicU64::new(u64::MAX),
            max: AtomicU64::new(0),
            sum: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    pub fn update(&self, val: u64) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum.fetch_add(val, Ordering::Relaxed);
        
        let mut current_min = self.min.load(Ordering::Relaxed);
        while val < current_min {
            match self.min.compare_exchange_weak(current_min, val, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(actual) => current_min = actual,
            }
        }

        let mut current_max = self.max.load(Ordering::Relaxed);
        while val > current_max {
            match self.max.compare_exchange_weak(current_max, val, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(actual) => current_max = actual,
            }
        }
    }

    pub fn reset(&self) {
        self.min.store(u64::MAX, Ordering::Relaxed);
        self.max.store(0, Ordering::Relaxed);
        self.sum.store(0, Ordering::Relaxed);
        self.count.store(0, Ordering::Relaxed);
    }

    pub fn get_avg(&self) -> f64 {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 { return 0.0; }
        self.sum.load(Ordering::Relaxed) as f64 / count as f64
    }

    pub fn get_max(&self) -> u64 {
        self.max.load(Ordering::Relaxed)
    }

    pub fn get_min(&self) -> u64 {
        let min = self.min.load(Ordering::Relaxed);
        if min == u64::MAX { 0 } else { min }
    }
}

pub struct HotPathStats {
    pub json_parsing_sum_us: AtomicU64,
    pub json_parsing_count: AtomicU64,
    pub lob_update_sum_us: AtomicU64,
    pub lob_update_count: AtomicU64,
    pub feature_calc_sum_us: AtomicU64,
    pub feature_calc_count: AtomicU64,
    pub inference_sum_us: AtomicU64,
    pub inference_count: AtomicU64,
    pub max_inference_us: AtomicU64,
}

impl HotPathStats {
    pub fn new() -> Self {
        Self {
            json_parsing_sum_us: AtomicU64::new(0),
            json_parsing_count: AtomicU64::new(0),
            lob_update_sum_us: AtomicU64::new(0),
            lob_update_count: AtomicU64::new(0),
            feature_calc_sum_us: AtomicU64::new(0),
            feature_calc_count: AtomicU64::new(0),
            inference_sum_us: AtomicU64::new(0),
            inference_count: AtomicU64::new(0),
            max_inference_us: AtomicU64::new(0),
        }
    }

    pub fn record_json_parsing(&self, us: u64) {
        self.json_parsing_sum_us.fetch_add(us, Ordering::Relaxed);
        self.json_parsing_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_lob(&self, us: u64) {
        self.lob_update_sum_us.fetch_add(us, Ordering::Relaxed);
        self.lob_update_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_feature(&self, us: u64) {
        self.feature_calc_sum_us.fetch_add(us, Ordering::Relaxed);
        self.feature_calc_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_inference(&self, us: u64) {
        self.inference_sum_us.fetch_add(us, Ordering::Relaxed);
        self.inference_count.fetch_add(1, Ordering::Relaxed);

        let mut current_max = self.max_inference_us.load(Ordering::Relaxed);
        while us > current_max {
            match self.max_inference_us.compare_exchange_weak(
                current_max,
                us,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_max = actual,
            }
        }
    }

    pub fn reset(&self) {
        self.json_parsing_sum_us.store(0, Ordering::Relaxed);
        self.json_parsing_count.store(0, Ordering::Relaxed);
        self.lob_update_sum_us.store(0, Ordering::Relaxed);
        self.lob_update_count.store(0, Ordering::Relaxed);
        self.feature_calc_sum_us.store(0, Ordering::Relaxed);
        self.feature_calc_count.store(0, Ordering::Relaxed);
        self.inference_sum_us.store(0, Ordering::Relaxed);
        self.inference_count.store(0, Ordering::Relaxed);
        self.max_inference_us.store(0, Ordering::Relaxed);
    }

    pub fn get_avg_json_parsing(&self) -> u64 {
        let count = self.json_parsing_count.load(Ordering::Relaxed);
        if count == 0 { return 0; }
        self.json_parsing_sum_us.load(Ordering::Relaxed) / count
    }

    pub fn get_avg_lob(&self) -> u64 {
        let count = self.lob_update_count.load(Ordering::Relaxed);
        if count == 0 { return 0; }
        self.lob_update_sum_us.load(Ordering::Relaxed) / count
    }

    pub fn get_avg_feature(&self) -> u64 {
        let count = self.feature_calc_count.load(Ordering::Relaxed);
        if count == 0 { return 0; }
        self.feature_calc_sum_us.load(Ordering::Relaxed) / count
    }

    pub fn get_avg_inference(&self) -> u64 {
        let count = self.inference_count.load(Ordering::Relaxed);
        if count == 0 { return 0; }
        self.inference_sum_us.load(Ordering::Relaxed) / count
    }

    pub fn get_max_inference(&self) -> u64 {
        self.max_inference_us.load(Ordering::Relaxed)
    }

    pub fn get_count(&self) -> u64 {
        self.inference_count.load(Ordering::Relaxed)
    }
}

lazy_static! {
    pub static ref E2E_LATENCY: LatencyStats = LatencyStats::new();
    pub static ref PROC_LATENCY: LatencyStats = LatencyStats::new();
    pub static ref HOT_PATH_STATS: HotPathStats = HotPathStats::new();
    pub static ref LATENCY_MONITOR: Arc<LatencyMonitor> = Arc::new(LatencyMonitor::new());
}
