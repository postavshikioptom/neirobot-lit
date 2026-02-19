use std::sync::atomic::{AtomicU64, Ordering};

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
        
        // Обновление min/max через циклы для атомарности
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
        if count == 0 {
            return 0.0;
        }
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

// Глобальные статсы для удобства доступа
lazy_static::lazy_static! {
    pub static ref E2E_LATENCY: LatencyStats = LatencyStats::new();
    pub static ref PROC_LATENCY: LatencyStats = LatencyStats::new();
}
