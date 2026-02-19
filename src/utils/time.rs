use std::time::{SystemTime, UNIX_EPOCH, Instant};

/// Unix timestamp в миллисекундах (u64 — стандарт для Bybit API)
pub fn timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("System time before UNIX epoch")
        .as_millis() as u64
}

/// Unix timestamp в микросекундах (для детальных логов)
pub fn timestamp_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("System time before UNIX epoch")
        .as_micros() as u64
}

/// Unix timestamp в наносекундах
pub fn timestamp_ns() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("System time before UNIX epoch")
        .as_nanos()
}

/// Текущее время в формате ISO 8601 UTC с миллисекундами
pub fn now_iso() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

/// Монотонный таймер для замера задержек (Inference, Network round-trip).
/// Не зависит от корректировок системного времени (NTP).
pub struct LatencyTimer(Instant);

impl LatencyTimer {
    pub fn new() -> Self {
        Self(Instant::now())
    }

    pub fn elapsed_ms(&self) -> u64 {
        self.0.elapsed().as_millis() as u64
    }

    pub fn elapsed_us(&self) -> u64 {
        self.0.elapsed().as_micros() as u64
    }

    pub fn elapsed_ns(&self) -> u128 {
        self.0.elapsed().as_nanos()
    }
}
