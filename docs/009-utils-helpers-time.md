# 009 - Utils Helpers Time

Цель задачи: Реализовать централизованные утилиты для работы со временем. Нам нужны два типа инструментов: абсолютные Unix Timestamps (для API Bybit и меток в данных) и монотонный таймер (для точного замера задержек инференса и сетевых запросов без влияния системных скачков времени).

Файлы для изменения/создания:

src/utils/time.rs (создать)
src/utils/mod.rs (обновить)
Инструкции для Gemini:

Добавить зависимость в Cargo.toml:

chrono = { version = "0.4", features = ["clock"] }
src/utils/time.rs: Реализовать функции получения системного времени и структуру для замера Latency.

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
src/utils/mod.rs:

pub mod logger;
pub mod time;

pub use time::*; // Реэкспорт для доступа как crate::utils::timestamp_ms()
Технические требования:

Использовать беззнаковые типы (u64, u128), так как время не может быть отрицательным.
LatencyTimer должен базироваться на std::time::Instant для защиты от "прыжков" времени.
Все функции должны быть публичными.
Почему это важно: Bybit требует подписи запросов с точностью до миллисекунды. Монотонный таймер позволит нам точно измерить "чистое" время работы нашей модели LiT, отделив его от системных задержек.