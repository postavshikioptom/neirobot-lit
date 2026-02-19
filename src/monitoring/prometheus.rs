use prometheus::{Encoder, IntCounterVec, GaugeVec, Registry, TextEncoder};
use std::sync::OnceLock;
use axum::{http::StatusCode, response::IntoResponse};

static REGISTRY: OnceLock<Registry> = OnceLock::new();
static SYMBOL: OnceLock<String> = OnceLock::new();
pub static TICK_COUNTER: OnceLock<IntCounterVec> = OnceLock::new();
pub static MEMORY_GAUGE: OnceLock<GaugeVec> = OnceLock::new();
pub static CPU_GAUGE: OnceLock<GaugeVec> = OnceLock::new();
pub static WATCHDOG_STALL_GAUGE: OnceLock<GaugeVec> = OnceLock::new();
pub static WATCHDOG_CHECK_GAUGE: OnceLock<GaugeVec> = OnceLock::new();
pub static OSCILLATION_COUNTER: OnceLock<IntCounterVec> = OnceLock::new();
pub static TIME_DECAY_EXIT_COUNTER: OnceLock<IntCounterVec> = OnceLock::new();
pub static MAKER_FILL_COUNTER: OnceLock<IntCounterVec> = OnceLock::new();
pub static TAKER_FILL_COUNTER: OnceLock<IntCounterVec> = OnceLock::new();

pub fn registry() -> &'static Registry {
    REGISTRY.get_or_init(Registry::new)
}

pub fn init_metrics(symbol: &str) {
    let r = registry();
    let _ = SYMBOL.set(symbol.to_string());
    
    let ticks = IntCounterVec::new(
        prometheus::opts!("bot_ticks_total", "Total incoming ticks"),
        &["symbol"]
    ).unwrap();
    r.register(Box::new(ticks.clone())).ok();
    ticks.with_label_values(&[symbol]).inc_by(0); // Force init
    TICK_COUNTER.set(ticks).ok();

    let osc = IntCounterVec::new(
        prometheus::opts!("bot_signal_oscillations_handled_total", "Total signal oscillations suppressed"),
        &["symbol"]
    ).unwrap();
    r.register(Box::new(osc.clone())).ok();
    OSCILLATION_COUNTER.set(osc).ok();

    let memory = GaugeVec::new(
        prometheus::opts!("bot_memory_usage_bytes", "Memory usage in bytes"),
        &["symbol"]
    ).unwrap();
    r.register(Box::new(memory.clone())).ok();
    MEMORY_GAUGE.set(memory).ok();

    let cpu = GaugeVec::new(
        prometheus::opts!("bot_cpu_usage_percent", "CPU usage in percent"),
        &["symbol"]
    ).unwrap();
    r.register(Box::new(cpu.clone())).ok();
    CPU_GAUGE.set(cpu).ok();

    let stall = GaugeVec::new(
        prometheus::opts!("bot_watchdog_stall_seconds", "Hot path stall duration in seconds"),
        &["symbol"]
    ).unwrap();
    r.register(Box::new(stall.clone())).ok();
    WATCHDOG_STALL_GAUGE.set(stall).ok();

    let last_check = GaugeVec::new(
        prometheus::opts!("bot_watchdog_last_check_timestamp", "Last watchdog check timestamp"),
        &["symbol"]
    ).unwrap();
    r.register(Box::new(last_check.clone())).ok();
    WATCHDOG_CHECK_GAUGE.set(last_check).ok();

    let time_decay_exits = IntCounterVec::new(
        prometheus::opts!("bot_time_decay_exits_total", "Total exits triggered by time decay"),
        &["symbol"]
    ).unwrap();
    r.register(Box::new(time_decay_exits.clone())).ok();
    time_decay_exits.with_label_values(&[symbol]).inc_by(0); // Force init
    TIME_DECAY_EXIT_COUNTER.set(time_decay_exits).ok();

    let maker_fills = IntCounterVec::new(
        prometheus::opts!("bot_maker_fills_total", "Total maker fills (rebate-eligible trades)"),
        &["symbol"]
    ).unwrap();
    r.register(Box::new(maker_fills.clone())).ok();
    maker_fills.with_label_values(&[symbol]).inc_by(0); // Force init
    MAKER_FILL_COUNTER.set(maker_fills).ok();

    let taker_fills = IntCounterVec::new(
        prometheus::opts!("bot_taker_fills_total", "Total taker fills (fee-paying trades)"),
        &["symbol"]
    ).unwrap();
    r.register(Box::new(taker_fills.clone())).ok();
    taker_fills.with_label_values(&[symbol]).inc_by(0); // Force init
    TAKER_FILL_COUNTER.set(taker_fills).ok();
}

pub async fn metrics_handler() -> impl IntoResponse {
    let mut buffer = Vec::new();
    let encoder = TextEncoder::new();
    
    // Сбор системных метрик перед отдачей
    update_system_metrics();

    let metric_families = registry().gather();
    encoder.encode(&metric_families, &mut buffer).unwrap();

    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4")],
        buffer
    )
}

fn update_system_metrics() {
    use crate::utils::sys::METRICS;
    use std::sync::atomic::Ordering;
    
    let rss = METRICS.rss_bytes.load(Ordering::Relaxed);
    let cpu = f32::from_bits(METRICS.cpu_usage.load(Ordering::Relaxed));
    
    let symbol = SYMBOL.get().map(|s| s.as_str()).unwrap_or("unknown");

    if let Some(gauge) = MEMORY_GAUGE.get() {
        gauge.with_label_values(&[symbol]).set(rss as f64);
    }
    if let Some(gauge) = CPU_GAUGE.get() {
        gauge.with_label_values(&[symbol]).set(cpu as f64);
    }
}
