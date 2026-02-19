use anyhow::Result;
use metrics_exporter_prometheus::PrometheusBuilder;
use std::net::SocketAddr;

/// Инициализирует Prometheus metrics exporter с HTTP-слушателем
///
/// # Arguments
/// * `port` - Порт для HTTP-слушателя (например, 9090)
///
/// # Returns
/// * `Result<()>` - Ok если инициализация прошла успешно
pub fn init_metrics_exporter(port: u16) -> Result<()> {
    let addr: SocketAddr = format!("0.0.0.0:{}", port).parse()?;
    
    // Создаем builder и устанавливаем адрес через переменную окружения
    // metrics-exporter-prometheus использует METRICS_EXPORTER_PROMETHEUS_LISTEN_ADDRESS
    std::env::set_var("METRICS_EXPORTER_PROMETHEUS_LISTEN_ADDRESS", addr.to_string());
    
    // Регистрируем описания метрик для улучшения читаемости в Prometheus/Grafana
    
    // === Метрики WebSocket и сетевые ===
    metrics::describe_counter!(
        "bot_ws_messages_total",
        "Total number of WebSocket messages received from Bybit"
    );
    
    // === Метрики ML инференса ===
    metrics::describe_histogram!(
        "bot_inference_duration_us",
        metrics::Unit::Microseconds,
        "Duration of ML model inference in microseconds"
    );
    
    // === Метрики PnL ===
    metrics::describe_gauge!(
        "bot_realized_pnl_bps",
        metrics::Unit::BasisPoints,
        "Realized PnL in basis points"
    );
    
    metrics::describe_gauge!(
        "bot_unrealized_pnl_bps",
        metrics::Unit::BasisPoints,
        "Unrealized PnL in basis points"
    );
    
    // === Метрики здоровья бота ===
    metrics::describe_gauge!(
        "bot_health_status",
        "Health status of the bot (1 = OK, 0 = critical error/blocked)"
    );
    
    // === Метрики заказов (Задача 189) ===
    metrics::describe_counter!(
        "bot_orders_placed_total",
        "Total number of orders placed"
    );
    
    metrics::describe_counter!(
        "bot_order_rejections_total",
        "Total number of order rejections"
    );
    
    // === Метрики тиков и сигналов (из prometheus.rs) ===
    metrics::describe_counter!(
        "bot_ticks_total",
        "Total incoming ticks"
    );
    
    metrics::describe_counter!(
        "bot_signal_oscillations_handled_total",
        "Total signal oscillations suppressed"
    );
    
    // === Метрики ресурсов системы ===
    metrics::describe_gauge!(
        "bot_memory_usage_bytes",
        metrics::Unit::Bytes,
        "Memory usage in bytes"
    );
    
    metrics::describe_gauge!(
        "bot_cpu_usage_percent",
        metrics::Unit::Percent,
        "CPU usage in percent"
    );
    
    // === Метрики Watchdog ===
    metrics::describe_gauge!(
        "bot_watchdog_stall_seconds",
        metrics::Unit::Seconds,
        "Hot path stall duration in seconds"
    );
    
    metrics::describe_gauge!(
        "bot_watchdog_last_check_timestamp",
        metrics::Unit::Seconds,
        "Last watchdog check timestamp"
    );
    
    // === Метрики выходов и заполнений ===
    metrics::describe_counter!(
        "bot_time_decay_exits_total",
        "Total exits triggered by time decay"
    );
    
    metrics::describe_counter!(
        "bot_maker_fills_total",
        "Total maker fills (rebate-eligible trades)"
    );
    
    metrics::describe_counter!(
        "bot_taker_fills_total",
        "Total taker fills (fee-paying trades)"
    );
    
    // Инициализируем exporter с HTTP listener
    PrometheusBuilder::new()
        .install()
        .map_err(|e| anyhow::anyhow!("Failed to install Prometheus exporter: {}", e))?;
    
    tracing::info!("Prometheus metrics exporter initialized on {}", addr);
    
    Ok(())
}
