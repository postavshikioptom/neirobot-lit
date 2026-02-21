use serde::{Deserialize, Serialize};

use rust_decimal::Decimal;
use rust_decimal::prelude::FromPrimitive;

// --- Типы для динамических порогов на основе режимов рынка (Задача 161) ---

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegimeId {
    Quiet = 0,
    Trend = 1,
    Volatile = 2,
    Unknown = 255,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ThresholdOverride {
    pub regime: RegimeId,
    pub buy_threshold: f32,
    pub sell_threshold: f32,
    pub min_confidence: f32,
}

// --- Вспомогательные функции для значений по умолчанию ---

fn default_env() -> String { "dev".to_string() }
fn default_timezone() -> String { "UTC".to_string() }
fn default_log_level() -> String { "info".to_string() }
fn default_log_format() -> String { "full".to_string() }
fn default_log_file() -> String { "bot.log".to_string() }
fn default_rotate() -> bool { true }
fn default_max_size_mb() -> u64 { 100 }
fn default_rotation() -> String { "daily".to_string() }
fn default_max_files() -> usize { 7 }
fn default_console_enabled() -> bool { true }
fn default_max_latency() -> u64 { 500 }
fn default_retry_interval() -> u64 { 1000 }
fn default_max_retries() -> u32 { 5 }
fn default_drawdown_stop() -> f64 { 5.0 }
fn default_max_orders_min() -> u32 { 30 }
fn default_auto_reset() -> bool { true }
fn default_threshold_buy() -> f32 { 0.5 }
fn default_threshold_sell() -> f32 { 0.5 }
fn default_threshold_flat() -> f32 { 0.3 }
fn default_long_threshold() -> Decimal { Decimal::from_f64(0.6).unwrap() }
fn default_short_threshold() -> Decimal { Decimal::from_f64(0.6).unwrap() }
fn default_seq_len() -> usize { 10 }
fn default_features_dim() -> usize { 200 }
fn default_past_returns_lags() -> Vec<usize> { vec![10, 50, 100] }
fn default_initial_balance() -> Decimal { Decimal::from_f64(1000.0).unwrap() }
fn default_close_on_flat() -> bool { false }
fn default_position_sync_interval() -> u64 { 60 }
fn default_buffer_pct() -> Decimal { Decimal::from_f64(0.01).unwrap() }
fn default_leverage() -> Decimal { Decimal::ONE }
fn default_taker_fee_bps() -> Decimal { Decimal::from_f64(6.0).unwrap() } // 0.06% typical for Bybit Taker
fn default_post_only() -> bool { true }
fn default_post_only_retry_limit() -> u32 { 3 }
fn default_limit_timeout_ms() -> u64 { 10000 } // 10 seconds default
fn default_base_delay_ms() -> u64 { 1000 }
fn default_rest_requests_per_second() -> u64 { 20 }
fn default_private_endpoint_per_minute() -> u64 { 600 }
fn default_backoff_base_ms() -> u64 { 500 }
fn default_order_rate() -> u64 { 20 }
fn default_private_rate() -> u64 { 50 }
fn default_max_delay_ms() -> u64 { 60000 }
fn default_max_price_deviation() -> Decimal { Decimal::from_f64(0.02).unwrap() } // 2% по умолчанию
fn default_ping_interval_sec() -> u64 { 20 }
fn default_pong_timeout_sec() -> u64 { 30 }
fn default_warn_rtt_ms() -> u64 { 500 }
fn default_ws_retry_initial_ms() -> u64 { 1000 }
fn default_ws_retry_max_ms() -> u64 { 60000 }
fn default_ws_retry_multiplier() -> f64 { 2.0 }
fn default_ws_retry_jitter() -> f64 { 0.1 }
fn default_tcp_nodelay() -> bool { true }
fn default_socket_recv_buffer_size() -> usize { 1048576 }
fn default_socket_send_buffer_size() -> usize { 1048576 }
fn default_rest_retry_initial_ms() -> u64 { 100 }
fn default_rest_retry_max_ms() -> u64 { 5000 }
fn default_rest_retry_multiplier() -> f64 { 2.0 }
fn default_rest_retry_jitter() -> f64 { 0.1 }
fn default_rest_max_retries() -> u32 { 3 }
fn default_required_permissions() -> Vec<String> { 
    vec!["ContractTrade".to_string(), "Order".to_string(), "Position".to_string()] 
}
fn default_check_api_expiry() -> bool { true }
fn default_min_api_days_left() -> u32 { 7 }
fn default_mass_cancel_threshold() -> usize { 3 }
fn default_ep() -> String { "cpu".to_string() }
fn default_device_id() -> i32 { 0 }
fn default_fusion_method() -> FusionMethod { FusionMethod::WeightedAverage }
fn default_fusion_weights() -> Vec<Decimal> { vec![Decimal::ONE] }
fn default_fusion_min_horizons() -> usize { 2 }
fn default_fusion_principal_idx() -> usize { 0 }
fn default_volatility_target_bps() -> f64 { 7.5 }
fn default_volatility_window() -> usize { 100 }
fn default_volatility_default() -> f64 { 10.0 }
fn default_size_min_multiplier() -> f64 { 0.5 }
fn default_size_max_multiplier() -> f64 { 2.0 }
fn default_stats_window_ms() -> i64 { 60000 } // 1 минута
fn default_stats_max_trades() -> usize { 5000 } // Лимит сделок в очереди
fn default_desync_tolerance() -> f64 { 0.01 } // 1%

// --- Функции default для URL и конфигурации обмена ---
fn default_bybit_category() -> String { "linear".to_string() }
fn default_bybit_api_key_path() -> String { "api_key".to_string() }
fn default_public_url() -> String { "wss://stream.bybit.com/v5/public/linear".to_string() }
fn default_private_ws_url() -> String { "wss://stream.bybit.com/v5/private".to_string() }
fn default_base_url() -> String { "https://api.bybit.com".to_string() }
fn default_request_timeout_sec() -> u64 { 30 }

/// Стратегия входа в позицию (задача 168: Volume-Weighted Entry)
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum EntryStyle {
    AggressiveMarket,  // Market Order (Sweep) - быстрый вход
    PassiveLimit,      // Limit Order с отступом - ждем исполнения
    ChaseBest,         // Limit на Best Bid/Ask - преследуем лучшую цену
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum ChaseMode {
    ToBest,          // Ставить на Best Bid/Ask
    InsideSpread,    // Ставить внутрь спреда на distance_bps
    ToVWAP,          // Ставить по цене текущего VWAP (если возможно)
}

fn default_entry_style() -> EntryStyle { EntryStyle::AggressiveMarket }
fn default_max_entry_slippage_bps() -> u32 { 50 } // 0.5% максимальное проскальзывание
fn default_entry_participation_ratio() -> f64 { 0.1 } // 10% от доступного объема на первых N уровнях
fn default_slicing_enabled() -> bool { true } // Разбиение крупных ордеров на части

// --- Конфигурация проверки свежести сигнала (Задача 169) ---

/// Действие при обнаружении устаревшего сигнала
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StalenessAction {
    Skip,    // Пропустить сигнал
    LogOnly, // Только логировать, но выполнить
}

fn default_max_signal_age_ms() -> u64 { 100 } // 100ms лимит свежести
fn default_max_clock_skew_ms() -> i64 { 5000 } // 5 секунд допустимое расхождение времени
fn default_staleness_action() -> StalenessAction { StalenessAction::Skip }

// --- Конфигурация контроля времени жизни ордеров (Задача 179) ---

/// Действие при обнаружении "зависшего" (stale) ордера
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StaleOrderAction {
    CancelOnly,           // Только отменить ордер
    CancelAndMarketFill,  // Отменить лимит и исполнить по рынку
    Repeg,                // Переставить ордер к best_bid/ask
}

fn default_max_order_life_ms() -> u64 { 30000 }
fn default_stale_order_action() -> StaleOrderAction { StaleOrderAction::CancelOnly }
fn default_min_fill_pct_to_keep() -> f64 { 0.8 }
fn default_stale_check_interval_ms() -> u64 { 1000 }

// Задача 180: Расширенная проверка целостности данных (Data Integrity Checksum Extended)
fn default_checksum_validation_enabled() -> bool { true }
fn default_max_checksum_mismatches() -> u32 { 3 }

fn default_chase_mode() -> ChaseMode { ChaseMode::ToBest }
fn default_chase_threshold() -> Decimal { Decimal::from_f64(2.0).unwrap() }
fn default_chase_distance() -> Decimal { Decimal::from_f64(0.5).unwrap() }
fn default_chase_max_attempts() -> usize { 3 }
fn default_chase_interval() -> u64 { 1000 }
fn default_vwap_filter() -> bool { false }
fn default_vwap_filter_threshold() -> Decimal { Decimal::from_f64(10.0).unwrap() } // 10 bps (0.1%)
fn default_close_on_exit() -> bool { true }
fn default_emergency_timeout() -> u64 { 5000 }
fn default_signal_min_confidence() -> f64 { 0.45 }
fn default_signal_full_confidence() -> f64 { 0.85 }
fn default_signal_size_mult_min() -> f64 { 0.5 }
fn default_signal_size_mult_max() -> f64 { 2.0 }
fn default_total_size_mult_max() -> f64 { 3.0 }
fn default_max_daily_loss() -> Decimal { Decimal::from_f64(2.0).unwrap() }
fn default_daily_reset_hour() -> u32 { 0 }
fn default_max_trades_limit() -> usize { 0 }
fn default_max_trades_window() -> u64 { 3600 }
fn default_max_inactivity_ms() -> u64 { 5000 }
fn default_close_on_inactivity() -> bool { false }
fn default_threshold_base() -> f64 { 0.55 }
fn default_threshold_loss_mult() -> f64 { 0.05 }
fn default_threshold_max() -> f64 { 0.85 }
fn default_threshold_min() -> f64 { 0.51 }
fn default_threshold_max_streak() -> usize { 5 }
fn default_max_network_latency_micros() -> u64 { 200_000 } // 200ms
fn default_max_inference_latency_micros() -> u64 { 50_000 } // 50ms
fn default_max_total_latency_micros() -> u64 { 250_000 } // 250ms
fn default_max_latency_rejects_limit() -> usize { 3 }
fn default_obi_threshold() -> f64 { 0.7 }
fn default_obi_depth() -> usize { 10 }
fn default_lockout_period_sec() -> u64 { 0 }
fn default_lockout_streak_threshold() -> usize { 0 }
fn default_stop_file_name() -> String { "STOP".to_string() }
fn default_ack_extension() -> String { "DONE".to_string() }
fn default_global_stop_enabled() -> bool { true }
fn default_stop_check_interval_ms() -> u64 { 1000 }
fn default_reconciliation_interval_sec() -> u64 { 60 }
fn default_sync_on_desync() -> bool { true }
fn default_price_desync_threshold() -> Decimal { Decimal::from_f64(0.01).unwrap() } // 1 tick_size по умолчанию
fn default_min_flip_interval_ms() -> u64 { 200 } // 200ms default
fn default_max_slice_size() -> f64 { 0.0 } // 0 = disabled
fn default_regime_overrides() -> Vec<ThresholdOverride> { vec![] }
fn default_max_impact_bps() -> f64 { 8.0 }
fn default_min_top_multiple() -> f64 { 1.5 }
fn default_adjust_size_if_thin() -> bool { true }
fn default_time_decay_enabled() -> bool { false }
fn default_max_age_long_ms() -> u64 { 10000 }
fn default_max_age_short_ms() -> u64 { 10000 }
fn default_force_taker_confidence() -> f64 { 0.95 }
fn default_maker_offset_step_ticks() -> u32 { 1 }
fn default_max_post_only_rejects() -> u32 { 3 }
fn default_repeg_threshold_ticks() -> u32 { 2 }
fn default_rebate_wait_timeout_ms() -> u64 { 5000 }
fn default_vpin_buckets_count() -> usize { 50 }
fn default_vpin_volume_threshold() -> f64 { 1000.0 }
fn default_layering_std_threshold() -> f64 { 0.0001 }
fn default_spoofing_min_vol_multiple() -> f64 { 3.0 }
fn default_tp_stages() -> Vec<TpStage> { vec![] }
fn default_tp_close_all_on_min_qty() -> bool { true }

fn default_max_process_memory_mb() -> u64 { 512 }
fn default_max_avg_latency_ms() -> u64 { 200 }
fn default_health_check_interval_s() -> u64 { 10 }
fn default_pnl_volatility_window() -> usize { 20 }
fn default_max_pnl_std_dev_bps() -> u32 { 100 }
fn default_max_pnl_z_score_threshold() -> f64 { 3.0 }

// --- Задача 187: Политика автоматизированной очистки данных ---
fn default_raw_data_retention_days() -> u32 { 14 } // Срок хранения снимков стакана и сделок
fn default_max_data_dir_size_gb() -> u64 { 50 } // Жесткий лимит объема папки data/
fn default_cleanup_interval_hours() -> u32 { 12 } // Периодичность проверки

// --- Конфигурация динамического сокращения лимитов позиции (Задача 178) ---
fn default_drawdown_scaling_start_pct() -> f64 { 5.0 } // Просадка, при которой начинается сокращение лимитов
fn default_volatility_threshold() -> f64 { 1.5 } // Коэффициент превышения медианной волатильности
fn default_min_scale_factor() -> f64 { 0.2 } // Минимальный порог сокращения (20% от базы)
fn default_recovery_rate() -> f64 { 0.05 } // Скорость восстановления лимитов (5% за шаг)

// --- Конфигурация частичной фиксации прибыли (Задача 166) ---

// --- Конфигурация динамического скользящего стоп-лосса (Задача 167) ---

// --- Конфигурация фильтра по ставкам финансирования (Задача 170) ---
fn default_max_funding_rate_bps() -> u32 { 30 } // 30 bps = 0.03%
fn default_avoid_settlement_window_ms() -> u64 { 300_000 } // 5 minutes
fn default_min_confidence_to_ignore_funding() -> f64 { 0.75 } // 75% confidence
fn default_max_state_age_ms() -> u64 { 60000 }
fn default_enable_fill_rate_logging() -> bool { false }
fn default_enable_impact_logging() -> bool { false }
 // 60 секунд (задача 190)
fn default_critical_signal() -> f32 { 0.75 }
fn default_max_size_ratio() -> f64 { 0.3 }
fn default_sor_default_urgency() -> f32 { 0.5 }
fn default_slice_interval_ms() -> u64 { 100 }
fn default_iceberg_randomize() -> f32 { 0.2 }
fn default_iceberg_price_dev_bps() -> u32 { 10 }
// Задача 208: Параметры переключения Passive -> Aggressive
fn default_switch_base_timeout_ms() -> u64 { 500 }  // 500мс базовый timeout
// Задача 210: Адаптивные пороги отмены ордеров
// Параметры установлены максимально мягко, чтобы не мешать торговле на разных монетах
// Реальные спреды: BTC ~10 bps, альткойны ~200-2000 bps, экстремальные ~20000 bps
// Реальная волатильность: низкая ~100 bps, средняя ~300 bps, высокая ~1000 bps, экстремальная ~2000+ bps
fn default_adaptive_thresholds_enabled() -> bool { false }  // По умолчанию выключено
fn default_base_threshold_bps() -> f64 { 500.0 }  // Базовый порог 500 bps (5%) - очень мягко
fn default_vol_multiplier() -> f64 { 0.01 }  // Коэффициент для волатильности (очень маленький)
fn default_spread_multiplier_adaptive() -> f64 { 0.001 }  // Коэффициент для спреда (очень маленький)
fn default_min_threshold_bps() -> f64 { 100.0 }  // Минимальный порог 100 bps (1%)
fn default_max_threshold_bps() -> f64 { 1000.0 }  // Максимальный порог 1000 bps (10%)

fn default_switch_base_distance_bps() -> u32 { 5 }  // 5 базисных пунктов
fn default_max_switches_per_signal() -> u8 { 1 }    // Максимум 1 переключение
fn default_persistence_interval_sec() -> u64 { 60 }
fn default_max_state_backups() -> u32 { 3 }

// Задача 221: Real-time Equity Streamer
fn default_monitoring_port() -> u16 { 9001 }
fn default_min_update_ms() -> u64 { 100 }
fn default_balance_sync_interval() -> u64 { 60 }
fn default_alert_dedup_ttl_secs() -> u64 { 600 }
fn default_confidence_sample_rate() -> u32 { 100 }
fn default_entropy_drift_threshold() -> f32 { 1.5 }
fn default_enable_realtime_drift_check() -> bool { true }
fn default_drift_stop_enabled() -> bool { false }
fn default_drift_scale_factor() -> f32 { 0.5 }

// Задача 228: Model Hot-Swap defaults
fn default_enable_model_hotswap() -> bool { false }

// Задача 225: Resource Profiler defaults
fn default_cpu_max_pct() -> f32 { 80.0 }
fn default_mem_growth_kb_min() -> u64 { 10240 } // 10 MB
fn default_sample_interval_sec() -> u64 { 5 }
fn default_ema_alpha() -> f32 { 0.2 }
fn default_leak_detection_window() -> usize { 10 } // 10 минут

// Задача 232: Self-Match Prevention (SMP) defaults
fn default_smp_type() -> String { "None".to_string() }
fn default_local_smp_enabled() -> bool { false }

// Задача 234: Адаптивный Rate Limit и динамический Backoff defaults
fn default_rate_limit_threshold_pct() -> f64 { 0.15 } // 15% порог включения замедления
fn default_backoff_base_ms() -> u64 { 250 }
fn default_cleanup_interval_min() -> u64 { 60 }
fn default_max_stale_age_min() -> u64 { 120 }
fn default_auto_cancel_stale() -> bool { false } // 250 мс базовая задержка

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TSLMode {
    Bot,      // Логика на стороне бота
    Exchange, // Нативный функционал биржи
}

/// Конфигурация мониторинга системных ресурсов (задача 225)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceThresholdsConfig {
    /// Максимальный процент использования CPU (EMA)
    #[serde(default = "default_cpu_max_pct")]
    pub cpu_max_pct: f32,
    /// Минимальный рост памяти в KB для детекции утечки
    #[serde(default = "default_mem_growth_kb_min")]
    pub mem_growth_kb_min: u64,
    /// Интервал сбора метрик в секундах
    #[serde(default = "default_sample_interval_sec")]
    pub sample_interval_sec: u64,
    /// Коэффициент сглаживания EMA (0.0-1.0)
    #[serde(default = "default_ema_alpha")]
    pub ema_alpha: f32,
    /// Размер окна для детекции утечек памяти
    #[serde(default = "default_leak_detection_window")]
    pub leak_detection_window: usize,
}

impl Default for ResourceThresholdsConfig {
    fn default() -> Self {
        Self {
            cpu_max_pct: default_cpu_max_pct(),
            mem_growth_kb_min: default_mem_growth_kb_min(),
            sample_interval_sec: default_sample_interval_sec(),
            ema_alpha: default_ema_alpha(),
            leak_detection_window: default_leak_detection_window(),
        }
    }
}

/// Конфигурация изоляции ресурсов процесса (задача 230)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SystemConfig {
    /// Номер ядра CPU для привязки процесса (None = без привязки)
    pub cpu_core: Option<usize>,
    /// Максимальный лимит памяти в мегабайтах (для мягкого контроля)
    #[serde(default = "default_max_memory_mb")]
    pub max_memory_mb: u64,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            cpu_core: None,
            max_memory_mb: default_max_memory_mb(),
        }
    }
}

fn default_max_memory_mb() -> u64 { 512 } // 512 MB по умолчанию

fn default_tsl_mode() -> TSLMode { TSLMode::Bot }
fn default_tsl_activation_bps() -> u32 { 200 }  // 2% профита для активации
fn default_tsl_distance_bps() -> u32 { 100 }   // 1% отступа от экстремума
fn default_tsl_step_bps() -> u32 { 10 }        // 0.1% минимальный шаг обновления

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrailingStopConfig {
    #[serde(default = "default_tsl_mode")]
    pub tsl_mode: TSLMode,
    #[serde(default = "default_tsl_activation_bps")]
    pub tsl_activation_bps: u32,
    #[serde(default = "default_tsl_distance_bps")]
    pub tsl_distance_bps: u32,
    #[serde(default = "default_tsl_step_bps")]
    pub tsl_step_bps: u32,
}

/// Этап частичной фиксации прибыли
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TpStage {
    /// Порог в базисных пунктах от цены входа (например, 50 = 0.5%)
    pub threshold_bps: u32,
    /// Процент от initial_size для закрытия (0.0 - 1.0, например 0.5 = 50%)
    pub close_pct: f64,
}

// --- Конфигурация анти-адверсариальной защиты ---

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AdversarialConfig {
    /// Количество корзин для расчета VPIN (например, 50)
    #[serde(default = "default_vpin_buckets_count")]
    pub vpin_buckets_count: usize,
    /// Объем одной корзины для VPIN расчета
    #[serde(default = "default_vpin_volume_threshold")]
    pub vpin_volume_threshold: f64,
    /// Порог стандартного отклонения ценовых шагов для детекции Layering
    #[serde(default = "default_layering_std_threshold")]
    pub layering_std_threshold: f64,
    /// Во сколько раз объем уровня должен превышать средний, чтобы считаться спуфингом
    #[serde(default = "default_spoofing_min_vol_multiple")]
    pub spoofing_min_vol_multiple: f64,
}

impl Default for AdversarialConfig {
    fn default() -> Self {
        Self {
            vpin_buckets_count: default_vpin_buckets_count(),
            vpin_volume_threshold: default_vpin_volume_threshold(),
            layering_std_threshold: default_layering_std_threshold(),
            spoofing_min_vol_multiple: default_spoofing_min_vol_multiple(),
        }
    }
}

// --- Конфигурация Backtest ---

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct BacktestConfig {
    pub history_path: String,
    pub output_dir: String,
    pub start_time: Option<String>,
    pub end_time: Option<String>,
}

// --- Глобальная конфигурация ---

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct GlobalConfig {
    #[serde(default)]
    pub general: GeneralConfig,
    #[serde(default)]
    pub logging: LoggingConfig,
    #[serde(default)]
    pub trading_defaults: TradingDefaultsConfig,
    #[serde(default)]
    pub risk_defaults: RiskDefaultsConfig,
    pub backtest: Option<BacktestConfig>,
    #[serde(default)]
    pub monitoring: Option<crate::monitoring::health::HealthConfig>,
    /// Telegram Bot Token для отправки алертов (поддерживает префикс ENC:)
    pub telegram_token: Option<String>,
    /// ID чата по умолчанию для отправки алертов
    pub default_chat_id: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct GeneralConfig {
    #[serde(default = "default_env")]
    pub env: String,
    #[serde(default = "default_timezone")]
    pub timezone: String,
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self { env: default_env(), timezone: default_timezone() }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct LoggingConfig {
    #[serde(default = "default_log_level")]
    pub level: String,
    #[serde(default = "default_log_format")]
    pub format: String,
    #[serde(default = "default_log_file")]
    pub file_name: String,
    #[serde(default = "default_rotate")]
    pub rotate: bool,
    #[serde(default = "default_max_size_mb")]
    pub max_size_mb: u64,
    #[serde(default = "default_rotation")]
    pub rotation: String,
    #[serde(default = "default_max_files")]
    pub max_files: usize,
    #[serde(default = "default_console_enabled")]
    pub console_enabled: bool,
    #[serde(default = "default_logger_queue_size")]
    pub logger_queue_size: usize,
    #[serde(default = "default_log_retention_days")]
    pub log_retention_days: u64,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            format: default_log_format(),
            file_name: default_log_file(),
            rotate: default_rotate(),
            max_size_mb: default_max_size_mb(),
            rotation: default_rotation(),
            max_files: default_max_files(),
            console_enabled: default_console_enabled(),
            logger_queue_size: default_logger_queue_size(),
            log_retention_days: default_log_retention_days(),
        }
    }
}

fn default_snapshot_interval() -> u64 { 100 }
fn default_latency_report_interval() -> u64 { 60 }
fn default_logger_queue_size() -> usize { 10_000 }
fn default_log_retention_days() -> u64 { 7 }

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct TradingDefaultsConfig {
    #[serde(default = "default_max_latency")]
    pub max_latency_ms: u64,
    #[serde(default = "default_retry_interval")]
    pub retry_interval_ms: u64,
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    #[serde(default = "default_snapshot_interval")]
    pub snapshot_interval_ms: u64,
}

impl Default for TradingDefaultsConfig {
    fn default() -> Self {
        Self {
            max_latency_ms: default_max_latency(),
            retry_interval_ms: default_retry_interval(),
            max_retries: default_max_retries(),
            snapshot_interval_ms: default_snapshot_interval(),
        }
    }
}

fn default_max_consecutive_rejections() -> u32 { 5 }
fn default_max_total_rejections_in_window() -> u32 { 10 }
fn default_rejection_window_ms() -> u64 { 60000 }
fn default_ignored_rejection_codes() -> Vec<i32> { vec![34026] }

// Задача 176: Защита от дублирования ордеров
fn default_duplicate_window_ms() -> u64 { 5000 } // 5 секунд окно детекции
fn default_duplicate_qty_tolerance_pct() -> f64 { 0.01 } // 1% допуск по объему
fn default_duplicate_price_tolerance_ticks() -> u32 { 2 } // 2 тика допуск по цене
fn default_order_intent_timeout_ms() -> u64 { 30000 } // 30 секунд таймаут очистки

// Задача 177: Расширенный фильтр отклонения цены
fn default_max_price_deviation_bps() -> u32 { 500 } // 500 bps = 5%
fn default_halt_on_extreme_deviation() -> bool { false }

// Задача 231: Обработка ошибок при нехватке маржи
fn default_margin_error_backoff_minutes() -> u64 { 5 } // 5 минут штрафного периода
fn default_margin_penalty_multiplier() -> f64 { 0.5 } // Снижение размера позиции до 50%

// Задача 233: Обработка ошибок Price Band и стабилизация цен
fn default_price_band_cooldown_sec() -> u64 { 60 } // 60 секунд базового охлаждения
fn default_max_mark_deviation() -> f64 { 0.02 } // 2% максимального отклонения мида от марки
fn default_max_spread_bps_shock() -> f64 { 15.0 } // 15 bps максимального спреда для выхода из шока

/// Эталон цены для проверки отклонения (Задача 177)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PriceReferenceSource {
    MidPrice,  // Средняя цена между best_bid и best_ask
    LastPrice, // Цена последней сделки
    MarkPrice, // Маркированная цена (индексная цена)
    Both,      // Проверка по обоим эталонам
}

impl Default for PriceReferenceSource {
    fn default() -> Self {
        PriceReferenceSource::MidPrice
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(default)]
pub struct RiskDefaultsConfig {
    #[serde(default = "default_drawdown_stop")]
    pub drawdown_stop_pct: f64,
    pub max_orders_per_minute: u32,
    pub max_open_orders: Option<u32>,
    pub max_position_size: Option<Decimal>,
    pub max_notional_usd: Option<Decimal>,
    pub max_margin_usd: Option<Decimal>,
    #[serde(default = "default_leverage")]
    pub leverage: Decimal,
    pub max_daily_drawdown_usd: Option<Decimal>,
    pub max_daily_drawdown_pct: Option<Decimal>,
    pub auto_reset_at_midnight: bool,
    pub max_drawdown_pct: Option<Decimal>,
    pub max_spread_bps: Option<u32>, // Макс. спред в базисных пунктах (1 bps = 0.01%)
    pub max_price_deviation_pct: Option<Decimal>, // Макс. отклонение цены (2% по умолчанию)
    pub max_trades_limit: usize,         // Макс. кол-во сделок (0 = отключено)
    pub max_trades_window_sec: u64,     // Скользящее окно в секундах
    pub max_inactivity_ms: u64,
    pub close_position_on_inactivity: bool,

    #[serde(default = "default_max_process_memory_mb")]
    pub max_process_memory_mb: u64,
    #[serde(default = "default_max_avg_latency_ms")]
    pub max_avg_latency_ms: u64,
    #[serde(default = "default_health_check_interval_s")]
    pub health_check_interval_s: u64,

    #[serde(default = "default_pnl_volatility_window")]
    pub pnl_volatility_window: usize,
    #[serde(default = "default_max_pnl_std_dev_bps")]
    pub max_pnl_std_dev_bps: u32,
    #[serde(default = "default_max_pnl_z_score_threshold")]
    pub max_pnl_z_score_threshold: f64,

    // Задача 175: Лимиты отклонения ордеров
    #[serde(default = "default_max_consecutive_rejections")]
    pub max_consecutive_rejections: u32,
    #[serde(default = "default_max_total_rejections_in_window")]
    pub max_total_rejections_in_window: u32,
    #[serde(default = "default_rejection_window_ms")]
    pub rejection_window_ms: u64,
    #[serde(default = "default_ignored_rejection_codes")]
    pub ignored_rejection_codes: Vec<i32>,

    // Задача 176: Защита от дублирования ордеров
    #[serde(default = "default_duplicate_window_ms")]
    pub duplicate_window_ms: u64,
    #[serde(default = "default_duplicate_qty_tolerance_pct")]
    pub duplicate_qty_tolerance_pct: f64,
    #[serde(default = "default_duplicate_price_tolerance_ticks")]
    pub duplicate_price_tolerance_ticks: u32,
    #[serde(default = "default_order_intent_timeout_ms")]
    pub order_intent_timeout_ms: u64,

    // Задача 177: Расширенный фильтр отклонения цены
    #[serde(default = "default_max_price_deviation_bps")]
    pub max_price_deviation_bps: u32,
    #[serde(default)]
    pub price_reference_source: PriceReferenceSource,
    #[serde(default = "default_halt_on_extreme_deviation")]
    pub halt_on_extreme_deviation: bool,

    // Задача 178: Динамическое сокращение лимитов позиции
    #[serde(default = "default_drawdown_scaling_start_pct")]
    pub drawdown_scaling_start_pct: f64,
    #[serde(default = "default_volatility_threshold")]
    pub volatility_threshold: f64,
    #[serde(default = "default_min_scale_factor")]
    pub min_scale_factor: f64,
    #[serde(default = "default_recovery_rate")]
    pub recovery_rate: f64,

    // Задача 179: Лимит времени на неисполненные ордера
    #[serde(default = "default_max_order_life_ms")]
    pub max_order_life_ms: u64,
    #[serde(default = "default_stale_order_action")]
    pub stale_order_action: StaleOrderAction,
    #[serde(default = "default_min_fill_pct_to_keep")]
    pub min_fill_pct_to_keep: f64,
    #[serde(default = "default_stale_check_interval_ms")]
    pub stale_check_interval_ms: u64,

    // Задача 180: Расширенная проверка целостности данных (Data Integrity Checksum Extended)
    #[serde(default = "default_checksum_validation_enabled")]
    pub checksum_validation_enabled: bool,
    #[serde(default = "default_max_checksum_mismatches")]
    pub max_checksum_mismatches: u32,

    // Задача 187: Политика автоматизированной очистки данных
    #[serde(default = "default_raw_data_retention_days")]
    pub raw_data_retention_days: u32,
    #[serde(default = "default_max_data_dir_size_gb")]
    pub max_data_dir_size_gb: u64,
    #[serde(default = "default_cleanup_interval_hours")]
    pub cleanup_interval_hours: u32,

    // Задача №198: Перенос фильтра ликвидности в общие риски для статической диспетчеризации
    #[serde(default)]
    pub liquidity_filter: Option<LiquidityFilterConfig>,

    // Задача 231: Обработка ошибок при нехватке маржи
    #[serde(default = "default_margin_error_backoff_minutes")]
    pub margin_error_backoff_minutes: u64,
    #[serde(default = "default_margin_penalty_multiplier")]
    pub margin_penalty_multiplier: f64,

    // Задача 233: Обработка ошибок Price Band и стабилизация цен
    #[serde(default = "default_price_band_cooldown_sec")]
    pub price_band_cooldown_sec: u64,
    #[serde(default = "default_max_mark_deviation")]
    pub max_mark_deviation: f64,
    #[serde(default = "default_max_spread_bps_shock")]
    pub max_spread_bps_shock: f64,
}

pub type RiskConfig = RiskDefaultsConfig;

impl Default for RiskDefaultsConfig {
    fn default() -> Self {
        Self {
            drawdown_stop_pct: default_drawdown_stop(),
            max_orders_per_minute: default_max_orders_min(),
            max_open_orders: None,
            max_position_size: None,
            max_notional_usd: None,
            max_margin_usd: None,
            max_daily_drawdown_usd: None,
            max_daily_drawdown_pct: None,
            auto_reset_at_midnight: default_auto_reset(),
            max_drawdown_pct: None,
            max_spread_bps: None,
            max_price_deviation_pct: Some(default_max_price_deviation()), // 2% по умолчанию
            max_trades_limit: 0,
            max_trades_window_sec: 60,
            max_inactivity_ms: 5000,
            close_position_on_inactivity: false,
            max_process_memory_mb: default_max_process_memory_mb(),
            max_avg_latency_ms: default_max_avg_latency_ms(),
            health_check_interval_s: default_health_check_interval_s(),
            pnl_volatility_window: default_pnl_volatility_window(),
            max_pnl_std_dev_bps: default_max_pnl_std_dev_bps(),
            max_pnl_z_score_threshold: default_max_pnl_z_score_threshold(),
            max_consecutive_rejections: default_max_consecutive_rejections(),
            max_total_rejections_in_window: default_max_total_rejections_in_window(),
            rejection_window_ms: default_rejection_window_ms(),
            ignored_rejection_codes: default_ignored_rejection_codes(),
            duplicate_window_ms: default_duplicate_window_ms(),
            duplicate_qty_tolerance_pct: default_duplicate_qty_tolerance_pct(),
            duplicate_price_tolerance_ticks: default_duplicate_price_tolerance_ticks(),
            order_intent_timeout_ms: default_order_intent_timeout_ms(),
            max_price_deviation_bps: default_max_price_deviation_bps(),
            price_reference_source: PriceReferenceSource::default(),
            halt_on_extreme_deviation: default_halt_on_extreme_deviation(),
            drawdown_scaling_start_pct: default_drawdown_scaling_start_pct(),
            volatility_threshold: default_volatility_threshold(),
            min_scale_factor: default_min_scale_factor(),
            recovery_rate: default_recovery_rate(),
            max_order_life_ms: default_max_order_life_ms(),
            stale_order_action: default_stale_order_action(),
            min_fill_pct_to_keep: default_min_fill_pct_to_keep(),
            stale_check_interval_ms: default_stale_check_interval_ms(),
            checksum_validation_enabled: default_checksum_validation_enabled(),
            max_checksum_mismatches: default_max_checksum_mismatches(),
            raw_data_retention_days: default_raw_data_retention_days(),
            max_data_dir_size_gb: default_max_data_dir_size_gb(),
            cleanup_interval_hours: default_cleanup_interval_hours(),
            liquidity_filter: None,
            margin_error_backoff_minutes: default_margin_error_backoff_minutes(),
            margin_penalty_multiplier: default_margin_penalty_multiplier(),
            price_band_cooldown_sec: default_price_band_cooldown_sec(),
            max_mark_deviation: default_max_mark_deviation(),
            max_spread_bps_shock: default_max_spread_bps_shock(),
        }
    }
}

// --- Конфигурация ONNX ---

fn default_intra_threads() -> Option<usize> { None }
fn default_inter_threads() -> Option<usize> { None }
fn default_execution_mode() -> OnnxExecutionMode { OnnxExecutionMode::Sequential }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OnnxExecutionMode {
    Sequential,
    Parallel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OnnxConfig {
    #[serde(default = "default_ep")]
    pub execution_provider: String, // "cpu", "cuda", "tensorrt"
    #[serde(default = "default_device_id")]
    pub device_id: i32,             // ID GPU (по умолчанию 0)
    pub intra_threads: Option<usize>, // Для CPU-оптимизации
    pub inter_threads: Option<usize>, // Для CPU-оптимизации
    #[serde(default = "default_execution_mode")]
    pub execution_mode: OnnxExecutionMode,
}

impl Default for OnnxConfig {
    fn default() -> Self {
        Self {
            execution_provider: default_ep(),
            device_id: default_device_id(),
            intra_threads: default_intra_threads(),
            inter_threads: default_inter_threads(),
            execution_mode: default_execution_mode(),
        }
    }
}

// --- Конфигурация слияния предсказаний (Multi-Horizon Fusion) ---

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FusionMethod {
    WeightedAverage, // Взвешенное среднее
    Consensus,       // Согласие большинства
    Principal,       // Приоритет одного горизонта
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FusionConfig {
    pub method: FusionMethod,
    pub weights: Vec<Decimal>,    // Веса горизонтов
    pub min_horizons: usize,      // Для Consensus
    pub principal_idx: usize,     // Для Principal
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            method: default_fusion_method(),
            weights: default_fusion_weights(),
            min_horizons: default_fusion_min_horizons(),
            principal_idx: default_fusion_principal_idx(),
        }
    }
}

// --- Конфигурация биржи ---

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct ExchangeConfig {
    #[serde(default)]
    pub bybit: BybitConfig,
    #[serde(default)]
    pub websocket: WebsocketConfig,
    #[serde(default)]
    pub rest: RestConfig,
    #[serde(default)]
    pub rate_limits: RateLimitsConfig,
    #[serde(default = "default_ws_retry_initial_ms")]
    pub ws_retry_initial_ms: u64,
    #[serde(default = "default_ws_retry_max_ms")]
    pub ws_retry_max_ms: u64,
    #[serde(default = "default_ws_retry_multiplier")]
    pub ws_retry_multiplier: f64,
    #[serde(default = "default_ws_retry_jitter")]
    pub ws_retry_jitter: f64,
    #[serde(default = "default_rest_retry_initial_ms")]
    pub rest_retry_initial_ms: u64,
    #[serde(default = "default_rest_retry_max_ms")]
    pub rest_retry_max_ms: u64,
    #[serde(default = "default_rest_retry_multiplier")]
    pub rest_retry_multiplier: f64,
    #[serde(default = "default_rest_retry_jitter")]
    pub rest_retry_jitter: f64,
    #[serde(default = "default_rest_max_retries")]
    pub rest_max_retries: u32,
    #[serde(default = "default_required_permissions")]
    pub required_permissions: Vec<String>,
    #[serde(default = "default_check_api_expiry")]
    pub check_api_expiry: bool,
    #[serde(default = "default_min_api_days_left")]
    pub min_api_days_left: u32,
    #[serde(default = "default_mass_cancel_threshold")]
    pub mass_cancel_threshold: usize,
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct BybitConfig {
    #[serde(default = "default_bybit_category")]
    pub category: String,
    #[serde(default = "default_bybit_api_key_path")]
    pub api_key_path: String,
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct ChaosConfig {
    #[serde(default = "default_packet_loss_rate")]
    pub packet_loss_rate: f64,
    #[serde(default = "default_mean_latency_ms")]
    pub mean_latency_ms: u64,
}

fn default_packet_loss_rate() -> f64 { 0.0 }
fn default_mean_latency_ms() -> u64 { 0 }

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct WebsocketConfig {
    #[serde(default = "default_public_url")]
    pub public_url: String,
    #[serde(default = "default_private_ws_url")]
    pub private_ws_url: String,
    #[serde(default = "default_ping_interval_sec")]
    pub ping_interval_sec: u64,
    #[serde(default = "default_pong_timeout_sec")]
    pub pong_timeout_sec: u64,
    #[serde(default = "default_warn_rtt_ms")]
    pub warn_rtt_ms: u64,
    pub max_subscriptions_per_connection: u64,
    #[serde(default = "default_base_delay_ms")]
    pub base_delay_ms: u64,
    #[serde(default = "default_max_delay_ms")]
    pub max_delay_ms: u64,
    pub max_attempts: Option<u32>,
    #[serde(default)]
    pub verify_checksum: bool,
    #[serde(default = "default_tcp_nodelay")]
    pub tcp_nodelay: bool,
    #[serde(default = "default_socket_recv_buffer_size")]
    pub socket_recv_buffer_size: usize,
    #[serde(default = "default_socket_send_buffer_size")]
    pub socket_send_buffer_size: usize,
    #[serde(default)]
    pub chaos: Option<ChaosConfig>,
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct RestConfig {
    #[serde(default = "default_base_url")]
    pub base_url: String,
    #[serde(default = "default_request_timeout_sec")]
    pub request_timeout_sec: u64,
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct RateLimitsConfig {
    #[serde(default = "default_order_rate")]
    pub order_rate: u64,
    #[serde(default = "default_private_rate")]
    pub private_rate: u64,
    #[serde(default = "default_rest_requests_per_second")]
    pub rest_requests_per_second: u64,
    #[serde(default = "default_private_endpoint_per_minute")]
    pub private_endpoint_per_minute: u64,
    #[serde(default = "default_backoff_base_ms")]
    pub backoff_base_ms: u64,
}

/// Конфигурация фильтра ликвидности (задача 162)
/// Защищает от входов в "тонкий" стакан с высоким проскальзыванием
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct LiquidityFilterConfig {
    /// Максимально допустимое отклонение VWAP от best_price в базисных пунктах (bps)
    /// Например, 8.0 = 0.08%
    #[serde(default = "default_max_impact_bps")]
    pub max_impact_bps: f64,
    
    /// Минимальное отношение объема на best_price к размеру нашего ордера
    /// Например, 1.5 означает, что на лучшем уровне должно быть в 1.5 раза больше ликвидности
    #[serde(default = "default_min_top_multiple")]
    pub min_top_multiple: f64,
    
    /// Уменьшать размер ордера вместо полной отмены при недостаточной ликвидности
    #[serde(default = "default_adjust_size_if_thin")]
    pub adjust_size_if_thin: bool,
}

impl Default for LiquidityFilterConfig {
    fn default() -> Self {
        Self {
            max_impact_bps: default_max_impact_bps(),
            min_top_multiple: default_min_top_multiple(),
            adjust_size_if_thin: default_adjust_size_if_thin(),
        }
    }
}

/// Конфигурация фильтра по ставкам финансирования (задача 170)
/// Защита от потерь на "дорогом" финансировании и фильтрация сигналов в периоды аномальных ставок
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(default)]
pub struct FundingFilterConfig {
    /// Максимально допустимая ставка финансирования в базисных пунктах (например, 30 bps = 0.03%)
    #[serde(default = "default_max_funding_rate_bps")]
    pub max_funding_rate_bps: u32,
    
    /// Окно перед клирингом, когда вход запрещен при плохой ставке (в миллисекундах)
    #[serde(default = "default_avoid_settlement_window_ms")]
    pub avoid_settlement_window_ms: u64,
    
    /// Порог уверенности сигнала, при котором мы игнорируем фандинг
    /// Если confidence >= этого значения, вход разрешен несмотря на высокий фандинг
    #[serde(default = "default_min_confidence_to_ignore_funding")]
    pub min_confidence_to_ignore_funding: f64,
}

impl Default for FundingFilterConfig {
    fn default() -> Self {
        Self {
            max_funding_rate_bps: default_max_funding_rate_bps(),
            avoid_settlement_window_ms: default_avoid_settlement_window_ms(),
            min_confidence_to_ignore_funding: default_min_confidence_to_ignore_funding(),
        }
    }
}

/// Конфигурация Time Decay Exit (задача 163)
/// Принудительно закрывает позиции, если они не достигли цели за установленное время
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(default)]
pub struct TimeDecayConfig {
    /// Включен ли механизм Time Decay
    #[serde(default = "default_time_decay_enabled")]
    pub enabled: bool,
    
    /// Максимальное время жизни Long позиции в миллисекундах
    #[serde(default = "default_max_age_long_ms")]
    pub max_age_long_ms: u64,
    
    /// Максимальное время жизни Short позиции в миллисекундах
    #[serde(default = "default_max_age_short_ms")]
    pub max_age_short_ms: u64,
}

impl Default for TimeDecayConfig {
    fn default() -> Self {
        Self {
            enabled: default_time_decay_enabled(),
            max_age_long_ms: default_max_age_long_ms(),
            max_age_short_ms: default_max_age_short_ms(),
        }
    }
}

// --- Конфигурация Smart Order Routing (SOR) ---

#[derive(Debug, Deserialize, Serialize, Clone, Copy)]
pub struct SorConfig {
    /// Порог силы сигнала для перехода в Aggressive (Taker) режим
    #[serde(default = "default_critical_signal")]
    pub critical_signal: f32,
    
    /// Процент от объема уровня, выше которого включается Slicing (TWAP)
    #[serde(default = "default_max_size_ratio")]
    pub max_size_ratio: f64,
    
    /// Базовая агрессивность (0.0 - 1.0)
    #[serde(default = "default_sor_default_urgency")]
    pub default_urgency: f32,
    
    /// Пауза между частями TWAP в миллисекундах
    #[serde(default = "default_slice_interval_ms")]
    pub slice_interval_ms: u64,
    
    /// Коэффициент вариации display_ratio для Iceberg (0.0 - 0.5)
    #[serde(default = "default_iceberg_randomize")]
    pub iceberg_randomize: f32,
    
    /// Максимальное отклонение цены от начальной для продолжения рефилла (в базисных пунктах)
    #[serde(default = "default_iceberg_price_dev_bps")]
    pub iceberg_price_dev_bps: u32,
    
    /// Задача 208: Базовый timeout для переключения Passive -> Aggressive (мс)
    #[serde(default = "default_switch_base_timeout_ms")]
    pub switch_base_timeout_ms: u64,
    
    /// Задача 208: Базовое расстояние для переключения (в базисных пунктах)
    #[serde(default = "default_switch_base_distance_bps")]
    pub switch_base_distance_bps: u32,
    
    /// Задача 208: Максимальное количество переключений на один сигнал
    #[serde(default = "default_max_switches_per_signal")]
    pub max_switches_per_signal: u8,
}

impl Default for SorConfig {
    fn default() -> Self {
        Self {
            critical_signal: default_critical_signal(),
            max_size_ratio: default_max_size_ratio(),
            default_urgency: default_sor_default_urgency(),
            slice_interval_ms: default_slice_interval_ms(),
            iceberg_randomize: default_iceberg_randomize(),
            iceberg_price_dev_bps: default_iceberg_price_dev_bps(),
            switch_base_timeout_ms: default_switch_base_timeout_ms(),
            switch_base_distance_bps: default_switch_base_distance_bps(),
            max_switches_per_signal: default_max_switches_per_signal(),
        }
    }
}

// --- Конфигурация бота (Per-Symbol) ---

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(default)]
pub struct BotConfig {
    pub symbol: String,
    pub model_path: std::path::PathBuf,
    #[serde(default = "default_seq_len")]
    pub seq_len: usize,
    #[serde(default = "default_features_dim")]
    pub features_dim: usize,
    #[serde(default = "default_past_returns_lags")]
    pub past_returns_lags: Vec<usize>,
    #[serde(default = "default_initial_balance")]
    pub initial_balance: Decimal,
    #[serde(default = "default_close_on_flat")]
    pub close_on_flat: bool,
    #[serde(default = "default_threshold_buy")]
    pub threshold_buy: f32,
    #[serde(default = "default_threshold_sell")]
    pub threshold_sell: f32,
    #[serde(default = "default_threshold_flat")]
    pub threshold_flat: f32,
    #[serde(default = "default_long_threshold")]
    pub long_threshold: Decimal,      // Порог для вероятности Up (например, 0.65)
    #[serde(default = "default_short_threshold")]
    pub short_threshold: Decimal,     // Порог для вероятности Down (например, 0.60)
    pub exit_threshold: Option<Decimal>, // Порог для выхода (если уверенность падает ниже X)
    pub position_idx: i32, // 0: One-Way, 1: Long, 2: Short
    #[serde(default = "default_position_sync_interval")]
    pub position_sync_interval_secs: u64,
    #[serde(default = "default_buffer_pct")]
    pub buffer_pct: Decimal,
    #[serde(default = "default_leverage")]
    pub leverage: Decimal,
    #[serde(default = "default_taker_fee_bps")]
    pub taker_fee_bps: Decimal,
    #[serde(default = "default_post_only")]
    pub post_only: bool,
    #[serde(default = "default_post_only_retry_limit")]
    pub post_only_retry_limit: u32,
    #[serde(default = "default_limit_timeout_ms")]
    pub limit_timeout_ms: u64,
    pub max_position_size: Option<Decimal>,
    pub max_notional_usd: Decimal,      // Максимальный риск в USD (0 = отключено)
    #[serde(rename = "max_drawdown")]
    pub max_drawdown_pct: Option<Decimal>,
    #[serde(default)]
    pub risk: RiskConfig,
    #[serde(default)]
    pub onnx: OnnxConfig,
    #[serde(default)]
    pub fusion: FusionConfig,
    /// Абсолютный максимум спреда (базисные пункты). По умолчанию: 200 (2%)
    pub max_spread_static_bps: Option<u32>,
    /// Во сколько раз текущий спред может превышать средний. По умолчанию: 5.0
    pub spread_multiplier: Option<f32>,
    /// Целевая волатильность (например, 5.0 - 10.0 для HFT)
    #[serde(default = "default_volatility_target_bps")]
    pub volatility_target_bps: f64,
    /// Размер окна для расчета волатильности (например, 100 снапшотов)
    #[serde(default = "default_volatility_window")]
    pub volatility_window: usize,
    /// Значение волатильности по умолчанию до заполнения окна
    #[serde(default = "default_volatility_default")]
    pub volatility_default: f64,
    /// Минимальный мультипликатор размера (например, 0.5)
    #[serde(default = "default_size_min_multiplier")]
    pub size_min_multiplier: f64,
    /// Максимальный мультипликатор размера (например, 2.0)
    #[serde(default = "default_size_max_multiplier")]
    pub size_max_multiplier: f64,
    /// Окно расчета VWAP/TWAP (например, 60000 для 1 минуты)
    #[serde(default = "default_stats_window_ms")]
    pub stats_window_ms: i64,
    /// Лимит сделок в очереди (защита памяти, например, 5000)
    #[serde(default = "default_stats_max_trades")]
    pub stats_max_trades: usize,
    #[serde(default = "default_desync_tolerance")]
    pub desync_tolerance_pct: f64,
    #[serde(default = "default_signal_min_confidence")]
    pub signal_min_confidence: f64,    // Минимальный порог входа (например, 0.45)
    #[serde(default = "default_signal_full_confidence")]
    pub signal_full_confidence: f64,   // Порог для максимального скейлинга (например, 0.85)
    #[serde(default = "default_signal_size_mult_min")]
    pub signal_size_mult_min: f64,     // Множитель при min_confidence (например, 0.5)
    #[serde(default = "default_signal_size_mult_max")]
    pub signal_size_mult_max: f64,     // Множитель при full_confidence (например, 2.0)
    #[serde(default = "default_total_size_mult_max")]
    pub total_size_mult_max: f64,      // Глобальный лимит (vol_mult * signal_mult)
    #[serde(default = "default_chase_mode")]
    pub chase_mode: ChaseMode,
    #[serde(default = "default_chase_threshold")]
    pub chase_threshold_bps: Decimal,
    #[serde(default = "default_chase_distance")]
    pub chase_distance_bps: Decimal,
    #[serde(default = "default_chase_max_attempts")]
    pub chase_max_attempts: usize,
    #[serde(default = "default_chase_interval")]
    pub chase_interval_ms: u64,
    #[serde(default = "default_vwap_filter")]
    pub use_vwap_filter: bool,
    #[serde(default = "default_vwap_filter_threshold")]
    pub vwap_filter_threshold_bps: Decimal,
    #[serde(default = "default_close_on_exit")]
    pub close_on_exit: bool,
    #[serde(default = "default_emergency_timeout")]
    pub emergency_timeout_ms: u64,
    #[serde(default = "default_max_daily_loss")]
    pub max_daily_loss_pct: Decimal,    // Лимит в % (например, 2.0)
    #[serde(default = "default_daily_reset_hour")]
    pub daily_reset_hour_utc: u32,      // Час сброса по UTC (0-23)
    #[serde(default = "default_max_trades_limit")]
    pub max_trades_limit: usize,         // Макс. кол-во сделок (0 = отключено)
    #[serde(default = "default_max_trades_window")]
    pub max_trades_window_sec: u64,     // Скользящее окно в секундах
    #[serde(default = "default_max_inactivity_ms")]
    pub max_inactivity_ms: u64,
    #[serde(default = "default_close_on_inactivity")]
    pub close_position_on_inactivity: bool,
    #[serde(default = "default_threshold_base")]
    pub threshold_base: f64,
    #[serde(default = "default_threshold_loss_mult")]
    pub threshold_loss_mult: f64,
    #[serde(default = "default_threshold_max")]
    pub threshold_max: f64,
    #[serde(default = "default_threshold_min")]
    pub threshold_min: f64,
    #[serde(default = "default_threshold_max_streak")]
    pub threshold_max_streak: usize,
    #[serde(default = "default_max_network_latency_micros")]
    pub max_network_latency_micros: u64,
    #[serde(default = "default_max_inference_latency_micros")]
    pub max_inference_latency_micros: u64,
    #[serde(default = "default_max_total_latency_micros")]
    pub max_total_latency_micros: u64,
    #[serde(default = "default_max_latency_rejects_limit")]
    pub max_latency_rejects_limit: usize,
    #[serde(default = "default_obi_threshold")]
    pub obi_threshold: f64,
    #[serde(default = "default_obi_depth")]
    pub obi_depth: usize,
    #[serde(default = "default_lockout_period_sec")]
    pub lockout_period_sec: u64,          // Длительность блокировки (например, 3600)
    #[serde(default = "default_lockout_streak_threshold")]
    pub lockout_streak_threshold: usize,  // Порог серии убытков (например, 2)
    #[serde(default = "default_stop_file_name")]
    pub stop_file_name: String,           // Имя файла остановки (по умолчанию "STOP")
    #[serde(default = "default_ack_extension")]
    pub ack_extension: String,            // Расширение подтверждения (по умолчанию "DONE")
    #[serde(default = "default_global_stop_enabled")]
    pub global_stop_enabled: bool,        // Проверять ли STOP_ALL в корне (default: true)
    #[serde(default = "default_stop_check_interval_ms")]
    pub stop_check_interval_ms: u64,      // Интервал проверки (default: 1000)
    #[serde(default = "default_reconciliation_interval_sec")]
    pub reconciliation_interval_sec: u64, // Интервал сверки (например, 60)
    #[serde(default = "default_sync_on_desync")]
    pub sync_on_desync: bool,            // Пытаться ли синхронизировать (default: true)
    #[serde(default = "default_price_desync_threshold")]
    pub price_desync_threshold: Decimal, // Допустимая разница в цене (например, tick_size)
    #[serde(default = "default_snapshot_interval")]
    pub snapshot_interval_ms: u64,       // Интервал дампа снимков стакана (задача 132)
    #[serde(default = "default_latency_report_interval")]
    pub latency_report_interval_sec: u64, // Интервал отчета о задержках (задача 133)
    #[serde(default = "default_min_flip_interval_ms")]
    pub min_flip_interval_ms: u64,       // Мин. интервал между переворотами (задача 148)
    #[serde(default = "default_max_slice_size")]
    pub max_slice_size: f64,             // Макс. размер одного слайса (задача 149, 0 = отключено)
    #[serde(default = "default_regime_overrides")]
    pub regime_overrides: Vec<ThresholdOverride>, // Переопределения порогов для режимов рынка (задача 161)
    #[serde(default)]
    pub liquidity_filter: Option<LiquidityFilterConfig>, // Фильтр ликвидности (задача 162)
    #[serde(default)]
    pub time_decay: TimeDecayConfig, // Time Decay Exit (задача 163)
    /// Порог уверенности для немедленного Market-ордера (задача 164)
    #[serde(default = "default_force_taker_confidence")]
    pub force_taker_confidence: f64,
    /// Базовый отступ в тиках от Best Bid/Ask для Maker-ордеров (задача 164)
    #[serde(default = "default_maker_offset_step_ticks")]
    pub maker_offset_step_ticks: u32,
    /// Лимит отклонений Post-Only до переключения в Taker-режим (задача 164)
    #[serde(default = "default_max_post_only_rejects")]
    pub max_post_only_rejects: u32,
    /// Дистанция от цены ордера до Best Price для триггера переустановки (задача 164)
    #[serde(default = "default_repeg_threshold_ticks")]
    pub repeg_threshold_ticks: u32,
    /// Таймаут ожидания исполнения Maker-ордера перед отменой (задача 164)
    #[serde(default = "default_rebate_wait_timeout_ms")]
    pub rebate_wait_timeout_ms: u64,
    /// Конфигурация анти-адверсариальной защиты (задача 165)
    #[serde(default)]
    pub adversarial: AdversarialConfig,
    /// Этапы частичной фиксации прибыли (задача 166)
    #[serde(default = "default_tp_stages")]
    pub tp_stages: Vec<TpStage>,
    /// Закрывать ли позицию полностью, если остаток меньше min_qty (задача 166)
    #[serde(default = "default_tp_close_all_on_min_qty")]
    pub tp_close_all_on_min_qty: bool,
    /// Конфигурация динамического скользящего стоп-лосса (задача 167)
    #[serde(default)]
    pub trailing_stop: TrailingStopConfig,
    /// Стратегия входа в позицию (задача 168: Volume-Weighted Entry)
    #[serde(default = "default_entry_style")]
    pub entry_style: EntryStyle,
    /// Максимальное отклонение VWAP от Mid Price в базисных пунктах (задача 168)
    #[serde(default = "default_max_entry_slippage_bps")]
    pub max_entry_slippage_bps: u32,
    /// Лимит объема: не более X% от доступного на первых N уровнях (задача 168)
    #[serde(default = "default_entry_participation_ratio")]
    pub entry_participation_ratio: f64,
    /// Разбиение крупного ордера на части во времени (задача 168)
    #[serde(default = "default_slicing_enabled")]
    pub slicing_enabled: bool,
    /// Максимальный возраст сигнала в миллисекундах (задача 169)
    #[serde(default = "default_max_signal_age_ms")]
    pub max_signal_age_ms: u64,
    /// Максимально допустимое расхождение времени с биржей в миллисекундах (задача 169)
    #[serde(default = "default_max_clock_skew_ms")]
    pub max_clock_skew_ms: i64,
    /// Действие при обнаружении устаревшего сигнала (задача 169)
    #[serde(default = "default_staleness_action")]
    pub staleness_action: StalenessAction,
    /// Конфигурация фильтра по ставкам финансирования (задача 170)
    #[serde(default)]
    pub funding_filter: FundingFilterConfig,
    /// Максимальный возраст сохраненного состояния в миллисекундах (задача 190)
    #[serde(default = "default_max_state_age_ms")]
    pub max_state_age_ms: u64,
    /// Включить логирование контекста стакана для анализа Fill Rate (задача 203)
    #[serde(default = "default_enable_fill_rate_logging")]
    pub enable_fill_rate_logging: bool,
    /// Включить логирование влияния сделок на Mid-Price (задача 204)
    #[serde(default = "default_enable_impact_logging")]
    pub enable_impact_logging: bool,
    /// Конфигурация Smart Order Routing (задача 206)
    #[serde(default)]
    pub sor: SorConfig,
    /// Задача 210: Адаптивные пороги отмены ордеров
    #[serde(default = "default_adaptive_thresholds_enabled")]
    pub adaptive_thresholds_enabled: bool,
    #[serde(default = "default_base_threshold_bps")]
    pub base_threshold_bps: f64,
    #[serde(default = "default_vol_multiplier")]
    pub vol_multiplier: f64,
    #[serde(default = "default_spread_multiplier_adaptive")]
    pub spread_multiplier_adaptive: f64,
    #[serde(default = "default_min_threshold_bps")]
    pub min_threshold_bps: f64,
    #[serde(default = "default_max_threshold_bps")]
    pub max_threshold_bps: f64,
    /// Интервал сохранения состояния в секундах (задача 218)
    #[serde(default = "default_persistence_interval_sec")]
    pub persistence_interval_sec: u64,
    /// Максимальное количество резервных копий состояния (задача 218)
    #[serde(default = "default_max_state_backups")]
    pub max_state_backups: u32,
    /// Порт для WebSocket мониторинга (задача 221)
    #[serde(default = "default_monitoring_port")]
    pub monitoring_port: u16,
    /// Минимальный интервал между обновлениями equity в миллисекундах (задача 221)
    #[serde(default = "default_min_update_ms")]
    pub min_update_ms: u64,
    /// Интервал синхронизации баланса с биржей в секундах (задача 221)
    #[serde(default = "default_balance_sync_interval")]
    pub balance_sync_interval: u64,
    /// Переопределение chat_id для отправки алертов (задача 222)
    pub override_chat_id: Option<String>,
    /// TTL дедупликации алертов в секундах (задача 222)
    #[serde(default = "default_alert_dedup_ttl_secs")]
    pub alert_dedup_ttl_secs: u64,
    /// Частота сэмплирования для записи полных вероятностей (1 из N) (задача 224)
    #[serde(default = "default_confidence_sample_rate")]
    pub confidence_sample_rate: u32,
    /// Порог энтропии для генерации алерта о дрейфе модели (задача 224)
    #[serde(default = "default_entropy_drift_threshold")]
    pub entropy_drift_threshold: f32,
    /// Включить проверку дрейфа модели в реальном времени (задача 224)
    #[serde(default = "default_enable_realtime_drift_check")]
    pub enable_realtime_drift_check: bool,
    /// Полностью остановить торговлю при обнаружении дрейфа (задача 224)
    #[serde(default = "default_drift_stop_enabled")]
    pub drift_stop_enabled: bool,
    /// Коэффициент сокращения размера позиции при дрейфе (задача 224)
    #[serde(default = "default_drift_scale_factor")]
    pub drift_scale_factor: f32,
    /// Включить автоматическую перезагрузку моделей (hot-swap) (задача 228)
    #[serde(default = "default_enable_model_hotswap")]
    pub enable_model_hotswap: bool,
    /// Конфигурация мониторинга системных ресурсов (задача 225)
    #[serde(default)]
    pub resource_thresholds: ResourceThresholdsConfig,
    /// Конфигурация изоляции ресурсов процесса (задача 230)
    #[serde(default)]
    pub system: SystemConfig,
    /// Тип предотвращения самосделок (SMP) на стороне биржи (задача 232)
    /// "None", "CancelMaker", "CancelTaker", "CancelBoth"
    #[serde(default = "default_smp_type")]
    pub smp_type: String,
    /// Включить локальную проверку противоположных ордеров перед выставлением (задача 232)
    #[serde(default = "default_local_smp_enabled")]
    pub local_smp_enabled: bool,
    /// Порог включения превентивного замедления в процентах (задача 234)
    #[serde(default = "default_rate_limit_threshold_pct")]
    pub rate_limit_threshold_pct: f64,
    /// Базовая задержка для exponential backoff в миллисекундах (задача 234)
    #[serde(default = "default_backoff_base_ms")]
    pub backoff_base_ms: u64,
    /// Интервал запуска процедуры очистки зависших ордеров в минутах (задача 235)
    #[serde(default = "default_cleanup_interval_min")]
    pub cleanup_interval_min: u64,
    /// Максимальный возраст ордера в минутах для автоматической отмены (задача 235)
    #[serde(default = "default_max_stale_age_min")]
    pub max_stale_age_min: u64,
    /// Разрешить ли автоматическую отмену старых, но известных ордеров (задача 235)
    #[serde(default = "default_auto_cancel_stale")]
    pub auto_cancel_stale: bool,
}

impl Default for BotConfig {
    fn default() -> Self {
        Self {
            symbol: "UNKNOWN".to_string(),
            model_path: std::path::PathBuf::from("models/default.onnx"),
            seq_len: default_seq_len(),
            features_dim: default_features_dim(),
            past_returns_lags: default_past_returns_lags(),
            initial_balance: default_initial_balance(),
            close_on_flat: default_close_on_flat(),
            threshold_buy: default_threshold_buy(),
            threshold_sell: default_threshold_sell(),
            threshold_flat: default_threshold_flat(),
            long_threshold: default_long_threshold(),
            short_threshold: default_short_threshold(),
            exit_threshold: None,
            position_idx: 0,
            position_sync_interval_secs: default_position_sync_interval(),
            buffer_pct: default_buffer_pct(),
            leverage: default_leverage(),
            taker_fee_bps: default_taker_fee_bps(),
            post_only: default_post_only(),
            post_only_retry_limit: default_post_only_retry_limit(),
            limit_timeout_ms: default_limit_timeout_ms(),
            max_position_size: None,
            max_notional_usd: Decimal::ZERO,
            max_drawdown_pct: None,
            risk: RiskConfig::default(),
            onnx: OnnxConfig::default(),
            fusion: FusionConfig::default(),
            max_spread_static_bps: None,
            spread_multiplier: None,
            volatility_target_bps: default_volatility_target_bps(),
            volatility_window: default_volatility_window(),
            volatility_default: default_volatility_default(),
            size_min_multiplier: default_size_min_multiplier(),
            size_max_multiplier: default_size_max_multiplier(),
            stats_window_ms: default_stats_window_ms(),
            stats_max_trades: default_stats_max_trades(),
            desync_tolerance_pct: default_desync_tolerance(),
            signal_min_confidence: 0.45,
            signal_full_confidence: 0.85,
            signal_size_mult_min: 0.5,
            signal_size_mult_max: 2.0,
            total_size_mult_max: 3.0,
            chase_mode: default_chase_mode(),
            chase_threshold_bps: default_chase_threshold(),
            chase_distance_bps: default_chase_distance(),
            chase_max_attempts: default_chase_max_attempts(),
            chase_interval_ms: default_chase_interval(),
            use_vwap_filter: default_vwap_filter(),
            vwap_filter_threshold_bps: default_vwap_filter_threshold(),
            close_on_exit: default_close_on_exit(),
            emergency_timeout_ms: default_emergency_timeout(),
            max_daily_loss_pct: Decimal::from_f64(2.0).unwrap(),
            daily_reset_hour_utc: 0,
            max_trades_limit: 0,
            max_trades_window_sec: 3600,
            max_inactivity_ms: default_max_inactivity_ms(),
            close_position_on_inactivity: default_close_on_inactivity(),
            threshold_base: default_threshold_base(),
            threshold_loss_mult: default_threshold_loss_mult(),
            threshold_max: default_threshold_max(),
            threshold_min: default_threshold_min(),
            threshold_max_streak: default_threshold_max_streak(),
            max_network_latency_micros: default_max_network_latency_micros(),
            max_inference_latency_micros: default_max_inference_latency_micros(),
            max_total_latency_micros: default_max_total_latency_micros(),
            max_latency_rejects_limit: default_max_latency_rejects_limit(),
            obi_threshold: default_obi_threshold(),
            obi_depth: default_obi_depth(),
            lockout_period_sec: default_lockout_period_sec(),
            lockout_streak_threshold: default_lockout_streak_threshold(),
            stop_file_name: default_stop_file_name(),
            ack_extension: default_ack_extension(),
            global_stop_enabled: default_global_stop_enabled(),
            stop_check_interval_ms: default_stop_check_interval_ms(),
            reconciliation_interval_sec: default_reconciliation_interval_sec(),
            sync_on_desync: default_sync_on_desync(),
            price_desync_threshold: default_price_desync_threshold(),
            snapshot_interval_ms: default_snapshot_interval(),
            latency_report_interval_sec: default_latency_report_interval(),
            min_flip_interval_ms: default_min_flip_interval_ms(),
            max_slice_size: default_max_slice_size(),
            regime_overrides: default_regime_overrides(),
            liquidity_filter: None,
            time_decay: TimeDecayConfig::default(),
            force_taker_confidence: default_force_taker_confidence(),
            maker_offset_step_ticks: default_maker_offset_step_ticks(),
            max_post_only_rejects: default_max_post_only_rejects(),
            repeg_threshold_ticks: default_repeg_threshold_ticks(),
            rebate_wait_timeout_ms: default_rebate_wait_timeout_ms(),
            adversarial: AdversarialConfig::default(),
            tp_stages: default_tp_stages(),
            tp_close_all_on_min_qty: default_tp_close_all_on_min_qty(),
            trailing_stop: TrailingStopConfig {
                tsl_mode: default_tsl_mode(),
                tsl_activation_bps: default_tsl_activation_bps(),
                tsl_distance_bps: default_tsl_distance_bps(),
                tsl_step_bps: default_tsl_step_bps(),
            },
            entry_style: default_entry_style(),
            max_entry_slippage_bps: default_max_entry_slippage_bps(),
            entry_participation_ratio: default_entry_participation_ratio(),
            slicing_enabled: default_slicing_enabled(),
            max_signal_age_ms: default_max_signal_age_ms(),
            max_clock_skew_ms: default_max_clock_skew_ms(),
            staleness_action: default_staleness_action(),
            funding_filter: FundingFilterConfig::default(),
            max_state_age_ms: default_max_state_age_ms(),
            enable_fill_rate_logging: default_enable_fill_rate_logging(),
            enable_impact_logging: default_enable_impact_logging(),
            sor: SorConfig::default(),
            adaptive_thresholds_enabled: default_adaptive_thresholds_enabled(),
            base_threshold_bps: default_base_threshold_bps(),
            vol_multiplier: default_vol_multiplier(),
            spread_multiplier_adaptive: default_spread_multiplier_adaptive(),
            min_threshold_bps: default_min_threshold_bps(),
            max_threshold_bps: default_max_threshold_bps(),
            persistence_interval_sec: default_persistence_interval_sec(),
            max_state_backups: default_max_state_backups(),
            monitoring_port: default_monitoring_port(),
            min_update_ms: default_min_update_ms(),
            balance_sync_interval: default_balance_sync_interval(),
            override_chat_id: None,
            alert_dedup_ttl_secs: default_alert_dedup_ttl_secs(),
            confidence_sample_rate: default_confidence_sample_rate(),
            entropy_drift_threshold: default_entropy_drift_threshold(),
            enable_realtime_drift_check: default_enable_realtime_drift_check(),
            drift_stop_enabled: default_drift_stop_enabled(),
            drift_scale_factor: default_drift_scale_factor(),
            enable_model_hotswap: default_enable_model_hotswap(),
            resource_thresholds: ResourceThresholdsConfig::default(),
            system: SystemConfig::default(),
            smp_type: default_smp_type(),
            local_smp_enabled: default_local_smp_enabled(),
            rate_limit_threshold_pct: default_rate_limit_threshold_pct(),
            backoff_base_ms: default_backoff_base_ms(),
            cleanup_interval_min: default_cleanup_interval_min(),
            max_stale_age_min: default_max_stale_age_min(),
            auto_cancel_stale: default_auto_cancel_stale(),
        }
    }
}

// --- Итоговая конфигурация ---

#[derive(Debug, Clone)]
pub struct FullConfig {
    pub symbol: String,
    pub general: GeneralConfig,
    pub logging: LoggingConfig,
    pub exchange: ExchangeConfig,
    pub trading: TradingDefaultsConfig,
    pub risk: RiskDefaultsConfig,
    pub bot: BotConfig,
    pub monitoring: crate::monitoring::health::HealthConfig,
    pub global: GlobalConfig,
}
