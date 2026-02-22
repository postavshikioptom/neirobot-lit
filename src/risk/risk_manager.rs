use crate::data::orderbook::{OrderBook, OrderBookSnapshot};
use crate::data::types::Side;
use anyhow::{Result, bail};
use rust_decimal::Decimal;
use rust_decimal::prelude::{ToPrimitive, FromPrimitive};
use chrono::{Utc, NaiveDate, DateTime, Timelike};
use crate::trading::rest_client::BybitRestClientTrait;
use std::collections::{VecDeque, HashMap};
use std::time::Instant;
use std::path::{Path, PathBuf};
use crate::risk::health_monitor::HealthMonitor;
use crate::risk::pnl_stats::PnlStats;
use crate::config::types::{BotConfig, PriceReferenceSource, RiskConfig};
use crate::trading::types::OrderSide;
use crate::trading::position_manager::Position;
use crate::trading::RiskState;
use crate::trading::PositionManager;

use tracing::{info, warn, debug, error};

/// Трейт для статической диспетчеризации риск-гейтов (Задача №198)
pub trait RiskGate {
    fn check(
        &self, 
        config: &RiskConfig,
        manager: &RiskManager, 
        intent: &OrderIntent,
        current_pos: &Position,
        mid_price: Decimal,
        ob: &OrderBookSnapshot
    ) -> RiskResult;
}

#[derive(Debug, Clone, Copy)]
pub struct MaxPositionGate;
impl RiskGate for MaxPositionGate {
    #[inline(always)]
    fn check(&self, config: &RiskConfig, manager: &RiskManager, intent: &OrderIntent, current_pos: &Position, _mid_price: Decimal, _ob: &OrderBookSnapshot) -> RiskResult {
        let order_qty = Decimal::from_f64(intent.qty).unwrap_or_default();
        let signed_qty = match intent.side {
            OrderSide::Buy => order_qty,
            OrderSide::Sell => -order_qty,
        };
        let projected_qty = current_pos.qty + signed_qty;

        if projected_qty.abs() < current_pos.qty.abs() {
            return RiskResult::Allow;
        }

        if let Some(max_size_dec) = config.max_position_size {
            let effective_limit = manager.get_effective_position_limit(max_size_dec);
            if projected_qty.abs() > effective_limit {
                return RiskResult::Reject(format!(
                    "MaxPosition exceeded: {} > {}", 
                    projected_qty.abs(), 
                    effective_limit
                ));
            }
        }
        RiskResult::Allow
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PriceDeviationGate;
impl RiskGate for PriceDeviationGate {
    #[inline(always)]
    fn check(&self, config: &RiskConfig, _manager: &RiskManager, intent: &OrderIntent, _current_pos: &Position, mid_price: Decimal, _ob: &OrderBookSnapshot) -> RiskResult {
        let order_price = Decimal::from_f64(intent.price).unwrap_or_default();
        let limit = match config.max_price_deviation_pct {
            Some(l) => l,
            None => return RiskResult::Allow,
        };

        if mid_price <= Decimal::ZERO {
            return RiskResult::Reject("Invalid mid_price".to_string());
        }

        let deviation = (order_price - mid_price).abs() / mid_price;
        if deviation > limit {
            return RiskResult::Reject(format!("Price deviation too high: {:.2}%", deviation * Decimal::from(100)));
        }
        RiskResult::Allow
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SlippageCheckGate;
impl RiskGate for SlippageCheckGate {
    #[inline(always)]
    fn check(&self, config: &RiskConfig, manager: &RiskManager, intent: &OrderIntent, _current_pos: &Position, _mid_price: Decimal, ob: &OrderBookSnapshot) -> RiskResult {
        let side = match intent.side {
            OrderSide::Buy => Side::Buy,
            OrderSide::Sell => Side::Sell,
        };
        let size_usd = intent.price * intent.qty;
        manager.check_liquidity_filter_internal_snapshot(config, ob, side, size_usd)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NotionalLimitGate;
impl RiskGate for NotionalLimitGate {
    #[inline(always)]
    fn check(&self, config: &RiskConfig, _manager: &RiskManager, intent: &OrderIntent, current_pos: &Position, mid_price: Decimal, _ob: &OrderBookSnapshot) -> RiskResult {
        let order_price = Decimal::from_f64(intent.price).unwrap_or_default();
        let order_qty = Decimal::from_f64(intent.qty).unwrap_or_default();
        
        // 1. Min Notional (Bybit default $5)
        let min_notional = Decimal::from_f64(5.0).unwrap_or(Decimal::from(5));
        if order_qty * order_price < min_notional {
             return RiskResult::Reject(format!("Order value too small: {} < {}", order_qty * order_price, min_notional));
        }

        // 2. Max Notional
        if let Some(max_notional_dec) = config.max_notional_usd {
            let signed_qty = match intent.side {
                OrderSide::Buy => order_qty,
                OrderSide::Sell => -order_qty,
            };
            let projected_qty = current_pos.qty + signed_qty;
            let projected_notional = projected_qty.abs() * mid_price;
            
            if projected_notional > max_notional_dec {
                return RiskResult::Reject(format!("MaxNotionalExceeded: {} > {}", projected_notional, max_notional_dec));
            }
            
            // 3. Max Margin Check (Задача 071)
            if let Some(max_margin_dec) = config.max_margin_usd {
                let projected_margin = projected_notional / config.leverage;
                if projected_margin > max_margin_dec {
                    return RiskResult::Reject(format!("MaxMarginExceeded: {} > {}", projected_margin, max_margin_dec));
                }
            }
        }
        
        RiskResult::Allow
    }
}

/// Задача 233: Gate для проверки отклонения цены от Mark Price
#[derive(Debug, Clone, Copy)]
pub struct PriceBandViolationGate;
impl RiskGate for PriceBandViolationGate {
    #[inline(always)]
    fn check(&self, config: &RiskConfig, _manager: &RiskManager, intent: &OrderIntent, _current_pos: &Position, _mid_price: Decimal, ob: &OrderBookSnapshot) -> RiskResult {
        let order_price = intent.price;
        let mark_price = ob.mark_price;
        
        // Если mark_price не установлена, пропускаем проверку
        if mark_price <= 0.0 {
            return RiskResult::Allow;
        }
        
        // Расчет отклонения цены ордера от Mark Price
        let mark_dev = (order_price - mark_price).abs() / mark_price;
        
        // Проверка превышения лимита отклонения
        if mark_dev > config.max_mark_deviation {
            return RiskResult::Reject(format!(
                "Price deviation from Mark Price too high: {:.4}% > {:.4}%",
                mark_dev * 100.0, config.max_mark_deviation * 100.0
            ));
        }
        
        RiskResult::Allow
    }
}

#[derive(Debug, Clone, Copy)]
pub enum RiskGateKind {
    MaxPosition(MaxPositionGate),
    PriceDeviation(PriceDeviationGate),
    SlippageCheck(SlippageCheckGate),
    NotionalLimit(NotionalLimitGate),
    PriceBandViolation(PriceBandViolationGate), // Задача 233
}

impl RiskGate for RiskGateKind {
    #[inline(always)]
    fn check(&self, config: &RiskConfig, manager: &RiskManager, intent: &OrderIntent, current_pos: &Position, mid_price: Decimal, ob: &OrderBookSnapshot) -> RiskResult {
        match self {
            RiskGateKind::MaxPosition(g) => g.check(config, manager, intent, current_pos, mid_price, ob),
            RiskGateKind::PriceDeviation(g) => g.check(config, manager, intent, current_pos, mid_price, ob),
            RiskGateKind::SlippageCheck(g) => g.check(config, manager, intent, current_pos, mid_price, ob),
            RiskGateKind::NotionalLimit(g) => g.check(config, manager, intent, current_pos, mid_price, ob),
            RiskGateKind::PriceBandViolation(g) => g.check(config, manager, intent, current_pos, mid_price, ob),
        }
    }
}

/// Намерение разместить ордер (Задача 176: Duplicate Order Prevention)
/// Используется для fuzzy matching и предотвращения дублирования ордеров
#[derive(Debug, Clone)]
pub struct OrderIntent {
    pub side: OrderSide,
    pub price: f64,
    pub qty: f64,
    pub timestamp: u64, // timestamp в миллисекундах (created_at)
    pub filled_qty: f64, // Задача 179: Исполненный объем для контроля времени жизни
}

/// Результат проверки риска для ордера
#[derive(Debug, Clone)]
pub enum RiskResult {
    /// Ордер разрешен с исходным размером
    Allow,
    /// Ордер разрешен, но размер должен быть скорректирован (новый размер в USD)
    AdjustSize(f64),
    /// Ордер отклонен (причина)
    Reject(String),
}

pub struct RiskManager {
    config: RiskConfig,
    initial_equity: Decimal,
    peak_equity: Decimal,
    daily_start_equity: Decimal,
    peak_daily_equity: Decimal,
    last_reset_date: Option<NaiveDate>,
    pub is_blocked: bool,
    trade_history: VecDeque<i64>,
    pub consecutive_latency_rejects: usize,
    // Задача 164: Отслеживание Maker/Taker сделок
    maker_fills_count: usize,
    taker_fills_count: usize,
    // Задача 169: Мониторинг устаревших сигналов
    stale_signal_history: VecDeque<(i64, bool)>, // (timestamp_ms, is_stale)
    stale_signal_window_ms: i64, // Окно для расчета (5 минут)
    // Задача 171: Система проверки здоровья (System Health Sanity Check)
    pub health_monitor: HealthMonitor,
    // Задача 173: Гейт аномальной волатильности PnL
    pub pnl_stats: PnlStats,
    last_pnl_bps: f64,
    // Задача 175: Лимиты отклонения ордеров
    consecutive_rejections: u32,
    rejection_history: VecDeque<u64>,
    // Задача 176: Защита от дублирования ордеров
    active_intents: HashMap<String, OrderIntent>, // ключ - order_link_id
    // Задача 178: Динамическое сокращение лимитов позиции
    current_scale: f64, // Текущий масштаб лимита позиции (1.0 = 100%, 0.2 = 20%)
    // Задача №198: Статическая диспетчеризация риск-гейтов
    pub gates: Vec<RiskGateKind>,
    // Задача 217: Аудит чувствительных действий
    pub audit_logger: Option<crate::utils::AuditLogger>,
    // Задача 222: Отправка алертов
    pub alert_manager: Option<std::sync::Arc<crate::monitoring::alert_manager::AlertManager>>,
    
    // Задача 231: Обработка ошибок при нехватке маржи
    margin_multiplier: f64, // Множитель для снижения размера позиции (1.0 = 100%, 0.5 = 50%)
    last_margin_error_ts: Option<Instant>, // Время последней ошибки маржи
    
    // Задача 233: Флаг ценового шока (Price Band Violation)
    pub is_price_shock: bool, // true если обнаружена ошибка 110010
}

impl RiskManager {
    pub fn new(config: RiskConfig, initial_equity: Decimal) -> Self {
        let today = Utc::now().date_naive();
        Self { 
            config: config.clone(), 
            peak_equity: initial_equity,
            initial_equity,
            daily_start_equity: initial_equity,
            peak_daily_equity: initial_equity,
            last_reset_date: Some(today),
            is_blocked: false,
            trade_history: VecDeque::new(),
            consecutive_latency_rejects: 0,
            maker_fills_count: 0,
            taker_fills_count: 0,
            stale_signal_history: VecDeque::new(),
            stale_signal_window_ms: 300_000, // 5 минут
            health_monitor: HealthMonitor::new(config.clone()),
            pnl_stats: PnlStats::new(),
            last_pnl_bps: 0.0,
            consecutive_rejections: 0,
            rejection_history: VecDeque::new(),
            active_intents: HashMap::new(),
            current_scale: 1.0, // Начинаем с полного лимита
            gates: vec![
                RiskGateKind::MaxPosition(MaxPositionGate),
                RiskGateKind::PriceDeviation(PriceDeviationGate),
                RiskGateKind::SlippageCheck(SlippageCheckGate),
                RiskGateKind::NotionalLimit(NotionalLimitGate),
                RiskGateKind::PriceBandViolation(PriceBandViolationGate), // Задача 233
            ],
            audit_logger: None,
            alert_manager: None,
            margin_multiplier: 1.0, // Начинаем с полного размера позиции
            last_margin_error_ts: None,
            is_price_shock: false, // Задача 233: Инициализация флага ценового шока
        }
    }

    /// Устанавливает AuditLogger для логирования событий (Задача 217)
    pub fn set_audit_logger(&mut self, logger: crate::utils::AuditLogger) {
        self.audit_logger = Some(logger);
    }

    /// Устанавливает AlertManager для отправки алертов (Задача 222)
    pub fn set_alert_manager(&mut self, manager: std::sync::Arc<crate::monitoring::alert_manager::AlertManager>) {
        self.alert_manager = Some(manager);
    }

    /// Устанавливает флаг ценового шока (Задача 233)
    pub fn set_price_shock(&mut self, is_shock: bool) {
        self.is_price_shock = is_shock;
        if is_shock {
            error!("Price shock flag set to true. Trading will be suspended.");
        } else {
            info!("Price shock flag cleared. Market has stabilized.");
        }
    }

    /// Проверяет системные ресурсы и снижает риски при перегрузке (Задача 225)
    pub fn check_system_resources(&mut self, metrics: &crate::monitoring::resource_profiler::SystemMetricsUpdate, bot_config: &crate::config::types::BotConfig) {
        // Проверяем превышение CPU
        if metrics.cpu_usage_pct > bot_config.resource_thresholds.cpu_max_pct {
            warn!(
                "CPU usage exceeded threshold: {:.1}% > {}%, entering Degraded mode",
                metrics.cpu_usage_pct,
                bot_config.resource_thresholds.cpu_max_pct
            );

            // Принудительно переводим в Degraded mode через снижение current_scale
            self.current_scale = 0.5; // Снижаем лимиты позиций на 50%

            // Отправляем алерт
            if let Some(ref alert_mgr) = self.alert_manager {
                let msg = format!(
                    "⚠️ HIGH CPU USAGE: {:.1}% (threshold: {}%)\nPosition limits reduced to 50%",
                    metrics.cpu_usage_pct,
                    bot_config.resource_thresholds.cpu_max_pct
                );
                if let Err(e) = alert_mgr.send_alert(
                    crate::monitoring::alert_manager::AlertLevel::Warning,
                    "System Resources",
                    &msg,
                ) {
                    error!("Failed to send CPU alert: {}", e);
                }
            }
        } else if self.current_scale < 1.0 && metrics.cpu_usage_pct < bot_config.resource_thresholds.cpu_max_pct * 0.8 {
            // Восстанавливаем лимиты, если CPU упал ниже 80% от порога
            info!("CPU usage normalized, restoring position limits");
            self.current_scale = 1.0;
        }

        // Проверяем утечку памяти
        if metrics.memory_leak_detected {
            error!("Memory leak detected: consistent growth detected");

            // Отправляем Warning алерт (не Critical)
            if let Some(ref alert_mgr) = self.alert_manager {
                let msg = format!(
                    "⚠️ MEMORY LEAK DETECTED\nCurrent RSS: {} MB\nConsistent growth over {} samples",
                    metrics.memory_rss_kb / 1024,
                    bot_config.resource_thresholds.leak_detection_window
                );
                if let Err(e) = alert_mgr.send_alert(
                    crate::monitoring::alert_manager::AlertLevel::Warning,
                    "Memory Leak",
                    &msg,
                ) {
                    error!("Failed to send memory leak alert: {}", e);
                }
            }
        }
    }

    /// Основная проверка рисков через статическую диспетчеризацию (Задача №198)
    #[inline(always)]
    pub fn check_risk(&self, intent: &OrderIntent, current_pos: &Position, ob: &OrderBookSnapshot) -> RiskResult {
        let mid_price = ob.get_mid_price_dec();
        for gate in &self.gates {
            match gate.check(&self.config, self, intent, current_pos, mid_price, ob) {
                RiskResult::Reject(reason) => return RiskResult::Reject(reason),
                RiskResult::AdjustSize(new_size) => return RiskResult::AdjustSize(new_size),
                RiskResult::Allow => {}
            }
        }
        RiskResult::Allow
    }

    /// Внутренний метод для проверки ликвидности через снапшот (используется SlippageCheckGate)
    #[inline(always)]
    pub fn check_liquidity_filter_internal_snapshot(
        &self,
        config: &RiskConfig,
        ob: &OrderBookSnapshot,
        side: Side,
        size_usd: f64,
    ) -> RiskResult {
        // Если фильтр не настроен, пропускаем проверку
        let filter_config = match &config.liquidity_filter {
            Some(cfg) => cfg,
            None => return RiskResult::Allow,
        };

        if size_usd <= 0.0 {
            return RiskResult::Reject("Order size must be positive".to_string());
        }

        // Проверка 1: Объем на лучшем уровне
        let volume_at_best = ob.get_volume_at_best(side);
        let min_required_volume = size_usd * filter_config.min_top_multiple;

        if volume_at_best < min_required_volume {
            let msg = format!(
                "[Risk] Insufficient liquidity at best level: ${:.2} < ${:.2} (required {}x of ${:.2})",
                volume_at_best, min_required_volume, filter_config.min_top_multiple, size_usd
            );
            
            if filter_config.adjust_size_if_thin {
                let adjusted_size = volume_at_best / filter_config.min_top_multiple;
                if adjusted_size > 0.0 {
                    return RiskResult::AdjustSize(adjusted_size);
                } else {
                    return RiskResult::Reject(msg);
                }
            } else {
                return RiskResult::Reject(msg);
            }
        }

        // Проверка 2: Проскальзывание (VWAP impact)
        let impact_bps = ob.calculate_vwap_impact(side, size_usd);
        
        if impact_bps > filter_config.max_impact_bps {
            let msg = format!(
                "[Risk] Slippage impact {:.2}bps > limit {:.2}bps",
                impact_bps, filter_config.max_impact_bps
            );
            
            if filter_config.adjust_size_if_thin {
                // Бинарный поиск максимального размера, который проходит по фильтру
                let mut low = 0.0;
                let mut high = size_usd;
                let mut best_size = 0.0;
                
                // Максимум 10 итераций бинарного поиска
                for _ in 0..10 {
                    let mid = (low + high) / 2.0;
                    if mid <= 0.0 {
                        break;
                    }
                    
                    let test_impact = ob.calculate_vwap_impact(side, mid);
                    if test_impact <= filter_config.max_impact_bps {
                        best_size = mid;
                        low = mid;
                    } else {
                        high = mid;
                    }
                    
                    // Если разница меньше $1, останавливаемся
                    if (high - low).abs() < 1.0 {
                        break;
                    }
                }
                
                if best_size > 0.0 {
                    return RiskResult::AdjustSize(best_size);
                }
            }
            return RiskResult::Reject(msg);
        }

        RiskResult::Allow
    }

    /// Обновление текущего эквити и пиковых значений (HWM)
    #[inline(always)]
    pub fn update_equity(&mut self, current_pnl: Decimal) {
        // Проверка полуночного сброса перед обновлением
        self.check_midnight_reset(current_pnl);

        let current_equity = self.initial_equity + current_pnl;
        
        // Глобальный пик
        if current_equity > self.peak_equity {
            self.peak_equity = current_equity;
        }

        // Дневной пик
        if current_equity > self.peak_daily_equity {
            self.peak_daily_equity = current_equity;
        }
    }

    /// Сброс дневных показателей (вызывать раз в сутки или автоматически)
    pub fn reset_daily(&mut self, current_pnl: Decimal) {
        let today = Utc::now().date_naive();
        self.daily_start_equity = self.initial_equity + current_pnl;
        self.peak_daily_equity = self.initial_equity + current_pnl;
        self.last_reset_date = Some(today);
        self.is_blocked = false;
        // Задача 164: Сброс счетчиков Maker/Taker
        self.reset_fill_counters();
        info!("Daily reset: Maker/Taker counters cleared");
        info!("RiskManager: Daily metrics reset. New peak: {}", self.peak_daily_equity);
    }

    fn check_midnight_reset(&mut self, current_pnl: Decimal) {
        if !self.config.auto_reset_at_midnight {
            return;
        }

        let today = Utc::now().date_naive();
        if let Some(last_date) = self.last_reset_date {
            if today > last_date {
                info!("Midnight reached (UTC). Resetting daily risk limits.");
                self.reset_daily(current_pnl);
            }
        } else {
            self.last_reset_date = Some(today);
        }
    }

    /// Обновление статистики PnL (Задача 173)
    pub fn update_pnl_stats(&mut self, pnl_bps: f64) {
        self.last_pnl_bps = pnl_bps;
        let window = self.config.pnl_volatility_window;
        self.pnl_stats.update(pnl_bps, window);
        
        debug!("RiskManager: PnL Stats updated. Last: {} bps, StdDev: {:.2} bps (n={})", 
            pnl_bps, self.pnl_stats.std_dev(), self.pnl_stats.n);
    }

    /// Задача 175: Регистрация отклоненного ордера
    pub fn report_rejection(&mut self) {
        let now = Utc::now().timestamp_millis() as u64;
        self.handle_rejection(now);
    }

    /// Вспомогательный метод для обработки реджектов (выделен для тестирования)
    fn handle_rejection(&mut self, now: u64) {
        self.rejection_history.push_back(now);
        
        // Lazy Cleanup: Удаление устаревших меток
        let window_start = now.saturating_sub(self.config.rejection_window_ms);
        while let Some(&ts) = self.rejection_history.front() {
            if ts < window_start {
                self.rejection_history.pop_front();
            } else {
                break;
            }
        }
        
        self.consecutive_rejections += 1;
        warn!("RiskManager: Rejection reported. Consecutive: {}, Window count: {}", 
            self.consecutive_rejections, self.rejection_history.len());
    }

    /// Helper для тестов, чтобы симулировать время
    #[cfg(test)]
    pub fn report_rejection_for_test(&mut self, timestamp: u64) {
        self.handle_rejection(timestamp);
    }

    /// Задача 175: Регистрация успешного ордера
    pub fn report_success(&mut self) {
        if self.consecutive_rejections > 0 {
            debug!("RiskManager: Success reported. Resetting consecutive rejections (was {}).", 
                self.consecutive_rejections);
            self.consecutive_rejections = 0;
        }
    }

    /// Глобальная проверка здоровья и рисков (Задача 171)
    #[inline(always)]
    pub fn check_risk_gates(&mut self, current_pnl: Decimal) -> Result<()> {
        // Задача 233: Проверка флага ценового шока
        if self.is_price_shock {
            self.is_blocked = true;
            error!("CRITICAL_RISK_STOP: Price shock detected. Trading suspended until market stabilizes.");
            // Логируем в аудит (Задача 217)
            if let Some(logger) = &self.audit_logger {
                let _ = logger.log_risk_gate("PRICE_SHOCK", true, "Market in shock mode");
            }
            // Отправляем алерт в Telegram (Задача 222)
            if let Some(am) = &self.alert_manager {
                am.send_alert(crate::monitoring::alert_manager::Alert::new(
                    crate::monitoring::alert_manager::AlertLevel::Critical,
                    "CRITICAL_RISK_STOP: Price shock detected. Trading suspended.",
                    "RiskManager".to_string(),
                ));
            }
            bail!("Risk: Price shock detected. Trading suspended.");
        }

        // 1. Проверка здоровья системы с очисткой интентов (Задача 176)
        if let Err(e) = self.health_monitor.is_sane_with_intent_cleanup(&mut self.active_intents) {
            self.is_blocked = true;
            error!("CRITICAL_HEALTH_STOP: System health sanity check failed: {}", e);
            // Логируем в аудит (Задача 217)
            if let Some(logger) = &self.audit_logger {
                let _ = logger.log_risk_gate("HEALTH_CHECK", true, &e.to_string());
            }
            // Отправляем алерт в Telegram (Задача 222)
            if let Some(am) = &self.alert_manager {
                am.send_alert(crate::monitoring::alert_manager::Alert::new(
                    crate::monitoring::alert_manager::AlertLevel::Critical,
                    format!("CRITICAL_HEALTH_STOP: {}", e),
                    "RiskManager".to_string(),
                ));
            }
            bail!("Risk: System health sanity check failed: {}", e);
        }

        // 2. Глобальные проверки рисков (DD)
        self.check_global_risk(current_pnl)?;

        // 3. Проверка аномальной волатильности PnL (Задача 173)
        if self.pnl_stats.n >= 2 {
            let std_dev = self.pnl_stats.std_dev();
            if std_dev > self.config.max_pnl_std_dev_bps as f64 {
                self.is_blocked = true;
                error!("CRITICAL_RISK_STOP: Unusual PnL volatility: {:.2} bps > {} bps", 
                    std_dev, self.config.max_pnl_std_dev_bps);
                // Логируем в аудит (Задача 217)
                if let Some(logger) = &self.audit_logger {
                    let _ = logger.log_risk_gate("PNL_VOLATILITY", true, 
                        &format!("{:.2} bps > {} bps", std_dev, self.config.max_pnl_std_dev_bps));
                }
                bail!("Risk: Unusual PnL volatility detected");
            }
            
            if self.pnl_stats.is_outlier(self.last_pnl_bps, self.config.max_pnl_z_score_threshold) {
                self.is_blocked = true;
                error!("CRITICAL_RISK_STOP: PnL Outlier detected: last {} bps, Z-score > {}", 
                    self.last_pnl_bps, self.config.max_pnl_z_score_threshold);
                // Логируем в аудит (Задача 217)
                if let Some(logger) = &self.audit_logger {
                    let _ = logger.log_risk_gate("PNL_OUTLIER", true, 
                        &format!("{} bps, Z-score > {}", self.last_pnl_bps, self.config.max_pnl_z_score_threshold));
                }
                bail!("Risk: PnL Outlier detected");
            }
        }

        // 4. Проверка лимитов отклонения ордеров (Задача 175)
        if self.consecutive_rejections >= self.config.max_consecutive_rejections {
            self.is_blocked = true;
            error!("CRITICAL_RISK_STOP: Too many consecutive rejections: {} >= {}", 
                self.consecutive_rejections, self.config.max_consecutive_rejections);
            // Логируем в аудит (Задача 217)
            if let Some(logger) = &self.audit_logger {
                let _ = logger.log_risk_gate("CONSECUTIVE_REJECTIONS", true, 
                    &format!("{} >= {}", self.consecutive_rejections, self.config.max_consecutive_rejections));
            }
            bail!("Risk: Too many consecutive rejections");
        }

        let window_rejections = self.rejection_history.len() as u32;
        if window_rejections >= self.config.max_total_rejections_in_window {
            self.is_blocked = true;
            error!("CRITICAL_RISK_STOP: Too many rejections in window: {} >= {}", 
                window_rejections, self.config.max_total_rejections_in_window);
            // Логируем в аудит (Задача 217)
            if let Some(logger) = &self.audit_logger {
                let _ = logger.log_risk_gate("REJECTIONS_WINDOW", true, 
                    &format!("{} >= {}", window_rejections, self.config.max_total_rejections_in_window));
            }
            bail!("Risk: Too many rejections in window");
        }

        Ok(())
    }

    /// Глобальная проверка: можно ли боту открывать новые позиции?
    pub fn check_global_risk(&mut self, current_pnl: Decimal) -> Result<()> {
        // 0. Автосброс если нужно
        self.check_midnight_reset(current_pnl);

        if self.is_blocked {
            bail!("Risk: Trading is blocked due to HardStop (Drawdown limit reached)");
        }

        // Задача 169: Проверка circuit breaker для устаревших сигналов
        if self.check_stale_signal_circuit_breaker() {
            self.is_blocked = true;
            error!("CRITICAL_RISK_STOP: Stale signal circuit breaker triggered (>50% stale signals in last 5min)");
            bail!("Risk: Stale signal circuit breaker triggered");
        }

        let current_equity = self.initial_equity + current_pnl;

        // 1. Проверка Max Drawdown от глобального пика (Cumulative)
        if let Some(max_dd_pct_dec) = self.config.max_drawdown_pct {
            if self.peak_equity > Decimal::ZERO {
                let dd_pct = (self.peak_equity - current_equity) / self.peak_equity;
                if dd_pct > max_dd_pct_dec {
                    self.is_blocked = true;
                    error!("CRITICAL_RISK_STOP: Cumulative drawdown exceeded: {}% > {}%", 
                        dd_pct * Decimal::from(100), max_dd_pct_dec * Decimal::from(100));
                    bail!("Risk: Cumulative drawdown exceeded");
                }
            }
        }

        // 2. Проверка дневной просадки (Task 072)
        self.check_drawdown(current_pnl)?;

        Ok(())
    }

    /// Проверка дневной просадки от HWM
    pub fn check_drawdown(&mut self, current_pnl: Decimal) -> Result<()> {
        let current_equity = self.initial_equity + current_pnl;
        let drawdown_usd = self.peak_daily_equity - current_equity;
        
        let mut triggered = false;
        let mut reason = String::new();

        // Проверка по USD
        if let Some(limit_dec) = self.config.max_daily_drawdown_usd {
            if drawdown_usd >= limit_dec {
                triggered = true;
                reason = format!("Daily DD USD limit reached: {} >= {}", drawdown_usd, limit_dec);
            }
        }

        // Проверка по %
        if let Some(limit_dec) = self.config.max_daily_drawdown_pct {
            if self.peak_daily_equity > Decimal::ZERO {
                let drawdown_pct = (drawdown_usd / self.peak_daily_equity) * Decimal::from(100);
                if drawdown_pct >= limit_dec {
                    triggered = true;
                    reason = format!("Daily DD % limit reached: {}% >= {}%", drawdown_pct, limit_dec);
                }
            }
        }

        if triggered {
            self.is_blocked = true;
            error!("CRITICAL_RISK_STOP: {}. Peak: {}, Current: {}", reason, self.peak_daily_equity, current_equity);
            // Логируем в аудит (Задача 217)
            if let Some(logger) = &self.audit_logger {
                let _ = logger.log_risk_gate("MAX_DAILY_LOSS", true, &reason);
            }
            bail!("Risk: HardStop triggered");
        }

        Ok(())
    }

    /// Проверка спреда (Задача 073)
    #[inline(always)]
    pub fn check_spread_gate(
        &self, 
        best_bid: Decimal, 
        bid_vol: Decimal, 
        best_ask: Decimal, 
        ask_vol: Decimal
    ) -> bool {
        // 1. Валидация цен
        if best_bid <= Decimal::ZERO || best_ask <= Decimal::ZERO {
            warn!("Risk: Invalid prices detected: bid={}, ask={}", best_bid, best_ask);
            return false;
        }

        if best_ask <= best_bid {
            error!("Risk: Inverted spread detected: ask {} <= bid {}", best_ask, best_bid);
            return false;
        }

        // 2. Ликвидность
        if bid_vol.is_zero() || ask_vol.is_zero() {
            warn!("Risk: Zero volume at top levels: bid_vol={}, ask_vol={}", bid_vol, ask_vol);
            return false;
        }

        // 3. Расчет спреда
        let spread_abs = best_ask - best_bid;
        let mid_price = (best_ask + best_bid) / Decimal::from(2);
        let spread_bps = (spread_abs / mid_price) * Decimal::from(10_000);

        // 4. Логика проверки
        if let Some(max_bps) = self.config.max_spread_bps {
            let limit_dec = Decimal::from(max_bps);
            
            if spread_bps > limit_dec {
                warn!("Risk: Spread too wide: {} bps (Limit: {} bps)", spread_bps, limit_dec);
                return false;
            }

            if spread_bps > limit_dec * Decimal::from_f64(0.8).unwrap() {
                info!("Risk: Spread nearing limit: {} bps", spread_bps);
            } else {
                debug!("Risk: Spread OK: {} bps", spread_bps);
            }
        }

        true
    }

    /// Финальный фильтр перед отправкой ордера
    #[inline(always)]
    pub fn check_order_gate(
        &mut self,
        side: OrderSide,
        qty: Decimal,
        price: Decimal,
        current_pos: &Position,
        active_orders_count: usize,
        current_pnl: Decimal,
        mid_price: Decimal,
        tick_size: Decimal, // Задача 176: Для проверки дубликатов
    ) -> Result<()> {
        // 0. Проверка на дубликат ордера (Задача 176)
        let price_f64 = price.to_f64().unwrap_or(0.0);
        let qty_f64 = qty.to_f64().unwrap_or(0.0);
        let tick_size_f64 = tick_size.to_f64().unwrap_or(0.0);
        
        if self.is_duplicate(side, price_f64, qty_f64, tick_size_f64) {
            bail!("Risk: Duplicate order detected (fuzzy match)");
        }

        // 1. Проверка глобальных рисков и здоровья (Задача 171)
        self.check_risk_gates(current_pnl)?;

        // 2. Проверка минимальной стоимости ордера (Notional)
        let min_notional = Decimal::from_f64(5.0).unwrap(); // Bybit default
        if qty * price < min_notional {
            bail!("Risk: Order value too small: {} < {}", qty * price, min_notional);
        }

        // 3. Валидация параметров ордера (лимиты позиции и кол-во ордеров)
        self.validate_order(side, qty, current_pos, active_orders_count, mid_price)?;

        Ok(())
    }

    /// Расчет эффективного лимита позиции с учетом динамического скейлинга (Задача 178)
    #[inline(always)]
    pub fn get_effective_position_limit(&self, base_limit: Decimal) -> Decimal {
        let scale = Decimal::from_f64(self.current_scale).unwrap_or(Decimal::ONE);
        base_limit * scale
    }

    /// Проверка параметров ордера (размер позиции, номинал, маржа и количество)
    #[inline(always)]
    pub fn validate_order(
        &self, 
        side: OrderSide, 
        qty: Decimal, 
        current_pos: &Position,
        _active_orders_count: usize,
        mid_price: Decimal,
    ) -> Result<()> {
        // 1. Расчет проекции позиции
        let signed_qty = if side == OrderSide::Buy { qty } else { -qty };
        let projected_qty = current_pos.qty + signed_qty;

        // --- ЛОГИКА REDUCE-ONLY ---
        // Если новая позиция по модулю МЕНЬШЕ текущей — это закрытие/уменьшение. 
        // Такие сделки ВСЕГДА разрешены для безопасности.
        if projected_qty.abs() < current_pos.qty.abs() {
            return Ok(());
        }

        // 3. Лимит размера позиции (в базовой валюте, например BTC)
        // Задача 178: Используем динамический лимит с учетом просадки и волатильности
        if let Some(max_size_dec) = self.config.max_position_size {
            let effective_limit = self.get_effective_position_limit(max_size_dec);
            if projected_qty.abs() > effective_limit {
                bail!(
                    "Risk: DynamicPositionLimit violated. Projected: {}, Effective Limit: {} (Base: {}, Scale: {:.1}%)", 
                    projected_qty.abs(), 
                    effective_limit,
                    max_size_dec,
                    self.current_scale * 100.0
                );
            }
        }

        // 4. Лимит номинальной стоимости (Notional USD)
        if let Some(max_notional_dec) = self.config.max_notional_usd {
            let projected_notional = projected_qty.abs() * mid_price;
            if projected_notional > max_notional_dec {
                warn!("Risk Gate: Order blocked. Projected Notional: {} (Limit: {})", projected_notional, max_notional_dec);
                bail!("Risk: MaxNotionalExceeded. Projected: {}, Limit: {}", projected_notional, max_notional_dec);
            }
        }

        // 5. Лимит используемой маржи (Margin USD)
        if let Some(max_margin_dec) = self.config.max_margin_usd {
            let projected_notional = projected_qty.abs() * mid_price;
            let leverage = if current_pos.leverage.is_zero() { Decimal::ONE } else { current_pos.leverage };
            let projected_margin = projected_notional / leverage;
            
            if projected_margin > max_margin_dec {
                warn!("Risk Gate: Order blocked. Projected Margin: {} (Limit: {})", projected_margin, max_margin_dec);
                bail!("Risk: MaxMarginExceeded. Projected: {}, Limit: {}", projected_margin, max_margin_dec);
            }
        }

        Ok(())
    }

    /// Риск-гейт по количеству открытых ордеров (Задача 074)
    ///
    /// Возвращает `true`, если можно продолжать размещение новых ордеров.
    /// Если лимит отключен (`max_open_orders == None`) — всегда `true`.
    pub fn check_orders_limit_gate(&self, active_orders_count: usize) -> bool {
        let Some(limit) = self.config.max_open_orders else {
            return true;
        };

        let count = active_orders_count as u32;

        if count >= limit {
            warn!(
                "MAX ORDERS REACHED: {}/{} (limit reached, gate closed)",
                count, limit
            );
            return false;
        }

        let warn_threshold = limit * 80 / 100;

        if count >= warn_threshold {
            info!(
                "Approaching orders limit: {}/{} (>= 80%)",
                count, limit
            );
        } else {
            debug!(
                "Active orders: {}/{} (gate OK)",
                count, limit
            );
        }

        true
    }

    /// Проверка необходимости сброса дневной статистики (Задача 111)
    pub fn should_reset_daily_stats(&self, state: &RiskState, now: DateTime<Utc>, reset_hour: u32) -> bool {
        let last_reset = DateTime::from_timestamp(state.last_pnl_reset_ts / 1000, 0)
            .unwrap_or_else(|| DateTime::from_timestamp(0, 0).unwrap());
        
        // Сброс если: сменилась дата ИЛИ наступил час сброса (если в прошлые сутки не сбрасывали)
        now.date_naive() != last_reset.date_naive() && now.hour() >= reset_hour
    }

    /// Основная проверка дневного лимита убытка (Задача 111)
    pub async fn check_daily_limit(
        &mut self,
        bot_config: &BotConfig,
        state: &mut RiskState,
        position_manager: &PositionManager,
        rest_client: &impl BybitRestClientTrait,
        mid_price: Decimal,
    ) -> Result<bool> {
        let now = Utc::now();
        
        // 1. Получаем актуальный PnL (реализованный + нереализованный)
        let current_pnl = position_manager.get_total_pnl(mid_price);
        
        // 2. Получаем актуальный Equity с биржи (с ретраями)
        let equity = rest_client.get_equity_with_retry(3).await?;

        // 3. Проверка и выполнение сброса начала дня
        if self.should_reset_daily_stats(state, now, bot_config.daily_reset_hour_utc) {
            state.day_start_pnl = current_pnl;
            state.last_pnl_reset_ts = now.timestamp_millis();
            info!("Daily PnL limit reset. New day_start_pnl: {}", current_pnl);
            // Сохранение стейта будет выполнено вызывающей стороной (ExecutionEngine)
        }

        // 4. Расчет просадки за сегодня
        let daily_pnl = current_pnl - state.day_start_pnl;
        
        if daily_pnl < Decimal::ZERO {
            if equity.is_zero() {
                return Ok(true);
            }
            let loss_pct = daily_pnl.abs() / equity;
            let limit_pct = bot_config.max_daily_loss_pct / Decimal::from(100);
            
            if loss_pct > limit_pct {
                error!(
                    "DAILY LOSS LIMIT BREACHED: {:.2}% (Limit: {}%)", 
                    (loss_pct * Decimal::from(100)), 
                    bot_config.max_daily_loss_pct
                );
                return Ok(false); 
            }
        }
        
        Ok(true)
    }

    /// Проверка цены ордера на отклонение от середины стакана и кратность шагу цены (Задача 075)
    pub fn validate_order_price(
        &self,
        order_price: Decimal,
        mid_price: Decimal,
        tick_size: Decimal,
    ) -> Result<bool> {
        // 1. Проверка на кратность tick_size (Bybit rejection 10001)
        if tick_size > Decimal::ZERO && !(order_price % tick_size).is_zero() {
            bail!("Price {} is not a multiple of tick_size {}", order_price, tick_size);
        }

        // 2. Проверка отклонения (Price Deviation)
        let limit = match self.config.max_price_deviation_pct {
            Some(l) => l,
            None => return Ok(true), // Лимит отключен
        };

        if mid_price <= Decimal::ZERO {
            bail!("Invalid mid_price: {}", mid_price);
        }

        let deviation = (order_price - mid_price).abs() / mid_price;

        if deviation > limit {
            warn!("Price deviation too high: {:.2}% (Order: {}, Mid: {})",
                   deviation * Decimal::from(100), order_price, mid_price);
            return Ok(false);
        }

        Ok(true)
    }

    /// Проверка лимита номинального риска на символ (Задача 114)
    pub fn check_notional_limit(
        &self, 
        current_size: Decimal, 
        pending_size: Decimal, // Объем активных ордеров на той же стороне
        order_size: Decimal, 
        order_side: OrderSide, 
        mid_price: Decimal,
        max_notional_usd: Decimal,
    ) -> bool {
        // 1. Если лимит не установлен — пропускаем
        if max_notional_usd.is_zero() {
            return true;
        }

        // 2. Рассчитываем целевой размер позиции (интегральный риск)
        // Важно: учитываем знак позиции (Long +, Short -)
        let new_size = match order_side {
            OrderSide::Buy => current_size + pending_size + order_size,
            OrderSide::Sell => current_size - pending_size - order_size,
        };

        let new_notional = new_size.abs() * mid_price;

        // 3. Логика Reduce-Only: если новый риск меньше текущего — разрешаем всегда
        let current_notional = (current_size.abs() + pending_size) * mid_price;
        if new_notional <= current_notional {
            return true; 
        }

        // 4. Проверка лимита на увеличение позиции
        if new_notional > max_notional_usd {
            warn!(
                "NOTIONAL BLOCKED: Future exposure ${:.2} > Limit ${:.2}", 
                new_notional, 
                max_notional_usd
            );
            return false;
        }

        true
    }

    /// Проверка неактивности котировок (Dead Man's Switch) (Задача 113)
    pub fn check_inactivity(&self, last_update: Option<Instant>) -> bool {
        let last_update = match last_update {
            Some(t) => t,
            None => return true, // Еще не получали данных
        };

        if self.config.max_inactivity_ms == 0 {
            return true;
        }

        let elapsed = last_update.elapsed().as_millis() as u64;

        if elapsed > self.config.max_inactivity_ms / 2 && elapsed < self.config.max_inactivity_ms {
            warn!("Inactivity warning: No book updates for {}ms", elapsed);
        }

        elapsed < self.config.max_inactivity_ms
    }

    /// Проверяет лимит на количество сделок за окно времени
    pub fn check_overtrading_limit(&mut self, state: &mut RiskState) -> bool {
        if self.config.max_trades_limit == 0 {
            return true;
        }

        let now_ms = Utc::now().timestamp_millis();
        let cutoff = now_ms - (self.config.max_trades_window_sec as i64 * 1000);

        // 1. Очистка устаревших записей
        let initial_len = self.trade_history.len();
        self.trade_history.retain(|&ts| ts > cutoff);

        // 2. Синхронизация со стейтом только если были изменения
        if self.trade_history.len() != initial_len {
            state.recent_trade_timestamps = self.trade_history.iter().copied().collect();
        }

        // 3. Проверка лимита
        if self.trade_history.len() >= self.config.max_trades_limit {
            error!(
                "OVERTRADING PROTECT: {} trades in last {}s window (Limit: {})",
                self.trade_history.len(),
                self.config.max_trades_window_sec,
                self.config.max_trades_limit
            );
            return false;
        }

        true
    }

    /// Регистрирует факт исполнения ордера (Fill)
    pub fn register_fill(&mut self, timestamp_ms: i64) {
        self.trade_history.push_back(timestamp_ms);
        info!(
            "Fill registered at {}. Total in window: {}",
            timestamp_ms,
            self.trade_history.len()
        );
    }

    /// Обновляет историю сделок из стейта (при запуске)
    pub fn update_trade_history(&mut self, timestamps: &[i64], state: &mut RiskState) {
        self.trade_history = VecDeque::from(timestamps.to_vec());
        info!("RiskManager: Trade history loaded ({} entries)", self.trade_history.len());
        // Сразу очищаем устаревшие записи
        self.check_overtrading_limit(state);
    }

    /// Расчет эффективного порога входа на основе серии убытков (Задача 115)
    pub fn get_effective_threshold(&self, current_streak: usize, bot_config: &BotConfig) -> f64 {
        // Ограничиваем влияние серии лимитом из конфига
        let effective_streak = current_streak.min(bot_config.threshold_max_streak) as f64;
        
        let dynamic_part = bot_config.threshold_loss_mult * effective_streak;
        
        (bot_config.threshold_base + dynamic_part)
            .clamp(bot_config.threshold_min, bot_config.threshold_max)
    }

    /// Проверка задержки (Latency Kill Switch) (Задача 116)
    #[inline(always)]
    pub fn check_latency(&mut self, network_micros: u64, inference_micros: u64, bot_config: &BotConfig) -> bool {
        let total = network_micros + inference_micros;
        let limit = bot_config.max_total_latency_micros;

        if total > limit && limit > 0 {
            self.consecutive_latency_rejects += 1;
            warn!(
                "LATENCY REJECT: total {}µs (Net: {}µs, Inf: {}µs). Count: {}", 
                total, network_micros, inference_micros, self.consecutive_latency_rejects
            );
            return false;
        }

        self.consecutive_latency_rejects = 0; // Сброс при успешном проходе
        true
    }

    /// Проверка гейта дисбаланса стакана (Order Book Imbalance Gate) (Задача 117)
    /// Блокирует вход против доминирующей стороны стакана
    #[inline(always)]
    pub fn check_imbalance_gate(&self, side: OrderSide, current_obi: f64, bot_config: &BotConfig) -> bool {
        if bot_config.obi_threshold <= 0.0 {
            return true;
        }

        match side {
            OrderSide::Buy => {
                // Блокируем покупку, если в стакане доминируют продавцы
                if current_obi < -bot_config.obi_threshold {
                    warn!("BUY BLOCKED: OBI {:.2} < -{:.2}", current_obi, bot_config.obi_threshold);
                    return false;
                }
            }
            OrderSide::Sell => {
                // Блокируем продажу, если в стакане доминируют покупатели
                if current_obi > bot_config.obi_threshold {
                    warn!("SELL BLOCKED: OBI {:.2} > {:.2}", current_obi, bot_config.obi_threshold);
                    return false;
                }
            }
        }
        true
    }

    /// Проверка периода блокировки после серии убытков (Lockout Period) (Задача 118)
    #[inline(always)]
    pub fn is_in_lockout(&self, state: &RiskState, bot_config: &BotConfig) -> bool {
        let period = bot_config.lockout_period_sec;
        let streak = state.loss_streak;
        let threshold = bot_config.lockout_streak_threshold;

        // Если лимит не настроен или серия убытков не достигла порога
        if period == 0 || threshold == 0 || streak < threshold {
            return false;
        }

        let now_ms = Utc::now().timestamp_millis();
        // Используем saturating_sub для защиты от скачков времени
        let elapsed_ms = now_ms.saturating_sub(state.last_loss_timestamp_ms);
        let elapsed_sec = (elapsed_ms / 1000) as u64;

        if elapsed_sec < period {
            let remaining = period - elapsed_sec;
            warn!(
                "LOCKOUT ACTIVE: Streak {} >= {}. Cooling down for another {}s", 
                streak, threshold, remaining
            );
            return true;
        }

        false
    }

    /// Проверка ручного экстренного стопа через файл (Задача 119)
    pub fn check_manual_stop(&self, bot_dir: &Path, bot_config: &BotConfig) -> Option<PathBuf> {
        // 1. Проверка локального файла в папке бота bots/SYMBOL/STOP
        let local_stop = bot_dir.join(&bot_config.stop_file_name);
        if local_stop.exists() {
            return Some(local_stop);
        }

        // 2. Проверка глобального файла в корне проекта STOP_ALL
        if bot_config.global_stop_enabled {
            let global_stop = Path::new("STOP_ALL");
            if global_stop.exists() {
                return Some(global_stop.to_path_buf());
            }
        }

        None
    }

    /// Детальная верификация состояния с биржей (Задача 120)
    /// Сравнивает локальную позицию и ордера с данными биржи
    pub fn verify_consistency(
        &self,
        local_pos: &Position,
        ex_pos: &Position,
        local_orders: &std::collections::HashMap<String, crate::trading::types::OrderInfo>,
        ex_orders: &[crate::trading::types::OrderInfo],
        price_threshold: Decimal,
    ) -> bool {
        // 1. Сверка позиции: объем и средняя цена
        let size_diff = (local_pos.qty - ex_pos.qty).abs();
        let price_diff = (local_pos.avg_price - ex_pos.avg_price).abs();

        if size_diff > Decimal::ZERO {
            error!(
                "POS DESYNC: Local {}@{} vs Ex {}@{}",
                local_pos.qty, local_pos.avg_price, ex_pos.qty, ex_pos.avg_price
            );
            return false;
        }

        if price_diff > price_threshold && !ex_pos.avg_price.is_zero() {
            error!(
                "PRICE DESYNC: Local avg_price {} vs Ex {} (threshold: {})",
                local_pos.avg_price, ex_pos.avg_price, price_threshold
            );
            return false;
        }

        // 2. Сверка ордеров: маппинг по link_id
        if local_orders.len() != ex_orders.len() {
            error!(
                "ORDER COUNT DESYNC: Local {} vs Ex {}",
                local_orders.len(),
                ex_orders.len()
            );
            return false;
        }

        // Создаем HashMap из ex_orders для быстрого поиска по link_id
        let ex_orders_map: std::collections::HashMap<String, &crate::trading::types::OrderInfo> = 
            ex_orders.iter()
                .filter_map(|o| {
                    o.link_id.as_ref().map(|id| (id.clone(), o))
                })
                .collect();

        // Проверяем каждый локальный ордер
        for (link_id, local_order) in local_orders {
            match ex_orders_map.get(link_id) {
                Some(ex_order) => {
                    if local_order.price != ex_order.price || local_order.qty != ex_order.qty {
                        error!(
                            "ORDER DATA DESYNC for {}: Local {}@{} vs Ex {}@{}",
                            link_id,
                            local_order.qty,
                            local_order.price,
                            ex_order.qty,
                            ex_order.price
                        );
                        return false;
                    }
                }
                None => {
                    error!(
                        "ORPHAN ORDER: {} found locally but not on exchange",
                        link_id
                    );
                    return false;
                }
            }
        }

        // Проверка на orphan orders на бирже (ордера на бирже, которых нет локально)
        for ex_order in ex_orders {
            if let Some(ex_link_id) = &ex_order.link_id {
                if !local_orders.contains_key(ex_link_id) {
                    error!(
                        "ORPHAN ORDER: {} found on exchange but not locally",
                        ex_link_id
                    );
                    return false;
                }
            }
        }

        true
    }

    /// Проверяет ликвидность стакана перед исполнением ордера (задача 162).
    /// Блокирует или корректирует ордера в зависимости от доступной ликвидности.
    /// 
    /// # Параметры
    /// - `ob`: Ссылка на стакан ордеров
    /// - `side`: Сторона ордера (Buy/Sell)
    /// - `size_usd`: Размер ордера в USD
    /// - `bot_config`: Конфигурация бота с настройками liquidity_filter
    /// 
    /// # Возвращает
    /// - `RiskResult::Allow` - ордер разрешен
    /// - `RiskResult::AdjustSize(new_size)` - ордер разрешен с уменьшенным размером
    /// - `RiskResult::Reject(reason)` - ордер отклонен
    pub fn check_liquidity_gate(
        &self,
        ob: &OrderBook,
        side: Side,
        size_usd: f64,
        bot_config: &BotConfig,
    ) -> RiskResult {
        // Если фильтр не настроен, пропускаем проверку
        let filter_config = match &bot_config.liquidity_filter {
            Some(cfg) => cfg,
            None => return RiskResult::Allow,
        };

        if size_usd <= 0.0 {
            return RiskResult::Reject("Order size must be positive".to_string());
        }

        // Проверка 1: Объем на лучшем уровне
        let volume_at_best = ob.get_volume_at_best(side);
        let min_required_volume = size_usd * filter_config.min_top_multiple;

        if volume_at_best < min_required_volume {
            let msg = format!(
                "[Risk] Insufficient liquidity at best level: ${:.2} < ${:.2} (required {}x of ${:.2})",
                volume_at_best, min_required_volume, filter_config.min_top_multiple, size_usd
            );
            
            if filter_config.adjust_size_if_thin {
                // Корректируем размер до доступной ликвидности
                let adjusted_size = volume_at_best / filter_config.min_top_multiple;
                if adjusted_size > 0.0 {
                    warn!("{} - Adjusting size to ${:.2}", msg, adjusted_size);
                    return RiskResult::AdjustSize(adjusted_size);
                } else {
                    warn!("{} - Cannot adjust (too small)", msg);
                    return RiskResult::Reject(msg);
                }
            } else {
                warn!("{}", msg);
                return RiskResult::Reject(msg);
            }
        }

        // Проверка 2: Проскальзывание (VWAP impact)
        let impact_bps = ob.calculate_vwap_impact(side, size_usd);
        
        if impact_bps > filter_config.max_impact_bps {
            let msg = format!(
                "[Risk] Slippage impact {:.2}bps > limit {:.2}bps. Available at best: ${:.2}, Order: ${:.2}",
                impact_bps, filter_config.max_impact_bps, volume_at_best, size_usd
            );
            
            if filter_config.adjust_size_if_thin {
                // Бинарный поиск максимального размера, который проходит по фильтру
                let mut low = 0.0;
                let mut high = size_usd;
                let mut best_size = 0.0;
                
                // Максимум 10 итераций бинарного поиска
                for _ in 0..10 {
                    let mid = (low + high) / 2.0;
                    if mid <= 0.0 {
                        break;
                    }
                    
                    let test_impact = ob.calculate_vwap_impact(side, mid);
                    if test_impact <= filter_config.max_impact_bps {
                        best_size = mid;
                        low = mid;
                    } else {
                        high = mid;
                    }
                    
                    // Если разница меньше $1, останавливаемся
                    if (high - low).abs() < 1.0 {
                        break;
                    }
                }
                
                if best_size > 0.0 {
                    warn!("{} - Adjusting size to ${:.2}", msg, best_size);
                    return RiskResult::AdjustSize(best_size);
                } else {
                    warn!("{} - Cannot adjust (impact too high)", msg);
                    return RiskResult::Reject(msg);
                }
            } else {
                warn!("{}", msg);
                return RiskResult::Reject(msg);
            }
        }

        // Все проверки пройдены
        debug!(
            "[Risk] Liquidity check passed: size=${:.2}, volume_at_best=${:.2}, impact={:.2}bps",
            size_usd, volume_at_best, impact_bps
        );
        RiskResult::Allow
    }

    /// Проверка Time Decay Stop (задача 163)
    /// Возвращает true если позиция "протухла" и должна быть закрыта
    pub fn check_time_stop(&self, position: &Position, bot_config: &BotConfig) -> bool {
        // Если механизм отключен, пропускаем проверку
        if !bot_config.time_decay.enabled {
            return false;
        }

        // Получаем текущее время
        let now = crate::utils::helpers::get_unix_ms();

        // Проверяем возраст позиции
        position.is_aged(now, &bot_config.time_decay)
    }
    
    /// Задача 164: Регистрация Maker/Taker филла
    pub fn register_fill_type(&mut self, is_maker: bool) {
        if is_maker {
            self.maker_fills_count += 1;
        } else {
            self.taker_fills_count += 1;
        }
    }
    
    /// Задача 164: Расчет текущего Taker Ratio
    pub fn get_taker_ratio(&self) -> f64 {
        let total = self.maker_fills_count + self.taker_fills_count;
        if total == 0 {
            return 0.0;
        }
        self.taker_fills_count as f64 / total as f64
    }
    
    /// Задача 164: Получение скорректированного offset с учетом Soft Limit
    /// Если taker_ratio > taker_ratio_limit, увеличиваем offset для более глубокого захода в стакан
    pub fn get_adjusted_maker_offset(&self, base_offset_ticks: u32) -> u32 {
        let taker_ratio = self.get_taker_ratio();
        let limit = self.config.taker_ratio_limit;
        
        if taker_ratio > limit {
            // Увеличиваем offset пропорционально превышению порога
            // Например, при taker_ratio = 0.3 (превышение на 0.1), увеличиваем на 50%
            let excess = taker_ratio - limit;
            let multiplier = 1.0 + (excess * 5.0); // 5x множитель для агрессивного увеличения
            let adjusted = (base_offset_ticks as f64 * multiplier).ceil() as u32;
            
            debug!(
                "Soft Limit activated: taker_ratio={:.2}%, offset: {} -> {} ticks",
                taker_ratio * 100.0, base_offset_ticks, adjusted
            );
            
            adjusted
        } else {
            base_offset_ticks
        }
    }
    
    /// Задача 164: Сброс счетчиков Maker/Taker (например, при дневном сбросе)
    pub fn reset_fill_counters(&mut self) {
        self.maker_fills_count = 0;
        self.taker_fills_count = 0;
    }
    
    /// Задача 164: Проверка необходимости принудительно устанавливать Post-Only флаг
    /// Возвращает true если taker_ratio превысил limit (Soft Limit активирован)
    pub fn should_force_post_only(&self) -> bool {
        self.get_taker_ratio() > self.config.taker_ratio_limit
    }

    // ============================================================================
    // Задача 176: Защита от дублирования ордеров (Duplicate Order Prevention)
    // ============================================================================

    /// Регистрация намерения разместить ордер
    pub fn register_order_intent(&mut self, link_id: String, side: OrderSide, price: f64, qty: f64) {
        // Задача 176: Валидация параметров для предотвращения регистрации с нулевым объемом
        if qty <= 0.0 {
            warn!("Attempted to register order intent with invalid qty: {}. Skipping registration.", qty);
            return;
        }
        if price <= 0.0 {
            warn!("Attempted to register order intent with invalid price: {}. Skipping registration.", price);
            return;
        }
        
        let now = Utc::now().timestamp_millis() as u64;
        let intent = OrderIntent {
            side,
            price,
            qty,
            timestamp: now,
            filled_qty: 0.0, // Задача 179: Инициализируем filled_qty = 0
        };
        self.active_intents.insert(link_id.clone(), intent);
        debug!("Registered order intent: {} (side: {:?}, price: {}, qty: {})", link_id, side, price, qty);
    }

    /// Удаление намерения после исполнения или отмены ордера
    pub fn remove_order_intent(&mut self, link_id: &str) {
        if self.active_intents.remove(link_id).is_some() {
            debug!("Removed order intent: {}", link_id);
        }
    }

    /// Обновление исполненного объема для интента (Задача 179)
    pub fn update_order_intent_filled_qty(&mut self, link_id: &str, filled_qty: f64) {
        if let Some(intent) = self.active_intents.get_mut(link_id) {
            intent.filled_qty = filled_qty;
            debug!("Updated order intent {} filled_qty: {}", link_id, filled_qty);
        }
    }

    /// Проверка адекватности цены ордера (Задача 177: Extreme Price Deviation & Fat Finger Protection)
    /// 
    /// Защищает от "Fat Finger" ошибок и исполнения в условиях аномальных рыночных зазоров.
    /// Блокирует любой ордер (Limit или Market), цена которого отклоняется от рыночного эталона
    /// более чем на заданный критический порог.
    /// 
    /// # Параметры
    /// - `side`: Сторона ордера (Buy/Sell)
    /// - `order_price`: Цена ордера (None для маркет-ордеров)
    /// - `order_book`: Ссылка на стакан
    /// - `qty`: Объем ордера (для расчета VWAP маркет-ордеров)
    /// 
    /// # Возвращает
    /// - `Ok(())` если цена в пределах допустимого отклонения
    /// - `Err` если отклонение превышает порог или стакан не инициализирован
    pub fn check_price_sanity(
        &self,
        side: OrderSide,
        order_price: Option<f64>,
        order_book: &OrderBook,
        qty: f64,
    ) -> Result<()> {
        // 1. Проверка инициализации стакана
        let (best_bid, _, best_ask, _) = order_book.get_best_bid_ask_with_vol();
        if best_bid <= 0.0 || best_ask <= 0.0 {
            bail!("OrderBook not initialized: no valid bid/ask levels");
        }

        // 2. Получение эталонной цены (reference price)
        let mid_price = order_book.get_mid_price();
        let ref_price = match self.config.price_reference_source {
            PriceReferenceSource::MidPrice => {
                if mid_price <= 0.0 {
                    bail!("Invalid mid_price: {}", mid_price);
                }
                mid_price
            }
            PriceReferenceSource::LastPrice => {
                // Используем last_trade_price, fallback на mid_price
                match order_book.last_trade_price {
                    Some(last) if last > 0.0 => last,
                    _ => {
                        warn!("last_trade_price not available, using mid_price as fallback");
                        if mid_price <= 0.0 {
                            bail!("Invalid mid_price fallback: {}", mid_price);
                        }
                        mid_price
                    }
                }
            }
            PriceReferenceSource::MarkPrice => {
                // Задача 233: Используем маркированную цену (индексную цену)
                let mark_price = order_book.get_mark_price();
                if mark_price <= 0.0 {
                    bail!("Invalid mark_price: {}", mark_price);
                }
                mark_price
            }
            PriceReferenceSource::Both => {
                // Проверяем оба эталона, используем mid_price как основной
                if mid_price <= 0.0 {
                    bail!("Invalid mid_price: {}", mid_price);
                }
                mid_price
            }
        };

        // 3. Определение ожидаемой цены исполнения
        let expected_price = match order_price {
            // Лимитный ордер: используем указанную цену
            Some(price) => price,
            // Маркет-ордер: рассчитываем ожидаемую цену через VWAP
            None => {
                let side_for_vwap = match side {
                    OrderSide::Buy => Side::Buy,
                    OrderSide::Sell => Side::Sell,
                };
                
                // Пытаемся получить VWAP для заданного объема
                match order_book.get_execution_vwap(side_for_vwap, qty) {
                    Some(vwap) => vwap,
                    None => {
                        // Недостаточно ликвидности, используем mid_price как оценку
                        warn!("Insufficient liquidity for VWAP calculation, using mid_price");
                        mid_price
                    }
                }
            }
        };

        // 4. Валидация для лимитных ордеров: сравнение с противоположной стороной стакана
        if let Some(limit_price) = order_price {
            let comparison_price = match side {
                OrderSide::Buy => best_ask,  // Для покупки сравниваем с best_ask
                OrderSide::Sell => best_bid, // Для продажи сравниваем с best_bid
            };

            let limit_bps = self.config.max_price_deviation_bps as f64;
            let max_deviation = comparison_price * (limit_bps / 10000.0);

            let is_invalid = match side {
                OrderSide::Buy => limit_price > comparison_price + max_deviation,
                OrderSide::Sell => limit_price < comparison_price - max_deviation,
            };

            if is_invalid {
                let deviation_bps = ((limit_price - comparison_price).abs() / comparison_price) * 10000.0;
                bail!(
                    "Limit order price deviation too high: {:?} order at {} vs {} (best_{}), deviation: {:.2} bps > {} bps",
                    side,
                    limit_price,
                    comparison_price,
                    if matches!(side, OrderSide::Buy) { "ask" } else { "bid" },
                    deviation_bps,
                    limit_bps
                );
            }
        }

        // 5. Проверка отклонения ожидаемой цены от эталона
        let deviation_bps = ((expected_price - ref_price).abs() / ref_price) * 10000.0;
        let limit_bps = self.config.max_price_deviation_bps as f64;

        if deviation_bps > limit_bps {
            bail!(
                "Price deviation too high: expected {} vs ref {} ({:?}), deviation: {:.2} bps > {} bps",
                expected_price,
                ref_price,
                self.config.price_reference_source,
                deviation_bps,
                limit_bps
            );
        }

        // 6. Дополнительная проверка для режима Both
        if matches!(self.config.price_reference_source, PriceReferenceSource::Both) {
            if let Some(last_trade) = order_book.last_trade_price {
                if last_trade > 0.0 {
                    let last_deviation_bps = ((expected_price - last_trade).abs() / last_trade) * 10000.0;
                    if last_deviation_bps > limit_bps {
                        bail!(
                            "Price deviation vs LastTrade too high: expected {} vs last_trade {}, deviation: {:.2} bps > {} bps",
                            expected_price,
                            last_trade,
                            last_deviation_bps,
                            limit_bps
                        );
                    }
                }
            }
        }

        Ok(())
    }

    /// Проверка на дубликат ордера с fuzzy matching
    /// 
    /// Возвращает true, если найден похожий активный ордер:
    /// - Та же сторона (side)
    /// - Объем в пределах tolerance (duplicate_qty_tolerance_pct)
    /// - Цена в пределах tolerance (duplicate_price_tolerance_ticks * tick_size)
    /// - Время в пределах окна (duplicate_window_ms)
    pub fn is_duplicate(&self, side: OrderSide, price: f64, qty: f64, tick_size: f64) -> bool {
        let now = Utc::now().timestamp_millis() as u64;
        let window_ms = self.config.duplicate_window_ms;
        let qty_tolerance = self.config.duplicate_qty_tolerance_pct;
        let price_tolerance_ticks = self.config.duplicate_price_tolerance_ticks;

        for (link_id, intent) in &self.active_intents {
            // 1. Проверка стороны
            if intent.side != side {
                continue;
            }

            // 2. Проверка времени (окно детекции)
            if now.saturating_sub(intent.timestamp) > window_ms {
                continue;
            }

            // 3. Проверка объема (относительный допуск)
            // Задача 176: Добавлена валидация для предотвращения деления на ноль
            if intent.qty <= 0.0 || qty <= 0.0 {
                warn!("Invalid qty in duplicate check: intent.qty={}, new qty={}", intent.qty, qty);
                continue; // Пропускаем некорректные объемы
            }
            
            let qty_diff_pct = (qty - intent.qty).abs() / intent.qty;

            if qty_diff_pct > qty_tolerance {
                continue;
            }

            // 4. Проверка цены (абсолютный допуск в тиках)
            let price_diff = (price - intent.price).abs();
            let price_tolerance = tick_size * price_tolerance_ticks as f64;

            if price_diff > price_tolerance {
                continue;
            }

            // Все условия совпали - это дубликат!
            warn!(
                "DUPLICATE ORDER DETECTED: side={:?}, price={}, qty={} matches intent {} (price={}, qty={}, age={}ms)",
                side, price, qty, link_id, intent.price, intent.qty, now.saturating_sub(intent.timestamp)
            );
            return true;
        }

        false
    }

    /// Получение мутабельной ссылки на активные интенты (для cleanup в HealthMonitor)
    pub fn get_active_intents_mut(&mut self) -> &mut HashMap<String, OrderIntent> {
        &mut self.active_intents
    }

    /// Получение количества активных интентов (для мониторинга)
    pub fn get_active_intents_count(&self) -> usize {
        self.active_intents.len()
    }

    /// Обработка обнаруженного дрейфа модели (Задача 224)
    /// 
    /// Действия:
    /// 1. Отправка Critical Alert через AlertManager
    /// 2. Сокращение размера позиций (Degraded mode)
    /// 3. Опционально: полная остановка торговли (Emergency mode)
    pub fn handle_model_drift(&mut self, entropy: f32, bot_config: &crate::config::types::BotConfig) {
        use tracing::warn;
        
        warn!(
            "Model drift detected! Entropy: {:.4}, threshold: {:.4}",
            entropy, bot_config.entropy_drift_threshold
        );
        
        // 1. Отправка алерта через AlertManager
        if let Some(ref alert_manager) = self.alert_manager {
            use crate::monitoring::alert_manager::{Alert, AlertLevel};
            
            let alert = Alert::new(
                AlertLevel::Critical,
                format!(
                    "🚨 MODEL DRIFT DETECTED\nEntropy: {:.4}\nThreshold: {:.4}\nAction: Reducing position size to {:.0}%",
                    entropy,
                    bot_config.entropy_drift_threshold,
                    bot_config.drift_scale_factor * 100.0
                ),
                "ModelDriftMonitor".to_string(),
            );
            alert_manager.send_alert(alert);
        }
        
        // 2. Сокращение размера позиций (Degraded mode)
        self.current_scale = bot_config.drift_scale_factor as f64;
        warn!(
            "Position size reduced to {:.0}% due to model drift",
            self.current_scale * 100.0
        );
        
        // 3. Опционально: полная остановка торговли (Emergency mode)
        if bot_config.drift_stop_enabled {
            self.is_blocked = true;
            warn!("Trading blocked due to model drift (drift_stop_enabled = true)");
            
            if let Some(ref alert_manager) = self.alert_manager {
                use crate::monitoring::alert_manager::{Alert, AlertLevel};
                
                let emergency_alert = Alert::new(
                    AlertLevel::Critical,
                    "🛑 EMERGENCY: Trading blocked due to model drift".to_string(),
                    "ModelDriftMonitor".to_string(),
                );
                alert_manager.send_alert(emergency_alert);
            }
        }
    }

    /// Восстановление после дрейфа модели (Задача 224)
    /// 
    /// Восстанавливает нормальный размер позиций и разблокирует торговлю
    pub fn recover_from_drift(&mut self) {
        use tracing::info;
        
        self.current_scale = 1.0;
        self.is_blocked = false;
        
        info!("Model drift recovery: position size restored to 100%, trading unblocked");
        
        if let Some(ref alert_manager) = self.alert_manager {
            use crate::monitoring::alert_manager::{Alert, AlertLevel};
            
            let alert = Alert::new(
                AlertLevel::Info,
                "✅ Model drift recovered. Position size restored to 100%".to_string(),
                "ModelDriftMonitor".to_string(),
            );
            alert_manager.send_alert(alert);
        }
    }

    /// Применение штрафа за ошибку маржи (Задача 231)
    /// 
    /// Уменьшает margin_multiplier до значения из конфига и устанавливает timestamp
    pub fn apply_margin_penalty(&mut self, config: &RiskConfig) {
        use tracing::warn;
        
        self.margin_multiplier = config.margin_penalty_multiplier;
        self.last_margin_error_ts = Some(Instant::now());
        
        warn!(
            "Margin penalty applied: position size reduced to {:.1}% for {} minutes",
            self.margin_multiplier * 100.0,
            config.margin_error_backoff_minutes
        );
        
        if let Some(ref alert_manager) = self.alert_manager {
            use crate::monitoring::alert_manager::{Alert, AlertLevel};
            
            let alert = Alert::new(
                AlertLevel::Warning,
                format!(
                    "⚠️ Margin penalty: position size reduced to {:.1}% for {} minutes",
                    self.margin_multiplier * 100.0,
                    config.margin_error_backoff_minutes
                ),
                "MarginErrorRecovery".to_string(),
            );
            alert_manager.send_alert(alert);
        }
    }

    /// Проверка и сброс штрафа за ошибку маржи (Задача 231)
    /// 
    /// Проверяет, прошло ли достаточно времени с момента последней ошибки маржи.
    /// Если да, восстанавливает margin_multiplier до 1.0
    pub fn check_and_reset_margin_penalty(&mut self, config: &RiskConfig) {
        if let Some(last_error_ts) = self.last_margin_error_ts {
            let elapsed = last_error_ts.elapsed();
            let backoff_duration = std::time::Duration::from_secs(config.margin_error_backoff_minutes * 60);
            
            if elapsed >= backoff_duration {
                use tracing::info;
                
                self.margin_multiplier = 1.0;
                self.last_margin_error_ts = None;
                
                info!("Margin penalty expired: position size restored to 100%");
                
                if let Some(ref alert_manager) = self.alert_manager {
                    use crate::monitoring::alert_manager::{Alert, AlertLevel};
                    
                    let alert = Alert::new(
                        AlertLevel::Info,
                        "✅ Margin penalty expired: position size restored to 100%".to_string(),
                        "MarginErrorRecovery".to_string(),
                    );
                    alert_manager.send_alert(alert);
                }
            }
        }
    }

    /// Получить текущий множитель маржи (Задача 231)
    /// 
    /// Возвращает текущее значение margin_multiplier для использования при расчете размера позиции
    pub fn get_margin_multiplier(&self) -> f64 {
        self.margin_multiplier
    }

    /// Задача 233: Проверить отклонение цены ордера от Mark Price
    /// Возвращает true если отклонение превышает лимит
    pub fn check_mark_price_deviation(&self, order_price: f64, mark_price: f64) -> bool {
        if mark_price <= 0.0 {
            return false;
        }
        
        let mark_dev = (order_price - mark_price).abs() / mark_price;
        mark_dev > self.config.max_mark_deviation
    }

    /// Задача 233: Проверить спред и отклонение от Mark Price для выхода из режима шока
    pub fn is_market_stabilized(&self, ob: &OrderBookSnapshot) -> bool {
        let spread_bps = ob.get_spread_bps();
        let mid_price = ob.get_mid_price();
        let mark_price = ob.mark_price;
        
        if mark_price <= 0.0 {
            return spread_bps < self.config.max_spread_bps_shock;
        }
        
        let mark_dev = (mid_price - mark_price).abs() / mark_price;
        spread_bps < self.config.max_spread_bps_shock && mark_dev < self.config.max_mark_deviation
    }

    /// Задача 065: Проверка возможности открытия позиции
    /// Проверяет, не заблокирован ли риск-менеджер и не превышает ли qty максимальный размер позиции
    pub fn can_open_position(&self, qty: Decimal) -> bool {
        // Проверка блокировки
        if self.is_blocked {
            return false;
        }
        
        // Проверка максимального размера позиции
        if let Some(max_size) = self.config.max_position_size {
            if qty > max_size {
                return false;
            }
        }
        
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dec(val: f64) -> Decimal {
        Decimal::from_f64(val).unwrap()
    }

    fn create_test_risk_config(max_spread_bps: Option<u32>) -> RiskConfig {
        RiskConfig {
            drawdown_stop_pct: 5.0,
            max_orders_per_minute: 30,
            max_open_orders: None,
            max_position_size: None,
            max_notional_usd: None,
            max_margin_usd: None,
            max_daily_drawdown_usd: None,
            max_daily_drawdown_pct: None,
            auto_reset_at_midnight: true,
            max_drawdown_pct: None,
            max_spread_bps,
            max_price_deviation_pct: Some(dec(0.02)),
        }
    }

    #[test]
    fn test_inverted_spread_detection() {
        let config = create_test_risk_config(Some(10));
        let risk_manager = RiskManager::new(config, dec(1000.0));

        let best_bid = dec(10.00);
        let best_ask = dec(9.99);
        let bid_vol = dec(100.0);
        let ask_vol = dec(100.0);

        let result = risk_manager.check_spread_gate(best_bid, bid_vol, best_ask, ask_vol);
        assert_eq!(result, false, "Inverted spread should be rejected");
    }

    #[test]
    fn test_zero_volume_detection() {
        let config = create_test_risk_config(Some(10));
        let risk_manager = RiskManager::new(config, dec(1000.0));

        let best_bid = dec(10.00);
        let best_ask = dec(10.01);
        let bid_vol = dec(0.0);
        let ask_vol = dec(100.0);

        let result = risk_manager.check_spread_gate(best_bid, bid_vol, best_ask, ask_vol);
        assert_eq!(result, false, "Zero bid volume should be rejected");

        let bid_vol = dec(100.0);
        let ask_vol = dec(0.0);

        let result = risk_manager.check_spread_gate(best_bid, bid_vol, best_ask, ask_vol);
        assert_eq!(result, false, "Zero ask volume should be rejected");
    }

    #[test]
    fn test_spread_exceeding_limit() {
        let config = create_test_risk_config(Some(10));
        let risk_manager = RiskManager::new(config, dec(1000.0));

        // mid = 100.0, spread = 0.5 -> 50 bps
        let best_bid = dec(99.75);
        let best_ask = dec(100.25);
        let bid_vol = dec(100.0);
        let ask_vol = dec(100.0);

        let result = risk_manager.check_spread_gate(best_bid, bid_vol, best_ask, ask_vol);
        assert_eq!(result, false, "Spread exceeding limit (50 vs 10) should be rejected");

        let config_wide = create_test_risk_config(Some(100));
        let risk_manager_wide = RiskManager::new(config_wide, dec(1000.0));

        let result = risk_manager_wide.check_spread_gate(best_bid, bid_vol, best_ask, ask_vol);
        assert_eq!(result, true, "Spread within limit should be accepted");
    }

    #[test]
    fn test_orders_limit_gate_disabled() {
        let mut config = create_test_risk_config(None);
        config.max_open_orders = None;
        let rm = RiskManager::new(config, dec(1000.0));

        // При max_open_orders = None гейт всегда открыт
        assert!(rm.check_orders_limit_gate(0));
        assert!(rm.check_orders_limit_gate(10));
        assert!(rm.check_orders_limit_gate(10_000));
    }

    #[test]
    fn test_orders_limit_gate_thresholds() {
        let mut config = create_test_risk_config(Some(10));
        config.max_open_orders = Some(10);
        let rm = RiskManager::new(config, dec(1000.0));

        // До 80% лимита — просто debug (гейт открыт)
        assert!(rm.check_orders_limit_gate(0));
        assert!(rm.check_orders_limit_gate(7)); // 70% от 10

        // На уровне 80% лимита — гейт все еще открыт
        assert!(rm.check_orders_limit_gate(8));

        // Чуть ниже лимита — тоже открыт
        assert!(rm.check_orders_limit_gate(9));

        // На лимите и выше — гейт должен закрываться
        assert!(!rm.check_orders_limit_gate(10));
        assert!(!rm.check_orders_limit_gate(11));
    }

    #[test]
    fn test_price_deviation_and_precision() {
        let mut config = create_test_risk_config(None);
        config.max_price_deviation_pct = Some(dec(0.02)); // 2%
        let rm = RiskManager::new(config, dec(1000.0));

        let mid = dec(10.0);
        let tick = dec(0.01);

        // 1. Правильная цена
        assert!(rm.validate_order_price(dec(10.1), mid, tick).unwrap());

        // 2. Неверная кратность (10.1005 не кратно 0.01)
        assert!(rm.validate_order_price(dec(10.1005), mid, tick).is_err());

        // 3. Превышение отклонения (10.3 > 10.2 (2%))
        assert_eq!(rm.validate_order_price(dec(10.3), mid, tick).unwrap(), false);
        
        // 4. Лимит отключен
        config.max_price_deviation_pct = None;
        let rm_no_limit = RiskManager::new(config, dec(1000.0));
        assert!(rm_no_limit.validate_order_price(dec(20.0), mid, tick).unwrap());
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::types::RiskConfig;
    use std::time::Duration;

    #[test]
    fn test_check_inactivity() {
        let mut config = RiskConfig::default();
        config.max_inactivity_ms = 100;
        let risk = RiskManager::new(config, dec(10000.0));

        // 1. None (еще нет данных)
        assert!(risk.check_inactivity(None));

        // 2. Some (активно)
        let last_update = Some(Instant::now());
        assert!(risk.check_inactivity(last_update));

        // 3. Таймаут
        std::thread::sleep(Duration::from_millis(150));
        assert!(!risk.check_inactivity(last_update));
    }
}

    #[test]
    fn test_lockout_disabled() {
        let config = RiskConfig::default();
        let risk = RiskManager::new(config, dec(10000.0));
        
        let mut state = RiskState::default();
        state.loss_streak = 5;
        state.last_loss_timestamp_ms = chrono::Utc::now().timestamp_millis();
        
        let mut bot_config = BotConfig::default();
        bot_config.lockout_period_sec = 0; // Отключено
        bot_config.lockout_streak_threshold = 2;
        
        // Lockout должен быть отключен
        assert!(!risk.is_in_lockout(&state, &bot_config));
    }

    #[test]
    fn test_lockout_below_threshold() {
        let config = RiskConfig::default();
        let risk = RiskManager::new(config, dec(10000.0));
        
        let mut state = RiskState::default();
        state.loss_streak = 1;
        state.last_loss_timestamp_ms = chrono::Utc::now().timestamp_millis();
        
        let mut bot_config = BotConfig::default();
        bot_config.lockout_period_sec = 3600;
        bot_config.lockout_streak_threshold = 2;
        
        // Серия убытков ниже порога
        assert!(!risk.is_in_lockout(&state, &bot_config));
    }

    #[test]
    fn test_lockout_active() {
        let config = RiskConfig::default();
        let risk = RiskManager::new(config, dec(10000.0));
        
        let mut state = RiskState::default();
        state.loss_streak = 3;
        state.last_loss_timestamp_ms = chrono::Utc::now().timestamp_millis();
        
        let mut bot_config = BotConfig::default();
        bot_config.lockout_period_sec = 10; // 10 секунд
        bot_config.lockout_streak_threshold = 2;
        
        // Lockout должен быть активен
        assert!(risk.is_in_lockout(&state, &bot_config));
    }

    #[test]
    fn test_lockout_expired() {
        let config = RiskConfig::default();
        let risk = RiskManager::new(config, dec(10000.0));
        
        let mut state = RiskState::default();
        state.loss_streak = 3;
        // Устанавливаем время убытка 2 секунды назад
        state.last_loss_timestamp_ms = chrono::Utc::now().timestamp_millis() - 2000;
        
        let mut bot_config = BotConfig::default();
        bot_config.lockout_period_sec = 1; // 1 секунда
        bot_config.lockout_streak_threshold = 2;
        
        // Lockout должен истечь
        assert!(!risk.is_in_lockout(&state, &bot_config));
    }

    #[test]
    fn test_lockout_saturating_sub() {
        let config = RiskConfig::default();
        let risk = RiskManager::new(config, dec(10000.0));
        
        let mut state = RiskState::default();
        state.loss_streak = 3;
        // Устанавливаем время убытка в будущем (симуляция глюка NTP)
        state.last_loss_timestamp_ms = chrono::Utc::now().timestamp_millis() + 10000;
        
        let mut bot_config = BotConfig::default();
        bot_config.lockout_period_sec = 10;
        bot_config.lockout_streak_threshold = 2;
        
        // Благодаря saturating_sub не должно быть паники, lockout не активен
        assert!(!risk.is_in_lockout(&state, &bot_config));
    }

    // ============================================================================
    // Мониторинг устаревших сигналов (Задача 169)
    // ============================================================================

    /// Регистрирует факт обработки сигнала (устаревший или свежий)
    /// 
    /// # Параметры
    /// - `is_stale`: true если сигнал был устаревшим
    pub fn register_signal_staleness(&mut self, is_stale: bool) {
        let now_ms = Utc::now().timestamp_millis();
        self.stale_signal_history.push_back((now_ms, is_stale));
        
        // Очищаем старые записи за пределами окна
        let cutoff = now_ms - self.stale_signal_window_ms;
        while let Some(&(ts, _)) = self.stale_signal_history.front() {
            if ts < cutoff {
                self.stale_signal_history.pop_front();
            } else {
                break;
            }
        }
    }

    /// Alias для register_signal_staleness (Задача 169)
    /// Регистрирует статус сигнала (свежий или устаревший) для отслеживания
    pub fn report_stale_signal(&mut self, is_stale: bool) {
        self.register_signal_staleness(is_stale);
    }

    /// Проверяет, не превышен ли порог устаревших сигналов (Circuit Breaker)
    /// 
    /// # Возвращает
    /// - `true` если более 50% сигналов за последние 5 минут были устаревшими
    pub fn check_stale_signal_circuit_breaker(&self) -> bool {
        if self.stale_signal_history.is_empty() {
            return false;
        }

        let total = self.stale_signal_history.len();
        let stale_count = self.stale_signal_history.iter()
            .filter(|(_, is_stale)| *is_stale)
            .count();

        let stale_ratio = stale_count as f64 / total as f64;
        
        if stale_ratio > 0.5 {
            error!(
                "CIRCUIT BREAKER: Stale signal ratio {:.1}% ({}/{}) exceeds 50% threshold",
                stale_ratio * 100.0,
                stale_count,
                total
            );
            return true;
        }

        if stale_ratio > 0.3 {
            warn!(
                "WARNING: High stale signal ratio {:.1}% ({}/{})",
                stale_ratio * 100.0,
                stale_count,
                total
            );
        }

        false
    }

    /// Возвращает текущую статистику устаревших сигналов
    /// 
    /// # Возвращает
    /// - (total_signals, stale_count, stale_ratio)
    pub fn get_staleness_stats(&self) -> (usize, usize, f64) {
        let total = self.stale_signal_history.len();
        if total == 0 {
            return (0, 0, 0.0);
        }

        let stale_count = self.stale_signal_history.iter()
            .filter(|(_, is_stale)| *is_stale)
            .count();

        let stale_ratio = stale_count as f64 / total as f64;
        
        (total, stale_count, stale_ratio)
    }

    // ============================================================================
    // Динамическое сокращение лимитов позиции (Задача 178)
    // ============================================================================

    /// Расчет масштабирующего фактора на основе просадки (Drawdown Scaling)
    /// 
    /// # Параметры
    /// - `current_drawdown_pct`: Текущая просадка в процентах (например, 8.0 для 8%)
    /// 
    /// # Возвращает
    /// - Масштабирующий фактор от min_scale_factor до 1.0
    fn calculate_drawdown_scaling(&self, current_drawdown_pct: f64) -> f64 {
        let start = self.config.drawdown_scaling_start_pct;
        let max_dd = self.config.drawdown_stop_pct;
        let min_factor = self.config.min_scale_factor;

        if current_drawdown_pct <= start {
            return 1.0; // Нет сокращения
        }

        if current_drawdown_pct >= max_dd {
            return min_factor; // Максимальное сокращение
        }

        // Линейное сокращение между start и max_dd
        let f_dd = 1.0 - (current_drawdown_pct - start) / (max_dd - start) * (1.0 - min_factor);
        f_dd.clamp(min_factor, 1.0)
    }

    /// Расчет масштабирующего фактора на основе волатильности (Volatility Scaling)
    /// 
    /// # Параметры
    /// - `current_vol`: Текущая волатильность
    /// - `hist_vol`: Историческая (медианная) волатильность
    /// 
    /// # Возвращает
    /// - Масштабирующий фактор от min_scale_factor до 1.0
    fn calculate_volatility_scaling(&self, current_vol: f64, hist_vol: f64) -> f64 {
        if hist_vol <= 0.0 || current_vol <= 0.0 {
            return 1.0; // Недостаточно данных
        }

        let vol_ratio = current_vol / hist_vol;
        let threshold = self.config.volatility_threshold;

        if vol_ratio > threshold {
            // Обратная волатильность (Inverse Vol)
            let f_vol = 1.0 / vol_ratio;
            f_vol.clamp(self.config.min_scale_factor, 1.0)
        } else {
            1.0 // Волатильность в норме
        }
    }

    /// Обновление масштаба лимита позиции с применением гистерезиса
    /// 
    /// # Параметры
    /// - `current_drawdown_pct`: Текущая просадка в процентах
    /// - `current_vol`: Текущая волатильность
    /// - `hist_vol`: Историческая волатильность
    pub fn update_position_scale(
        &mut self,
        current_drawdown_pct: f64,
        current_vol: f64,
        hist_vol: f64,
    ) {
        // Расчет масштабирующих факторов
        let f_dd = self.calculate_drawdown_scaling(current_drawdown_pct);
        let f_vol = self.calculate_volatility_scaling(current_vol, hist_vol);

        // Итоговый целевой масштаб (берем минимум из двух факторов)
        let target_scale = f_dd.min(f_vol).clamp(self.config.min_scale_factor, 1.0);

        // Применение гистерезиса (асимметричное обновление)
        if target_scale < self.current_scale {
            // Мгновенное сокращение при росте риска
            self.current_scale = target_scale;
            debug!(
                "Position limit reduced: {:.1}% (dd_factor={:.2}, vol_factor={:.2})",
                self.current_scale * 100.0,
                f_dd,
                f_vol
            );
        } else {
            // Плавное восстановление при снижении риска
            let new_scale = (self.current_scale + self.config.recovery_rate).min(target_scale);
            if new_scale > self.current_scale {
                debug!(
                    "Position limit recovering: {:.1}% -> {:.1}% (target={:.1}%)",
                    self.current_scale * 100.0,
                    new_scale * 100.0,
                    target_scale * 100.0
                );
            }
            self.current_scale = new_scale;
        }
    }

    /// Получение текущего эффективного лимита позиции
    /// 
    /// # Параметры
    /// - `base_max_pos`: Базовый максимальный размер позиции из конфига
    /// 
    /// # Возвращает
    /// - Эффективный лимит с учетом текущего масштаба
    pub fn get_effective_position_limit(&self, base_max_pos: Decimal) -> Decimal {
        let scale = Decimal::from_f64(self.current_scale).unwrap_or(Decimal::ONE);
        base_max_pos * scale
    }

    /// Получение текущего масштаба лимита (для тестирования и мониторинга)
    pub fn get_current_scale(&self) -> f64 {
        self.current_scale
    }

    /// Расчет текущей просадки в процентах от пика дневного эквити
    /// Используется для передачи в update_position_scale
    pub fn get_current_drawdown_pct(&self, current_pnl: Decimal) -> f64 {
        let current_equity = self.initial_equity + current_pnl;
        
        if self.peak_daily_equity <= Decimal::ZERO {
            return 0.0;
        }

        let drawdown_usd = self.peak_daily_equity - current_equity;
        let drawdown_pct = (drawdown_usd / self.peak_daily_equity) * Decimal::from(100);
        
        drawdown_pct.to_f64().unwrap_or(0.0).max(0.0)
    }



    /// Задача 184: Обновление конфигурации при SIGHUP
    /// Применяет новые параметры риска к risk manager и health monitor
    pub fn update_config(&mut self, config: RiskConfig) {
        tracing::info!("[Audit] Updating RiskManager config");
        self.config = config.clone();
        self.health_monitor.update_config(config);
    }
