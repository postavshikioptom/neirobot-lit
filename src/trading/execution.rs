use crate::ml::types::{Signal, InferenceOutput};
use crate::trading::order_manager::OrderManager;
use crate::trading::position_manager::{PositionManager, Position};
use crate::trading::rest_client::{BybitRestClient, BybitRestClientTrait};
use crate::risk::risk_manager::RiskManager;
use crate::trading::types::{OrderSide, MarketInfo, OrderUpdate, OrderState, OrderStatus, CreateOrderRequest, ExecutionAction, BybitOrderResult, RiskOrderInfo};
use crate::trading::types::BotState;
use crate::config::types::{BotConfig, ExchangeConfig};
use crate::utils::trade_logger::TradeRecord;
use tokio::sync::{mpsc, Mutex};
use std::sync::Arc;
use anyhow::Result;
use tracing::{info, warn, debug, error};
use rust_decimal::Decimal;
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};
use std::str::FromStr;
use std::time::{Duration, Instant};
use crate::utils::timestamp_ms;
use std::collections::VecDeque;
use std::path::PathBuf;
use crate::utils::helpers::RollingPriceStats;
use crate::data::types::{PublicTradeArc, OrderBookUpdateOwned};

/// Стратегия исполнения ордера (задача 206: Smart Order Routing)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExecutionStrategy {
    /// Пассивная стратегия (Limit ордер на лучшей цене)
    Passive,
    /// Агрессивная стратегия (Market/Cross ордер)
    Aggressive,
    /// TWAP Slicing - разбиение на части
    TwapSlice { slices: u32, interval_ms: u64 },
    /// Iceberg Order - скрытый ордер с частичным отображением объема
    Iceberg { 
        /// Неизменяемый целевой объем всего ордера
        total_size: f64, 
        /// Базовый процент для отображения (0.0 - 1.0)
        display_ratio: f64 
    },
}

/// Инструкция по исполнению ордера (без exchange_id для избежания аллокаций в Hot Path)
#[derive(Debug, Clone, Copy)]
pub struct ExecutionInstruction {
    /// Выбранная стратегия исполнения
    pub strategy: ExecutionStrategy,
    /// Цена исполнения
    pub price: Decimal,
    /// Количество для исполнения
    pub quantity: Decimal,
    /// Уровень агрессивности (0.0 - 1.0)
    pub urgency: f32,
}

pub struct ExecutionEngine {
    pub order_manager: OrderManager,
    pub position_manager: PositionManager,
    pub risk_manager: RiskManager,
    pub health_monitor: crate::risk::HealthMonitor, // Задача 179: Мониторинг здоровья системы
    pub symbol: String,
    pub bot_config: BotConfig,
    pub thresh_buy: f32,      // Задача 044: Порог вероятности для сигнала покупки
    pub thresh_sell: f32,     // Задача 044: Порог вероятности для сигнала продажи
    pub close_on_flat: bool, 
    pub market_info: MarketInfo,
    pub trade_tx: mpsc::Sender<TradeRecord>,
    pub execution_quality_tx: mpsc::Sender<crate::utils::logger::ExecutionQualityLog>, // Задача 202: Канал для логирования метрик
    pub market_impact_tx: Option<mpsc::Sender<crate::utils::logger::MarketImpactLog>>, // Задача 204: Канал для логирования влияния на цену
    pub last_probabilities: [f32; 3], // [Flat, Up, Down]
    pub last_signal_timestamp_ms: u64, // Время получения последнего сигнала (Задача 169)
    pub spread_ema: Option<Decimal>,  // EMA спреда для динамического фильтра
    pub price_stats: RollingPriceStats, // Статистика цен для VWAP/TWAP
    pub mid_history: VecDeque<f64>,
    pub sum_returns: f64,
    pub sum_returns_sq: f64,
    pub emergency_mode: bool,
    pub waiting_mode: bool,
    pub last_book_update: Option<Instant>,
    pub state_path: PathBuf,
    pub state: Arc<Mutex<BotState>>,
    pub state_persistence: crate::trading::StatePersistenceManager, // Задача 218: Менеджер персистентности состояния
    pub last_overtrade_warn_ts: i64,
    pub last_flip_ts: i64, // Задача 148: Время последнего переворота сигнала
    // Задача 149: Поля для механизма нарезки крупных ордеров
    pub pending_slice_qty: Option<Decimal>,     // Оставшийся объем для нарезки
    pub pending_slice_side: Option<OrderSide>,  // Сторона нарезки
    pub pending_slice_signal: Option<crate::ml::types::SignalWithTimestamp>, // Задача 201: Сигнал с временной меткой
    pub pending_slice_probs: Option<[f32; 3]>,  // Вероятности для нарезки
    pub last_signal_price: f64,                 // Задача 201: Mid price в момент генерации сигнала
    // Задача 161: Детектор режимов рынка
    pub regime_detector: Option<crate::trading::regime_detector::RegimeDetector>,
    // Задача 165: Детектор адверсариальной активности
    pub adversarial_detector: crate::risk::AdversarialDetector,
    // Задача 165: Накопление сделок для VPIN расчета
    pub pending_trades: Vec<PublicTradeArc>,
    // Задача 170: Текущая ставка финансирования и время следующего клиринга
    pub current_funding_rate: f64,
    pub next_funding_time: u64,
    // Задача 202: Очередь ордеров, ожидающих захвата цены через 100мс
    pub pending_price_checks: VecDeque<(String, u64)>, // (order_link_id, fill_time_ms)
    // Задача 149/168: Время последнего слайса для неблокирующего rate limiting
    pub last_slice_time: Instant,
}

impl ExecutionEngine {
    pub fn new(
        symbol: String, 
        risk_manager: RiskManager, 
        bot_config: BotConfig,
        market_info: MarketInfo,
        trade_tx: mpsc::Sender<TradeRecord>,
        state_path: PathBuf,
    ) -> Self {
        let close_on_flat = bot_config.close_on_flat;
        let thresh_buy = bot_config.threshold_buy;
        let thresh_sell = bot_config.threshold_sell;
        let leverage = bot_config.leverage;
        let risk_config = bot_config.risk.clone();
        let adversarial_config = bot_config.adversarial.clone();
        
        // Задача 202: Запускаем фоновый worker для логирования метрик исполнения
        let execution_quality_tx = crate::utils::logger::spawn_execution_quality_logger();
        
        // Задача 204: Инициализируем market_impact_logger если включено
        let market_impact_tx = if bot_config.enable_impact_logging {
            Some(crate::utils::logger::spawn_market_impact_logger())
        } else {
            None
        };

        // Задача 218: Инициализируем менеджер персистентности состояния
        let state_dir = state_path.parent().unwrap_or_else(|| std::path::Path::new("."));
        let state_persistence = crate::trading::StatePersistenceManager::new(
            state_dir,
            bot_config.max_state_backups,
        ).unwrap_or_else(|e| {
            tracing::error!("Failed to initialize state persistence manager: {}", e);
            // Fallback: создать менеджер с дефолтными параметрами
            crate::trading::StatePersistenceManager::new(state_dir, 3)
                .expect("Failed to create fallback state persistence manager")
        });
        
        Self {
            order_manager: OrderManager::new(),
            position_manager: PositionManager::new(symbol.clone(), leverage, market_info.qty_step),
            risk_manager,
            health_monitor: crate::risk::HealthMonitor::new(risk_config), // Задача 179
            symbol,
            close_on_flat,
            thresh_buy,
            thresh_sell,
            bot_config: bot_config.clone(),
            market_info,
            trade_tx,
            execution_quality_tx,
            market_impact_tx,
            last_probabilities: [0.0; 3],
            last_signal_timestamp_ms: 0,
            spread_ema: None,  // Инициализируется при первом вызове check_spread_barrier
            price_stats: RollingPriceStats::new(
                bot_config.stats_window_ms,
                bot_config.stats_max_trades
            ),
            mid_history: VecDeque::new(),
            sum_returns: 0.0,
            sum_returns_sq: 0.0,
            emergency_mode: false,
            waiting_mode: false,
            last_book_update: None,
            state_path: state_path.clone(),
            state: Arc::new(Mutex::new(crate::trading::BotState::default())),
            state_persistence,
            last_overtrade_warn_ts: 0,
            last_flip_ts: 0,
            pending_slice_qty: None,
            pending_slice_side: None,
            pending_slice_signal: None,
            pending_slice_probs: None,
            last_signal_price: 0.0,
            regime_detector: None,
            adversarial_detector: crate::risk::AdversarialDetector::new(adversarial_config),
            pending_trades: Vec::new(),
            current_funding_rate: 0.0,
            next_funding_time: 0,
            pending_price_checks: VecDeque::new(), // Задача 202: Инициализация очереди для захвата цены через 100мс
            last_slice_time: Instant::now(),
        }
    }

    /// Обработка публичных сделок для обновления статистики VWAP/TWAP и VPIN
    #[inline(always)]
    pub fn on_public_trade(&mut self, trade: PublicTradeArc) {
        self.price_stats.update(trade.clone());
        // Задача 165: Накапливаем сделку для VPIN расчета
        self.pending_trades.push(trade);
    }

    /// Обработка обновлений системных метрик (Задача 225)
    /// Проверяет ресурсы и автоматически снижает риски при перегрузке
    pub fn on_system_metrics(&mut self, metrics: crate::monitoring::resource_profiler::SystemMetricsUpdate) {
        self.risk_manager.check_system_resources(&metrics, &self.bot_config);
    }
    
    /// Установка sender для логирования влияния на цену (Задача 204)
    pub fn set_market_impact_logger(&mut self, tx: mpsc::Sender<crate::utils::logger::MarketImpactLog>) {
        self.market_impact_tx = Some(tx.clone());
        self.order_manager.set_market_impact_logger(tx);
    }
    
    /// Обновление текущей ставки финансирования и времени следующего клиринга (Задача 170)
    pub fn update_funding_info(&mut self, funding_rate: f64, next_funding_time: u64, mark_price: Decimal) {
        // Детектирование settlement: если next_funding_time изменился и стал больше старого значения,
        // значит произошел клиринг и нужно применить фандинг
        if self.next_funding_time > 0 && next_funding_time > self.next_funding_time {
            // Settlement occurred! Применяем фандинг используя СТАВКУ из ПРЕДЫДУЩЕГО периода
            self.position_manager.apply_funding(self.current_funding_rate, mark_price);
            debug!("[{}] Funding settlement detected! Applied funding rate: {:.6} at mark_price: {}", 
                self.symbol, self.current_funding_rate, mark_price);
        }
        
        // Обновляем сохраненные значения
        self.current_funding_rate = funding_rate;
        self.next_funding_time = next_funding_time;
        debug!("[{}] Updated funding rate: {:.6} ({}%), next settlement: {}", 
            self.symbol, funding_rate, (funding_rate * 100.0), next_funding_time);
    }
    
    /// Установка детектора режимов рынка (Задача 161)
    pub fn set_regime_detector(&mut self, detector: crate::trading::regime_detector::RegimeDetector) {
        self.regime_detector = Some(detector);
    }
    
    /// Получение текущего режима рынка (Задача 161)
    pub fn get_current_regime(&self) -> crate::config::types::RegimeId {
        self.regime_detector
            .as_ref()
            .map(|d| d.current_regime())
            .unwrap_or(crate::config::types::RegimeId::Unknown)
    }
    
    /// Получение порогов для конкретного режима рынка (Задача 161)
    /// Возвращает (buy_threshold, sell_threshold, min_confidence)
    /// 
    /// ИСПОЛЬЗОВАНИЕ: Этот метод должен вызываться в логике принятия торговых решений
    /// для получения динамических порогов на основе текущего режима рынка.
    /// Пример: let (buy_th, sell_th, min_conf) = self.get_thresholds_for_regime(current_regime);
    pub fn get_thresholds_for_regime(&self, regime: crate::config::types::RegimeId) -> (f32, f32, f32) {
        // Ищем переопределение для данного режима
        for override_cfg in &self.bot_config.regime_overrides {
            if override_cfg.regime == regime {
                debug!(
                    "[{}] Using regime-specific thresholds for {:?}: buy={}, sell={}, min_conf={}",
                    self.symbol, regime, override_cfg.buy_threshold, 
                    override_cfg.sell_threshold, override_cfg.min_confidence
                );
                return (
                    override_cfg.buy_threshold,
                    override_cfg.sell_threshold,
                    override_cfg.min_confidence,
                );
            }
        }
        
        // Если переопределения нет - используем базовые значения
        (
            self.bot_config.threshold_buy,
            self.bot_config.threshold_sell,
            self.bot_config.signal_min_confidence as f32,
        )
    }

    /// Выбор оптимальной стратегии исполнения на основе сигнала и состояния стакана (Задача 206)
    /// 
    /// Логика:
    /// - Если signal.strength > config.critical_signal → Aggressive (Market/Cross)
    /// - Если order_size > level_total_vol * config.max_size_ratio → TwapSlice
    /// - В остальных случаях → Passive (Limit)
    pub fn select_strategy(
        &self,
        _signal: &Signal,
        side: OrderSide,
        strength: f32,
        order_book: &crate::data::orderbook::OrderBook,
        order_size: Decimal,
    ) -> ExecutionInstruction {
        let sor_config = &self.bot_config.sor;
        let (best_bid, best_ask) = order_book.get_best_bid_ask();
        let best_bid = Decimal::from_f64(best_bid).unwrap_or(Decimal::ZERO);
        let best_ask = Decimal::from_f64(best_ask).unwrap_or(Decimal::ZERO);
        let mid_price = (best_bid + best_ask) / Decimal::from(2);
        
        // Получаем объем на лучшем уровне
        let level_volume = if side == OrderSide::Buy {
            let vol = order_book.get_volume_at_best(crate::data::types::Side::Buy);
            Decimal::from_f64(vol).unwrap_or(Decimal::ZERO)
        } else {
            let vol = order_book.get_volume_at_best(crate::data::types::Side::Sell);
            Decimal::from_f64(vol).unwrap_or(Decimal::ZERO)
        };
        
        // Определяем стратегию
        let strategy = if strength > sor_config.critical_signal {
            // Агрессивная стратегия при сильном сигнале
            ExecutionStrategy::Aggressive
        } else if level_volume > Decimal::ZERO {
            let size_ratio = (order_size / level_volume).to_f64().unwrap_or(1.0);
            
            // Iceberg для очень крупных ордеров (>5x объема уровня)
            if size_ratio > 5.0 {
                let total_size = order_size.to_f64().unwrap_or(0.0);
                let display_ratio = (1.0 / size_ratio).clamp(0.1, 0.3); // 10-30% от total
                ExecutionStrategy::Iceberg {
                    total_size,
                    display_ratio,
                }
            }
            // TWAP Slicing при превышении объема уровня (но не слишком большом)
            else if size_ratio > sor_config.max_size_ratio as f64 {
                let slices = (size_ratio.ceil() as u32).max(2);
                ExecutionStrategy::TwapSlice {
                    slices,
                    interval_ms: sor_config.slice_interval_ms,
                }
            } else {
                // Пассивная стратегия
                ExecutionStrategy::Passive
            }
        } else {
            // Пассивная стратегия по умолчанию
            ExecutionStrategy::Passive
        };
        
        // Определяем цену в зависимости от стратегии
        let price = match strategy {
            ExecutionStrategy::Passive | ExecutionStrategy::Iceberg { .. } => {
                // Для Passive и Iceberg используем лучшую цену на противоположной стороне
                if side == OrderSide::Buy {
                    best_ask
                } else {
                    best_bid
                }
            }
            ExecutionStrategy::Aggressive | ExecutionStrategy::TwapSlice { .. } => {
                // Для Aggressive и TwapSlice используем Mid Price
                mid_price
            }
        };
        
        ExecutionInstruction {
            strategy,
            price,
            quantity: order_size,
            urgency: sor_config.default_urgency,
        }
    }

    /// Обновление времени активности стакана (Задача 113)
    pub fn poke_book_activity(&mut self) {
        self.last_book_update = Some(Instant::now());
        if self.waiting_mode {
            info!("[{}] Data stream resumed. Resuming trading mode.", self.symbol);
            self.waiting_mode = false;
        }
    }

    /// Переход в безопасный режим при неактивности (Задача 113)
    pub async fn handle_inactivity_trigger(
        &mut self,
        rest_client: &impl BybitRestClientTrait,
        exchange_config: &ExchangeConfig,
    ) -> Result<()> {
        if self.waiting_mode {
            return Ok(());
        }

        error!("[{}] INACTIVITY TRIGGERED: Quotes frozen. Entering safety mode.", self.symbol);
        self.waiting_mode = true;

        // 1. Отмена всех ордеров (ретраи как в 109)
        if let Err(e) = self.order_manager.cancel_all_orders(rest_client, &mut self.risk_manager, &self.bot_config, exchange_config).await {
            error!("[{}] Failed to cancel orders during inactivity trigger: {}", self.symbol, e);
        }

        // 2. Опциональное закрытие позиции
        if self.bot_config.close_position_on_inactivity {
            warn!("[{}] Closing position due to inactivity config.", self.symbol);
            if let Err(e) = self.order_manager.execute_emergency_close(rest_client, &mut self.risk_manager, &self.position_manager, &self.bot_config, exchange_config).await {
                error!("[{}] Failed to close position during inactivity trigger: {}", self.symbol, e);
            }
        }

        Ok(())
    }

    /// Расчет мультипликатора силы сигнала (Задача 110)
    #[inline(always)]
    fn get_signal_multiplier(&self, signal: Signal, probs: &[f32; 3]) -> f64 {
        if signal.is_flat() {
            return 0.0;
        }

        // Находим уверенность (максимальная вероятность среди Up/Down)
        let confidence = probs[1].max(probs[2]) as f64;

        if confidence < self.bot_config.signal_min_confidence {
            return 0.0;
        }

        // Линейная интерполяция между min_confidence и full_confidence
        let range = self.bot_config.signal_full_confidence - self.bot_config.signal_min_confidence;
        let t = if range > 0.0 {
            ((confidence - self.bot_config.signal_min_confidence) / range).clamp(0.0, 1.0)
        } else {
            1.0
        };

        self.bot_config.signal_size_mult_min + t * (self.bot_config.signal_size_mult_max - self.bot_config.signal_size_mult_min)
    }

    /// Расчет объема ордера на основе баланса, цены и рисков
    #[inline(always)]
    pub fn calculate_order_size(
        &self, 
        available_balance: Decimal, 
        current_price: Decimal,
        signal: Signal,
        probs: &[f32; 3],
    ) -> Decimal {
        let fee_rate = self.bot_config.taker_fee_bps / Decimal::from(10000);
        let buffer = self.bot_config.buffer_pct;
        let leverage = self.bot_config.leverage;
        
        // effective_balance = available_balance / (1.0 + fee_rate + buffer)
        let divisor = Decimal::ONE + fee_rate + buffer;
        let effective_balance = available_balance / divisor;

        // target_qty = min(effective_balance * leverage, bot_config.max_position_size) / current_price
        let mut target_val = effective_balance * leverage;
        if let Some(max_pos_dec) = self.bot_config.max_position_size {
            if target_val > max_pos_dec {
                target_val = max_pos_dec;
            }
        }
        
        let target_qty = target_val / current_price;

        // Задача 137: Используем clamp_qty для округления и валидации
        let base_qty = crate::utils::helpers::clamp_qty(
            target_qty.to_f64().unwrap_or(0.0),
            self.market_info.min_order_qty.to_f64().unwrap_or(0.01),
            self.market_info.max_order_qty.to_f64().unwrap_or(1000000.0),
            self.market_info.qty_step.to_f64().unwrap_or(0.01),
        );
        let base_qty = Decimal::from_f64(base_qty).unwrap_or(Decimal::ZERO);

        // Валидация по лимитам
        if base_qty < self.market_info.min_order_qty {
            return Decimal::ZERO;
        }

        // Применяем скейлинг (Волатильность + Сигнал)
        let mut scaled_qty = self.calculate_combined_scaled_size(base_qty, &signal, probs);
        
        // Задача 231: Применяем margin_multiplier для снижения размера позиции после ошибки маржи
        let margin_multiplier = self.risk_manager.get_margin_multiplier();
        if margin_multiplier < 1.0 {
            scaled_qty = scaled_qty * Decimal::from_f64(margin_multiplier).unwrap_or(Decimal::ONE);
            debug!(
                "Order size reduced by margin_multiplier ({:.1}%): {} -> {}",
                margin_multiplier * 100.0,
                self.calculate_combined_scaled_size(base_qty, &signal, probs),
                scaled_qty
            );
        }

        // Задача 065: Проверка риск-менеджера
        if !self.risk_manager.can_open_position(scaled_qty) {
            debug!("Order size rejected by risk manager: {}", scaled_qty);
            return Decimal::ZERO;
        }

        info!(
            "Order size calculated: {} (original target: {}, balance used: {})", 
            scaled_qty, target_qty, effective_balance
        );

        scaled_qty
    }

    /// Расчет скейлированного размера на основе волатильности и силы сигнала (Задача 110)
    fn calculate_combined_scaled_size(&self, base_size: Decimal, signal: &Signal, probs: &[f32; 3]) -> Decimal {
        // 1. Мультипликатор волатильности (Задача 105)
        let current_vol = self.get_current_vol();
        let effective_vol = if self.mid_history.len() < self.bot_config.volatility_window {
            self.bot_config.volatility_default
        } else if current_vol < 1e-9 {
            1e-9
        } else {
            current_vol
        };

        let vol_mult = (self.bot_config.volatility_target_bps / effective_vol)
            .clamp(self.bot_config.size_min_multiplier, self.bot_config.size_max_multiplier);

        // 2. Мультипликатор силы сигнала (Задача 110)
        let signal_mult = self.get_signal_multiplier(signal.clone(), probs);

        if signal_mult <= 0.0 {
            return Decimal::ZERO;
        }

        // 3. Комбинированный мультипликатор с ограничением
        let total_mult = (vol_mult * signal_mult).min(self.bot_config.total_size_mult_max);

        let multiplier_dec = Decimal::from_f64(total_mult).unwrap_or(Decimal::ONE);
        
        // Задача 137: Используем clamp_qty для округления и валидации
        let scaled_size = base_size * multiplier_dec;
        let rounded_size = crate::utils::helpers::clamp_qty(
            scaled_size.to_f64().unwrap_or(0.0),
            self.market_info.min_order_qty.to_f64().unwrap_or(0.01),
            self.market_info.max_order_qty.to_f64().unwrap_or(1000000.0),
            self.market_info.qty_step.to_f64().unwrap_or(0.01),
        );
        let rounded_size = Decimal::from_f64(rounded_size).unwrap_or(Decimal::ZERO);
        
        // Валидация по лимитам биржи
        if rounded_size < self.market_info.min_order_qty {
            return Decimal::ZERO;
        }
        
        if rounded_size > self.market_info.max_order_qty {
            return self.market_info.max_order_qty;
        }
        
        debug!(
            "[{}] Scaling: vol_mult={:.3}, signal_mult={:.3}, total_mult={:.3}, base={}, scaled={}",
            self.symbol, vol_mult, signal_mult, total_mult, base_size, rounded_size
        );
        
        rounded_size
    }

    /// Обновление буфера mid_price и расчет волатильности
    #[inline(always)]
    fn update_volatility(&mut self, new_mid: f64) {
        let window = self.bot_config.volatility_window;
        
        if let Some(&last_mid) = self.mid_history.back() {
            if last_mid > 0.0 {
                let ret = (new_mid - last_mid) / last_mid;
                
                // Обновляем суммы для O(1) расчета дисперсии
                self.sum_returns += ret;
                self.sum_returns_sq += ret * ret;
            }
        }
        
        self.mid_history.push_back(new_mid);

        // Удаляем старые значения, если превышен размер окна
        if self.mid_history.len() > window + 1 {
            if let Some(old_mid) = self.mid_history.pop_front() {
                // Пересчитываем returns для удаленного элемента
                if let Some(&next_mid) = self.mid_history.front() {
                    if old_mid > 0.0 {
                        let old_ret = (next_mid - old_mid) / old_mid;
                        self.sum_returns -= old_ret;
                        self.sum_returns_sq -= old_ret * old_ret;
                    }
                }
            }
        }
    }

    /// Расчет текущей волатильности в базисных пунктах (bps)
    #[inline(always)]
    fn get_current_vol(&self) -> f64 {
        let n = self.mid_history.len();
        
        // Нужно минимум 2 точки для расчета returns
        if n < 2 {
            return self.bot_config.volatility_default;
        }

        let n_returns = (n - 1) as f64;
        
        if n_returns < 1.0 {
            return self.bot_config.volatility_default;
        }

        // Расчет среднего и дисперсии
        let mean = self.sum_returns / n_returns;
        let variance = (self.sum_returns_sq / n_returns) - (mean * mean);
        
        // Защита от отрицательной дисперсии (из-за ошибок округления)
        let variance = if variance < 0.0 { 0.0 } else { variance };
        
        let std_dev = variance.sqrt();
        
        // Конвертируем в базисные пункты (1 bps = 0.01%)
        let vol_bps = std_dev * 10000.0;
        
        vol_bps
    }

    /// Проверка спреда с двойным барьером (Static Cap + Dynamic Multiplier)
    #[inline(always)]
    fn check_spread_barrier(&mut self, best_bid: Decimal, best_ask: Decimal) -> bool {
        // Проверка валидности bid/ask
        if best_bid.is_zero() || best_ask.is_zero() {
            warn!("[{}] Spread barrier: invalid bid/ask (bid={}, ask={})", self.symbol, best_bid, best_ask);
            return false;
        }

        let mid = (best_bid + best_ask) / Decimal::from(2);
        let current_spread_pct = (best_ask - best_bid) / mid;
        let current_bps = (current_spread_pct * Decimal::from(10000))
            .to_u32()
            .unwrap_or(200);

        // 1. Проверка жесткого лимита (Static Cap)
        let static_limit = self.bot_config.max_spread_static_bps.unwrap_or(200);
        if current_bps > static_limit {
            warn!(
                "[{}] Spread blocked: {} bps > {} bps (Static Cap)",
                self.symbol, current_bps, static_limit
            );
            return false;
        }

        // 2. Инициализация EMA при первом вызове
        if self.spread_ema.is_none() {
            self.spread_ema = Some(current_spread_pct);
            info!(
                "[{}] Spread EMA initialized: {:.4}%",
                self.symbol,
                current_spread_pct * Decimal::from(100)
            );
            return true; // Пропускаем первый сигнал
        }

        // 3. Проверка динамического лимита (5x от нормы)
        let spread_ema = self.spread_ema.unwrap();
        let multiplier_f32 = self.bot_config.spread_multiplier.unwrap_or(5.0);
        let multiplier = Decimal::from_f32(multiplier_f32).unwrap_or(Decimal::from(5));
        let dynamic_limit_pct = spread_ema * multiplier;

        if current_spread_pct > dynamic_limit_pct {
            warn!(
                "[{}] Spread blocked: {:.4}% > {:.4}% ({}x Norm)",
                self.symbol,
                current_spread_pct * Decimal::from(100),
                dynamic_limit_pct * Decimal::from(100),
                multiplier_f32
            );
            return false;
        }

        // 4. Обновляем EMA (коэффициент 0.01 для плавности)
        let alpha = Decimal::from_str("0.01").unwrap();
        self.spread_ema = Some(spread_ema * (Decimal::ONE - alpha) + current_spread_pct * alpha);

        true
    }

    /// Синхронизация состояния с биржей при старте (Reconciliation)
    /// Задача 190: Переписано для использования новой системы persistence
    pub async fn sync_state(
        &mut self,
        rest_client: &BybitRestClient,
        exchange_config: &ExchangeConfig,
    ) -> Result<()> {
        info!("[{}] Starting state synchronization with exchange...", self.symbol);
        
        // Получаем состояние с биржи
        let pos_idx = self.bot_config.position_idx;
        let ex_pos = rest_client.get_position(
            &exchange_config.bybit.category,
            &self.symbol,
            pos_idx
        ).await?;

        let ex_size = ex_pos.as_ref().map(|p| p.size).unwrap_or(Decimal::ZERO);
        let ex_side = ex_pos.as_ref().map(|p| p.side.as_str()).unwrap_or("");
        
        // Bybit возвращает положительный size, знак определяется по side
        let signed_ex_size = if ex_side == "Sell" { -ex_size } else { ex_size };
        let ex_avg_price = ex_pos.as_ref().map(|p| p.avg_price).unwrap_or(Decimal::ZERO);

        // Сверяем с локальным состоянием
        let local_size = self.position_manager.get_position().qty;
        let diff = (local_size - signed_ex_size).abs();
        
        // Используем desync_tolerance_pct из конфига
        let tolerance_pct = Decimal::from_f64(self.bot_config.desync_tolerance_pct).unwrap_or(Decimal::ZERO);
        
        // Проверка расхождения
        let is_desynced = if !signed_ex_size.is_zero() {
            (diff / signed_ex_size.abs()) > tolerance_pct
        } else {
            !local_size.is_zero() // Если на бирже 0, а у нас нет — это расхождение
        };

        if is_desynced {
            error!(
                "[{}] CRITICAL DESYNC: Local {} vs Exchange {}. diff={}. STOPPING.", 
                self.symbol, local_size, signed_ex_size, diff
            );
            self.emergency_mode = true;
            return Ok(());
        } else {
            info!(
                "[{}] State synced successfully. Position: {} (Exchange: {})",
                self.symbol, local_size, signed_ex_size
            );
            
            // Обновляем PositionManager данными биржи
            self.position_manager.set_position(signed_ex_size, ex_avg_price);
            
            // Сохранить состояние после синхронизации позиции (Задача 107)
            if let Err(e) = self.save_current_state().await {
                error!("[{}] Failed to save state after sync: {}", self.symbol, e);
            }
        }

        Ok(())
    }

    /// Сохранение текущего состояния на диск
    /// Задача 218: Загрузить состояние при запуске
    pub async fn load_state_on_startup(
        &mut self,
        rest_client: &impl BybitRestClientTrait,
        exchange_config: &ExchangeConfig,
    ) -> Result<()> {
        match self.state_persistence.load_state() {
            Ok(bot_state) => {
                tracing::info!(
                    "[{}] State loaded on startup: position={}, pnl={}, active_orders={}",
                    self.symbol,
                    bot_state.data.position,
                    bot_state.data.pnl,
                    bot_state.data.active_order_ids.len()
                );
                
                // Обновить position_manager с загруженным состоянием
                let position = rust_decimal::Decimal::from_f64(bot_state.data.position)
                    .unwrap_or(rust_decimal::Decimal::ZERO);
                self.position_manager.set_position(position, rust_decimal::Decimal::ZERO);
                
                Ok(())
            }
            Err(e) => {
                if e.to_string().contains("All state files corrupted") {
                    tracing::error!(
                        "[{}] CRITICAL: All state files are corrupted! Force syncing with exchange (Task 066)...",
                        self.symbol
                    );
                    
                    // Задача 066/120: Принудительная синхронизация с биржей при потере состояния
                    self.perform_reconciliation(rest_client, exchange_config).await?;
                    
                    tracing::info!("[{}] State successfully recovered from exchange via REST.", self.symbol);
                } else {
                    tracing::warn!(
                        "[{}] Failed to load state on startup: {}. Starting with empty state.",
                        self.symbol,
                        e
                    );
                }
                Ok(())
            }
        }
    }

    /// Задача 218: Сохранение текущего состояния на диск с использованием StatePersistenceManager
    pub async fn save_current_state(&mut self) -> Result<()> {
        // Получить текущее состояние из position_manager и order_manager
        let position = self.position_manager.get_position();
        let pnl = self.position_manager.get_position().realized_pnl;
        let active_order_ids: Vec<String> = self.order_manager
            .get_active_orders().await
            .values()
            .map(|o| o.link_id.clone())
            .collect();

        // Создать BotStateData
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("symbol".to_string(), self.symbol.clone());
        metadata.insert("timestamp".to_string(), 
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs()
                .to_string()
        );

        let state_data = crate::trading::BotStateData {
            position: position.qty.to_f64().unwrap_or(0.0),
            pnl: pnl.to_f64().unwrap_or(0.0),
            active_order_ids,
            metadata,
        };

        // Создать PersistentState с чексуммой
        let bot_state = crate::trading::PersistentState::new(state_data)?;

        // Сохранить состояние через StatePersistenceManager
        self.state_persistence.save_state(&bot_state)?;

        tracing::info!(
            "[{}] State saved: position={}, pnl={}, active_orders={}",
            self.symbol,
            bot_state.data.position,
            bot_state.data.pnl,
            bot_state.data.active_order_ids.len()
        );

        Ok(())
    }

    /// Экстренное закрытие позиций (Panic Exit) с таймаутом и ретраями
    pub async fn emergency_market_close(
        &mut self,
        rest_client: &impl BybitRestClientTrait,
        exchange_config: &ExchangeConfig,
    ) -> Result<()> {
        let timeout_ms = self.bot_config.emergency_timeout_ms;
        tokio::time::timeout(
            Duration::from_millis(timeout_ms),
            self.perform_panic_exit(rest_client, exchange_config)
        ).await
        .map_err(|_| anyhow::anyhow!("Emergency close timed out after {}ms!", timeout_ms))?
    }

    async fn perform_panic_exit(
        &mut self,
        rest_client: &impl BybitRestClientTrait,
        exchange_config: &ExchangeConfig,
    ) -> Result<()> {
        warn!("PANIC EXIT STARTED for {}", self.symbol);
        self.emergency_mode = true; // Блокируем новые сигналы

        // 1. Отмена ордеров с ретраями (max 3 попытки)
        let mut attempts = 0;
        while attempts < 3 {
            match self.order_manager.cancel_all_orders(rest_client, &mut self.risk_manager, &self.bot_config, exchange_config).await {
                Ok(_) => {
                    info!("All orders cancelled");
                    break;
                }
                Err(e) if attempts < 2 => {
                    attempts += 1;
                    let delay = 100 * (2_u64.pow(attempts as u32)); // 200ms, 400ms
                    warn!("Cancel orders failed, retrying in {}ms: {}", delay, e);
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                }
                Err(e) => {
                    error!("Failed to cancel orders after 3 attempts: {}. Continuing to market close.", e);
                    break; 
                }
            }
        }

        // 2. Закрытие позиции
        if let Err(e) = self.order_manager.execute_emergency_close(rest_client, &mut self.risk_manager, &self.position_manager, &self.bot_config, exchange_config).await {
            error!("CRITICAL: Failed to place market close order: {}", e);
            // Даже если не удалось выставить ордер (например, позиции нет), продолжаем к финализации стейта
        } else {
            info!("Emergency market order placed or no position to close");
        }

        // 3. Финализация стейта
        {
            let mut state_guard = self.state.lock().await;
            state_guard.position_size = Decimal::ZERO;
            state_guard.active_orders.clear();
            // Сохранить состояние после экстренного закрытия (Задача 107)
            drop(state_guard); // Освобождаем блокировку перед вызовом save_current_state
        }
        
        if let Err(e) = self.save_current_state().await {
            error!("[{}] Failed to save state after panic exit: {}", self.symbol, e);
        }
        
        Ok(())
    }

    /// Детальная сверка состояния с биржей (Reconciliation) (Задача 120)
    /// Использует блокировку для защиты от race conditions с WS-апдейтами
    pub async fn perform_reconciliation(
        &mut self,
        rest_client: &impl BybitRestClientTrait,
        exchange_config: &ExchangeConfig,
    ) -> Result<()> {
        info!("[{}] Starting reconciliation...", self.symbol);

        // Блокируем стейт, чтобы WS-апдейты не мешали сверке
        let mut state_guard = self.state.lock().await;

        // 1. Fetch fresh data from REST
        let category = &exchange_config.bybit.category;
        let pos_idx = self.bot_config.position_idx;
        
        let ex_position_info = rest_client.get_position(category, &self.symbol, pos_idx).await?;
        let ex_orders = rest_client.get_open_orders(category, &self.symbol).await?;

        // 2. Конвертируем биржевую позицию в локальный формат Position
        let ex_position = if let Some(pos_info) = ex_position_info {
            let qty = if pos_info.side == "Buy" { pos_info.size } else { -pos_info.size };
            Position {
                symbol: self.symbol.clone(),
                qty,
                avg_price: pos_info.avg_price,
                realized_pnl: Decimal::ZERO,
                unrealized_pnl: pos_info.unrealised_pnl,
                unrealized_pnl_pct: Decimal::ZERO,
                mark_pnl: Decimal::ZERO,
                leverage: pos_info.leverage,
                updated_at: crate::utils::timestamp_ms(),
                opened_at: None,
                completed_tp_stages: Default::default(),
                initial_size: 0.0,
                side: crate::trading::types::OrderSide::Buy,
                extreme_water_mark: 0.0,
                current_stop_loss: 0.0,
                tsl_active: false,
                accumulated_funding: Decimal::ZERO,
                total_rebates: Decimal::ZERO,
            }
        } else {
            Position {
                symbol: self.symbol.clone(),
                qty: Decimal::ZERO,
                avg_price: Decimal::ZERO,
                realized_pnl: Decimal::ZERO,
                unrealized_pnl: Decimal::ZERO,
                unrealized_pnl_pct: Decimal::ZERO,
                mark_pnl: Decimal::ZERO,
                leverage: Decimal::ONE,
                updated_at: crate::utils::timestamp_ms(),
                opened_at: None,
                completed_tp_stages: Default::default(),
                initial_size: 0.0,
                side: crate::trading::types::OrderSide::Buy,
                extreme_water_mark: 0.0,
                current_stop_loss: 0.0,
                tsl_active: false,
                accumulated_funding: Decimal::ZERO,
                total_rebates: Decimal::ZERO,
            }
        };

        // 3. Получаем локальное состояние
        let local_position = self.position_manager.get_position().clone();
        let local_orders = &state_guard.active_orders;

        // 4. Преобразуем RiskOrderInfo в OrderInfo
        let local_orders_converted: std::collections::HashMap<String, crate::trading::types::OrderInfo> = 
            local_orders.iter().map(|(k, v)| {
                (k.clone(), crate::trading::types::OrderInfo {
                    side: v.side,
                    price: v.price,
                    qty: v.qty,
                    status: match v.state {
                        crate::trading::types::OrderState::Created => crate::trading::types::OrderStatus::Created,
                        crate::trading::types::OrderState::PendingNew => crate::trading::types::OrderStatus::New,
                        crate::trading::types::OrderState::Active => crate::trading::types::OrderStatus::New,
                        crate::trading::types::OrderState::PartiallyFilled => crate::trading::types::OrderStatus::PartiallyFilled,
                        crate::trading::types::OrderState::Filled => crate::trading::types::OrderStatus::Filled,
                        crate::trading::types::OrderState::PendingCancel => crate::trading::types::OrderStatus::Cancelled,
                        crate::trading::types::OrderState::Cancelled => crate::trading::types::OrderStatus::Cancelled,
                        crate::trading::types::OrderState::Expired => crate::trading::types::OrderStatus::Expired,
                        crate::trading::types::OrderState::Rejected(_) => crate::trading::types::OrderStatus::Rejected,
                    },
                    chase_count: 0,
                    last_chase_ts: 0,
                    link_id: v.link_id.clone(),
                })
            }).collect();

        // 5. Проверка консистентности
        let is_consistent = self.risk_manager.verify_consistency(
            &local_position,
            &ex_position,
            &local_orders_converted,
            &ex_orders,
            self.bot_config.price_desync_threshold,
        );

        if !is_consistent {
            if self.bot_config.sync_on_desync {
                info!("[{}] Desync detected. Force syncing local state...", self.symbol);
                
                // Синхронизация позиции
                self.position_manager.set_position(ex_position.qty, ex_position.avg_price);
                state_guard.position_size = ex_position.qty;
                state_guard.avg_price = ex_position.avg_price;

                // Синхронизация ордеров: отменяем локальные "призраки" и принимаем биржевые
                state_guard.active_orders.clear();
                for order in ex_orders {
                    // Преобразуем OrderInfo в RiskOrderInfo
                    let state = match order.status {
                        OrderStatus::New => OrderState::Active,
                        OrderStatus::PartiallyFilled => OrderState::PartiallyFilled,
                        OrderStatus::Filled => OrderState::Filled,
                        OrderStatus::Cancelled => OrderState::Cancelled,
                        OrderStatus::Rejected => OrderState::Rejected("Rejected".to_string()),
                        OrderStatus::Expired => OrderState::Expired,
                        OrderStatus::Created => OrderState::Created,
                        OrderStatus::PostOnlyRejected => OrderState::Rejected("PostOnlyRejected".to_string()),
                        OrderStatus::Untracked => OrderState::Active,
                    };
                    
                    let risk_order = RiskOrderInfo {
                        side: order.side,
                        price: order.price,
                        qty: order.qty,
                        state,
                        link_id: order.link_id.clone(),
                    };
                    
                    // Используем link_id из биржевого ордера
                    if let Some(link_id) = risk_order.link_id.clone() {
                        state_guard.active_orders.insert(link_id, risk_order);
                    } else {
                        // Если link_id отсутствует, генерируем его на основе параметров
                        let link_id = format!("{:?}_{}_{}",  risk_order.side, risk_order.price, risk_order.qty);
                        state_guard.active_orders.insert(link_id, risk_order);
                    }
                }
                
                // Сохранить состояние после синхронизации (Задача 107)
                drop(state_guard); // Освобождаем блокировку перед вызовом save_current_state
                if let Err(e) = self.save_current_state().await {
                    error!("[{}] Failed to save state after reconciliation: {}", self.symbol, e);
                }
                info!("[{}] State synchronized successfully", self.symbol);
            } else {
                self.emergency_mode = true;
                error!("[{}] Critical desync! Manual intervention required.", self.symbol);
                return Err(anyhow::anyhow!("Critical desync! Manual intervention required."));
            }
        } else {
            info!("[{}] Reconciliation passed: state is consistent", self.symbol);
        }

        Ok(())
    }

    /// Проверка условий частичной фиксации прибыли (Задача 166)
    /// Вызывается при обновлении стакана для проверки достижения уровней TP
    pub async fn check_partial_profit_taking(
        &mut self,
        mid_price: Decimal,
        best_bid: Decimal,
        best_ask: Decimal,
        rest_client: &impl BybitRestClientTrait,
        exchange_config: &ExchangeConfig,
    ) -> Result<()> {
        // Проверяем, есть ли открытая позиция
        let position = self.position_manager.get_position();
        if position.qty.is_zero() {
            return Ok(());
        }

        // Проверяем, есть ли настроенные этапы TP
        if self.bot_config.tp_stages.is_empty() {
            return Ok(());
        }

        let entry_price = position.avg_price;
        if entry_price.is_zero() {
            return Ok(());
        }

        // Расчет отклонения в базисных пунктах с учетом направления позиции
        let deviation_bps = if position.side == OrderSide::Buy {
            // Long позиция: профит при росте цены
            ((mid_price - entry_price) / entry_price * Decimal::from(10000))
                .to_f64()
                .unwrap_or(0.0)
        } else {
            // Short позиция: профит при падении цены
            ((entry_price - mid_price) / entry_price * Decimal::from(10000))
                .to_f64()
                .unwrap_or(0.0)
        };

        // Итерация по этапам TP (Задача 166)
        // Клонируем конфигурацию этапов, чтобы избежать конфликтов заимствования при вызове &mut self методов
        let tp_stages = self.bot_config.tp_stages.clone();
        for (stage_idx, stage) in tp_stages.iter().enumerate() {
            // Проверяем, не выполнен ли уже этот этап
            let is_completed = {
                let position = self.position_manager.get_position();
                position.completed_tp_stages.contains(&stage_idx)
            };
            if is_completed {
                continue;
            }

            // Проверяем, достигнут ли порог
            if deviation_bps >= stage.threshold_bps as f64 {
                info!(
                    "[{}] TP Stage {} triggered: deviation={:.2} bps >= threshold={} bps",
                    self.symbol, stage_idx, deviation_bps, stage.threshold_bps
                );

                // Рассчитываем размер закрытия на основе текущей позиции
                let (close_size_decimal, pos_qty_abs, pos_side) = {
                    let position = self.position_manager.get_position();
                    let close_size = position.initial_size * stage.close_pct;
                    let close_size_decimal = Decimal::from_f64(close_size).unwrap_or(Decimal::ZERO);
                    (close_size_decimal, position.qty.abs(), position.side)
                };

                // Выполняем частичное закрытие
                // Проверка min_qty и логика tp_close_all_on_min_qty теперь в place_limit_order (order_manager.rs)
                self.execute_partial_close(
                    close_size_decimal,
                    pos_qty_abs,
                    pos_side,
                    best_bid,
                    best_ask,
                    rest_client,
                    exchange_config,
                ).await?;

                // Помечаем этап как выполненный
                // ВАЖНО: Нужно получить мутабельную ссылку на позицию через position_manager
                // Но у нас нет прямого доступа к мутабельной позиции здесь
                // Поэтому добавим метод в PositionManager для пометки этапа
                self.mark_tp_stage_completed(stage_idx);
            }
        }

        Ok(())
    }

    /// Выполнение частичного закрытия позиции (Задача 166)
    async fn execute_partial_close(
        &mut self,
        size: Decimal,
        position_qty: Decimal,
        position_side: OrderSide,
        best_bid: Decimal,
        best_ask: Decimal,
        rest_client: &impl BybitRestClientTrait,
        exchange_config: &ExchangeConfig,
    ) -> Result<()> {
        // Определяем сторону ордера (противоположную позиции)
        let order_side = match position_side {
            OrderSide::Buy => OrderSide::Sell,  // Закрываем Long через Sell
            OrderSide::Sell => OrderSide::Buy,  // Закрываем Short через Buy
        };

        let mid_price = (best_bid + best_ask) / Decimal::from(2); // Задача 201: Для логирования slippage

        info!(
            "[{}] Executing partial close: {} {} (maker offset: {} ticks will be applied in place_limit_order)",
            self.symbol, order_side, size, self.bot_config.maker_offset_step_ticks
        );

        // Задача 166: Выставляем лимитный ордер с Post-Only и reduce_only
        // Расчет Maker-цены и проверка min_qty теперь в place_limit_order
        let _order_link_id = self.order_manager.place_limit_order(
            rest_client,
            &mut self.risk_manager,
            &self.bot_config,
            exchange_config,
            None,
            order_side,
            mid_price, // Будет использована как fallback если не reduce_only
            size,
            true,  // post_only = true для минимизации комиссий
            true,  // reduce_only = true для закрытия позиции
            mid_price,
            Some(best_bid),
            Some(best_ask),
            Some(position_qty),
        ).await?;

        Ok(())
    }

    /// Помечает этап TP как выполненный (Задача 166)
    fn mark_tp_stage_completed(&mut self, stage_idx: usize) {
        self.position_manager.mark_tp_stage_completed(stage_idx);
        info!("[{}] TP Stage {} marked as completed", self.symbol, stage_idx);
    }

    /// Обновляет динамический скользящий стоп-лосс (Задача 167)
    /// Вызывается при каждом обновлении цены для расчета и обновления уровня стопа
    pub fn update_tsl(&mut self, mid_price: f64) {
        let position = self.position_manager.get_position_mut();
        
        // Если позиция закрыта, ничего не делаем
        if position.qty.is_zero() {
            return;
        }

        let config = &self.bot_config.trailing_stop;
        let entry_price = position.avg_price.to_f64().unwrap_or(0.0);
        
        // Шаг 1: Проверяем активацию трейлинга по профиту
        if !position.tsl_active {
            let profit_bps = if position.side == OrderSide::Buy {
                (mid_price - entry_price) / entry_price * 10000.0
            } else {
                (entry_price - mid_price) / entry_price * 10000.0
            };
            
            if profit_bps >= config.tsl_activation_bps as f64 {
                position.tsl_active = true;
                debug!(
                    "[{}] TSL activated at profit: {:.2} bps (threshold: {} bps)",
                    self.symbol, profit_bps, config.tsl_activation_bps
                );
            } else {
                return; // Трейлинг еще не активирован
            }
        }

        // Шаг 2: Обновляем extreme_water_mark
        let distance_bps = config.tsl_distance_bps as f64;
        let mut new_extreme = position.extreme_water_mark;
        let mut updated_extreme = false;
        
        if position.side == OrderSide::Buy {
            // Для Long: отслеживаем максимум
            if mid_price > new_extreme {
                new_extreme = mid_price;
                updated_extreme = true;
            }
        } else {
            // Для Short: отслеживаем минимум
            if mid_price < new_extreme {
                new_extreme = mid_price;
                updated_extreme = true;
            }
        }

        // Шаг 3: Рассчитываем новый уровень стопа
        let new_sl = if position.side == OrderSide::Buy {
            // Для Long: new_sl = extreme * (1 - distance / 10000)
            new_extreme * (1.0 - distance_bps / 10000.0)
        } else {
            // Для Short: new_sl = extreme * (1 + distance / 10000)
            new_extreme * (1.0 + distance_bps / 10000.0)
        };

        // Шаг 4: Обновляем extreme_water_mark ВСЕГДА при улучшении цены
        if updated_extreme {
            position.extreme_water_mark = new_extreme;
        }

        // Шаг 5: Применяем фильтр шага (step_bps) ТОЛЬКО для current_stop_loss
        let step_bps = config.tsl_step_bps as f64;
        let current_sl = position.current_stop_loss;
        let mut should_update_sl = false;
        
        if current_sl > 0.0 {
            let sl_change_bps = (new_sl - current_sl).abs() / current_sl * 10000.0;
            if sl_change_bps >= step_bps {
                should_update_sl = true;
            }
        } else {
            // Если это первое обновление (current_sl = 0), обновляем
            should_update_sl = true;
        }

        if should_update_sl {
            position.current_stop_loss = new_sl;
            debug!(
                "[{}] TSL updated: extreme={:.8}, new_sl={:.8}, distance_bps={}",
                self.symbol, new_extreme, new_sl, distance_bps
            );
        } else {
            debug!(
                "[{}] TSL extreme updated: {:.8}, but sl update suppressed by step filter (change={:.2} bps < {:.2} bps)",
                self.symbol, new_extreme,
                (new_sl - current_sl).abs() / current_sl * 10000.0,
                step_bps
            );
        }
    }

    /// Проверяет триггер TSL и закрывает позицию при пересечении (Задача 167)
    /// Используется только для Bot-side режима
    async fn check_tsl_trigger(
        &mut self,
        mid_price: f64,
        best_bid: Decimal,
        best_ask: Decimal,
        rest_client: &impl BybitRestClientTrait,
        exchange_config: &ExchangeConfig,
    ) -> Result<()> {
        let position = self.position_manager.get_position();
        
        // Если позиция закрыта или TSL не активирован, ничего не делаем
        if position.qty.is_zero() || !position.tsl_active {
            return Ok(());
        }

        let current_sl = position.current_stop_loss;
        let should_close = if position.side == OrderSide::Buy {
            // Для Long: закрываем если цена упала ниже стопа
            mid_price <= current_sl
        } else {
            // Для Short: закрываем если цена выросла выше стопа
            mid_price >= current_sl
        };

        if should_close {
            info!(
                "[{}] TSL triggered! Mid price: {:.8}, Stop Loss: {:.8}. Closing position by market.",
                self.symbol, mid_price, current_sl
            );

            // Закрываем позицию по рынку
            let close_side = if position.side == OrderSide::Buy {
                OrderSide::Sell
            } else {
                OrderSide::Buy
            };

            let close_price = if close_side == OrderSide::Buy {
                best_ask
            } else {
                best_bid
            };

            let close_qty = position.qty.abs();

            // Создаем рыночный ордер на закрытие
            let close_request = CreateOrderRequest {
                symbol: self.symbol.clone(),
                category: exchange_config.bybit.category.clone(),
                side: close_side.to_string(),
                order_type: "Market".to_string(),
                qty: close_qty.to_string(),
                price: Some(close_price.to_string()),
                time_in_force: "IOC".to_string(),
                order_link_id: format!("TSL_CLOSE_{}", chrono::Utc::now().timestamp_millis()),
                position_idx: self.bot_config.position_idx,
                reduce_only: Some(true),
                trailing_stop: None,
                active_price: None,
                smp_type: None,
            };

            match rest_client.post::<CreateOrderRequest, BybitOrderResult>("/v5/order/create", &close_request).await {
                Ok(order_id) => {
                    info!("[{}] TSL close order created: {:?}", self.symbol, order_id);
                }
                Err(e) => {
                    error!("[{}] Failed to create TSL close order: {}", self.symbol, e);
                }
            }
        }

        Ok(())
    }

    /// Проверяет, должен ли вход быть заблокирован из-за высокой ставки финансирования (Задача 170)
    /// 
    /// Логика:
    /// 1. Определяем, является ли направление "adverse" (вредным для нас)
    /// 2. Если adverse и rate_bps > max_funding_rate_bps:
    ///    - Проверяем, находимся ли мы в окне перед клирингом
    ///    - Проверяем, достаточно ли уверен сигнал
    /// 3. Если оба условия не выполнены, блокируем вход
    pub fn should_block_by_funding_rate(&self, side: OrderSide, confidence: f64) -> bool {
        use crate::trading::types::OrderSide;
        
        let funding_rate = self.current_funding_rate;
        let rate_bps = (funding_rate.abs() * 10000.0) as u32;
        
        // Определяем, является ли направление adverse (вредным)
        let is_adverse = (funding_rate > 0.0 && side == OrderSide::Buy) || 
                        (funding_rate < 0.0 && side == OrderSide::Sell);
        
        if !is_adverse {
            // Направление благоприятно, фандинг нас не волнует
            return false;
        }
        
        // Направление adverse, проверяем величину ставки
        if rate_bps <= self.bot_config.funding_filter.max_funding_rate_bps {
            // Ставка в пределах допустимого
            return false;
        }
        
        // Ставка высокая и adverse. Проверяем условия исключения.
        
        // Условие 1: Проверяем время до клиринга
        let current_time_ms = crate::utils::helpers::unix_ms();
        let time_to_settlement_ms = self.next_funding_time.saturating_sub(current_time_ms);
        let in_settlement_window = time_to_settlement_ms < self.bot_config.funding_filter.avoid_settlement_window_ms;
        
        // Условие 2: Проверяем уверенность сигнала
        let high_confidence = confidence >= self.bot_config.funding_filter.min_confidence_to_ignore_funding;
        
        // Блокируем вход, если:
        // - Мы в окне перед клирингом И уверенность низкая
        if in_settlement_window && !high_confidence {
            warn!("[{}] Funding filter: Blocking entry. Rate: {:.6} ({} bps), Time to settlement: {}ms, Confidence: {:.2}%",
                self.symbol, funding_rate, rate_bps, time_to_settlement_ms, confidence * 100.0);
            return true;
        }
        
        // Если мы не в окне перед клирингом, но ставка очень высокая и уверенность низкая
        if !high_confidence && rate_bps > self.bot_config.funding_filter.max_funding_rate_bps * 2 {
            warn!("[{}] Funding filter: Blocking entry due to very high rate. Rate: {:.6} ({} bps), Confidence: {:.2}%",
                self.symbol, funding_rate, rate_bps, confidence * 100.0);
            return true;
        }
        
        false
    }

    /// Проверяет проскальзывание входа и выбирает стратегию входа (задача 168: Volume-Weighted Entry)
    /// Использует direction-aware логику для определения приемлемого проскальзывания
    /// 
    /// # Параметры
    /// - `side`: Сторона ордера (Buy или Sell)
    /// - `size`: Требуемый размер ордера
    /// - `mid_price`: Текущая средняя цена
    /// - `orderbook`: Текущий снапшот стакана (Задача 191: lock-free доступ)
    /// 
    /// # Возвращает
    /// (entry_style, vwap) - выбранная стратегия входа и рассчитанный VWAP
    fn check_entry_slippage(
        &self,
        side: crate::data::types::Side,
        size: f64,
        mid_price: f64,
        orderbook: &crate::data::orderbook::OrderBook,
    ) -> (crate::config::types::EntryStyle, Option<f64>) {
        use crate::data::types::Side;
        use crate::config::types::EntryStyle;

        // Получаем VWAP для исполнения
        let vwap = orderbook.get_execution_vwap(side, size);
        
        if vwap.is_none() {
            // Недостаточно ликвидности - используем PassiveLimit
            return (EntryStyle::PassiveLimit, None);
        }

        let vwap_val = vwap.unwrap();
        let max_slippage_bps = self.bot_config.max_entry_slippage_bps as f64;
        
        // Direction-aware проверка проскальзывания
        let slippage_bps = match side {
            Side::Buy => {
                // Для покупки: VWAP должен быть близко к mid_price
                // Если VWAP > mid * (1 + max_bps / 10000), то проскальзывание высокое
                ((vwap_val - mid_price) / mid_price) * 10000.0
            }
            Side::Sell => {
                // Для продажи: VWAP должен быть близко к mid_price
                // Если VWAP < mid * (1 - max_bps / 10000), то проскальзывание высокое
                ((mid_price - vwap_val) / mid_price) * 10000.0
            }
        };

        // Выбираем стратегию входа на основе проскальзывания
        let entry_style = if slippage_bps <= max_slippage_bps {
            // Проскальзывание в норме - используем AggressiveMarket
            EntryStyle::AggressiveMarket
        } else if slippage_bps <= max_slippage_bps * 1.5 {
            // Проскальзывание немного повышено - используем ChaseBest
            EntryStyle::ChaseBest
        } else {
            // Проскальзывание слишком высокое - используем PassiveLimit
            EntryStyle::PassiveLimit
        };

        debug!(
            "[{}] Entry slippage check: side={:?}, vwap={:.8}, mid={:.8}, slippage_bps={:.2}, strategy={:?}",
            self.symbol, side, vwap_val, mid_price, slippage_bps, entry_style
        );

        (entry_style, vwap)
    }

    /// Обработка обновления orderbook (Задача 166)
    /// Вызывается при каждом обновлении orderbook для проверки условий частичной фиксации прибыли
    pub async fn on_ob_update(
        &mut self,
        mid_price: Decimal,
        best_bid: Decimal,
        best_ask: Decimal,
        rest_client: &impl BybitRestClientTrait,
        exchange_config: &ExchangeConfig,
    ) -> Result<()> {
        // Задача 042: Обновление эквити для отслеживания пикового значения и просадки
        let pos = self.position_manager.get_position();
        let current_pnl = pos.realized_pnl + pos.unrealized_pnl;
        self.risk_manager.update_equity(current_pnl);

        // Обновление динамического скользящего стоп-лосса (Задача 167)
        let mid_price_f64 = mid_price.to_f64().unwrap_or(0.0);
        self.update_tsl(mid_price_f64);

        // Задача 178: Обновление истории цен для отслеживания волатильности
        self.risk_manager.health_monitor.update_price(mid_price_f64);

        // Проверка триггера TSL для Bot-side режима (Задача 167)
        if self.bot_config.trailing_stop.tsl_mode == crate::config::types::TSLMode::Bot {
            self.check_tsl_trigger(mid_price_f64, best_bid, best_ask, rest_client, exchange_config).await?;
        }

        // Проверка условий частичной фиксации прибыли (Задача 166)
        self.check_partial_profit_taking(
            mid_price,
            best_bid,
            best_ask,
            rest_client,
            exchange_config,
        ).await?;

        Ok(())
    }

    /// Проверка возможности исполнения сигнала (Задача 165: Anti-Adversarial Protection, Пункт 3 плана)
    /// Применяет защитный гейт для входа: проверка токсичности потока и других рисков
    /// 
    /// Возвращает ExecutionAction::Execute если все проверки пройдены, иначе ExecutionAction::Skip
    /// Публичная обертка для тестирования гейтов (Задача 169)
    #[cfg(test)]
    pub fn can_execute_public(&mut self, orderbook_update: &OrderBookUpdateOwned, signal: &crate::ml::types::Signal) -> ExecutionAction {
        self.can_execute(orderbook_update, signal)
    }

    fn can_execute(&mut self, orderbook_update: &OrderBookUpdateOwned, signal: &crate::ml::types::Signal) -> ExecutionAction {
        // Задача 169: Проверка свежести сигнала (Signal Staleness Check) - финальный гейт
        let current_time_ms = crate::utils::helpers::unix_ms();
        let signal_age_ms = current_time_ms - signal.source_timestamp_ms;
        
        if signal_age_ms > self.bot_config.max_signal_age_ms {
            tracing::warn!(
                "[{}] [Stale Signal] Age {}ms > {}ms limit. Skipping execution.",
                self.symbol, signal_age_ms, self.bot_config.max_signal_age_ms
            );
            
            // Регистрируем устаревший сигнал
            self.risk_manager.register_signal_staleness(true);
            
            // Проверяем circuit breaker
            if self.risk_manager.check_stale_signal_circuit_breaker() {
                tracing::error!(
                    "[{}] CIRCUIT BREAKER TRIGGERED: Too many stale signals. Entering emergency mode.",
                    self.symbol
                );
                self.emergency_mode = true;
            }
            
            match self.bot_config.staleness_action {
                crate::config::types::StalenessAction::Skip => {
                    return ExecutionAction::Skip;
                },
                crate::config::types::StalenessAction::LogOnly => {
                    tracing::warn!("[{}] Proceeding with stale signal (LogOnly mode)", self.symbol);
                },
            }
        } else {
            // Регистрируем свежий сигнал
            self.risk_manager.register_signal_staleness(false);
        }
        
        // Проверка адверсариальной активности (VPIN, Layering, Spoofing)
        let is_toxic = self.adversarial_detector.update_and_check(orderbook_update, &self.pending_trades);
        self.pending_trades.clear(); // Очищаем накопленные сделки для следующего обновления
        
        if is_toxic {
            tracing::warn!(
                "[{}] Adversarial protection: Toxic flow detected, skipping signal",
                self.symbol
            );
            return ExecutionAction::Skip;
        }

        ExecutionAction::Execute
    }

    /// Точка входа для новых предсказаний модели
    pub async fn on_inference_output(
        &mut self, 
        output: InferenceOutput, 
        _current_price_f64: f64,
        best_bid: Decimal,
        bid_vol: Decimal,
        best_ask: Decimal,
        ask_vol: Decimal,
        orderbook: &crate::data::orderbook::OrderBook,
        orderbook_update: &OrderBookUpdateOwned,
        rest_client: &impl BybitRestClientTrait,
        exchange_config: &ExchangeConfig,
        regime: crate::config::types::RegimeId,  // Задача 161: текущий режим рынка
    ) -> Result<()> {
        // Задача 231: Проверка и сброс штрафа за ошибку маржи
        self.risk_manager.check_and_reset_margin_penalty(&self.bot_config.risk);
        
        // Задача 169: Сохраняем timestamp источника сигнала для проверки свежести
        self.last_signal_timestamp_ms = output.signal.source_timestamp_ms;
        
        // Задача 169: Проверка синхронизации времени с биржей (Clock Skew Check)
        // Это критично для корректного расчета возраста сигнала
        match crate::utils::helpers::check_clock_skew(
            &exchange_config.rest.base_url,
            self.bot_config.max_clock_skew_ms,
        ).await {
            Ok(delta) => {
                if delta.abs() > self.bot_config.max_clock_skew_ms {
                    error!(
                        "[{}] CRITICAL: Clock skew {}ms exceeds limit {}ms. Signal age calculations unreliable. Skipping signal.",
                        self.symbol, delta, self.bot_config.max_clock_skew_ms
                    );
                    self.risk_manager.register_signal_staleness(true);
                    return Ok(());
                }
            },
            Err(e) => {
                warn!(
                    "[{}] WARNING: Failed to check clock skew: {}. Proceeding with caution.",
                    self.symbol, e
                );
            }
        }
        
        // 0. Проверка аварийного режима (Emergency Mode) - Задача 107
        if self.emergency_mode {
            error!("[{}] BLOCKED: Emergency mode is active. Manual intervention required.", self.symbol);
            return Err(anyhow::anyhow!("Trading blocked: emergency_mode is active"));
        }

        // Задача 224: Обработка дрейфа модели
        if output.drift_detected {
            if let Some(entropy) = output.entropy {
                warn!(
                    "[{}] Model drift detected! Entropy: {:.4}, threshold: {:.4}",
                    self.symbol, entropy, self.bot_config.entropy_drift_threshold
                );
                self.risk_manager.handle_model_drift(entropy, &self.bot_config);
                
                // Если включена полная остановка при дрейфе
                if self.bot_config.drift_stop_enabled {
                    error!("[{}] EMERGENCY: Model drift detected and drift_stop_enabled=true. Entering emergency mode.", self.symbol);
                    self.emergency_mode = true;
                    if let Err(e) = self.emergency_market_close(rest_client, exchange_config).await {
                        error!("[{}] CRITICAL: Emergency exit failed after model drift: {}", self.symbol, e);
                    }
                }
            }
        }

        let mid_price = (best_bid + best_ask) / Decimal::from(2);
        let prev_reset_ts = {
            let state_guard = self.state.lock().await;
            state_guard.last_pnl_reset_ts
        };

        // 0.1. Проверка дневного лимита убытка (Задача 111)
        {
            let mut state_guard = self.state.lock().await;
            match self.risk_manager.check_daily_limit(
                &self.bot_config, 
                &mut *state_guard, 
                &self.position_manager, 
                rest_client, 
                mid_price
            ).await {
                Ok(true) => {
                    // Сохраняем стейт только если произошел сброс дня (last_pnl_reset_ts изменился)
                    // Это предотвращает избыточный disk I/O на каждый сигнал в HFT.
                    if state_guard.last_pnl_reset_ts != prev_reset_ts {
                        drop(state_guard); // Освобождаем блокировку перед вызовом save_current_state
                        let _ = self.save_current_state().await;
                    }
                }, 
                Ok(false) => {
                    error!("FATAL: Bot stopped due to daily risk limit breached");
                    self.emergency_mode = true;
                    drop(state_guard); // Освобождаем блокировку
                    if let Err(e) = self.emergency_market_close(rest_client, exchange_config).await {
                        error!("[{}] CRITICAL: Emergency exit failed after daily risk breach: {}", self.symbol, e);
                    }
                    return Ok(());
                },
                Err(e) => {
                    error!("FATAL: Bot stopped due to API failure during daily risk check: {}", e);
                    self.emergency_mode = true;
                    drop(state_guard); // Освобождаем блокировку
                    if let Err(panic_err) = self.emergency_market_close(rest_client, exchange_config).await {
                        error!("[{}] CRITICAL: Emergency exit failed after risk check error: {}", self.symbol, panic_err);
                    }
                    return Err(e);
                }
            }
        }

        // 0. Слияние предсказаний для разных горизонтов
        let fused_probs = self.fuse_probs(&output).await?;
        self.last_probabilities = fused_probs;
        
        let mid_price = (best_bid + best_ask) / Decimal::from(2);
        let mid_price_f64 = mid_price.to_f64().unwrap_or(0.0);
        
        // 0.1. Обновляем буфер волатильности
        self.update_volatility(mid_price_f64);
        
        // 1. Обновляем плавающий PnL перед расчетами (используем mid как mark заглушку)
        self.position_manager.update_unrealized_pnl(mid_price, mid_price);

        let position = self.position_manager.get_position().clone();
        let current_pnl = position.realized_pnl + position.unrealized_pnl;

        // Задача 178: Обновление масштаба лимита позиции на основе просадки и волатильности
        {
            let drawdown_pct = self.risk_manager.get_current_drawdown_pct(current_pnl);
            let current_vol = self.risk_manager.health_monitor.get_current_volatility();
            let hist_vol = self.risk_manager.health_monitor.get_historical_volatility();
            self.risk_manager.update_position_scale(drawdown_pct, current_vol, hist_vol);
            
            debug!(
                "[{}] Position Scale Updated: {:.1}% (DD: {:.1}%, Vol: {:.2}/{:.2})",
                self.symbol,
                self.risk_manager.get_current_scale() * 100.0,
                drawdown_pct,
                current_vol,
                hist_vol
            );
        }

        // 2. Проверка глобального риска (Drawdown) перед любыми действиями
        if let Err(e) = self.risk_manager.check_global_risk(current_pnl) {
            let error_msg = e.to_string();
            if error_msg.contains("HardStop") || error_msg.contains("drawdown") {
                // ЯВНЫЙ HARDSTOP TRIGGER — выполняем экстренные действия
                tracing::error!("[{}] HARD STOP TRIGGERED: {}", self.symbol, error_msg);
                // 1. Отмена всех ордеров
                let _ = self.order_manager.cancel_all_orders(rest_client, &mut self.risk_manager, &self.bot_config, exchange_config).await;
                // 2. Экстренное закрытие позиции
                let _ = self.order_manager.execute_emergency_close(rest_client, &mut self.risk_manager, &self.position_manager, &self.bot_config, exchange_config).await;
                // 3. Система уже в режиме Blocked (is_blocked = true в risk_manager)
                self.emergency_mode = true;
                return Ok(());
            }
            return Err(e);
        }

        // 2.0.5. Проверка периода блокировки после убытков (Lockout Period) (Задача 118)
        {
            let state_guard = self.state.lock().await;
            if self.risk_manager.is_in_lockout(&*state_guard, &self.bot_config) {
                // Просто выходим, не генерируя лишних логов, если уже предупреждали
                return Ok(());
            }
        }

        // 2.1. Проверка лимита переторговки (Задача 112)
        {
            let mut state_guard = self.state.lock().await;
            if !self.risk_manager.check_overtrading_limit(&mut *state_guard) {
                let now = timestamp_ms() as i64;
                if now - self.last_overtrade_warn_ts > 60_000 {
                    warn!("[{}] OVERTRADING PROTECTION: Signal ignored (limit reached).", self.symbol);
                    self.last_overtrade_warn_ts = now;
                }
                return Ok(());
            }
        }

        // 3. Проверка спреда с двойным барьером (Задача 104)
        if !self.check_spread_barrier(best_bid, best_ask) {
            debug!("[{}] Signal suppressed: Spread Barrier Filter blocked", self.symbol);
            return Ok(()); // Отклоняем сигнал
        }

        // 3.1. Проверка спреда (Задача 073) - дополнительная проверка объемов
        if !self.risk_manager.check_spread_gate(best_bid, bid_vol, best_ask, ask_vol) {
            debug!("[{}] Signal suppressed: Spread Filter Gate is closed", self.symbol);
            return Ok(()); // Отклоняем сигнал (эквивалентно Signal::Flat)
        }

        // 3.2. Проверка гейта дисбаланса стакана (Задача 117)
        let current_obi = orderbook.calculate_imbalance(self.bot_config.obi_depth);
        debug!("[{}] Current OBI: {:.4}", self.symbol, current_obi);

        // 4. Фильтрация сигнала по порогам вероятности (Задача 044) с учетом режима рынка (Задача 161)
        let effective_signal = self.filter_signal(&fused_probs, regime);
        let signal_side = effective_signal.side; // Сохраняем side до перемещения сигнала
        
        // Задача 201: Создаем SignalWithTimestamp для замера latency
        let signal_with_ts = crate::ml::types::SignalWithTimestamp {
            signal: effective_signal,
            start_instant: std::time::Instant::now(),
        };
        
        // Сохраняем сигнал и цену для логирования
        self.pending_slice_signal = Some(signal_with_ts);
        self.last_signal_price = mid_price.to_f64().unwrap_or(0.0);

        // 4.0. Проверка адверсариальной активности (Задача 165: Anti-Adversarial Protection, Пункт 3 плана)
        // Применяет защитный гейт через метод can_execute() для определения ExecutionAction
        // Задача 169: Передаем signal для проверки свежести сигнала
        let execution_action = self.can_execute(orderbook_update, &output.signal);
        match execution_action {
            ExecutionAction::Execute => {
                // Продолжаем с исполнением сигнала
            }
            ExecutionAction::PartialClose | ExecutionAction::Skip => {
                tracing::warn!("[{}] can_execute returned {:?}, aborting inference signal processing", self.symbol, execution_action);
                return Ok(());
            }
        }

        // 4.2. Защита от осцилляций (Throttling) - Задача 148
        let now_ms = timestamp_ms() as i64;
        let is_flip = match signal_side {
            crate::ml::types::SignalSide::Up => position.qty.is_sign_negative(),
            crate::ml::types::SignalSide::Down => position.qty.is_sign_positive(),
            _ => false,
        };

        if is_flip {
            let elapsed = now_ms - self.last_flip_ts;
            if elapsed < self.bot_config.min_flip_interval_ms as i64 {
                let symbol = self.symbol.as_str();
                let active_ids: Vec<String> = self.order_manager.get_active_orders().await.keys().cloned().collect();
                warn!(
                    "[{}] SIGNAL OSCILLATION: Suppressing flip {} -> {} (elapsed {}ms < {}ms). Active orders: {:?}",
                    symbol,
                    if position.qty.is_sign_positive() { "Long" } else { "Short" },
                    match signal_side { crate::ml::types::SignalSide::Up => "Up", crate::ml::types::SignalSide::Down => "Down", _ => "Flat" },
                    elapsed,
                    self.bot_config.min_flip_interval_ms,
                    active_ids
                );
                
                if let Some(counter) = crate::monitoring::prometheus::OSCILLATION_COUNTER.get() {
                    counter.with_label_values(&[symbol]).inc();
                }
                
                return Ok(());
            }
            // Обновляем время последнего переворота ПЕРЕД исполнением
            self.last_flip_ts = now_ms;
        }

        // 4.1. Проверка exit_threshold для текущей позиции
        if let Some(exit_th) = self.bot_config.exit_threshold {
            let exit_th_f = exit_th.to_f32().unwrap_or(0.4);
            let prob_up = fused_probs[1];
            let prob_down = fused_probs[2];
            
            // Если мы в позиции, но уверенность в направлении упала ниже порога выхода
            if !position.qty.is_zero() {
                let should_exit = if position.qty.is_sign_positive() {
                    // В лонге: проверяем уверенность в Up
                    prob_up < exit_th_f
                } else {
                    // В шорте: проверяем уверенность в Down
                    prob_down < exit_th_f
                };

                if should_exit {
                    let side = if position.qty.is_sign_positive() { OrderSide::Sell } else { OrderSide::Buy };
                    tracing::info!(
                        "[{}] Exit threshold triggered: position={}, prob_up={:.4}, prob_down={:.4}, exit_threshold={:.4}",
                        self.symbol, 
                        if position.qty.is_sign_positive() { "Long" } else { "Short" },
                        prob_up, prob_down, exit_th_f
                    );
                    self.execute_trade(side, Signal::flat(), &fused_probs, best_bid, best_ask, mid_price, orderbook, orderbook_update, rest_client, exchange_config).await?;
                    return Ok(());
                }
            }
        }

        // 5. Логика исполнения на основе отфильтрованного сигнала
        match signal_side {
            crate::ml::types::SignalSide::Up => {
                if position.qty.is_sign_negative() || position.qty.is_zero() {
                    // Проверка OBI гейта перед покупкой
                    if !self.risk_manager.check_imbalance_gate(OrderSide::Buy, current_obi, &self.bot_config) {
                        debug!("[{}] BUY signal suppressed: OBI Gate blocked", self.symbol);
                        return Ok(());
                    }
                    // Если мы в кэше или в шорте — покупаем (переворот или вход)
                    self.execute_trade(OrderSide::Buy, Signal::up(), &fused_probs, best_bid, best_ask, mid_price, orderbook, orderbook_update, rest_client, exchange_config).await?;
                }
            }
            crate::ml::types::SignalSide::Down => {
                if position.qty.is_sign_positive() || position.qty.is_zero() {
                    // Проверка OBI гейта перед продажей
                    if !self.risk_manager.check_imbalance_gate(OrderSide::Sell, current_obi, &self.bot_config) {
                        debug!("[{}] SELL signal suppressed: OBI Gate blocked", self.symbol);
                        return Ok(());
                    }
                    // Если мы в кэше или в лонге — продаем (переворот или вход)
                    self.execute_trade(OrderSide::Sell, Signal::down(), &fused_probs, best_bid, best_ask, mid_price, orderbook, orderbook_update, rest_client, exchange_config).await?;
                }
            }
            crate::ml::types::SignalSide::Flat => {
                if self.close_on_flat && !position.qty.is_zero() {
                    // Закрытие текущей позиции при сигнале Flat
                    let side = if position.qty.is_sign_positive() { OrderSide::Sell } else { OrderSide::Buy };
                    info!("[{}] Closing position due to Flat signal", self.symbol);
                    self.execute_trade(side, Signal::flat(), &fused_probs, best_bid, best_ask, mid_price, orderbook, orderbook_update, rest_client, exchange_config).await?;
                }
            }
        }

        // Задача 202: Проверка очереди ордеров, ожидающих захвата цены через 100мс
        self.check_and_capture_100ms_prices(best_bid, best_ask).await;

        Ok(())
    }

    /// Слияние предсказаний для разных временных горизонтов
    async fn fuse_probs(&self, output: &InferenceOutput) -> Result<[f32; 3]> {
        use crate::config::types::FusionMethod;
        
        let num_horizons = output.probs.shape()[0];
        
        // Валидация конфигурации в runtime
        match self.bot_config.fusion.method {
            FusionMethod::WeightedAverage => {
                if self.bot_config.fusion.weights.len() != num_horizons {
                    anyhow::bail!(
                        "Fusion config error: weights.len() ({}) != num_horizons ({})",
                        self.bot_config.fusion.weights.len(),
                        num_horizons
                    );
                }
            }
            FusionMethod::Principal => {
                if self.bot_config.fusion.principal_idx >= num_horizons {
                    anyhow::bail!(
                        "Fusion config error: principal_idx ({}) >= num_horizons ({})",
                        self.bot_config.fusion.principal_idx,
                        num_horizons
                    );
                }
            }
            _ => {}
        }
        
        match self.bot_config.fusion.method {
            FusionMethod::WeightedAverage => {
                let mut fused = [0.0f32; 3];
                for (i, weight) in self.bot_config.fusion.weights.iter().enumerate() {
                    let w = weight.to_f32().unwrap_or(0.0);
                    for cls in 0..3 {
                        fused[cls] += output.probs[[i, cls]] * w;
                    }
                }
                Ok(fused)
            },
            FusionMethod::Consensus => {
                let mut votes_up = 0;
                let mut votes_down = 0;
                // Используем динамические пороги для консистентности (Задача 115)
                let current_streak = {
                    let state_guard = self.state.lock().await;
                    state_guard.loss_streak
                };
                let dynamic_threshold = self.risk_manager.get_effective_threshold(current_streak, &self.bot_config) as f32;

                for i in 0..num_horizons {
                    if output.probs[[i, 1]] > dynamic_threshold { votes_up += 1; }
                    if output.probs[[i, 2]] > dynamic_threshold { votes_down += 1; }
                }

                if votes_up >= self.bot_config.fusion.min_horizons { 
                    Ok([0.0, 1.0, 0.0]) // 1 = Up
                } else if votes_down >= self.bot_config.fusion.min_horizons { 
                    Ok([0.0, 0.0, 1.0]) // 2 = Down
                } else { 
                    Ok([1.0, 0.0, 0.0]) 
                }
            },
            FusionMethod::Principal => {
                let idx = self.bot_config.fusion.principal_idx;
                Ok([output.probs[[idx, 0]], output.probs[[idx, 1]], output.probs[[idx, 2]]])
            }
        }
    }

    /// Превращает вероятности в сигнал на основе асимметричных порогов
    /// Задача 101: Использование long_threshold и short_threshold вместо единых порогов
    /// Задача 161: Использование динамических порогов на основе режима рынка
    fn filter_signal(&self, probs: &[f32; 3], regime: crate::config::types::RegimeId) -> Signal {
        // Индексы: [0] = Flat, [1] = Up, [2] = Down
        let prob_flat = probs[0];
        let prob_up = probs[1];
        let prob_down = probs[2];

        // Задача 161: Получаем динамические пороги на основе режима рынка
        let (buy_threshold, sell_threshold, _min_confidence) = self.get_thresholds_for_regime(regime);
        
        // Используем динамические пороги, если они не равны базовым (режим переопределен)
        // Иначе используем базовые пороги из конфига
        let long_th = buy_threshold;
        let short_th = sell_threshold;

        // Логирование всех трёх вероятностей [F, D, U] при принятии решения (Задача 101)
        // Задача 161: также логируем текущий режим
        tracing::info!(
            "[{}] Signal probabilities: [Flat={:.4}, Down={:.4}, Up={:.4}], thresholds: [long={:.4}, short={:.4}], regime: {:?}",
            self.symbol, prob_flat, prob_down, prob_up, long_th, short_th, regime
        );

        // Приоритет входа: игнорировать противоречивый сигнал (Задача 101)
        // Если обе вероятности выше порогов - редкий случай при softmax
        if prob_up > long_th && prob_down > short_th {
            tracing::warn!(
                "[{}] Contradictory signal ignored: Up({:.4})>{:.4} AND Down({:.4})>{:.4}",
                self.symbol, prob_up, long_th, prob_down, short_th
            );
            return Signal::flat();
        }

        if prob_up > long_th {
            Signal::up()
        } else if prob_down > short_th {
            Signal::down()
        } else {
            // Если уверенность ниже порогов, считаем сигнал нейтральным
            if prob_up > 0.4 || prob_down > 0.4 {
                tracing::debug!(
                    "[{}] Signal suppressed by thresholds: Up={:.2} (>{:.2}), Down={:.2} (>{:.2})",
                    self.symbol, prob_up, long_th, prob_down, short_th
                );
            }
            Signal::flat()
        }
    }

    /// Вспомогательный метод для валидации риска и подготовки ордера
    async fn execute_trade(
        &mut self, 
        side: OrderSide, 
        signal: Signal,
        probs: &[f32; 3],
        best_bid: Decimal, 
        best_ask: Decimal, 
        mid_price: Decimal,
        orderbook: &crate::data::orderbook::OrderBook,
        orderbook_update: &OrderBookUpdateOwned,
        rest_client: &impl BybitRestClientTrait,
        exchange_config: &ExchangeConfig,
    ) -> Result<()> {
        // Задача 170: Фильтр по ставкам финансирования (Funding Rate Arbitrage Filter)
        let max_confidence = probs.iter().cloned().fold(0.0f32, f32::max) as f64;
        if self.should_block_by_funding_rate(side, max_confidence) {
            info!("[{}] Entry blocked by funding rate filter. Side: {:?}, Funding: {:.6}, Confidence: {:.2}%",
                self.symbol, side, self.current_funding_rate, max_confidence * 100.0);
            return Ok(());
        }
        
        let position = self.position_manager.get_position();
        
        // Используем актуальный PnL из обновленной структуры
        let current_pnl = position.realized_pnl + position.unrealized_pnl;
        
        // В будущем здесь будет получаться реальный баланс
        let available_balance = self.bot_config.initial_balance;
        
        // Задача 164: Safety Valve (Force-Taker)
        // max_confidence уже вычислена выше для фильтра по фандингу
        let force_taker = max_confidence > self.bot_config.force_taker_confidence;
        
        // Определяем, будем ли использовать Post-Only режим
        let use_post_only = self.bot_config.post_only && !force_taker;
        
        // Задача 164: Получаем скорректированный offset с учетом Soft Limit
        let base_offset_ticks = self.bot_config.maker_offset_step_ticks;
        let adjusted_offset_ticks = self.risk_manager.get_adjusted_maker_offset(base_offset_ticks);
        let offset = self.market_info.tick_size * Decimal::from(adjusted_offset_ticks);
        
        // Определяем цену исполнения
        let mut price = if force_taker {
            // Force-Taker: агрессивный вход по рынку
            info!("[{}] Force-Taker mode activated (confidence: {:.4})", self.symbol, max_confidence);
            match side {
                OrderSide::Buy => best_ask,
                OrderSide::Sell => best_bid,
            }
        } else if use_post_only {
            // Maker-режим: отступаем от Best Price на offset
            match side {
                OrderSide::Buy => best_bid - offset,
                OrderSide::Sell => best_ask + offset,
            }
        } else {
            // Обычный Taker-режим
            match side {
                OrderSide::Buy => best_ask,
                OrderSide::Sell => best_bid,
            }
        };
        
        // Проверка цены ордера на отклонение и кратность шагу цены (Задача 075)
        match self.risk_manager.validate_order_price(price, mid_price, self.market_info.tick_size) {
            Ok(valid) => {
                if !valid {
                    debug!("[{}] Order price validation failed: Price {} deviates too much from mid price {}",
                           self.symbol, price, mid_price);
                    return Ok(());
                }
            },
            Err(e) => {
                debug!("[{}] Order price validation error: {}", self.symbol, e);
                return Ok(());
            }
        }

        // Проверка Slippage Tolerance через VWAP (Задача 106)
        let vwap = self.price_stats.get_vwap(None);
        if !vwap.is_zero() && !self.price_stats.is_empty() {
            let deviation = ((price - vwap) / vwap).abs();
            let max_deviation = Decimal::from_f64(0.02).unwrap_or(Decimal::from(2) / Decimal::from(100)); // 2%
            
            if deviation > max_deviation {
                warn!(
                    "[{}] High slippage detected: price={}, vwap={}, deviation={:.2}%",
                    self.symbol, price, vwap, deviation * Decimal::from(100)
                );
            }
        }

        let qty = self.calculate_order_size(available_balance, price, signal.clone(), probs);
        
        if qty.is_zero() {
            info!("Skipping trade: calculated quantity is zero");
            return Ok(());
        }

        // Задача 168: Volume-Weighted Entry - проверка проскальзывания и выбор стратегии входа
        let mid_price_f64 = mid_price.to_f64().unwrap_or(0.0);
        let qty_f64 = qty.to_f64().unwrap_or(0.0);
        let data_side = match side {
            OrderSide::Buy => crate::data::types::Side::Buy,
            OrderSide::Sell => crate::data::types::Side::Sell,
        };
        
        let (entry_style, vwap_opt) = self.check_entry_slippage(data_side, qty_f64, mid_price_f64, orderbook);
        
        // Применяем логику выбора стратегии входа
        match entry_style {
            crate::config::types::EntryStyle::AggressiveMarket => {
                // Агрессивный вход - используем рыночный ордер
                debug!("[{}] Using AggressiveMarket entry strategy", self.symbol);
                
                // Задача 168: Проверка адверсариальной активности перед AggressiveMarket
                // Если поток токсичен, отменяем вход
                let is_toxic = self.adversarial_detector.update_and_check(orderbook_update, &self.pending_trades);
                if is_toxic {
                    warn!(
                        "[{}] AggressiveMarket entry REJECTED: Toxic flow detected by AdversarialDetector",
                        self.symbol
                    );
                    return Ok(());
                }
                
                // Цена уже установлена как best_ask/best_bid выше
            },
            crate::config::types::EntryStyle::ChaseBest => {
                // Преследуем лучшую цену - ставим лимит на Best Bid/Ask
                debug!("[{}] Using ChaseBest entry strategy", self.symbol);
                price = match side {
                    OrderSide::Buy => best_ask,
                    OrderSide::Sell => best_bid,
                };
            },
            crate::config::types::EntryStyle::PassiveLimit => {
                // Пассивный вход - ставим лимит с отступом
                debug!("[{}] Using PassiveLimit entry strategy", self.symbol);
                price = match side {
                    OrderSide::Buy => best_bid - offset,
                    OrderSide::Sell => best_ask + offset,
                };
            },
        }

        // Логирование VWAP для отладки
        if let Some(vwap_val) = vwap_opt {
            debug!(
                "[{}] VWAP: {:.8}, Mid: {:.8}, Entry Style: {:?}",
                self.symbol, vwap_val, mid_price_f64, entry_style
            );
        }


        // Проверка лимита номинального риска (Задача 114)
        let current_pos = position.qty;
        let pending_same_side = self.order_manager.get_pending_size(side).await;
        
        if !self.risk_manager.check_notional_limit(
            current_pos, 
            pending_same_side, 
            qty, 
            side, 
            mid_price,
            self.bot_config.max_notional_usd,
        ) {
            debug!(
                "[{}] Signal suppressed: Notional limit exceeded (current: {}, pending: {}, order: {})",
                self.symbol, current_pos, pending_same_side, qty
            );
            return Ok(());
        }

        let mut final_qty = qty;

        // Задача №198: Консолидированная проверка рисков через статическую диспетчеризацию
        let intent = crate::risk::risk_manager::OrderIntent {
            side,
            price: price.to_f64().unwrap_or(0.0),
            qty: final_qty.to_f64().unwrap_or(0.0),
            timestamp: crate::utils::helpers::unix_ms(),
            filled_qty: 0.0,
        };

        match self.risk_manager.check_risk(&intent, position, orderbook) {
            crate::risk::risk_manager::RiskResult::Reject(reason) => {
                info!("[{}] Trade REJECTED by RiskManager: {}", self.symbol, reason);
                return Ok(());
            },
            crate::risk::risk_manager::RiskResult::AdjustSize(new_size_usd) => {
                let new_qty = Decimal::from_f64(new_size_usd / price.to_f64().unwrap_or(1.0))
                    .unwrap_or(Decimal::ZERO);
                let rounded_qty_f64 = crate::utils::helpers::round_down_to_step(
                    new_qty.to_f64().unwrap_or(0.0),
                    self.market_info.qty_step.to_f64().unwrap_or(0.01),
                );
                let rounded_qty = Decimal::from_f64(rounded_qty_f64).unwrap_or(Decimal::ZERO);
                
                if rounded_qty < self.market_info.min_order_qty {
                    info!("[{}] Trade REJECTED: Adjusted size below min_order_qty", self.symbol);
                    return Ok(());
                }
                
                info!("[{}] Trade ADJUSTED: {} -> {} by RiskManager", self.symbol, final_qty, rounded_qty);
                final_qty = rounded_qty;
            },
            crate::risk::risk_manager::RiskResult::Allow => {},
        }

        // Количество реально "ожидающих" ордеров (New / PartiallyFilled / Untracked)
        let active_orders = self.order_manager.count_pending_orders().await;

        // Риск-гейт по количеству открытых ордеров (Задача 074).
        // ВАЖНО: не блокируем сделки, которые уменьшают или закрывают позицию (Reduce-Only).
        let is_closing = if position.qty.is_zero() {
            false
        } else if position.qty.is_sign_positive() {
            side == OrderSide::Sell
        } else {
            side == OrderSide::Buy
        };

        if !is_closing && !self.risk_manager.check_orders_limit_gate(active_orders) {
            debug!(
                "[{}] Signal suppressed: Max Open Orders Gate is closed (pending: {})",
                self.symbol,
                active_orders
            );
            return Ok(());
        }

        // Проверка через Risk Gate с логикой урезания объема
        // Задача 177: Проверка адекватности цены (Extreme Price Deviation & Fat Finger Protection)
        let qty_f64_for_sanity = final_qty.to_f64().unwrap_or(0.0);
        let sanity_result = self.risk_manager.check_price_sanity(
            side,
            Some(price.to_f64().unwrap_or(0.0)),
            orderbook,
            qty_f64_for_sanity,
        );
        
        if let Err(e) = sanity_result {
            error!("[RISK] Price Deviation Block: {:?}", e);
            // Логируем в аудит (Задача 217)
            if let Some(logger) = &self.risk_manager.audit_logger {
                let _ = logger.log_risk_gate("PRICE_DEVIATION", true, &e.to_string());
            }
            if self.bot_config.risk.halt_on_extreme_deviation {
                error!("[{}] EMERGENCY STOP: Extreme price deviation detected. Entering emergency mode.", self.symbol);
                self.emergency_mode = true;
                return Err(anyhow::anyhow!("Emergency stop due to extreme price deviation"));
            }
            info!("[{}] Trade REJECTED: Price sanity check failed", self.symbol);
            return Ok(());
        }
        
        let gate_result = self.risk_manager.check_order_gate(
            side,
            final_qty,
            price,
            position,
            active_orders,
            current_pnl,
            mid_price,
            self.market_info.tick_size, // Задача 176
        );

        if let Err(e) = gate_result {
            let error_msg = e.to_string();
            
            // Если превышен лимит размера позиции, пробуем урезать объем
            if error_msg.contains("MaxPositionSize violated") {
                // Вычисляем максимально допустимый остаток (Задача 042: Приоритет у лимитов риска)
                let max_size = if let Some(max_pos) = self.bot_config.risk.max_position_size {
                    max_pos
                } else if let Some(max_pos) = self.bot_config.max_position_size {
                    max_pos
                } else {
                    // Если лимит не установлен, возвращаем ошибку
                    return Err(e);
                };

                // Расчет остатка с учетом текущей позиции
                let signed_qty = if side == OrderSide::Buy { final_qty } else { -final_qty };
                let projected_qty = position.qty + signed_qty;
                
                // Если проекция превышает лимит, урезаем
                if projected_qty.abs() > max_size {
                    let remaining = max_size - position.qty.abs();
                    
                    // Задача 137: Используем round_down_to_step для округления
                    let trimmed_qty_f64 = crate::utils::helpers::round_down_to_step(
                        remaining.to_f64().unwrap_or(0.0),
                        self.market_info.qty_step.to_f64().unwrap_or(0.01),
                    );
                    let trimmed_qty = Decimal::from_f64(trimmed_qty_f64).unwrap_or(Decimal::ZERO);
                    
                    if trimmed_qty >= self.market_info.min_order_qty {
                        warn!(
                            "[{}] Trimming size due to MaxPositionSize limit: {} -> {} (max: {}, current: {})",
                            self.symbol, final_qty, trimmed_qty, max_size, position.qty.abs()
                        );
                        final_qty = trimmed_qty;
                        
                        // Повторная проверка с урезанным объемом
                        self.risk_manager.check_order_gate(
                            side,
                            final_qty,
                            price,
                            position,
                            active_orders,
                            current_pnl,
                            mid_price,
                            self.market_info.tick_size, // Задача 176
                        )?;
                    } else {
                        debug!(
                            "[{}] Trimmed size {} is below min_order_qty {}. Cancelling trade.",
                            self.symbol, trimmed_qty, self.market_info.min_order_qty
                        );
                        return Ok(());
                    }
                } else {
                    // Если проекция не превышает лимит, возвращаем исходную ошибку
                    return Err(e);
                }
            } else {
                // Для других ошибок риска - отменяем сделку
                return Err(e);
            }
        }

        // 9. Перед выставлением нового ордера отменяем все текущие активные (Задача 148)
        // Это предотвращает накопление ордеров при быстрой смене сигналов.
        // Задача 232: Локальная проверка SMP (Self-Match Prevention)
        let mut smp_triggered = false;
        if self.bot_config.trading.local_smp_enabled {
            let has_opposite_orders = match side {
                OrderSide::Buy => self.order_manager.has_active_sell_orders().await,
                OrderSide::Sell => self.order_manager.has_active_buy_orders().await,
            };
            if has_opposite_orders {
                info!("[{}] Local SMP: Cancelling opposite orders before {:?}", self.symbol, side);
                smp_triggered = true;
            }
        }

        // Если сработал SMP ИЛИ просто есть активные ордера (Задача 148), отменяем всё
        if smp_triggered || active_orders > 0 {
            if !smp_triggered {
                debug!("[{}] Cancelling {} active orders before new trade", self.symbol, active_orders);
            }
            let _ = self.order_manager.cancel_all_orders(rest_client, &mut self.risk_manager, &self.bot_config, exchange_config).await;
        }

        // 9.3. Логика слайсинга для больших ордеров (Задача 168: Volume-Weighted Entry)
        // Если размер ордера превышает лимит участия в объеме, разбиваем его на части
        let mut slice_qty = final_qty;
        let mut remaining_qty_168 = None;
        
        if self.bot_config.slicing_enabled {
            // Рассчитываем максимальный объем на основе entry_participation_ratio
            let available_vol = orderbook.get_volume_at_best(data_side);
            let max_participation_vol = available_vol * self.bot_config.entry_participation_ratio;
            
            if max_participation_vol > 0.0 && final_qty.to_f64().unwrap_or(0.0) > max_participation_vol {
                let max_qty = Decimal::from_f64(max_participation_vol).unwrap_or(Decimal::ZERO);
                let remaining = final_qty - max_qty;
                
                info!(
                    "[{}] Slicing order due to participation limit: {} / {} total (available: {}, ratio: {})",
                    self.symbol, max_qty, final_qty, available_vol, self.bot_config.entry_participation_ratio
                );
                
                slice_qty = max_qty;
                remaining_qty_168 = Some(remaining);
            }
        }

        // 9.5. Логика нарезки крупных ордеров (Задача 149)
        let max_slice = self.bot_config.max_slice_size;
        let (_final_slice_qty, remaining_qty) = if max_slice > 0.0 {
            let max_slice_dec = Decimal::from_f64(max_slice).unwrap_or(Decimal::ZERO);
            if slice_qty > max_slice_dec {
                // Нарезка активирована: ограничиваем текущий ордер
                let remaining = slice_qty - max_slice_dec;
                info!(
                    "[{}] Slicing order: {} / {} total (max_slice_size: {})",
                    self.symbol, max_slice_dec, slice_qty, max_slice
                );
                (max_slice_dec, Some(remaining))
            } else {
                // Объем меньше лимита слайса - отправляем как есть
                (slice_qty, remaining_qty_168)
            }
        } else {
            // Нарезка отключена, но может быть remaining_qty_168 от participation_ratio
            (slice_qty, remaining_qty_168)
        };

        // Сохраняем состояние нарезки для последующих слайсов
        if let Some(remaining) = remaining_qty {
            self.pending_slice_qty = Some(remaining);
            self.pending_slice_side = Some(side);
            // Задача 201: Создаем SignalWithTimestamp для замера latency
            self.pending_slice_signal = Some(crate::ml::types::SignalWithTimestamp {
                signal,
                start_instant: std::time::Instant::now(),
            });
            self.pending_slice_probs = Some(*probs);
            info!(
                "[{}] Pending slices: remaining {} (side: {:?})",
                self.symbol, remaining, side
            );
        } else {
            // Очищаем состояние нарезки если это последний/единственный слайс
            self.pending_slice_qty = None;
            self.pending_slice_side = None;
            self.pending_slice_signal = None;
            self.pending_slice_probs = None;
        }

        info!(
            "Execution: Placing {:?} order (Post-Only: {}) for {} at price {} with qty {}", 
            side, use_post_only, self.symbol, price, slice_qty
        );
        
        // Задача 207: Проверка необходимости использования Iceberg-ордера
        // Если размер ордера значительно превышает объем на лучшем уровне, используем Iceberg
        let level_volume = if side == OrderSide::Buy {
            orderbook.take_snapshot().get_ask_volume_at_level(0)
        } else {
            orderbook.take_snapshot().get_bid_volume_at_level(0)
        };
        
        let should_use_iceberg = if level_volume > 0.0 {
            let size_ratio = slice_qty.to_f64().unwrap_or(0.0) / level_volume;
            size_ratio > 5.0 // Iceberg для ордеров >5x объема уровня
        } else {
            false
        };
        
        // 10. Отправка ордера на биржу
        let result = if should_use_iceberg {
            let total_size = slice_qty.to_f64().unwrap_or(0.0);
            let display_ratio = (level_volume / total_size).clamp(0.1, 0.3);
            
            info!(
                "[{}] Using Iceberg strategy: total={:.4}, display_ratio={:.2}",
                self.symbol, total_size, display_ratio
            );
            
            self.order_manager.place_iceberg_order(
                rest_client,
                &mut self.risk_manager,
                &self.bot_config,
                exchange_config,
                Some(orderbook),
                side,
                price,
                total_size,
                display_ratio,
                use_post_only,
                false, // reduce_only = false для открытия позиции
                mid_price,
            ).await
        } else {
            self.order_manager.place_limit_order(
                rest_client,
                &mut self.risk_manager,
                &self.bot_config,
                exchange_config,
                Some(orderbook),
                side,
                price,
                slice_qty,
                use_post_only,
                false, // reduce_only = false для открытия позиции
                mid_price, // Задача 202: Signal price для отслеживания expected_price
                None, // best_bid - не используется для обычных входов
                None, // best_ask - не используется для обычных входов
                None, // position_qty - не используется для обычных входов
            ).await
        };

        if let Err(e) = result {
            // Задача 232: Обработка "мягкой" ошибки SMP (Self-Match Prevention)
            if e.to_string() == "SMP_TRIGGERED" {
                debug!("[{}] Entry skipped due to SMP cooling down", self.symbol);
                return Ok(());
            }
            return Err(e);
        }

        // Сохраняем состояние после выставления ордера
        let _ = self.save_current_state();
        
        Ok(())
    }

    /// Обработка обновлений ордеров и логика Retry для Post-Only
    pub async fn handle_order_update(
        &mut self,
        update: OrderUpdate,
        rest_client: &BybitRestClient,
        exchange_config: &ExchangeConfig,
        best_bid: Decimal,
        best_ask: Decimal,
        orderbook: &crate::data::orderbook::OrderBook,
    ) -> Result<()> {
        let order_link_id = update.order_link_id.clone();
        
        // Задача 137: Создаем LotFilter из market_info для проверки пыли
        let lot_filter = crate::trading::types::LotFilter::from_market_info(&self.market_info);
        
        // 1. Извлекаем данные до обновления (чтобы знать side и qty)
        let (side, qty) = if let Some(order) = self.order_manager.get_by_client_id(&order_link_id).await {
            (order.side, order.qty)
        } else {
            // Если ордера нет в активных, пробуем обновить состояние (он мог быть в истории)
            if let Some((fill_event, realized_pnl, _position_closed, _entry_price)) = self.order_manager.update_order_state(&order_link_id, &update, &mut self.position_manager, &lot_filter, &mut self.risk_manager).await? {
                self.log_trade(fill_event, realized_pnl);
            }
            return Ok(());
        };

        let is_filled = update.status == OrderStatus::Filled;

        // 2. Обновляем состояние в OrderManager
        if let Some((fill_event, realized_pnl, position_closed, entry_price)) = self.order_manager.update_order_state(&order_link_id, &update, &mut self.position_manager, &lot_filter, &mut self.risk_manager).await? {
            // Задача 202: Добавляем ордер в очередь для захвата цены через 100мс
            if is_filled {
                let fill_time_ms = crate::utils::timestamp_ms();
                self.pending_price_checks.push_back((order_link_id.clone(), fill_time_ms));
                debug!("[{}] Order {} added to price check queue (fill_time: {})", self.symbol, order_link_id, fill_time_ms);
            }
            
            // Задача 202: Логирование метрик качества исполнения
            if let Some(order) = self.order_manager.get_by_client_id(&order_link_id).await {
                if let Some(log) = self.order_manager.create_execution_quality_log(&order, &self.bot_config.model_path) {
                    // Отправляем лог в фоновый worker через канал
                    let _ = self.execution_quality_tx.send(log).await;
                }
            }
            
            // Задача 204: Логирование влияния сделок на Mid-Price
            if self.bot_config.enable_impact_logging {
                let mid_price = (best_bid + best_ask) / Decimal::from(2);
                let mid_price_f64 = mid_price.to_f64().unwrap_or(0.0);
                
                if let Err(e) = self.order_manager.log_fill_with_mid_price(
                    &order_link_id,
                    fill_event.exec_qty.to_f64().unwrap_or(0.0),
                    mid_price_f64,
                    &self.bot_config.model_path,
                ).await {
                    warn!("[{}] Failed to log market impact for order {}: {}", self.symbol, order_link_id, e);
                }
            }
            
            // Задача 201: Логирование slippage
            // Получаем сигнал из pending_slice_signal если доступен
            if let Some(signal_with_ts) = &self.pending_slice_signal {
                let fill_price = fill_event.exec_price.to_f64().unwrap_or(0.0);
                let signal_price = self.last_signal_price; // Используем сохраненную цену сигнала
                
                // Вычисляем slippage в базисных пунктах
                let slippage_bps = if signal_price > 0.0 {
                    ((fill_price - signal_price) / signal_price) * 10000.0
                } else {
                    0.0
                };
                
                // Вычисляем spread в базисных пунктах
                let mid_price = (best_bid + best_ask) / Decimal::from(2);
                let spread_bps = if mid_price > Decimal::ZERO {
                    ((best_ask - best_bid) / mid_price) * Decimal::from(10000)
                } else {
                    Decimal::ZERO
                };
                
                // Вычисляем latency через Instant
                let latency_ms = signal_with_ts.start_instant.elapsed().as_millis() as u64;
                
                // Логируем в trades.csv
                let bot_path = std::path::Path::new("bots").join(&self.symbol);
                if let Err(e) = crate::utils::logger::log_trade_execution(
                    &bot_path,
                    fill_event.timestamp,
                    signal_price,
                    fill_price,
                    slippage_bps,
                    latency_ms,
                    spread_bps.to_f64().unwrap_or(0.0),
                ) {
                    warn!("[{}] Failed to log trade execution: {}", self.symbol, e);
                }
            }
            
            // Задача 164: Регистрируем тип филла (Maker/Taker)
            self.risk_manager.register_fill_type(fill_event.is_maker);
            
            // Задача 164: Инкрементируем Prometheus метрики для Maker/Taker fills
            if fill_event.is_maker {
                if let Some(counter) = crate::monitoring::prometheus::MAKER_FILL_COUNTER.get() {
                    counter.with_label_values(&[&self.symbol]).inc();
                }
            } else {
                if let Some(counter) = crate::monitoring::prometheus::TAKER_FILL_COUNTER.get() {
                    counter.with_label_values(&[&self.symbol]).inc();
                }
            }
            
            self.log_trade(fill_event.clone(), realized_pnl);
            
            // Задача 167: Активация Exchange-side TSL при первом исполнении ордера открытия
            if let Some(mut order) = self.order_manager.get_order_mut(&order_link_id).await {
                if order.tsl_trailing_stop.is_some() {
                    let req = crate::trading::types::TradingStopRequest {
                        category: exchange_config.bybit.category.clone(),
                        symbol: self.symbol.clone(),
                        trailing_stop: order.tsl_trailing_stop.take(), // .take() гарантирует выполнение только один раз
                        active_price: order.tsl_active_price.take(),
                        position_idx: self.bot_config.position_idx,
                    };
                    
                    info!("[{}] Activating Exchange-side TSL for order {}: ts={:?}, active={:?}", 
                        self.symbol, order_link_id, req.trailing_stop, req.active_price);
                        
                    if let Err(e) = rest_client.set_trading_stop(&req).await {
                        error!("[{}] Failed to activate Exchange-side TSL: {}", self.symbol, e);
                    }
                }
            }
            
            // Задача 173: Обновление статистики волатильности PnL (Поддержка частичных закрытий)
            if let Some(trade_pnl) = realized_pnl {
                if fill_event.exec_qty > Decimal::ZERO {
                    // Используем entry_price для номинала (если его нет, fallback на exec_price)
                    let base_price = entry_price.unwrap_or(fill_event.exec_price);
                    if base_price > Decimal::ZERO {
                        let notional = fill_event.exec_qty * base_price;
                        let pnl_bps = (trade_pnl / notional) * Decimal::from(10000);
                        if let Some(pnl_bps_f64) = pnl_bps.to_f64() {
                            self.risk_manager.update_pnl_stats(pnl_bps_f64);
                        }
                    }
                }

                // Обновляем серию убытков ТОЛЬКО при полном закрытии позиции (Задача 115, исправление Multiple Fills Bug)
                if position_closed {
                    let mut state_guard = self.state.lock().await;
                    let state = &mut *state_guard;
                    self.position_manager.update_streak(trade_pnl, &mut state.loss_streak, &mut state.last_loss_timestamp_ms, update.timestamp);
                }
            }
            
            // Регистрируем сделку только при полном исполнении ордера (Задача 112)
            if is_filled {
                self.risk_manager.register_fill(update.timestamp as i64);
                
                // Задача 207: Проверка Iceberg Refill
                let mid_price = (best_bid + best_ask) / Decimal::from(2);
                let mid_price_f64 = mid_price.to_f64().unwrap_or(0.0);
                
                // Проверяем, является ли это Iceberg-ордером
                let is_iceberg = {
                    if let Some(order) = self.order_manager.get_by_client_id(&order_link_id).await {
                        order.iceberg_total_size.is_some()
                    } else {
                        false
                    }
                };
                
                if is_iceberg {
                    // Обновляем iceberg_filled_total перед проверкой refill
                    if let Some(mut order_mut) = self.order_manager.get_order_mut(&order_link_id).await {
                        order_mut.iceberg_filled_total += order_mut.executed_qty;
                    }
                    
                    // Получаем ордер для передачи в check_iceberg_refill
                    if let Some(order) = self.order_manager.get_by_client_id(&order_link_id).await {
                        let order_clone = order.clone();
                        // Проверяем необходимость refill
                        match self.order_manager.check_iceberg_refill(
                            &order_clone,
                            mid_price_f64,
                            &self.bot_config.sor,
                            rest_client,
                            exchange_config,
                            &mut self.risk_manager,
                            &self.bot_config,
                            Some(orderbook),
                        ).await {
                            Ok(Some(new_link_id)) => {
                                info!("[{}] Iceberg refill placed: {}", self.symbol, new_link_id);
                            }
                            Ok(None) => {
                                // Refill не требуется или прерван
                            }
                            Err(e) => {
                                warn!("[{}] Iceberg refill failed: {}", self.symbol, e);
                            }
                        }
                    }
                }
                
                // Задача 149: Продолжение нарезки после исполнения слайса
                if let Some(remaining_qty) = self.pending_slice_qty {
                    if let (Some(slice_side), Some(_slice_signal), Some(_slice_probs)) = 
                        (self.pending_slice_side, self.pending_slice_signal.clone(), self.pending_slice_probs) {
                        
                        info!(
                            "[{}] Slice filled. Continuing slicing: remaining {} (side: {:?})",
                            self.symbol, remaining_qty, slice_side
                        );
                        
                        // Рассчитываем размер следующего слайса
                        let max_slice = self.bot_config.max_slice_size;
                        let max_slice_dec = Decimal::from_f64(max_slice).unwrap_or(Decimal::ZERO);
                        
                        let (next_slice_qty, new_remaining) = if remaining_qty > max_slice_dec {
                            (max_slice_dec, Some(remaining_qty - max_slice_dec))
                        } else {
                            (remaining_qty, None)
                        };
                        
                        // Обновляем состояние нарезки
                        self.pending_slice_qty = new_remaining;
                        if new_remaining.is_none() {
                            // Это последний слайс - очищаем состояние
                            self.pending_slice_side = None;
                            self.pending_slice_signal = None;
                            self.pending_slice_probs = None;
                            info!("[{}] Last slice: {}", self.symbol, next_slice_qty);
                        } else {
                            info!(
                                "[{}] Next slice: {} / {} remaining",
                                self.symbol, next_slice_qty, new_remaining.unwrap()
                            );
                        }
                        
                        // Определяем цену для следующего слайса
                        let mid_price = (best_bid + best_ask) / Decimal::from(2);
                        let price = match slice_side {
                            OrderSide::Buy => best_bid,
                            OrderSide::Sell => best_ask,
                        };
                        
                        // Проверка цены ордера
                        match self.risk_manager.validate_order_price(price, mid_price, self.market_info.tick_size) {
                            Ok(valid) if valid => {
                                // Проверка риск-гейта для следующего слайса
                                let position = self.position_manager.get_position();
                                let current_pnl = position.realized_pnl + position.unrealized_pnl;
                                let active_orders = self.order_manager.count_pending_orders().await;
                                
                                match self.risk_manager.check_order_gate(
                                    slice_side,
                                    next_slice_qty,
                                    price,
                                    position,
                                    active_orders,
                                    current_pnl,
                                    mid_price,
                                    self.market_info.tick_size, // Задача 176
                                ) {
                                    Ok(_) => {
                                        // Задача 149: Rate Limiting между слайсами
                                        // Добавляем задержку на основе настроек биржи
                                        let rate_limit_delay_ms = if exchange_config.rate_limits.rest_requests_per_second > 0 {
                                            // Рассчитываем минимальный интервал между ордерами
                                            // rest_requests_per_second - это количество запросов в секунду
                                            (1000 / exchange_config.rate_limits.rest_requests_per_second).max(100)
                                        } else {
                                            100 // Минимальная задержка 100ms
                                        };
                                        
                                        debug!(
                                            "[{}] Rate limiting: waiting {}ms before next slice",
                                            self.symbol, rate_limit_delay_ms
                                        );
                                        
                                        // Задача 149/168: Неблокирующая задержка вместо sleep
                                        if self.last_slice_time.elapsed() < tokio::time::Duration::from_millis(rate_limit_delay_ms) {
                                            debug!("[{}] Slicing rate limited: skipping this tick", self.symbol);
                                            return Ok(());
                                        }
                                        self.last_slice_time = Instant::now();
                                        
                                        // Выставляем следующий слайс
                                        if let Err(e) = self.order_manager.place_limit_order(
                                            rest_client,
                                            &mut self.risk_manager,
                                            &self.bot_config,
                                            exchange_config,
                                            None,
                                            slice_side,
                                            price,
                                            next_slice_qty,
                                            self.bot_config.post_only,
                                            false, // reduce_only = false для открытия позиции
                                            mid_price, // Задача 201: Signal price для анализа slippage
                                            None, // best_bid - не используется для обычных входов
                                            None, // best_ask - не используется для обычных входов
                                            None, // position_qty - не используется для обычных входов
                                        ).await {
                                            error!("[{}] Failed to place next slice: {}", self.symbol, e);
                                            // Очищаем состояние нарезки при ошибке
                                            self.pending_slice_qty = None;
                                            self.pending_slice_side = None;
                                            self.pending_slice_signal = None;
                                            self.pending_slice_probs = None;
                                        }
                                    }
                                    Err(e) => {
                                        warn!("[{}] Risk gate blocked next slice: {}. Stopping slicing.", self.symbol, e);
                                        // Очищаем состояние нарезки
                                        self.pending_slice_qty = None;
                                        self.pending_slice_side = None;
                                        self.pending_slice_signal = None;
                                        self.pending_slice_probs = None;
                                    }
                                }
                            }
                            _ => {
                                warn!("[{}] Price validation failed for next slice. Stopping slicing.", self.symbol);
                                // Очищаем состояние нарезки
                                self.pending_slice_qty = None;
                                self.pending_slice_side = None;
                                self.pending_slice_signal = None;
                                self.pending_slice_probs = None;
                            }
                        }
                    }
                }
            }
        }

        // 4. Сохраняем состояние после любого обновления (исполнение, отмена и т.д.)
        if let Err(e) = self.save_current_state().await {
            error!("[{}] Failed to save state after order update: {}", self.symbol, e);
        }

        // 5. Проверка глобальных рисков после обновления PnL/позиции (Заметка Claude)
        let pos = self.position_manager.get_position();
        let current_pnl = pos.realized_pnl + pos.unrealized_pnl;
        
        // Задача 042: Обновление эквити для отслеживания пикового значения и просадки
        self.risk_manager.update_equity(current_pnl);
        
        if let Err(e) = self.risk_manager.check_global_risk(current_pnl) {
            let error_msg = e.to_string();
            if error_msg.contains("HardStop") || error_msg.contains("drawdown") {
                // ЯВНЫЙ HARDSTOP TRIGGER — выполняем экстренные действия
                tracing::error!("[{}] HARD STOP TRIGGERED: {}", self.symbol, error_msg);
            } else {
                warn!("[{}] Risk breach detected after order update: {}. Triggering emergency exit.", self.symbol, e);
            }
            // Очищаем состояние нарезки при аварийном выходе
            self.pending_slice_qty = None;
            self.pending_slice_side = None;
            self.pending_slice_signal = None;
            self.pending_slice_probs = None;
            
            if let Err(panic_err) = self.emergency_market_close(rest_client, exchange_config).await {
                error!("[{}] CRITICAL: Emergency exit failed after risk breach: {}", self.symbol, panic_err);
            }
            return Err(e);
        }

        // 3. Задача 164: Если ордер был отклонен по Post-Only, обрабатываем с экспоненциальным увеличением offset
        let is_post_only_rejected = self.order_manager.get_history().await.iter()
            .any(|o| o.link_id == order_link_id && matches!(o.state, OrderState::Rejected(ref r) if r.contains("PostOnly")));

        if is_post_only_rejected {
            // Используем новый метод handle_post_only_reject
            let base_offset_ticks = self.bot_config.maker_offset_step_ticks;
            let max_rejects = self.bot_config.max_post_only_rejects;
            
            if let Some(new_offset_ticks) = self.order_manager.handle_post_only_reject(&order_link_id, base_offset_ticks, max_rejects) {
                // Повторяем с увеличенным offset
                debug!("[{}] Post-Only rejected for {}. Retrying with offset {} ticks...", 
                    self.symbol, order_link_id, new_offset_ticks);

                // Перед повторной отправкой проверяем лимит открытых ордеров (Задача 074)
                let pending = self.order_manager.count_pending_orders().await;
                if !self.risk_manager.check_orders_limit_gate(pending) {
                    warn!(
                        "[{}] Max Open Orders Gate closed (pending: {}). Skipping Post-Only retry for {}",
                        self.symbol,
                        pending,
                        order_link_id
                    );
                    return Ok(());
                }
                
                // Вычисляем новую цену с увеличенным offset
                let offset = self.market_info.tick_size * Decimal::from(new_offset_ticks);
                let new_price = match side {
                    OrderSide::Buy => best_bid - offset,
                    OrderSide::Sell => best_ask + offset,
                };

                // Проверка цены ордера на отклонение и кратность шагу цены (Задача 075)
                let mid_price = (best_bid + best_ask) / Decimal::from(2);
                match self.risk_manager.validate_order_price(new_price, mid_price, self.market_info.tick_size) {
                    Ok(valid) => {
                        if !valid {
                            debug!("[{}] Post-Only retry order price validation failed: Price {} deviates too much from mid price {}",
                                   self.symbol, new_price, mid_price);
                            return Ok(());
                        }
                    },
                    Err(e) => {
                        debug!("[{}] Post-Only retry order price validation error: {}", self.symbol, e);
                        return Ok(());
                    }
                }

                let new_link_id = self.order_manager.place_limit_order(
                    rest_client,
                    &mut self.risk_manager,
                    &self.bot_config,
                    exchange_config,
                    None,
                    side,
                    new_price,
                    Decimal::from_f64(qty).unwrap_or(Decimal::ZERO),
                    true, // Снова пробуем Post-Only
                    false, // reduce_only = false для открытия позиции
                    mid_price, // Задача 201: Signal price для анализа slippage
                    None, // best_bid - не используется для обычных входов
                    None, // best_ask - не используется для обычных входов
                    None, // position_qty - не используется для обычных входов
                ).await?;

                // Сохраняем состояние после выставления нового ордера
                let _ = self.save_current_state();

                // Копируем счетчик режектов в новый ордер
                if let Some(old_order) = self.order_manager.get_history().await.iter().find(|o| o.link_id == order_link_id) {
                    if let Some(mut new_order) = self.order_manager.get_order_mut(&new_link_id).await {
                        new_order.post_only_reject_count = old_order.post_only_reject_count;
                    }
                }
            } else {
                // Лимит режектов достигнут - переключаемся на Taker
                warn!("[{}] Post-Only retries exhausted for {}. Falling back to Taker mode.", self.symbol, order_link_id);
                
                let fallback_price = match side {
                    OrderSide::Buy => best_ask, 
                    OrderSide::Sell => best_bid,
                };

                // Проверка цены ордера на отклонение и кратность шагу цены (Задача 075)
                let mid_price = (best_bid + best_ask) / Decimal::from(2);
                match self.risk_manager.validate_order_price(fallback_price, mid_price, self.market_info.tick_size) {
                    Ok(valid) => {
                        if !valid {
                            debug!("[{}] Taker fallback order price validation failed: Price {} deviates too much from mid price {}",
                                   self.symbol, fallback_price, mid_price);
                            return Ok(());
                        }
                    },
                    Err(e) => {
                        debug!("[{}] Taker fallback order price validation error: {}", self.symbol, e);
                        return Ok(());
                    }
                }

                // Перед Taker-фоллбеком также проверяем лимит открытых ордеров
                let pending = self.order_manager.count_pending_orders().await;
                if !self.risk_manager.check_orders_limit_gate(pending) {
                    warn!(
                        "[{}] Max Open Orders Gate closed (pending: {}). Skipping Taker fallback for {}",
                        self.symbol,
                        pending,
                        order_link_id
                    );
                } else {
                    self.order_manager.place_limit_order(
                        rest_client,
                        &mut self.risk_manager,
                        &self.bot_config,
                        exchange_config,
                        None,
                        side,
                        fallback_price,
                        Decimal::from_f64(qty).unwrap_or(Decimal::ZERO),
                        false, // GTC (Taker)
                        false, // reduce_only = false для открытия позиции
                        mid_price, // Задача 201: Signal price для анализа slippage
                        None, // best_bid - не используется для обычных входов
                        None, // best_ask - не используется для обычных входов
                        None, // position_qty - не используется для обычных входов
                    ).await?;

                    let _ = self.save_current_state();
                }
            }
        }

        Ok(())
    }

    /// Проверка таймаутов для "зависших" лимитных ордеров
    /// Задача 164: Добавлена логика Dynamic Re-pegging и Rebate Timeout
    pub async fn check_timeouts(
        &mut self,
        rest_client: &BybitRestClient,
        exchange_config: &ExchangeConfig,
        best_bid: Decimal,
        best_ask: Decimal,
    ) -> Result<()> {
        // Задача 164: Проверка необходимости ре-пеггинга Maker-ордеров
        let to_repeg = self.order_manager.check_repeg_needed(
            best_bid,
            best_ask,
            self.market_info.tick_size,
            self.bot_config.repeg_threshold_ticks,
        );
        
        for (order_link_id, new_price) in to_repeg {
            info!("[{}] Re-pegging order {} to price {}", self.symbol, order_link_id, new_price);
            
            // Используем amend_active_order для обновления цены
            if let Err(e) = self.order_manager.amend_active_order(
                rest_client,
                &mut self.risk_manager,
                &self.bot_config,
                exchange_config,
                None,
                &order_link_id,
                Some(new_price),
                None, // qty не меняем
                None, // trigger_price не меняем
            ).await {
                warn!("[{}] Failed to repeg order {}: {}", self.symbol, order_link_id, e);
            }
        }
        
        // Задача 164: Проверка Rebate Timeout для Maker-ордеров
        let timed_out_maker = self.order_manager.check_rebate_timeout(
            self.bot_config.rebate_wait_timeout_ms,
        );
        
        for order_link_id in timed_out_maker {
            warn!("[{}] Maker order {} exceeded rebate timeout. Cancelling and switching to Taker.", 
                self.symbol, order_link_id);
            
            let (side, qty) = {
                let o = self.order_manager.get_by_client_id(&order_link_id).await.unwrap();
                let remaining = o.remaining_qty();
                (o.side, Decimal::from_f64_retain(remaining).unwrap_or(Decimal::ZERO))
            };

            // 1. Отменяем старый Maker-ордер
            self.order_manager.cancel_order(rest_client, &mut self.risk_manager, &self.bot_config, exchange_config, &order_link_id, true).await?;

            // 2. Выставляем Taker-ордер на остаток
            if qty >= self.market_info.min_order_qty {
                let taker_price = match side {
                    OrderSide::Buy => best_ask,
                    OrderSide::Sell => best_bid,
                };

                let mid_price = (best_bid + best_ask) / Decimal::from(2);
                match self.risk_manager.validate_order_price(taker_price, mid_price, self.market_info.tick_size) {
                    Ok(valid) => {
                        if !valid {
                            debug!("[{}] Rebate timeout Taker order price validation failed", self.symbol);
                            continue;
                        }
                    },
                    Err(e) => {
                        debug!("[{}] Rebate timeout Taker order price validation error: {}", self.symbol, e);
                        continue;
                    }
                }

                let pending = self.order_manager.count_pending_orders().await;
                if !self.risk_manager.check_orders_limit_gate(pending) {
                    warn!("[{}] Max Open Orders Gate closed. Skipping rebate timeout Taker order", self.symbol);
                } else {
                    self.order_manager.place_limit_order(
                        rest_client,
                        &mut self.risk_manager,
                        &self.bot_config,
                        exchange_config,
                        None,
                        side,
                        taker_price,
                        qty,
                        false, // GTC (Taker)
                        false, // reduce_only = false для открытия позиции
                        mid_price, // Задача 201: Signal price для анализа slippage
                        None, // best_bid - не используется для обычных входов
                        None, // best_ask - не используется для обычных входов
                        None, // position_qty - не используется для обычных входов
                    ).await?;

                    let _ = self.save_current_state();
                }
            }
        }
        
        // Задача 208: Логика переключения Passive -> Aggressive
        // Проверяем ордера, которые готовы к переключению
        let now = timestamp_ms();
        let base_switch_timeout = self.bot_config.sor.switch_base_timeout_ms;
        let max_switches = self.bot_config.sor.max_switches_per_signal;
        
        let to_switch: Vec<(String, f32)> = self.order_manager.get_active_orders().await
            .iter()
            .filter(|(_, o)| {
                // Пропускаем Post-Only ордера (они обрабатываются выше через rebate_timeout)
                !o.is_post_only && 
                o.state == OrderState::Active && 
                o.switch_count < max_switches
            })
            .filter_map(|(id, o)| {
                // Модулируем timeout по urgency каждого ордера
                let (effective_switch_timeout, _) = self.calculate_modulated_switch_params(
                    base_switch_timeout,
                    self.bot_config.sor.switch_base_distance_bps,
                    o.urgency,
                );
                
                if (now - o.created_at) > effective_switch_timeout {
                    Some((id.clone(), o.urgency))
                } else {
                    None
                }
            })
            .collect();

        for (id, urgency) in to_switch {
            info!("[{}] Order {} ready for Passive->Aggressive switch (urgency: {})", 
                  self.symbol, id, urgency);
            
            // Вызываем handle_switch_trigger для безопасного переключения
            match self.order_manager.handle_switch_trigger(
                rest_client,
                &mut self.risk_manager,
                &self.bot_config,
                exchange_config,
                &id,
                max_switches,
            ).await {
                Ok(Some((side, remaining, _original_price))) => {
                    // Переключение успешно, выставляем агрессивный ордер
                    let qty = Decimal::from_f64_retain(remaining).unwrap_or(Decimal::ZERO);

                    if qty >= self.market_info.min_order_qty {
                        // Выставляем Market или Cross-Limit ордер
                        let aggressive_price = match side {
                            OrderSide::Buy => best_ask,
                            OrderSide::Sell => best_bid,
                        };

                        let mid_price = (best_bid + best_ask) / Decimal::from(2);
                        
                        // Проверка цены ордера
                        match self.risk_manager.validate_order_price(aggressive_price, mid_price, self.market_info.tick_size) {
                            Ok(valid) => {
                                if !valid {
                                    debug!("[{}] Aggressive order price validation failed", self.symbol);
                                    continue;
                                }
                            },
                            Err(e) => {
                                debug!("[{}] Aggressive order price validation error: {}", self.symbol, e);
                                continue;
                            }
                        }

                        // Проверяем лимит открытых ордеров
                        let pending = self.order_manager.count_pending_orders().await;
                        if !self.risk_manager.check_orders_limit_gate(pending) {
                            warn!("[{}] Max Open Orders Gate closed. Skipping aggressive order", self.symbol);
                        } else {
                            // Выставляем агрессивный ордер (Taker)
                            self.order_manager.place_limit_order(
                                rest_client,
                                &mut self.risk_manager,
                                &self.bot_config,
                                exchange_config,
                                None,
                                side,
                                aggressive_price,
                                qty,
                                false, // GTC (Taker)
                                false, // reduce_only = false
                                mid_price,
                                None, // best_bid - не используется для обычных входов
                                None, // best_ask - не используется для обычных входов
                                None, // position_qty - не используется для обычных входов
                            ).await?;

                            info!("[{}] Aggressive order placed for {} at price {}", self.symbol, id, aggressive_price);
                            let _ = self.save_current_state();
                        }
                    }
                }
                Ok(None) => {
                    // Переключение не требуется или ордер уже исполнен
                }
                Err(e) => {
                    warn!("[{}] Switch trigger failed for order {}: {}", self.symbol, id, e);
                }
            }
        }

        // Оригинальная логика таймаута для обычных лимитных ордеров
        let timeout = self.bot_config.limit_timeout_ms;

        let to_cancel: Vec<String> = self.order_manager.get_active_orders().await
            .iter()
            .filter(|(_, o)| {
                // Пропускаем Post-Only ордера (они обрабатываются выше через rebate_timeout)
                // Пропускаем ордера, которые уже переключались
                !o.is_post_only && o.state == OrderState::Active && o.switch_count == 0 && (now - o.created_at) > timeout
            })
            .map(|(id, _)| id.clone())
            .collect();

        // Задача 209: Оптимизация массовых отмен
        // Если количество ордеров для отмены > mass_cancel_threshold, используем cancel-all
        if to_cancel.len() > exchange_config.mass_cancel_threshold {
            warn!("[{}] Mass cancellation triggered: {} orders > threshold {}. Using cancel-all endpoint.",
                  self.symbol, to_cancel.len(), exchange_config.mass_cancel_threshold);
            
            let _ = self.order_manager.cancel_all_orders(rest_client, &mut self.risk_manager, &self.bot_config, exchange_config).await;
            
            // После cancel-all, пропускаем цикл отмены, так как все ордера уже отменены
            return Ok(());
        }

        for id in to_cancel {
            warn!("[{}] Order {} timed out after {}ms. Cancelling and falling back to GTC.", self.symbol, id, timeout);
            
            let (side, qty) = {
                let o = self.order_manager.get_by_client_id(&id).await.unwrap();
                let remaining = o.remaining_qty();
                (o.side, Decimal::from_f64_retain(remaining).unwrap_or(Decimal::ZERO))
            };

            // 1. Отменяем старый ордер
            self.order_manager.cancel_order(rest_client, &mut self.risk_manager, &self.bot_config, exchange_config, &id, true).await?;

            // 2. Выставляем гарантированный ордер на остаток
            if qty >= self.market_info.min_order_qty {
                let fallback_price = match side {
                    OrderSide::Buy => best_ask,
                    OrderSide::Sell => best_bid,
                };

                // Проверка цены ордера на отклонение и кратность шагу цены (Задача 075)
                let mid_price = (best_bid + best_ask) / Decimal::from(2);
                match self.risk_manager.validate_order_price(fallback_price, mid_price, self.market_info.tick_size) {
                    Ok(valid) => {
                        if !valid {
                            debug!("[{}] Timeout fallback order price validation failed: Price {} deviates too much from mid price {}",
                                   self.symbol, fallback_price, mid_price);
                            return Ok(());
                        }
                    },
                    Err(e) => {
                        debug!("[{}] Timeout fallback order price validation error: {}", self.symbol, e);
                        return Ok(());
                    }
                }

                // Перед выставлением нового GTC-ордера проверяем лимит открытых ордеров (Задача 074)
                let pending = self.order_manager.count_pending_orders().await;
                if !self.risk_manager.check_orders_limit_gate(pending) {
                    warn!(
                        "[{}] Max Open Orders Gate closed (pending: {}). Skipping timeout fallback order for {}",
                        self.symbol,
                        pending,
                        id
                    );
                } else {
                    self.order_manager.place_limit_order(
                        rest_client,
                        &mut self.risk_manager,
                        &self.bot_config,
                        exchange_config,
                        None,
                        side,
                        fallback_price,
                        qty,
                        false, // GTC
                        false, // reduce_only = false для открытия позиции
                        mid_price, // Задача 201: Signal price для анализа slippage
                        None, // best_bid - не используется для обычных входов
                        None, // best_ask - не используется для обычных входов
                        None, // position_qty - не используется для обычных входов
                    ).await?;

                    let _ = self.save_current_state();
                }
            }
        }

        Ok(())
    }

    /// Задача 210: Расчет адаптивного порога отмены ордеров
    /// Формула: adaptive_threshold = base_threshold / (1.0 + (volatility * vol_multiplier) + (spread * spread_multiplier))
    /// 
    /// ПАРАМЕТРЫ УСТАНОВЛЕНЫ МАКСИМАЛЬНО МЯГКО:
    /// - base_threshold_bps: 500 bps (5%) - очень большой базовый порог
    /// - vol_multiplier: 0.01 - минимальное влияние волатильности
    /// - spread_multiplier: 0.001 - минимальное влияние спреда
    /// - min_threshold: 100 bps (1%), max_threshold: 1000 bps (10%)
    /// 
    /// Это гарантирует, что механизм не будет мешать торговле на разных монетах,
    /// включая те с большими спредами (альткойны, низколиквидные пары).
    fn calculate_adaptive_threshold(&self, orderbook: &crate::data::orderbook::OrderBook) -> Decimal {
        // Если адаптивные пороги выключены, используем статический порог
        if !self.bot_config.adaptive_thresholds_enabled {
            return self.bot_config.chase_threshold_bps;
        }

        // Получаем текущую волатильность и спред из стакана (Задача 191)
        let volatility_bps = orderbook.get_volatility_bps();
        let spread_bps = orderbook.get_spread_bps();
        
        // Для volatility_level используем упрощенную логику на основе volatility_bps
        let volatility_level = if volatility_bps < 50.0 {
            "Low"
        } else if volatility_bps < 150.0 {
            "Medium"
        } else {
            "High"
        };
        
        let buffer_fill_percent = 100.0; // Снапшот всегда полный

        // Применяем инверсную формулу: порог уменьшается при росте риска
        let base = self.bot_config.base_threshold_bps;
        let denominator = 1.0 
            + (volatility_bps * self.bot_config.vol_multiplier) 
            + (spread_bps * self.bot_config.spread_multiplier_adaptive);

        let adaptive_threshold = if denominator > 0.0 {
            base / denominator
        } else {
            base
        };

        // Применяем жесткие границы
        let clamped = adaptive_threshold
            .max(self.bot_config.min_threshold_bps)
            .min(self.bot_config.max_threshold_bps);

        // Задача 210: Логирование волатильности и спреда для мониторинга
        // Классификация уровней волатильности (в bps, за последние 500 тиков)
        // Логируем только если волатильность высокая или экстремальная
        if volatility_bps >= 500.0 {
            warn!(
                "[{}] ADAPTIVE_THRESHOLD: volatility={:.1} bps ({}), buffer_fill={:.0}%, spread={:.1} bps, threshold={:.1} bps -> {:.1} bps | Consider adjusting vol_multiplier or spread_multiplier",
                self.symbol,
                volatility_bps,
                volatility_level,
                buffer_fill_percent,
                spread_bps,
                base,
                clamped
            );
        } else if volatility_bps >= 200.0 {
            // Логируем на уровне debug для средней волатильности
            debug!(
                "[{}] ADAPTIVE_THRESHOLD: volatility={:.1} bps ({}), buffer_fill={:.0}%, spread={:.1} bps, threshold={:.1} bps -> {:.1} bps",
                self.symbol,
                volatility_bps,
                volatility_level,
                buffer_fill_percent,
                spread_bps,
                base,
                clamped
            );
        }

        Decimal::from_f64(clamped).unwrap_or(self.bot_config.chase_threshold_bps)
    }

    /// Основная логика погони (Order Chasing)
    pub async fn check_and_chase(
        &mut self,
        rest_client: &BybitRestClient,
        exchange_config: &ExchangeConfig,
        best_bid: Decimal,
        best_ask: Decimal,
        orderbook: &crate::data::orderbook::OrderBook,  // Задача 210: Добавлен для адаптивных порогов
    ) -> Result<()> {
        let now = timestamp_ms() as i64;
        let mid = (best_bid + best_ask) / Decimal::from(2);
        
        // Задача 210: Используем адаптивный порог вместо статического
        let th_bps = self.calculate_adaptive_threshold(orderbook);
        
        // Собираем ID ордеров, требующих переставления
        let mut to_chase = Vec::new();
        
        for (id, order) in self.order_manager.get_active_orders().await {
            if order.chase_count >= self.bot_config.chase_max_attempts || 
               now - order.last_chase_ts < self.bot_config.chase_interval_ms as i64 {
                continue;
            }

            let th_amount = mid * th_bps / Decimal::from(10000);
            let mut needs_chase = false;

            // Конвертируем цену ордера в Decimal для сравнения
            let order_price_decimal = Decimal::from_f64(order.price).unwrap_or(Decimal::ZERO);

            if order.side == OrderSide::Buy && order_price_decimal < (best_bid - th_amount) {
                needs_chase = true;
            } else if order.side == OrderSide::Sell && order_price_decimal > (best_ask + th_amount) {
                needs_chase = true;
            }

            if needs_chase && self.is_signal_still_valid(order.side).await {
                let new_price = self.calc_new_peg(order.side, mid, best_bid, best_ask);
                
                // Фильтр по VWAP (если включен в логике is_price_fair)
                if !self.is_price_fair(new_price, order.side) {
                    continue;
                }
                
                // Валидация новой цены по риск-гейтам
                if let Ok(valid) = self.risk_manager.validate_order_price(new_price, mid, self.market_info.tick_size) {
                    if !valid { continue; }
                } else {
                    continue;
                }

                to_chase.push((id.to_string(), new_price));
            }
        }

        for (id, new_price) in to_chase {
            let current_price = self.order_manager.get_by_client_id(&id).await.map(|o| o.price).unwrap_or_default();
            info!("[{}] Chasing order {}: price {} -> {}", &self.symbol, id, current_price, new_price);
            
            match self.order_manager.amend_active_order(rest_client, &mut self.risk_manager, &self.bot_config, exchange_config, None, &id, Some(new_price), None, None).await {
                Ok(_) => {
                    if let Some(mut order) = self.order_manager.get_order_mut(&id).await {
                        order.chase_count += 1;
                        order.last_chase_ts = now;
                    }
                    let _ = self.save_current_state();
                }
                Err(e) => {
                    let err_msg = e.to_string();
                    if err_msg.contains("not found") || err_msg.contains("Order not exists") {
                        // Ордер уже исполнился или отменен
                        warn!("[{}] Chase failed: order {} not found. Might be filled.", &self.symbol, id);
                    } else {
                        // Если amend не поддерживается или другая ошибка — пробуем переставить через Cancel + Replace
                        warn!("[{}] Amend failed for {}: {}. Attempting Cancel + Replace...", &self.symbol, id, &err_msg);
                        self.cancel_and_replace_chase(rest_client, exchange_config, id.to_string(), new_price, mid).await?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Вспомогательная функция расчета цены для перепривязки
    fn calc_new_peg(&self, side: OrderSide, mid: Decimal, best_bid: Decimal, best_ask: Decimal) -> Decimal {
        use crate::config::types::ChaseMode;
        
        let offset = mid * self.bot_config.chase_distance_bps / Decimal::from(10000);
        let raw_price = match self.bot_config.chase_mode {
            ChaseMode::ToBest => if side == OrderSide::Buy { best_bid } else { best_ask },
            ChaseMode::InsideSpread => {
                if side == OrderSide::Buy { best_bid + offset } else { best_ask - offset }
            },
            ChaseMode::ToVWAP => {
                let vwap = self.price_stats.get_vwap(None);
                if vwap.is_zero() { mid } else { vwap }
            },
        };

        // Округление до tick_size
        let tick = self.market_info.tick_size;
        if tick.is_zero() {
            raw_price
        } else {
            (raw_price / tick).round_dp(0) * tick
        }
    }

    /// Проверка актуальности сигнала нейросети для данного направления
    async fn is_signal_still_valid(&self, side: OrderSide) -> bool {
        // Используем динамические пороги (Задача 115)
        let current_streak = {
            let state_guard = self.state.lock().await;
            state_guard.loss_streak
        };
        let dynamic_threshold = self.risk_manager.get_effective_threshold(current_streak, &self.bot_config) as f32;
        
        match side {
            OrderSide::Buy => self.last_probabilities[1] > dynamic_threshold, // Index 1: Up
            OrderSide::Sell => self.last_probabilities[2] > dynamic_threshold, // Index 2: Down
        }
    }

    /// Проверка справедливости цены (например, не покупать слишком дорого относительно VWAP)
    fn is_price_fair(&self, price: Decimal, side: OrderSide) -> bool {
        if !self.bot_config.use_vwap_filter {
            return true;
        }

        let vwap = self.price_stats.get_vwap(None);
        if vwap.is_zero() || self.price_stats.is_empty() {
            return true;
        }

        let threshold_bps = self.bot_config.vwap_filter_threshold_bps;
        let threshold = vwap * threshold_bps / Decimal::from(10000);
        
        match side {
            OrderSide::Buy => price <= (vwap + threshold),
            OrderSide::Sell => price >= (vwap - threshold),
        }
    }

    /// Экстренный Cancel + Replace при ошибках Amend
    async fn cancel_and_replace_chase(
        &mut self,
        rest_client: &BybitRestClient,
        exchange_config: &ExchangeConfig,
        old_id: String,
        new_price: Decimal,
        mid_price: Decimal, // Задача 201: Signal price для анализа slippage
    ) -> Result<()> {
        let (side, qty, chase_count) = {
            if let Some(o) = self.order_manager.get_by_client_id(&old_id).await {
                (o.side, o.qty - o.executed_qty, o.chase_count)
            } else {
                return Ok(());
            }
        };

        let qty_decimal = Decimal::from_f64(qty).unwrap_or(Decimal::ZERO);
        if qty_decimal < self.market_info.min_order_qty {
            return Ok(());
        }

        // 1. Отменяем старый
        let _ = self.order_manager.cancel_order(rest_client, &mut self.risk_manager, &self.bot_config, exchange_config, &old_id, false).await;

        // 2. Выставляем новый
        let new_id = self.order_manager.place_limit_order(
            rest_client,
            &mut self.risk_manager,
            &self.bot_config,
            exchange_config,
            None,
            side,
            new_price,
            qty_decimal,
            self.bot_config.post_only,
            false, // reduce_only = false для открытия позиции
            mid_price, // Задача 201: Signal price для анализа slippage
            None, // best_bid - не используется для обычных входов
            None, // best_ask - не используется для обычных входов
            None, // position_qty - не используется для обычных входов
        ).await?;

        // 3. Переносим счетчик погони
        if let Some(mut new_order) = self.order_manager.get_order_mut(&new_id).await {
            new_order.chase_count = chase_count + 1;
            new_order.last_chase_ts = timestamp_ms() as i64;
        }
        
        let _ = self.save_current_state();
        Ok(())
    }

    /// Логирует сделку в CSV через асинхронный канал
    fn log_trade(&self, fill: crate::trading::types::FillEvent, realized_pnl: Option<Decimal>) {
        let record = TradeRecord {
            time: chrono::Utc::now().to_rfc3339(),
            symbol: fill.symbol,
            side: fill.side.to_string(),
            price: fill.exec_price,
            qty: fill.exec_qty,
            order_type: "Limit".to_string(), 
            is_maker: fill.is_maker, 
            signal_up: self.last_probabilities[1],   // Индекс 1: Up
            signal_down: self.last_probabilities[2], // Индекс 2: Down
            realized_pnl,
            fee: fill.exec_fee, 
        };

        if let Err(e) = self.trade_tx.try_send(record) {
            warn!("Failed to send trade record to logger: {}", e);
        }
    }

    /// Проверка очереди ордеров, ожидающих захвата цены через 100мс (Задача 202)
    /// Вызывается при каждом обновлении orderbook
    pub async fn check_and_capture_100ms_prices(&mut self, best_bid: Decimal, best_ask: Decimal) {
        let now = crate::utils::timestamp_ms();
        let price_check_delay_ms = 100u64;
        
        // Проверяем очередь и захватываем цены для ордеров, прошедших 100мс
        while let Some((_order_link_id, fill_time_ms)) = self.pending_price_checks.front() {
            let elapsed_ms = now.saturating_sub(*fill_time_ms);
            
            if elapsed_ms >= price_check_delay_ms {
                // Извлекаем ордер из очереди
                if let Some((order_link_id, _)) = self.pending_price_checks.pop_front() {
                    // Пытаемся найти ордер в истории (он должен быть там, так как он терминальный)
                    if let Some(order) = self.order_manager.get_history().await.iter().find(|o| o.link_id == order_link_id) {
                        let mid_price = (best_bid + best_ask) / Decimal::from(2);
                        let mid_price_f64 = mid_price.to_f64().unwrap_or(0.0);
                        
                        // Клонируем ордер и обновляем mid_price_100ms
                        let mut order_copy = order.clone();
                        order_copy.set_mid_price_100ms(mid_price_f64);
                        
                        // Отправляем обновленный лог в канал
                        let bot_path = std::path::Path::new("bots").join(&self.symbol);
                        if let Some(log) = self.order_manager.create_execution_quality_log(&order_copy, &bot_path) {
                            let _ = self.execution_quality_tx.try_send(log);
                        }
                        
                        debug!(
                            "[{}] Captured 100ms price for order {}: mid_price={:.8}",
                            self.symbol, order_link_id, mid_price_f64
                        );
                    } else {
                        warn!("[{}] Order {} not found in history for 100ms price capture", self.symbol, order_link_id);
                    }
                }
            } else {
                // Очередь отсортирована по времени, поэтому если первый ордер не готов, остальные тоже не готовы
                break;
            }
        }
    }

    /// Задача 208: Модуляция параметров переключения по urgency
    /// Применяет формулы:
    /// - effective_timeout = base_timeout / (1.0 + urgency)
    /// - effective_distance = base_distance / (1.0 + urgency)
    /// Чем выше urgency, тем быстрее переключение на агрессивный режим
    pub fn calculate_modulated_switch_params(
        &self,
        base_timeout_ms: u64,
        base_distance_bps: u32,
        urgency: f32,
    ) -> (u64, u32) {
        // Ограничиваем urgency в диапазон [0.0, 1.0]
        let urgency_clamped = urgency.max(0.0).min(1.0);
        
        // Применяем модуляцию
        let urgency_factor = 1.0 + urgency_clamped;
        
        let effective_timeout_ms = (base_timeout_ms as f32 / urgency_factor).max(50.0) as u64;
        let effective_distance_bps = (base_distance_bps as f32 / urgency_factor).max(1.0) as u32;
        
        info!(
            "[{}] Switch params modulated by urgency {}: timeout {} -> {}ms, distance {} -> {}bps",
            self.symbol, urgency_clamped, base_timeout_ms, effective_timeout_ms, 
            base_distance_bps, effective_distance_bps
        );
        
        (effective_timeout_ms, effective_distance_bps)
    }


    /// Задача 184: Обновление конфигурации при SIGHUP
    /// Применяет новые параметры бота к execution engine
    pub fn update_config(&mut self, bot_config: crate::config::types::BotConfig) {
        tracing::info!("[Audit] Updating ExecutionEngine config");
        self.bot_config = bot_config.clone();
        self.close_on_flat = bot_config.close_on_flat;
        self.thresh_buy = bot_config.threshold_buy;
        self.thresh_sell = bot_config.threshold_sell;
        
        // Обновляем risk manager с новыми параметрами риска
        self.risk_manager.update_config(bot_config.risk.clone());
    }
} 
