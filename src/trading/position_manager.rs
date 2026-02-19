use crate::trading::types::{OrderSide, FillEvent, MarketInfo, CreateOrderRequest};
use crate::config::types::TimeDecayConfig;
use crate::utils::timestamp_ms;
use tracing::{info, warn, debug};
use rust_decimal::Decimal;
use rust_decimal::prelude::Zero;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    pub qty: Decimal,               // Положительная для Long, отрицательная для Short
    pub avg_price: Decimal,         // Средневзвешенная цена входа
    pub realized_pnl: Decimal,
    pub unrealized_pnl: Decimal,     // PnL по Mid Price
    pub unrealized_pnl_pct: Decimal, // Leveraged ROI (%)
    pub mark_pnl: Decimal,           // PnL по Mark Price
    pub leverage: Decimal,           // Текущее плечо
    pub updated_at: u64,
    pub opened_at: Option<u64>,      // Время открытия позиции (задача 163)
    // Задача 166: Поля для частичной фиксации прибыли
    pub completed_tp_stages: HashSet<usize>, // Индексы пройденных этапов TP
    pub initial_size: f64,           // Размер позиции в момент открытия (для расчета долей)
    pub side: OrderSide,             // Направление позиции (Buy для Long, Sell для Short)
    // Задача 167: Поля для динамического скользящего стоп-лосса
    pub extreme_water_mark: f64,     // Максимальная цена для Long, минимальная для Short
    pub current_stop_loss: f64,      // Актуальный уровень стопа
    pub tsl_active: bool,            // Флаг состояния активации трейлинга
    // Задача 170: Учет фандинга
    pub accumulated_funding: Decimal, // Накопленный фандинг (отрицательный = мы платим, положительный = нам платят)
}

impl Position {
    /// Проверяет, превысила ли позиция максимальное время жизни (задача 163)
    /// Метод максимально дешевый - без аллокаций
    pub fn is_aged(&self, now: u64, config: &TimeDecayConfig) -> bool {
        // Если opened_at не установлен, позиция не может быть "протухшей"
        let opened_at = match self.opened_at {
            Some(t) => t,
            None => return false,
        };

        // Определяем направление позиции
        let is_long = self.qty.is_sign_positive();
        
        // Выбираем соответствующий лимит
        let max_age = if is_long {
            config.max_age_long_ms
        } else {
            config.max_age_short_ms
        };

        // Проверяем возраст
        let age = now.saturating_sub(opened_at);
        age > max_age
    }
}

pub struct PositionManager {
    position: Position,
    min_qty_step: Decimal,
    // Задача 221: Real-time Equity Streamer
    equity_tx: Option<tokio::sync::broadcast::Sender<crate::monitoring::types::EquityUpdate>>,
    initial_balance: Decimal,
    last_equity_update_ms: i64,
    taker_fee_bps: Decimal,
    min_update_ms: u64,
}

impl PositionManager {
    pub fn new(symbol: String, leverage: Decimal, min_qty_step: Decimal) -> Self {
        Self {
            position: Position {
                symbol,
                qty: Decimal::zero(),
                avg_price: Decimal::zero(),
                realized_pnl: Decimal::zero(),
                unrealized_pnl: Decimal::zero(),
                unrealized_pnl_pct: Decimal::zero(),
                mark_pnl: Decimal::zero(),
                leverage,
                updated_at: timestamp_ms(),
                opened_at: None,
                completed_tp_stages: HashSet::new(),
                initial_size: 0.0,
                side: OrderSide::Buy, // Будет обновлено при открытии позиции
                extreme_water_mark: 0.0,
                current_stop_loss: 0.0,
                tsl_active: false,
                accumulated_funding: Decimal::zero(),
            },
            min_qty_step,
            equity_tx: None,
            initial_balance: Decimal::zero(),
            last_equity_update_ms: 0,
            taker_fee_bps: Decimal::zero(),
            min_update_ms: 100,
        }
    }

    /// Устанавливает broadcast канал для трансляции equity обновлений (Задача 221)
    pub fn set_equity_channel(
        &mut self, 
        tx: tokio::sync::broadcast::Sender<crate::monitoring::types::EquityUpdate>,
        initial_balance: Decimal,
        taker_fee_bps: Decimal,
        min_update_ms: u64,
    ) {
        self.equity_tx = Some(tx);
        self.initial_balance = initial_balance;
        self.taker_fee_bps = taker_fee_bps;
        self.min_update_ms = min_update_ms;
    }

    pub fn set_leverage(&mut self, leverage: Decimal) {
        self.position.leverage = leverage;
    }

    /// Обновляет нереализованный PnL и Leveraged ROI на основе текущих цен
    /// Задача 221: Отправляет обновления через broadcast канал с учетом throttling
    pub fn update_unrealized_pnl(&mut self, mid_price: Decimal, mark_price: Decimal) {
        if self.position.qty.is_zero() {
            self.position.unrealized_pnl = Decimal::zero();
            self.position.unrealized_pnl_pct = Decimal::zero();
            self.position.mark_pnl = Decimal::zero();
            
            // Отправляем обновление даже для пустой позиции
            self.send_equity_update(mid_price);
            return;
        }

        // 1. Расчет абсолютного PnL
        let raw_pnl = (mid_price - self.position.avg_price) * self.position.qty;
        
        // Задача 221: Учитываем Taker fee на закрытие позиции
        let position_value = mid_price * self.position.qty.abs();
        let close_fee = position_value * self.taker_fee_bps / Decimal::from(10000);
        self.position.unrealized_pnl = raw_pnl - close_fee;
        
        self.position.mark_pnl = (mark_price - self.position.avg_price) * self.position.qty;

        // 2. Расчет Leveraged ROI (%)
        // entry_value = avg_price * |qty|
        // entry_margin = entry_value / leverage
        let entry_value = self.position.avg_price * self.position.qty.abs();
        let entry_margin = if self.position.leverage.is_zero() {
            entry_value // Без плеча маржа равна стоимости входа
        } else {
            entry_value / self.position.leverage
        };

        self.position.unrealized_pnl_pct = if entry_margin.is_zero() {
            Decimal::zero()
        } else {
            (self.position.unrealized_pnl / entry_margin) * Decimal::from(100)
        };

        debug!(
            "[{}] PnL: {:+} USDT (ROI: {:+}%), Mark PnL: {:+}", 
            self.position.symbol, self.position.unrealized_pnl, self.position.unrealized_pnl_pct, self.position.mark_pnl
        );

        // Обновление метрик PnL (задача 189)
        // Конвертируем в базисные пункты (bps): 1 bps = 0.01%
        let unrealized_pnl_bps = self.position.unrealized_pnl_pct.to_f64().unwrap_or(0.0) * 100.0;
        metrics::gauge!("bot_unrealized_pnl_bps").set(unrealized_pnl_bps);
        
        // Realized PnL также в bps от начального баланса
        let realized_pnl_bps = if !self.initial_balance.is_zero() {
            (self.position.realized_pnl / self.initial_balance * Decimal::from(10000)).to_f64().unwrap_or(0.0)
        } else {
            0.0
        };
        metrics::gauge!("bot_realized_pnl_bps").set(realized_pnl_bps);
        
        // Задача 221: Отправляем обновление через broadcast канал
        self.send_equity_update(mid_price);
    }
    
    /// Отправляет обновление equity через broadcast канал с учетом throttling (Задача 221)
    fn send_equity_update(&mut self, mid_price: Decimal) {
        if let Some(ref tx) = self.equity_tx {
            let now = chrono::Utc::now().timestamp_millis();
            
            // Throttling: не отправляем обновления чаще min_update_ms
            // Для первого обновления last_equity_update_ms == 0, поэтому оно пройдет
            if now - self.last_equity_update_ms < self.min_update_ms as i64 {
                return;
            }
            
            let total_equity = self.initial_balance + self.position.realized_pnl + self.position.unrealized_pnl;
            
            let update = crate::monitoring::types::EquityUpdate::new(
                now,
                total_equity.to_f64().unwrap_or(0.0),
                self.position.unrealized_pnl.to_f64().unwrap_or(0.0),
                self.position.realized_pnl.to_f64().unwrap_or(0.0),
                self.position.qty.to_f64().unwrap_or(0.0),
            );
            
            // Отправляем через broadcast, игнорируем ошибки (нет подписчиков - не проблема)
            let _ = tx.send(update);
            self.last_equity_update_ms = now;
        }
    }
    
    /// Синхронизирует баланс с биржей для устранения дрейфа (Задача 221)
    pub fn sync_balance(&mut self, remote_balance: Decimal) {
        let drift = (self.initial_balance + self.position.realized_pnl - remote_balance).abs();
        
        if drift > Decimal::from_f64(0.01).unwrap_or_default() {
            warn!(
                "[{}] Balance drift detected! Local: {}, Remote: {}. Drift: {}. Syncing...",
                self.position.symbol,
                self.initial_balance + self.position.realized_pnl,
                remote_balance,
                drift
            );
            
            // Корректируем realized_pnl чтобы сохранить баланс
            self.position.realized_pnl = remote_balance - self.initial_balance;
            
            info!("[{}] Balance synced. New realized_pnl: {}", self.position.symbol, self.position.realized_pnl);
        }
    }

    /// Обновление позиции на основе исполненной сделки (Fill).
    /// Возвращает (Some(realized_pnl), position_closed, entry_price), где:
    /// - realized_pnl: PnL от частичного/полного закрытия
    /// - position_closed: true если позиция полностью закрыта (для обновления loss_streak)
    /// - entry_price: цена входа (avg_price) ДО обновления/сброса, полезна для расчета PnL в bps
    pub fn update_from_fill(&mut self, fill: FillEvent) -> (Option<Decimal>, bool, Option<Decimal>) {
        // Учет комиссий сразу уменьшает realized PnL
        self.position.realized_pnl -= fill.exec_fee;
        
        let qty_change = if fill.side == OrderSide::Buy { fill.exec_qty } else { -fill.exec_qty };
        let old_qty = self.position.qty;
        let new_qty = old_qty + qty_change;
        let mut fill_pnl = None;
        let entry_price = if old_qty.is_zero() { None } else { Some(self.position.avg_price) };

        info!(
            "Fill received: {} @ {}. Prev Pos: {}, New Pos: {}, Fee: {}", 
            fill.side, fill.exec_price, old_qty, new_qty, fill.exec_fee
        );

        // Задача 163: Установка opened_at при открытии позиции
        // Задача 166: Установка side и initial_size при открытии позиции
        // Задача 167: Инициализация TSL при открытии позиции
        if old_qty.is_zero() && !new_qty.is_zero() {
            self.position.opened_at = Some(timestamp_ms());
            self.position.side = fill.side;
            self.position.initial_size = new_qty.abs().to_f64().unwrap_or(0.0);
            self.position.completed_tp_stages.clear(); // Сброс этапов TP
            // Инициализируем extreme_water_mark и current_stop_loss по цене входа
            let entry_price = fill.exec_price.to_f64().unwrap_or(0.0);
            self.position.extreme_water_mark = entry_price;
            self.position.current_stop_loss = entry_price;
            self.position.tsl_active = false;
            debug!(
                "[{}] Position opened at {}, side: {:?}, initial_size: {}, TSL initialized",
                self.position.symbol, self.position.opened_at.unwrap(), 
                self.position.side, self.position.initial_size
            );
        }

        if old_qty.is_zero() {
            // 1. Открытие позиции с нуля
            self.position.avg_price = fill.exec_price;
        } else if old_qty.is_sign_positive() == qty_change.is_sign_positive() {
            // 2. Увеличение существующей позиции (пирамидинг)
            // new_avg = (qty * avg_price + qty_change * exec_price) / (qty + qty_change)
            self.position.avg_price = (old_qty * self.position.avg_price + qty_change * fill.exec_price) / new_qty;
            // Задача 166: Обновляем initial_size при усреднении для корректного расчета close_pct
            self.position.initial_size = new_qty.abs().to_f64().unwrap_or(0.0);
            debug!(
                "[{}] Position averaged, new avg_price: {}, updated initial_size: {}",
                self.position.symbol, self.position.avg_price, self.position.initial_size
            );
        } else if qty_change.abs() <= old_qty.abs() {
            // 3. Уменьшение или полное закрытие
            // avg_price не меняется. Расчет PnL: (exec - avg) * qty_change_abs * (1 if Long else -1)
            let side_sign = if old_qty.is_sign_positive() { Decimal::ONE } else { -Decimal::ONE };
            let pnl = (fill.exec_price - self.position.avg_price) * qty_change.abs() * side_sign;
            self.position.realized_pnl += pnl;
            fill_pnl = Some(pnl);
            info!("Realized PnL updated (Reduction): {}", pnl);

            if new_qty.is_zero() {
                self.position.avg_price = Decimal::zero();
            }
        } else {
            // 4. Переворот (Flip): qty_change.abs() > old_qty.abs()
            // Сначала закрываем старую позицию полностью
            let closed_qty_abs = old_qty.abs();
            let side_sign = if old_qty.is_sign_positive() { Decimal::ONE } else { -Decimal::ONE };
            let pnl = (fill.exec_price - self.position.avg_price) * closed_qty_abs * side_sign;
            self.position.realized_pnl += pnl;
            fill_pnl = Some(pnl);
            info!("Realized PnL updated (Flip Close): {}", pnl);

            // Открываем новую на остаток
            self.position.avg_price = fill.exec_price;
            // Задача 163: При флипе обновляем opened_at для новой позиции
            // Задача 166: При флипе обновляем side и initial_size для новой позиции
            // Задача 167: При флипе переинициализируем TSL для новой позиции
            self.position.opened_at = Some(timestamp_ms());
            self.position.side = fill.side;
            self.position.initial_size = new_qty.abs().to_f64().unwrap_or(0.0);
            self.position.completed_tp_stages.clear(); // Сброс этапов TP
            let entry_price = fill.exec_price.to_f64().unwrap_or(0.0);
            self.position.extreme_water_mark = entry_price;
            self.position.current_stop_loss = entry_price;
            self.position.tsl_active = false;
            debug!(
                "[{}] Position flipped, new opened_at: {}, side: {:?}, initial_size: {}, TSL reinitialized",
                self.position.symbol, self.position.opened_at.unwrap(),
                self.position.side, self.position.initial_size
            );
        }

        self.position.qty = new_qty;
        
        // Dust Cleanup: если остаток меньше min_qty_step, закрываем позицию полностью
        let mut position_closed = new_qty.is_zero();
        if self.position.qty.abs() < self.min_qty_step && !self.position.qty.is_zero() {
            info!(
                "[{}] Dust cleanup: qty {} < min_qty_step {}. Setting to Flat.", 
                self.position.symbol, self.position.qty.abs(), self.min_qty_step
            );
            self.position.qty = Decimal::zero();
            self.position.avg_price = Decimal::zero();
            position_closed = true;
        }
        
        // Задача 163: Сброс opened_at при закрытии позиции
        // Задача 166: Сброс completed_tp_stages при закрытии позиции
        // Задача 167: Сброс TSL при закрытии позиции
        if position_closed {
            self.position.opened_at = None;
            self.position.completed_tp_stages.clear();
            self.position.initial_size = 0.0;
            self.position.extreme_water_mark = 0.0;
            self.position.current_stop_loss = 0.0;
            self.position.tsl_active = false;
            debug!("[{}] Position closed, opened_at, TP stages and TSL reset", self.position.symbol);
        }
        
        self.position.updated_at = timestamp_ms();
        
        info!(
            "[{}] Position state: qty={}, avg_price={}, realized_pnl={}", 
            self.position.symbol, self.position.qty, self.position.avg_price, self.position.realized_pnl
        );

        (fill_pnl, position_closed, entry_price)
    }

    pub fn get_position(&self) -> &Position {
        &self.position
    }

    pub fn get_position_mut(&mut self) -> &mut Position {
        &mut self.position
    }

    /// Возвращает суммарный PnL (реализованный + нереализованный по mid_price)
    pub fn get_total_pnl(&self, mid_price: Decimal) -> Decimal {
        if self.position.qty.is_zero() {
            return self.position.realized_pnl;
        }
        
        let unrealized = (mid_price - self.position.avg_price) * self.position.qty;
        self.position.realized_pnl + unrealized
    }

    /// Прямая установка позиции (используется при синхронизации со стейтом)
    pub fn set_position(&mut self, qty: Decimal, avg_price: Decimal) {
        self.position.qty = qty;
        self.position.avg_price = avg_price;
        self.position.updated_at = timestamp_ms();
    }

    /// Принудительная синхронизация с биржей (Источник Истины)
    pub fn sync_from_remote(
        &mut self, 
        remote_qty: Decimal, 
        remote_avg_price: Decimal, 
        remote_leverage: Decimal, 
        remote_pnl: Decimal,
        market_info: &MarketInfo
    ) {
        let diff = (self.position.qty - remote_qty).abs();
        
        if diff >= market_info.qty_step {
            warn!(
                "[{}] Position mismatch detected! Local: {}, Remote: {}. Forcing sync.", 
                self.position.symbol, self.position.qty, remote_qty
            );
            self.position.qty = remote_qty;
            self.position.avg_price = remote_avg_price;
            self.position.updated_at = timestamp_ms();
        } else {
            debug!(
                "[{}] Position sync OK. Local: {}, Remote: {}", 
                self.position.symbol, self.position.qty, remote_qty
            );
        }

        // Синхронизация плеча
        if self.position.leverage != remote_leverage && !remote_leverage.is_zero() {
            info!("[{}] Leverage sync: {} -> {}", self.position.symbol, self.position.leverage, remote_leverage);
            self.position.leverage = remote_leverage;
        }

        // Сверка PnL (Биржа считает по Mark Price)
        let pnl_drift = (self.position.mark_pnl - remote_pnl).abs();
        if pnl_drift > Decimal::from_f64(0.1).unwrap_or_default() { // Порог 0.1 USDT
            warn!(
                "[{}] PnL Drift detected! Local Mark PnL: {}, Remote PnL: {}. Diff: {}", 
                self.position.symbol, self.position.mark_pnl, remote_pnl, pnl_drift
            );
        }

        // Логирование разницы между Mid и Mark PnL (влияние спреда)
        let spread_impact = (self.position.unrealized_pnl - self.position.mark_pnl).abs();
        if !self.position.qty.is_zero() {
            debug!(
                "[{}] Mid vs Mark PnL Variance: {} USDT. Total ROI: {:.2}%", 
                self.position.symbol, 
                spread_impact,
                self.position.unrealized_pnl_pct
            );
        }
    }

    /// Обновление серии убытков на основе PnL закрытой сделки (Задача 115, 118)
    /// Примечание: loss_streak и last_loss_timestamp_ms передаются как изменяемые ссылки, так как они хранятся в BotState
    pub fn update_streak(&self, trade_pnl: Decimal, loss_streak: &mut usize, last_loss_timestamp_ms: &mut i64) {
        if trade_pnl < Decimal::ZERO {
            *loss_streak += 1;
            *last_loss_timestamp_ms = chrono::Utc::now().timestamp_millis();
            warn!("[{}] Loss registered. Streak: {}", self.position.symbol, loss_streak);
        } else if trade_pnl > Decimal::ZERO {
            // Сброс серии И выход из блокировки при профите
            *loss_streak = 0;
            *last_loss_timestamp_ms = 0;
            info!("[{}] Profit registered. Loss streak and lockout reset.", self.position.symbol);
        }
        // Если trade_pnl == 0, серия не меняется
        // Сохранение стейта произойдет в execution.rs после вызова этого метода
    }

    /// Формирует параметры для экстренного рыночного закрытия позиции
    pub fn emergency_market_close(&self, category: String, order_link_id: String, position_idx: i32) -> Option<CreateOrderRequest> {
        if self.position.qty.is_zero() {
            return None;
        }

        let side = if self.position.qty.is_sign_positive() {
            OrderSide::Sell
        } else {
            OrderSide::Buy
        };

        Some(CreateOrderRequest {
            category,
            symbol: self.position.symbol.to_uppercase(),
            side: side.to_string(),
            order_type: "Market".to_string(),
            qty: self.position.qty.abs().to_string(),
            price: None,
            time_in_force: "GTC".to_string(),
            order_link_id,
            position_idx,
            reduce_only: Some(true),
        })
    }

    /// Помечает этап TP как выполненный (Задача 166)
    pub fn mark_tp_stage_completed(&mut self, stage_idx: usize) {
        self.position.completed_tp_stages.insert(stage_idx);
    }
    
    /// Обновляет накопленный фандинг при клиринге (Задача 170)
    /// Вычисляет фандинг как: funding_rate * qty * mark_price
    pub fn apply_funding(&mut self, funding_rate: f64, mark_price: Decimal) {
        if self.position.qty.is_zero() {
            // Позиция закрыта, фандинг не применяется
            return;
        }
        
        // Вычисляем фандинг: funding_rate * qty * mark_price
        let funding_amount = Decimal::from_f64_retain(funding_rate)
            .unwrap_or_default()
            * self.position.qty
            * mark_price;
        
        // Для Long (qty > 0): если funding_rate > 0, мы платим (отрицательный PnL)
        // Для Short (qty < 0): если funding_rate < 0, мы платим (отрицательный PnL)
        // Поэтому просто вычитаем из realized_pnl
        self.position.realized_pnl -= funding_amount;
        self.position.accumulated_funding += funding_amount;
        
        debug!("[{}] Applied funding: rate={:.6}, qty={}, mark_price={}, amount={}", 
            self.position.symbol, funding_rate, self.position.qty, mark_price, funding_amount);
    }
}
