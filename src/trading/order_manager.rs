use std::str::FromStr;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use anyhow::{Result, bail};
use tracing::{info, warn, error, debug};
use crate::trading::order::Order; // Новая Order с f64 и State Machine
use crate::trading::types::{
    OrderStatus, OrderSide, CreateOrderRequest, BybitOrderResult, 
    CancelOrderRequest, CancelAllOrdersRequest, OrderUpdate, FillEvent, AmendOrderRequest,
    OrderState, OrderEvent, // Задача 136: State Machine types
};
use crate::trading::rest_client::{BybitRestClientTrait, BybitOrderListResponse, RemoteOrder, BybitError};
use crate::trading::position_manager::PositionManager;
use crate::config::types::{BotConfig, ExchangeConfig, RiskConfig};
use crate::risk::risk_manager::RiskManager;
use crate::utils::timestamp_ms;
use crate::utils::logger::MarketImpactLog;
use crate::utils::rate_limiter::RateLimiter;
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use rust_decimal::prelude::FromPrimitive;
use tokio::sync::mpsc;

pub struct OrderManager {
    active_orders: HashMap<String, Order>, // Задача 136: Используем новую Order
    exchange_map: HashMap<String, String>, // order_id -> link_id
    history: Vec<Order>, // Задача 136: Используем новую Order
    nonce: AtomicU64, // Задача 176: Атомарный счетчик для уникальности order_link_id
    market_impact_tx: Option<mpsc::Sender<MarketImpactLog>>, // Задача 204: Sender для логирования влияния на цену
    is_price_shock: bool, // Задача 233: Флаг режима шока при ошибке Price Band
    rate_limiter: RateLimiter, // Задача 063: Rate limiter для REST API запросов
}

impl OrderManager {
    pub fn new() -> Self {
        Self {
            active_orders: HashMap::new(),
            exchange_map: HashMap::new(),
            history: Vec::new(),
            nonce: AtomicU64::new(0),
            market_impact_tx: None,
            is_price_shock: false,
            rate_limiter: RateLimiter::new(10), // Задача 063: 10 запросов в секунду
        }
    }

    /// Устанавливает sender для логирования влияния на цену (Задача 204)
    pub fn set_market_impact_logger(&mut self, tx: mpsc::Sender<MarketImpactLog>) {
        self.market_impact_tx = Some(tx);
    }

    /// Добавляет новый ордер. Возвращает ошибку, если link_id уже существует.
    pub fn add_order(&mut self, order: Order) -> Result<()> {
        if self.active_orders.contains_key(&order.link_id) {
            bail!("Duplicate link_id: {}", order.link_id);
        }
        self.active_orders.insert(order.link_id.clone(), order);
        Ok(())
    }

    /// Обновляет состояние ордера через State Machine transition.
    /// Если состояние терминальное, перемещает в историю.
    pub fn update_order(&mut self, link_id: &str, order_id: Option<String>, new_state: OrderState, exec_qty_delta: Option<f64>) {
        if let Some(order) = self.active_orders.get_mut(link_id) {
            let old_state = order.state.clone();
            
            // Обновляем order_id если пришел
            if let Some(id) = order_id.clone() {
                self.exchange_map.insert(id.clone(), link_id.to_string());
                order.order_id = Some(id);
            }

            // Создаем событие для transition
            let event = match (&new_state, exec_qty_delta) {
                (OrderState::Active, _) if matches!(old_state, OrderState::PendingNew) => {
                    OrderEvent::Accepted {
                        order_id: order.order_id.clone().unwrap_or_default(),
                    }
                }
                (OrderState::PartiallyFilled | OrderState::Filled, Some(delta)) => {
                    OrderEvent::Trade {
                        exec_qty: delta,
                        price: order.price,
                    }
                }
                (OrderState::Cancelled, _) if matches!(old_state, OrderState::PendingCancel) => {
                    OrderEvent::CancelAck
                }
                (OrderState::Rejected(reason), _) => {
                    OrderEvent::Rejected {
                        reason: reason.clone(),
                    }
                }
                (OrderState::Expired, _) => OrderEvent::Expired,
                _ => {
                    // Прямое изменение состояния без события (fallback)
                    order.state = new_state.clone();
                    order.updated_at = timestamp_ms();
                    
                    if order.is_terminal() {
                        self.move_to_history(link_id);
                    }
                    return;
                }
            };

            // Применяем transition
            match order.transition(event) {
                Ok(()) => {
                    info!("Order {} transitioned: {:?} -> {:?}", link_id, old_state, order.state);
                }
                Err(e) => {
                    warn!("Failed to transition order {}: {}. Forcing state change.", link_id, e);
                    order.state = new_state;
                    order.updated_at = timestamp_ms();
                }
            }

            // Перемещаем в историю если терминальное состояние
            if order.is_terminal() {
                self.move_to_history(link_id);
            }
        }
    }

    /// Перемещает ордер из активных в историю
    fn move_to_history(&mut self, link_id: &str) {
        if let Some(order) = self.active_orders.remove(link_id) {
            if let Some(ref id) = order.order_id {
                self.exchange_map.remove(id);
            }
            self.history.push(order);
        }
    }

    /// Основной метод обработки обновлений жизненного цикла ордера от биржи (WS или REST).
    /// Возвращает Ok(Some((fill_event, realized_pnl, position_closed, entry_price))), если произошло исполнение.
    pub fn update_order_state(
        &mut self,
        order_link_id: &str,
        event: OrderUpdate,
        position_manager: &mut PositionManager,
        lot_filter: &crate::trading::types::LotFilter, // Задача 137: Добавлен параметр для проверки пыли
        risk_manager: &mut RiskManager, // Задача 176: Для удаления интентов
    ) -> Result<Option<(FillEvent, Option<Decimal>, bool, Option<Decimal>)>> {
        let order = match self.active_orders.get_mut(order_link_id) {
            Some(o) => o,
            None => bail!("Order {} not found in active_orders", order_link_id),
        };

        let old_state = order.state.clone();
        let new_status = event.status;
        let old_executed_qty = order.executed_qty;
        let mut new_executed_qty = event.cum_exec_qty.to_f64().unwrap_or(0.0);
        let mut fill_info = None;

        // Критическое требование: При переходе в статус Filled убеждаемся, что исполнен весь объем
        if new_status == OrderStatus::Filled && new_executed_qty < order.qty {
            warn!(
                "Order {} status is Filled, but executed_qty ({}) < qty ({}). Adjusting to full qty.",
                order_link_id, new_executed_qty, order.qty
            );
            new_executed_qty = order.qty;
        }

        // 1. Обработка дельты исполнения (Partial Fill logic)
        if new_executed_qty > old_executed_qty {
            let delta_qty = new_executed_qty - old_executed_qty;
            let exec_price = event.exec_price.unwrap_or(Decimal::from_f64_retain(order.price).unwrap_or(Decimal::ZERO));

            let fill_event = FillEvent {
                symbol: order.symbol.clone(),
                side: order.side,
                exec_qty: Decimal::from_f64_retain(delta_qty).unwrap_or(Decimal::ZERO),
                exec_price,
                exec_fee: event.exec_fee.unwrap_or_default(),
                is_maker: event.is_maker.unwrap_or(true),
                exec_id: format!("exec_{}_{}", order_link_id, timestamp_ms()), 
                order_id: event.order_id.clone(),
                order_link_id: Some(order_link_id.to_string()),
                timestamp: event.timestamp,
            };

            info!(
                "Order {} fill delta: {} @ {} (Total: {}/{})",
                order_link_id, delta_qty, exec_price, new_executed_qty, order.qty
            );

            let (realized_pnl, position_closed, entry_price) = position_manager.update_from_fill(fill_event.clone());
            
            // Задача 137: Применяем apply_trade вместо transition для учета фильтров лота и проверки пыли
            order.apply_trade(delta_qty, lot_filter);
            
            fill_info = Some((fill_event, realized_pnl, position_closed, entry_price));
        }

        // 2. Обновление метаданных
        let new_state = match new_status {
            OrderStatus::New => OrderState::Active,
            OrderStatus::PartiallyFilled => OrderState::PartiallyFilled,
            OrderStatus::Filled => OrderState::Filled,
            OrderStatus::Cancelled => {
                // Распознавание причины отмены Post-Only
                if let Some(ref reason) = event.reason {
                    if reason.contains("CancelByPostOnly") || reason.contains("PostOnly") {
                        OrderState::Rejected("PostOnly".to_string())
                    } else {
                        OrderState::Cancelled
                    }
                } else {
                    OrderState::Cancelled
                }
            }
            OrderStatus::Rejected | OrderStatus::PostOnlyRejected => {
                OrderState::Rejected(event.reason.clone().unwrap_or_default())
            }
            OrderStatus::Expired => OrderState::Expired,
            _ => order.state.clone(),
        };

        // Применяем изменение состояния если оно изменилось
        if new_state != old_state {
            order.state = new_state.clone();
            order.updated_at = timestamp_ms();
            info!("Order {} state: {:?} -> {:?}", order_link_id, old_state, new_state);
        }
        
        // Синхронизация цены и объема после amendment (из WebSocket)
        if let Some(new_price) = event.new_price {
            let new_price_f64 = new_price.to_f64().unwrap_or(0.0);
            if (new_price_f64 - order.price).abs() > 1e-8 {
                info!("Order {} price updated via WebSocket: {} -> {}", order_link_id, order.price, new_price_f64);
                order.price = new_price_f64;
            }
        }
        if let Some(new_qty) = event.new_qty {
            let new_qty_f64 = new_qty.to_f64().unwrap_or(0.0);
            if (new_qty_f64 - order.qty).abs() > 1e-8 {
                info!("Order {} qty updated via WebSocket: {} -> {}", order_link_id, order.qty, new_qty_f64);
                order.qty = new_qty_f64;
            }
        }
        
        // Связываем order_id если он пришел впервые
        if order.order_id.is_none() {
            order.order_id = Some(event.order_id.clone());
            self.exchange_map.insert(event.order_id, order_link_id.to_string());
        }

        // 3. Обработка терминальных состояний
        if order.is_terminal() {
            if matches!(order.state, OrderState::Expired | OrderState::Rejected(_)) {
                warn!(
                    "Order {} {:?} by exchange. Reason: {:?}", 
                    order_link_id, order.state, event.reason
                );
            }

            // Задача 176: Удаляем интент при терминальном состоянии
            risk_manager.remove_order_intent(order_link_id);

            // Задача 202: Логирование метрик качества исполнения перед перемещением в историю
            if let Some(order_to_log) = self.active_orders.get(order_link_id) {
                // Отметить ордер как отменённый если он в состоянии Cancelled
                let mut order_copy = order_to_log.clone();
                if matches!(order_copy.state, OrderState::Cancelled) {
                    order_copy.mark_cancelled();
                }
                // Логирование происходит асинхронно через канал (см. ExecutionEngine)
                // Здесь мы просто подготавливаем данные
            }

            // Перенос в историю
            self.move_to_history(order_link_id);
        }

        Ok(fill_info)
    }

    /// Генерация уникального order_link_id с атомарным nonce (Задача 176)
    /// Формат: LIT_{symbol}_{unix_ms}_{nonce}
    /// Гарантирует уникальность даже при генерации нескольких ордеров в одну миллисекунду
    pub fn generate_order_link_id(&self, symbol: &str) -> String {
        let now = chrono::Utc::now().timestamp_millis();
        let nonce = self.nonce.fetch_add(1, Ordering::SeqCst);
        format!("LIT_{}_{}_{}", symbol.to_uppercase(), now, nonce)
    }

    /// Отправка лимитного ордера на биржу
    /// 
    /// Задача 136: Ордер создается в состоянии Created, затем переходит в PendingNew
    /// перед отправкой запроса на биржу.
    /// Задача 166: Добавлен параметр reduce_only для частичного закрытия позиций
    /// Задача 166: Добавлены параметры best_bid, best_ask для расчета Maker-цены внутри метода
    /// Задача 166: Добавлен параметр position_qty для проверки min_qty при TP close
    /// Задача 202: Добавлен параметр mid_price для отслеживания expected_price
    pub async fn place_limit_order(
        &mut self,
        rest_client: &impl BybitRestClientTrait,
        risk_manager: &mut RiskManager,
        bot_config: &BotConfig,
        exchange_config: &ExchangeConfig,
        ob: Option<&crate::data::orderbook::OrderBook>,
        side: OrderSide,
        price: Decimal,
        qty: Decimal,
        post_only: bool,
        reduce_only: bool,
        mid_price: Decimal,
        best_bid: Option<Decimal>,
        best_ask: Option<Decimal>,
        position_qty: Option<Decimal>,
    ) -> Result<String> {
        let order_link_id = self.generate_order_link_id(&bot_config.symbol);
        
        // Проверка на дубликат
        if self.active_orders.contains_key(&order_link_id) {
            bail!("Duplicate order_link_id: {}", order_link_id);
        }

        // Задача 166: Проверка min_qty для reduce_only ордеров (TP close)
        let min_qty = Decimal::from_str(&exchange_config.bybit.min_order_qty)
            .unwrap_or(Decimal::ZERO);
        let final_qty = if reduce_only && qty < min_qty {
            if bot_config.tp_close_all_on_min_qty {
                // Закрываем всю позицию
                if let Some(full_qty) = position_qty {
                    warn!(
                        "[{}] TP close qty {} < min_qty {}. Closing entire position with qty {}.",
                        bot_config.symbol, qty, min_qty, full_qty
                    );
                    full_qty
                } else {
                    warn!(
                        "[{}] TP close qty {} < min_qty {} but position_qty not provided. Using qty as is.",
                        bot_config.symbol, qty, min_qty
                    );
                    qty
                }
            } else {
                warn!(
                    "[{}] TP close qty {} < min_qty {} and tp_close_all_on_min_qty is false. Skipping order.",
                    bot_config.symbol, qty, min_qty
                );
                bail!("Order qty {} less than min_qty {} and tp_close_all_on_min_qty is false", qty, min_qty);
            }
        } else {
            qty
        };

        // Задача 166: Расчет Maker-цены для reduce_only ордеров (TP close)
        let final_price = if reduce_only && best_bid.is_some() && best_ask.is_some() {
            let best_bid = best_bid.unwrap();
            let best_ask = best_ask.unwrap();
            let tick_size = exchange_config.bybit.tick_size
                .parse::<Decimal>()
                .unwrap_or(Decimal::from_f64(0.01).unwrap());
            let maker_offset_ticks = bot_config.maker_offset_step_ticks;
            let offset = tick_size * Decimal::from(maker_offset_ticks);
            
            match side {
                OrderSide::Buy => best_bid - offset,
                OrderSide::Sell => best_ask + offset,
            }
        } else {
            price
        };

        // Создаем новый ордер в состоянии Created
        let mut order = Order::new(
            order_link_id.clone(),
            bot_config.symbol.clone(),
            side,
            final_price.to_f64().unwrap_or(0.0),
            final_qty.to_f64().unwrap_or(0.0),
            timestamp_ms(),
        );
        
        // Задача 202: Установка expected_price (цена в момент сигнала)
        order.expected_price = mid_price.to_f64().unwrap_or(0.0);
        
        // Задача 204: Установка mid_price_before (цена в момент создания ордера)
        order.set_mid_price_before(mid_price.to_f64().unwrap_or(0.0));
        
        // Задача 164: Устанавливаем флаг Post-Only
        order.is_post_only = post_only;
        
        // Задача 208: Установка urgency для модуляции параметров переключения
        order.urgency = bot_config.sor_config.default_urgency;

        // Переход в PendingNew перед отправкой
        order.mark_pending_new();
        info!("Order {} marked as PendingNew", order_link_id);

        // Задача 176: Регистрация намерения разместить ордер (для защиты от дубликатов)
        risk_manager.register_order_intent(
            order_link_id.clone(),
            side,
            final_price.to_f64().unwrap_or(0.0),
            final_qty.to_f64().unwrap_or(0.0),
        );

        let time_in_force = if post_only { "PostOnly" } else { "GTC" }.to_string();

        // Задача 167: Подготовка параметров TSL для Exchange-side режима
        let (trailing_stop, active_price) = if bot_config.trailing_stop.tsl_mode == crate::config::types::TSLMode::Exchange && !reduce_only {
            // Для Exchange-side режима при открытии позиции (не reduce_only)
            let distance_bps = bot_config.trailing_stop.tsl_distance_bps as f64;
            let activation_bps = bot_config.trailing_stop.tsl_activation_bps as f64;
            let order_price_f = final_price.to_f64().unwrap_or(0.0);
            
            // 1. Дистанция отступа в абсолютных единицах цены
            let absolute_distance = order_price_f * (distance_bps / 10000.0);
            
            // 2. Цена активации (Profit-only Activation) с учетом стороны
            let activation_price_f = if side == crate::trading::types::OrderSide::Buy {
                order_price_f * (1.0 + activation_bps / 10000.0)
            } else {
                order_price_f * (1.0 - activation_bps / 10000.0)
            };
            
            (Some(format!("{:.8}", absolute_distance)), Some(format!("{:.8}", activation_price_f)))
        } else {
            (None, None)
        };

        let request = CreateOrderRequest {
            category: exchange_config.bybit.category.clone(),
            symbol: bot_config.symbol.to_uppercase(),
            side: side.to_string(),
            order_type: "Limit".to_string(),
            qty: final_qty.to_string(),
            price: Some(final_price.to_string()),
            time_in_force,
            order_link_id: order_link_id.clone(),
            position_idx: bot_config.position_idx,
            reduce_only: if reduce_only { Some(true) } else { None },
            trailing_stop,
            active_price,
            // Задача 232: Self-Match Prevention (SMP)
            smp_type: if bot_config.smp_type != "None" {
                Some(bot_config.smp_type.clone())
            } else {
                None
            },
        };

        // Сохраняем параметры TSL в объект Order для последующей активации при Fill (Задача 167)
        order.tsl_trailing_stop = request.trailing_stop.clone();
        order.tsl_active_price = request.active_price.clone();

        // Задача 202: Установка sent_time_us перед отправкой запроса
        let sent_time_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        order.set_sent_time(sent_time_us);

        // Отправка через REST
        let result: Result<BybitOrderResult> = rest_client.post("/v5/order/create", &request).await;

        match result {
            Ok(res) => {
                risk_manager.report_success();
                
                // Задача 202: Установка confirmed_time_us при получении подтверждения
                let confirmed_time_us = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_micros() as u64;
                order.set_confirmed_time(confirmed_time_us);
                
                // Сохраняем order_id
                order.order_id = Some(res.order_id);
                // Добавляем в активные
                self.add_order(order)?;
                
                // Задача 189: Инкремент метрики успешно размещенных ордеров
                metrics::counter!("bot_orders_placed_total").increment(1);
                
                Ok(order_link_id)
            }
            Err(e) => {
                // Задача 176: Удаляем интент при ошибке отправки
                risk_manager.remove_order_intent(&order_link_id);
                
                // Задача 232: Проверка на ошибку SMP (110037)
                if let Some(bybit_err) = e.downcast_ref::<BybitError>() {
                    if bybit_err.code == 110037 {
                        warn!(
                            "[Order] SMP Triggered for {}. Cooling down...",
                            bot_config.symbol
                        );
                        // Небольшая задержка, чтобы стакан очистился от отмененных ордеров
                        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                        return Err(anyhow::anyhow!("SMP error {}: {}", bybit_err.code, bybit_err.msg));
                    }
                }
                
                // Задача 233: Проверка на ошибку Price Band Violation (110010)
                if let Some(bybit_err) = e.downcast_ref::<BybitError>() {
                    if bybit_err.code == 110010 {
                        error!(
                            target: "CRITICAL",
                            "[Order] Price Band Violation detected: {} - {}. Entering shock mode...",
                            bybit_err.code, bybit_err.msg
                        );
                        
                        // Вызываем обработчик ценового шока, если есть OrderBook
                        if let Some(order_book) = ob {
                            self.handle_price_band_violation(
                                order_book,
                                risk_manager,
                                &bot_config.risk,
                            ).await?;
                        } else {
                            // Если OrderBook не передан, просто устанавливаем флаг
                            risk_manager.set_price_shock(true);
                        }
                        
                        // Возвращаем ошибку после обработки
                        return Err(anyhow::anyhow!("Price band violation {}: {}", bybit_err.code, bybit_err.msg));
                    }
                }
                
                // Задача 231: Проверка на ошибки маржи (110004, 110007)
                if let Some(bybit_err) = e.downcast_ref::<BybitError>() {
                    // Проверяем коды ошибок маржи
                    if bybit_err.code == 110004 || bybit_err.code == 110007 {
                        error!(
                            target: "CRITICAL",
                            "[Order] Margin error detected: {} - {}. Initiating recovery...",
                            bybit_err.code, bybit_err.msg
                        );
                        
                        // Вызываем обработчик ошибки маржи
                        self.handle_margin_error(
                            rest_client,
                            risk_manager,
                            bot_config,
                            exchange_config,
                            bybit_err.code,
                        ).await?;
                        
                        // Возвращаем ошибку после обработки
                        return Err(anyhow::anyhow!("Margin error {}: {}", bybit_err.code, bybit_err.msg));
                    }
                    
                    // Обработка других ошибок Bybit
                    if !bot_config.risk.ignored_rejection_codes.contains(&(bybit_err.code as i32)) {
                        risk_manager.report_rejection();
                        error!("[Order] Rejected: {} - {}", bybit_err.code, bybit_err.msg);
                        
                        // Задача 189: Инкремент метрики отклоненных ордеров
                        metrics::counter!("bot_order_rejections_total").increment(1);
                    }
                } else {
                    risk_manager.report_rejection();
                    error!("[Order] API Error: {}", e);
                    
                    // Задача 189: Инкремент метрики отклоненных ордеров при API ошибке
                    metrics::counter!("bot_order_rejections_total").increment(1);
                }
                Err(e)
            }
        }
    }

    /// Отмена конкретного ордера по order_link_id
    /// 
    /// Задача 136: Перед отправкой запроса ордер переводится в PendingCancel.
    pub async fn cancel_order(
        &mut self,
        rest_client: &impl BybitRestClientTrait,
        risk_manager: &mut RiskManager,
        bot_config: &BotConfig,
        exchange_config: &ExchangeConfig,
        order_link_id: &str,
        force: bool,
    ) -> Result<()> {
        let exists_locally = self.active_orders.contains_key(order_link_id);
        
        if !exists_locally && !force {
            bail!("Order {} not found. Use force=true to cancel anyway.", order_link_id);
        }

        // Переход в PendingCancel перед отправкой
        if let Some(order) = self.active_orders.get_mut(order_link_id) {
            order.mark_pending_cancel();
            info!("Order {} marked as PendingCancel", order_link_id);
        }

        let request = CancelOrderRequest {
            category: exchange_config.bybit.category.clone(),
            symbol: bot_config.symbol.to_uppercase(),
            order_link_id: Some(order_link_id.to_string()),
            order_id: None,
        };

        // Rate limiting перед REST API запросом (Задача 063)
        self.rate_limiter.wait().await;
        
        // Отправка запроса
        let result: Result<serde_json::Value> = rest_client.post("/v5/order/cancel", &request).await;

        match result {
            Ok(_) => {
                risk_manager.report_success();
                info!("Order {} cancelled on exchange", order_link_id);
                self.update_order(order_link_id, None, OrderState::Cancelled, None);
                Ok(())
            }
            Err(e) => {
                let mut is_ignored = false;
                if let Some(bybit_err) = e.downcast_ref::<BybitError>() {
                    let code = bybit_err.code as i32;
                    if bot_config.risk.ignored_rejection_codes.contains(&code) || code == 110001 {
                        is_ignored = true;
                    }
                }

                if !is_ignored {
                    risk_manager.report_rejection();
                    error!("[Order] Cancel failed for {}: {}", order_link_id, e);
                } else {
                    debug!("[Order] Cancel error ignored for {}: {}", order_link_id, e);
                }

                let err_str = e.to_string();
                if err_str.contains("110001") || err_str.contains("Order not exists") {
                    warn!("Order {} not found on exchange. Syncing local state.", order_link_id);
                    if exists_locally {
                        self.update_order(order_link_id, None, OrderState::Cancelled, None);
                    }
                    Ok(())
                } else {
                    bail!("Failed to cancel order {}: {}", order_link_id, e);
                }
            }
        }
    }

    /// Изменение параметров активного ордера (amendment)
    pub async fn amend_active_order(
        &mut self,
        rest_client: &impl BybitRestClientTrait,
        risk_manager: &mut RiskManager,
        bot_config: &BotConfig,
        exchange_config: &ExchangeConfig,
        ob: Option<&crate::data::orderbook::OrderBook>,
        order_link_id: &str,
        new_price: Option<Decimal>,
        new_qty: Option<Decimal>,
        new_trigger_price: Option<Decimal>,
    ) -> Result<()> {
        use tracing::error;

        // Проверяем, существует ли ордер локально
        let order = match self.active_orders.get(order_link_id) {
            Some(o) => o,
            None => bail!("Order {} not found in active_orders", order_link_id),
        };

        // Проверяем, нужно ли отправлять запрос (экономия лимитов)
        // Конвертируем Decimal в f64 для сравнения
        let price_unchanged = new_price.map_or(true, |p| {
            let p_f64 = p.to_f64().unwrap_or(0.0);
            (p_f64 - order.price).abs() < 1e-8
        });
        let qty_unchanged = new_qty.map_or(true, |q| {
            let q_f64 = q.to_f64().unwrap_or(0.0);
            (q_f64 - order.qty).abs() < 1e-8
        });
        
        if price_unchanged && qty_unchanged && new_trigger_price.is_none() {
            info!("Order {} amendment skipped: no changes detected", order_link_id);
            return Ok(());
        }

        // Формируем запрос
        let request = AmendOrderRequest {
            category: exchange_config.bybit.category.clone(),
            symbol: bot_config.symbol.to_uppercase(),
            order_link_id: order_link_id.to_string(),
            price: new_price.map(|p| p.to_string()),
            qty: new_qty.map(|q| q.to_string()),
            trigger_price: new_trigger_price.map(|tp| tp.to_string()),
        };

        // Отправляем запрос через специализированный метод amend_order
        let result = rest_client.amend_order(&request).await;

        match result {
            Ok(_) => {
                risk_manager.report_success();
                info!("Order {} successfully amended on exchange", order_link_id);
                
                // Обновляем локальные данные при успехе
                if let Some(order) = self.active_orders.get_mut(order_link_id) {
                    let old_price = order.price;
                    let old_qty = order.qty;
                    let side = order.side;
                    
                    if let Some(p) = new_price {
                        order.price = p.to_f64().unwrap_or(0.0);
                    }
                    if let Some(q) = new_qty {
                        order.qty = q.to_f64().unwrap_or(0.0);
                    }
                    order.updated_at = timestamp_ms();
                    
                    // Задача 176: Обновляем интент в RiskManager с новыми параметрами
                    let final_price = new_price.unwrap_or(Decimal::from_f64_retain(old_price).unwrap_or(Decimal::ZERO)).to_f64().unwrap_or(0.0);
                    let final_qty = new_qty.unwrap_or(Decimal::from_f64_retain(old_qty).unwrap_or(Decimal::ZERO)).to_f64().unwrap_or(0.0);
                    
                    risk_manager.remove_order_intent(order_link_id);
                    risk_manager.register_order_intent(order_link_id.to_string(), side, final_price, final_qty);
                    
                    info!("Updated order intent for {} with new params: price={}, qty={}", order_link_id, final_price, final_qty);
                }
                
                Ok(())
            }
            Err(e) => {
                if let Some(bybit_err) = e.downcast_ref::<BybitError>() {
                    // Задача 233: Проверка на ошибку Price Band Violation (110010)
                    if bybit_err.code == 110010 {
                        error!(
                            target: "CRITICAL",
                            "[Order] Price Band Violation on Amend for {}. Entering shock mode...",
                            order_link_id
                        );
                        
                        // Вызываем обработчик ценового шока, если есть OrderBook
                        if let Some(order_book) = ob {
                            self.handle_price_band_violation(
                                order_book,
                                risk_manager,
                                &bot_config.risk,
                            ).await?;
                        } else {
                            // Если OrderBook не передан, просто устанавливаем флаг
                            risk_manager.set_price_shock(true);
                        }
                        
                        // Возвращаем ошибку после обработки
                        return Err(anyhow::anyhow!("Price band violation {}: {}", bybit_err.code, bybit_err.msg));
                    }
                    
                    let code = bybit_err.code as i32;
                    // 110004 (Not modified) и 110001 (Not exists) не считаются критическими отклонениями для лимита
                    if code != 110004 && code != 110001 && !bot_config.risk.ignored_rejection_codes.contains(&code) {
                        risk_manager.report_rejection();
                        error!("[Order] Amend rejected for {}: {} - {}", order_link_id, bybit_err.code, bybit_err.msg);
                    }
                } else {
                    risk_manager.report_rejection();
                    error!("[Order] API Error during amend for {}: {}", order_link_id, e);
                }

                let err_str = e.to_string();
                
                // 110004: Order not modified (параметры не изменились)
                if err_str.contains("110004") || err_str.contains("Order not modified") {
                    info!("Order {} not modified (110004): parameters unchanged", order_link_id);
                    return Ok(());
                }
                
                // 110001: Order not found (уже исполнен или отменен)
                if err_str.contains("110001") || err_str.contains("Order not exists") {
                    warn!("Order {} not found on exchange (110001): removing from local state", order_link_id);
                    self.update_order(order_link_id, None, OrderState::Cancelled, None);
                    return Ok(());
                }
                
                // 170139: Order qty out of range
                if err_str.contains("170139") || err_str.contains("qty out of range") {
                    error!("Order {} amendment failed (170139): qty out of range", order_link_id);
                    return Err(e);
                }
                
                // Другие ошибки
                bail!("Failed to amend order {}: {}", order_link_id, e);
            }
        }
    }

    /// Массовая отмена всех ордеров по текущему символу
    pub async fn cancel_all_orders(
        &mut self,
        rest_client: &impl BybitRestClientTrait,
        risk_manager: &mut RiskManager,
        bot_config: &BotConfig,
        exchange_config: &ExchangeConfig,
    ) -> Result<()> {
        let request = CancelAllOrdersRequest {
            category: exchange_config.bybit.category.clone(),
            symbol: bot_config.symbol.to_uppercase(),
        };

        info!("Sending cancel-all orders for {}", bot_config.symbol);
        
        // Rate limiting перед REST API запросом (Задача 063)
        self.rate_limiter.wait().await;
        
        // Отправка запроса на массовую отмену
        let result: Result<serde_json::Value> = rest_client.post("/v5/order/cancel-all", &request).await;

        match result {
            Ok(_) => {
                risk_manager.report_success();
                // Очистка локального реестра активных ордеров (Задача 063)
                let count = self.active_orders.len();
                self.active_orders.clear();
                info!("Successfully cancelled all {} active orders for {}", count, bot_config.symbol);
                Ok(())
            }
            Err(e) => {
                if let Some(bybit_err) = e.downcast_ref::<BybitError>() {
                    if !bot_config.risk.ignored_rejection_codes.contains(&(bybit_err.code as i32)) {
                        risk_manager.report_rejection();
                        error!("[Order] CancelAll rejected: {} - {}", bybit_err.code, bybit_err.msg);
                    }
                } else {
                    risk_manager.report_rejection();
                    error!("[Order] API Error during CancelAll: {}", e);
                }
                Err(e)
            }
        }
    }

    /// Сверка локального состояния ордеров с биржей (REST API)
    pub async fn reconcile_with_exchange(
        &mut self,
        rest_client: &impl BybitRestClientTrait,
        bot_config: &BotConfig,
        exchange_config: &ExchangeConfig,
        position_manager: &mut PositionManager,
        lot_filter: &crate::trading::types::LotFilter, // Задача 137: Добавлен параметр
        risk_manager: &mut RiskManager, // Задача 176: Для удаления интентов
    ) -> Result<()> {
        let category = &exchange_config.bybit.category;
        let symbol = &bot_config.symbol;

        // 1. Получаем список активных ордеров с биржи
        let params = format!("category={}&symbol={}", category, symbol);
        let realtime_resp: BybitOrderListResponse = rest_client.get_signed("/v5/order/realtime", &params).await?;
        
        let remote_active: HashMap<String, RemoteOrder> = realtime_resp.list.into_iter()
            .map(|o| (o.order_link_id.clone(), o))
            .collect();

        // 2. Перебираем локальные активные ордера для сверки
        let local_ids: Vec<String> = self.active_orders.keys().cloned().collect();

        for link_id in local_ids {
            if let Some(remote) = remote_active.get(&link_id) {
                // Ордер найден в активных на бирже
                let update = OrderUpdate {
                    order_link_id: link_id.clone(),
                    order_id: remote.order_id.clone(),
                    status: OrderStatus::from_bybit_status(&remote.order_status),
                    cum_exec_qty: remote.cum_exec_qty,
                    exec_price: None,
                    exec_fee: None,
                    is_maker: None,
                    reason: None,
                    timestamp: remote.updated_time.parse().unwrap_or_else(|_| timestamp_ms()),
                    new_price: Some(remote.price),
                    new_qty: Some(remote.qty),
                };
                let _ = self.update_order_state(&link_id, update, position_manager, lot_filter, risk_manager)?;
            } else {
                // Ордер не найден в активных, проверяем историю для уточнения причины
                info!("Order {} not found in realtime list, checking history...", link_id);
                let hist_params = format!("category={}&symbol={}&orderLinkId={}", category, symbol, link_id);
                let history_resp: BybitOrderListResponse = rest_client.get_signed("/v5/order/history", &hist_params).await?;
                
                if let Some(remote_hist) = history_resp.list.first() {
                    let update = OrderUpdate {
                        order_link_id: link_id.clone(),
                        order_id: remote_hist.order_id.clone(),
                        status: OrderStatus::from_bybit_status(&remote_hist.order_status),
                        cum_exec_qty: remote_hist.cum_exec_qty,
                        exec_price: None,
                        exec_fee: None,
                        is_maker: None,
                        reason: Some("Found in history during reconciliation".to_string()),
                        timestamp: remote_hist.updated_time.parse().unwrap_or_else(|_| timestamp_ms()),
                        new_price: Some(remote_hist.price),
                        new_qty: Some(remote_hist.qty),
                    };
                    let _ = self.update_order_state(&link_id, update, position_manager, lot_filter, risk_manager)?;
                } else {
                    // Ордера нет ни в активных, ни в истории (возможно, очень старый или ошибка ID)
                    warn!("Order {} not found in exchange history! Closing locally as Cancelled.", link_id);
                    self.update_order(&link_id, None, OrderState::Cancelled, None);
                }
            }
        }

        Ok(())
    }

    pub fn get_by_client_id(&self, client_oid: &str) -> Option<&Order> {
        self.active_orders.get(client_oid)
    }

    pub fn get_order_mut(&mut self, client_oid: &str) -> Option<&mut Order> {
        self.active_orders.get_mut(client_oid)
    }

    pub fn get_by_exchange_id(&self, order_id: &str) -> Option<&Order> {
        let client_oid = self.exchange_map.get(order_id)?;
        self.active_orders.get(client_oid)
    }

    pub fn get_active_orders(&self) -> &HashMap<String, Order> {
        &self.active_orders
    }

    /// Количество "ожидающих" (pending) ордеров в смысле задачи 074:
    /// Количество "ожидающих" (pending) ордеров в смысле задачи 074:
    /// считаем только ордера, реально висящие в стакане или частично исполненные.
    pub fn count_pending_orders(&self) -> usize {
        self.active_orders
            .values()
            .filter(|o| matches!(o.state, OrderState::Active | OrderState::PartiallyFilled | OrderState::PendingNew))
            .count()
    }

    /// Подсчет суммарного объема активных ордеров по конкретной стороне (Задача 114)
    pub fn get_pending_size(&self, side: OrderSide) -> Decimal {
        let sum_f64: f64 = self.active_orders.values()
            .filter(|o| o.side == side)
            .map(|o| o.qty)
            .sum();
        Decimal::from_f64_retain(sum_f64).unwrap_or(Decimal::ZERO)
    }

    /// Задача 232: Проверка наличия активных ордеров на покупку (Buy)
    pub fn has_active_buy_orders(&self) -> bool {
        self.active_orders.values().any(|o| o.side == OrderSide::Buy)
    }

    /// Задача 232: Проверка наличия активных ордеров на продажу (Sell)
    pub fn has_active_sell_orders(&self) -> bool {
        self.active_orders.values().any(|o| o.side == OrderSide::Sell)
    }

    pub fn get_active_count(&self) -> usize {
        self.active_orders.len()
    }

    pub fn get_history(&self) -> &[Order] {
        &self.history
    }

    /// Обработка события State Machine для ордера (Задача 136)
    /// 
    /// Применяет transition к ордеру на основе события.
    pub fn process_order_event(&mut self, order_link_id: &str, event: OrderEvent) -> Result<()> {
        let order = match self.active_orders.get_mut(order_link_id) {
            Some(o) => o,
            None => bail!("Order {} not found", order_link_id),
        };

        // Применяем transition
        match order.transition(event) {
            Ok(()) => {
                info!("Order {} transitioned to {:?}", order_link_id, order.state);
                
                // Перемещаем в историю если терминальное состояние
                if order.is_terminal() {
                    self.move_to_history(order_link_id);
                }
                
                Ok(())
            }
            Err(e) => {
                warn!("Invalid transition for order {}: {}", order_link_id, e);
                // Не возвращаем ошибку - это может быть race condition
                Ok(())
            }
        }
    }

    /// Логирует Fill событие с текущим mid_price (Задача 204)
    /// 
    /// Вызывается при получении Fill события для логирования влияния на цену
    pub fn log_fill_with_mid_price(
        &mut self,
        order_link_id: &str,
        fill_size: f64,
        mid_price_at_fill: f64,
        bot_path: &std::path::Path,
    ) -> Result<()> {
        let order = match self.active_orders.get_mut(order_link_id) {
            Some(o) => o,
            None => bail!("Order {} not found for logging", order_link_id),
        };

        // Инкрементируем fill_counter и получаем fill_id
        let fill_id = order.increment_fill_counter();

        // Если логирование включено, отправляем лог
        if let Some(ref tx) = self.market_impact_tx {
            let log = MarketImpactLog {
                timestamp_ms: timestamp_ms(),
                order_id: order.order_id.clone().unwrap_or_else(|| order.link_id.clone()),
                fill_id,
                side: format!("{:?}", order.side),
                fill_size,
                mid_before: order.mid_price_before,
                mid_at_fill: mid_price_at_fill,
                bot_path: bot_path.to_path_buf(),
            };

            // Отправляем в фоновый worker (не блокируем если канал переполнен)
            let _ = tx.try_send(log);
        }

        Ok(())
    }

    /// Выполняет экстренное рыночное закрытие
    pub async fn execute_emergency_close(
        &mut self,
        rest_client: &impl BybitRestClientTrait,
        risk_manager: &mut RiskManager,
        position_manager: &PositionManager,
        bot_config: &BotConfig,
        exchange_config: &ExchangeConfig,
    ) -> Result<()> {
        let order_link_id = self.generate_order_link_id(&bot_config.symbol);
        
        if let Some(request) = position_manager.emergency_market_close(
            exchange_config.bybit.category.clone(),
            order_link_id.clone(),
            bot_config.position_idx
        ) {
            info!("HARD STOP: Sending emergency market close order for {}", bot_config.symbol);
            let result: Result<BybitOrderResult> = rest_client.post("/v5/order/create", &request).await;

            match result {
                Ok(res) => {
                    risk_manager.report_success();
                    let side = OrderSide::from_str(&request.side)?;
                    let qty = Decimal::from_str(&request.qty)?;

                    // Используем новую Order структуру с f64
                    let mut order = Order::new(
                        order_link_id.clone(),
                        bot_config.symbol.clone(),
                        side,
                        0.0, // Market order - цена 0
                        qty.to_f64().unwrap_or(0.0),
                        timestamp_ms(),
                    );
                    
                    order.order_id = Some(res.order_id);
                    order.mark_pending_new();

                    self.add_order(order)?;
                }
                Err(e) => {
                    if let Some(bybit_err) = e.downcast_ref::<BybitError>() {
                        if !bot_config.risk.ignored_rejection_codes.contains(&(bybit_err.code as i32)) {
                            risk_manager.report_rejection();
                            error!("[Order] EmergencyClose rejected: {} - {}", bybit_err.code, bybit_err.msg);
                        }
                    } else {
                        risk_manager.report_rejection();
                        error!("[Order] API Error during EmergencyClose: {}", e);
                    }
                    return Err(e);
                }
            }
        }

        Ok(())
    }
    
    /// Задача 164: Обработка Post-Only Reject с экспоненциальным увеличением offset
    /// Возвращает новый offset в тиках или None, если нужно переключиться на Taker
    pub fn handle_post_only_reject(
        &mut self,
        order_link_id: &str,
        base_offset_ticks: u32,
        max_rejects: u32,
    ) -> Option<u32> {
        if let Some(order) = self.active_orders.get_mut(order_link_id) {
            order.post_only_reject_count += 1;
            
            if order.post_only_reject_count >= max_rejects {
                info!(
                    "[{}] Post-Only reject limit reached ({}/{}). Switching to Taker mode.",
                    order.symbol, order.post_only_reject_count, max_rejects
                );
                return None; // Переключаемся на Taker
            }
            
            // Экспоненциальное увеличение offset: offset * 1.5^reject_count
            let multiplier = 1.5_f64.powi(order.post_only_reject_count as i32);
            let new_offset = (base_offset_ticks as f64 * multiplier).ceil() as u32;
            
            info!(
                "[{}] Post-Only rejected ({}/{}). Increasing offset: {} -> {} ticks",
                order.symbol, order.post_only_reject_count, max_rejects, base_offset_ticks, new_offset
            );
            
            return Some(new_offset);
        }
        
        Some(base_offset_ticks)
    }
    
    /// Задача 164: Проверка необходимости ре-пеггинга активных Maker-ордеров
    /// Возвращает список (order_link_id, new_price) для ордеров, требующих обновления
    pub fn check_repeg_needed(
        &self,
        best_bid: Decimal,
        best_ask: Decimal,
        tick_size: Decimal,
        repeg_threshold_ticks: u32,
    ) -> Vec<(String, Decimal)> {
        let mut to_repeg = Vec::new();
        let threshold = tick_size * Decimal::from(repeg_threshold_ticks);
        
        for (link_id, order) in &self.active_orders {
            // Проверяем только Post-Only ордера в активном состоянии
            if !order.is_post_only {
                continue;
            }
            
            if !matches!(order.state, OrderState::Active | OrderState::PartiallyFilled) {
                continue;
            }
            
            let order_price = Decimal::from_f64(order.price).unwrap_or(Decimal::ZERO);
            let best_price = match order.side {
                OrderSide::Buy => best_bid,
                OrderSide::Sell => best_ask,
            };
            
            let distance = (order_price - best_price).abs();
            
            if distance > threshold {
                // Вычисляем новую цену (на 1 тик лучше best_price)
                let new_price = match order.side {
                    OrderSide::Buy => best_bid + tick_size,
                    OrderSide::Sell => best_ask - tick_size,
                };
                
                debug!(
                    "[{}] Order {} needs repeg: current={}, best={}, distance={} ticks",
                    order.symbol, link_id, order_price, best_price, 
                    (distance / tick_size).to_f64().unwrap_or(0.0)
                );
                
                to_repeg.push((link_id.clone(), new_price));
            }
        }
        
        to_repeg
    }
    
    /// Задача 164: Проверка таймаута Maker-ордеров
    /// Возвращает список order_link_id ордеров, превысивших timeout
    pub fn check_rebate_timeout(
        &self,
        timeout_ms: u64,
    ) -> Vec<String> {
        let now = timestamp_ms();
        let mut timed_out = Vec::new();
        
        for (link_id, order) in &self.active_orders {
            // Проверяем только Post-Only ордера
            if !order.is_post_only {
                continue;
            }
            
            if !matches!(order.state, OrderState::Active | OrderState::PartiallyFilled) {
                continue;
            }
            
            let age_ms = now.saturating_sub(order.created_at);
            
            if age_ms > timeout_ms {
                info!(
                    "[{}] Order {} timed out: age={}ms, timeout={}ms",
                    order.symbol, link_id, age_ms, timeout_ms
                );
                timed_out.push(link_id.clone());
            }
        }
        
        timed_out
    }
    
    /// Задача 190: Валидация восстановленных ордеров с биржей
    /// Сверяет восстановленные ордера с реальным состоянием на бирже через GET /v5/order/realtime
    /// Возвращает список link_id ордеров, которые не найдены на бирже или имеют расхождения
    pub async fn validate_restored_orders(
        &self,
        rest_client: &crate::trading::rest_client::BybitRestClient,
        category: &str,
        symbol: &str,
    ) -> Result<Vec<String>> {
        use tracing::{info, warn};
        
        if self.active_orders.is_empty() {
            info!("[Persistence] No restored orders to validate");
            return Ok(vec![]);
        }
        
        info!("[Persistence] Validating {} restored orders with exchange...", self.active_orders.len());
        
        // Получаем список активных ордеров с биржи
        let exchange_orders = match rest_client.get_open_orders(category, symbol).await {
            Ok(orders) => orders,
            Err(e) => {
                warn!("[Persistence] Failed to fetch open orders from exchange: {}", e);
                // Возвращаем все восстановленные ордера как невалидные
                return Ok(self.active_orders.keys().cloned().collect());
            }
        };
        
        // Создаем мапу биржевых ордеров по order_link_id
        let mut exchange_map: std::collections::HashMap<String, &crate::trading::types::OrderInfo> = 
            std::collections::HashMap::new();
        
        for order in &exchange_orders {
            exchange_map.insert(order.order_link_id.clone(), order);
        }
        
        let mut invalid_orders = Vec::new();
        
        // Проверяем каждый восстановленный ордер
        for (link_id, local_order) in &self.active_orders {
            match exchange_map.get(link_id) {
                Some(exchange_order) => {
                    // Ордер найден на бирже, проверяем соответствие
                    let price_match = (local_order.price - exchange_order.price.to_f64().unwrap_or(0.0)).abs() < 0.0001;
                    let qty_match = (local_order.qty - exchange_order.qty.to_f64().unwrap_or(0.0)).abs() < 0.0001;
                    
                    if !price_match || !qty_match {
                        warn!(
                            "[Persistence] Order {} has mismatched data: local(price={}, qty={}) vs exchange(price={}, qty={})",
                            link_id, local_order.price, local_order.qty, exchange_order.price, exchange_order.qty
                        );
                        invalid_orders.push(link_id.clone());
                    } else {
                        info!("[Persistence] Order {} validated successfully", link_id);
                    }
                }
                None => {
                    // Ордер не найден на бирже
                    warn!("[Persistence] Order {} not found on exchange", link_id);
                    invalid_orders.push(link_id.clone());
                }
            }
        }
        
        if invalid_orders.is_empty() {
            info!("[Persistence] All restored orders validated successfully");
        } else {
            warn!("[Persistence] {} orders failed validation", invalid_orders.len());
        }
        
        Ok(invalid_orders)
    }

    /// Задача 233: Обработка ошибки Price Band (110010) и стабилизация цен
    /// Переводит бота в режим ожидания до нормализации рыночных условий
    pub async fn handle_price_band_violation(
        &mut self,
        ob: &crate::data::orderbook::OrderBook,
        risk_manager: &mut RiskManager,
        config: &RiskConfig,
    ) -> Result<()> {
        self.is_price_shock = true;
        risk_manager.set_price_shock(true);
        error!("Price band violation detected. Suspending trading...");
        
        // Ожидание базового периода охлаждения
        tokio::time::sleep(tokio::time::Duration::from_secs(config.price_band_cooldown_sec)).await;
        
        // Цикл проверки стабилизации (Spread + Mark Deviation)
        loop {
            let spread_bps = ob.get_spread_bps();
            let mid_price = ob.get_mid_price();
            let mark_price = ob.get_mark_price();
            
            // Расчет отклонения мида от марки
            let mark_dev = if mark_price > 0.0 {
                (mid_price - mark_price).abs() / mark_price
            } else {
                0.0
            };
            
            // Проверка условий стабилизации
            if spread_bps < config.max_spread_bps_shock && mark_dev < config.max_mark_deviation {
                info!(
                    "Market stabilized. Spread: {:.2} bps, Mark Deviation: {:.4}%. Resuming trading...",
                    spread_bps, mark_dev * 100.0
                );
                self.is_price_shock = false;
                risk_manager.set_price_shock(false);
                break;
            }
            
            // Логирование текущего состояния
            debug!(
                "Waiting for stabilization. Spread: {:.2} bps (max: {:.2}), Mark Dev: {:.4}% (max: {:.4}%)",
                spread_bps, config.max_spread_bps_shock, mark_dev * 100.0, config.max_mark_deviation * 100.0
            );
            
            tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
        }
        
        Ok(())
    }

    /// Задача 233: Получить статус режима шока
    #[inline(always)]
    pub fn is_in_price_shock(&self) -> bool {
        self.is_price_shock
    }
}


// ============================================================================
// Методы для логирования метрик исполнения (Задача 202)
// ============================================================================

impl OrderManager {
    /// Создаёт структуру ExecutionQualityLog для передачи через канал
    /// Вызывается при достижении ордером терминального состояния
    pub fn create_execution_quality_log(
        &self,
        order: &Order,
        bot_path: &std::path::Path,
    ) -> Option<crate::utils::logger::ExecutionQualityLog> {
        // Получаем метрики задержки
        let internal_lat_us = order.get_internal_latency_us().unwrap_or(0);
        let network_lat_us = order.get_network_latency_us().unwrap_or(0);
        let fill_rate = order.get_fill_rate();
        
        // Получаем order_id или используем link_id если order_id не установлен
        let order_id = order.order_id.as_ref().unwrap_or(&order.link_id);
        
        Some(crate::utils::logger::ExecutionQualityLog {
            timestamp_ms: order.created_at,
            order_id: order_id.clone(),
            internal_lat_us,
            network_lat_us,
            fill_rate,
            is_cancelled: order.is_cancelled,
            bot_path: bot_path.to_path_buf(),
        })
    }

    /// Задача 208: Безопасное переключение Passive -> Aggressive
    /// Паттерн: cancel -> confirm -> send aggressive order
    /// Использует tokio::select! для параллельного ожидания исполнения и отмены
    pub async fn handle_switch_trigger(
        &mut self,
        rest_client: &impl BybitRestClientTrait,
        risk_manager: &mut RiskManager,
        bot_config: &BotConfig,
        exchange_config: &ExchangeConfig,
        order_link_id: &str,
        max_switches: u8,
    ) -> Result<bool> {
        // Получаем ордер
        let order = match self.active_orders.get_mut(order_link_id) {
            Some(o) => o,
            None => {
                warn!("Order {} not found for switch trigger", order_link_id);
                return Ok(false);
            }
        };

        // Проверяем лимит переключений
        if order.switch_count >= max_switches {
            info!("Order {} reached max switches limit ({})", order_link_id, max_switches);
            return Ok(false);
        }

        // Проверяем, есть ли еще что исполнять
        let remaining = order.remaining_qty();
        if remaining <= 0.0 {
            info!("Order {} has no remaining qty to switch", order_link_id);
            return Ok(false);
        }

        info!("Switching order {} to aggressive mode. Remaining: {}", order_link_id, remaining);

        // Шаг 1: Отправляем запрос на отмену
        let cancel_request = CancelOrderRequest {
            category: exchange_config.bybit.category.clone(),
            symbol: bot_config.symbol.to_uppercase(),
            order_link_id: Some(order_link_id.to_string()),
            order_id: None,
        };

        // Шаг 2: Дождаться ответа на отмену
        let cancel_result: Result<serde_json::Value> = rest_client.post("/v5/order/cancel", &cancel_request).await;

        match cancel_result {
            Ok(_) => {
                risk_manager.report_success();
                info!("Cancel request confirmed for order {}", order_link_id);
                
                // Шаг 3: Проверяем remaining_size после отмены
                if let Some(order) = self.active_orders.get_mut(order_link_id) {
                    let remaining_after_cancel = order.remaining_qty();
                    
                    if remaining_after_cancel > 0.0 {
                        // Увеличиваем счетчик переключений
                        order.switch_count += 1;
                        info!("Order {} switched to aggressive. Switch count: {}", order_link_id, order.switch_count);
                        
                        // Обновляем состояние ордера
                        self.update_order(order_link_id, None, OrderState::Cancelled, None);
                        return Ok(true);
                    } else {
                        info!("Order {} was fully filled before cancel confirmation", order_link_id);
                        return Ok(false);
                    }
                }
                Ok(false)
            }
            Err(e) => {
                let err_str = e.to_string();
                
                // Проверяем, был ли ордер уже исполнен или удален
                if err_str.contains("110001") || err_str.contains("Order not exists") {
                    warn!("Order {} not found on exchange (already filled or deleted)", order_link_id);
                    
                    // Проверяем локальное состояние
                    if let Some(order) = self.active_orders.get(order_link_id) {
                        let remaining = order.remaining_qty();
                        if remaining > 0.0 {
                            // Ордер был отменен на бирже, но у нас есть остаток
                            if let Some(o) = self.active_orders.get_mut(order_link_id) {
                                o.switch_count += 1;
                            }
                            self.update_order(order_link_id, None, OrderState::Cancelled, None);
                            return Ok(true);
                        }
                    }
                    return Ok(false);
                }
                
                // Другие ошибки
                risk_manager.report_rejection();
                error!("Failed to cancel order {} for switch: {}", order_link_id, e);
                bail!("Switch trigger failed for {}: {}", order_link_id, e)
            }
        }
    }

    /// Задача 231: Обработка ошибок при нехватке маржи (Insufficient Margin Recovery)
    /// 
    /// Выполняет восстановление после ошибок 110004 (Insufficient wallet balance) и 110007 (Insufficient margin):
    /// 1. Отменяет все активные ордера для высвобождения заблокированного баланса
    /// 2. Применяет штраф к размеру позиции через risk_manager
    /// 3. Логирует инцидент с тегом CRITICAL
    pub async fn handle_margin_error(
        &mut self,
        rest_client: &impl BybitRestClientTrait,
        risk_manager: &mut RiskManager,
        bot_config: &BotConfig,
        exchange_config: &ExchangeConfig,
        error_code: i64,
    ) -> Result<()> {
        use tracing::error;
        
        // Логирование инцидента с тегом CRITICAL
        error!(
            target: "CRITICAL",
            "Margin error detected (code: {}). Initiating recovery: cancelling all orders and applying position size penalty",
            error_code
        );
        
        // Action 1: Cancel All - отменяем все активные ордера для высвобождения баланса
        match self.cancel_all_orders(rest_client, risk_manager, bot_config, exchange_config).await {
            Ok(_) => {
                info!("Successfully cancelled all orders for {} during margin error recovery", bot_config.symbol);
            }
            Err(e) => {
                error!("Failed to cancel all orders during margin error recovery: {}", e);
                // Продолжаем выполнение, даже если отмена не удалась
            }
        }
        
        // Action 2: Backoff & Sizing - применяем штраф к размеру позиции
        risk_manager.apply_margin_penalty(&bot_config.risk);
        
        // Action 3: Отправка алерта (если настроен alert_manager)
        if let Some(ref alert_manager) = risk_manager.alert_manager {
            use crate::monitoring::alert_manager::{Alert, AlertLevel};
            
            let alert = Alert::new(
                AlertLevel::Critical,
                format!(
                    "🚨 CRITICAL: Margin error {} on {}. All orders cancelled, position size reduced to {:.1}%",
                    error_code,
                    bot_config.symbol,
                    bot_config.risk.margin_penalty_multiplier * 100.0
                ),
                "MarginErrorRecovery".to_string(),
            );
            alert_manager.send_alert(alert);
        }
        
        Ok(())
    }
}

    /// Задача 207: Обработка Iceberg Refill
    /// Проверяет, нужно ли выставить следующее "плечо" айсберга после исполнения текущего
    pub async fn check_iceberg_refill(
        &mut self,
        order: &Order,
        current_mid_price: f64,
        sor_config: &crate::config::types::SorConfig,
        rest_client: &impl BybitRestClientTrait,
        exchange_config: &ExchangeConfig,
        risk_manager: &mut RiskManager,
        bot_config: &BotConfig,
        ob: Option<&crate::data::orderbook::OrderBook>,
    ) -> Result<Option<String>> {
        // Проверяем, является ли это Iceberg-ордером
        let total_size = match order.iceberg_total_size {
            Some(size) => size,
            None => return Ok(None), // Не Iceberg-ордер
        };

        // Проверяем, что ордер полностью исполнен
        if !matches!(order.state, OrderState::Filled) {
            return Ok(None);
        }

        // Вычисляем остаток для исполнения
        let remaining = total_size - order.iceberg_filled_total;
        if remaining <= 0.0 {
            info!("Iceberg order {} fully completed: {}/{}", order.link_id, order.iceberg_filled_total, total_size);
            return Ok(None);
        }

        // Проверка отклонения цены от начальной
        if let Some(initial_price) = order.iceberg_initial_price {
            let price_deviation_bps = ((current_mid_price - initial_price).abs() / initial_price * 10000.0) as u32;
            if price_deviation_bps > sor_config.iceberg_price_dev_bps {
                warn!(
                    "Iceberg order {} aborted: price deviation {}bps > {}bps (initial: {}, current: {})",
                    order.link_id, price_deviation_bps, sor_config.iceberg_price_dev_bps,
                    initial_price, current_mid_price
                );
                return Ok(None);
            }
        }

        // Рассчитываем новый display_size с рандомизацией
        let base_display_ratio = order.qty / total_size; // Восстанавливаем базовый ratio
        let randomize_factor = 1.0 + (rand::random::<f32>() - 0.5) * 2.0 * sor_config.iceberg_randomize;
        let randomized_ratio = (base_display_ratio * randomize_factor).clamp(0.05, 1.0);
        let calculated_display = remaining * randomized_ratio as f64;
        
        // Используем min для предотвращения переполнения на последнем шаге
        let next_display_size = f64::min(calculated_display, remaining);

        info!(
            "Iceberg refill for {}: remaining={:.4}, display={:.4} (ratio={:.2}%)",
            order.link_id, remaining, next_display_size, randomized_ratio * 100.0
        );

        // Создаем новый ордер
        let price = Decimal::from_f64(order.price).unwrap_or(Decimal::ZERO);
        let qty = Decimal::from_f64(next_display_size).unwrap_or(Decimal::ZERO);
        let mid_price = Decimal::from_f64(current_mid_price).unwrap_or(Decimal::ZERO);
        
        let new_link_id = self.place_limit_order(
            rest_client,
            risk_manager,
            bot_config,
            exchange_config,
            ob,
            order.side,
            price,
            qty,
            order.is_post_only,
            false, // reduce_only
            mid_price,
            None, // best_bid
            None, // best_ask
        ).await?;
        
        // Обновляем Iceberg-метаданные в новом ордере
        if let Some(new_order) = self.active_orders.get_mut(&new_link_id) {
            new_order.iceberg_total_size = Some(total_size);
            new_order.iceberg_initial_price = order.iceberg_initial_price;
            new_order.iceberg_filled_total = order.iceberg_filled_total;
        }

        Ok(Some(new_link_id))
    }

    /// Задача 207: Создание первого "плеча" Iceberg-ордера
    pub async fn place_iceberg_order(
        &mut self,
        rest_client: &impl BybitRestClientTrait,
        risk_manager: &mut RiskManager,
        bot_config: &BotConfig,
        exchange_config: &ExchangeConfig,
        ob: Option<&crate::data::orderbook::OrderBook>,
        side: OrderSide,
        price: Decimal,
        total_size: f64,
        display_ratio: f64,
        post_only: bool,
        reduce_only: bool,
        mid_price: Decimal,
    ) -> Result<String> {
        // Рассчитываем размер первого отображаемого "плеча"
        let display_size = total_size * display_ratio;
        let display_qty = Decimal::from_f64(display_size).unwrap_or(Decimal::ZERO);
        
        // Создаем link_id для ордера
        let link_id = self.generate_order_link_id(&bot_config.symbol);
        
        // Создаем ордер
        let mut order = Order::new(
            link_id.clone(),
            bot_config.symbol.clone(),
            side,
            price.to_f64().unwrap_or(0.0),
            display_size,
            crate::utils::timestamp_ms(),
        );
        
        // Устанавливаем Iceberg-метаданные
        order.iceberg_total_size = Some(total_size);
        order.iceberg_initial_price = Some(price.to_f64().unwrap_or(0.0));
        order.iceberg_filled_total = 0.0;
        order.mid_price_before = mid_price.to_f64().unwrap_or(0.0);
        order.is_post_only = post_only;
        order.expected_price = mid_price.to_f64().unwrap_or(0.0);
        
        // Переход в PendingNew
        order.mark_pending_new();
        
        // Добавляем ордер в менеджер
        self.add_order(order)?;
        
        info!(
            "[{}] Creating Iceberg order: total={:.4}, display={:.4} ({:.1}%), price={}",
            bot_config.symbol, total_size, display_size, display_ratio * 100.0, price
        );
        
        // Регистрируем намерение
        risk_manager.register_order_intent(
            link_id.clone(),
            side,
            price.to_f64().unwrap_or(0.0),
            display_size,
        );
        
        // Отправляем на биржу используя стандартную логику place_limit_order
        // Но сначала нужно удалить ордер из active_orders, так как place_limit_order создаст его заново
        self.active_orders.remove(&link_id);
        
        // Вызываем place_limit_order для отправки на биржу
        self.place_limit_order(
            rest_client,
            risk_manager,
            bot_config,
            exchange_config,
            ob,
            side,
            price,
            display_qty,
            post_only,
            reduce_only,
            mid_price,
            None, // best_bid
            None, // best_ask
        ).await?;
        
        // Обновляем Iceberg-метаданные в созданном ордере
        if let Some(order) = self.active_orders.get_mut(&link_id) {
            order.iceberg_total_size = Some(total_size);
            order.iceberg_initial_price = Some(price.to_f64().unwrap_or(0.0));
            order.iceberg_filled_total = 0.0;
        }
        
        Ok(link_id)
    }

    /// Задача 235: Фоновая процедура очистки «зависших» ордеров (Stale Order Cleanup)
    /// Сверяет локальное состояние active_orders с реальными данными на бирже
    /// и автоматически отменяет неучтенные (untracked) или слишком старые (stale) ордера
    pub async fn run_cleanup_routine(
        &mut self,
        rest_client: &impl BybitRestClientTrait,
        risk_manager: &mut RiskManager,
        bot_config: &BotConfig,
        exchange_config: &ExchangeConfig,
    ) -> Result<()> {
        // 1. Получение всех открытых ордеров с биржи
        let params = format!("category={}&symbol={}", 
            exchange_config.bybit.category, 
            bot_config.symbol.to_uppercase()
        );
        
        let realtime_resp: BybitOrderListResponse = rest_client
            .get_signed("/v5/order/realtime", &params)
            .await?;
        
        let exchange_orders = realtime_resp.list;
        let mut orders_to_cancel = Vec::new();
        let now_ms = crate::utils::timestamp_ms();

        // 2. Логика фильтрации
        for remote_order in exchange_orders {
            // Проверяем, отслеживается ли ордер локально
            let is_untracked = !self.active_orders.contains_key(&remote_order.order_link_id) 
                && !self.exchange_map.contains_key(&remote_order.order_id);
            
            // Проверяем возраст ордера (если есть created_time)
            let mut is_too_old = false;
            if let Some(created_time_str) = &remote_order.created_time {
                if let Ok(created_ms) = created_time_str.parse::<u64>() {
                    let age_minutes = (now_ms - created_ms) / 60_000;
                    is_too_old = age_minutes > bot_config.max_stale_age_min;
                }
            }
            
            // Определяем, нужно ли отменять ордер
            if is_untracked {
                warn!(
                    "[Cleanup] Found untracked order on exchange: {} (link_id: {})",
                    remote_order.order_id, remote_order.order_link_id
                );
                orders_to_cancel.push(remote_order.order_link_id.clone());
            } else if bot_config.auto_cancel_stale && is_too_old {
                warn!(
                    "[Cleanup] Found stale order: {} (link_id: {}, age > {} min)",
                    remote_order.order_id, remote_order.order_link_id, bot_config.max_stale_age_min
                );
                orders_to_cancel.push(remote_order.order_link_id.clone());
            }
        }

        // 3. Массовая отмена найденных ордеров
        for link_id in orders_to_cancel {
            warn!("[Cleanup] Cancelling stale/untracked order: {}", link_id);
            // Используем force=true, так как ордер может не быть в active_orders
            let _ = self.cancel_order(
                rest_client,
                risk_manager,
                bot_config,
                exchange_config,
                &link_id,
                true, // force
            ).await;
        }
        
        Ok(())
    }
