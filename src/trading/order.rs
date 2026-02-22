// Задача 136: State Machine для жизненного цикла ордера
//
// ## Обзор
//
// Этот модуль реализует State Machine для управления жизненным циклом ордеров.
// Основная цель - обеспечить прозрачный и надежный переход между состояниями,
// исключая "зависшие" состояния и ошибки в расчете текущей позиции.
//
// ## Архитектура
//
// ### Состояния (OrderState)
// - Created: Ордер создан локально, но еще не отправлен на биржу
// - PendingNew: Запрос на создание отправлен, ожидаем подтверждения
// - Active: Ордер подтвержден биржей и активен в стакане
// - PartiallyFilled: Ордер частично исполнен
// - Filled: Ордер полностью исполнен (терминальное состояние)
// - PendingCancel: Запрос на отмену отправлен, ожидаем подтверждения
// - Cancelled: Ордер отменен (терминальное состояние)
// - Rejected(reason): Ордер отклонен биржей (терминальное состояние)
// - Expired: Ордер истек (терминальное состояние)
//
// ### События (OrderEvent)
// - Accepted { order_id }: Биржа подтвердила создание ордера
// - Trade { exec_qty, price }: Произошло частичное или полное исполнение
// - CancelAck: Биржа подтвердила отмену ордера
// - Rejected { reason }: Биржа отклонила ордер
// - Expired: Ордер истек (Time-in-force)
//
// ## Использование f64 вместо Decimal
//
// Согласно решению в задаче 136: "Поскольку Bybit API и наш стакан (013) используют f64,
// использование Decimal создаст лишние касты и оверхед в 20%. Мы принимаем риск потери
// точности на 10-й значащей цифре ради скорости."

use crate::trading::types::{OrderState, OrderEvent, OrderSide};
use serde::{Deserialize, Serialize};

/// Структура ордера с поддержкой State Machine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub link_id: String,           // Наш уникальный ID (client_oid)
    pub order_id: Option<String>,  // ID от биржи
    pub state: OrderState,         // Текущее состояние
    pub price: f64,                // Цена ордера
    pub qty: f64,                  // Объем ордера
    pub executed_qty: f64,         // Исполненный объем
    pub side: OrderSide,           // Сторона (Buy/Sell)
    pub symbol: String,            // Торговая пара
    pub created_at: u64,           // Timestamp создания
    pub updated_at: u64,           // Timestamp последнего обновления
    pub post_only_reject_count: u32, // Задача 164: Счетчик Post-Only режектов
    pub is_post_only: bool,        // Задача 164: Флаг Post-Only режима
    // Задача 202: Метрики задержки (в микросекундах)
    pub signal_time_us: u64,       // Время создания сигнала
    pub sent_time_us: Option<u64>, // Время отправки запроса на биржу
    pub confirmed_time_us: Option<u64>, // Время получения подтверждения от биржи
    pub expected_price: f64,       // Цена в момент сигнала
    pub mid_price_100ms: Option<f64>, // Цена через 100мс после исполнения
    pub is_cancelled: bool,        // Флаг отмены ордера
    // Задача 204: Метрики влияния на цену
    pub mid_price_before: f64,     // Mid-price в момент создания ордера
    pub fill_counter: u32,         // Счетчик для отслеживания каждого Fill
    // Задача 207: Iceberg Order tracking
    pub iceberg_total_size: Option<f64>,    // Общий размер айсберга (если это Iceberg)
    pub iceberg_initial_price: Option<f64>, // Начальная цена первого плеча
    pub iceberg_filled_total: f64,          // Сколько исполнено от всего айсберга
    // Задача 208: Счетчик переключений Passive -> Aggressive
    pub switch_count: u8,          // Количество переключений на один сигнал
    // Задача 208: Urgency сигнала для модуляции параметров переключения
    pub urgency: f32,              // Уровень агрессивности (0.0 - 1.0)
    // Задача 108: Order Chasing tracking
    pub chase_count: usize,        // Счетчик попыток переставления ордера
    pub last_chase_ts: i64,        // Timestamp последней попытки погони (в миллисекундах)
    // Задача 167: Параметры для Exchange-side TSL
    pub tsl_trailing_stop: Option<String>,
    pub tsl_active_price: Option<String>,
}

impl Order {
    /// Создание нового ордера в состоянии Created
    pub fn new(
        link_id: String,
        symbol: String,
        side: OrderSide,
        price: f64,
        qty: f64,
        created_at: u64,
    ) -> Self {
        let signal_time_us = created_at * 1000; // Конвертируем из мс в микросекунды
        Self {
            link_id,
            order_id: None,
            state: OrderState::Created,
            price,
            qty,
            executed_qty: 0.0,
            side,
            symbol,
            created_at,
            updated_at: created_at,
            post_only_reject_count: 0,
            is_post_only: false,
            signal_time_us,
            sent_time_us: None,
            confirmed_time_us: None,
            expected_price: price,
            mid_price_100ms: None,
            is_cancelled: false,
            mid_price_before: price,  // Инициализируем ценой ордера по умолчанию
            fill_counter: 0,           // Счетчик Fill событий
            iceberg_total_size: None,  // Задача 207: Iceberg tracking
            iceberg_initial_price: None,
            iceberg_filled_total: 0.0,
            switch_count: 0,           // Задача 208: Инициализируем счетчик переключений
            urgency: 0.5,              // Задача 208: Default urgency
            chase_count: 0,                // Задача 108: Инициализируем счетчик погони
            last_chase_ts: 0,              // Задача 108: Инициализируем timestamp
            tsl_trailing_stop: None,       // Задача 167: TSL параметры
            tsl_active_price: None,
        }
    }

    /// Расчет остаточного объема
    pub fn remaining_qty(&self) -> f64 {
        self.qty - self.executed_qty
    }

    /// Проверка, является ли состояние терминальным
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.state,
            OrderState::Filled | OrderState::Cancelled | OrderState::Rejected(_) | OrderState::Expired
        )
    }

    /// Переход состояния на основе события
    /// 
    /// Возвращает Ok(()) при успешном переходе или Err с описанием недопустимого перехода.
    /// 
    /// Логика переходов:
    /// - PendingNew + Accepted -> Active (сохраняем order_id)
    /// - Active/PartiallyFilled + Trade -> PartiallyFilled или Filled (обновляем executed_qty)
    /// - PendingCancel + CancelAck -> Cancelled
    /// - Любое состояние + Rejected -> Rejected
    /// - Любое состояние + Expired -> Expired
    pub fn transition(&mut self, event: OrderEvent) -> Result<(), String> {
        let next_state = match (&self.state, event) {
            // Подтверждение создания ордера биржей
            (OrderState::PendingNew, OrderEvent::Accepted { order_id }) => {
                self.order_id = Some(order_id);
                OrderState::Active
            }

            // Частичное или полное исполнение
            (OrderState::Active | OrderState::PartiallyFilled, OrderEvent::Trade { exec_qty, price: _ }) => {
                self.executed_qty += exec_qty;
                
                // Критическое требование: при достижении полного объема переходим в Filled
                if self.executed_qty >= self.qty {
                    OrderState::Filled
                } else {
                    OrderState::PartiallyFilled
                }
            }

            // Подтверждение отмены
            (OrderState::PendingCancel, OrderEvent::CancelAck) => {
                OrderState::Cancelled
            }

            // Отклонение ордера (может произойти в любом состоянии)
            (_, OrderEvent::Rejected { reason }) => {
                OrderState::Rejected(reason)
            }

            // Истечение срока действия ордера
            (_, OrderEvent::Expired) => {
                OrderState::Expired
            }

            // Недопустимый переход
            (current_state, event) => {
                return Err(format!(
                    "Invalid transition from {:?} via {:?}",
                    current_state, event
                ));
            }
        };

        self.state = next_state;
        self.updated_at = crate::utils::timestamp_ms();
        Ok(())
    }

    /// Переход в состояние PendingNew (вызывается перед отправкой на биржу)
    pub fn mark_pending_new(&mut self) {
        self.state = OrderState::PendingNew;
        self.updated_at = crate::utils::timestamp_ms();
    }

    /// Переход в состояние PendingCancel (вызывается перед отправкой запроса на отмену)
    pub fn mark_pending_cancel(&mut self) {
        self.state = OrderState::PendingCancel;
        self.updated_at = crate::utils::timestamp_ms();
    }

    /// Применение сделки с учетом фильтров лота (Задача 137)
    /// 
    /// Обновляет executed_qty и определяет новое состояние ордера:
    /// - Если остаток после округления меньше min_qty (пыль) -> Filled
    /// - Иначе -> PartiallyFilled
    /// 
    /// Этот метод должен вызываться вместо прямого использования transition(Trade)
    /// для корректной обработки частичных исполнений с учетом правил биржи.
    pub fn apply_trade(&mut self, exec_qty: f64, filter: &crate::trading::types::LotFilter) {
        self.executed_qty += exec_qty;
        
        // Расчитываем реальный остаток, который можно выставить на биржу
        let remaining = self.qty - self.executed_qty;
        let rounded_remaining = crate::utils::helpers::round_down_to_step(remaining, filter.qty_step);

        if crate::utils::helpers::is_dust(rounded_remaining, filter.min_qty) {
            // Если остаток — "пыль", помечаем ордер как полностью исполненный
            self.state = OrderState::Filled;
            if rounded_remaining > 0.0 {
                tracing::debug!(
                    "Dust detected ({:.8}), marking order {} as Filled", 
                    rounded_remaining, 
                    self.link_id
                );
            }
        } else {
            self.state = OrderState::PartiallyFilled;
        }
        
        self.updated_at = crate::utils::timestamp_ms();
    }

    // ============================================================================
    // Методы для отслеживания метрик задержки (Задача 202)
    // ============================================================================

    /// Установить время отправки запроса на биржу (в микросекундах)
    pub fn set_sent_time(&mut self, sent_time_us: u64) {
        self.sent_time_us = Some(sent_time_us);
    }

    /// Установить время получения подтверждения от биржи (в микросекундах)
    pub fn set_confirmed_time(&mut self, confirmed_time_us: u64) {
        self.confirmed_time_us = Some(confirmed_time_us);
    }

    /// Установить цену через 100мс после исполнения
    pub fn set_mid_price_100ms(&mut self, price: f64) {
        self.mid_price_100ms = Some(price);
    }

    pub fn set_mid_price_before(&mut self, price: f64) {
        self.mid_price_before = price;
    }

    pub fn increment_fill_counter(&mut self) -> u32 {
        self.fill_counter += 1;
        self.fill_counter
    }

    /// Отметить ордер как отменённый
    pub fn mark_cancelled(&mut self) {
        self.is_cancelled = true;
    }

    /// Получить внутреннюю задержку в микросекундах (от сигнала до отправки)
    pub fn get_internal_latency_us(&self) -> Option<u64> {
        self.sent_time_us.map(|sent| sent.saturating_sub(self.signal_time_us))
    }

    /// Получить сетевую задержку в микросекундах (от отправки до подтверждения)
    pub fn get_network_latency_us(&self) -> Option<u64> {
        match (self.sent_time_us, self.confirmed_time_us) {
            (Some(sent), Some(confirmed)) => Some(confirmed.saturating_sub(sent)),
            _ => None,
        }
    }

    /// Получить Fill Rate (исполненный объём / заявленный объём)
    pub fn get_fill_rate(&self) -> f64 {
        if self.qty > 0.0 {
            self.executed_qty / self.qty
        } else {
            0.0
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_lifecycle_full_fill() {
        let mut order = Order::new(
            "test_001".to_string(),
            "BTCUSDT".to_string(),
            OrderSide::Buy,
            50000.0,
            1.0,
            1000,
        );

        // Created -> PendingNew
        order.mark_pending_new();
        assert_eq!(order.state, OrderState::PendingNew);

        // PendingNew -> Active
        let result = order.transition(OrderEvent::Accepted {
            order_id: "EX123".to_string(),
        });
        assert!(result.is_ok());
        assert_eq!(order.state, OrderState::Active);
        assert_eq!(order.order_id, Some("EX123".to_string()));

        // Active -> Filled (полное исполнение)
        let result = order.transition(OrderEvent::Trade {
            exec_qty: 1.0,
            price: 50000.0,
        });
        assert!(result.is_ok());
        assert_eq!(order.state, OrderState::Filled);
        assert_eq!(order.executed_qty, 1.0);
        assert_eq!(order.remaining_qty(), 0.0);
    }

    #[test]
    fn test_order_lifecycle_partial_fill() {
        let mut order = Order::new(
            "test_002".to_string(),
            "BTCUSDT".to_string(),
            OrderSide::Sell,
            50000.0,
            2.0,
            1000,
        );

        order.mark_pending_new();
        order.transition(OrderEvent::Accepted {
            order_id: "EX456".to_string(),
        }).unwrap();

        // Active -> PartiallyFilled
        order.transition(OrderEvent::Trade {
            exec_qty: 0.5,
            price: 50000.0,
        }).unwrap();
        assert_eq!(order.state, OrderState::PartiallyFilled);
        assert_eq!(order.executed_qty, 0.5);
        assert_eq!(order.remaining_qty(), 1.5);

        // PartiallyFilled -> PartiallyFilled
        order.transition(OrderEvent::Trade {
            exec_qty: 0.5,
            price: 50000.0,
        }).unwrap();
        assert_eq!(order.state, OrderState::PartiallyFilled);
        assert_eq!(order.executed_qty, 1.0);

        // PartiallyFilled -> Filled
        order.transition(OrderEvent::Trade {
            exec_qty: 1.0,
            price: 50000.0,
        }).unwrap();
        assert_eq!(order.state, OrderState::Filled);
        assert_eq!(order.executed_qty, 2.0);
    }

    #[test]
    fn test_order_cancellation() {
        let mut order = Order::new(
            "test_003".to_string(),
            "BTCUSDT".to_string(),
            OrderSide::Buy,
            50000.0,
            1.0,
            1000,
        );

        order.mark_pending_new();
        order.transition(OrderEvent::Accepted {
            order_id: "EX789".to_string(),
        }).unwrap();

        // Active -> PendingCancel
        order.mark_pending_cancel();
        assert_eq!(order.state, OrderState::PendingCancel);

        // PendingCancel -> Cancelled
        order.transition(OrderEvent::CancelAck).unwrap();
        assert_eq!(order.state, OrderState::Cancelled);
        assert!(order.is_terminal());
    }

    #[test]
    fn test_order_rejection() {
        let mut order = Order::new(
            "test_004".to_string(),
            "BTCUSDT".to_string(),
            OrderSide::Buy,
            50000.0,
            1.0,
            1000,
        );

        order.mark_pending_new();

        // PendingNew -> Rejected
        let result = order.transition(OrderEvent::Rejected {
            reason: "Insufficient balance".to_string(),
        });
        assert!(result.is_ok());
        assert!(matches!(order.state, OrderState::Rejected(_)));
        assert!(order.is_terminal());
    }

    #[test]
    fn test_invalid_transition() {
        let mut order = Order::new(
            "test_005".to_string(),
            "BTCUSDT".to_string(),
            OrderSide::Buy,
            50000.0,
            1.0,
            1000,
        );

        // Попытка Trade без Accepted
        let result = order.transition(OrderEvent::Trade {
            exec_qty: 1.0,
            price: 50000.0,
        });
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid transition"));
    }

    #[test]
    fn test_race_condition_handling() {
        let mut order = Order::new(
            "test_006".to_string(),
            "BTCUSDT".to_string(),
            OrderSide::Buy,
            50000.0,
            1.0,
            1000,
        );

        order.mark_pending_new();
        order.transition(OrderEvent::Accepted {
            order_id: "EX999".to_string(),
        }).unwrap();

        // Полное исполнение
        order.transition(OrderEvent::Trade {
            exec_qty: 1.0,
            price: 50000.0,
        }).unwrap();
        assert_eq!(order.state, OrderState::Filled);

        // Попытка Accepted после Filled (race condition из-за лага REST)
        let result = order.transition(OrderEvent::Accepted {
            order_id: "EX999_LATE".to_string(),
        });
        assert!(result.is_err());
        // Состояние не должно измениться
        assert_eq!(order.state, OrderState::Filled);
    }

    // ============================================================================
    // Тесты для apply_trade с фильтрами лота (Задача 137)
    // ============================================================================

    #[test]
    fn test_apply_trade_normal_partial_fill() {
        use crate::trading::types::LotFilter;
        
        let mut order = Order::new(
            "test_007".to_string(),
            "BTCUSDT".to_string(),
            OrderSide::Buy,
            50000.0,
            2.0,
            1000,
        );
        
        order.mark_pending_new();
        order.transition(OrderEvent::Accepted {
            order_id: "EX_PARTIAL".to_string(),
        }).unwrap();

        let filter = LotFilter {
            min_qty: 0.01,
            max_qty: 100.0,
            qty_step: 0.01,
        };

        // Первое частичное исполнение: 0.5 из 2.0
        order.apply_trade(0.5, &filter);
        assert_eq!(order.state, OrderState::PartiallyFilled);
        assert_eq!(order.executed_qty, 0.5);
        assert_eq!(order.remaining_qty(), 1.5);

        // Второе частичное исполнение: еще 1.0
        order.apply_trade(1.0, &filter);
        assert_eq!(order.state, OrderState::PartiallyFilled);
        assert_eq!(order.executed_qty, 1.5);
        assert_eq!(order.remaining_qty(), 0.5);
    }

    #[test]
    fn test_apply_trade_dust_detection() {
        use crate::trading::types::LotFilter;
        
        let mut order = Order::new(
            "test_008".to_string(),
            "BTCUSDT".to_string(),
            OrderSide::Sell,
            50000.0,
            1.0,
            1000,
        );
        
        order.mark_pending_new();
        order.transition(OrderEvent::Accepted {
            order_id: "EX_DUST".to_string(),
        }).unwrap();

        let filter = LotFilter {
            min_qty: 0.01,
            max_qty: 100.0,
            qty_step: 0.01,
        };

        // Исполнение 0.995 из 1.0, остаток 0.005 - это пыль (< 0.01)
        order.apply_trade(0.995, &filter);
        
        // Должен перейти в Filled, т.к. остаток меньше min_qty
        assert_eq!(order.state, OrderState::Filled);
        assert_eq!(order.executed_qty, 0.995);
    }

    #[test]
    fn test_apply_trade_full_fill() {
        use crate::trading::types::LotFilter;
        
        let mut order = Order::new(
            "test_009".to_string(),
            "ETHUSDT".to_string(),
            OrderSide::Buy,
            3000.0,
            1.0,
            1000,
        );
        
        order.mark_pending_new();
        order.transition(OrderEvent::Accepted {
            order_id: "EX_FULL".to_string(),
        }).unwrap();

        let filter = LotFilter {
            min_qty: 0.01,
            max_qty: 100.0,
            qty_step: 0.01,
        };

        // Полное исполнение
        order.apply_trade(1.0, &filter);
        assert_eq!(order.state, OrderState::Filled);
        assert_eq!(order.executed_qty, 1.0);
        assert_eq!(order.remaining_qty(), 0.0);
    }

    #[test]
    fn test_apply_trade_with_different_steps() {
        use crate::trading::types::LotFilter;
        
        // Тест с шагом 1.0 (для щиткоинов)
        let mut order = Order::new(
            "test_010".to_string(),
            "SHITCOINUSDT".to_string(),
            OrderSide::Buy,
            0.001,
            1000.0,
            1000,
        );
        
        order.mark_pending_new();
        order.transition(OrderEvent::Accepted {
            order_id: "EX_STEP1".to_string(),
        }).unwrap();

        let filter = LotFilter {
            min_qty: 1.0,
            max_qty: 100000.0,
            qty_step: 1.0,
        };

        // Исполнение 999.5 из 1000.0
        // Остаток 0.5 округляется вниз до 0.0 (шаг 1.0)
        // 0.0 < 1.0 (min_qty) -> пыль -> Filled
        order.apply_trade(999.5, &filter);
        assert_eq!(order.state, OrderState::Filled);
        assert_eq!(order.executed_qty, 999.5);
    }

    #[test]
    fn test_apply_trade_with_step_10() {
        use crate::trading::types::LotFilter;
        
        let mut order = Order::new(
            "test_011".to_string(),
            "DOGEUSDT".to_string(),
            OrderSide::Sell,
            0.1,
            1000.0,
            1000,
        );
        
        order.mark_pending_new();
        order.transition(OrderEvent::Accepted {
            order_id: "EX_STEP10".to_string(),
        }).unwrap();

        let filter = LotFilter {
            min_qty: 10.0,
            max_qty: 100000.0,
            qty_step: 10.0,
        };

        // Исполнение 985.0 из 1000.0
        // Остаток 15.0 округляется вниз до 10.0 (шаг 10.0)
        // 10.0 >= 10.0 (min_qty) -> не пыль -> PartiallyFilled
        order.apply_trade(985.0, &filter);
        assert_eq!(order.state, OrderState::PartiallyFilled);
        assert_eq!(order.executed_qty, 985.0);
        assert_eq!(order.remaining_qty(), 15.0);

        // Еще одно исполнение 8.0
        // Остаток 7.0 округляется вниз до 0.0 (шаг 10.0)
        // 0.0 < 10.0 (min_qty) -> пыль -> Filled
        order.apply_trade(8.0, &filter);
        assert_eq!(order.state, OrderState::Filled);
        assert_eq!(order.executed_qty, 993.0);
    }
}


// ============================================================================
// Конвертер между LegacyOrder (Decimal) и Order (f64)
// ============================================================================

use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;

impl Order {
    /// Конвертация из LegacyOrder (с Decimal) в Order (с f64)
    pub fn from_legacy(legacy: &crate::trading::types::LegacyOrder) -> Self {
        let created_at = legacy.created_at;
        let signal_time_us = created_at * 1000;
        let price = legacy.price.to_f64().unwrap_or(0.0);
        Self {
            link_id: legacy.client_oid.clone(),
            order_id: legacy.order_id.clone(),
            state: Self::status_to_state(&legacy.status),
            price,
            qty: legacy.qty.to_f64().unwrap_or(0.0),
            executed_qty: legacy.cum_exec_qty.to_f64().unwrap_or(0.0),
            side: legacy.side,
            symbol: legacy.symbol.clone(),
            created_at,
            updated_at: legacy.updated_at,
            post_only_reject_count: 0,
            is_post_only: false,
            signal_time_us,
            sent_time_us: None,
            confirmed_time_us: None,
            expected_price: price,
            mid_price_100ms: None,
            is_cancelled: false,
            mid_price_before: price,
            fill_counter: 0,
            // Задача 207: Iceberg-поля для восстановленных ордеров
            iceberg_total_size: None,
            iceberg_initial_price: None,
            iceberg_filled_total: 0.0,
            // Задача 208: Инициализируем счетчик переключений
            switch_count: 0,
            // Задача 208: Default urgency для восстановленных ордеров
            urgency: 0.5,
            // Задача 108: Order Chasing tracking для восстановленных ордеров
            chase_count: 0,
            last_chase_ts: 0,
            tsl_trailing_stop: None,
            tsl_active_price: None,
        }
    }

    /// Конвертация OrderStatus в OrderState
    fn status_to_state(status: &crate::trading::types::OrderStatus) -> OrderState {
        use crate::trading::types::OrderStatus;
        match status {
            OrderStatus::Created => OrderState::Created,
            OrderStatus::New => OrderState::Active,
            OrderStatus::PartiallyFilled => OrderState::PartiallyFilled,
            OrderStatus::Filled => OrderState::Filled,
            OrderStatus::Cancelled => OrderState::Cancelled,
            OrderStatus::Rejected => OrderState::Rejected("Unknown".to_string()),
            OrderStatus::Expired => OrderState::Expired,
            OrderStatus::PostOnlyRejected => OrderState::Rejected("PostOnly".to_string()),
            OrderStatus::Untracked => OrderState::Created, // Fallback
        }
    }

    /// Конвертация в LegacyOrder (с Decimal) для обратной совместимости
    pub fn to_legacy(&self) -> crate::trading::types::LegacyOrder {
        crate::trading::types::LegacyOrder {
            client_oid: self.link_id.clone(),
            order_id: self.order_id.clone(),
            symbol: self.symbol.clone(),
            side: self.side,
            price: Decimal::from_f64_retain(self.price).unwrap_or(Decimal::ZERO),
            qty: Decimal::from_f64_retain(self.qty).unwrap_or(Decimal::ZERO),
            status: Self::state_to_status(&self.state),
            cum_exec_qty: Decimal::from_f64_retain(self.executed_qty).unwrap_or(Decimal::ZERO),
            post_only_retry_count: 0,
            chase_count: 0,
            last_chase_ts: 0,
            created_at: self.created_at,
            updated_at: self.updated_at,
        }
    }

    /// Конвертация OrderState в OrderStatus
    fn state_to_status(state: &OrderState) -> crate::trading::types::OrderStatus {
        use crate::trading::types::OrderStatus;
        match state {
            OrderState::Created => OrderStatus::Created,
            OrderState::PendingNew => OrderStatus::Created, // Нет прямого аналога
            OrderState::Active => OrderStatus::New,
            OrderState::PartiallyFilled => OrderStatus::PartiallyFilled,
            OrderState::Filled => OrderStatus::Filled,
            OrderState::PendingCancel => OrderStatus::New, // Нет прямого аналога
            OrderState::Cancelled => OrderStatus::Cancelled,
            OrderState::Rejected(_) => OrderStatus::Rejected,
            OrderState::Expired => OrderStatus::Expired,
        }
    }
}
