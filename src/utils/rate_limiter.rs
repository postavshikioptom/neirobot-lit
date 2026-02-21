use governor::{Quota, RateLimiter as GovernorLimiter, state::NotKeyed};
use nonzero_ext::nonzero;
use std::num::NonZeroU32;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct RateLimiter {
    limiter: GovernorLimiter<NotKeyed, governor::state::InMemoryState, governor::clock::DefaultClock>,
}

impl RateLimiter {
    pub fn new(rate_per_sec: u32) -> Self {
        let quota = Quota::per_second(NonZeroU32::new(rate_per_sec).unwrap_or(nonzero!(10u32)));
        Self {
            limiter: GovernorLimiter::direct(quota),
        }
    }

    pub async fn wait(&self) {
        self.limiter.until_ready().await;
    }
}

/// Категории лимитов Bybit V5
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LimitCategory {
    /// Создание/отмена ордеров (самые строгие лимиты)
    Orders,
    /// Запросы рыночных данных (глубина стакана, история)
    Market,
    /// Запросы информации об аккаунте (балансы, плечи, позиции)
    Account,
}

/// Состояние лимита для конкретной категории
#[derive(Debug, Clone)]
pub struct LimitState {
    /// Оставшееся количество запросов
    pub remaining: u32,
    /// Общий лимит запросов
    pub limit: u32,
    /// Временная метка сброса лимита (Unix timestamp в миллисекундах)
    pub reset_ts: u64,
}

impl Default for LimitState {
    fn default() -> Self {
        Self {
            remaining: 1000,
            limit: 1000,
            reset_ts: 0,
        }
    }
}

/// Трекер лимитов запросов для разных категорий
#[derive(Debug, Clone)]
pub struct RateLimitTracker {
    states: Arc<RwLock<HashMap<LimitCategory, LimitState>>>,
}

impl RateLimitTracker {
    /// Создает новый трекер с инициализированными состояниями
    pub fn new() -> Self {
        let mut states = HashMap::new();
        states.insert(LimitCategory::Orders, LimitState::default());
        states.insert(LimitCategory::Market, LimitState::default());
        states.insert(LimitCategory::Account, LimitState::default());

        Self {
            states: Arc::new(RwLock::new(states)),
        }
    }

    /// Получает текущее состояние лимита для категории
    pub async fn get_state(&self, category: LimitCategory) -> LimitState {
        self.states
            .read()
            .await
            .get(&category)
            .cloned()
            .unwrap_or_default()
    }

    /// Обновляет состояние лимита на основе HTTP-заголовков Bybit
    pub async fn update_from_headers(
        &self,
        category: LimitCategory,
        remaining: u32,
        limit: u32,
        reset_ts: u64,
    ) {
        let mut states = self.states.write().await;
        states.insert(
            category,
            LimitState {
                remaining,
                limit,
                reset_ts,
            },
        );
    }

    /// Проверяет, находится ли лимит ниже порога (в процентах)
    pub async fn is_below_threshold(&self, category: LimitCategory, threshold_pct: f64) -> bool {
        let state = self.get_state(category).await;
        let threshold = (state.limit as f64 * threshold_pct) as u32;
        state.remaining < threshold
    }

    /// Получает процент оставшихся запросов
    pub async fn get_remaining_pct(&self, category: LimitCategory) -> f64 {
        let state = self.get_state(category).await;
        if state.limit == 0 {
            100.0
        } else {
            (state.remaining as f64 / state.limit as f64) * 100.0
        }
    }

    /// Проверяет, нужно ли ждать перед следующим запросом
    pub async fn should_throttle(&self, category: LimitCategory, threshold_pct: f64) -> bool {
        self.is_below_threshold(category, threshold_pct).await
    }
}

impl Default for RateLimitTracker {
    fn default() -> Self {
        Self::new()
    }
}
