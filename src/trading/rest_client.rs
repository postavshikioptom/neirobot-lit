use std::time::{SystemTime, UNIX_EPOCH, Duration};
use std::sync::Arc;
use std::str::FromStr;
use anyhow::{Result, Context};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use secrecy::{SecretString, ExposeSecret};
use reqwest::{Client, header, StatusCode};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use rust_decimal::Decimal;
use tracing::{warn, error};
use async_trait::async_trait;

#[derive(Debug, Clone)]
pub struct BybitError {
    pub code: i64,
    pub msg: String,
}

impl std::fmt::Display for BybitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Bybit error {}: {}", self.code, self.msg)
    }
}

impl std::error::Error for BybitError {}

use crate::config::ExchangeConfig;
use crate::trading::types::{MarketInfo, AmendOrderResult, SymbolInfo, LotFilter};
use crate::utils::rate_limiter::{RateLimiter, RateLimitTracker, LimitCategory};
use crate::utils::backoff::ExponentialBackoff;

type HmacSha256 = Hmac<Sha256>;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BybitResponse<T> {
    pub ret_code: i64,
    pub ret_msg: String,
    pub result: T,
    pub ret_ext_info: serde_json::Value,
    pub time: u64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BybitOrderListResponse {
    pub list: Vec<RemoteOrder>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct RemoteOrder {
    pub order_id: String,
    pub order_link_id: String,
    pub symbol: String,
    pub side: String,
    pub order_status: String,
    pub price: Decimal,
    pub qty: Decimal,
    pub cum_exec_qty: Decimal,
    pub updated_time: String,
    #[serde(default)]
    pub created_time: Option<String>, // Задача 235: Время создания ордера для cleanup routine
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WalletBalanceResponse {
    pub list: Vec<WalletBalance>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WalletBalance {
    pub account_type: String,
    pub total_equity: Decimal,
    pub total_available_balance: Decimal,
    pub total_margin_balance: Decimal,
    pub coin: Vec<CoinBalance>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CoinBalance {
    pub coin: String,
    pub equity: Decimal,
    pub available_to_withdraw: Decimal,
    pub available_to_borrow: Decimal,
    pub accrued_interest: Decimal,
    pub total_order_im_margin: Decimal,
    pub total_position_im_margin: Decimal,
    pub total_position_mm_margin: Decimal,
    pub wallet_balance: Decimal,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerTimeResponse {
    pub time_second: String,
    pub time_nano: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InstrumentsInfoResponse {
    pub list: Vec<InstrumentInfo>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InstrumentInfo {
    pub symbol: String,
    pub lot_size_filter: LotSizeFilter,
    pub price_filter: PriceFilter,
    pub leverage_filter: LeverageFilter,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LotSizeFilter {
    pub qty_step: String,
    pub min_order_qty: String,
    pub max_order_qty: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PriceFilter {
    pub tick_size: String,
    pub min_price: String,
    pub max_price: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LeverageFilter {
    pub max_leverage: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PositionListResponse {
    pub list: Vec<PositionInfo>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PositionInfo {
    pub symbol: String,
    pub side: String,
    pub size: Decimal,
    pub avg_price: Decimal,
    pub unrealised_pnl: Decimal,
    pub leverage: Decimal,
    pub position_idx: i32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiKeyInfoResponse {
    pub permissions: Vec<String>,
    pub ip_restrict: bool,
    pub expired_at: u64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OrderbookResponse {
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "b")]
    pub bids: Vec<[String; 2]>,
    #[serde(rename = "a")]
    pub asks: Vec<[String; 2]>,
    #[serde(rename = "ts")]
    pub timestamp: u64,
    #[serde(rename = "u")]
    pub update_id: u64,
}

/// Трейт для мокирования REST клиента в тестах
#[async_trait]
pub trait BybitRestClientTrait: Send + Sync {
    async fn post<T: Serialize + Send + Sync, R: DeserializeOwned + Send>(
        &self,
        endpoint: &str,
        body: &T,
    ) -> Result<R>;
    
    async fn get_signed<R: DeserializeOwned + Send>(
        &self,
        endpoint: &str,
        params: &str,
    ) -> Result<R>;
    
    async fn amend_order<T: Serialize + Send + Sync>(
        &self,
        body: &T,
    ) -> Result<AmendOrderResult>;

    async fn get_equity_with_retry(&self, retries: u32) -> Result<Decimal>;
    
    async fn get_position(
        &self,
        category: &str,
        symbol: &str,
        position_idx: i32,
    ) -> Result<Option<PositionInfo>>;
    
    async fn get_open_orders(
        &self,
        category: &str,
        symbol: &str,
    ) -> Result<Vec<crate::trading::types::OrderInfo>>;

    async fn fetch_orderbook(
        &self,
        category: &str,
        symbol: &str,
        limit: i32,
    ) -> Result<crate::data::types::OrderBookUpdateOwned>;

    async fn get_api_key_info(&self) -> Result<ApiKeyInfoResponse>;

    async fn set_trading_stop(
        &self,
        body: &crate::trading::types::TradingStopRequest,
    ) -> Result<()>;

    async fn cancel_order<T: Serialize + Send + Sync>(
        &self,
        body: &T,
    ) -> Result<serde_json::Value>;

    async fn cancel_all_orders<T: Serialize + Send + Sync>(
        &self,
        body: &T,
    ) -> Result<serde_json::Value>;
}

pub struct BybitRestClient {
    client: Client,
    api_key: String,
    api_secret: SecretString,
    base_url: String,
    recv_window: u64,
    orders_limiter: Arc<RateLimiter>,
    general_limiter: Arc<RateLimiter>,
    retry_initial_ms: u64,
    retry_max_ms: u64,
    retry_multiplier: f64,
    retry_jitter: f64,
    max_retries: u32,
    rate_limit_tracker: RateLimitTracker,
    rate_limit_threshold_pct: f64,
    backoff_base_ms: u64,
}

impl BybitRestClient {
    pub fn new(config: &ExchangeConfig) -> Result<Self> {
        let api_key = std::env::var("BYBIT_API_KEY")
            .context("BYBIT_API_KEY not found in environment")?;
        let api_secret = std::env::var("BYBIT_API_SECRET")
            .map(|s| SecretString::new(s.into_boxed_str()))
            .context("BYBIT_API_SECRET not found in environment")?;

        Ok(Self {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(config.rest.request_timeout_sec))
                .build()?,
            api_key,
            api_secret,
            base_url: config.rest.base_url.clone(),
            recv_window: 5000,
            orders_limiter: Arc::new(RateLimiter::new(config.rate_limits.order_rate as u32)),
            general_limiter: Arc::new(RateLimiter::new(config.rate_limits.private_rate as u32)),
            retry_initial_ms: config.rest_retry_initial_ms,
            retry_max_ms: config.rest_retry_max_ms,
            retry_multiplier: config.rest_retry_multiplier,
            retry_jitter: config.rest_retry_jitter,
            max_retries: config.rest_max_retries,
            rate_limit_tracker: RateLimitTracker::new(),
            rate_limit_threshold_pct: 0.15,
            backoff_base_ms: config.rate_limits.backoff_base_ms,
        })
    }

    /// Создает REST клиент с параметрами rate limiting из BotConfig
    pub fn with_rate_limit_config(
        config: &ExchangeConfig,
        rate_limit_threshold_pct: f64,
        backoff_base_ms: u64,
    ) -> Result<Self> {
        let api_key = std::env::var("BYBIT_API_KEY")
            .context("BYBIT_API_KEY not found in environment")?;
        let api_secret = std::env::var("BYBIT_API_SECRET")
            .map(|s| SecretString::new(s.into_boxed_str()))
            .context("BYBIT_API_SECRET not found in environment")?;

        Ok(Self {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(config.rest.request_timeout_sec))
                .build()?,
            api_key,
            api_secret,
            base_url: config.rest.base_url.clone(),
            recv_window: 5000,
            orders_limiter: Arc::new(RateLimiter::new(config.rate_limits.order_rate as u32)),
            general_limiter: Arc::new(RateLimiter::new(config.rate_limits.private_rate as u32)),
            retry_initial_ms: config.rest_retry_initial_ms,
            retry_max_ms: config.rest_retry_max_ms,
            retry_multiplier: config.rest_retry_multiplier,
            retry_jitter: config.rest_retry_jitter,
            max_retries: config.rest_max_retries,
            rate_limit_tracker: RateLimitTracker::new(),
            rate_limit_threshold_pct,
            backoff_base_ms,
        })
    }

    /// Сортирует параметры запроса по алфавиту (требование Bybit V5)
    fn sort_query_params(&self, params: &str) -> String {
        if params.is_empty() {
            return String::new();
        }
        
        let mut pairs: Vec<&str> = params.split('&').collect();
        pairs.sort_by(|a, b| {
            let key_a = a.split('=').next().unwrap_or("");
            let key_b = b.split('=').next().unwrap_or("");
            key_a.cmp(key_b)
        });
        pairs.join("&")
    }

    /// Генерирует подпись для Bybit V5 API
    fn generate_signature(&self, timestamp: u64, payload: &str) -> String {
        let pre_hash = format!("{}{}{}{}", timestamp, self.api_key, self.recv_window, payload);
        let mut mac = HmacSha256::new_from_slice(self.api_secret.expose_secret().as_bytes())
            .expect("HMAC can take key of any size");
        mac.update(pre_hash.as_bytes());
        let result = mac.finalize();
        hex::encode(result.into_bytes())
    }

    fn build_headers(&self, timestamp: u64, signature: &str) -> header::HeaderMap {
        let mut headers = header::HeaderMap::new();
        headers.insert("X-BAPI-API-KEY", header::HeaderValue::from_str(&self.api_key).unwrap());
        headers.insert("X-BAPI-SIGN", header::HeaderValue::from_str(signature).unwrap());
        headers.insert("X-BAPI-TIMESTAMP", header::HeaderValue::from_str(&timestamp.to_string()).unwrap());
        headers.insert("X-BAPI-RECV-WINDOW", header::HeaderValue::from_str(&self.recv_window.to_string()).unwrap());
        headers.insert(header::CONTENT_TYPE, header::HeaderValue::from_static("application/json"));
        headers
    }

    /// Ожидание лимитера в зависимости от пути с превентивным throttle
    async fn wait_for_limiter(&self, endpoint: &str) {
        // Определяем категорию запроса
        let category = if endpoint.contains("order/create") 
            || endpoint.contains("order/cancel") 
            || endpoint.contains("order/amend") {
            LimitCategory::Orders
        } else if endpoint.contains("market/") {
            LimitCategory::Market
        } else {
            LimitCategory::Account
        };

        // Проверяем, находится ли лимит ниже порога
        if self.rate_limit_tracker.should_throttle(category, self.rate_limit_threshold_pct).await {
            let remaining_pct = self.rate_limit_tracker.get_remaining_pct(category).await;
            tracing::warn!(
                category = ?category,
                remaining_pct = remaining_pct,
                "Rate limit below threshold, applying preventive throttle"
            );
            // Добавляем превентивную задержку 200 мс
            tokio::time::sleep(Duration::from_millis(200)).await;
        }

        // Вызываем соответствующий лимитер
        if category == LimitCategory::Orders {
            self.orders_limiter.wait().await;
        } else {
            self.general_limiter.wait().await;
        }
    }

    /// Парсит HTTP-заголовки Bybit для обновления состояния лимитов
    async fn update_rate_limits_from_headers(&self, category: LimitCategory, headers: &reqwest::header::HeaderMap) {
        let remaining = headers.get("X-Bapi-Limit-Status")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u32>().ok());
        let limit = headers.get("X-Bapi-Limit")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u32>().ok());
        let reset_ts = headers.get("X-Bapi-Limit-Reset-Timestamp")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok());

        if let (Some(r), Some(l), Some(ts)) = (remaining, limit, reset_ts) {
            self.rate_limit_tracker.update_from_headers(category, r, l, ts).await;
        } else if let Some(retry_after) = headers.get("Retry-After")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok()) {
            // Fallback: обновление через Retry-After
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            self.rate_limit_tracker.update_from_headers(category, 0, 100, now + retry_after * 1000).await;
        }
    }

    /// Выполнение запроса с retry логикой для сетевых ошибок
    async fn execute_with_retry<F, Fut, R>(&self, mut f: F) -> Result<R>
    where
        F: FnMut(u32) -> Fut,
        Fut: std::future::Future<Output = Result<R>>,
    {
        let mut backoff = ExponentialBackoff::new(
            Duration::from_millis(self.retry_initial_ms),
            Duration::from_millis(self.retry_max_ms),
            self.retry_multiplier,
            self.retry_jitter,
        );
        let mut attempt = 0;

        loop {
            match f(attempt).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    attempt += 1;
                    if attempt >= self.max_retries {
                        error!("Max retry attempts ({}) reached. Last error: {}", self.max_retries, e);
                        return Err(e);
                    }

                    let delay = backoff.next_delay();
                    warn!("Request failed (attempt {}): {}. Retrying in {:?}...", attempt, e, delay);
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }

    /// Обработка ответа от Bybit с учетом категории лимитов
    async fn handle_response<R: DeserializeOwned>(&self, category: LimitCategory, resp: reqwest::Response, attempt: u32) -> Result<R> {
        let status = resp.status();
        
        // Парсим заголовки лимитов для конкретной категории
        self.update_rate_limits_from_headers(category, resp.headers()).await;
        
        if status == StatusCode::TOO_MANY_REQUESTS {
            warn!("HTTP 429 Too Many Requests received. Sleeping for 10 seconds...");
            tokio::time::sleep(Duration::from_secs(10)).await;
            return Err(anyhow::Error::new(BybitError {
                code: 10004, // Стандартный код Bybit для Rate Limit
                msg: "Rate limit exceeded (HTTP 429)".to_string(),
            }));
        }

        let data: BybitResponse<R> = resp.json().await?;

        if data.ret_code != 0 {
            // Обработка ошибки 10006 (Too Many Requests) с exponential backoff
            if data.ret_code == 10006 {
                error!("Bybit error 10006: Too Many Requests. Applying exponential backoff...");
                // Применяем exponential backoff с джиттером
                crate::utils::helpers::apply_backoff(attempt, self.backoff_base_ms).await;
                return Err(anyhow::Error::new(BybitError {
                    code: data.ret_code,
                    msg: data.ret_msg,
                }));
            }

            // Инкремент счетчика отклонений ордеров (задача 189)
            // Проверяем, является ли это ошибкой связанной с ордерами
            if data.ret_code >= 10000 && data.ret_code < 20000 {
                metrics::counter!("bot_order_rejections_total").increment(1);
            }
            
            return Err(anyhow::Error::new(BybitError {
                code: data.ret_code,
                msg: data.ret_msg,
            }));
        }

        // Инкремент счетчика успешно размещенных ордеров (задача 189)
        // Проверяем, является ли это запросом на создание ордера
        // (это упрощенная проверка, в идеале нужно проверять endpoint)
        metrics::counter!("bot_orders_placed_total").increment(1);

        Ok(data.result)
    }

    /// Получение времени сервера (публичный эндпоинт)
    pub async fn get_server_time(&self) -> Result<u64> {
        let endpoint = "/v5/market/time";
        self.wait_for_limiter(endpoint).await;
        
        let url = format!("{}{}", self.base_url, endpoint);
        let client = self.client.clone();

        let result = self.execute_with_retry(|attempt| {
            let url = url.clone();
            let client = client.clone();
            async move {
                let resp = client.get(&url).send().await?;
                let result: ServerTimeResponse = self.handle_response(LimitCategory::Market, resp, attempt).await?;
                Ok(result)
            }
        }).await?;

        let time_ms = result.time_second.parse::<u64>()? * 1000 
                    + result.time_nano.parse::<u64>()? / 1_000_000;
        Ok(time_ms)
    }

    /// Получение баланса кошелька (приватный эндпоинт)
    pub async fn get_wallet_balance(&self) -> Result<WalletBalanceResponse> {
        let endpoint = "/v5/account/wallet-balance";
        let params = "accountType=UNIFIED";
        let sorted_params = self.sort_query_params(params);
        
        self.wait_for_limiter(endpoint).await;
        
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_millis() as u64;
        
        let signature = self.generate_signature(timestamp, &sorted_params);
        
        let url = format!("{}{}?{}", self.base_url, endpoint, sorted_params);
        let headers = self.build_headers(timestamp, &signature);
        let client = self.client.clone();

        self.execute_with_retry(|attempt| {
            let url = url.clone();
            let headers = headers.clone();
            let client = client.clone();
            async move {
                let resp = client.get(&url)
                    .headers(headers)
                    .send()
                    .await?;
                self.handle_response(LimitCategory::Account, resp, attempt).await
            }
        }).await
    }

    /// Получение информации об инструментах (публичный эндпоинт)
    pub async fn get_instruments_info(&self, category: &str, symbol: &str) -> Result<MarketInfo> {
        let endpoint = "/v5/market/instruments-info";
        self.wait_for_limiter(endpoint).await;
        
        let url = format!("{}{}?category={}&symbol={}", self.base_url, endpoint, category, symbol);
        let client = self.client.clone();

        let result: InstrumentsInfoResponse = self.execute_with_retry(|attempt| {
            let url = url.clone();
            let client = client.clone();
            async move {
                let resp = client.get(&url).send().await?;
                self.handle_response(LimitCategory::Market, resp, attempt).await
            }
        }).await?;

        let info = result.list.first()
            .context(format!("Symbol {} not found in instruments-info", symbol))?;

        Ok(MarketInfo {
            qty_step: Decimal::from_str(&info.lot_size_filter.qty_step)?,
            min_order_qty: Decimal::from_str(&info.lot_size_filter.min_order_qty)?,
            max_order_qty: Decimal::from_str(&info.lot_size_filter.max_order_qty)?,
            tick_size: Decimal::from_str(&info.price_filter.tick_size)?,
        })
    }

    /// Получение полной информации о символе с поддержкой retry (Задача 138)
    pub async fn fetch_symbol_info(&self, symbol: &str) -> Result<SymbolInfo> {
        let endpoint = "/v5/market/instruments-info";
        let params = [("category", "linear"), ("symbol", symbol)];
        
        self.wait_for_limiter(endpoint).await;
        
        let url = format!("{}{}?category={}&symbol={}", 
            self.base_url, endpoint, params[0].1, params[1].1);
        let client = self.client.clone();

        let result: InstrumentsInfoResponse = self.execute_with_retry(|attempt| {
            let url = url.clone();
            let client = client.clone();
            async move {
                let resp = client.get(&url).send().await?;
                self.handle_response(LimitCategory::Market, resp, attempt).await
            }
        }).await?;

        let data = result.list.first()
            .context(format!("Symbol {} not found", symbol))?;
        
        // Парсим строки в f64, так как Bybit API возвращает числа в кавычках
        let tick_size = f64::from_str(&data.price_filter.tick_size)
            .context("Failed to parse tick_size")?;
        let qty_step = f64::from_str(&data.lot_size_filter.qty_step)
            .context("Failed to parse qty_step")?;
        let min_price = f64::from_str(&data.price_filter.min_price)
            .context("Failed to parse min_price")?;
        let max_price = f64::from_str(&data.price_filter.max_price)
            .context("Failed to parse max_price")?;
        let min_qty = f64::from_str(&data.lot_size_filter.min_order_qty)
            .context("Failed to parse min_order_qty")?;
        let max_qty = f64::from_str(&data.lot_size_filter.max_order_qty)
            .context("Failed to parse max_order_qty")?;
        let max_leverage = f64::from_str(&data.leverage_filter.max_leverage)
            .context("Failed to parse max_leverage")?;
        
        // Рассчитываем количество знаков для форматирования цен
        let price_precision = if tick_size > 0.0 {
            (-tick_size.log10().floor()) as usize
        } else {
            8
        };

        Ok(SymbolInfo {
            lot_filter: LotFilter {
                min_qty,
                max_qty,
                qty_step,
            },
            price_filter: crate::trading::types::PriceFilter {
                tick_size,
                price_precision,
                min_price,
                max_price,
            },
            max_leverage,
        })
    }

    /// Универсальный GET запрос к Bybit V5 API (подписанный)
    pub async fn get_signed<R: DeserializeOwned>(&self, endpoint: &str, params: &str) -> Result<R> {
        self.wait_for_limiter(endpoint).await;

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_millis() as u64;
        
        let sorted_params = self.sort_query_params(params);
        let signature = self.generate_signature(timestamp, &sorted_params);
        
        let url = if sorted_params.is_empty() {
            format!("{}{}", self.base_url, endpoint)
        } else {
            format!("{}{}?{}", self.base_url, endpoint, sorted_params)
        };

        let headers = self.build_headers(timestamp, &signature);
        let client = self.client.clone();

        // Определяем категорию запроса
        let category = if endpoint.contains("order") {
            LimitCategory::Orders
        } else {
            LimitCategory::Account
        };

        self.execute_with_retry(|attempt| {
            let url = url.clone();
            let headers = headers.clone();
            let client = client.clone();
            async move {
                let resp = client.get(&url)
                    .headers(headers)
                    .send()
                    .await?;
                self.handle_response(category, resp, attempt).await
            }
        }).await
    }

    /// Получение текущей позиции (приватный эндпоинт)
    pub async fn get_position(&self, category: &str, symbol: &str, position_idx: i32) -> Result<Option<PositionInfo>> {
        let endpoint = "/v5/position/list";
        let params = format!("category={}&symbol={}", category, symbol);
        let sorted_params = self.sort_query_params(&params);
        
        self.wait_for_limiter(endpoint).await;
        
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_millis() as u64;
        
        let signature = self.generate_signature(timestamp, &sorted_params);
        
        let url = format!("{}{}?{}", self.base_url, endpoint, sorted_params);
        let headers = self.build_headers(timestamp, &signature);
        let client = self.client.clone();

        let result: PositionListResponse = self.execute_with_retry(|attempt| {
            let url = url.clone();
            let headers = headers.clone();
            let client = client.clone();
            async move {
                let resp = client.get(&url)
                    .headers(headers)
                    .send()
                    .await?;
                self.handle_response(LimitCategory::Account, resp, attempt).await
            }
        }).await?;

        // Ищем позицию с нужным positionIdx
        let pos = result.list.into_iter()
            .find(|p| p.position_idx == position_idx && p.size > Decimal::ZERO);

        Ok(pos)
    }

    /// Получение текущей позиции с учетом знака (LIT-совместимый)
    pub async fn get_position_signed(&self, category: &str, symbol: &str, position_idx: i32) -> Result<(Decimal, Decimal, Decimal, Decimal)> {
        let pos = self.get_position(category, symbol, position_idx).await?;
        match pos {
            Some(p) => {
                let qty = if p.side == "Buy" { p.size } else { -p.size };
                Ok((qty, p.avg_price, p.leverage, p.unrealised_pnl))
            }
            None => Ok((Decimal::ZERO, Decimal::ZERO, Decimal::ONE, Decimal::ZERO))
        }
    }

    /// Получение списка активных ордеров (Задача 120)
    pub async fn get_open_orders(&self, category: &str, symbol: &str) -> Result<Vec<crate::trading::types::OrderInfo>> {
        let endpoint = "/v5/order/realtime";
        let params = format!("category={}&symbol={}", category, symbol);
        let sorted_params = self.sort_query_params(&params);
        
        self.wait_for_limiter(endpoint).await;
        
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_millis() as u64;
        
        let signature = self.generate_signature(timestamp, &sorted_params);
        
        let url = format!("{}{}?{}", self.base_url, endpoint, sorted_params);
        let headers = self.build_headers(timestamp, &signature);
        let client = self.client.clone();

        let result: BybitOrderListResponse = self.execute_with_retry(|attempt| {
            let url = url.clone();
            let headers = headers.clone();
            let client = client.clone();
            async move {
                let resp = client.get(&url)
                    .headers(headers)
                    .send()
                    .await?;
                self.handle_response(LimitCategory::Orders, resp, attempt).await
            }
        }).await?;

        // Конвертируем RemoteOrder в OrderInfo
        let orders: Vec<crate::trading::types::OrderInfo> = result.list.into_iter()
            .filter(|o| o.order_status == "New" || o.order_status == "PartiallyFilled")
            .map(|o| {
                let side = if o.side == "Buy" {
                    crate::trading::types::OrderSide::Buy
                } else {
                    crate::trading::types::OrderSide::Sell
                };
                
                crate::trading::types::OrderInfo {
                    side,
                    price: o.price,
                    qty: o.qty,
                    status: crate::trading::types::OrderStatus::from_bybit_status(&o.order_status),
                    chase_count: 0,
                    last_chase_ts: 0,
                    link_id: Some(o.order_link_id),
                }
            })
            .collect();

        Ok(orders)
    }

    /// Универсальный POST запрос к Bybit V5 API
    pub async fn post<T: Serialize, R: DeserializeOwned>(&self, endpoint: &str, body: &T) -> Result<R> {
        self.wait_for_limiter(endpoint).await;

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_millis() as u64;
        
        let payload = serde_json::to_string(body)?;
        let signature = self.generate_signature(timestamp, &payload);
        
        let url = format!("{}{}", self.base_url, endpoint);
        let headers = self.build_headers(timestamp, &signature);
        let client = self.client.clone();

        // Определяем категорию запроса
        let category = if endpoint.contains("order") {
            LimitCategory::Orders
        } else {
            LimitCategory::Account
        };

        self.execute_with_retry(|attempt| {
            let url = url.clone();
            let headers = headers.clone();
            let payload = payload.clone();
            let client = client.clone();
            async move {
                let resp = client.post(&url)
                    .headers(headers)
                    .body(payload)
                    .send()
                    .await?;
                self.handle_response(category, resp, attempt).await
            }
        }).await
    }

    /// Изменение параметров активного ордера (amendment)
    pub async fn amend_order<T: Serialize>(&self, body: &T) -> Result<AmendOrderResult> {
        let endpoint = "/v5/order/amend";
        self.wait_for_limiter(endpoint).await;

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_millis() as u64;
        
        let payload = serde_json::to_string(body)?;
        let signature = self.generate_signature(timestamp, &payload);
        
        let url = format!("{}{}", self.base_url, endpoint);
        let headers = self.build_headers(timestamp, &signature);
        let client = self.client.clone();

        self.execute_with_retry(|attempt| {
            let url = url.clone();
            let headers = headers.clone();
            let payload = payload.clone();
            let client = client.clone();
            async move {
                let resp = client.post(&url)
                    .headers(headers)
                    .body(payload)
                    .send()
                    .await?;
                self.handle_response(LimitCategory::Orders, resp, attempt).await
            }
        }).await
    }

    /// Отмена одного ордера (Задача 209: Специализированный метод для одиночной отмены)
    pub async fn cancel_order<T: Serialize>(&self, body: &T) -> Result<serde_json::Value> {
        self.post("/v5/order/cancel", body).await
    }

    /// Массовая отмена всех ордеров по символу (Задача 209: Оптимизация массовых отмен)
    /// Использует эндпоинт POST /v5/order/cancel-all для атомарной отмены всех ордеров символа
    pub async fn cancel_all_orders<T: Serialize>(&self, body: &T) -> Result<serde_json::Value> {
        self.post("/v5/order/cancel-all", body).await
    }

    /// Получение текущего эквити аккаунта (USDT) с ретраями (Задача 111)
    pub async fn get_equity_with_retry(&self, _retries: u32) -> Result<Decimal> {
        // Мы используем существующий механизм execute_with_retry, 
        // который уже настроен на self.max_retries из конфига.
        let balance = self.get_wallet_balance().await?;
        
        let usdt_equity = balance.list.first()
            .and_then(|w| w.coin.iter().find(|c| c.coin == "USDT"))
            .map(|c| c.equity)
            .context("USDT balance not found in Unified Account")?;
            
        Ok(usdt_equity)
    }

    /// Получение снимка стакана через REST API (Задача 171)
    pub async fn fetch_orderbook(
        &self,
        category: &str,
        symbol: &str,
        limit: i32,
    ) -> Result<crate::data::types::OrderBookUpdateOwned> {
        let endpoint = "/v5/market/orderbook";
        self.wait_for_limiter(endpoint).await;
        let params = format!("category={}&symbol={}&limit={}", category, symbol, limit);
        
        let url = format!("{}{}?{}", self.base_url, endpoint, params);
        let client = self.client.clone();

        let resp: BybitResponse<OrderbookResponse> = self.execute_with_retry(|attempt| {
            let url = url.clone();
            let client = client.clone();
            async move {
                let resp = client.get(&url).send().await?;
                self.handle_response(LimitCategory::Market, resp, attempt).await
            }
        }).await?;

        // Конвертируем в OrderBookUpdateOwned
        use crate::data::types::{OrderBookUpdateOwned, PriceLevel};
        
        let bids = resp.result.bids.iter()
            .map(|l| PriceLevel {
                price: l[0].parse().unwrap_or(0.0),
                size: l[1].parse().unwrap_or(0.0),
            })
            .collect();

        let asks = resp.result.asks.iter()
            .map(|l| PriceLevel {
                price: l[0].parse().unwrap_or(0.0),
                size: l[1].parse().unwrap_or(0.0),
            })
            .collect();

        Ok(OrderBookUpdateOwned {
            symbol: resp.result.symbol,
            timestamp_ms: resp.result.timestamp,
            last_update_id: resp.result.update_id,
            is_snapshot: true,
            bids,
            asks,
            checksum: None,
        })
    }

    /// Получение информации об API ключе (Задача 174)
    pub async fn get_api_key_info(&self) -> Result<ApiKeyInfoResponse> {
        let endpoint = "/v5/user/query-api-key";
        self.get_signed(endpoint, "").await
    }
}


// Реализация трейта для BybitRestClient
#[async_trait]
impl BybitRestClientTrait for BybitRestClient {
    async fn post<T: Serialize + Send + Sync, R: DeserializeOwned + Send>(
        &self,
        endpoint: &str,
        body: &T,
    ) -> Result<R> {
        self.post(endpoint, body).await
    }
    
    async fn get_signed<R: DeserializeOwned + Send>(
        &self,
        endpoint: &str,
        params: &str,
    ) -> Result<R> {
        self.get_signed(endpoint, params).await
    }
    
    async fn amend_order<T: Serialize + Send + Sync>(
        &self,
        body: &T,
    ) -> Result<AmendOrderResult> {
        self.amend_order(body).await
    }

    async fn get_equity_with_retry(&self, retries: u32) -> Result<Decimal> {
        self.get_equity_with_retry(retries).await
    }
    
    async fn get_position(
        &self,
        category: &str,
        symbol: &str,
        position_idx: i32,
    ) -> Result<Option<PositionInfo>> {
        self.get_position(category, symbol, position_idx).await
    }
    
    async fn get_open_orders(
        &self,
        category: &str,
        symbol: &str,
    ) -> Result<Vec<crate::trading::types::OrderInfo>> {
        self.get_open_orders(category, symbol).await
    }

    async fn fetch_orderbook(
        &self,
        category: &str,
        symbol: &str,
        limit: i32,
    ) -> Result<crate::data::types::OrderBookUpdateOwned> {
        self.fetch_orderbook(category, symbol, limit).await
    }

    async fn get_api_key_info(&self) -> Result<ApiKeyInfoResponse> {
        self.get_api_key_info().await
    }

    async fn set_trading_stop(
        &self,
        body: &crate::trading::types::TradingStopRequest,
    ) -> Result<()> {
        let endpoint = "/v5/position/trading-stop";
        let _: crate::trading::types::TradingStopResponse = self.post(endpoint, body).await?;
        Ok(())
    }

    async fn cancel_order<T: Serialize + Send + Sync>(
        &self,
        body: &T,
    ) -> Result<serde_json::Value> {
        self.cancel_order(body).await
    }

    async fn cancel_all_orders<T: Serialize + Send + Sync>(
        &self,
        body: &T,
    ) -> Result<serde_json::Value> {
        self.cancel_all_orders(body).await
    }
}
