use anyhow::{Result, Context};
use futures_util::{SinkExt, StreamExt};
#[cfg(feature = "chaos")]
use rand_distr::{Exponential, Distribution};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use tokio::sync::mpsc::Sender;
use tokio::time::{sleep, Duration, interval, Instant};
use tokio_util::sync::CancellationToken;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message, tungstenite::protocol::frame::CloseFrame};
use tracing::{info, warn, error, debug};
use serde::Deserialize;
use hmac::{Hmac, Mac};
use sha2::Sha256;
use tokio::sync::mpsc::Receiver;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReconnectSignal {
    Immediate,
}

use crate::config::types::ExchangeConfig;

#[cfg(feature = "chaos")]
pub async fn inject_chaos(config: &ChaosConfig, rng: &mut StdRng) -> bool {
    // 1. Packet Loss (simulating sequence gaps)
    if config.packet_loss_rate > 0.0 && rng.gen_bool(config.packet_loss_rate) {
        warn!("Chaos Monkey: Simulating packet loss");
        return true; // Skip message, trigger gap detection
    }
    
    // 2. Latency with Long Tail (Exponential)
    if config.mean_latency_ms > 0 {
        let exp = Exponential::new(1.0 / config.mean_latency_ms as f64).unwrap();
        let delay = exp.sample(rng) as u64;
        debug!("Chaos Monkey: Injecting latency {}ms", delay);
        sleep(Duration::from_millis(delay)).await;
    }
    false
}
use crate::data::types::WsData;
use crate::data::parser::{parse_orderbook_msg, parse_public_trade_msg, parse_ticker_msg};
use crate::utils::helpers::now_secs;
use crate::utils::backoff::ExponentialBackoff;
use smallvec::SmallVec;

pub struct BybitWsClient {
    config: ExchangeConfig,
    symbol: String,
    last_activity: Arc<AtomicU64>,
    last_ping_sent_at: Arc<Mutex<Option<Instant>>>,
    #[cfg(feature = "chaos")]
    rng: Arc<Mutex<StdRng>>,
}

pub struct BybitPrivateWsClient {
    config: ExchangeConfig,
    last_activity: Arc<AtomicU64>,
    last_ping_sent_at: Arc<Mutex<Option<Instant>>>,
}

#[derive(Debug, Deserialize)]
pub struct BybitPong {
    pub op: String,                // Всегда "pong"
    pub req_id: Option<String>,    // Если передавали в ping
    pub conn_id: String,           // ID соединения на стороне Bybit
    pub ts: u64,                   // Timestamp сервера в мс
    #[serde(default)]
    pub ret_msg: Option<String>,   // Может содержать "pong" или "OK"
}

#[derive(Debug, Deserialize)]
pub struct BybitAuthResponse {
    pub op: String,
    pub success: bool,
    pub ret_msg: String,
    pub conn_id: String,
}

impl BybitPrivateWsClient {
    pub fn new(config: ExchangeConfig) -> Self {
        Self {
            config,
            last_activity: Arc::new(AtomicU64::new(now_secs())),
            last_ping_sent_at: Arc::new(Mutex::new(None)),
        }
    }

    fn generate_auth_message(&self) -> Result<Message> {
        let api_key = std::env::var("BYBIT_API_KEY")
            .context("BYBIT_API_KEY not found in environment")?;
        let api_secret = std::env::var("BYBIT_API_SECRET")
            .context("BYBIT_API_SECRET not found in environment")?;

        let expires = chrono::Utc::now().timestamp_millis() + 10000;
        let prehash = format!("GET/realtime{}", expires);
        
        let mut mac = Hmac::<Sha256>::new_from_slice(api_secret.as_bytes())
            .expect("HMAC can take key of any size");
        mac.update(prehash.as_bytes());
        let signature = hex::encode(mac.finalize().into_bytes());

        let auth_msg = serde_json::json!({
            "op": "auth",
            "args": [api_key, expires, signature]
        });

        Ok(Message::Text(auth_msg.to_string().into()))
    }

    pub async fn run(&self, tx: Sender<serde_json::Value>, mut reconnect_rx: Receiver<ReconnectSignal>, token: CancellationToken) -> Result<()> {
        let mut backoff = ExponentialBackoff::new(
            Duration::from_millis(self.config.ws_retry_initial_ms),
            Duration::from_millis(self.config.ws_retry_max_ms),
            self.config.ws_retry_multiplier,
            self.config.ws_retry_jitter,
        );
        let max_attempts = self.config.websocket.max_attempts;
        let mut attempt = 0;

        loop {
            if token.is_cancelled() {
                return Ok(());
            }

            match self.connect_and_auth(&tx, token.clone()).await {
                Ok(_) => {
                    info!("Private WS Connection closed cleanly");
                    backoff.reset();
                    attempt = 0;
                }
                Err(e) => {
                    if token.is_cancelled() {
                        return Ok(());
                    }
                    attempt += 1;
                    if let Some(max) = max_attempts {
                        if attempt > max {
                            error!("Max private WS connection attempts reached ({}). Stopping.", max);
                            return Err(e);
                        }
                    }

                    let delay = backoff.next_delay();
                    warn!("Private WS Disconnected: {}. Reconnecting in {:?} (Attempt {})", e, delay, attempt);
                    
                    tokio::select! {
                        _ = sleep(delay) => {},
                        Some(signal) = reconnect_rx.recv() => {
                            if signal == ReconnectSignal::Immediate {
                                info!("Private WS: Received Immediate reconnect signal during backoff.");
                            }
                        }
                        _ = token.cancelled() => return Ok(()),
                    }
                }
            }
        }
    }

    async fn connect_and_auth(&self, tx: &Sender<serde_json::Value>, token: CancellationToken) -> Result<()> {
        let url = &self.config.websocket.private_ws_url;
        
        // Используем connect_async для WebSocket. SChannel (TLS) в Windows обработается автоматически.
        let (ws_stream, _) = connect_async(url).await
            .context("Failed to connect to Bybit Private WS")?;

        info!("Connected to Bybit Private WS");

        let (mut ws_sink, mut ws_read) = ws_stream.split();

        // 1. Авторизация
        let auth_msg = self.generate_auth_message()?;
        ws_sink.send(auth_msg).await?;

        // 2. Ожидание подтверждения авторизации
        loop {
            tokio::select! {
                _ = token.cancelled() => {
                    let _ = ws_sink.send(Message::Close(Some(CloseFrame {
                        code: 1000.into(),
                        reason: "Graceful shutdown".into(),
                    }))).await;
                    return Ok(());
                }
                msg = ws_read.next() => {
                    let msg = msg.context("Private WS connection closed during auth")??;
                    if let Message::Text(text) = msg {
                        let resp: serde_json::Value = serde_json::from_str(&text)?;
                        if resp["op"] == "auth" {
                            if resp["success"] == true {
                                info!("Private WS authenticated successfully");
                                break;
                            } else {
                                error!("Private WS Auth failed: {}", text);
                                return Err(anyhow::anyhow!("Private WS Auth failed"));
                            }
                        }
                    }
                }
            }
        }

        // 3. Подписка
        let sub_msg = serde_json::json!({
            "op": "subscribe",
            "args": ["order", "execution", "position", "wallet"]
        });
        ws_sink.send(Message::Text(sub_msg.to_string().into())).await?;
        info!("Private WS subscriptions sent");

        // 4. Heartbeat логика
        self.last_activity.store(now_secs(), Ordering::Relaxed);
        let ping_interval = Duration::from_secs(self.config.websocket.ping_interval_sec);
        let timeout = Duration::from_secs(self.config.websocket.pong_timeout_sec);
        
        let (reconnect_tx, mut reconnect_rx) = tokio::sync::mpsc::channel(1);
        let (ping_tx, mut ping_rx) = tokio::sync::mpsc::channel(1);
        
        let last_activity_clone = self.last_activity.clone();
        let last_ping_sent_at_clone = self.last_ping_sent_at.clone();
        let token_heartbeat = token.clone();

        tokio::spawn(async move {
            let mut interval = interval(ping_interval);
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let last = last_activity_clone.load(Ordering::Relaxed);
                        let elapsed = now_secs().saturating_sub(last);
                        
                        if elapsed > timeout.as_secs() {
                            warn!("Private WS Heartbeat timeout: {}s. Triggering reconnect...", elapsed);
                            let _ = reconnect_tx.send(ReconnectSignal::Immediate).await;
                            break;
                        }

                        if let Ok(mut lock) = last_ping_sent_at_clone.lock() {
                            *lock = Some(Instant::now());
                        }

                        if let Err(_) = ping_tx.send(()).await {
                            error!("Failed to send ping signal");
                            let _ = reconnect_tx.send(ReconnectSignal::Immediate).await;
                            break;
                        }
                    }
                    _ = token_heartbeat.cancelled() => {
                        break;
                    }
                }
            }
        });

        // 5. Основной цикл чтения
        loop {
            tokio::select! {
                _ = token.cancelled() => {
                    info!("Private WS stopping due to cancellation");
                    let _ = ws_sink.send(Message::Close(Some(CloseFrame {
                        code: 1000.into(),
                        reason: "Graceful shutdown".into(),
                    }))).await;
                    return Ok(());
                }
                _ = ping_rx.recv() => {
                    if let Err(e) = ws_sink.send(Message::Text(r#"{"op":"ping"}"#.into())).await {
                        error!("Failed to send Private WS ping: {}", e);
                        return Err(anyhow::anyhow!("Failed to send ping"));
                    }
                }
                msg = ws_read.next() => {
                    match msg {
                        Some(Ok(msg)) => {
                            self.last_activity.store(now_secs(), Ordering::Relaxed);

                            match msg {
                                Message::Text(text) => {
                                    // Проверка на Pong
                                    if text.contains(r#""op":"pong""#) || text.contains(r#""op": "pong""#) {
                                        if let Ok(pong) = serde_json::from_str::<BybitPong>(&text) {
                                            if let Ok(mut lock) = self.last_ping_sent_at.lock() {
                                                if let Some(sent_at) = lock.take() {
                                                    let rtt = sent_at.elapsed();
                                                    debug!("Private WS Pong [{}]: RTT = {:?}, ServerTS = {}", pong.conn_id, rtt, pong.ts);
                                                    if rtt.as_millis() > self.config.websocket.warn_rtt_ms as u128 {
                                                        warn!("Private WS High RTT: {:?}", rtt);
                                                    }
                                                }
                                            }
                                            continue;
                                        }
                                    }

                                    // Отправляем событие в канал
                                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(&text) {
                                        // Расчет E2E Latency для приватных сообщений (если есть ts)
                                        if let Some(ts) = val["ts"].as_u64() {
                                            let local_recv_ms = chrono::Utc::now().timestamp_millis() as u64;
                                            let e2e = local_recv_ms.saturating_sub(ts);
                                            crate::monitoring::latency::E2E_LATENCY.update(e2e);
                                        }

                                        if let Err(e) = tx.try_send(val) {
                                            error!("Private WS channel overflow or closed: {}", e);
                                            return Err(anyhow::anyhow!("Private channel overflow"));
                                        }
                                    }
                                }
                                Message::Close(_) => return Err(anyhow::anyhow!("Private WS closed by server")),
                                _ => {}
                            }
                        }
                        Some(Err(e)) => return Err(anyhow::anyhow!("Private WS error: {}", e)),
                        None => return Err(anyhow::anyhow!("Private WS connection lost")),
                    }
                }
                _ = reconnect_rx.recv() => {
                    return Err(anyhow::anyhow!("Private Heartbeat watchdog triggered reconnect"));
                }
            }
        }
    }
}

impl BybitWsClient {
    pub fn new(config: ExchangeConfig, symbol: String) -> Self {
        Self { 
            config, 
            symbol,
            last_activity: Arc::new(AtomicU64::new(now_secs())),
            last_ping_sent_at: Arc::new(Mutex::new(None)),
            #[cfg(feature = "chaos")]
            rng: Arc::new(Mutex::new(StdRng::from_os_rng())),
        }
    }

    /// Главный цикл с логикой переподключения (Exponential Backoff + Jitter)
    pub async fn run(&self, tx: Sender<crate::data::types::WsData>, mut reconnect_rx: Receiver<ReconnectSignal>, token: CancellationToken) -> Result<()> {
        let mut backoff = ExponentialBackoff::new(
            Duration::from_millis(self.config.ws_retry_initial_ms),
            Duration::from_millis(self.config.ws_retry_max_ms),
            self.config.ws_retry_multiplier,
            self.config.ws_retry_jitter,
        );
        let max_attempts = self.config.websocket.max_attempts;
        let mut attempt = 0;

        loop {
            if token.is_cancelled() {
                return Ok(());
            }

            match self.connect_and_subscribe(&tx, token.clone()).await {
                Ok(_) => {
                    info!("[{}] Connection closed cleanly", self.symbol);
                    backoff.reset();
                    attempt = 0;
                }
                Err(e) => {
                    if token.is_cancelled() {
                        return Ok(());
                    }
                    attempt += 1;
                    if let Some(max) = max_attempts {
                        if attempt > max {
                            error!("[{}] Max connection attempts reached ({}). Stopping.", self.symbol, max);
                            return Err(e);
                        }
                    }

                    let delay = backoff.next_delay();
                    warn!(
                        "[{}] WS Disconnected: {}. Reconnecting in {:?} (Attempt {})", 
                        self.symbol, e, delay, attempt
                    );

                    tokio::select! {
                        _ = sleep(delay) => {},
                        Some(signal) = reconnect_rx.recv() => {
                            if signal == ReconnectSignal::Immediate {
                                info!("[{}] WS: Received Immediate reconnect signal during backoff.", self.symbol);
                            }
                        }
                        _ = token.cancelled() => return Ok(()),
                    }
                }
            }
        }
    }

    async fn connect_and_subscribe(&self, tx: &Sender<WsData>, token: CancellationToken) -> Result<()> {
        let url = &self.config.websocket.public_url;
        
        // Используем connect_async для WebSocket. SChannel (TLS) в Windows обработается автоматически.
        let (ws_stream, _) = connect_async(url).await
            .context("Failed to connect to Bybit Public WS")?;

        info!("[{}] Connected to Bybit WS", self.symbol);

        let (mut ws_sink, mut ws_read) = ws_stream.split();

        // Подписка на стакан (глубина 50 уровней), публичные сделки и тикеры (включают markPrice в V5)
        let sub_msg = format!(
            r#"{{"op": "subscribe", "args": ["orderbook.50.{}", "publicTrade.{}", "tickers.{}"]}}"#, 
            self.symbol, self.symbol, self.symbol
        );
        ws_sink.send(Message::Text(sub_msg.into())).await?;
        
        info!("[{}] WS Connected and Subscribed successfully (including markPrice)", self.symbol);

        self.last_activity.store(now_secs(), Ordering::Relaxed);
        let ping_interval = Duration::from_secs(self.config.websocket.ping_interval_sec);
        let timeout = Duration::from_secs(self.config.websocket.pong_timeout_sec);
        
        let (reconnect_tx, mut reconnect_rx) = tokio::sync::mpsc::channel::<ReconnectSignal>(1);
        
        let last_activity_clone = self.last_activity.clone();
        let last_ping_sent_at_clone = self.last_ping_sent_at.clone();
        let symbol_clone = self.symbol.clone();
        let token_heartbeat = token.clone();

        // Фоновая задача Heartbeat
        tokio::spawn(async move {
            let mut interval = interval(ping_interval);
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let last = last_activity_clone.load(Ordering::Relaxed);
                        let elapsed = now_secs().saturating_sub(last);
                        
                        if elapsed > timeout.as_secs() {
                            warn!("[{}] WS Heartbeat timeout: {}s. Triggering reconnect...", symbol_clone, elapsed);
                            let _ = reconnect_tx.send(ReconnectSignal::Immediate).await;
                            break;
                        }

                        // Фиксируем время отправки пинга
                        {
                            if let Ok(mut lock) = last_ping_sent_at_clone.lock() {
                                *lock = Some(Instant::now());
                            }
                        }

                        // Отправка пинга
                        if let Err(e) = ws_sink.send(Message::Text(r#"{"op":"ping"}"#.into())).await {
                            error!("[{}] Failed to send WS ping: {}", symbol_clone, e);
                            let _ = reconnect_tx.send(ReconnectSignal::Immediate).await;
                            break;
                        }
                    }
                    _ = token_heartbeat.cancelled() => {
                        let _ = ws_sink.send(Message::Close(Some(CloseFrame {
                            code: 1000.into(),
                            reason: "Shutdown".into(),
                        }))).await;
                        break;
                    }
                }
            }
        });

        // Логика инициализации (буферизация до Snapshot)
        let mut is_synced = false;
        let mut init_buffer: SmallVec<[crate::data::types::OrderBookUpdateArc; 100]> = SmallVec::new();
        let mut last_u: u64 = 0;

        loop {
            tokio::select! {
                _ = token.cancelled() => {
                    info!("[{}] WS stopping due to cancellation", self.symbol);
                    return Ok(());
                }
                msg = ws_read.next() => {
                    let local_recv_ms = chrono::Utc::now().timestamp_millis() as u64;
                    match msg {
                        Some(Ok(msg)) => {
                            // Chaos Monkey (задача 147)
                            #[cfg(feature = "chaos")]
                            {
                                if let Some(ref chaos_config) = self.config.websocket.chaos {
                                    if let Ok(mut rng) = self.rng.lock() {
                                        if inject_chaos(chaos_config, &mut rng).await {
                                            continue; // Skip message, trigger gap detection
                                        }
                                    }
                                }
                            }

                            // Обновляем время активности при любом сообщении
                            self.last_activity.store(now_secs(), Ordering::Relaxed);

                            match msg {
                                Message::Text(text) => {
                                    // Инкремент счетчика WebSocket сообщений (задача 189)
                                    metrics::counter!("bot_ws_messages_total").increment(1);
                                    
                                    // Сначала проверяем на Pong
                                    if text.contains(r#""op":"pong""#) || text.contains(r#""op": "pong""#) {
                                        // Обновляем время активности при получении Pong (Задача 303)
                                        self.last_activity.store(now_secs(), Ordering::Relaxed);
                                        
                                        if let Ok(pong) = serde_json::from_str::<BybitPong>(&text) {
                                            let mut rtt = None;
                                            if let Ok(mut lock) = self.last_ping_sent_at.lock() {
                                                if let Some(sent_at) = lock.take() {
                                                    rtt = Some(sent_at.elapsed());
                                                }
                                            }
                                            
                                            if let Some(r) = rtt {
                                                debug!("[{}] WS Pong [{}]: RTT = {:?}, ServerTS = {}", self.symbol, pong.conn_id, r, pong.ts);
                                                if r.as_millis() > self.config.websocket.warn_rtt_ms as u128 {
                                                    warn!("[{}] High RTT: {:?}", self.symbol, r);
                                                }
                                            }
                                            continue;
                                        }
                                    }

                                    // Замер времени парсинга JSON (Задача 199)
                                    let start_parse = Instant::now();
                                    let parse_res = parse_orderbook_msg(&text);
                                    let parse_duration = start_parse.elapsed().as_micros() as u64;
                                    crate::monitoring::latency::HOT_PATH_STATS.record_json_parsing(parse_duration);

                                    match parse_res {
                                        Ok(Some(update)) => {
                                            // Инкремент счетчика тиков для Prometheus (задача 143)
                                            if let Some(counter) = crate::monitoring::prometheus::TICK_COUNTER.get() {
                                                counter.with_label_values(&[self.symbol.as_str()]).inc();
                                            }
                                            
                                            // Расчет E2E Latency
                                            let e2e = local_recv_ms.saturating_sub(update.timestamp_ms);
                                            if e2e < 5 {
                                                warn!("[{}] Local clock drift detected or ultra-low latency: {}ms", self.symbol, e2e);
                                            }
                                            crate::monitoring::latency::E2E_LATENCY.update(e2e);

                                            if !is_synced {
                                                if update.is_snapshot {
                                                    info!("[{}] Received Snapshot (u={}). Synchronizing buffer...", self.symbol, update.last_update_id);
                                                    
                                                    // Логирование checksum для отладки (Задача 049)
                                                    if self.config.websocket.verify_checksum {
                                                        if let Some(remote_cs) = update.checksum {
                                                            debug!("[{}] Snapshot received with checksum: {} (will be verified in main loop)", self.symbol, remote_cs);
                                                        }
                                                    }
                                                    
                                                    is_synced = true;
                                                    last_u = update.last_update_id;
                                                    
                                                    // Конвертируем borrowed в Arc версию (zero-copy для symbol)
                                                    if let Err(e) = tx.try_send(crate::data::types::WsData::OrderBook(update.to_arc())) {
                                                        error!("[{}] Failed to send snapshot to channel: {}", self.symbol, e);
                                                        return Err(anyhow::anyhow!("Channel overflow or closed"));
                                                    }

                                                    // Применяем буферизованные дельты
                                                    init_buffer.sort_by_key(|u| u.last_update_id);
                                                    for delta in init_buffer.drain(..) {
                                                        if delta.last_update_id <= last_u { continue; }
                                                        
                                                        // Проверка на пропуск (Bybit: дельты должны быть последовательны u, u+1)
                                                        if delta.last_update_id != last_u + 1 {
                                                            error!("[{}] Gap detected in init buffer: {} -> {}. Reconnecting...", self.symbol, last_u, delta.last_update_id);
                                                            return Err(anyhow::anyhow!("Init buffer sequence gap"));
                                                        }
                                                        
                                                        last_u = delta.last_update_id;
                                                        if let Err(e) = tx.try_send(crate::data::types::WsData::OrderBook(delta)) {
                                                            error!("[{}] Failed to send buffered delta to channel: {}", self.symbol, e);
                                                            return Err(anyhow::anyhow!("Channel overflow or closed"));
                                                        }
                                                    }
                                                    info!("[{}] WS Stream synchronized and live.", self.symbol);
                                                } else {
                                                    // Буферизуем дельту до снапшота (Arc версия)
                                                    if init_buffer.len() >= 100 {
                                                        error!("[{}] Init buffer overflow. Snapshot not received in time.", self.symbol);
                                                        return Err(anyhow::anyhow!("Init buffer overflow"));
                                                    }
                                                    init_buffer.push(update.to_arc());
                                                }
                                            } else {
                                                // Прямая трансляция с проверкой последовательности
                                                if update.last_update_id != last_u + 1 && !update.is_snapshot {
                                                    error!("[{}] Sequence gap: {} -> {}. Reconnecting...", self.symbol, last_u, update.last_update_id);
                                                    return Err(anyhow::anyhow!("Sequence gap"));
                                                }
                                                
                                                last_u = update.last_update_id;
                                                if let Err(e) = tx.try_send(crate::data::types::WsData::OrderBook(update.to_arc())) {
                                                    error!("[{}] Channel overflow or closed for {}: {}", self.symbol, self.symbol, e);
                                                    return Err(anyhow::anyhow!("Channel overflow or closed"));
                                                }
                                            }
                                        }
                                        Ok(None) => {} 
                                        Err(e) => warn!("[{}] Parser error: {}", self.symbol, e),
                                    }
                                    
                                    // Обработка публичных сделок
                                    match parse_public_trade_msg(&text) {
                                        Ok(Some(trades)) => {
                                            let arc_trades: Vec<_> = trades.into_iter()
                                                .map(|t| t.to_arc())
                                                .collect();
                                            if let Err(e) = tx.try_send(crate::data::types::WsData::Trades(arc_trades)) {
                                                error!("[{}] Failed to send public trades to channel: {}", self.symbol, e);
                                                return Err(anyhow::anyhow!("Channel overflow or closed"));
                                            }
                                        }
                                        Ok(None) => {} 
                                        Err(e) => warn!("[{}] PublicTrade parser error: {}", self.symbol, e),
                                    }
                                    
                                    // Обработка тикеров (задача 170: Funding Rate Filter)
                                    match parse_ticker_msg(&text) {
                                        Ok(Some(ticker)) => {
                                            let symbol: Arc<str> = Arc::from(ticker.symbol);
                                            
                                            // Если есть маркированная цена, отправляем её отдельно (Задача 233)
                                            if let Some(mp) = ticker.mark_price {
                                                if let Err(e) = tx.try_send(crate::data::types::WsData::MarkPrice(symbol.clone(), mp)) {
                                                    error!("[{}] Failed to send mark price from ticker: {}", self.symbol, e);
                                                }
                                            }

                                            // Отправляем сам тикер
                                            if let Err(e) = tx.try_send(crate::data::types::WsData::Ticker(ticker.to_arc())) {
                                                error!("[{}] Failed to send ticker to channel: {}", self.symbol, e);
                                                return Err(anyhow::anyhow!("Channel overflow or closed"));
                                            }
                                        }
                                        Ok(None) => {} 
                                        Err(e) => warn!("[{}] Ticker parser error: {}", self.symbol, e),
                                    }

                                    // Задача 303: Удалено ручное парсинг mark_price, так как он идет в тикерах
                                }
                                Message::Pong(_) => {}
                                Message::Close(_) => return Err(anyhow::anyhow!("WS connection closed by server")),
                                _ => {}
                            }
                        }
                        Some(Err(e)) => return Err(anyhow::anyhow!("WS message error: {}", e)),
                        None => return Err(anyhow::anyhow!("WS connection closed by server")),
                    }
                }
                Some(signal) = reconnect_rx.recv() => {
                    if signal == ReconnectSignal::Immediate {
                        return Err(anyhow::anyhow!("External Immediate reconnect signal received"));
                    } else {
                        return Err(anyhow::anyhow!("Heartbeat watchdog triggered reconnect"));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pong_parsing() {
        let json = r#"{"op":"pong","req_id":"123","conn_id":"abc","ts":1700000000000,"ret_msg":"OK"}"#;
        let pong: BybitPong = serde_json::from_str(json).unwrap();
        assert_eq!(pong.op, "pong");
        assert_eq!(pong.conn_id, "abc");
    }
}
