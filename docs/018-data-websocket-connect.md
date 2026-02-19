# 018 - Data WebSocket Connect

Цель задачи: Реализовать WebSocket-клиент для публичного потока Bybit V5. Клиент должен поддерживать стабильное соединение для одного символа, выполнять подписку на стакан (orderbook.50.SYMBOL), отправлять регулярные Heartbeat (Ping) сообщения и автоматически переподключаться при разрыве соединения с использованием экспоненциальной задержки (backoff).

Файлы для изменения/создания:

Cargo.toml (обновить зависимости)
src/data/websocket.rs (создать)
src/data/mod.rs (обновить)
Инструкции для Gemini:

Добавить зависимости в Cargo.toml:

tokio-tungstenite = { version = "0.23", features = ["native-tls"] }
futures-util = "0.3"
url = "2.5"
src/data/websocket.rs: Реализовать структуру BybitWsClient, которая разделяет логику управления соединением и самого цикла обработки сообщений.

use anyhow::{Result, Context};
use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc::Sender;
use tokio::time::{sleep, Duration, interval};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use tracing::{info, warn, error};
use url::Url;

use crate::config::types::ExchangeConfig;

pub struct BybitWsClient {
    config: ExchangeConfig,
    symbol: String,
}

impl BybitWsClient {
    pub fn new(config: ExchangeConfig, symbol: String) -> Self {
        Self { config, symbol }
    }

    /// Главный цикл с логикой переподключения (Backoff)
    pub async fn run(&self, tx: Sender<String>) -> Result<()> {
        let mut backoff_ms = 1000;
        loop {
            match self.connect_and_subscribe(&tx).await {
                Ok(_) => info!("[{}] Connection closed cleanly", self.symbol),
                Err(e) => warn!("[{}] Connection error: {}. Reconnecting in {}ms...", self.symbol, e, backoff_ms),
            }

            sleep(Duration::from_millis(backoff_ms)).await;
            backoff_ms = (backoff_ms * 2).min(60000); // Максимум 60 секунд
        }
    }

    async fn connect_and_subscribe(&self, tx: &Sender<String>) -> Result<()> {
        let url = &self.config.websocket.public_url;
        let (mut ws_stream, _) = connect_async(Url::parse(url)?).await
            .context("Failed to connect to Bybit WS")?;

        info!("[{}] Connected to Bybit WS", self.symbol);

        // Подписка на стакан
        let sub_msg = format!(r#"{{"op": "subscribe", "args": ["orderbook.50.{}"]}}"#, self.symbol);
        ws_stream.send(Message::Text(sub_msg.into())).await?;

        // Интервал для пингов (Bybit требует каждые 20 сек)
        let mut ping_interval = interval(Duration::from_secs(self.config.websocket.ping_interval_sec));

        loop {
            tokio::select! {
                // Отправка Ping
                _ = ping_interval.tick() => {
                    ws_stream.send(Message::Text(r#"{"op": "ping"}"#.into())).await?;
                }
                // Получение сообщений
                msg = ws_stream.next() => {
                    match msg {
                        Some(Ok(Message::Text(text))) => {
                            // Передаем сырой JSON в канал
                            if tx.send(text).await.is_err() {
                                error!("[{}] Receiver dropped, stopping WS client", self.symbol);
                                return Ok(());
                            }
                        }
                        Some(Ok(Message::Pong(_))) => { /* Ок, игнорируем */ }
                        Some(Err(e)) => return Err(anyhow::anyhow!("WS message error: {}", e)),
                        None => return Err(anyhow::anyhow!("WS connection closed by server")),
                        _ => {}
                    }
                }
            }
        }
    }
}
Технические требования:

Переподключение: Обязательный бесконечный цикл loop в методе run с удвоением времени ожидания при ошибках.
Heartbeat: Использование tokio::select! для одновременного ожидания сообщений и тиков таймера ping_interval.
Конфигурация: URL и интервал пинга брать строго из ExchangeConfig.
Изоляция: Один инстанс клиента — один символ.
Почему это важно: В трейдинге соединение может обрываться из-за сетевых лагов или перезагрузки серверов биржи. Без автоматического реконнекта и ручного пинга бот перестанет получать данные через несколько минут после запуска. Использование mpsc канала позволяет передавать данные на обработку мгновенно, не блокируя чтение из сокета.