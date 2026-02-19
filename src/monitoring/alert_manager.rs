//! Alert Manager для интеллектуальной фильтрации и отправки уведомлений
//! 
//! Реализует систему алертов с:
//! - Маршрутизацией по уровням (Info, Warning, Critical)
//! - Дедупликацией с TTL
//! - Retry механизмом с экспоненциальным backoff
//! - Rate limiting для Telegram API
//! - Интеграцией с AuditLogger

use anyhow::{Context, Result};
use chrono::Utc;
use dashmap::DashMap;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time::sleep;
use tracing::{debug, error, info, warn};

use crate::utils::audit::AuditLogger;
use crate::utils::crypto;

/// Уровень критичности алерта
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlertLevel {
    Info,
    Warning,
    Critical,
}

impl std::fmt::Display for AlertLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlertLevel::Info => write!(f, "INFO"),
            AlertLevel::Warning => write!(f, "WARNING"),
            AlertLevel::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Структура алерта
#[derive(Debug, Clone)]
pub struct Alert {
    pub level: AlertLevel,
    pub message: String,
    pub source: String,
    pub timestamp: chrono::DateTime<Utc>,
}

impl Alert {
    /// Создает новый алерт
    pub fn new(level: AlertLevel, message: String, source: String) -> Self {
        Self {
            level,
            message,
            source,
            timestamp: Utc::now(),
        }
    }

    /// Вычисляет хеш для дедупликации
    fn compute_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.level.hash(&mut hasher);
        self.message.hash(&mut hasher);
        self.source.hash(&mut hasher);
        hasher.finish()
    }
}

/// Запись в кеше дедупликации
struct DeduplicationEntry {
    last_sent: Instant,
}

/// Конфигурация AlertManager
#[derive(Debug, Clone)]
pub struct AlertManagerConfig {
    pub telegram_token: String,
    pub chat_id: String,
    pub dedup_ttl_secs: u64,
    pub max_retries: u32,
    pub initial_retry_delay_ms: u64,
    pub max_retry_delay_ms: u64,
}

/// Alert Manager
pub struct AlertManager {
    config: AlertManagerConfig,
    client: Client,
    dedup_cache: Arc<DashMap<u64, DeduplicationEntry>>,
    tx: mpsc::Sender<Alert>,
    audit_logger: Option<Arc<AuditLogger>>,
}

impl AlertManager {
    /// Создает новый AlertManager
    /// 
    /// # Аргументы
    /// * `telegram_token` - Telegram Bot Token (поддерживает префикс ENC:)
    /// * `chat_id` - ID чата для отправки алертов
    /// * `master_password` - Мастер-пароль для расшифровки токена
    /// * `audit_logger` - Опциональный AuditLogger для логирования критических алертов
    pub fn new(
        telegram_token: String,
        chat_id: String,
        dedup_ttl_secs: u64,
        master_password: Option<&str>,
        audit_logger: Option<Arc<AuditLogger>>,
    ) -> Result<Arc<Self>> {
        // Расшифровываем токен, если он зашифрован
        let decrypted_token = if crypto::is_encrypted(&telegram_token) {
            let password = master_password
                .context("Master password required for encrypted telegram_token")?;
            crypto::decrypt(&telegram_token, password)
                .context("Failed to decrypt telegram_token")?
        } else {
            telegram_token
        };

        let config = AlertManagerConfig {
            telegram_token: decrypted_token,
            chat_id,
            dedup_ttl_secs,
            max_retries: 3,
            initial_retry_delay_ms: 1000,
            max_retry_delay_ms: 30000,
        };

        let client = Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .context("Failed to create HTTP client")?;

        let dedup_cache = Arc::new(DashMap::new());
        // Ограниченная очередь (bounded queue) для предотвращения OOM
        let (tx, rx) = mpsc::channel(100);

        let manager = Arc::new(Self {
            config: config.clone(),
            client: client.clone(),
            dedup_cache: dedup_cache.clone(),
            tx,
            audit_logger,
        });

        // Запускаем фоновую задачу обработки алертов
        let manager_clone = manager.clone();
        tokio::spawn(async move {
            manager_clone.process_alerts(rx).await;
        });

        // Запускаем фоновую задачу очистки кеша
        let dedup_cache_clone = dedup_cache.clone();
        let ttl_secs = config.dedup_ttl_secs;
        tokio::spawn(async move {
            Self::cleanup_cache(dedup_cache_clone, ttl_secs).await;
        });

        info!("AlertManager initialized");
        Ok(manager)
    }

    /// Отправляет алерт
    pub fn send_alert(&self, alert: Alert) {
        // Используем try_send для неблокирующей отправки
        // Если очередь переполнена, логируем ошибку
        if let Err(e) = self.tx.try_send(alert) {
            error!("Failed to send alert to queue (queue full or closed): {}", e);
        }
    }

    /// Обрабатывает очередь алертов
    async fn process_alerts(self: Arc<Self>, mut rx: mpsc::Receiver<Alert>) {
        while let Some(alert) = rx.recv().await {
            self.handle_alert(alert).await;
        }
    }

    /// Обрабатывает один алерт
    async fn handle_alert(&self, alert: Alert) {
        // Проверяем дедупликацию
        let hash = alert.compute_hash();
        if let Some(entry) = self.dedup_cache.get(&hash) {
            let elapsed = entry.last_sent.elapsed();
            if elapsed.as_secs() < self.config.dedup_ttl_secs {
                debug!(
                    "Alert deduplicated: {} (last sent {} seconds ago)",
                    alert.message,
                    elapsed.as_secs()
                );
                return;
            }
        }

        // Маршрутизация по уровням
        match alert.level {
            AlertLevel::Info => {
                // Только логирование
                info!("[ALERT] {}: {}", alert.source, alert.message);
            }
            AlertLevel::Warning => {
                // Логирование + запись в аудит
                warn!("[ALERT] {}: {}", alert.source, alert.message);
                if let Some(ref audit) = self.audit_logger {
                    if let Err(e) = audit.log_event(
                        &alert.source,
                        "ALERT_WARNING",
                        "TRIGGERED",
                        "",
                        &alert.message,
                    ) {
                        error!("Failed to log warning alert to audit: {}", e);
                    }
                }
            }
            AlertLevel::Critical => {
                // Логирование + аудит + Telegram
                error!("[ALERT] {}: {}", alert.source, alert.message);
                
                // Записываем в аудит
                if let Some(ref audit) = self.audit_logger {
                    if let Err(e) = audit.log_event(
                        &alert.source,
                        "ALERT_CRITICAL",
                        "TRIGGERED",
                        "",
                        &alert.message,
                    ) {
                        error!("Failed to log critical alert to audit: {}", e);
                    }
                }

                // Отправляем в Telegram
                if let Err(e) = self.send_to_telegram(&alert).await {
                    error!("Failed to send alert to Telegram: {}", e);
                }
            }
        }

        // Обновляем кеш дедупликации
        self.dedup_cache.insert(hash, DeduplicationEntry {
            last_sent: Instant::now(),
        });
    }

    /// Отправляет алерт в Telegram с retry механизмом
    async fn send_to_telegram(&self, alert: &Alert) -> Result<()> {
        let formatted_message = format!(
            "🚨 *{}*\n\n*Source:* {}\n*Time:* {}\n\n{}",
            alert.level,
            alert.source,
            alert.timestamp.format("%Y-%m-%d %H:%M:%S UTC"),
            alert.message
        );

        let mut attempt = 0;
        let mut delay_ms = self.config.initial_retry_delay_ms;

        loop {
            match self.try_send_telegram(&formatted_message).await {
                Ok(_) => {
                    debug!("Alert sent to Telegram successfully");
                    return Ok(());
                }
                Err(e) => {
                    attempt += 1;
                    
                    if attempt >= self.config.max_retries {
                        return Err(anyhow::anyhow!(
                            "Failed to send alert after {} attempts: {}",
                            attempt,
                            e
                        ));
                    }

                    warn!(
                        "Failed to send alert (attempt {}/{}): {}. Retrying in {}ms...",
                        attempt, self.config.max_retries, e, delay_ms
                    );

                    sleep(Duration::from_millis(delay_ms)).await;
                    
                    // Экспоненциальный backoff
                    delay_ms = (delay_ms * 2).min(self.config.max_retry_delay_ms);
                }
            }
        }
    }

    /// Пытается отправить сообщение в Telegram
    async fn try_send_telegram(&self, message: &str) -> Result<()> {
        let url = format!(
            "https://api.telegram.org/bot{}/sendMessage",
            self.config.telegram_token
        );

        let payload = serde_json::json!({
            "chat_id": self.config.chat_id,
            "text": message,
            "parse_mode": "Markdown",
        });

        let response = self.client
            .post(&url)
            .json(&payload)
            .send()
            .await
            .context("Failed to send HTTP request")?;

        let status = response.status();

        if status.is_success() {
            return Ok(());
        }

        // Обработка rate limiting (429)
        if status.as_u16() == 429 {
            let body: serde_json::Value = response.json().await
                .context("Failed to parse 429 response")?;
            
            let retry_after = body["parameters"]["retry_after"]
                .as_u64()
                .unwrap_or(60);

            warn!("Telegram rate limit hit. Retry after {} seconds", retry_after);
            sleep(Duration::from_secs(retry_after)).await;
            
            return Err(anyhow::anyhow!("Rate limited by Telegram API"));
        }

        // Другие ошибки
        let body = response.text().await
            .unwrap_or_else(|_| "Failed to read response body".to_string());
        
        Err(anyhow::anyhow!(
            "Telegram API error (status {}): {}",
            status,
            body
        ))
    }

    /// Фоновая задача очистки кеша дедупликации
    async fn cleanup_cache(cache: Arc<DashMap<u64, DeduplicationEntry>>, ttl_secs: u64) {
        let mut interval = tokio::time::interval(Duration::from_secs(60));
        
        loop {
            interval.tick().await;
            
            let now = Instant::now();
            let mut removed = 0;

            cache.retain(|_, entry| {
                let keep = now.duration_since(entry.last_sent).as_secs() < ttl_secs;
                if !keep {
                    removed += 1;
                }
                keep
            });

            if removed > 0 {
                debug!("Cleaned up {} expired deduplication entries", removed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alert_hash() {
        let alert1 = Alert::new(
            AlertLevel::Critical,
            "Test message".to_string(),
            "TestSource".to_string(),
        );

        let alert2 = Alert::new(
            AlertLevel::Critical,
            "Test message".to_string(),
            "TestSource".to_string(),
        );

        // Одинаковые алерты должны иметь одинаковый хеш
        assert_eq!(alert1.compute_hash(), alert2.compute_hash());

        let alert3 = Alert::new(
            AlertLevel::Warning,
            "Test message".to_string(),
            "TestSource".to_string(),
        );

        // Разные уровни - разные хеши
        assert_ne!(alert1.compute_hash(), alert3.compute_hash());
    }

    #[test]
    fn test_alert_level_display() {
        assert_eq!(format!("{}", AlertLevel::Info), "INFO");
        assert_eq!(format!("{}", AlertLevel::Warning), "WARNING");
        assert_eq!(format!("{}", AlertLevel::Critical), "CRITICAL");
    }
}
