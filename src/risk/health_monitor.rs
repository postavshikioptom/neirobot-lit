use sysinfo::{System, RefreshKind, ProcessRefreshKind, ProcessesToUpdate};
use std::collections::VecDeque;
use tracing::{error, info};
use crate::config::types::RiskConfig;
use std::time::Instant;
use std::path::PathBuf;

use crate::trading::rest_client::ApiKeyInfoResponse;
use crate::config::ExchangeConfig;

/// Система самодиагностики для обеспечения технической безопасности торгов (Задача 171)
pub struct HealthMonitor {
    config: RiskConfig,
    last_u: u64,
    book_corrupted: bool,
    latencies: VecDeque<f64>,
    sys: System,
    pid: sysinfo::Pid,
    last_health_check: Instant,
    api_permissions_valid: bool,
    
    // Задача 178: Отслеживание волатильности для динамического сокращения лимитов
    price_changes: VecDeque<f64>,        // Последние 100 изменений цены
    last_mid_price: Option<f64>,         // Предыдущая mid_price для расчета дельты
    volatility_history: VecDeque<f64>,   // История волатильности для расчета медианы за 24 часа
    
    // Задача 180: Счетчик ошибок контрольной суммы
    checksum_mismatch_count: u32,
    
    // Задача 182: Архивация логов
    log_dir: Option<PathBuf>,
    last_archive_check: Instant,
}

impl HealthMonitor {
    pub fn new(config: RiskConfig) -> Self {
        let mut sys = System::new_with_specifics(
            RefreshKind::nothing().with_processes(ProcessRefreshKind::nothing().with_memory())
        );
        let pid = sysinfo::get_current_pid().expect("Failed to get current PID");
        
        // Первичная подгрузка данных о процессе
        sys.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[pid]),
            true,
            ProcessRefreshKind::nothing().with_memory(),
        );

        Self {
            config,
            last_u: 0,
            book_corrupted: false,
            latencies: VecDeque::with_capacity(60),
            sys,
            pid,
            last_health_check: Instant::now(),
            api_permissions_valid: true,
            price_changes: VecDeque::with_capacity(100),
            last_mid_price: None,
            volatility_history: VecDeque::with_capacity(1440), // ~24 часа при обновлении каждую минуту
            checksum_mismatch_count: 0,
            log_dir: None,
            last_archive_check: Instant::now(),
        }
    }

    /// Установка директории логов для архивации (Задача 182)
    pub fn set_log_dir(&mut self, log_dir: PathBuf) {
        self.log_dir = Some(log_dir);
    }

    /// Контроль целостности стакана (OrderBook Integrity)
    /// Возвращает true, если обнаружен разрыв в последовательности update ID
    pub fn check_u(&mut self, new_u: u64) -> bool {
        if self.last_u == 0 {
            self.last_u = new_u;
            return false;
        }

        if new_u != self.last_u + 1 {
            error!("OrderBook sequence gap detected! last_u: {}, new_u: {}", self.last_u, new_u);
            self.book_corrupted = true;
            self.last_u = new_u;
            return true; // Corrupted
        }

        self.last_u = new_u;
        false
    }

    /// Сброс флага коррупции после успешного ресинка (Задача 171)
    pub fn reset_corruption(&mut self) {
        if self.book_corrupted {
            info!("HealthMonitor: OrderBook corruption flag reset.");
            self.book_corrupted = false;
            self.last_u = 0;
        }
        // Сбрасываем счетчик ошибок checksum при успешном ресинке (Задача 180)
        self.checksum_mismatch_count = 0;
    }

    /// Обработка несоответствия контрольной суммы (Задача 180)
    /// Инкрементирует счетчик ошибок и при достижении лимита устанавливает флаг коррупции
    /// Возвращает true, если достигнут лимит и требуется полный ресинк
    pub fn checksum_mismatch(&mut self) -> bool {
        self.checksum_mismatch_count += 1;
        error!(
            "OrderBook checksum mismatch detected! Count: {} / {}",
            self.checksum_mismatch_count,
            self.config.max_checksum_mismatches
        );

        if self.checksum_mismatch_count >= self.config.max_checksum_mismatches {
            error!(
                "Max checksum mismatches reached ({} >= {}). Marking OrderBook as corrupted.",
                self.checksum_mismatch_count,
                self.config.max_checksum_mismatches
            );
            self.book_corrupted = true;
            return true; // Требуется полный ресинк
        }

        false
    }

    /// Мониторинг задержек (Latency Tracking)
    pub fn update_latency(&mut self, elapsed_ms: f64) {
        if self.latencies.len() >= 60 {
            self.latencies.pop_front();
        }
        self.latencies.push_back(elapsed_ms);
    }

    /// Проверка средней задержки пайплайна за последнюю минуту
    pub fn is_latency_ok(&self) -> bool {
        if self.latencies.is_empty() {
            return true;
        }
        let avg: f64 = self.latencies.iter().sum::<f64>() / self.latencies.len() as f64;
        avg <= self.config.max_avg_latency_ms as f64
    }

    /// Проверка разрешений API-ключа (Задача 174)
    pub fn validate_api_permissions(&mut self, info: &ApiKeyInfoResponse, config: &ExchangeConfig) -> anyhow::Result<()> {
        // 1. Проверка прав
        for req in &config.required_permissions {
            if !info.permissions.contains(req) {
                error!("[Health] Missing critical API permission: {}", req);
                self.api_permissions_valid = false;
                anyhow::bail!("Missing critical API permission: {}", req);
            }
        }
        self.api_permissions_valid = true;

        // 2. IP Security
        if !info.ip_restrict {
            tracing::warn!("[Security] No IP restriction on API key!");
        }

        // 3. Expiry Logic
        if config.check_api_expiry && info.expired_at != 0 {
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
                
            if info.expired_at > now_ms {
                let days_left = (info.expired_at - now_ms) / 86_400_000;
                if days_left < config.min_api_days_left as u64 {
                    tracing::warn!("[Health] API Key expires in {} days", days_left);
                }
            } else {
                error!("[Health] API Key has expired!");
                self.api_permissions_valid = false;
                anyhow::bail!("API Key has expired");
            }
        }

        Ok(())
    }

    /// Проверка здоровья системы (Sanity Check)
    pub fn is_sane(&mut self) -> anyhow::Result<()> {
        // 0. Проверка прав API
        if !self.api_permissions_valid {
            metrics::gauge!("bot_health_status").set(0.0); // Критическая ошибка
            anyhow::bail!("API Key permissions invalid or expired");
        }

        // 1. Проверка целостности стакана
        if self.book_corrupted {
            metrics::gauge!("bot_health_status").set(0.0); // Критическая ошибка
            anyhow::bail!("OrderBook sequence corrupted");
        }

        // 2. Проверка задержек
        if !self.is_latency_ok() {
            let avg: f64 = self.latencies.iter().sum::<f64>() / self.latencies.len() as f64;
            error!("System Health: Pipeline latency too high: {:.2}ms (limit: {}ms)", 
                   avg, self.config.max_avg_latency_ms);
            metrics::gauge!("bot_health_status").set(0.0); // Критическая ошибка
            anyhow::bail!("Pipeline latency too high: {:.2}ms", avg);
        }

        // 3. Периодическая проверка потребления ресурсов (Memory RSS)
        if self.last_health_check.elapsed().as_secs() >= self.config.health_check_interval_s {
            self.sys.refresh_processes_specifics(
                ProcessesToUpdate::Some(&[self.pid]),
                true,
                ProcessRefreshKind::nothing().with_memory(),
            );

            if let Some(process) = self.sys.process(self.pid) {
                let rss_mb = process.memory() / 1024 / 1024;
                if rss_mb > self.config.max_process_memory_mb {
                    error!("System Health: Memory usage too high: {}MB (limit: {}MB)", 
                           rss_mb, self.config.max_process_memory_mb);
                    metrics::gauge!("bot_health_status").set(0.0); // Критическая ошибка
                    anyhow::bail!("Memory usage too high: {}MB", rss_mb);
                }
            }
            self.last_health_check = Instant::now();
        }

        // Все проверки пройдены - статус OK (задача 189)
        metrics::gauge!("bot_health_status").set(1.0);

        Ok(())
    }

    /// Проверка здоровья системы с очисткой интентов (Задача 176)
    /// Вызывается из RiskManager::check_risk_gates
    pub fn is_sane_with_intent_cleanup(
        &mut self,
        active_intents: &mut std::collections::HashMap<String, crate::risk::risk_manager::OrderIntent>,
    ) -> anyhow::Result<()> {
        // Сначала проверяем здоровье
        self.is_sane()?;
        
        // Затем очищаем устаревшие интенты (периодически)
        if self.last_health_check.elapsed().as_secs() >= self.config.health_check_interval_s {
            self.cleanup_stale_intents(active_intents);
        }
        
        Ok(())
    }

    /// Периодическая очистка устаревших интентов (Задача 176)
    /// Удаляет интенты, чей возраст превышает order_intent_timeout_ms
    /// Защита от утечек памяти при потере WebSocket сообщений
    pub fn cleanup_stale_intents(
        &self,
        active_intents: &mut std::collections::HashMap<String, crate::risk::risk_manager::OrderIntent>,
    ) {
        let now = chrono::Utc::now().timestamp_millis() as u64;
        let timeout_ms = self.config.order_intent_timeout_ms;
        
        let initial_count = active_intents.len();
        active_intents.retain(|link_id, intent| {
            let age_ms = now.saturating_sub(intent.timestamp);
            if age_ms > timeout_ms {
                tracing::warn!(
                    "Cleaning up stale order intent: {} (age: {}ms > {}ms)",
                    link_id, age_ms, timeout_ms
                );
                false
            } else {
                true
            }
        });
        
        let removed_count = initial_count - active_intents.len();
        if removed_count > 0 {
            tracing::info!("Cleaned up {} stale order intents", removed_count);
        }
    }

    /// Обновление истории изменений цены (Задача 178)
    /// Вызывается при каждом обновлении mid_price из OrderBook
    pub fn update_price(&mut self, mid_price: f64) {
        if let Some(last_price) = self.last_mid_price {
            // Вычисляем абсолютное изменение цены
            let price_change = (mid_price - last_price).abs();
            
            // Добавляем в историю изменений
            if self.price_changes.len() >= 100 {
                self.price_changes.pop_front();
            }
            self.price_changes.push_back(price_change);
        }
        
        self.last_mid_price = Some(mid_price);
    }

    /// Расчет текущей волатильности (Задача 178)
    /// Возвращает стандартное отклонение изменений цены за последние ~15 минут
    /// Если данных недостаточно, возвращает 0.0
    pub fn get_current_volatility(&self) -> f64 {
        if self.price_changes.len() < 10 {
            return 0.0;
        }

        // Берем последние 100 изменений (примерно 15 минут при частоте обновления ~9 сек)
        let changes: Vec<f64> = self.price_changes.iter().copied().collect();
        
        // Вычисляем среднее
        let mean = changes.iter().sum::<f64>() / changes.len() as f64;
        
        // Вычисляем стандартное отклонение
        let variance = changes.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / changes.len() as f64;
        
        variance.sqrt()
    }

    /// Расчет исторической (медианной) волатильности за 24 часа (Задача 178)
    /// Возвращает медианное значение волатильности
    /// Если данных недостаточно, возвращает текущую волатильность
    pub fn get_historical_volatility(&mut self) -> f64 {
        let current_vol = self.get_current_volatility();
        
        // Периодически сохраняем текущую волатильность в историю
        // (можно вызывать это каждую минуту или при каждом обновлении)
        if self.volatility_history.len() >= 1440 {
            self.volatility_history.pop_front();
        }
        self.volatility_history.push_back(current_vol);
        
        if self.volatility_history.len() < 10 {
            return current_vol;
        }

        // Вычисляем медиану
        let mut sorted: Vec<f64> = self.volatility_history.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }
}


impl HealthMonitor {
    /// Сжатие старых логов с использованием Zstd (Задача 182)
    /// Находит файлы без расширения .zst и сжимает их
    /// Использует tokio::spawn_blocking для предотвращения блокировки асинхронного рантайма
    async fn compress_old_logs(&self) -> anyhow::Result<()> {
        let log_dir = match &self.log_dir {
            Some(dir) => dir.clone(),
            None => return Ok(()),
        };

        // Используем spawn_blocking для синхронной операции сжатия
        tokio::task::spawn_blocking(move || {
            if !log_dir.exists() {
                return Ok(());
            }

            // Читаем директорию логов
            let entries = std::fs::read_dir(&log_dir)?;
            let mut files_to_compress: Vec<PathBuf> = Vec::new();

            for entry in entries {
                let entry = entry?;
                let path = entry.path();
                
                // Пропускаем директории и файлы с расширением .zst
                if path.is_file() {
                    if let Some(ext) = path.extension() {
                        if ext != "zst" {
                            files_to_compress.push(path);
                        }
                    }
                }
            }

            // Сжимаем каждый файл
            for file_path in files_to_compress {
                // Пропускаем активный лог и уже сжатые файлы
                if let Some(file_name) = file_path.file_name().and_then(|n| n.to_str()) {
                    // Не трогаем текущий лог (без даты в названии)
                    if file_name == "bot.log" || file_name.ends_with(".zst") {
                        continue;
                    }
                }

                // Читаем исходный файл
                match std::fs::read(&file_path) {
                    Ok(data) => {
                        // Сжимаем с уровнем 1 (минимальная нагрузка на CPU)
                        match zstd::encode_all(&data[..], 1) {
                            Ok(compressed) => {
                                // Создаем путь для сжатого файла
                                let mut compressed_path = file_path.clone();
                                compressed_path.set_extension("zst");

                                // Записываем сжатый файл
                                match std::fs::write(&compressed_path, &compressed) {
                                    Ok(_) => {
                                        // Удаляем оригинальный файл
                                        if let Err(e) = std::fs::remove_file(&file_path) {
                                            tracing::warn!(
                                                "Failed to remove original log file {:?}: {}",
                                                file_path, e
                                            );
                                        } else {
                                            let original_size = data.len();
                                            let compressed_size = compressed.len();
                                            let ratio = if original_size > 0 {
                                                (compressed_size as f64 / original_size as f64) * 100.0
                                            } else {
                                                0.0
                                            };
                                            tracing::info!(
                                                "Compressed log file: {:?} ({} -> {} bytes, {:.1}%)",
                                                file_path, original_size, compressed_size, ratio
                                            );
                                        }
                                    }
                                    Err(e) => {
                                        tracing::error!(
                                            "Failed to write compressed log file {:?}: {}",
                                            compressed_path, e
                                        );
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::error!("Failed to compress log file {:?}: {}", file_path, e);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to read log file {:?}: {}", file_path, e);
                    }
                }
            }

            Ok::<(), anyhow::Error>(())
        })
        .await??;

        Ok(())
    }

    /// Удаление старых архивов логов (Задача 182)
    /// Удаляет .zst файлы, дата создания которых старше log_retention_days
    async fn cleanup_old_archives(&self, retention_days: u64) -> anyhow::Result<()> {
        let log_dir = match &self.log_dir {
            Some(dir) => dir.clone(),
            None => return Ok(()),
        };

        // Используем spawn_blocking для синхронной операции удаления
        tokio::task::spawn_blocking(move || {
            if !log_dir.exists() {
                return Ok(());
            }

            let now = std::time::SystemTime::now();
            let retention_duration = std::time::Duration::from_secs(retention_days * 86400);

            // Читаем директорию логов
            let entries = std::fs::read_dir(&log_dir)?;

            for entry in entries {
                let entry = entry?;
                let path = entry.path();

                // Обрабатываем только .zst файлы
                if path.is_file() {
                    if let Some(ext) = path.extension() {
                        if ext == "zst" {
                            // Получаем время модификации файла
                            if let Ok(metadata) = std::fs::metadata(&path) {
                                if let Ok(modified) = metadata.modified() {
                                    // Вычисляем возраст файла
                                    if let Ok(age) = now.duration_since(modified) {
                                        if age > retention_duration {
                                            // Удаляем старый архив
                                            match std::fs::remove_file(&path) {
                                                Ok(_) => {
                                                    let size_mb = metadata.len() / 1024 / 1024;
                                                    tracing::info!(
                                                        "Removed old log archive: {:?} (age: {:?}, size: {}MB)",
                                                        path, age, size_mb
                                                    );
                                                }
                                                Err(e) => {
                                                    tracing::warn!(
                                                        "Failed to remove old log archive {:?}: {}",
                                                        path, e
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            Ok::<(), anyhow::Error>(())
        })
        .await??;

        Ok(())
    }

    /// Периодическая задача архивации логов (Задача 182)
    /// Вызывается раз в час для сжатия и очистки логов
    pub async fn run_log_archival_task(&mut self, retention_days: u64) -> anyhow::Result<()> {
        // Проверяем, прошел ли час с последней проверки
        if self.last_archive_check.elapsed().as_secs() < 3600 {
            return Ok(());
        }

        // Сжимаем старые логи
        if let Err(e) = self.compress_old_logs().await {
            tracing::error!("Log compression failed: {}", e);
        }

        // Удаляем старые архивы
        if let Err(e) = self.cleanup_old_archives(retention_days).await {
            tracing::error!("Log cleanup failed: {}", e);
        }

        self.last_archive_check = Instant::now();
        Ok(())
    }
}


impl HealthMonitor {
    /// Проверка и обработка "зависших" (stale) ордеров (Задача 179)
    /// Вызывается периодически из run-bot.rs
    /// 
    /// Логика:
    /// 1. Итерирует по активным интентам
    /// 2. Проверяет, превышено ли время жизни ордера (max_order_life_ms)
    /// 3. Если заполнено менее min_fill_pct_to_keep, выполняет действие:
    ///    - CancelOnly: отменить ордер
    ///    - CancelAndMarketFill: отменить лимит и исполнить по рынку
    ///    - Repeg: переставить ордер к best_bid/ask
    /// 4. Удаляет интент после завершения действия
    pub async fn check_stale_orders(
        &self,
        rest_client: &impl crate::trading::rest_client::BybitRestClientTrait,
        bot_config: &crate::config::types::BotConfig,
        exchange_config: &crate::config::types::ExchangeConfig,
        active_intents: &mut std::collections::HashMap<String, crate::risk::risk_manager::OrderIntent>,
        order_manager: &mut crate::trading::order_manager::OrderManager,
        risk_manager: &mut crate::risk::risk_manager::RiskManager,
    ) -> anyhow::Result<()> {
        use crate::config::types::StaleOrderAction;
        
        let now = chrono::Utc::now().timestamp_millis() as u64;
        let max_order_life_ms = self.config.max_order_life_ms;
        let min_fill_pct_to_keep = self.config.min_fill_pct_to_keep;
        let stale_order_action = self.config.stale_order_action;
        
        // Собираем список "зависших" ордеров
        let mut stale_orders: Vec<String> = Vec::new();
        
        for (link_id, intent) in active_intents.iter() {
            let age_ms = now.saturating_sub(intent.timestamp);
            
            // Проверяем, превышено ли время жизни
            if age_ms > max_order_life_ms {
                // Проверяем процент исполнения
                let fill_pct = if intent.qty > 0.0 {
                    intent.filled_qty / intent.qty
                } else {
                    0.0
                };
                
                // Если заполнено менее порога, добавляем в список для обработки
                if fill_pct < min_fill_pct_to_keep {
                    stale_orders.push(link_id.clone());
                    tracing::warn!(
                        "Stale order detected: {} (age: {}ms, fill_pct: {:.2}%)",
                        link_id, age_ms, fill_pct * 100.0
                    );
                }
            }
        }
        
        // Обрабатываем каждый "зависший" ордер
        for link_id in stale_orders {
            match stale_order_action {
                StaleOrderAction::CancelOnly => {
                    tracing::info!("Cancelling stale order: {} (action: CancelOnly)", link_id);
                    // Отменяем ордер с force=true (ордер может быть уже исполнен на бирже)
                    if let Err(e) = order_manager.cancel_order(
                        rest_client,
                        risk_manager,
                        bot_config,
                        exchange_config,
                        &link_id,
                        true,
                    ).await {
                        tracing::error!("Failed to cancel stale order {}: {}", link_id, e);
                    }
                    // Удаляем интент
                    risk_manager.remove_order_intent(&link_id);
                    active_intents.remove(&link_id);
                }
                
                StaleOrderAction::CancelAndMarketFill => {
                    tracing::info!("Market filling stale order: {} (action: CancelAndMarketFill)", link_id);
                    // TODO: Реализовать логику CancelAndMarketFill
                    // Требует доступа к текущему сигналу (задача 169) и best_bid/ask
                    // Пока просто отменяем
                    if let Err(e) = order_manager.cancel_order(
                        rest_client,
                        risk_manager,
                        bot_config,
                        exchange_config,
                        &link_id,
                        true,
                    ).await {
                        tracing::error!("Failed to cancel stale order {}: {}", link_id, e);
                    }
                    risk_manager.remove_order_intent(&link_id);
                    active_intents.remove(&link_id);
                }
                
                StaleOrderAction::Repeg => {
                    tracing::info!("Repegging stale order: {} (action: Repeg)", link_id);
                    // TODO: Реализовать логику Repeg
                    // Требует доступа к best_bid/ask (задача 108) и сохранения оригинального created_at
                    // Пока просто отменяем
                    if let Err(e) = order_manager.cancel_order(
                        rest_client,
                        risk_manager,
                        bot_config,
                        exchange_config,
                        &link_id,
                        true,
                    ).await {
                        tracing::error!("Failed to cancel stale order {}: {}", link_id, e);
                    }
                    risk_manager.remove_order_intent(&link_id);
                    active_intents.remove(&link_id);
                }
            }
        }
        
        Ok(())
    }
}


    /// Задача 184: Обновление конфигурации при SIGHUP
    /// Применяет новые параметры риска к health monitor
    pub fn update_config(&mut self, config: crate::config::types::RiskConfig) {
        tracing::info!("[Audit] Updating HealthMonitor config");
        self.config = config;
    }

    /// Задача 187: Автоматическая очистка данных
    /// Удаляет устаревшие файлы из директории data/raw
    /// Возвращает (освобожденные байты, количество удаленных файлов)
    pub fn perform_data_cleanup(
        data_dir: &std::path::Path,
        config: &RiskConfig,
    ) -> anyhow::Result<(u64, usize)> {
        use std::fs;
        use std::time::{SystemTime, Duration};

        if !data_dir.exists() {
            tracing::warn!("[Health] Data directory does not exist: {:?}", data_dir);
            return Ok((0, 0));
        }

        // Структура для хранения информации о файле
        struct FileInfo {
            path: std::path::PathBuf,
            size: u64,
            modified: SystemTime,
        }

        // Сканируем директорию рекурсивно
        let mut files = Vec::new();
        let mut total_size: u64 = 0;

        fn scan_dir(dir: &std::path::Path, files: &mut Vec<FileInfo>, total_size: &mut u64) -> anyhow::Result<()> {
            if !dir.is_dir() {
                return Ok(());
            }

            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                
                if path.is_dir() {
                    scan_dir(&path, files, total_size)?;
                } else if path.is_file() {
                    if let Ok(metadata) = fs::metadata(&path) {
                        let size = metadata.len();
                        if let Ok(modified) = metadata.modified() {
                            files.push(FileInfo {
                                path: path.clone(),
                                size,
                                modified,
                            });
                            *total_size += size;
                        }
                    }
                }
            }
            Ok(())
        }

        if let Err(e) = scan_dir(data_dir, &mut files, &mut total_size) {
            tracing::warn!("[Health] Failed to scan data directory: {}", e);
            return Err(e);
        }

        let now = SystemTime::now();
        let retention_duration = Duration::from_secs(config.raw_data_retention_days as u64 * 24 * 3600);
        let buffer_duration = Duration::from_secs(24 * 3600); // 24 часа защитный буфер
        let max_size_bytes = config.max_data_dir_size_gb * 1024 * 1024 * 1024;

        let mut freed_bytes: u64 = 0;
        let mut deleted_count: usize = 0;

        // Фаза 1: Очистка по времени (retention)
        // Удаляем файлы старше retention_days, но не младше 24 часов
        for file in &files {
            if let Ok(age) = now.duration_since(file.modified) {
                // Защитный буфер: не удаляем файлы младше 24 часов
                if age < buffer_duration {
                    continue;
                }

                // Удаляем файлы старше retention_days
                if age > retention_duration {
                    match fs::remove_file(&file.path) {
                        Ok(_) => {
                            freed_bytes += file.size;
                            deleted_count += 1;
                            tracing::debug!("[Health] Deleted old file: {:?} (age: {:?})", file.path, age);
                        }
                        Err(e) => {
                            tracing::warn!("[Health] Failed to delete file {:?}: {}", file.path, e);
                        }
                    }
                }
            }
        }

        // Пересканируем после удаления по времени
        files.clear();
        total_size = 0;
        if let Err(e) = scan_dir(data_dir, &mut files, &mut total_size) {
            tracing::warn!("[Health] Failed to rescan data directory: {}", e);
            return Ok((freed_bytes, deleted_count));
        }

        // Фаза 2: Очистка по квоте (size)
        if total_size > max_size_bytes {
            // Сортируем файлы по времени модификации (от старых к новым)
            files.sort_by_key(|f| f.modified);

            // Удаляем файлы по FIFO, пока не достигнем лимита
            for file in &files {
                if total_size <= max_size_bytes {
                    break;
                }

                // Защитный буфер: не удаляем файлы младше 24 часов
                if let Ok(age) = now.duration_since(file.modified) {
                    if age < buffer_duration {
                        continue;
                    }
                }

                match fs::remove_file(&file.path) {
                    Ok(_) => {
                        freed_bytes += file.size;
                        deleted_count += 1;
                        total_size -= file.size;
                        tracing::debug!("[Health] Deleted file to free space: {:?}", file.path);
                    }
                    Err(e) => {
                        tracing::warn!("[Health] Failed to delete file {:?}: {}", file.path, e);
                    }
                }
            }
        }

        Ok((freed_bytes, deleted_count))
    }