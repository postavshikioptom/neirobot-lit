//! Hot-Swap Engine для безопасной замены ONNX моделей без остановки бота
//! 
//! Задача 228: Автоматизированная дистрибуция и безопасный Hot-Swap моделей
//! 
//! Основные возможности:
//! - File watcher для model.hash (избегание race condition)
//! - SIGUSR1 signal handler для принудительной перезагрузки
//! - ArcSwap для атомарной замены сессии
//! - Warmup цикл (100 итераций на dummy данных)
//! - Auto Rollback при ошибках или деградации латентности
//! - Интеграция с AlertManager

use anyhow::{Context, Result, bail};
use arc_swap::ArcSwap;
use notify::{Watcher, RecursiveMode, Event, EventKind};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{info, warn, error, debug};

use crate::ml::onnx::{OnnxEngine, InferenceResult};
use crate::ml::types::InferenceOutput;
use crate::config::types::{OnnxConfig, BotConfig};
use crate::monitoring::alert_manager::{AlertManager, Alert, AlertLevel};

/// Результат warmup проверки
#[derive(Debug)]
struct WarmupResult {
    /// Средняя латентность в микросекундах
    avg_latency_us: u64,
    /// Максимальная латентность в микросекундах
    max_latency_us: u64,
    /// Успешность warmup
    success: bool,
    /// Сообщение об ошибке (если есть)
    error_message: Option<String>,
}

/// Hot-Swap Engine с автоматической перезагрузкой моделей
pub struct HotSwapEngine {
    /// Текущая ONNX сессия (атомарно заменяемая)
    engine: Arc<ArcSwap<OnnxEngine>>,
    /// Путь к директории модели
    model_dir: PathBuf,
    /// Путь к файлу model.onnx
    model_path: PathBuf,
    /// Путь к файлу model.hash
    hash_file: PathBuf,
    /// Путь к backup модели
    backup_path: PathBuf,
    /// Конфигурация ONNX
    onnx_config: OnnxConfig,
    /// Конфигурация бота
    bot_config: BotConfig,
    /// Символ (для логирования и алертов)
    symbol: String,
    /// Seq length модели
    seq_len: usize,
    /// Input features модели
    input_features: usize,
    /// Alert Manager (опционально)
    alert_manager: Option<Arc<AlertManager>>,
    /// Baseline латентность (для проверки деградации)
    baseline_latency_us: Arc<parking_lot::RwLock<Option<u64>>>,
}

impl HotSwapEngine {
    /// Создает новый HotSwapEngine и загружает модель
    pub fn new(
        model_path: PathBuf,
        seq_len: usize,
        input_features: usize,
        onnx_config: OnnxConfig,
        bot_config: BotConfig,
        symbol: String,
        alert_manager: Option<Arc<AlertManager>>,
    ) -> Result<Arc<Self>> {
        info!("[HotSwap] Initializing HotSwapEngine for {}", symbol);
        
        // Проверяем, включен ли hot-swap в конфигурации
        if !bot_config.enable_model_hotswap {
            info!("[HotSwap] Model hot-swap is disabled in config");
        }
        
        let model_dir = model_path.parent()
            .context("Failed to get model directory")?
            .to_path_buf();
        
        let hash_file = model_dir.join("model.hash");
        let backup_path = model_dir.join("model.onnx.bak");
        
        // Загружаем начальную модель
        info!("[HotSwap] Loading initial model from {}", model_path.display());
        let engine = OnnxEngine::load(
            &model_path,
            seq_len,
            input_features,
            &onnx_config,
            &symbol,
            Some(&bot_config),
        )?;
        
        // Выполняем warmup и замеряем baseline латентность
        info!("[HotSwap] Performing initial warmup...");
        let warmup_result = Self::warmup_engine(&engine, seq_len, input_features)?;
        
        if !warmup_result.success {
            bail!("Initial warmup failed: {}", warmup_result.error_message.unwrap_or_default());
        }
        
        let baseline_latency = warmup_result.avg_latency_us;
        info!("[HotSwap] Baseline latency: {}μs (max: {}μs)", 
              baseline_latency, warmup_result.max_latency_us);
        
        let hotswap_engine = Arc::new(Self {
            engine: Arc::new(ArcSwap::from_pointee(engine)),
            model_dir,
            model_path,
            hash_file,
            backup_path,
            onnx_config,
            bot_config,
            symbol,
            seq_len,
            input_features,
            alert_manager,
            baseline_latency_us: Arc::new(parking_lot::RwLock::new(Some(baseline_latency))),
        });
        
        // Запускаем file watcher если hot-swap включен
        if hotswap_engine.bot_config.enable_model_hotswap {
            hotswap_engine.clone().start_file_watcher()?;
        }
        
        // Запускаем signal handler для SIGUSR1 (только на Unix)
        #[cfg(unix)]
        if hotswap_engine.bot_config.enable_model_hotswap {
            hotswap_engine.clone().start_signal_handler()?;
        }
        
        Ok(hotswap_engine)
    }
    
    /// Выполняет warmup модели и возвращает статистику латентности
    fn warmup_engine(
        engine: &OnnxEngine,
        seq_len: usize,
        input_features: usize,
    ) -> Result<WarmupResult> {
        let dummy_data = vec![0.0f32; seq_len * input_features];
        let regime_id = if engine.use_regime_embedding { Some(0) } else { None };
        
        let mut latencies = Vec::with_capacity(100);
        let mut error_message = None;
        
        for i in 0..100 {
            let start = Instant::now();
            
            match engine.predict(&dummy_data, regime_id) {
                Ok(result) => {
                    let latency_us = result.duration_us;
                    latencies.push(latency_us);
                }
                Err(e) => {
                    error_message = Some(format!("Warmup iteration {} failed: {}", i, e));
                    return Ok(WarmupResult {
                        avg_latency_us: 0,
                        max_latency_us: 0,
                        success: false,
                        error_message,
                    });
                }
            }
        }
        
        let avg_latency = latencies.iter().sum::<u64>() / latencies.len() as u64;
        let max_latency = *latencies.iter().max().unwrap_or(&0);
        
        Ok(WarmupResult {
            avg_latency_us: avg_latency,
            max_latency_us: max_latency,
            success: true,
            error_message: None,
        })
    }
    
    /// Перезагружает модель с проверкой и rollback
    fn reload_model(self: &Arc<Self>) -> Result<()> {
        info!("[HotSwap] Starting model reload for {}", self.symbol);
        
        // 1. Загружаем новую модель
        let new_engine = match OnnxEngine::load(
            &self.model_path,
            self.seq_len,
            self.input_features,
            &self.onnx_config,
            &self.symbol,
            Some(&self.bot_config),
        ) {
            Ok(engine) => engine,
            Err(e) => {
                error!("[HotSwap] Failed to load new model: {}", e);
                self.send_alert(
                    AlertLevel::Critical,
                    format!("Model reload failed: {}", e),
                );
                return Err(e);
            }
        };
        
        // 2. Warmup новой модели
        info!("[HotSwap] Warming up new model...");
        let warmup_result = Self::warmup_engine(&new_engine, self.seq_len, self.input_features)?;
        
        if !warmup_result.success {
            let error_msg = warmup_result.error_message.unwrap_or_else(|| "Unknown error".to_string());
            error!("[HotSwap] Warmup failed: {}", error_msg);
            self.send_alert(
                AlertLevel::Critical,
                format!("Model warmup failed: {}", error_msg),
            );
            
            // Пытаемся загрузить backup
            self.rollback_to_backup()?;
            bail!("Warmup failed: {}", error_msg);
        }
        
        // 3. Проверяем деградацию латентности
        let baseline = self.baseline_latency_us.read().unwrap_or(None);
        if let Some(baseline_latency) = baseline {
            let latency_increase = warmup_result.avg_latency_us as f64 / baseline_latency as f64;
            
            if latency_increase > 1.5 {
                warn!(
                    "[HotSwap] Latency degradation detected: {}μs -> {}μs ({:.1}% increase)",
                    baseline_latency,
                    warmup_result.avg_latency_us,
                    (latency_increase - 1.0) * 100.0
                );
                
                self.send_alert(
                    AlertLevel::Critical,
                    format!(
                        "Model latency degradation: {}μs -> {}μs ({:.1}% increase). Rolling back.",
                        baseline_latency,
                        warmup_result.avg_latency_us,
                        (latency_increase - 1.0) * 100.0
                    ),
                );
                
                // Rollback
                self.rollback_to_backup()?;
                bail!("Latency degradation detected, rolled back to backup");
            }
        }
        
        // 4. Атомарная замена сессии
        info!("[HotSwap] Swapping to new model...");
        let old_engine = self.engine.swap(Arc::new(new_engine));
        
        // 5. Обновляем baseline латентность
        *self.baseline_latency_us.write() = Some(warmup_result.avg_latency_us);
        
        // 6. Явно освобождаем старую сессию
        // ArcSwap гарантирует, что старая сессия будет удалена когда все ссылки освободятся
        drop(old_engine);
        
        // 7. Обновляем Prometheus метрику
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        metrics::gauge!("model_version_timestamp", "symbol" => self.symbol.clone())
            .set(timestamp as f64);
        
        info!(
            "[HotSwap] ✓ Model reload successful. New latency: {}μs (max: {}μs)",
            warmup_result.avg_latency_us,
            warmup_result.max_latency_us
        );
        
        self.send_alert(
            AlertLevel::Info,
            format!(
                "Model reloaded successfully. Latency: {}μs",
                warmup_result.avg_latency_us
            ),
        );
        
        Ok(())
    }
    
    /// Откатывается на backup модель
    fn rollback_to_backup(&self) -> Result<()> {
        warn!("[HotSwap] Rolling back to backup model...");
        
        if !self.backup_path.exists() {
            error!("[HotSwap] Backup model not found: {}", self.backup_path.display());
            return Err(anyhow::anyhow!("Backup model not found"));
        }
        
        // Загружаем backup модель
        let backup_engine = OnnxEngine::load(
            &self.backup_path,
            self.seq_len,
            self.input_features,
            &self.onnx_config,
            &self.symbol,
            Some(&self.bot_config),
        )?;
        
        // Warmup backup модели
        let warmup_result = Self::warmup_engine(&backup_engine, self.seq_len, self.input_features)?;
        
        if !warmup_result.success {
            error!("[HotSwap] Backup model warmup failed!");
            return Err(anyhow::anyhow!("Backup model warmup failed"));
        }
        
        // Атомарная замена
        self.engine.store(Arc::new(backup_engine));
        
        info!("[HotSwap] ✓ Rolled back to backup model successfully");
        
        self.send_alert(
            AlertLevel::Warning,
            "Rolled back to backup model due to errors".to_string(),
        );
        
        Ok(())
    }
    
    /// Запускает file watcher для model.hash
    fn start_file_watcher(self: Arc<Self>) -> Result<()> {
        info!("[HotSwap] Starting file watcher for {}", self.hash_file.display());
        
        let (tx, mut rx) = mpsc::channel(10);
        
        // Создаем watcher
        let mut watcher = notify::recommended_watcher(move |res: Result<Event, notify::Error>| {
            match res {
                Ok(event) => {
                    // Отправляем событие в канал
                    let _ = tx.blocking_send(event);
                }
                Err(e) => {
                    error!("[HotSwap] File watcher error: {}", e);
                }
            }
        })?;
        
        // Наблюдаем за файлом model.hash
        watcher.watch(&self.hash_file, RecursiveMode::NonRecursive)?;
        
        // Запускаем обработчик событий в отдельной задаче
        let engine = self.clone();
        tokio::spawn(async move {
            // Держим watcher живым
            let _watcher = watcher;
            
            while let Some(event) = rx.recv().await {
                // Фильтруем только события изменения файла
                match event.kind {
                    EventKind::Modify(_) | EventKind::Create(_) => {
                        info!("[HotSwap] Detected model.hash change, triggering reload...");
                        
                        // Небольшая задержка для гарантии завершения записи
                        tokio::time::sleep(Duration::from_millis(100)).await;
                        
                        if let Err(e) = engine.reload_model() {
                            error!("[HotSwap] Model reload failed: {}", e);
                        }
                    }
                    _ => {
                        debug!("[HotSwap] Ignoring file event: {:?}", event.kind);
                    }
                }
            }
        });
        
        info!("[HotSwap] File watcher started successfully");
        Ok(())
    }
    
    /// Запускает signal handler для SIGUSR1 (только Unix)
    #[cfg(unix)]
    fn start_signal_handler(self: Arc<Self>) -> Result<()> {
        use tokio::signal::unix::{signal, SignalKind};
        
        info!("[HotSwap] Starting SIGUSR1 signal handler");
        
        let engine = self.clone();
        tokio::spawn(async move {
            let mut sigusr1 = signal(SignalKind::user_defined1())
                .expect("Failed to create SIGUSR1 handler");
            
            loop {
                sigusr1.recv().await;
                info!("[HotSwap] Received SIGUSR1 signal, triggering model reload...");
                
                if let Err(e) = engine.reload_model() {
                    error!("[HotSwap] Model reload failed: {}", e);
                }
            }
        });
        
        info!("[HotSwap] SIGUSR1 signal handler started");
        Ok(())
    }
    
    /// Отправляет алерт через AlertManager
    fn send_alert(&self, level: AlertLevel, message: String) {
        if let Some(ref alert_manager) = self.alert_manager {
            let alert = Alert::new(
                level,
                message,
                format!("HotSwap:{}", self.symbol),
            );
            alert_manager.send_alert(alert);
        }
    }
    
    /// Выполняет инференс с текущей моделью
    pub fn predict(&self, input_data: &[f32], regime_id: Option<usize>) -> Result<InferenceResult> {
        let engine = self.engine.load();
        engine.predict(input_data, regime_id)
    }
    
    /// Принудительная перезагрузка модели (для тестирования)
    pub fn force_reload(&self) -> Result<()> {
        info!("[HotSwap] Force reload triggered");
        self.reload_model()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_warmup_result() {
        let result = WarmupResult {
            avg_latency_us: 1000,
            max_latency_us: 2000,
            success: true,
            error_message: None,
        };
        
        assert!(result.success);
        assert_eq!(result.avg_latency_us, 1000);
        assert_eq!(result.max_latency_us, 2000);
    }
}
