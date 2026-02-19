//! Задача 225: Живой профилировщик системных ресурсов
//!
//! Система непрерывного мониторинга технического состояния бота:
//! - CPU usage (с EMA сглаживанием)
//! - Memory RSS (с EMA сглаживанием)
//! - Disk I/O (read/write bytes)
//! - Network I/O (rx/tx bytes)
//! - Детекция утечек памяти (slope detection)

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;
use sysinfo::{Networks, Pid, ProcessRefreshKind, RefreshKind, System};
use tokio::sync::broadcast;
use tokio::time;
use tracing::{debug, error, info, warn};

use crate::config::types::ResourceThresholdsConfig;

/// Обновление системных метрик (отдельная структура, НЕ EquityUpdate)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetricsUpdate {
    /// Временная метка
    pub timestamp: DateTime<Utc>,
    /// Использование CPU (EMA сглаженное), %
    pub cpu_usage_pct: f32,
    /// Использование памяти RSS (EMA сглаженное), KB
    pub memory_rss_kb: u64,
    /// Прочитано с диска, байты (с момента запуска)
    pub disk_read_bytes: u64,
    /// Записано на диск, байты (с момента запуска)
    pub disk_write_bytes: u64,
    /// Получено по сети, байты (с момента запуска)
    pub network_rx_bytes: u64,
    /// Отправлено по сети, байты (с момента запуска)
    pub network_tx_bytes: u64,
    /// Обнаружена ли утечка памяти
    pub memory_leak_detected: bool,
    /// Достигнут ли мягкий лимит памяти (90% от max_memory_mb) - задача 230
    pub soft_limit_reached: bool,
}

/// Профилировщик системных ресурсов
pub struct ResourceProfiler {
    /// Конфигурация порогов
    config: ResourceThresholdsConfig,
    /// Система sysinfo
    system: System,
    /// Сетевые интерфейсы
    networks: Networks,
    /// PID текущего процесса
    current_pid: Pid,
    /// EMA для CPU usage
    cpu_ema: Option<f32>,
    /// EMA для Memory RSS
    memory_ema: Option<f64>,
    /// История EMA памяти для детекции утечек
    memory_history: VecDeque<f64>,
    /// Broadcast канал для трансляции метрик
    tx: broadcast::Sender<SystemMetricsUpdate>,
    /// Максимальный лимит памяти в KB (задача 230) - для проверки мягкого лимита
    pub max_memory_kb: u64,
}

impl ResourceProfiler {
    /// Создать новый профилировщик
    pub fn new(config: ResourceThresholdsConfig) -> Result<(Self, broadcast::Receiver<SystemMetricsUpdate>)> {
        // Создаем broadcast канал
        let (tx, rx) = broadcast::channel(100);

        // Получаем PID текущего процесса
        let current_pid = sysinfo::get_current_pid()
            .context("Failed to get current process PID")?;

        // Инициализируем System с минимальными обновлениями
        let system = System::new_with_specifics(
            RefreshKind::new()
                .with_processes(ProcessRefreshKind::new())
                .with_cpu()
                .with_memory(),
        );

        // Инициализируем Networks
        let networks = Networks::new_with_refreshed_list();

        let profiler = Self {
            config,
            system,
            networks,
            current_pid,
            cpu_ema: None,
            memory_ema: None,
            memory_history: VecDeque::with_capacity(config.leak_detection_window),
            tx,
            max_memory_kb: 0, // Будет установлено из конфига в run-bot.rs (задача 230)
        };

        info!(
            "ResourceProfiler initialized: cpu_max={}%, mem_growth_min={}KB, interval={}s",
            profiler.config.cpu_max_pct,
            profiler.config.mem_growth_kb_min,
            profiler.config.sample_interval_sec
        );

        Ok((profiler, rx))
    }

    /// Запустить профилировщик в фоновом режиме
    pub fn spawn(mut self) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let interval = Duration::from_secs(self.config.sample_interval_sec);
            let mut ticker = time::interval(interval);
            ticker.set_missed_tick_behavior(time::MissedTickBehavior::Skip);

            info!("ResourceProfiler started");

            loop {
                ticker.tick().await;

                if let Err(e) = self.collect_and_broadcast().await {
                    error!("Failed to collect metrics: {}", e);
                }
            }
        })
    }

    /// Собрать метрики и отправить в broadcast канал
    async fn collect_and_broadcast(&mut self) -> Result<()> {
        // Обновляем информацию о процессе
        self.system.refresh_process_specifics(
            self.current_pid,
            ProcessRefreshKind::new()
                .with_cpu()
                .with_memory()
                .with_disk_usage(),
        );

        // Обновляем сетевые интерфейсы
        self.networks.refresh();

        // Получаем процесс
        let process = self.system.process(self.current_pid)
            .context("Current process not found")?;

        // Собираем сырые метрики
        let cpu_raw = process.cpu_usage();
        let memory_raw = process.memory() / 1024; // bytes -> KB
        let disk_usage = process.disk_usage();
        let disk_read = disk_usage.total_read_bytes;
        let disk_write = disk_usage.total_written_bytes;

        // Собираем system-wide network stats (так как per-process недоступно в sysinfo)
        let (network_rx_bytes, network_tx_bytes) = self.get_network_stats();

        // Применяем EMA сглаживание
        let cpu_ema = self.apply_ema_f32(self.cpu_ema, cpu_raw);
        let memory_ema = self.apply_ema_f64(self.memory_ema, memory_raw as f64);

        self.cpu_ema = Some(cpu_ema);
        self.memory_ema = Some(memory_ema);

        // Добавляем в историю для детекции утечек
        self.memory_history.push_back(memory_ema);
        if self.memory_history.len() > self.config.leak_detection_window {
            self.memory_history.pop_front();
        }

        // Проверяем утечку памяти
        let memory_leak_detected = self.detect_memory_leak();

        // Создаем обновление
        let update = SystemMetricsUpdate {
            timestamp: Utc::now(),
            cpu_usage_pct: cpu_ema,
            memory_rss_kb: memory_ema as u64,
            disk_read_bytes: disk_read,
            disk_write_bytes: disk_write,
            network_rx_bytes,
            network_tx_bytes,
            memory_leak_detected,
        };

        // Логируем метрики
        debug!(
            "System metrics: CPU={:.1}%, MEM={}KB, DISK_R={}B, DISK_W={}B, NET_RX={}B, NET_TX={}B, LEAK={}",
            update.cpu_usage_pct,
            update.memory_rss_kb,
            update.disk_read_bytes,
            update.disk_write_bytes,
            update.network_rx_bytes,
            update.network_tx_bytes,
            update.memory_leak_detected
        );

        // Проверяем превышение порогов
        if cpu_ema > self.config.cpu_max_pct {
            warn!(
                "CPU usage exceeded threshold: {:.1}% > {}%",
                cpu_ema, self.config.cpu_max_pct
            );
        }

        if memory_leak_detected {
            warn!(
                "Memory leak detected: consistent growth over {} samples",
                self.config.leak_detection_window
            );
        }

        // Отправляем в broadcast канал
        if let Err(e) = self.tx.send(update) {
            debug!("No active receivers for system metrics: {}", e);
        }

        Ok(())
    }

    /// Применить EMA сглаживание для f32
    fn apply_ema_f32(&self, prev: Option<f32>, current: f32) -> f32 {
        match prev {
            Some(prev_val) => {
                let alpha = self.config.ema_alpha;
                alpha * current + (1.0 - alpha) * prev_val
            }
            None => current,
        }
    }

    /// Применить EMA сглаживание для f64
    fn apply_ema_f64(&self, prev: Option<f64>, current: f64) -> f64 {
        match prev {
            Some(prev_val) => {
                let alpha = self.config.ema_alpha as f64;
                alpha * current + (1.0 - alpha) * prev_val
            }
            None => current,
        }
    }

    /// Детектор утечек памяти (slope detection)
    /// Возвращает true, если память растет на протяжении N циклов подряд
    fn detect_memory_leak(&self) -> bool {
        if self.memory_history.len() < self.config.leak_detection_window {
            return false;
        }

        // Проверяем, что каждое следующее значение больше предыдущего
        let mut growing = true;
        let mut prev = self.memory_history[0];

        for &current in self.memory_history.iter().skip(1) {
            if current <= prev {
                growing = false;
                break;
            }
            prev = current;
        }

        if !growing {
            return false;
        }

        // Проверяем, что рост превышает минимальный порог
        let first = self.memory_history.front().unwrap();
        let last = self.memory_history.back().unwrap();
        let growth_kb = (last - first) as u64;

        growth_kb >= self.config.mem_growth_kb_min
    }

    /// Получить system-wide network stats
    /// Примечание: sysinfo не предоставляет per-process network stats,
    /// поэтому используем system-wide данные
    fn get_network_stats(&self) -> (u64, u64) {
        let mut total_rx = 0u64;
        let mut total_tx = 0u64;

        for (_interface_name, data) in &self.networks {
            total_rx = total_rx.saturating_add(data.total_received());
            total_tx = total_tx.saturating_add(data.total_transmitted());
        }

        (total_rx, total_tx)
    }

    /// Получить текущие метрики (синхронно)
    pub fn get_current_metrics(&mut self) -> Result<SystemMetricsUpdate> {
        // Обновляем информацию
        self.system.refresh_process_specifics(
            self.current_pid,
            ProcessRefreshKind::new()
                .with_cpu()
                .with_memory()
                .with_disk_usage(),
        );

        let process = self.system.process(self.current_pid)
            .context("Current process not found")?;

        let cpu_raw = process.cpu_usage();
        let memory_raw = process.memory() / 1024;
        let disk_usage = process.disk_usage();

        let cpu_ema = self.cpu_ema.unwrap_or(cpu_raw);
        let memory_ema = self.memory_ema.unwrap_or(memory_raw as f64);

        // Задача 230: Проверка мягкого лимита (90%)
        let soft_limit_reached = if self.max_memory_kb > 0 {
            memory_ema as u64 > (self.max_memory_kb * 9 / 10)
        } else {
            false
        };

        Ok(SystemMetricsUpdate {
            timestamp: Utc::now(),
            cpu_usage_pct: cpu_ema,
            memory_rss_kb: memory_ema as u64,
            disk_read_bytes: disk_usage.total_read_bytes,
            disk_write_bytes: disk_usage.total_written_bytes,
            network_rx_bytes: 0,
            network_tx_bytes: 0,
            memory_leak_detected: self.detect_memory_leak(),
            soft_limit_reached,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema_calculation() {
        let config = ResourceThresholdsConfig {
            cpu_max_pct: 80.0,
            mem_growth_kb_min: 10240,
            sample_interval_sec: 5,
            ema_alpha: 0.2,
            leak_detection_window: 10,
        };

        let (profiler, _rx) = ResourceProfiler::new(config).unwrap();

        // Первое значение
        let ema1 = profiler.apply_ema_f32(None, 50.0);
        assert_eq!(ema1, 50.0);

        // Второе значение
        let ema2 = profiler.apply_ema_f32(Some(50.0), 60.0);
        // 0.2 * 60.0 + 0.8 * 50.0 = 12.0 + 40.0 = 52.0
        assert_eq!(ema2, 52.0);
    }

    #[test]
    fn test_memory_leak_detection() {
        let config = ResourceThresholdsConfig {
            cpu_max_pct: 80.0,
            mem_growth_kb_min: 100,
            sample_interval_sec: 5,
            ema_alpha: 0.2,
            leak_detection_window: 5,
        };

        let (mut profiler, _rx) = ResourceProfiler::new(config).unwrap();

        // Заполняем историю растущими значениями
        for i in 0..5 {
            profiler.memory_history.push_back(1000.0 + (i as f64 * 50.0));
        }

        // Должна быть обнаружена утечка (рост 200 KB > 100 KB)
        assert!(profiler.detect_memory_leak());
    }

    #[test]
    fn test_no_memory_leak_stable() {
        let config = ResourceThresholdsConfig {
            cpu_max_pct: 80.0,
            mem_growth_kb_min: 100,
            sample_interval_sec: 5,
            ema_alpha: 0.2,
            leak_detection_window: 5,
        };

        let (mut profiler, _rx) = ResourceProfiler::new(config).unwrap();

        // Заполняем историю стабильными значениями
        for _ in 0..5 {
            profiler.memory_history.push_back(1000.0);
        }

        // Утечка не должна быть обнаружена
        assert!(!profiler.detect_memory_leak());
    }
}
