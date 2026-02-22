#[cfg(target_os = "linux")]
use nix::sched::{set_scheduler, SchedParameters, Scheduler};
use core_affinity::CoreId;

use std::sync::atomic::{AtomicU64, AtomicU32, Ordering};
use std::sync::OnceLock;
use parking_lot::Mutex;
use sysinfo::{System, Pid, get_current_pid, ProcessesToUpdate};
use tokio::sync::mpsc;
use std::time::Duration;

/// События системного мониторинга (задача 230)
#[derive(Debug, Clone)]
pub enum SystemEvent {
    /// Достигнут мягкий лимит памяти (90% от max_memory_mb)
    SoftLimitReached { current_mb: u64, limit_mb: u64 },
}

/// Метрики ресурсов процесса
pub struct ResourceMetrics {
    pub rss_bytes: AtomicU64,
    pub cpu_usage: AtomicU32, // Храним f32 как биты u32
}

pub static METRICS: ResourceMetrics = ResourceMetrics {
    rss_bytes: AtomicU64::new(0),
    cpu_usage: AtomicU32::new(0),
};

static SYSTEM: OnceLock<Mutex<System>> = OnceLock::new();
static PID: OnceLock<Pid> = OnceLock::new();

/// Настройка приоритета и привязки потока к ядру ЦП
pub fn set_hot_thread_config(core_id: Option<usize>) {
    // 1. Привязка к конкретному физическому ядру
    if let Some(id) = core_id {
        if core_affinity::set_for_current(CoreId { id }) {
            tracing::info!("Thread pinned to core {}", id);
        } else {
            tracing::warn!("Failed to pin thread to core {}", id);
        }
    }

    // 2. Установка Real-Time приоритета (Round Robin)
    #[cfg(target_os = "linux")]
    {
        let param = SchedParameters { sched_priority: 10 };
        match set_scheduler(nix::unistd::Pid::from_raw(0), Scheduler::SchedRr, &param) {
            Ok(_) => tracing::info!("Thread priority set to SCHED_RR"),
            Err(nix::errno::Errno::EPERM) => {
                tracing::warn!("No CAP_SYS_NICE, falling back to nice -10");
                unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, -10); }
            }
            Err(e) => tracing::error!("Failed to set thread priority: {}", e),
        }
    }
}

/// Обновление метрик ресурсов процесса
pub fn update_resource_metrics() {
    let mut sys = SYSTEM.get_or_init(|| Mutex::new(System::new_all())).lock();
    let pid = *PID.get_or_init(|| get_current_pid().expect("Failed to get current PID"));
    
    sys.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
    if let Some(process) = sys.process(pid) {
        METRICS.rss_bytes.store(process.memory(), Ordering::Relaxed);
        METRICS.cpu_usage.store(process.cpu_usage().to_bits(), Ordering::Relaxed);
    }
}

/// Привязка процесса к конкретному ядру CPU (задача 230)
/// 
/// Использует core_affinity для установки CPU affinity текущего процесса.
/// Это обеспечивает предсказуемую производительность и снижает cache misses.
pub fn set_process_affinity(core_id: usize) {
    if core_affinity::set_for_current(CoreId { id: core_id }) {
        tracing::info!("Process pinned to CPU core {}", core_id);
    } else {
        tracing::warn!("Failed to pin process to CPU core {}", core_id);
    }
}

/// Мониторинг использования памяти процесса (задача 230)
/// 
/// Периодически проверяет RSS (Resident Set Size) текущего процесса.
/// При превышении 90% от max_mem_mb отправляет событие SoftLimitReached
/// для перехода в режим Graceful Degradation (задача 220).
/// 
/// # Аргументы
/// * `max_mem_mb` - Максимальный лимит памяти в мегабайтах
/// * `tx` - Канал для отправки событий SystemEvent
pub async fn monitor_resources(max_mem_mb: u64, tx: mpsc::Sender<SystemEvent>) {
    let mut sys = System::new_all();
    let pid = get_current_pid().expect("Failed to get current PID");
    let soft_limit_bytes = (max_mem_mb * 1024 * 1024 * 90) / 100; // 90% порог
    
    tracing::info!(
        "Starting resource monitor: max_memory={}MB, soft_limit={}MB",
        max_mem_mb,
        soft_limit_bytes / (1024 * 1024)
    );
    
    loop {
        tokio::time::sleep(Duration::from_secs(5)).await;
        
        sys.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
        if let Some(process) = sys.process(pid) {
            let rss_bytes = process.memory();
            let rss_mb = rss_bytes / (1024 * 1024);
            
            // Обновляем глобальные метрики
            METRICS.rss_bytes.store(rss_bytes, Ordering::Relaxed);
            METRICS.cpu_usage.store(process.cpu_usage().to_bits(), Ordering::Relaxed);
            
            // Проверка мягкого лимита
            if rss_bytes > soft_limit_bytes {
                tracing::warn!(
                    "Memory soft limit reached: current={}MB, limit={}MB",
                    rss_mb,
                    max_mem_mb
                );
                
                let event = SystemEvent::SoftLimitReached {
                    current_mb: rss_mb,
                    limit_mb: max_mem_mb,
                };
                
                if tx.send(event).await.is_err() {
                    tracing::error!("Failed to send SoftLimitReached event: receiver dropped");
                    break;
                }
            }
        }
    }
}
