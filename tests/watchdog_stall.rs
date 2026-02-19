use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;
use tokio::time::Instant;
use neirobot_lit::monitoring::health::{SharedState, HealthConfig};
use neirobot_lit::utils::helpers::unix_ms;

#[test]
fn test_watchdog_stall_trigger() {
    let mut config = HealthConfig::default();
    // Настраиваем очень короткие таймауты для быстрого теста
    config.watchdog.stall_timeout_ms = 50;
    config.watchdog.check_interval_ms = 20;
    config.watchdog.suspend_grace_ms = 1000;

    let state = Arc::new(SharedState {
        last_update: AtomicU64::new(0),
        last_heartbeat: AtomicU64::new(unix_ms()),
        ws_connected: AtomicBool::new(true),
        emergency_mode: AtomicBool::new(false),
        start_time: Instant::now(),
        config: config.clone(),
    });

    let state_clone = state.clone();
    
    // Поток Watchdog (упрощенная версия логики из run-bot.rs)
    let handle = std::thread::spawn(move || {
        let mut consecutive_misses = 0;
        let watchdog_config = state_clone.config.watchdog.clone();
        
        for _ in 0..100 { // Ограничиваем количество итераций для безопасности теста
            let now = unix_ms();
            let last = state_clone.last_heartbeat.load(Ordering::Relaxed);
            let delta = now.saturating_sub(last);
            
            if delta > watchdog_config.suspend_grace_ms {
                state_clone.last_heartbeat.store(now, Ordering::Relaxed);
                consecutive_misses = 0;
            } else if delta > watchdog_config.stall_timeout_ms {
                consecutive_misses += 1;
                if consecutive_misses >= 3 {
                    panic!("CRITICAL: HOT PATH STALLED for {}ms", delta);
                }
            } else {
                consecutive_misses = 0;
            }
            
            std::thread::sleep(Duration::from_millis(watchdog_config.check_interval_ms));
        }
    });

    // Мы НЕ обновляем heartbeat, поэтому через ~50ms * 3 + задержки он должен запаниковать
    let result = handle.join();
    
    // Проверяем, что поток завершился с паникой
    assert!(result.is_err(), "Watchdog should have panicked due to stall");
}

#[test]
fn test_watchdog_suspend_protection() {
    let mut config = HealthConfig::default();
    config.watchdog.stall_timeout_ms = 50;
    config.watchdog.check_interval_ms = 20;
    config.watchdog.suspend_grace_ms = 200; // Короткое время для теста

    let state = Arc::new(SharedState {
        last_update: AtomicU64::new(0),
        last_heartbeat: AtomicU64::new(unix_ms() - 500), // Имитируем старый heartbeat (больше suspend_grace)
        ws_connected: AtomicBool::new(true),
        emergency_mode: AtomicBool::new(false),
        start_time: Instant::now(),
        config: config.clone(),
    });

    let state_clone = state.clone();
    
    let handle = std::thread::spawn(move || {
        let mut consecutive_misses = 0;
        let watchdog_config = state_clone.config.watchdog.clone();
        
        // Выполняем одну проверку
        let now = unix_ms();
        let last = state_clone.last_heartbeat.load(Ordering::Relaxed);
        let delta = now.saturating_sub(last);
        
        if delta > watchdog_config.suspend_grace_ms {
            // Должно сработать это условие
            state_clone.last_heartbeat.store(now, Ordering::Relaxed);
            consecutive_misses = 0;
            return true;
        } else if delta > watchdog_config.stall_timeout_ms {
            consecutive_misses += 1;
            if consecutive_misses >= 3 {
                panic!("STALL");
            }
        }
        false
    });

    let result = handle.join().expect("Thread should not panic");
    assert!(result, "Watchdog should have detected suspend and reset heartbeat");
    
    // Проверяем, что heartbeat обновился
    let last = state.last_heartbeat.load(Ordering::Relaxed);
    assert!(unix_ms() - last < 100);
}
