mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::Duration;
use neirobot_lit::monitoring::health::{SharedState, HealthConfig, WatchdogConfig};
use neirobot_lit::utils::helpers;
use neirobot_lit::trading::emergency::cancel_all_sync;
use mockito::Server;
use tracing::info;

#[tokio::test]
async fn test_chaos_watchdog_trigger_emergency_cancel() {
    // 1. Запуск мок-сервера Bybit
    let mut server = Server::new_async().await;
    let mock = server.mock("POST", "/v5/order/cancel-all")
        .with_status(200)
        .with_body(r#"{"retCode":0,"retMsg":"OK"}"#)
        .expect(1) // Ожидаем ровно один вызов
        .create_async().await;

    // Переопределяем URL для экстренной отмены
    std::env::set_var("BYBIT_API_URL", server.url());

    // 2. Настройка SharedState с агрессивными таймаутами для теста
    let state = Arc::new(SharedState {
        last_update: AtomicU64::new(helpers::unix_ms()),
        last_heartbeat: AtomicU64::new(helpers::unix_ms()),
        ws_connected: AtomicBool::new(true),
        emergency_mode: AtomicBool::new(false),
        start_time: tokio::time::Instant::now(),
        config: HealthConfig {
            watchdog: WatchdogConfig {
                stall_timeout_ms: 100,  // Таймаут 100мс
                check_interval_ms: 50,  // Проверка каждые 50мс
                suspend_grace_ms: 10000,
            },
            ..Default::default()
        },
    });

    // 3. Установка кастомного panic hook, который вызывает отмену, но не завершает процесс теста
    // Важно: Hook глобальный, поэтому тест должен быть изолирован (или очищать за собой)
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        info!("Test Panic Hook: Detected panic, triggering emergency cancel. Info: {:?}", panic_info);
        // Вызываем реальную функцию отмены, которая пойдет на mockito сервер
        cancel_all_sync("test_key", "test_secret", "BTCUSDT");
    }));

    // 4. Запуск потока Watchdog (копия логики из run-bot.rs)
    let state_for_watchdog = state.clone();
    let watchdog_thread = std::thread::spawn(move || {
        let mut consecutive_misses = 0;
        let config = &state_for_watchdog.config.watchdog;
        
        loop {
            let now = helpers::unix_ms();
            let last = state_for_watchdog.last_heartbeat.load(Ordering::Relaxed);
            let delta = now.saturating_sub(last);
            
            if delta > config.stall_timeout_ms {
                consecutive_misses += 1;
                if consecutive_misses >= 3 {
                    panic!("CRITICAL: HOT PATH STALLED for {}ms (Simulated in Test)", delta);
                }
            } else {
                consecutive_misses = 0;
            }
            std::thread::sleep(Duration::from_millis(config.check_interval_ms));
            
            // Выход из цикла, если тест завершился (защита от бесконечного прогона при фейле)
            if state_for_watchdog.emergency_mode.load(Ordering::Relaxed) {
                break;
            }
        }
    });

    // 5. Симуляция "зависания": просто перестаем обновлять heartbeat
    // Watchdog должен сработать через ~300мс (3 пропуска по 100мс)
    tokio::time::sleep(Duration::from_millis(1000)).await;

    // 6. Проверка, что запрос на отмену был отправлен
    mock.assert_async().await;
    
    // Сигнал к завершению потока (если он еще жив) и восстановление хука
    state.emergency_mode.store(true, Ordering::Relaxed);
    std::panic::set_hook(default_hook);
}
