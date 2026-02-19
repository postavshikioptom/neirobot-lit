# Задача 146: Monitoring Stall Watchdog for Hot Path

## 1. Цель
Реализовать независимый компонент Watchdog, который отслеживает жизнеспособность основного торгового цикла (Hot Path) и инициирует аварийное завершение при обнаружении "зависания".

## 2. План реализации
1.  **Heartbeat**: Использовать `AtomicU64` в `SharedState` для хранения таймстемпа последнего успешного прохода цикла в [./src/bin/run-bot.rs](./src/bin/run-bot.rs).
2.  **Monitor Thread**: Запустить нативный поток (`std::thread::spawn`), который в бесконечном цикле проверяет дельту между `now` и `heartbeat`.
3.  **Stall Policy**: Реализовать проверку на 3 последовательных пропуска (например, проверка каждые 2с, таймаут 5с).
4.  **Suspend Protection**: Если дельта > 60с, считать это внешним событием (suspend сервера) и сбрасывать таймер без паники.
5.  **Metrics**: Добавить `Gauge` для `bot_watchdog_stall_seconds` и `bot_watchdog_last_check_timestamp`.

## 3. Технические детали
- **Panic Hook Integration**: При обнаружении Stall вызвать `panic!("HOT PATH STALLED")`. Это активирует `PanicHandler` из [./src/trading/emergency.rs](./src/trading/emergency.rs) для отмены ордеров.
- **Example Logic**:
```rust
let mut consecutive_misses = 0;
loop {
    let now = unix_ms();
    let last = state.last_heartbeat.load(Ordering::Relaxed);
    let delta = now.saturating_sub(last);

    if delta > config.suspend_grace_ms {
        warn!("Possible suspend detected (delta {}ms), resetting watchdog", delta);
        state.last_heartbeat.store(now, Ordering::Relaxed);
        consecutive_misses = 0;
    } else if delta > config.stall_timeout_ms {
        consecutive_misses += 1;
        error!("Watchdog: stall detected! Miss {}/3 (delta {}ms)", consecutive_misses, delta);
        if consecutive_misses >= 3 {
            panic!("CRITICAL: HOT PATH STALLED for {}ms", delta);
        }
    } else {
        consecutive_misses = 0;
    }
    
    // Update metrics
    METRICS.watchdog_last_check.set(now as f64 / 1000.0);
    thread::sleep(Duration::from_millis(config.check_interval_ms));
}
```

## 4. Критерии приемки
- [ ] Бот паникует и вызывает `cancel_all`, если основной цикл заблокирован более чем на `3 * timeout`.
- [ ] Метрика `bot_watchdog_last_check_timestamp` обновляется.
- [ ] Интеграционный тест `tests/watchdog_stall.rs` подтверждает срабатывание.

---
