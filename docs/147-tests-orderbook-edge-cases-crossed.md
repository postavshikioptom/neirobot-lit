# Задача 147: Integration Tests Chaos Monkey Latency Injector

## 1. Цель
Создать систему контролируемой деградации производительности и надежности для проверки реакции бота на сетевые сбои, пропуски пакетов и критические зависания.

## 2. План реализации
1.  **Feature Gate**: Добавить `feature "chaos"` в `Cargo.toml`. Весь код инжектора обернуть в `#[cfg(feature = "chaos")]`.
2.  **Middleware**: В [./src/data/websocket.rs](./src/data/websocket.rs) внедрить вызов `inject_chaos` перед парсингом сообщения.
3.  **Chaos State**: Реализовать структуру с `StdRng` и параметрами распределения задержек.
4.  **Sequence Gaps**: Реализовать имитацию потери пакетов (`packet_loss_rate`), которая просто пропускает инкремент последовательности, вынуждая систему вызвать `reconnect` или `checksum recovery`.

## 3. Технические детали
- **Distributions**:
```rust
#[cfg(feature = "chaos")]
pub async fn inject_chaos(config: &ChaosConfig, rng: &mut StdRng) {
    // 1. Packet Loss (simulating sequence gaps)
    if rng.gen_bool(config.packet_loss_rate) {
        return; // Skip message, trigger gap detection
    }
    
    // 2. Latency with Long Tail (Exponential)
    let exp = Exponential::new(1.0 / config.mean_latency_ms as f64).unwrap();
    let delay = exp.sample(rng) as u64;
    tokio::time::sleep(Duration::from_millis(delay)).await;
}
```
- **Watchdog Integration Test**:
  - В тестовом сценарии выставить `mean_latency_ms = 15000`.
  - Убедиться, что `Stall Watchdog` (задача 146) детектирует зависание через 3 итерации.
  - Проверить через `mockall` или `mockito`, что вызван аварийный `POST /cancel-all`.

## 4. Критерии приемки
- [ ] Инжектор работает только при `cargo build --features chaos`.
- [ ] Тест `tests/chaos_recovery.rs` подтверждает реконнект при потере последовательности.
- [ ] Тест `tests/chaos_watchdog.rs` подтверждает экстренную остановку при фризах.
- [ ] Использование `rand_distr` для имитации реалистичного сетевого джиттера.

---
