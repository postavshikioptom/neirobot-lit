
# Задача 145: Monitoring Memory and CPU Usage Tracking

## 1. Цель
Реализовать систему отслеживания потребления ресурсов (RSS, CPU) процессом бота с экспортом в Prometheus и интеграцией в Health-check.

## 2. План реализации
1.  **Utility**: В `src/utils/sys.rs` (или [./src/utils/mod.rs](./src/utils/mod.rs)) создать глобальный `OnceLock` для `sysinfo::System`.
2.  **Metrics**: Добавить атомарные переменные для хранения последних метрик (чтобы не блокировать Hot Path системными вызовами).
3.  **Background Task**: В фоновом рантайме (из задачи 144) запустить цикл (например, раз в 5 секунд), который обновляет `sysinfo`.
4.  **Integration**: Обновить эндпоинт `/health` (из задачи 135) и Prometheus (из задачи 143).

## 3. Технические детали
- **Crate**: `sysinfo = "0.30"` (или новее).
- **Global State**:
```rust
static SYSTEM: OnceLock<Mutex<System>> = OnceLock::new();
static PID: OnceLock<Pid> = OnceLock::new();

pub struct ResourceMetrics {
    pub rss_bytes: AtomicU64,
    pub cpu_usage: AtomicF32,
}
```
- **Update Logic**:
```rust
let mut sys = SYSTEM.get_or_init(|| Mutex::new(System::new_all())).lock().unwrap();
let pid = *PID.get_or_init(|| Pid::from_raw(std::process::id() as i32));
sys.refresh_process(pid);
if let Some(process) = sys.process(pid) {
    METRICS.rss_bytes.store(process.memory(), Ordering::Relaxed);
    METRICS.cpu_usage.store(process.cpu_usage(), Ordering::Relaxed);
}
```

## 4. Критерии приемки
- [ ] Метрика `bot_memory_usage_bytes` доступна в `/metrics`.
- [ ] Метрика `bot_cpu_usage_percent` доступна в `/metrics`.
- [ ] `/health` возвращает `"status": "degraded"` если `rss > config.max_memory_mb`.
- [ ] Отсутствие аллокаций `System::new()` в цикле обновления.
