
# Задача 144: Тюнинг приоритетов потоков и изоляция Hot Path (v2.0)

## 1. Системный тюнинг в [./src/utils/sys.rs](./src/utils/sys.rs)
Реализуй кроссплатформенную логику установки приоритетов и привязки к ядрам процессора (CPU Pinning).

```rust
// В [./src/utils/sys.rs](./src/utils/sys.rs)
#[cfg(target_os = "linux")]
use nix::sched::{set_scheduler, SchedParameters, Scheduler};
use core_affinity::CoreId;

pub fn set_hot_thread_config(core_id: Option<usize>) {
    // 1. Привязка к конкретному физическому ядру
    if let Some(id) = core_id {
        core_affinity::set_for_current(CoreId { id });
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
```

## 2. Разделение рантаймов в [./src/bin/run-bot.rs](./src/bin/run-bot.rs)
Создай два пула потоков: один для трейдинга, другой для логов и метрик.

```rust
fn main() {
    let physical_cores = num_cpus::get_physical();
    let hot_workers = (physical_cores - 2).max(1);

    // 1. Hot Path Runtime (Trading, Inference, WS)
    let hot_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(hot_workers)
        .thread_name("hot-worker")
        .on_thread_start(move || {
            // В будущем: прокидывать ID потока для точного пиннинга
            crate::utils::sys::set_hot_thread_config(None);
        })
        .enable_all()
        .build()
        .unwrap();

    // 2. Background Runtime (Logging, Metrics, Health)
    let bg_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("bg-worker")
        .on_thread_start(|| {
            #[cfg(target_os = "linux")]
            unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, 10); } // Низкий приоритет
        })
        .enable_all()
        .build()
        .unwrap();

    // Запуск фоновых задач
    bg_rt.spawn(async { /* start_health_server, prometheus, etc */ });

    // Запуск основного цикла
    hot_rt.block_on(async_main());
}
```

## 3. Спорные моменты и корректировки (Grok + Zencoder)

- **SCHED_RR vs SCHED_FIFO**: Полностью согласен с Grok. `SCHED_FIFO` опасен тем, что при ошибке (бесконечный цикл) он может полностью заморозить ядро ОС. `SCHED_RR` (Round Robin) позволяет системе прерывать поток, что безопаснее для стабильности.
- **Double Runtime**: Это ключевое решение. Разделяя `hot_rt` и `bg_rt`, мы гарантируем, что даже если `prometheus` или `logging` начнут потреблять 100% выделенных им ресурсов, они физически не смогут вытеснить торговые потоки с их ядер.
- **CPU Pinning**: Использование `core_affinity` снижает количество переключений контекста (Context Switches), что делает задержки более предсказуемыми (уменьшает jitter).
- **Permissions**: Обязательно обрабатываем `EPERM`. В Linux для установки RT-приоритетов нужны права `root` или `CAP_SYS_NICE`. Если их нет — откатываемся к `nice -10`.

## 4. Инструкции для Gemini (Coder AI):
1. **Cargo.toml**: Добавить `nix = "0.27"`, `libc = "0.2"`, `core_affinity = "0.8"`, `num_cpus = "1.16"`.
2. **[./src/utils/sys.rs](./src/utils/sys.rs)**: Реализовать кроссплатформенную функцию `set_hot_thread_config`.
3. **[./src/bin/run-bot.rs](./src/bin/run-bot.rs)**: Заменить `#[tokio::main]` на ручную инициализацию двух рантаймов.
4. **Validation**: Проверить в логах, что потоки запускаются с корректными именами (`hot-worker` / `bg-worker`).

**Результат**: Максимально детерминированная производительность. Торговый цикл защищен от влияния «тяжелых» системных операций, а задержки сведены к аппаратному минимуму.
