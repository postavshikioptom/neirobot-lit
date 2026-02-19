# Задача 230: Изоляция ресурсов и квоты (CPU/RAM)

Реализация механизмов жесткой и мягкой изоляции ресурсов для каждого инстанса бота. Основной упор на использование **Linux cgroups** через **systemd** для гарантированной аллокации и внутренние проверки в Rust для предотвращения критических сбоев.

## 1. Цель задачи
Обеспечить стабильность торговой фермы при запуске 100+ процессов на одном сервере. Исключить влияние утечек памяти или CPU-спайков одного бота на работу остальных.

## 2. Инструкции по реализации для Gemini

### А. Rust: Внутренний мониторинг и Affinity ([./src/utils/system.rs](./src/utils/system.rs))
1.  **Функция Affinity**:
    *   Реализовать `pub fn set_process_affinity(core_id: usize)`.
    *   Использовать библиотеку `core_affinity`. Вызывать сразу после загрузки конфига в [./src/bin/run-bot.rs](./src/bin/run-bot.rs).
2.  **Memory Guard**:
    *   Реализовать `pub fn monitor_resources(max_mem_mb: u64, tx: mpsc::Sender<SystemEvent>)`.
    *   Использовать `sysinfo` для получения `resident_set_size (RSS)` текущего процесса.
    *   При превышении `max_mem_mb * 0.9` (90% порог): отправить сигнал `SoftLimitReached` для перехода в режим `Graceful Degradation` (задача 220).

### Б. Orchestration: Hard Isolation (cgroups v2)
1.  **Запуск через systemd-run**:
    *   Для обеспечения жестких лимитов, которые процесс не может обойти, использовать `systemd-run`.
    *   Пример команды для интеграции в `farm_launcher.py`:
        ```bash
        systemd-run --user --scope \
          -p MemoryMax=512M \
          -p CPUQuota=100% \
          -p AllowedCPUs=1 \
          ./target/release/run-bot --symbol BTCUSDT
        ```
2.  **Опциональный аллокатор**:
    *   Добавить поддержку `jemalloc` под флагом в `Cargo.toml` для лучшего контроля фрагментации:
        ```rust
        #[cfg(feature = "jemalloc")]
        #[global_allocator]
        static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
        ```

## 3. Интеграция в код

1.  **В [./src/bin/run-bot.rs](./src/bin/run-bot.rs)**:
    ```rust
    // Инициализация Affinity
    if let Some(core) = config.system.cpu_core {
        system::set_process_affinity(core);
    }

    // Запуск потока мониторинга ресурсов
    let max_mem = config.system.max_memory_mb;
    tokio::spawn(async move {
        let mut sys = sysinfo::System::new_all();
        loop {
            sys.refresh_process(sysinfo::get_current_pid().unwrap());
            // Проверка лимитов и отправка событий
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
    });
    ```

## 4. Аргументация и уточнения (по Grok)
*   **Приоритет cgroups**: Внутренний `core_affinity` важен для HFT-оптимизации внутри процесса, но **AllowedCPUs** в systemd — единственный надежный способ гарантировать, что бот не займет чужое ядро при сбое.
*   **Отказ от Jemalloc по умолчанию**: Согласен, оставляем его как опциональную `feature`. Для большинства задач стандартного аллокатора достаточно, а отладка `jemalloc` сложнее.
*   **Периодичность проверки**: Проверка памяти раз в 5 секунд достаточна для "мягкого" лимита. "Жесткий" лимит (OOM Killer) отработает мгновенно на уровне ядра.

## 5. Ожидаемый результат
1.  Бот привязан к конкретному ядру CPU (подтверждается через `taskset -p <pid>`).
2.  При попытке занять более 512MB RAM (или другого лимита из конфига) бот либо корректно останавливается (Rust), либо убивается ОС (cgroups), не затрагивая соседей.

## 6. Необходимые зависимости
-   **Rust**: `core_affinity = "0.8"`, `sysinfo = "0.30"`.
-   **System**: `systemd` с поддержкой пользовательских `slices/scopes`.