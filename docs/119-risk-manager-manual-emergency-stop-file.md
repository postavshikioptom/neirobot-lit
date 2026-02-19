# Задача 119: Ручной экстренный стоп через «Kill-файл» (v2.0)

## 1. Изменения в конфигурации `src/config/types.rs`
Добавь параметры управления файлом остановки в `BotConfig`:
```rust
pub struct BotConfig {
    // ...
    pub stop_file_name: String,            // По умолчанию "STOP"
    pub ack_extension: String,             // По умолчанию "DONE" (результат STOP.DONE)
    pub global_stop_enabled: bool,         // Проверять ли STOP_ALL в корне (default: true)
    pub stop_check_interval_ms: u64,       // Интервал проверки (default: 1000)
}
```

## 2. Реализация в `src/risk/risk_manager.rs`
Добавь логику проверки локального и глобального файлов:
```rust
use std::fs;
use std::path::{Path, PathBuf};

impl RiskManager {
    pub fn check_manual_stop(&self, bot_dir: &Path) -> Option<PathBuf> {
        // 1. Проверка локального файла в папке бота bots/SYMBOL/STOP
        let local_stop = bot_dir.join(&self.config.stop_file_name);
        if local_stop.exists() {
            return Some(local_stop);
        }

        // 2. Проверка глобального файла в корне проекта STOP_ALL
        if self.config.global_stop_enabled {
            let global_stop = Path::new("STOP_ALL");
            if global_stop.exists() {
                return Some(global_stop.to_path_buf());
            }
        }

        None
    }
}
```

## 3. Интеграция в основной цикл `src/bin/run-bot.rs`
Реализуй интервальную проверку с корректным завершением:

```rust
let mut stop_check_interval = tokio::time::interval(
    Duration::from_millis(execution.config.stop_check_interval_ms)
);

loop {
    tokio::select! {
        _ = stop_check_interval.tick() => {
            if let Some(stop_path) = execution.risk_manager.check_manual_stop(&bot_path) {
                tracing::error!("MANUAL STOP DETECTED: {}. Shutting down...", stop_path.display());
                
                // 1. Экстренное закрытие (задача 109)
                execution.emergency_market_close().await.ok();
                
                // 2. Попытка переименования файла для подтверждения (ACK)
                let ack_path = stop_path.with_extension(&execution.config.ack_extension);
                if let Err(e) = fs::rename(&stop_path, &ack_path) {
                    tracing::error!("Failed to rename stop file to {}: {}", ack_path.display(), e);
                    // Даже если переименовать не удалось (права доступа), мы все равно выходим
                }
                
                return Ok(()); // Выход из процесса
            }
        }
        // ... другие события (WS, сигналы)
    }
}
```

## 4. Особенности реализации
- **Atomic Rename**: Переименование в `.DONE` предотвращает циклическую остановку при автоматическом перезапуске бота супервизором (например, `PM2` или `systemd`).
- **Global Stop**: Файл `STOP_ALL` в корне проекта позволяет одним действием остановить всю ферму ботов.
- **Fail-safe**: Если `fs::rename` завершился ошибкой (например, файл заблокирован или нет прав), бот обязан залоггировать это и **все равно завершить работу**, так как приоритет — безопасность депозита.

---

## Аргументация для Планировщика:
1.  **No ./**: Пути указаны относительно корня проекта, согласно архитектуре.
2.  **Configurable Interval**: Позволяет снизить нагрузку на диск на слабых серверах или ускорить реакцию на мощных.
3.  **Explicit ACK**: Переименование файла — это визуальное подтверждение для оператора, что бот «увидел» команду и выполнил её.

**Gemini, реализуй этот механизм, обеспечив кроссплатформенную работу с путями через `std::path::PathBuf`.**