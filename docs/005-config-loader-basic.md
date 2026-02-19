# 005 - Config Loader Basic

**Цель задачи:**  
Реализовать базовую загрузку TOML-файлов (global.toml, exchange.toml и bot/config.toml) в структуры из `types.rs`. Пока только чтение и десериализация — без merge (это в 006). Избежать hardcoded путей, обеспечить DRY-код и добавить базовое логирование через tracing.

**Файлы для изменения/создания:**

- `src/config/mod.rs` — обновить объявления модулей.
- `src/config/loader.rs` — основной код загрузчика.

**Инструкции для Gemini:**  

1. **Обновить src/config/mod.rs**

```rust
pub mod types;
pub mod loader;

pub use types::*;
```

2. **Создать src/config/loader.rs**

```rust
use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

use crate::config::types::{BotConfig, ExchangeConfig, GlobalConfig};

/// Приватная generic-функция для загрузки любого TOML-файла
fn load_toml<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let display_path = path.display();

    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read config file: {display_path}"))?;

    toml::from_str(&content)
        .with_context(|| format!("Failed to parse TOML in: {display_path}"))
}

/// Загрузка global.toml по пути к корню проекта
pub fn load_global(root_path: &Path) -> Result<GlobalConfig> {
    let path = root_path.join("global.toml");
    let config = load_toml(&path)?;
    info!("Global config successfully loaded from {}", path.display());
    Ok(config)
}

/// Загрузка exchange.toml по пути к корню проекта
pub fn load_exchange(root_path: &Path) -> Result<ExchangeConfig> {
    let path = root_path.join("exchange.toml");
    let config = load_toml(&path)?;
    info!("Exchange config successfully loaded from {}", path.display());
    Ok(config)
}

/// Загрузка config.toml конкретного бота по полному пути к файлу
pub fn load_bot(bot_config_path: &Path) -> Result<BotConfig> {
    if !bot_config_path.exists() {
        warn!("Bot config file not found: {}", bot_config_path.display());
    }
    let config = load_toml(bot_config_path)?;
    info!("Bot config successfully loaded from {}", bot_config_path.display());
    Ok(config)
}
```

**Технические требования:**

- Все пути передаются явно (root_path для global/exchange, полный путь для bot).
- Использовать generic `load_toml` для избежания дублирования.
- Обязательно `anyhow::Context` с указанием пути в сообщениях об ошибках.
- tracing::info! при успешной загрузке, tracing::warn! если файл не найден (но ошибка не критичная — зависит от вызова).
- Функции публичные, принимают `&Path` для гибкости.
- Не добавлять merge или загрузку .env.

**Почему это важно:**  
Передача путей явно делает загрузчик надёжным независимо от текущей рабочей директории. Generic-функция упрощает поддержку. Раннее логирование через tracing позволит видеть в логах успешную загрузку конфигов при старте бота.

**Ожидаемый результат:**  
- `cargo check` проходит.
- Функции можно вызвать, например: `loader::load_global(Path::new("."))?` или с тестовым путём.
- Ошибки содержат полный путь к файлу.

**Следующая задача:** 006-config-merge-logic.md

Эта версия полностью учитывает замечания Claude: нет hardcoded строк, DRY через generic, tracing добавлен, пути передаются параметрами. Готов к проверке или переходу к 006.