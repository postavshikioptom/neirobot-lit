use anyhow::{bail, Context, Result};
use std::fs;
use std::path::Path;
use tracing::{info, warn};
use rust_decimal::Decimal;
use sha2::{Sha256, Digest};
use serde::{Deserialize, Serialize};

use crate::config::types::*;
use crate::utils::crypto;

/// Структура для маппинга переменных окружения через serde
/// Задача 188: Использование serde для маппинга переменных окружения
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EnvConfig {
    /// Символ торговой пары (переопределяет значение из bot.toml)
    #[serde(default)]
    pub bot_symbol: Option<String>,
}

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

/// Загрузка конфигурации из переменных окружения через serde
/// Задача 188: Маппинг переменных окружения через serde
pub fn load_env_config() -> Result<EnvConfig> {
    let mut env_config = EnvConfig {
        bot_symbol: None,
    };
    
    // Загружаем BOT_SYMBOL если установлена
    if let Ok(symbol) = std::env::var("BOT_SYMBOL") {
        if !symbol.is_empty() {
            env_config.bot_symbol = Some(symbol);
            info!("Loaded BOT_SYMBOL from environment: {}", env_config.bot_symbol.as_ref().unwrap());
        }
    }
    
    Ok(env_config)
}

/// Сборка полной конфигурации с мерджем и валидацией
pub fn load_full_config(root_path: &Path, bot_config_path: &Path) -> Result<FullConfig> {
    let global = load_global(root_path)?;
    let exchange = load_exchange(root_path)?;
    let bot = load_bot(bot_config_path)?;
    
    // Задача 188: Загрузка конфигурации из переменных окружения через serde
    let env_config = load_env_config()?;

    // 1. Мерджим Риски (Bot > Global)
    let mut effective_risk = global.risk_defaults.clone();
    
    // Слияние значений из bot config в эффективный risk config
    if let Some(val) = bot.max_position_size {
        effective_risk.max_position_size = Some(val);
    }
    if let Some(val) = bot.max_drawdown_pct {
        effective_risk.max_drawdown_pct = Some(val);
    }
    
    // Слияние всех полей из bot.risk
    if let Some(val) = bot.risk.max_price_deviation_pct {
        effective_risk.max_price_deviation_pct = Some(val);
    }
    if let Some(val) = bot.risk.max_spread_bps {
        effective_risk.max_spread_bps = Some(val);
    }
    if let Some(val) = bot.risk.max_open_orders {
        effective_risk.max_open_orders = Some(val);
    }
    if let Some(val) = bot.risk.max_position_size {
        effective_risk.max_position_size = Some(val);
    }
    if let Some(val) = bot.risk.max_notional_usd {
        effective_risk.max_notional_usd = Some(val);
    }
    if let Some(val) = bot.risk.max_margin_usd {
        effective_risk.max_margin_usd = Some(val);
    }
    if let Some(val) = bot.risk.max_daily_drawdown_usd {
        effective_risk.max_daily_drawdown_usd = Some(val);
    }
    if let Some(val) = bot.risk.max_daily_drawdown_pct {
        effective_risk.max_daily_drawdown_pct = Some(val);
    }
    if let Some(val) = bot.risk.max_drawdown_pct {
        effective_risk.max_drawdown_pct = Some(val);
    }

    // 2. Мерджим Трейдинг
    let effective_trading = global.trading_defaults.clone();

    let mut full = FullConfig {
        symbol: bot.symbol.clone(),
        general: global.general.clone(),
        logging: global.logging.clone(),
        exchange,
        trading: effective_trading,
        risk: effective_risk,
        bot,
        monitoring: global.monitoring.unwrap_or_default(),
        global,
    };

    // Задача 188: Перекрытие параметров из переменных окружения через serde
    // Переменные окружения имеют высший приоритет и перекрывают значения из bot.toml
    if let Some(env_symbol) = env_config.bot_symbol {
        full.symbol = env_symbol;
        info!("Config parameter 'symbol' overridden from environment: {}", full.symbol);
    }

    // 3. Валидация
    validate_full_config(&full)?;

    Ok(full)
}

/// Строгая валидация итоговой конфигурации
pub fn validate_full_config(cfg: &FullConfig) -> Result<()> {
    // 1. Проверка символа (не должен быть пустым)
    if cfg.symbol.trim().is_empty() {
        bail!("Config validation error: 'symbol' is empty");
    }

    // Проверка формата символа (должен быть в верхнем регистре, без дефисов)
    if cfg.symbol.contains('-') || cfg.symbol.contains('_') {
        bail!(
            "Config validation error: symbol '{}' has invalid format. Expected format: BTCUSDT (uppercase, no separators)",
            cfg.symbol
        );
    }

    // Проверка, что символ в верхнем регистре
    if cfg.symbol != cfg.symbol.to_uppercase() {
        bail!(
            "Config validation error: symbol '{}' must be in uppercase. Expected: {}",
            cfg.symbol,
            cfg.symbol.to_uppercase()
        );
    }

    // 2. Проверка пути к модели
    let model_path = Path::new(&cfg.bot.model_path);
    if !model_path.exists() {
        bail!("Config validation error: model file not found at {:?}", model_path);
    }

    // 3. Проверка логики порогов (Buy должен быть выше Sell)
    if cfg.bot.threshold_buy <= cfg.bot.threshold_sell {
        bail!(
            "Config validation error: threshold_buy ({}) must be greater than threshold_sell ({})",
            cfg.bot.threshold_buy,
            cfg.bot.threshold_sell
        );
    }

    // Проверка диапазона threshold_flat
    if cfg.bot.threshold_flat < 0.0 || cfg.bot.threshold_flat > 1.0 {
        bail!("Config validation error: threshold_flat must be between 0.0 and 1.0");
    }

    // Проверка асимметричных порогов (long_threshold и short_threshold)
    if cfg.bot.long_threshold <= Decimal::ZERO || cfg.bot.long_threshold >= Decimal::ONE {
        bail!(
            "Config validation error: long_threshold must be between 0 and 1, got {}",
            cfg.bot.long_threshold
        );
    }
    
    if cfg.bot.short_threshold <= Decimal::ZERO || cfg.bot.short_threshold >= Decimal::ONE {
        bail!(
            "Config validation error: short_threshold must be between 0 and 1, got {}",
            cfg.bot.short_threshold
        );
    }
    
    // Проверка exit_threshold если задан
    if let Some(exit_th) = cfg.bot.exit_threshold {
        if exit_th <= Decimal::ZERO || exit_th >= Decimal::ONE {
            bail!(
                "Config validation error: exit_threshold must be between 0 and 1, got {}",
                exit_th
            );
        }
    }

    // 4. Проверка лимитов риска
    if let Some(max_pos) = cfg.risk.max_position_size {
        if max_pos <= Decimal::ZERO {
            bail!("Config validation error: max_position_size must be positive, got {}", max_pos);
        }
    }

    // Проверка max_open_orders
    if let Some(orders) = cfg.risk.max_open_orders {
        if orders == 0 {
            bail!("Config validation error: max_open_orders must be > 0");
        }
    }

    // Проверка max_spread_bps
    if let Some(spread) = cfg.risk.max_spread_bps {
        if spread == 0 {
            bail!("Config validation error: max_spread_bps must be > 0");
        }
    }

    // Проверка конфигурации слияния (Fusion)
    use crate::config::types::FusionMethod;
    
    match cfg.bot.fusion.method {
        FusionMethod::WeightedAverage => {
            if cfg.bot.fusion.weights.is_empty() {
                bail!("Config validation error: fusion weights cannot be empty for WeightedAverage method");
            }
            
            // Проверка, что сумма весов равна 1.0 (с допуском на погрешность)
            let sum: Decimal = cfg.bot.fusion.weights.iter().sum();
            let diff = (sum - Decimal::ONE).abs();
            if diff > Decimal::new(1, 4) { // 0.0001 допуск
                bail!(
                    "Config validation error: fusion weights must sum to 1.0, got {}",
                    sum
                );
            }
            
            // Проверка, что все веса положительные
            for (i, w) in cfg.bot.fusion.weights.iter().enumerate() {
                if *w < Decimal::ZERO {
                    bail!(
                        "Config validation error: fusion weight[{}] must be non-negative, got {}",
                        i, w
                    );
                }
            }
        }
        FusionMethod::Consensus => {
            if cfg.bot.fusion.min_horizons == 0 {
                bail!("Config validation error: fusion min_horizons must be > 0 for Consensus method");
            }
        }
        FusionMethod::Principal => {
            // principal_idx будет проверен в runtime, так как количество горизонтов неизвестно
        }
    }

    Ok(())
}


/// Загрузка секретов (API ключи) из переменных окружения
/// Приоритет: переменные окружения > .env файл
pub fn load_secrets() -> Result<(String, String)> {
    let key = std::env::var("BYBIT_API_KEY")
        .context("BYBIT_API_KEY not found in environment variables")?;
    let secret = std::env::var("BYBIT_API_SECRET")
        .context("BYBIT_API_SECRET not found in environment variables")?;
    
    // Обработка зашифрованных значений
    let key = decrypt_if_needed(&key)?;
    let secret = decrypt_if_needed(&secret)?;
    
    Ok((key, secret))
}

/// Вспомогательная функция для расшифровки значения, если оно зашифровано
fn decrypt_if_needed(value: &str) -> Result<String> {
    if crypto::is_encrypted(value) {
        // Если значение зашифровано, мастер-ключ ОБЯЗАТЕЛЕН
        let master_key = std::env::var("NEIRO_MASTER_KEY")
            .context("FATAL: Encrypted value found but NEIRO_MASTER_KEY is not set in environment")?;
        
        if master_key.is_empty() {
            bail!("FATAL: NEIRO_MASTER_KEY is empty");
        }
        
        // Расшифровываем
        match crypto::decrypt(value, &master_key) {
            Ok(decrypted) => {
                info!("Config decryption: SUCCESS");
                Ok(decrypted)
            }
            Err(e) => {
                warn!("Config decryption: FAILURE - {}", e);
                Err(e).context("Failed to decrypt value")
            }
        }
    } else {
        // Значение не зашифровано, возвращаем как есть
        Ok(value.to_string())
    }
}

/// Загрузка Telegram credentials из переменных окружения
/// 
/// Возвращает Ok(Some((token, chat_id))) если оба параметра найдены,
/// Ok(None) если хотя бы один отсутствует
pub fn load_telegram_credentials() -> Result<Option<(String, String)>> {
    match (
        std::env::var("TELEGRAM_TOKEN").ok(),
        std::env::var("TELEGRAM_CHAT_ID").ok(),
    ) {
        (Some(token), Some(chat_id)) => {
            if token.is_empty() || chat_id.is_empty() {
                Ok(None)
            } else {
                // Расшифровываем, если нужно
                let token = decrypt_if_needed(&token)?;
                let chat_id = decrypt_if_needed(&chat_id)?;
                Ok(Some((token, chat_id)))
            }
        }
        _ => Ok(None),
    }
}


/// Вычисление SHA-256 хэша содержимого файла конфигурации
pub fn compute_config_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let result = hasher.finalize();
    hex::encode(result)
}

/// Вычисление SHA-256 хэша файла конфигурации по пути
pub fn compute_config_hash_from_file(path: &Path) -> Result<String> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read config file for hashing: {}", path.display()))?;
    Ok(compute_config_hash(&content))
}

/// Структура для отслеживания состояния конфигурации
#[derive(Clone, Debug)]
pub struct ConfigAudit {
    pub hash: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Сравнение двух версий конфигурации и вывод diff
pub fn generate_config_diff(old_content: &str, new_content: &str) -> String {
    use diffy::create_patch;
    
    let patch = create_patch(old_content, new_content);
    let mut diff_output = String::new();
    
    for hunk in patch.hunks() {
        for line in hunk.lines() {
            match line {
                diffy::Line::Context(l) => {
                    diff_output.push_str(&format!(" {}\n", l));
                }
                diffy::Line::Delete(l) => {
                    diff_output.push_str(&format!("-{}\n", l));
                }
                diffy::Line::Add(l) => {
                    diff_output.push_str(&format!("+{}\n", l));
                }
            }
        }
    }
    
    diff_output
}

/// Логирование изменений конфигурации при перезагрузке
pub fn log_config_changes(old_content: &str, new_content: &str, old_hash: &str, new_hash: &str) {
    if old_hash != new_hash {
        info!("[Audit] Config SHA-256 changed: {} -> {}", old_hash, new_hash);
        
        let diff = generate_config_diff(old_content, new_content);
        if !diff.is_empty() {
            info!("[Audit] Config diff:\n{}", diff);
        }
    } else {
        info!("[Audit] Config hash unchanged: {}", new_hash);
    }
}

/// Создание резервной копии конфигурации в config_history
pub fn backup_config(bot_config_path: &Path) -> Result<PathBuf> {
    let bot_dir = bot_config_path.parent()
        .context("Failed to get bot directory")?;
    
    let history_dir = bot_dir.join("config_history");
    fs::create_dir_all(&history_dir)
        .with_context(|| format!("Failed to create config_history directory: {}", history_dir.display()))?;
    
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let backup_name = format!("config.toml.{}.bak", timestamp);
    let backup_path = history_dir.join(&backup_name);
    
    fs::copy(bot_config_path, &backup_path)
        .with_context(|| format!("Failed to backup config to: {}", backup_path.display()))?;
    
    // Задача 184: Установка прав доступа 600 на резервную копию (только владелец может читать/писать)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let permissions = fs::Permissions::from_mode(0o600);
        fs::set_permissions(&backup_path, permissions)
            .with_context(|| format!("Failed to set permissions on backup: {}", backup_path.display()))?;
    }
    
    info!("[Audit] Config backup created: {} (permissions: 600)", backup_path.display());
    
    Ok(backup_path)
}
