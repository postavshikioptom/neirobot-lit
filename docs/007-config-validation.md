# Задача 007 - Config Validation
Цель задачи: Реализовать строгую валидацию конфигурации после её загрузки и мерджа. Мы должны гарантировать, что бот не запустится с некорректными данными (пустой символ, отрицательные риски, неверные пороги модели).

Файлы для изменения:

src/config/loader.rs (добавить логику валидации)
src/config/types.rs (возможно, добавить методы валидации к структурам)
Инструкции для Gemini:

src/config/loader.rs: Добавить функцию validate_full_config, которая будет вызываться в конце load_full_config.

use anyhow::{bail, Result};
use std::path::Path;
use crate::config::types::FullConfig;

pub fn validate_full_config(cfg: &FullConfig) -> Result<()> {
    // 1. Проверка символа (не должен быть пустым)
    if cfg.symbol.trim().is_empty() {
        bail!("Config validation error: 'symbol' is empty");
    }

    // 2. Проверка пути к модели
    let model_path = Path::new(&cfg.bot.model_path);
    if !model_path.exists() {
        // Это может быть предупреждением, если модель скачивается позже, 
        // но для LiT модели она должна быть готова.
        bail!("Config validation error: model file not found at {:?}", model_path);
    }

    // 3. Проверка порогов (thresholds)
    if cfg.bot.threshold_buy <= cfg.bot.threshold_sell {
        bail!("Config validation error: threshold_buy ({}) must be greater than threshold_sell ({})", 
              cfg.bot.threshold_buy, cfg.bot.threshold_sell);
    }
    
    if cfg.bot.threshold_buy <= 0.0 || cfg.bot.threshold_sell >= 0.0 {
         // Обычно buy > 0 (long), sell < 0 (short). Для нейтральной стратегии это важно.
         // Но мы проверим просто логическую корректность buy > sell.
    }

    // 4. Проверка лимитов риска
    if let Some(max_pos) = cfg.risk.max_position_size {
        if max_pos <= 0.0 {
            bail!("Config validation error: max_position_size must be positive, got {}", max_pos);
        }
    }

    Ok(())
}
Обновить load_full_config:

pub fn load_full_config(root_path: &Path, bot_config_path: &Path) -> Result<FullConfig> {
    // ... код загрузки и мерджа из задачи 006 ...
    let full_cfg = FullConfig { /* ... */ };
    
    // Обязательная валидация перед возвратом
    validate_full_config(&full_cfg)?;
    
    Ok(full_cfg)
}
Технические требования:

Использовать anyhow::bail! для немедленного возврата ошибки с понятным описанием.
Проверять существование model_path через std::path::Path::exists.
Убедиться, что threshold_buy и threshold_sell не равны и имеют правильную иерархию.
Почему это важно: Ошибки в конфиге — самая частая причина падений в проде. Проверка buy > sell предотвратит ситуацию, когда бот будет одновременно слать ордера на покупку и продажу из-за неверных чисел.