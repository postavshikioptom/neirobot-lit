006 - Config Merge Logic & Validation
Цель задачи: Реализовать сборку итоговой конфигурации FullConfig. Основная логика: взять глобальные настройки и перекрыть их значениями из конфига конкретного бота, если они указаны. Добавить обязательную валидацию данных перед запуском.

Файлы для изменения:

src/config/types.rs
src/config/loader.rs
Инструкции для Gemini:

src/config/types.rs: Добавить структуру FullConfig. Она должна содержать как исходные части, так и "эффективные" (вычисленные) настройки.
```
#[derive(Debug, Clone)]
pub struct FullConfig {
    pub symbol: String, // Кэшируем для удобства
    pub general: GeneralConfig,
    pub logging: LoggingConfig,
    pub exchange: ExchangeConfig,
    pub trading: TradingDefaultsConfig, // Итоговые (после мерджа)
    pub risk: RiskDefaultsConfig,       // Итоговые (после мерджа)
    pub bot: BotConfig,                 // Исходный конфиг бота
}
```
src/config/loader.rs: Реализовать сборку с безопасной обработкой Option и валидацией.
```
use crate::config::types::*;
use std::path::Path;
use anyhow::{bail, Result};

pub fn load_full_config(root_path: &Path, bot_config_path: &Path) -> Result<FullConfig> {
    let global = load_global(root_path)?;
    let exchange = load_exchange(root_path)?;
    let bot = load_bot(bot_config_path)?;

    // 1. Мерджим Риски (Bot > Global)
    let mut effective_risk = global.risk_defaults.unwrap_or_default();
    if let Some(val) = bot.max_position_size { effective_risk.max_position_size = Some(val); }
    if let Some(val) = bot.max_drawdown_pct { effective_risk.max_drawdown_pct = Some(val); }

    // 2. Мерджим Трейдинг (пока берем дефолты, если будут оверрайды в BotConfig — добавим сюда)
    let effective_trading = global.trading_defaults.unwrap_or_default();

    let full = FullConfig {
        symbol: bot.symbol.clone(),
        general: global.general,
        logging: global.logging,
        exchange,
        trading: effective_trading,
        risk: effective_risk,
        bot,
    };

    // 3. Валидация
    validate_config(&full)?;

    Ok(full)
}

fn validate_config(cfg: &FullConfig) -> Result<()> {
    if cfg.symbol.is_empty() { bail!("Validation error: Symbol is empty"); }
    if cfg.bot.model_path.is_empty() { bail!("Validation error: model_path is empty"); }
    
    if cfg.bot.threshold_buy <= cfg.bot.threshold_sell {
        bail!("Validation error: threshold_buy must be greater than threshold_sell");
    }

    if let Some(pos) = cfg.risk.max_position_size {
        if pos <= 0.0 { bail!("Validation error: max_position_size must be positive"); }
    }

    Ok(())
}
```
Технические требования:

Использовать .unwrap_or_default() для Option<Config> из global.toml (требует #[derive(Default)] для структур в types.rs).
Метод validate_config должен возвращать anyhow::Error через bail!, если данные некорректны.
Логика мерджа полей risk должна быть явной: если у бота поле Some, оно заменяет глобальное.
Почему это важно: Это "ворота" приложения. Если мы пропустим пустой symbol или некорректные пороги (thresholds), бот упадет позже с невнятной ошибкой. Валидация на старте экономит часы отладки.

Grok, теперь я учел твои замечания:

Убрал несуществующий overrides.
Добавил unwrap_or_default() для безопасной работы с GlobalConfig.
Добавил реальную функцию validate_config с проверкой порогов и путей.
FullConfig теперь содержит и части, и "эффективные" значения.