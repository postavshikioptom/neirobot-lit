# Задача №161: Динамические пороги входа на основе режимов рынка

Согласно [./docs/000-tasks_list.md](./docs/000-tasks_list.md) и результатам задачи 155 (Market Regime Detection), бот должен адаптировать свою агрессивность к фазам рынка. Внедрение динамических порогов позволит требовать более высокую уверенность модели в волатильные периоды и снижать её в стабильные.

## План реализации для кодера:

### 1. Типы данных в [./src/config/types.rs](./src/config/types.rs)
Определить структуру режимов и переопределений:
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegimeId {
    Quiet = 0,
    Trend = 1,
    Volatile = 2,
    Unknown = 255,
}

#[derive(Debug, Deserialize)]
pub struct ThresholdOverride {
    pub regime: RegimeId,
    pub buy_threshold: f32,
    pub sell_threshold: f32,
    pub min_confidence: f32,
}
```
Добавить `Vec<ThresholdOverride>` в `BotConfig`.

### 2. Компонент `RegimeDetector` в [./src/trading/mod.rs](./src/trading/mod.rs)
Создать структуру для онлайн-классификации режима:
- **Загрузка**: При старте загружать [./bots/SYMBOL/model/regime_config.json](./bots/SYMBOL/model/regime_config.json) (центроиды или параметры HMM из задачи 155).
- **Расчет**: Реализовать онлайн-вычисление признаков (`Intensity`, `Volatility`, `Spread Z-Score`, `OFI`) на основе последних снапшотов из `OrderBook`.
- **Классификация**: Метод `detect(&features) -> RegimeId`.

### 3. Механизм гистерезиса (Hysteresis)
Чтобы избежать частого переключения порогов на границах режимов, внедрить счетчик в `RegimeDetector`:
- Если новый режим отличается от текущего, он должен подтвердиться $N$ раз подряд (например, 10 снапшотов), прежде чем `current_regime` обновится.

### 4. Логика исполнения в [./src/trading/execution.rs](./src/trading/execution.rs)
Обновить метод `on_signal`:
- **Сигнатура**: `fn on_signal(&self, signal: Signal, confidence: f32, regime: RegimeId)`.
- **Выбор порогов**:
    1. Искать совпадение `regime` в `config.regime_overrides`.
    2. Если найдено — использовать специфичные пороги.
    3. Если нет — использовать базовые значения `buy_threshold` / `sell_threshold`.

### 5. Интеграция в [./src/bin/run-bot.rs](./src/bin/run-bot.rs)
В основном цикле:
1. `Snapshot` -> `RegimeDetector` -> `current_regime` (с учетом гистерезиса).
2. `Snapshot` -> `InferenceEngine` -> `(Signal, Confidence)`.
3. `(Signal, Confidence, current_regime)` -> `ExecutionEngine`.

## Аргументация (Спор с Grok):
- **Согласен**: Режим **не должен** возвращаться из ONNX. Это внешняя бизнес-логика, которая рассчитывается в Rust на основе параметров из `regime_config.json`.
- **Согласен**: Использование `enum` для `RegimeId` обеспечивает типобезопасность и исключает ошибки при сопоставлении с Python-метками.
- **Согласен**: Алгоритм гистерезиса со счетчиком подтверждений — обязательное условие для стабильности торгового цикла.
- **Важное уточнение**: При загрузке `regime_config.json` в Rust необходимо обеспечить полную идентичность формул расчета признаков (Z-score, OFI) тем, что использовались в [./python_lab/src/dataset.py](./python_lab/src/dataset.py).

**Gemini, реализуй систему, которая сделает бота адаптивным к рыночному контексту.**