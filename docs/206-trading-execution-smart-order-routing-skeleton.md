# Задача 206: Каркас системы Smart Order Routing (SOR Skeleton)

## 1. Цель задачи
Реализовать интеллектуальный слой управления исполнением (SOR), который выбирает оптимальную тактику (Maker/Taker/Slice) на основе силы сигнала, состояния стакана и установленных риск-лимитов.

## 2. Инструкции для Gemini по реализации

### А. Изменения в Rust-ядре (Execution & Config)

1.  **Конфигурация ([./src/config/types.rs](./src/config/types.rs))**:
    *   Добавить секцию `SorConfig` в `BotConfig`:
        ```rust
        struct SorConfig {
            critical_signal: f32,    // Порог для перехода в Aggressive (Taker)
            max_size_ratio: f64,     // % от объема уровня, выше которого включается Slicing
            default_urgency: f32,    // Базовая агрессивность (0.0 - 1.0)
            slice_interval_ms: u64,  // Пауза между частями TWAP
        }
        ```

2.  **Новые абстракции ([./src/trading/execution.rs](./src/trading/execution.rs))**:
    *   **`ExecutionStrategy`**: Enum с вариантами `Passive` (Limit), `Aggressive` (Market/Cross), `TwapSlice { slices: u32, interval_ms: u64 }`.
    *   **`ExecutionInstruction`**: Структура без `exchange_id` (избегаем `String` аллокаций в Hot Path). Использовать только необходимые поля: `strategy`, `price`, `quantity`, `urgency`.

3.  **Логика выбора тактики (Decision Engine)**:
    *   Реализовать функцию `select_strategy`, принимающую `Signal`, `OrderBook` и `SorConfig`.
    *   **Логика**:
        *   Если `signal.strength > config.critical_signal` → **Aggressive**.
        *   Если `order_size > level_total_vol * config.max_size_ratio` → **TwapSlice**.
        *   В остальных случаях → **Passive**.

### Б. Изменения в Python Lab (Optimization)

1.  **Скрипт оптимизатора**:
    *   Путь: **[./python_lab/scripts/execution_optimizer.py](./python_lab/scripts/execution_optimizer.py)**.
    *   **Функционал**: Анализ исторических логов из задач 201–204 для подбора оптимальных значений `critical_signal` и `max_size_ratio`.
    *   **Выход**: Рекомендованные параметры для `config.toml` конкретного бота.

## 3. Аргументация и правки (По следам Grok)
*   **Спор об `exchange_id`**: Полностью согласен. Проект сейчас работает с одной биржей (Bybit). Введение `String` или даже `Enum` для биржи на данном этапе — это преждевременная оптимизация, которая только замусорит код и создаст лишние аллокации.
*   **Спор о структуре папок**: Скрипт перенесен в `scripts/`, так как это инструмент исследователя, а не часть обучаемого пайплайна модели.
*   **Спор о параметрах Slicing**: Параметры `slices` и `interval` теперь динамические и передаются в `ExecutionInstruction`, что позволяет SOR менять частоту дробления в зависимости от волатильности.

## 4. Ожидаемый результат
В логах `trades.csv` должна появиться колонка `strategy`, показывающая, какая тактика была выбрана. Бот должен начать эффективно «проедать» стакан частями, если объем сигнала превышает ликвидность уровня.

---
**Gemini, при реализации в `execution.rs`: убедись, что логика SOR не содержит блокирующих вызовов и работает исключительно с константными ссылками на `OrderBook` и `Config`.**