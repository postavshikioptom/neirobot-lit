# Задача №196: Оптимизация производительности: Тюнинг потоков ONNX Runtime (Intra/Inter-op)

[./docs/196-performance-onnx-intra-op-threading-config.md](./docs/196-performance-onnx-intra-op-threading-config.md)

## План реализации для Gemini

1.  **Изменения в [./src/config/types.rs](./src/config/types.rs)**:
    - Создать перечисление `OnnxExecutionMode` с вариантами `Sequential` и `Parallel`.
    - Добавить структуру `OnnxConfig` в `BotConfig` со следующими полями:
        - **intra_threads**: `u32` (по умолчанию `1` для Windows).
        - **inter_threads**: `u32` (по умолчанию `1`).
        - **execution_mode**: `OnnxExecutionMode` (по умолчанию `Sequential`).

2.  **Обновление зависимостей**:
    - Убедиться, что используется `ort = "2.0"` (или актуальная мажорная версия) в `Cargo.toml`.

3.  **Изменения в [./src/ml/onnx.rs](./src/ml/onnx.rs)**:
    - При инициализации `SessionBuilder` использовать параметры из `OnnxConfig`:
        - `.with_intra_threads(config.intra_threads)?`
        - `.with_inter_threads(config.inter_threads)?`
        - `.with_execution_mode(config.execution_mode.into())?` (реализовать конвертацию в типы `ort`).
    - Убрать любые попытки установки приоритетов потоков (priority) через `ort`, так как это не поддерживается библиотекой напрямую.

4.  **Оптимизация под Windows**:
    - Установить значения по умолчанию `intra_threads = 1` и `execution_mode = Sequential`. Это минимизирует переключения контекста и накладные расходы на синхронизацию потоков для небольших моделей, критичных к задержкам (latency-sensitive).

5.  **Валидация**:
    - Добавить логирование параметров сессии при загрузке модели: `[ML] ONNX Runtime configured: intra={}, inter={}, mode={:?}`.

## Ожидаемый результат
- Снижение задержки инференса на Windows за счет устранения избыточной конкуренции потоков.
- Полный контроль над ресурсами CPU, выделяемыми под ML-инференс, через `bot.toml`.
- Код соответствует актуальному API крейта `ort`.