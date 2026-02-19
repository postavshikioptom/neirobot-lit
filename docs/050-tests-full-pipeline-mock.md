Задача 050-tests-full-pipeline-mock.md
Цель: Реализовать сквозной интеграционный тест торгового цикла, проверяющий взаимодействие всех систем (OrderBook -> ML -> Risk -> Execution) в позитивных и негативных сценариях.

Инструкции для Gemini:
1. Подготовка структуры (Refactoring)
В ./src/bin/run-bot.rs вынесите основной цикл обработки сообщений в отдельную функцию:
pub async fn run_bot_loop<S, I, R, O>(
    mut stream: S, 
    mut orderbook: OrderBook,
    mut inference: I,
    mut risk: R,
    mut order_manager: O
) where S: Stream<Item = String> + Unpin ...
Это позволит в тесте подставить вместо реального WebSocket мок-стрим (tokio_stream::iter).
2. Создание Моков (в папке tests/common/)
MockInference: Простая структура с методом predict, которая возвращает сигнал на основе внутреннего флага (устанавливается в тесте: set_next_signal(Signal::Up)).
MockOrderManager: Фиксирует все входящие запросы на ордера в Vec, чтобы мы могли проверить их количество и параметры в конце теста.
3. Реализация теста ./tests/execution_flow.rs
Создать 3 сценария (Test Cases):

Scenario A (Success Buy):
Input: Snapshot JSON.
Mock ML: Signal::Up.
Risk: Allow.
Assert: В MockOrderManager появился 1 Buy-ордер.
Scenario B (Risk Block):
Input: Snapshot JSON.
Mock ML: Signal::Up.
Risk: Deny (например, имитация превышения лимита позиции).
Assert: В MockOrderManager пусто (ордер заблокирован).
Scenario C (Flat/No Action):
Input: Snapshot JSON.
Mock ML: Signal::Flat.
Assert: В MockOrderManager пусто.
4. Данные для теста
Не генерируйте случайные данные. Используйте одну и ту же константную строку JSON из документации Bybit или логов, чтобы стакан инициализировался предсказуемо.
5. Риск-гейты
В тесте инициализируйте RiskManager с жесткими лимитами (например, max_position = 0), чтобы гарантированно проверить сценарий блокировки.
Аргументация: Этот подход делает тест "белым ящиком" — мы точно знаем, что подали на вход и что должны получить на выходе. Вынос run_bot_loop в отдельную функцию — это стандартная практика для написания тестируемого кода на Rust (Hexagonal Architecture lite).