# Requirements Document

## Introduction

Данная функциональность реализует механизм адаптивного управления рисками для торгового бота на Rust. Система динамически повышает порог уверенности ML модели для входа в сделку после серии убыточных сделок, делая бота более консервативным в периоды непредсказуемого поведения рынка.

## Glossary

- **Trading_Bot**: Торговый бот на Rust, использующий ML модель для генерации торговых сигналов
- **ML_Model**: Машинное обучение модель в формате ONNX, генерирующая торговые сигналы с уровнем уверенности
- **Risk_Manager**: Модуль управления рисками
- **Position_Manager**: Модуль управления позициями
- **Execution_Module**: Модуль исполнения ордеров
- **Confidence_Threshold**: Порог уверенности модели для принятия торгового решения
- **Loss_Streak**: Серия убыточных сделок подряд
- **PnL**: Profit and Loss - прибыль или убыток от сделки
- **State_Persistence**: Сохранение состояния в файл state.json для восстановления после перезапуска

## Requirements

### Requirement 1: Configuration Parameters

**User Story:** Как администратор бота, я хочу настраивать параметры динамических порогов через конфигурацию, чтобы адаптировать поведение бота под разные рыночные условия.

#### Acceptance Criteria

1. THE Configuration SHALL include threshold_base parameter of type f64 representing the base confidence threshold
2. THE Configuration SHALL include threshold_loss_mult parameter of type f64 representing the threshold increase per loss
3. THE Configuration SHALL include threshold_max parameter of type f64 representing the maximum allowed threshold
4. THE Configuration SHALL include threshold_min parameter of type f64 representing the minimum allowed threshold
5. THE Configuration SHALL include threshold_max_streak parameter of type usize representing the maximum loss streak count for calculation

### Requirement 2: Loss Streak Tracking

**User Story:** Как торговый бот, я хочу отслеживать серию убыточных сделок, чтобы адаптировать свое поведение на основе недавней производительности.

#### Acceptance Criteria

1. THE Trading_Bot SHALL maintain a loss_streak counter of type usize in its state
2. WHEN a trade closes with PnL less than zero, THE Position_Manager SHALL increment loss_streak by one
3. WHEN a trade closes with PnL greater than zero, THE Position_Manager SHALL reset loss_streak to zero
4. WHEN a trade closes with PnL equal to zero, THE Position_Manager SHALL maintain the current loss_streak value unchanged
5. THE Trading_Bot SHALL persist loss_streak value to state.json file

### Requirement 3: State Persistence

**User Story:** Как торговый бот, я хочу сохранять серию убытков между перезапусками, чтобы поддерживать консистентное поведение управления рисками.

#### Acceptance Criteria

1. WHEN the Trading_Bot saves state, THE State_Persistence SHALL serialize loss_streak to state.json
2. WHEN the Trading_Bot starts, THE State_Persistence SHALL deserialize loss_streak from state.json
3. IF state.json does not exist or loss_streak is missing, THEN THE Trading_Bot SHALL initialize loss_streak to zero
4. THE State_Persistence SHALL maintain loss_streak value across bot restarts

### Requirement 4: Dynamic Threshold Calculation

**User Story:** Как модуль управления рисками, я хочу рассчитывать эффективный порог на основе текущей серии убытков, чтобы делать бота более консервативным после неудачных сделок.

#### Acceptance Criteria

1. THE Risk_Manager SHALL provide a method get_effective_threshold that accepts current_streak of type usize and returns f64
2. THE Risk_Manager SHALL calculate effective threshold using formula: threshold_base + (threshold_loss_mult × min(current_streak, threshold_max_streak))
3. WHEN calculated threshold exceeds threshold_max, THE Risk_Manager SHALL return threshold_max
4. WHEN calculated threshold is below threshold_min, THE Risk_Manager SHALL return threshold_min
5. THE Risk_Manager SHALL ensure the returned threshold is always within the range [threshold_min, threshold_max]

### Requirement 5: Signal Filtering

**User Story:** Как модуль исполнения, я хочу фильтровать торговые сигналы на основе динамического порога, чтобы открывать позиции только при достаточной уверенности модели.

#### Acceptance Criteria

1. WHEN the Execution_Module receives a trading signal, THE Execution_Module SHALL retrieve the current loss_streak from state
2. WHEN processing a signal, THE Execution_Module SHALL calculate the effective threshold using Risk_Manager
3. WHEN model confidence is less than the effective threshold, THE Execution_Module SHALL reject the signal and not open a position
4. WHEN model confidence is greater than or equal to the effective threshold, THE Execution_Module SHALL proceed with position opening logic
5. THE Execution_Module SHALL log the effective threshold and model confidence for each signal evaluation

### Requirement 6: Streak Update Integration

**User Story:** Как менеджер позиций, я хочу обновлять серию убытков при закрытии каждой сделки, чтобы система управления рисками имела актуальную информацию.

#### Acceptance Criteria

1. THE Position_Manager SHALL provide an update_streak method that accepts trade_pnl of type Decimal
2. WHEN a position is closed, THE Position_Manager SHALL call update_streak with the final PnL value
3. THE Position_Manager SHALL update the loss_streak in the bot state immediately after calculation
4. THE Position_Manager SHALL trigger state persistence after updating loss_streak
5. THE Position_Manager SHALL ensure update_streak is called exactly once per closed trade

### Requirement 7: Configuration Validation

**User Story:** Как администратор бота, я хочу получать ошибки при некорректной конфигурации порогов, чтобы избежать неправильного поведения бота.

#### Acceptance Criteria

1. WHEN loading configuration, THE Trading_Bot SHALL validate that threshold_min is less than or equal to threshold_max
2. WHEN loading configuration, THE Trading_Bot SHALL validate that threshold_base is within the range [threshold_min, threshold_max]
3. WHEN loading configuration, THE Trading_Bot SHALL validate that threshold_loss_mult is non-negative
4. WHEN loading configuration, THE Trading_Bot SHALL validate that threshold_max_streak is greater than zero
5. IF any validation fails, THEN THE Trading_Bot SHALL return a descriptive error and refuse to start

### Requirement 8: Monotonic Behavior

**User Story:** Как торговый бот, я хочу чтобы порог уверенности монотонно возрастал с увеличением серии убытков, чтобы обеспечить предсказуемое поведение управления рисками.

#### Acceptance Criteria

1. FOR ALL valid loss_streak values, THE Risk_Manager SHALL ensure that get_effective_threshold(n) is less than or equal to get_effective_threshold(n+1)
2. THE Risk_Manager SHALL ensure that increasing loss_streak never decreases the effective threshold
3. THE Risk_Manager SHALL ensure that the effective threshold reaches threshold_max when loss_streak equals or exceeds threshold_max_streak
