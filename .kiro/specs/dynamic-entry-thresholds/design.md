# Design Document: Dynamic Entry Thresholds

## Overview

The Dynamic Entry Thresholds system implements adaptive risk management for a Rust-based trading bot. It dynamically increases the ML model's confidence threshold for trade entry after consecutive losing trades, making the bot more conservative during periods of market unpredictability.

The system tracks loss streaks, persists state across restarts, calculates effective thresholds based on current performance, and filters trading signals accordingly. This creates a feedback loop where poor performance automatically triggers stricter entry criteria.

## Architecture

### High-Level Flow

```
Trading Signal (with confidence)
    ↓
Execution_Module receives signal
    ↓
Retrieve current loss_streak from state
    ↓
Risk_Manager calculates effective_threshold
    ↓
Compare: confidence >= effective_threshold?
    ├─ YES → Proceed with position opening
    └─ NO → Reject signal
    ↓
Position closes with PnL
    ↓
Position_Manager updates loss_streak
    ↓
State_Persistence saves to state.json
```

### Component Responsibilities

**Configuration Module**:
- Loads and validates threshold parameters
- Provides configuration to Risk_Manager and Position_Manager
- Validates all constraints at startup

**Risk_Manager**:
- Calculates effective threshold based on current loss streak
- Applies formula: `threshold_base + (threshold_loss_mult × min(current_streak, threshold_max_streak))`
- Clamps result to `[threshold_min, threshold_max]` range
- Ensures monotonic behavior

**Position_Manager**:
- Tracks loss streak updates when trades close
- Increments streak on losses, resets on wins, maintains on breakeven
- Triggers state persistence after updates
- Ensures exactly one update per closed trade

**Execution_Module**:
- Retrieves current loss streak from state
- Calls Risk_Manager to get effective threshold
- Compares model confidence against threshold
- Accepts or rejects trading signals
- Logs threshold and confidence for audit trail

**State_Persistence**:
- Serializes/deserializes loss_streak to/from state.json
- Handles missing state file (initializes to 0)
- Maintains consistency across bot restarts

## Components and Interfaces

### Configuration Structure

```rust
pub struct DynamicThresholdConfig {
    pub threshold_base: f64,           // Base confidence threshold
    pub threshold_loss_mult: f64,      // Threshold increase per loss
    pub threshold_max: f64,            // Maximum allowed threshold
    pub threshold_min: f64,            // Minimum allowed threshold
    pub threshold_max_streak: usize,   // Max streak for calculation
}
```

**Validation Rules**:
- `threshold_min <= threshold_max`
- `threshold_min <= threshold_base <= threshold_max`
- `threshold_loss_mult >= 0.0`
- `threshold_max_streak > 0`

### State Structure

```rust
pub struct BotState {
    pub loss_streak: usize,
    // ... other state fields
}
```

### Risk_Manager Interface

```rust
pub trait RiskManager {
    fn get_effective_threshold(&self, current_streak: usize) -> f64;
}

impl RiskManager for DynamicThresholdRiskManager {
    fn get_effective_threshold(&self, current_streak: usize) -> f64 {
        let streak_capped = current_streak.min(self.config.threshold_max_streak);
        let calculated = self.config.threshold_base 
            + (self.config.threshold_loss_mult * streak_capped as f64);
        calculated.clamp(self.config.threshold_min, self.config.threshold_max)
    }
}
```

### Position_Manager Interface

```rust
pub trait PositionManager {
    fn update_streak(&mut self, trade_pnl: Decimal) -> Result<(), Error>;
}

impl PositionManager for DefaultPositionManager {
    fn update_streak(&mut self, trade_pnl: Decimal) -> Result<(), Error> {
        if trade_pnl < Decimal::ZERO {
            self.state.loss_streak += 1;
        } else if trade_pnl > Decimal::ZERO {
            self.state.loss_streak = 0;
        }
        // else: maintain current streak for zero PnL
        
        self.persist_state()?;
        Ok(())
    }
}
```

### Execution_Module Interface

```rust
pub trait ExecutionModule {
    fn evaluate_signal(&self, signal: TradingSignal) -> Result<SignalDecision, Error>;
}

pub enum SignalDecision {
    Accept { effective_threshold: f64 },
    Reject { reason: String, effective_threshold: f64 },
}
```

### State_Persistence Interface

```rust
pub trait StatePersistence {
    fn save_state(&self, state: &BotState) -> Result<(), Error>;
    fn load_state(&self) -> Result<BotState, Error>;
}

impl StatePersistence for JsonStatePersistence {
    fn save_state(&self, state: &BotState) -> Result<(), Error> {
        let json = serde_json::to_string(state)?;
        std::fs::write("state.json", json)?;
        Ok(())
    }
    
    fn load_state(&self) -> Result<BotState, Error> {
        match std::fs::read_to_string("state.json") {
            Ok(json) => serde_json::from_str(&json),
            Err(_) => Ok(BotState { loss_streak: 0, .. }),
        }
    }
}
```

## Data Models

### Configuration Validation

```rust
impl DynamicThresholdConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.threshold_min > self.threshold_max {
            return Err(ConfigError::InvalidRange(
                "threshold_min must be <= threshold_max"
            ));
        }
        
        if self.threshold_base < self.threshold_min 
            || self.threshold_base > self.threshold_max {
            return Err(ConfigError::OutOfRange(
                "threshold_base must be within [threshold_min, threshold_max]"
            ));
        }
        
        if self.threshold_loss_mult < 0.0 {
            return Err(ConfigError::NegativeValue(
                "threshold_loss_mult must be non-negative"
            ));
        }
        
        if self.threshold_max_streak == 0 {
            return Err(ConfigError::InvalidValue(
                "threshold_max_streak must be greater than zero"
            ));
        }
        
        Ok(())
    }
}
```

### Loss Streak Update Logic

```rust
pub fn update_loss_streak(current_streak: usize, pnl: Decimal) -> usize {
    if pnl < Decimal::ZERO {
        current_streak + 1
    } else if pnl > Decimal::ZERO {
        0
    } else {
        current_streak
    }
}
```

### Threshold Calculation

```rust
pub fn calculate_effective_threshold(
    config: &DynamicThresholdConfig,
    current_streak: usize,
) -> f64 {
    let streak_capped = current_streak.min(config.threshold_max_streak);
    let calculated = config.threshold_base 
        + (config.threshold_loss_mult * streak_capped as f64);
    calculated.clamp(config.threshold_min, config.threshold_max)
}
```

## Correctness Properties

A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.

### Property 1: Loss Streak Increments on Negative PnL

*For any* initial loss streak value and any trade with negative PnL, updating the streak should result in exactly one increment.

**Validates: Requirements 2.2**

### Property 2: Loss Streak Resets on Positive PnL

*For any* initial loss streak value and any trade with positive PnL, updating the streak should result in zero.

**Validates: Requirements 2.3**

### Property 3: Loss Streak Unchanged on Zero PnL

*For any* initial loss streak value and any trade with zero PnL, updating the streak should leave it unchanged.

**Validates: Requirements 2.4**

### Property 4: Threshold Calculation Formula

*For any* valid configuration and loss streak value, the calculated effective threshold should equal `threshold_base + (threshold_loss_mult × min(current_streak, threshold_max_streak))` before clamping.

**Validates: Requirements 4.2**

### Property 5: Threshold Clamping to Maximum

*For any* valid configuration where the formula result exceeds `threshold_max`, the returned threshold should equal `threshold_max`.

**Validates: Requirements 4.3**

### Property 6: Threshold Clamping to Minimum

*For any* valid configuration where the formula result is below `threshold_min`, the returned threshold should equal `threshold_min`.

**Validates: Requirements 4.4**

### Property 7: Threshold Within Bounds

*For any* valid configuration and loss streak value, the returned effective threshold should always be within `[threshold_min, threshold_max]`.

**Validates: Requirements 4.5**

### Property 8: Monotonic Threshold Increase

*For any* valid configuration and two loss streak values n and n+1, `get_effective_threshold(n) <= get_effective_threshold(n+1)`.

**Validates: Requirements 8.1, 8.2**

### Property 9: Threshold Saturation

*For any* valid configuration, when `current_streak >= threshold_max_streak`, the effective threshold should equal `threshold_max`.

**Validates: Requirements 8.3**

### Property 10: State Persistence Round Trip

*For any* valid loss streak value, serializing to state.json and then deserializing should produce the same value.

**Validates: Requirements 3.1, 3.2, 3.4**

### Property 11: Signal Rejection Below Threshold

*For any* trading signal with confidence below the effective threshold, the Execution_Module should reject the signal and not proceed with position opening.

**Validates: Requirements 5.3**

### Property 12: Signal Acceptance At or Above Threshold

*For any* trading signal with confidence at or above the effective threshold, the Execution_Module should proceed with position opening logic.

**Validates: Requirements 5.4**

### Property 13: Configuration Validation - Min/Max Ordering

*For any* configuration where `threshold_min > threshold_max`, validation should fail with a descriptive error.

**Validates: Requirements 7.1**

### Property 14: Configuration Validation - Base Within Range

*For any* configuration where `threshold_base` is outside `[threshold_min, threshold_max]`, validation should fail with a descriptive error.

**Validates: Requirements 7.2**

### Property 15: Configuration Validation - Non-Negative Multiplier

*For any* configuration where `threshold_loss_mult < 0.0`, validation should fail with a descriptive error.

**Validates: Requirements 7.3**

### Property 16: Configuration Validation - Positive Max Streak

*For any* configuration where `threshold_max_streak == 0`, validation should fail with a descriptive error.

**Validates: Requirements 7.4**

### Property 17: Missing State File Initialization

*When* state.json does not exist or is missing the `loss_streak` field, loading state should initialize `loss_streak` to zero.

**Validates: Requirements 3.3**

## Error Handling

### Configuration Errors

```rust
pub enum ConfigError {
    InvalidRange(String),      // threshold_min > threshold_max
    OutOfRange(String),        // threshold_base outside bounds
    NegativeValue(String),     // threshold_loss_mult < 0
    InvalidValue(String),      // threshold_max_streak == 0
    ParseError(String),        // JSON parsing failed
}
```

### State Persistence Errors

```rust
pub enum PersistenceError {
    IoError(String),           // File I/O failed
    SerializationError(String), // JSON serialization failed
    DeserializationError(String), // JSON deserialization failed
}
```

### Execution Errors

```rust
pub enum ExecutionError {
    StateRetrievalFailed(String),
    ThresholdCalculationFailed(String),
    SignalProcessingFailed(String),
}
```

### Recovery Strategies

- **Configuration validation failure**: Bot refuses to start with descriptive error message
- **State file missing**: Initialize loss_streak to 0 and continue
- **State file corrupted**: Log warning, initialize to 0, continue
- **Threshold calculation failure**: Log error, use threshold_max as fallback
- **Signal evaluation failure**: Reject signal, log error, continue

## Testing Strategy

### Unit Testing Approach

Unit tests verify specific examples, edge cases, and error conditions:

1. **Configuration Validation Tests**
   - Valid configurations pass validation
   - Invalid configurations fail with descriptive errors
   - Boundary values (min=max, base at boundaries) are handled correctly

2. **Loss Streak Update Tests**
   - Negative PnL increments streak
   - Positive PnL resets streak to 0
   - Zero PnL maintains streak
   - Multiple consecutive updates work correctly

3. **Threshold Calculation Tests**
   - Formula applied correctly for various streak values
   - Clamping works at both boundaries
   - Saturation occurs at max_streak

4. **State Persistence Tests**
   - State saves to JSON correctly
   - State loads from JSON correctly
   - Missing file initializes to 0
   - Corrupted JSON handled gracefully

5. **Signal Filtering Tests**
   - Signals below threshold are rejected
   - Signals at threshold are accepted
   - Signals above threshold are accepted
   - Logging occurs for all evaluations

### Property-Based Testing Approach

Property tests verify universal properties across many generated inputs:

1. **Threshold Monotonicity** (Property 8)
   - Generate random valid configurations
   - Generate random streak values
   - Verify threshold never decreases with increasing streak

2. **Threshold Bounds** (Property 7)
   - Generate random valid configurations
   - Generate random streak values
   - Verify returned threshold always within [min, max]

3. **Streak Update Correctness** (Properties 1-3)
   - Generate random initial streaks
   - Generate random PnL values (negative, positive, zero)
   - Verify streak updates according to rules

4. **State Round Trip** (Property 10)
   - Generate random loss streak values
   - Serialize and deserialize
   - Verify values match

5. **Configuration Validation** (Properties 13-16)
   - Generate random invalid configurations
   - Verify validation fails appropriately
   - Verify error messages are descriptive

### Test Configuration

- **Minimum iterations per property test**: 100
- **Test framework**: `proptest` or `quickcheck` for Rust
- **Tag format**: `Feature: dynamic-entry-thresholds, Property {N}: {description}`
- **Coverage goal**: All acceptance criteria covered by unit or property tests

### Testing Balance

- **Unit tests**: ~40% of test suite (specific examples, edge cases)
- **Property tests**: ~60% of test suite (universal properties, comprehensive coverage)
- Both are complementary and necessary for comprehensive coverage
