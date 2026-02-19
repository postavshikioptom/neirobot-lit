# Implementation Plan: Dynamic Entry Thresholds

## Overview

This implementation plan breaks down the Dynamic Entry Thresholds feature into discrete, manageable Rust coding tasks. The system will be built incrementally, starting with configuration and state management, then moving to threshold calculation, signal filtering, and finally integration with the existing trading bot architecture.

The implementation follows the existing project structure: `src/config`, `src/data`, `src/ml`, `src/risk`, and `src/trading`.

## Tasks

- [ ] 1. Set up configuration module and validation
  - Create `src/config/dynamic_threshold.rs` with `DynamicThresholdConfig` struct
  - Implement all validation rules (min/max ordering, base within range, non-negative multiplier, positive max_streak)
  - Add configuration loading from config file
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 7.1, 7.2, 7.3, 7.4, 7.5_

  - [ ]* 1.1 Write unit tests for configuration validation
    - Test valid configurations pass validation
    - Test invalid configurations fail with descriptive errors
    - Test boundary values
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [ ]* 1.2 Write property tests for configuration validation
    - **Property 13: Configuration Validation - Min/Max Ordering**
    - **Property 14: Configuration Validation - Base Within Range**
    - **Property 15: Configuration Validation - Non-Negative Multiplier**
    - **Property 16: Configuration Validation - Positive Max Streak**
    - **Validates: Requirements 7.1, 7.2, 7.3, 7.4**

- [ ] 2. Implement state management and persistence
  - Create `src/data/state.rs` with `BotState` struct containing `loss_streak: usize`
  - Implement `StatePersistence` trait in `src/data/persistence.rs`
  - Implement JSON serialization/deserialization for state
  - Handle missing state.json file (initialize loss_streak to 0)
  - _Requirements: 2.1, 3.1, 3.2, 3.3, 3.4_

  - [ ]* 2.1 Write unit tests for state persistence
    - Test state saves to JSON correctly
    - Test state loads from JSON correctly
    - Test missing file initializes to 0
    - Test corrupted JSON handled gracefully
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [ ]* 2.2 Write property tests for state persistence
    - **Property 10: State Persistence Round Trip**
    - **Property 17: Missing State File Initialization**
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4**

- [ ] 3. Implement Risk_Manager with threshold calculation
  - Create `src/risk/dynamic_threshold.rs` with `DynamicThresholdRiskManager` struct
  - Implement `get_effective_threshold(current_streak: usize) -> f64` method
  - Apply formula: `threshold_base + (threshold_loss_mult × min(current_streak, threshold_max_streak))`
  - Implement clamping to `[threshold_min, threshold_max]` range
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 8.1, 8.2, 8.3_

  - [ ]* 3.1 Write unit tests for threshold calculation
    - Test formula applied correctly for various streak values
    - Test clamping at both boundaries
    - Test saturation at max_streak
    - Test monotonic behavior
    - _Requirements: 4.2, 4.3, 4.4, 4.5, 8.1, 8.3_

  - [ ]* 3.2 Write property tests for threshold calculation
    - **Property 4: Threshold Calculation Formula**
    - **Property 5: Threshold Clamping to Maximum**
    - **Property 6: Threshold Clamping to Minimum**
    - **Property 7: Threshold Within Bounds**
    - **Property 8: Monotonic Threshold Increase**
    - **Property 9: Threshold Saturation**
    - **Validates: Requirements 4.2, 4.3, 4.4, 4.5, 8.1, 8.2, 8.3**

- [ ] 4. Implement loss streak update logic
  - Create `src/trading/streak_manager.rs` with loss streak update functions
  - Implement `update_loss_streak(current_streak: usize, pnl: Decimal) -> usize`
  - Handle three cases: negative PnL (increment), positive PnL (reset to 0), zero PnL (maintain)
  - _Requirements: 2.2, 2.3, 2.4_

  - [ ]* 4.1 Write unit tests for streak updates
    - Test negative PnL increments streak
    - Test positive PnL resets streak to 0
    - Test zero PnL maintains streak
    - Test multiple consecutive updates
    - _Requirements: 2.2, 2.3, 2.4_

  - [ ]* 4.2 Write property tests for streak updates
    - **Property 1: Loss Streak Increments on Negative PnL**
    - **Property 2: Loss Streak Resets on Positive PnL**
    - **Property 3: Loss Streak Unchanged on Zero PnL**
    - **Validates: Requirements 2.2, 2.3, 2.4**

- [ ] 5. Integrate Position_Manager with streak updates
  - Modify `src/trading/position_manager.rs` to include `update_streak` method
  - Call `update_loss_streak` when position closes
  - Update `loss_streak` in bot state immediately after calculation
  - Trigger state persistence after updating loss_streak
  - Ensure update_streak is called exactly once per closed trade
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 2.5_

  - [ ]* 5.1 Write unit tests for Position_Manager integration
    - Test update_streak method exists and accepts Decimal
    - Test loss_streak updated in state after position close
    - Test state persistence triggered
    - Test update_streak called exactly once per trade
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ] 6. Implement Execution_Module signal filtering
  - Modify `src/trading/execution.rs` to add signal filtering logic
  - Retrieve current loss_streak from state when signal received
  - Calculate effective threshold using Risk_Manager
  - Compare model confidence against effective threshold
  - Reject signals below threshold, accept signals at or above threshold
  - Log effective threshold and model confidence for each evaluation
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ]* 6.1 Write unit tests for signal filtering
    - Test signals below threshold are rejected
    - Test signals at threshold are accepted
    - Test signals above threshold are accepted
    - Test logging occurs for all evaluations
    - _Requirements: 5.3, 5.4, 5.5_

  - [ ]* 6.2 Write property tests for signal filtering
    - **Property 11: Signal Rejection Below Threshold**
    - **Property 12: Signal Acceptance At or Above Threshold**
    - **Validates: Requirements 5.3, 5.4**

- [ ] 7. Checkpoint - Ensure all tests pass
  - Run all unit tests: `cargo test --lib`
  - Run all property tests: `cargo test --lib -- --test-threads=1`
  - Verify no compilation warnings
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. Integration testing and wiring
  - [ ] 8.1 Create integration test for complete flow
    - Test configuration loading → state initialization → threshold calculation → signal filtering
    - Test loss streak updates → state persistence → reload
    - Test multiple trades with varying PnL
    - _Requirements: 1.1, 2.1, 3.1, 4.1, 5.1, 6.1_

  - [ ]* 8.2 Write end-to-end property test
    - Generate random configurations, trades, and signals
    - Verify entire system maintains correctness properties
    - Test state persistence across simulated restarts
    - _Requirements: All_

- [ ] 9. Final checkpoint - Ensure all tests pass
  - Run full test suite: `cargo test`
  - Verify integration tests pass
  - Verify property tests pass with minimum 100 iterations
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties (minimum 100 iterations each)
- Unit tests validate specific examples and edge cases
- All code should follow Rust best practices and project conventions
- Use `Decimal` type from the project's existing dependencies for PnL calculations
- Ensure all new modules are properly integrated into the project's module hierarchy
