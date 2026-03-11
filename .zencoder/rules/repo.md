---
description: Repository Information Overview
alwaysApply: true
---

# Neirobot LiT Information

## Summary
The project is a high-frequency trading bot for cryptocurrency (Bybit) using the Limit Order Book Transformer (LiT) architecture. It consists of a Rust core for live trading and inference, and a Python-based ML lab for training and data analysis.

## Structure
- **src/**: Rust source code for the trading engine, order management, and ONNX inference.
- **python_lab/**: Python scripts for model training, dataset preparation, and experimentation.
- **docs/**: Comprehensive documentation of architecture, tasks, and data schemas.
- **bots/**: (Symbol-specific) Configuration, data, and model files (e.g., `bots/BTCUSDT/`).
- **tests/**: Integration and unit tests for both Rust and Python components.
- **deploy/**: Deployment scripts and guides (Systemd, Watchdog).

## Language & Runtime
**Languages**: Rust, Python  
**Rust Version**: 1.75+ (Alpine musl target in Docker)  
**Python Version**: 3.10+  
**Build System**: Cargo (Rust), Pip/Requirements (Python)  
**Package Managers**: cargo, pip

## Dependencies
**Main Dependencies**:
- **Rust**: `ort` (ONNX Runtime), `serde`, `tokio`, `polars`, `anyhow`, `tracing`.
- **Python**: `torch`, `polars`, `onnxruntime`, `pyarrow`, `optuna`, `psutil`.

## Build & Installation
```bash
# Rust Build
cargo build --release

# Python Environment
pip install -r python_lab/requirements.txt
```

## Docker
**Dockerfile**: `./Dockerfile` (Multi-stage build)
**Base Image**: `rust:1.75-alpine` (builder), `alpine:3.19` (runtime)
**Configuration**: Static linking for musl target, non-root user `neirobot`, mount points for `/app/bots`.

## Testing
**Frameworks**: `cargo test` (Rust), `pytest` (Python).
**Naming Convention**: `*_tests.rs`, `test_*.py`.
**Run Command**:
```bash
cargo test
python -m pytest python_lab/tests/
```

## Main Files & Resources
- **Rust Entrypoint**: `src/bin/run-bot.rs`, `src/bin/dump.rs`.
- **Python Entrypoint**: `python_lab/src/train.py`.
- **Data Schema**: `docs/012-data-schema-definition.md`.
- **Task List**: `docs/000-tasks_list.md`.
