# 001 - Architecture Final Verification

**Цель задачи:**  
Сверить и привести текущую структуру проекта в полное соответствие с 000-architecture.md. Убедиться в полной изоляции каждого токена в `bots/SYMBOL/`. Настроить Cargo.toml с минимальным, но достаточным набором зависимостей для всех последующих фаз (включая ML-инференс и работу с Parquet). Создать все директории, пустые модули и примерную папку одного бота для проверки изоляции.

**Предполагаемое состояние проекта на старте:** проект только создан (`cargo new neirobot-lit --bin`) или почти пустой.

**Что нужно сделать:**

1. **Создать/проверить полную файловую структуру**

   Создать все недостающие директории и пустые файлы-заглушки точно по схеме из 000-architecture.md:

   ```
   neirobot-lit/
   ├── Cargo.toml
   ├── .gitignore
   ├── global.toml                  # пустой или с комментариями
   ├── exchange.toml                # пустой или с комментариями
   ├── .env.example                 # шаблон с API_KEY=..., API_SECRET=...
   ├── src/
   │   ├── main.rs                  # заглушка (см. ниже)
   │   ├── bin/
   │   │   ├── run-bot.rs           # пустой main()
   │   │   └── dump.rs              # пустой main()
   │   ├── config/
   │   │   ├── mod.rs               # pub mod types; pub mod loader;
   │   │   ├── types.rs             # пустые struct'ы
   │   │   └── loader.rs
   │   ├── data/
   │   │   ├── mod.rs
   │   │   ├── types.rs
   │   │   ├── websocket.rs
   │   │   ├── orderbook.rs
   │   │   ├── snapshot.rs
   │   │   └── dump.rs
   │   ├── ml/
   │   │   ├── mod.rs
   │   │   ├── tensor.rs
   │   │   ├── onnx.rs
   │   │   └── types.rs
   │   ├── trading/
   │   │   ├── mod.rs
   │   │   ├── order_manager.rs
   │   │   ├── position_manager.rs
   │   │   └── execution.rs
   │   ├── risk/
   │   │   ├── mod.rs
   │   │   └── risk_manager.rs
   │   ├── utils/
   │   │   ├── mod.rs
   │   │   ├── logger.rs
   │   │   └── helpers.rs
   │   └── monitoring/
   │       └── mod.rs               # пока пустой
   ├── python_lab/
   │   ├── requirements.txt         # список библиотек (см. ниже)
   │   ├── README.md                # пустой или с заголовком
   │   ├── src/
   │   │   ├── __init__.py
   │   │   ├── lit_model.py
   │   │   ├── dataset.py
   │   │   ├── utils.py
   │   │   └── types.py
   │   └── scripts/                 # или просто в корне python_lab, но лучше отдельно
   │       ├── train.py
   │       ├── export_onnx.py
   │       └── backtest.py
   ├── bots/
   │   ├── symbols.txt              # пример: CAKEUSDT, FARTCOINUSDT
   │   └── CAKEUSDT/                # пример бота для верификации изоляции
   │       ├── config.toml          # пустой или минимальный шаблон
   │       ├── data/
   │       │   └── raw/             # пустая папка для будущих parquet
   │       ├── model/               # пустая, позже lit.onnx
   │       └── logs/                # пустая
   ├── docs/
   │   └── 000-architecture.md      # уже есть
   └── tests/
       ├── integration_test.rs
       └── common.rs                # опционально, пустой
   ```

2. **Настроить Cargo.toml**

   Добавить в `[dependencies]` актуальные версии (проверено на crates.io по состоянию на февраль 2026):

   ```toml
   [package]
   name = "neirobot-lit"
   version = "0.1.0"
   edition = "2021"

   [dependencies]
   anyhow = "1.0"
   serde = { version = "1.0", features = ["derive"] }
   toml = "0.8"
   tracing = "0.1"
   tracing-subscriber = { version = "0.3", features = ["env-filter", "fmt"] }
   clap = { version = "4.5", features = ["derive"] }
   tokio = { version = "1.40", features = ["full"] }
   tungstenite = { version = "0.24", features = ["native-tls"] }
   url = "2.5"
   dotenvy = "0.15"
   ort = "2.0.0-rc.9"                    # ONNX Runtime для инференса (обязательно на раннем этапе)
   polars = { version = "0.42", features = ["parquet", "lazy"] }  # для dump.rs и чтения данных
   ```

   В `[dev-dependencies]` добавить:
   ```toml
   tokio-test = "0.4"
   ```

3. **Создать .gitignore**

   ```gitignore
   target/
   **/Cargo.lock
   .env
   bots/*/logs/
   bots/*/data/
   python_lab/__pycache__/
   python_lab/*.pyc
   *.log
   ```

4. **Инициализировать python_lab/requirements.txt**

   ```txt
   torch
   polars
   numpy
   scikit-learn
   matplotlib
   tqdm
   pyyaml
   optuna
   shap
   pyarrow
   onnxruntime
   ```

5. **Добавить заглушку в src/main.rs**

   ```rust
   use anyhow::Result;

   #[tokio::main]
   async fn main() -> Result<()> {
       println!("Neirobot LIT — проект инициализирован");
       // TODO: в будущем — парсинг CLI-аргументов и выбор режима
       Ok(())
   }
   ```

6. **Проверить изоляцию ботов**

   - Создать тестовую папку `bots/CAKEUSDT/` с подпапками `data/raw/`, `model/`, `logs/` и пустым `config.toml`.
   - Убедиться, что пути в будущем коде будут строиться относительно этой папки (например, `--config bots/CAKEUSDT/config.toml`).

**Ожидаемый результат:**

- `cargo check` проходит без ошибок.
- Структура проекта на 100% соответствует 000-architecture.md.
- Все ключевые зависимости (включая ort и polars) добавлены сразу, чтобы избежать проблем в Phase 3–4.
- Есть пример бота CAKEUSDT для визуальной проверки изоляции.
- Python_lab готов к началу Phase 2.

**Следующая задача:** 002-global-toml-template.md

Эта версия учитывает все замечания Claude: добавлены ort и polars сразу, детализирована структура bots/SYMBOL/, расширен python_lab, уточнены версии. Готов к следующей задаче или к правкам, если нужно что-то скорректировать.