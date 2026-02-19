#  АРХИТЕКТУРА NEIROBOT LIT (для каждого токена отдельная подпапка в bots

neirobot-lit/
├── Cargo.toml
├── Cargo.lock
├── README.md
├── global.toml                # глобальные настройки (логи, env, общие параметры)
├── exchange.toml              # настройки биржи (Bybit/Binance: ws url, api keys path, rate limits и т.д.)
├── .env.example               # шаблон для API-ключей и секретов
├── .env					   # из example
│
├── src/
│   ├── main.rs                # Основная точка входа (парсит аргументы, загружает конфиг, запускает бот или выбирает режим: run-bot / dump / test)
│   │
│   ├── bin/                   # несколько точек входа
│   │   ├── run-bot.rs         # Основной бинарник: запуск живого бота (парсит --config bots/XXX/config.toml, инициализирует websocket, inference, trading loop)
│   │   └── dump.rs            # Отдельный бинарник для выгрузки данных (dump_data.sh будет его запускать: --symbol CAKEUSDT --days 7 --output bots/CAKEUSDT/data/raw/)
│   │
│   ├── config/                # парсинг, валидация, merge конфигов
│   │   ├── mod.rs			   # Объявление модуля: pub mod types; pub mod loader; (и pub use types::*;)
│	│	├── types.rs		   # Определения структур конфигов:    # - GlobalConfig (логирование, env, общие параметры),    # - ExchangeConfig (Bybit/Binance), BotConfig (per-token: symbol, model_path, thresholds, risk limits, strategy params)
│   │   └── loader.rs
│   │
│   ├── data/                  # всё, что связано с получением и хранением данных
│ 	│ 	├── mod.rs             # Объявление модуля: pub mod websocket; pub mod orderbook; pub mod snapshot; pub mod dump; (и pub use types::* если есть)
│ 	│ 	├── types.rs           # Общие типы для всего модуля data:    # - Level { price: f64, volume: f64 },  # - OrderBookSide (Bid/Ask ),   # - Update (enum: Snapshot, DeltaAdd, DeltaRemove, etc.),  # - Snapshot.
│   │   ├── websocket.rs       # live stream с биржи
│   │   ├── orderbook.rs       # in-memory стакан + update логика
│   │   ├── snapshot.rs        # формирование LOB-снимка → тензора для модели
│   │   └── dump.rs            # логика записи в parquet
│   │
│   ├── ml/                    # inference only, а не сама модель обучается
│   │   ├── mod.rs
│   │   ├── tensor.rs          # Формирование входного тензора из snapshot (LOB)
│   │   ├── onnx.rs            # ONNX Runtime, inference, error handling
│   │   └── types.rs           # общие типы (Signal, Logits, etc.)
│   │
│   ├── trading/
│   │   ├── mod.rs
│   │   ├── order_manager.rs   # отправка/отмена ордеров, fills handling
│   │   ├── position_manager.rs# учёт позиции и PnL в реальном времени
│   │   └── execution.rs       # логика исполнения на основе сигнала
│   │
│   ├── risk/
│	│	├── mod.rs
│   │   └── risk_manager.rs    # gates, limits, drawdown stop, per-token overrides
│   │
│   ├── utils/
│	│	├── mod.rs
│   │   ├── logger.rs          # структурированные логи
│   │   └── helpers.rs         # мелкие утилиты (time, math, etc.)
│   │
│   └── monitoring/            # пока пустая (метрики, latency tracking)
│
├── python_lab/                # обучение и бектест (пока пустая)
│   ├── requirements.txt       # библиотеки (torch, polars, numpy, scikit-learn, matplotlib, tqdm, pyyaml, optuna, shap, pyarrow, onnxruntime)
│	├── README.md              # как запускать train/backtest
│	├── src/                   # основной код (модули)
│	│   ├── __init__.py
│   │	├── lit_model.py       # архитектура LiT: Transformer с patching, embedding, attention (PyTorch реализация)
│   │	├── dataset.py         # загрузка parquet, preprocessing (event-time snapshots, normalization: z-score, (p-mid)/mid, log(vol))
│   │	├── utils.py           # helpers: metrics (precision, recall, MCC), plotting loss, early stopping callback
│   │	└── types.py           # типы данных (если нужно: Snapshot, Label enums: up/down/flat)
│	├── scripts/               # CLI-скрипты для запуска
│   ├── train.py           # обучение: argparse (--symbol, --data_dir, --epochs, --batch_size, --output_dir)
│   ├── export_onnx.py     # экспорт обученной модели в ONNX (--input_model bots/CAKE/model/lit.h5 --output bots/CAKEUSDT/model/lit.onnx)
│   └── backtest.py        # бектест: симуляция на holdout данных (--symbol, --model_path, metrics: sharpe, drawdown, pnpl)
│
├── bots/                      # экземпляры ботов (каждый токен — своя папка)
│	├── symbols.txt			   # Список монет, по которым будет выгружаться данные, каждый в отдельную папку bots/SYMBOL/
│   ├── CAKEUSDT/
│   │   ├── config.toml		   # per-token настройки (symbol, thresholds, model_path, risk limits, strategy params)
│   │   ├── data/              # raw/processed parquet для CAKEUSDT
│	│	│   ├── raw/           # сырые parquet из dump.rs
│   │   │	└── processed/     # если будешь хранить нормализованные (врятли)
│   │   ├── model/
│   │   │   ├── lit.onnx
│   │   │   └── trt_cache/     # Кэш TensorRT (Engine). ВНИМАНИЕ: Не переносим между разными моделями GPU!
│   │   └── logs/			# структурированные логи именно этого инстанса
│   └── FARTCOINUSDT/ ... (аналогично)
│
├── docs/                      # info/ — там все инфо документы
│   ├── 000-architecture.md     # этот файл — общая архитектура проекта
│   ├── 001-add-new-token.md   # и остальные твои .md файлы
│   └── ...
│
└──  tests/                     # unit + integration тесты
│   	├── integration_test.rs          # Основной файл для общих интеграционных тестов (один большой или несколько маленьких)
│		├── orderbook_integration.rs     # Тесты на полный цикл orderbook: apply delta → get snapshot → consistency
│		├── websocket_mock.rs            # (или в отдельном файле) Моки/симуляции websocket-обновлений для тестов без реальной биржи
│		├── snapshot_to_tensor.rs        # Интеграция data/snapshot.rs + ml/tensor.rs: проверка, что тензор формируется правильно (shape, значения, нормализация)
│		├── risk_gates.rs                # Тесты на risk_manager: check_before_trade блокирует/пропускает по лимитам, update_from_fill работает
│		├── execution_flow.rs            # Полный цикл: mock signal → execution.rs → order_manager place → position update
│		├── config_merge.rs              # Проверка merge global + exchange + bot config (overrides работают, ошибки ловятся)
│		└── common.rs                    # (опционально) Общие хелперы для тестов: mock OrderBook, fake Update, assert tensors equal и т.д.
└── 
