# Neirobot LiT

Высокочастотный торговый бот для криптовалют с использованием ML-модели LiT (Lightweight Transformer).

## Сборка с оптимизациями

Для сборки с максимальной производительностью используйте:

```bash
RUSTFLAGS="-C target-cpu=native -C link-arg=-fuse-ld=lld" cargo build --release
```

### Описание флагов

- `-C target-cpu=native` - использование всех инструкций текущего CPU (AVX/AVX-512)
- `-C link-arg=-fuse-ld=lld` - использование быстрого линкера lld

### Профиль Release

В `Cargo.toml` настроены оптимизации:
- `opt-level = 3` - полная оптимизация скорости
- `lto = "fat"` - Link-Time Optimization по всему графу зависимостей
- `codegen-units = 1` - генерация кода в один поток для лучшей оптимизации
- `panic = "abort"` - отключение раскрутки стека для уменьшения размера бинарника
- `strip = true` - удаление отладочных символов (уменьшение размера на 20-50%)

## Использование

```bash
# Запуск бота для конкретного символа
cargo run --release --bin neirobot-lit -- --config bots/BTCUSDT/config.toml

# Выгрузка данных
cargo run --release --bin crypto_tool -- --symbol BTCUSDT --days 7
```
