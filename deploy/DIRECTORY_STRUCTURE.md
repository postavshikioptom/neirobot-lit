# Структура директорий Neirobot LiT

## Обзор

```
neirobot-lit/
├── deploy/                          # Инструменты развертывания
│   ├── manage.sh                    # Главный скрипт управления
│   ├── bot.template.toml            # Шаблон конфигурации
│   ├── neirobot-lit@.service        # Systemd unit template
│   ├── docker-compose.yml           # Docker Compose конфигурация
│   ├── README.md                    # Документация
│   ├── QUICKSTART.md                # Быстрый старт
│   ├── EXAMPLES.md                  # Примеры использования
│   └── .gitignore                   # Игнорирование временных файлов
├── bots/                            # Директория ботов
│   ├── BTCUSDT/                     # Бот для BTCUSDT
│   │   ├── config.toml              # Конфигурация бота
│   │   ├── model/
│   │   │   └── model.onnx           # ONNX модель
│   │   ├── data/
│   │   │   └── raw/                 # Сырые данные
│   │   └── logs/                    # Логи бота
│   ├── ETHUSDT/                     # Бот для ETHUSDT
│   │   └── ...
│   └── symbols.txt                  # Список активных символов
├── python_lab/                      # ML лаборатория
│   ├── models/
│   │   ├── BTCUSDT.onnx             # Обученная модель
│   │   ├── ETHUSDT.onnx
│   │   └── metadata.json            # Метаданные моделей
│   ├── src/
│   │   ├── train.py                 # Скрипт обучения
│   │   ├── export_onnx.py           # Экспорт в ONNX
│   │   └── ...
│   └── requirements.txt             # Python зависимости
├── src/                             # Rust исходный код
│   ├── main.rs                      # Точка входа
│   ├── lib.rs
│   ├── config/                      # Конфигурация
│   ├── data/                        # Работа с данными
│   ├── ml/                          # ML компоненты
│   ├── trading/                     # Торговая логика
│   └── ...
├── tests/                           # Тесты
│   ├── deploy_script_test.sh        # Тесты развертывания
│   └── ...
├── Dockerfile                       # Docker образ
├── Cargo.toml                       # Rust зависимости
├── global.toml                      # Глобальные настройки
├── exchange.toml                    # Настройки биржи
├── .env                             # API ключи (не коммитить!)
└── .gitignore                       # Git игнорирование
```

## Native режим (systemd)

### Структура на сервере

```
/usr/bin/
└── neirobot-lit                     # Скомпилированный бинарник

/etc/neirobot/
└── bots/
    ├── BTCUSDT/
    │   ├── config.toml              # Конфигурация (из deploy)
    │   └── .env                     # API ключи (опционально)
    └── ETHUSDT/
        └── ...

/opt/neirobot-lit/
├── global.toml                      # Глобальные настройки
├── exchange.toml                    # Настройки биржи
└── bots/
    ├── BTCUSDT/
    │   └── model/
    │       └── model.onnx           # Рабочая копия модели
    └── ETHUSDT/
        └── ...

/var/log/neirobot/
├── BTCUSDT/
│   └── bot.log                      # Логи бота
└── ETHUSDT/
    └── ...

/var/lib/neirobot/
├── BTCUSDT/                         # Данные состояния
└── ETHUSDT/
    └── ...

/etc/systemd/system/
└── neirobot-lit@.service            # Systemd unit template
```

### Права доступа

```bash
# Бинарник
-rwxr-xr-x root:root /usr/bin/neirobot-lit

# Конфигурация
-rw-r----- neirobot:neirobot /etc/neirobot/bots/BTCUSDT/config.toml

# Модели
-rw-r----- neirobot:neirobot /opt/neirobot-lit/bots/BTCUSDT/model/model.onnx

# Логи
drwxr-x--- neirobot:neirobot /var/log/neirobot/BTCUSDT/

# Данные
drwxr-x--- neirobot:neirobot /var/lib/neirobot/BTCUSDT/
```

## Docker режим

### Структура volume

```
bots/BTCUSDT/
├── config.toml                      # Конфигурация (volume: ro)
├── model/
│   └── model.onnx                   # Модель (volume: ro)
├── logs/                            # Логи (volume: rw)
└── data/                            # Данные (volume: rw)
    └── raw/
```

### Docker контейнер

```
Container: neirobot-lit-BTCUSDT
Image: neirobot-lit:latest
User: neirobot (UID 1000)
WorkDir: /app

Volumes:
  - bots/BTCUSDT/model -> /app/bots/BTCUSDT/model (ro)
  - bots/BTCUSDT/logs -> /app/bots/BTCUSDT/logs (rw)
  - bots/BTCUSDT/data -> /app/bots/BTCUSDT/data (rw)
  - global.toml -> /app/global.toml (ro)
  - exchange.toml -> /app/exchange.toml (ro)
```

## Метаданные моделей

### Структура metadata.json

```json
{
  "symbol": "BTCUSDT",
  "version": "1.0",
  "timestamp": "2024-01-15T10:00:00Z",
  "model_type": "lit",
  "input_shape": [1, 100, 20],
  "output_shape": [1, 3],
  "training_data": {
    "start_date": "2023-01-01",
    "end_date": "2024-01-15",
    "samples": 50000,
    "timeframe": "1h"
  },
  "performance": {
    "accuracy": 0.72,
    "precision": 0.68,
    "recall": 0.75,
    "f1_score": 0.71
  },
  "features": {
    "count": 20,
    "normalization": "zscore",
    "lookback_period": 100
  },
  "quantization": {
    "enabled": false,
    "type": "int8"
  }
}
```

## Конфигурационные файлы

### global.toml

Глобальные настройки для всех ботов:
- Уровень логирования
- Таймзона
- Лимиты торговли
- Параметры риска

### exchange.toml

Настройки биржи Bybit:
- WebSocket URLs
- REST API endpoints
- Rate limits
- Категория рынка

### bot.template.toml

Шаблон конфигурации для отдельного бота:
- Символ торговой пары
- Пути к моделям
- Параметры стратегии
- Параметры риска
- Настройки ONNX

Переменные для подстановки:
- `${SYMBOL}` - торговая пара
- `${THRESHOLD_UP}` - порог для сигнала вверх
- `${THRESHOLD_DOWN}` - порог для сигнала вниз
- `${MAX_POSITION_SIZE}` - максимальный размер позиции
- `${MAX_DRAWDOWN}` - максимальная просадка
- `${EXECUTION_PROVIDER}` - провайдер выполнения (cpu/cuda/tensorrt)
- `${INTRA_THREADS}` - количество потоков

## Логирование

### Native режим

Логи хранятся в `/var/log/neirobot/SYMBOL/bot.log`

Просмотр:
```bash
journalctl -u neirobot-lit@BTCUSDT.service -f
tail -f /var/log/neirobot/BTCUSDT/bot.log
```

### Docker режим

Логи хранятся в `bots/SYMBOL/logs/bot.log` (volume)

Просмотр:
```bash
docker logs -f neirobot-lit-BTCUSDT
tail -f bots/BTCUSDT/logs/bot.log
```

## Безопасность

### Файлы, которые НЕ должны коммититься

```
.env                                # API ключи
bots/*/logs/                        # Логи
bots/*/data/                        # Данные
python_lab/models/*.onnx            # Обученные модели
```

### Файлы, которые ДОЛЖНЫ коммититься

```
deploy/                             # Скрипты развертывания
src/                                # Исходный код
python_lab/src/                     # ML код
Dockerfile                          # Docker конфигурация
Cargo.toml                          # Rust зависимости
global.toml                         # Глобальные настройки
exchange.toml                       # Настройки биржи
```

## Размеры файлов

Типичные размеры:

```
neirobot-lit (binary)               ~50-100 MB (musl static)
model.onnx (FP32)                   ~200-500 MB
model.onnx (INT8 quantized)         ~50-150 MB
metadata.json                       ~1-5 KB
config.toml                         ~1-2 KB
bot.log (per day)                   ~10-100 MB
```

## Масштабирование

### Добавление нового бота

```bash
# 1. Обучите модель
cd python_lab
python src/train.py --symbol NEWUSDT

# 2. Создайте метаданные
cat > models/metadata.json << EOF
{"symbol": "NEWUSDT", ...}
EOF

# 3. Разверните
cd ..
./deploy/manage.sh deploy NEWUSDT
```

### Удаление бота

```bash
# Native режим
sudo systemctl stop neirobot-lit@SYMBOL.service
sudo systemctl disable neirobot-lit@SYMBOL.service
sudo rm -rf /etc/neirobot/bots/SYMBOL
sudo rm -rf /opt/neirobot-lit/bots/SYMBOL
sudo rm -rf /var/log/neirobot/SYMBOL
sudo rm -rf /var/lib/neirobot/SYMBOL

# Docker режим
./deploy/manage.sh stop SYMBOL
docker rm neirobot-lit-SYMBOL
rm -rf bots/SYMBOL
```
