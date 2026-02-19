# Automatic Process Watchdog (Задача 219)

## Обзор

Система мониторинга живучести (Liveness) и автоматического восстановления бота состоит из двух компонентов:

1. **Внутренний механизм Liveness** (Rust) - обновляет файл heartbeat из основного торгового цикла
2. **Внешний Watchdog** (Shell/Systemd) - мониторит процесс и перезапускает его при необходимости

## Компоненты

### 1. Внутренний механизм Liveness

#### Heartbeat файл
- **Расположение**: `bots/{SYMBOL}/state/liveness.heartbeat`
- **Содержимое**: Unix timestamp (в секундах)
- **Права доступа**: 640 (rw-r-----)
- **Обновление**: Каждая итерация основного торгового цикла

#### Инициализация при старте
При запуске бота проверяется возраст heartbeat файла:
- Если файл старше 1 минуты → логируется "Recovery после сбоя"
- Файл пересоздается с текущим timestamp

#### Panic Hook
При панике процесса:
- Backtrace записывается в stderr
- Информация о месте паники логируется
- Сообщение паники выводится в консоль

### 2. Внешний Watchdog

#### Скрипт bot_watchdog.sh
```bash
./scripts/bot_watchdog.sh BTCUSDT
```

**Параметры**:
- `SYMBOL` - символ торговой пары (по умолчанию BTCUSDT)

**Переменные окружения**:
- `BOT_BINARY` - путь к бинарнику (по умолчанию `./target/release/run-bot`)
- `BOT_CONFIG` - путь к конфигурации (по умолчанию `./bots/$SYMBOL/config.toml`)

**Функциональность**:
- Запускает бот в цикле
- Обнаруживает crash процесса
- Проверяет свежесть heartbeat файла для обнаружения deadlock
- Применяет exponential backoff перед перезапуском
- Ограничивает количество рестартов (макс 10 за час)
- Корректно обрабатывает SIGTERM для плановой остановки

#### Exponential Backoff
При каждом краше пауза перед перезапуском увеличивается:
- 1-й краш: 5 секунд
- 2-й краш: 10 секунд
- 3-й краш: 20 секунд
- ...
- Максимум: 60 секунд

При успешном запуске backoff сбрасывается.

#### Лимит рестартов
- Максимум 10 рестартов за 1 час
- Если лимит превышен, watchdog выходит с ошибкой
- Это защита от "сжигания" API-лимитов и блокировки IP

### 3. Systemd интеграция

#### Конфигурация
Файл: `deploy/neirobot-lit@.service`

**Параметры перезапуска**:
```ini
Restart=on-failure
RestartSec=5s
StartLimitIntervalSec=3600
StartLimitBurst=10
```

- `Restart=on-failure` - перезапускать только при ошибке
- `RestartSec=5s` - пауза перед перезапуском
- `StartLimitIntervalSec=3600` - окно для подсчета рестартов (1 час)
- `StartLimitBurst=10` - максимум 10 рестартов в окне

## Использование

### Вариант 1: Systemd (рекомендуется для production)

```bash
# Установка сервиса
sudo cp deploy/neirobot-lit@.service /etc/systemd/system/

# Перезагрузка конфигурации systemd
sudo systemctl daemon-reload

# Запуск бота для BTCUSDT
sudo systemctl start neirobot-lit@BTCUSDT

# Проверка статуса
sudo systemctl status neirobot-lit@BTCUSDT

# Просмотр логов
sudo journalctl -u neirobot-lit@BTCUSDT -f

# Остановка
sudo systemctl stop neirobot-lit@BTCUSDT

# Перезагрузка конфигурации (SIGHUP)
sudo systemctl reload neirobot-lit@BTCUSDT
```

### Вариант 2: Прямой запуск watchdog скрипта

```bash
# Запуск watchdog для BTCUSDT
./scripts/bot_watchdog.sh BTCUSDT

# С переменными окружения
BOT_BINARY=./target/release/run-bot \
BOT_CONFIG=./bots/BTCUSDT/config.toml \
./scripts/bot_watchdog.sh BTCUSDT
```

### Вариант 3: Systemd с watchdog скриптом

Можно использовать watchdog скрипт как ExecStart в systemd:

```ini
[Service]
ExecStart=/path/to/scripts/bot_watchdog.sh %i
```

## Мониторинг

### Проверка heartbeat
```bash
# Просмотр текущего heartbeat
cat bots/BTCUSDT/state/liveness.heartbeat

# Проверка возраста heartbeat
current_ts=$(date +%s)
heartbeat_ts=$(cat bots/BTCUSDT/state/liveness.heartbeat)
age=$((current_ts - heartbeat_ts))
echo "Heartbeat age: $age seconds"
```

### Логирование
- **Systemd**: `sudo journalctl -u neirobot-lit@BTCUSDT -f`
- **Файл логов**: `bots/BTCUSDT/logs/bot.log`
- **Watchdog логи**: Выводятся в stderr

### Метрики
Prometheus метрики для мониторинга:
- `watchdog_stall_seconds` - время зависания (если > 0)
- `watchdog_check_timestamp` - время последней проверки

## Диагностика

### Бот часто падает
1. Проверьте логи: `tail -f bots/BTCUSDT/logs/bot.log`
2. Проверьте конфигурацию: `cat bots/BTCUSDT/config.toml`
3. Проверьте API ключи: убедитесь, что BYBIT_API_KEY и BYBIT_API_SECRET установлены
4. Проверьте права доступа: `ls -la bots/BTCUSDT/`

### Heartbeat не обновляется
1. Проверьте, запущен ли бот: `ps aux | grep run-bot`
2. Проверьте возраст heartbeat: `stat bots/BTCUSDT/state/liveness.heartbeat`
3. Проверьте права доступа на директорию state: `ls -la bots/BTCUSDT/state/`

### Watchdog не перезапускает бота
1. Проверьте, запущен ли watchdog: `ps aux | grep bot_watchdog`
2. Проверьте логи watchdog в stderr
3. Убедитесь, что лимит рестартов не превышен (макс 10 за час)

## Безопасность

### Права доступа
- Heartbeat файл: 640 (rw-r-----)
- Директория state: 750 (rwxr-x---)
- Логи: 640 (rw-r-----)

### Защита от бесконечных рестартов
- Systemd: `StartLimitBurst=10` за `StartLimitIntervalSec=3600`
- Watchdog: максимум 10 рестартов за 1 час
- Exponential backoff предотвращает "сжигание" ресурсов

## Примеры

### Запуск нескольких ботов
```bash
# Запуск для разных символов
sudo systemctl start neirobot-lit@BTCUSDT
sudo systemctl start neirobot-lit@ETHUSDT
sudo systemctl start neirobot-lit@BNBUSDT

# Проверка всех
sudo systemctl status neirobot-lit@*

# Остановка всех
sudo systemctl stop neirobot-lit@*
```

### Автозапуск при перезагрузке
```bash
# Включить автозапуск
sudo systemctl enable neirobot-lit@BTCUSDT

# Отключить автозапуск
sudo systemctl disable neirobot-lit@BTCUSDT

# Проверить статус
sudo systemctl is-enabled neirobot-lit@BTCUSDT
```

### Плановая остановка
```bash
# Graceful shutdown (отправляет SIGTERM)
sudo systemctl stop neirobot-lit@BTCUSDT

# Перезагрузка конфигурации (отправляет SIGHUP)
sudo systemctl reload neirobot-lit@BTCUSDT

# Перезапуск
sudo systemctl restart neirobot-lit@BTCUSDT
```

## Ограничения и особенности

1. **Heartbeat в основном цикле**: Heartbeat обновляется только если основной цикл работает. Если цикл зависнет, heartbeat не будет обновляться, и watchdog обнаружит это.

2. **Panic Hook**: Хук используется только для записи backtrace. Сохранение состояния происходит через механизм из задачи 218.

3. **Exponential Backoff**: Backoff сбрасывается при успешном запуске, но не при graceful shutdown.

4. **Лимит рестартов**: Считается за последний час. Если бот упал 10 раз за час, watchdog выходит.

## Связанные задачи

- **Задача 146**: Watchdog для Hot Path (внутренний мониторинг)
- **Задача 218**: Persistence (сохранение состояния)
- **Задача 184**: Config reload (SIGHUP)
- **Задача 135**: Health check (мониторинг здоровья)
