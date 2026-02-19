# Neirobot LiT Deployment Manager

Автоматизированная система развертывания торговых ботов с поддержкой Native (systemd) и Docker режимов.

## Быстрый старт

### 1. Сборка проекта

```bash
./deploy/manage.sh build
```

Эта команда:
- Устанавливает target `x86_64-unknown-linux-musl` (если не установлен)
- Собирает статический бинарник для Linux
- Использует `cargo-zigbuild` если доступен (рекомендуется для кросс-компиляции)

### 2. Развертывание бота

#### Native режим (systemd)

```bash
sudo ./deploy/manage.sh --mode native deploy BTCUSDT
```

Требования:
- Linux с systemd
- Root права (sudo)
- Обученная модель в `python_lab/models/BTCUSDT.onnx`

#### Docker режим

```bash
./deploy/manage.sh --mode docker deploy BTCUSDT
```

Требования:
- Docker и docker-compose
- Обученная модель в `python_lab/models/BTCUSDT.onnx`

### 3. Управление ботами

#### Просмотр статуса всех ботов

```bash
./deploy/manage.sh status
```

Вывод:
```
SYMBOL          MODE       PID/CONTAINER   UPTIME               STATUS
--------------------------------------------------------------------------------
BTCUSDT         native     12345           2024-01-15 10:30     active
ETHUSDT         docker     abc123def       Up 2 hours           running
```

#### Просмотр логов

```bash
./deploy/manage.sh logs BTCUSDT
```

#### Остановка бота

```bash
./deploy/manage.sh stop BTCUSDT
```

#### Перезапуск бота

```bash
./deploy/manage.sh restart BTCUSDT
```

## Конфигурация

### Переменные окружения

Перед развертыванием можно настроить параметры через переменные окружения:

```bash
export THRESHOLD_UP=0.8
export THRESHOLD_DOWN=0.75
export MAX_POSITION_SIZE=500.0
export MAX_DRAWDOWN=0.03
export EXECUTION_PROVIDER=cpu
export INTRA_THREADS=8

./deploy/manage.sh deploy BTCUSDT
```

### Секреты (API ключи)

API ключи должны быть в файле `.env` в корне проекта:

```env
API_KEY=your_api_key_here
API_SECRET=your_api_secret_here
```

**Важно**: Файл `.env` не должен попадать в git (добавлен в `.gitignore`)

## Архитектура

### Native режим (systemd)

**Преимущества**:
- Минимальная задержка (без сетевого оверхеда Docker)
- Приоритет CPU через `CPUSchedulingPolicy=fifo`
- Оптимально для HFT режима

**Структура**:
```
/usr/bin/neirobot-lit                                    # Бинарник
/etc/neirobot/bots/SYMBOL/config.toml                   # Конфигурация
/opt/neirobot-lit/                                      # Рабочая директория
├── global.toml                                         # Глобальные настройки
├── exchange.toml                                       # Настройки биржи
└── bots/SYMBOL/model/model.onnx                        # Модель
/var/log/neirobot/SYMBOL/                               # Логи
/var/lib/neirobot/SYMBOL/                               # Данные
```

**Systemd unit**: `neirobot-lit@SYMBOL.service`

### Docker режим

**Преимущества**:
- Переносимость между системами
- Изоляция окружения
- Простое управление зависимостями

**Структура**:
```
bots/SYMBOL/model/model.onnx                            # Модель (volume)
bots/SYMBOL/logs/                                       # Логи (volume)
bots/SYMBOL/data/                                       # Данные (volume)
docker container: neirobot-lit-SYMBOL
```

## Валидация моделей

Скрипт проверяет:
1. **Существование файла модели**: `python_lab/models/SYMBOL.onnx`
2. **Метаданные модели** (если доступны): `python_lab/models/metadata.json`
   - Проверяет соответствие символа в метаданных
   - Предупреждает, если метаданные отсутствуют

Пример metadata.json:
```json
{
  "symbol": "BTCUSDT",
  "version": "1.0",
  "timestamp": "2024-01-15T10:00:00Z",
  "model_type": "lit",
  "input_shape": [1, 100, 20],
  "output_shape": [1, 3]
}
```

## Тестирование

Запуск тестов развертывания:

```bash
chmod +x tests/deploy_script_test.sh
./tests/deploy_script_test.sh
```

Тесты проверяют:
- Существование и права доступа скриптов
- Наличие шаблонов
- Создание структуры директорий
- Корректность подстановки переменных в конфиг
- Валидацию моделей

## Требования

### Общие
- Rust toolchain с target `x86_64-unknown-linux-musl`
- `envsubst` (пакет `gettext`)
- Обученные модели в `python_lab/models/`

### Native режим
- Linux с systemd
- Root права для установки

### Docker режим
- Docker Engine 20.10+
- docker-compose 1.29+

### Опционально
- `cargo-zigbuild` для улучшенной кросс-компиляции

## Troubleshooting

### Ошибка: "Model not found"

Убедитесь, что модель обучена и находится в `python_lab/models/SYMBOL.onnx`:

```bash
ls -lh python_lab/models/
```

### Ошибка: "envsubst not found"

Установите пакет `gettext`:

```bash
# Ubuntu/Debian
sudo apt-get install gettext

# macOS
brew install gettext

# Arch Linux
sudo pacman -S gettext
```

### Native режим: Service failed to start

Проверьте логи systemd:

```bash
sudo journalctl -u neirobot-lit@SYMBOL.service -n 50
```

### Docker режим: Container failed to start

Проверьте логи Docker:

```bash
docker logs neirobot-lit-SYMBOL
```

## Безопасность

1. **API ключи**: Хранятся в `.env`, не попадают в git
2. **Systemd**: Запуск от непривилегированного пользователя `neirobot`
3. **Docker**: Запуск от пользователя `neirobot` (UID 1000)
4. **Изоляция**: Каждый бот имеет свою директорию и конфигурацию

## Производительность

### Native режим (рекомендуется для HFT)
- `CPUSchedulingPolicy=fifo` - реалтайм приоритет
- `Nice=-20` - максимальный приоритет планировщика
- Прямой доступ к сети без Docker bridge

### Docker режим
- CPU limits: 2 cores max, 1 core reserved
- Memory limits: 1GB max, 512MB reserved
- Подходит для dev/test окружений

### Изоляция ресурсов (Задача 230)

Для запуска 100+ ботов на одном сервере используйте механизмы изоляции ресурсов:
- **CPU Affinity**: Привязка процесса к конкретному ядру
- **Memory Limits**: Жесткие и мягкие лимиты памяти через cgroups
- **systemd-run**: Запуск с гарантированной аллокацией ресурсов

Подробнее см. [RESOURCE_ISOLATION_GUIDE.md](./RESOURCE_ISOLATION_GUIDE.md)

## Roadmap

- [ ] Поддержка Kubernetes (Helm charts)
- [ ] Автоматическое обновление моделей
- [ ] Мониторинг через Prometheus
- [ ] Health checks и автоматический failover
- [ ] Multi-region deployment

## Лицензия

См. LICENSE в корне проекта
