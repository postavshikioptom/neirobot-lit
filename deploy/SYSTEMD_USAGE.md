# Systemd Service Management Guide

## Обзор

Neirobot LiT использует systemd-шаблоны для управления ботами в Native-режиме. Это обеспечивает:
- Автоматический перезапуск при сбоях
- Глубокую изоляцию процессов (sandboxing)
- Приоритезацию в планировщике Linux для минимизации latency
- Graceful reload конфигурации без разрыва соединений

## Структура проекта

```
/opt/neirobot/
├── target/release/run-bot          # Бинарник бота (read-only)
├── global.toml                      # Глобальная конфигурация
├── exchange.toml                    # Конфигурация биржи
└── bots/
    └── SYMBOL/                      # Директория конкретного бота
        ├── config.toml              # Конфигурация бота
        ├── .env                     # Переменные окружения (опционально)
        ├── model/
        │   └── model.onnx          # ONNX модель
        ├── data/
        │   └── raw/                # Сырые данные
        └── logs/                    # Логи бота
```

## Команды управления

### Деплой бота

```bash
sudo ./deploy/manage.sh --mode native deploy BTCUSDT
```

Эта команда:
1. Создает пользователя `neirobot:neirobot` (если не существует)
2. Копирует проект в `/opt/neirobot`
3. Создает структуру директорий для бота
4. Копирует модель и конфигурацию
5. Устанавливает systemd-сервис
6. Запускает бота

### Включение автозапуска

```bash
sudo ./deploy/manage.sh enable BTCUSDT
```

Включает автоматический запуск бота при загрузке системы.

### Graceful reload конфигурации

```bash
sudo ./deploy/manage.sh reload BTCUSDT
```

Отправляет SIGHUP процессу бота для перезагрузки конфигурации без разрыва WebSocket-соединения.

**Важно**: Бот должен поддерживать обработку SIGHUP (задача 102).

### Мониторинг ресурсов

```bash
sudo ./deploy/manage.sh top
```

Запускает `systemd-cgtop` для мониторинга использования CPU/Memory всех ботов.

### Просмотр логов

```bash
./deploy/manage.sh logs BTCUSDT
```

Показывает логи бота через `journalctl -f`.

### Остановка/перезапуск

```bash
./deploy/manage.sh stop BTCUSDT
./deploy/manage.sh restart BTCUSDT
```

### Статус всех ботов

```bash
./deploy/manage.sh status
```

## Systemd команды

Прямое управление через systemctl:

```bash
# Запуск
sudo systemctl start neirobot-lit@BTCUSDT

# Остановка
sudo systemctl stop neirobot-lit@BTCUSDT

# Перезапуск
sudo systemctl restart neirobot-lit@BTCUSDT

# Reload конфигурации
sudo systemctl reload neirobot-lit@BTCUSDT

# Статус
sudo systemctl status neirobot-lit@BTCUSDT

# Логи
sudo journalctl -u neirobot-lit@BTCUSDT -f

# Включить автозапуск
sudo systemctl enable neirobot-lit@BTCUSDT

# Отключить автозапуск
sudo systemctl disable neirobot-lit@BTCUSDT
```

## Тестирование и валидация

### 1. Security Check

Проверка уровня безопасности (цель: оценка < 2.0 "OK"):

```bash
sudo systemd-analyze security neirobot-lit@BTCUSDT
```

### 2. Reload Test

Тест graceful reload:

```bash
# 1. Изменить параметр в config.toml
sudo nano /opt/neirobot/bots/BTCUSDT/config.toml

# 2. Отправить SIGHUP
sudo ./deploy/manage.sh reload BTCUSDT

# 3. Проверить логи
sudo journalctl -u neirobot-lit@BTCUSDT -n 50
```

Бот должен применить изменения без перезапуска процесса.

### 3. Priority Check

Проверка приоритета CPU:

```bash
# Получить PID процесса
PID=$(systemctl show -p MainPID --value neirobot-lit@BTCUSDT)

# Проверить политику планировщика и приоритет
sudo chrt -p $PID
```

Ожидаемый вывод:
```
pid 12345's current scheduling policy: SCHED_RR
pid 12345's current scheduling priority: 50
```

## Настройки безопасности

Systemd-сервис использует следующие настройки изоляции:

- **ProtectSystem=strict**: Вся файловая система read-only
- **ReadOnlyPaths**: Бинарник доступен только на чтение
- **ReadWritePaths**: Только директория бота доступна на запись
- **ProtectHome=yes**: Домашние директории недоступны
- **PrivateTmp=yes**: Изолированный /tmp
- **NoNewPrivileges=yes**: Запрет повышения привилегий
- **ProtectKernelTunables=yes**: Защита параметров ядра
- **ProtectKernelModules=yes**: Защита модулей ядра
- **ProtectControlGroups=yes**: Защита cgroups

## HFT-оптимизации

### CPU Scheduling

- **Policy**: Round Robin (RR) для предсказуемой latency
- **Priority**: 50 (баланс между производительностью и стабильностью системы)
- **Capabilities**: CAP_SYS_NICE для изменения приоритетов от имени непривилегированного пользователя

### CPU Affinity (опционально)

Для привязки бота к конкретным CPU ядрам создайте drop-in файл:

```bash
sudo mkdir -p /etc/systemd/system/neirobot-lit@BTCUSDT.service.d
sudo nano /etc/systemd/system/neirobot-lit@BTCUSDT.service.d/cpu-affinity.conf
```

Содержимое:
```ini
[Service]
CPUAffinity=0-3
```

Затем перезагрузите конфигурацию:
```bash
sudo systemctl daemon-reload
sudo systemctl restart neirobot-lit@BTCUSDT
```

## Troubleshooting

### Сервис не запускается

```bash
# Проверить статус
sudo systemctl status neirobot-lit@BTCUSDT

# Проверить логи
sudo journalctl -u neirobot-lit@BTCUSDT -n 100

# Проверить права доступа
ls -la /opt/neirobot/bots/BTCUSDT/
```

### Ошибка доступа к файлам

Убедитесь, что все файлы принадлежат пользователю `neirobot`:

```bash
sudo chown -R neirobot:neirobot /opt/neirobot
```

### Приоритет CPU не применяется

Проверьте, что capabilities установлены:

```bash
sudo systemctl show neirobot-lit@BTCUSDT | grep Capability
```

Должно быть:
```
CapabilityBoundingSet=cap_sys_nice
AmbientCapabilities=cap_sys_nice
```

## Ресурсы

- [systemd.service(5)](https://www.freedesktop.org/software/systemd/man/systemd.service.html)
- [systemd.exec(5)](https://www.freedesktop.org/software/systemd/man/systemd.exec.html)
- [systemd-analyze(1)](https://www.freedesktop.org/software/systemd/man/systemd-analyze.html)
