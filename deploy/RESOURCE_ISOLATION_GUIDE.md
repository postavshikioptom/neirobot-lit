# Руководство по изоляции ресурсов (Задача 230)

## Обзор

Задача 230 реализует механизмы жесткой и мягкой изоляции ресурсов для каждого инстанса бота. Основной упор на использование **Linux cgroups** через **systemd** для гарантированной аллокации и внутренние проверки в Rust для предотвращения критических сбоев.

## Цель

Обеспечить стабильность торговой фермы при запуске 100+ процессов на одном сервере. Исключить влияние утечек памяти или CPU-спайков одного бота на работу остальных.

## Компоненты решения

### 1. Внутренний мониторинг (Rust)

#### CPU Affinity
Привязка процесса к конкретному ядру CPU для предсказуемой производительности:

```toml
# bots/BTCUSDT/config.toml
[system]
cpu_core = 1  # Привязать к ядру #1
max_memory_mb = 512  # Мягкий лимит памяти
```

#### Memory Guard
Мониторинг использования памяти с порогом 90%:
- При превышении 90% от `max_memory_mb` отправляется событие `SoftLimitReached`
- Бот может перейти в режим Graceful Degradation (задача 220)
- Проверка выполняется каждые 5 секунд

### 2. Жесткая изоляция (cgroups v2)

#### Запуск через systemd-run

Для обеспечения жестких лимитов, которые процесс не может обойти, используйте `systemd-run`:

```bash
systemd-run --user --scope \
  -p MemoryMax=512M \
  -p CPUQuota=100% \
  -p AllowedCPUs=1 \
  ./target/release/run-bot BTCUSDT
```

**Параметры:**
- `MemoryMax=512M` - Жесткий лимит памяти (OOM Killer при превышении)
- `CPUQuota=100%` - Лимит использования CPU (100% = 1 ядро)
- `AllowedCPUs=1` - Разрешенные ядра CPU (можно указать диапазон: `0-3`)

#### Интеграция с farm_ctl.py

Пример интеграции в скрипт управления фермой:

```python
import subprocess

def start_bot_with_isolation(symbol, cpu_core, memory_mb):
    """Запуск бота с изоляцией ресурсов через systemd-run"""
    cmd = [
        "systemd-run",
        "--user",
        "--scope",
        f"-p MemoryMax={memory_mb}M",
        f"-p CPUQuota=100%",
        f"-p AllowedCPUs={cpu_core}",
        "./target/release/run-bot",
        symbol
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to start {symbol}: {result.stderr}")
        return False
    
    print(f"Started {symbol} on core {cpu_core} with {memory_mb}MB limit")
    return True

# Пример использования
start_bot_with_isolation("BTCUSDT", cpu_core=1, memory_mb=512)
start_bot_with_isolation("ETHUSDT", cpu_core=2, memory_mb=512)
```

### 3. Опциональный jemalloc

Для лучшего контроля фрагментации памяти можно использовать jemalloc:

```bash
# Сборка с jemalloc
cargo build --release --features jemalloc

# Запуск
./target/release/run-bot BTCUSDT
```

**Преимущества jemalloc:**
- Меньшая фрагментация памяти
- Лучшая производительность при многопоточности
- Детальная статистика использования памяти

**Недостатки:**
- Сложнее отладка
- Дополнительная зависимость

## Проверка работы

### 1. Проверка CPU Affinity

```bash
# Получить PID процесса
ps aux | grep run-bot

# Проверить привязку к ядру
taskset -p <PID>
```

Ожидаемый вывод:
```
pid 12345's current affinity mask: 2
```
(маска `2` = ядро #1, маска `1` = ядро #0)

### 2. Проверка лимитов памяти

```bash
# Проверить cgroup лимиты
systemctl --user status run-bot-BTCUSDT.scope

# Проверить текущее использование
cat /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/run-bot-BTCUSDT.scope/memory.current
cat /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/run-bot-BTCUSDT.scope/memory.max
```

### 3. Мониторинг в логах

Проверьте логи бота на наличие сообщений:

```
[INFO] Process pinned to CPU core 1
[INFO] Starting resource monitor: max_memory=512MB, soft_limit=460MB
[WARN] Memory soft limit reached: current=461MB, limit=512MB
```

## Рекомендации по распределению ресурсов

### Для сервера с 16 ядрами и 64GB RAM

**Стратегия 1: Равномерное распределение**
- 100 ботов
- По 6-7 ботов на ядро
- 512MB памяти на бота
- Итого: ~51GB используется

**Стратегия 2: Приоритизация**
- Высокочастотные боты: выделенные ядра (0-3)
- Среднечастотные боты: разделенные ядра (4-11)
- Низкочастотные боты: общие ядра (12-15)

### Пример конфигурации для фермы

```toml
# bots/BTCUSDT/config.toml (высокочастотный)
[system]
cpu_core = 0
max_memory_mb = 1024

# bots/ETHUSDT/config.toml (среднечастотный)
[system]
cpu_core = 4
max_memory_mb = 512

# bots/DOGEUSDT/config.toml (низкочастотный)
[system]
# cpu_core не указан = без привязки
max_memory_mb = 256
```

## Устранение неполадок

### Проблема: "Failed to pin process to CPU core"

**Причина:** Недостаточно прав или неверный номер ядра

**Решение:**
```bash
# Проверить доступные ядра
lscpu

# Проверить права
ulimit -a

# Запустить с правами
sudo setcap cap_sys_nice=eip ./target/release/run-bot
```

### Проблема: OOM Killer убивает процесс

**Причина:** Превышен жесткий лимит MemoryMax

**Решение:**
1. Увеличить лимит в systemd-run
2. Уменьшить max_memory_mb в конфиге для раннего предупреждения
3. Проверить утечки памяти через valgrind

### Проблема: Высокая задержка несмотря на CPU Affinity

**Причина:** Другие процессы на том же ядре

**Решение:**
```bash
# Изолировать ядра через kernel boot параметры
# Добавить в /etc/default/grub:
GRUB_CMDLINE_LINUX="isolcpus=0-3"

# Обновить grub
sudo update-grub
sudo reboot
```

## Интеграция с мониторингом

Метрики ресурсов доступны через Prometheus:

```
# Использование памяти (RSS)
process_resident_memory_bytes{symbol="BTCUSDT"}

# Использование CPU
process_cpu_usage_percent{symbol="BTCUSDT"}

# Алерты при превышении лимитов
rate(memory_soft_limit_reached_total[5m]) > 0
```

## Дополнительные ресурсы

- [systemd Resource Control](https://www.freedesktop.org/software/systemd/man/systemd.resource-control.html)
- [Linux cgroups v2](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html)
- [jemalloc Documentation](https://jemalloc.net/)
- [CPU Affinity in Linux](https://man7.org/linux/man-pages/man2/sched_setaffinity.2.html)
