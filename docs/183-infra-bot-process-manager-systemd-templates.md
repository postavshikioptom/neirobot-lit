# **Задача №183: Менеджер процессов ботов на базе системных шаблонов systemd**

**Цель**: Реализация отказоустойчивой системы управления жизненным циклом ботов в **Native**-режиме. Использование шаблонов `systemd` позволяет масштабировать ферму ботов, обеспечивая автоматический перезапуск, глубокую изоляцию (Sandboxing) и приоритезацию в планировщике Linux для минимизации Latency.

---

## **План реализации для Gemini AI Coder:**

### **1. Шаблон сервиса [./deploy/neirobot-lit@.service](./deploy/neirobot-lit@.service)**
Создать универсальный юнит для запуска через `systemctl start neirobot-lit@ETHUSDT`:
- **Пути**: Использовать абсолютные пути к проекту (предположительно `/opt/neirobot`).
- **Команды**:
    - `ExecStart=/opt/neirobot/target/release/run-bot --config /opt/neirobot/bots/%i/config.toml`
    - `ExecReload=/bin/kill -HUP $MAINPID` (для мягкой перезагрузки конфига без разрыва соединений).
- **Restart**: `Restart=always`, `RestartSec=5s`, `StartLimitIntervalSec=0` (бесконечные попытки перезапуска).

### **2. Безопасность и Изоляция (Hardening)**
Максимальное ограничение прав процесса согласно `systemd-analyze security`:
- **Пользователь**: `User=neirobot`, `Group=neirobot`.
- **ФС**: `ProtectSystem=strict`, `ProtectHome=yes`, `PrivateTmp=yes`, `NoNewPrivileges=yes`.
- **Доступ**: 
    - `ReadOnlyPaths=/opt/neirobot/target/release`
    - `ReadWritePaths=/opt/neirobot/bots/%i` (разрешить запись только в папку конкретного бота для логов и данных).

### **3. HFT-тюнинг и Приоритеты**
Настройка планировщика для стабильного инференса:
- **Capabilities**: `CapabilityBoundingSet=CAP_SYS_NICE`, `AmbientCapabilities=CAP_SYS_NICE` (необходимо для изменения приоритетов CPU от имени обычного пользователя).
- **Policy**: `CPUSchedulingPolicy=rr` (Round Robin) или `fifo`.
- **Priority**: `CPUSchedulingPriority=50` (баланс между отзывчивостью и стабильностью системы).
- **Affinity**: Проброс `CPUAffinity` через `EnvironmentFile=-/opt/neirobot/bots/%i/.env`.

### **4. Управление в [./deploy/manage.sh](./deploy/manage.sh)**
Добавить обертки для работы с `systemctl`:
- **`enable <SYMBOL>`**: `sudo systemctl enable neirobot-lit@$SYMBOL`.
- **`reload <SYMBOL>`**: Отправка `SIGHUP` для обновления параметров (задача [102](./docs/000-tasks_list.md)).
- **`top`**: Использование `systemd-cgtop` для мониторинга ресурсов конкретных слайсов ботов.

### **5. Тестирование и валидация**
- **Security Check**: `systemd-analyze security neirobot-lit@SYMBOL` (цель: оценка < 2.0 "OK").
- **Reload Test**: Изменить параметр в `config.toml`, вызвать `manage.sh reload` и проверить по логам, что бот применил изменения без перезагрузки процесса.
- **Priority Check**: Убедиться через `chrt -p <PID>`, что политика `RR/FIFO` и приоритет применились успешно.

---

## **Аргументы и обоснование (спор с Grok):**
1. **Проектные пути**: Отказ от `/etc/neirobot` в пользу `/opt/neirobot` (или корня проекта) делает деплой более предсказуемым и не замусоривает системные директории.
2. **Capabilities (CAP_SYS_NICE)**: Без этой настройки `systemd` не сможет применить `FIFO` приоритеты для пользователя `neirobot`, и запуск упадет с ошибкой доступа. Это критический момент для HFT-оптимизации.
3. **Graceful Reload**: Поддержка `SIGHUP` (ExecReload) крайне важна. В HFT-трейдинге полный перезапуск процесса — это потеря 10-20 секунд на переподключение к WebSocket и прогрев модели, что недопустимо.
4. **Strict Security**: Настройка `ProtectSystem=strict` гарантирует, что даже при взломе бота злоумышленник не сможет изменить бинарный файл или конфиги других ботов.

**Gemini, твоя задача — упаковать бота в надежную "броню" системного сервиса, обеспечив ему максимальный приоритет в борьбе за микросекунды.**