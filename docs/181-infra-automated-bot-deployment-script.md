# **Задача №181: Скрипт автоматизированного развертывания ботов (Automated Bot Deployment Script)**

**Цель**: Создание инструментария для быстрого запуска и управления изолированными инстансами ботов. Скрипт должен поддерживать два режима: **Native** (через `systemd` для минимальной задержки) и **Docker** (для переносимости), обеспечивая автоматическую сборку, шаблонизацию конфигов и проверку моделей.

---

## **План реализации для Gemini AI Coder:**

### **1. Управляющий скрипт [./deploy/manage.sh](./deploy/manage.sh)**
Реализовать Bash-скрипт с поддержкой флагов `--mode [native|docker]`:
- **`build`**: 
    - Поддержка кросс-компиляции: `cargo build --release --target x86_64-unknown-linux-musl` (для статических бинарников).
    - Опционально: использование `cargo-zigbuild` для деплоя с Mac/Windows на Linux.
- **`deploy <SYMBOL>`**:
    1. Создать структуру: `bots/SYMBOL/{model,data,logs}`.
    2. Синхронизировать модель: `cp python_lab/models/SYMBOL.onnx bots/SYMBOL/model/model.onnx`.
    3. Генерация конфига: использовать `envsubst` для замены переменных в [./deploy/bot.template.toml](./deploy/bot.template.toml).
    4. **Native Mode**: Создать и запустить `systemd` юнит `neirobot-lit@SYMBOL.service`.
    5. **Docker Mode**: Запустить `docker-compose` из `deploy/docker-compose.yml`.

### **2. Шаблоны инфраструктуры**
- **Systemd [./deploy/neirobot-lit@.service](./deploy/neirobot-lit@.service)**:
    - `ExecStart=/usr/bin/neirobot-lit --config /etc/neirobot/bots/%i/config.toml`
    - `Restart=always`, `CPUSchedulingPolicy=fifo` (для HFT приоритета).
- **Docker [./Dockerfile](./Dockerfile)**:
    - Multi-stage сборка на базе `alpine` или `distroless` для минимального веса и безопасности.

### **3. Валидация и безопасность**
- **Model Check**: Скрипт должен проверять соответствие версии модели и метаданных (задача [056](./docs/000-tasks_list.md)) перед запуском.
- **Secrets**: API ключи и секреты подставляются через переменные окружения ОС, не сохраняясь в итоговых `.toml` файлах.

### **4. Управление и мониторинг**
- **`status`**: Агрегированный вывод: `SYMBOL | MODE | PID/CONTAINER | UPTIME | STATUS`.
- **`logs <SYMBOL>`**: Удобный проброс к `journalctl -u neirobot-lit@SYMBOL` или `docker logs`.

### **5. Тестирование в [./tests/deploy_script_test.sh](./tests/deploy_script_test.sh)**
- **Mock Deploy**: Запуск скрипта в тестовой папке, проверка создания директорий, корректности подстановки переменных в конфиг и наличия исполняемых файлов.

---

## **Аргументы и обоснование (спор с Grok):**
1. **Native Priority**: Для HFT режима (задача [081](./docs/000-tasks_list.md)) запуск напрямую через `systemd` предпочтительнее Docker из-за отсутствия сетевого оверхеда и лучшего управления приоритетами CPU (FIFO/RR).
2. **Static Linking (musl)**: Использование `musl` таргета при сборке позволяет копировать один бинарный файл на любой Linux-сервер без забот о версиях `glibc`.
3. **envsubst**: Это стандартный и легковесный способ шаблонизации, который не требует установки тяжелых шаблонизаторов (Jinja2 и др.) на сервер деплоя.
4. **Separation of Concerns**: Скрипт не обучает модель, он лишь "потребляет" артефакты из `python_lab`. Это разделяет зоны ответственности между Data Science и DevOps/Infra.

**Gemini, твоя задача — превратить инфраструктуру в "код", чтобы запуск нового торгового инструмента занимал секунды, а не часы ручной настройки.**