# Изменения в задаче 181 (исправления)

## Исправления, внесенные на основе замечаний

### 1. Валидация метаданных моделей

**Файл**: `deploy/manage.sh` (функция `validate_model()`)

**Изменение**: Добавлена проверка метаданных модели согласно задаче 056.

**Что было**:
```bash
validate_model() {
    local symbol="$1"
    local model_path="${MODELS_DIR}/${symbol}.onnx"
    
    if [[ ! -f "$model_path" ]]; then
        log_error "Model not found: $model_path"
        return 1
    fi
    
    log_info "Model found: $model_path"
    return 0
}
```

**Что стало**:
```bash
validate_model() {
    local symbol="$1"
    local model_path="${MODELS_DIR}/${symbol}.onnx"
    local metadata_path="${MODELS_DIR}/metadata.json"
    
    if [[ ! -f "$model_path" ]]; then
        log_error "Model not found: $model_path"
        return 1
    fi
    
    # ПРОВЕРКА МЕТАДАННЫХ (Задача 056)
    if [[ -f "$metadata_path" ]]; then
        local meta_symbol=$(grep -oP '"symbol":\s*"\K[^"]+' "$metadata_path" 2>/dev/null || echo "")
        if [[ -n "$meta_symbol" && "$meta_symbol" != "$symbol" ]]; then
            log_error "Model metadata mismatch: expected $symbol, found $meta_symbol"
            return 1
        fi
        log_info "Model metadata verified for $symbol"
    else
        log_warn "Metadata file not found at $metadata_path, skipping deep check"
    fi
    
    log_info "Model found: $model_path"
    return 0
}
```

**Преимущества**:
- Проверка соответствия символа в метаданных
- Предотвращение развертывания неправильной модели
- Graceful handling отсутствия метаданных

---

### 2. Пути для Native режима

**Файл**: `deploy/manage.sh` (функция `deploy_native()`)

**Изменение**: Добавлено копирование модели в рабочую директорию `/opt/neirobot-lit/`.

**Что было**:
```bash
# Create system directories
mkdir -p "/etc/neirobot/bots/${symbol}"
mkdir -p "/var/log/neirobot/${symbol}"
mkdir -p "/var/lib/neirobot/${symbol}"

# Copy configuration
cp "${BOTS_DIR}/${symbol}/config.toml" "/etc/neirobot/bots/${symbol}/"
```

**Что стало**:
```bash
# Create system directories
mkdir -p "/etc/neirobot/bots/${symbol}"
mkdir -p "/var/log/neirobot/${symbol}"
mkdir -p "/var/lib/neirobot/${symbol}"
mkdir -p "/opt/neirobot-lit/bots/${symbol}/model"

# Copy model to working directory
log_info "Copying model to /opt/neirobot-lit..."
cp "${BOTS_DIR}/${symbol}/model/model.onnx" "/opt/neirobot-lit/bots/${symbol}/model/"
chown -R neirobot:neirobot /opt/neirobot-lit

# Copy configuration
cp "${BOTS_DIR}/${symbol}/config.toml" "/etc/neirobot/bots/${symbol}/"
```

**Преимущества**:
- Модель находится в рабочей директории сервиса
- Правильное разделение конфигурации и данных
- Соответствие production best practices

---

### 3. Пути в конфигурации

**Файл**: `deploy/bot.template.toml`

**Изменение**: Изменен путь к модели на абсолютный путь в рабочей директории.

**Что было**:
```toml
model_path = "bots/${SYMBOL}/model/model.onnx"
```

**Что стало**:
```toml
model_path = "/opt/neirobot-lit/bots/${SYMBOL}/model/model.onnx"
```

**Преимущества**:
- Абсолютный путь гарантирует корректную работу независимо от текущей директории
- Соответствует структуре native режима
- Более надежно для production

---

### 4. Docker Compose volume mapping

**Файл**: `deploy/docker-compose.yml`

**Изменение**: Разделены volume для модели, логов и данных с правильными правами доступа.

**Что было**:
```yaml
volumes:
  - ../bots/${SYMBOL}:/app/bots/${SYMBOL}
  - ../global.toml:/app/global.toml:ro
  - ../exchange.toml:/app/exchange.toml:ro
```

**Что стало**:
```yaml
volumes:
  - ../bots/${SYMBOL}/model:/app/bots/${SYMBOL}/model:ro
  - ../bots/${SYMBOL}/logs:/app/bots/${SYMBOL}/logs
  - ../bots/${SYMBOL}/data:/app/bots/${SYMBOL}/data
  - ../global.toml:/app/global.toml:ro
  - ../exchange.toml:/app/exchange.toml:ro
```

**Преимущества**:
- Модель только для чтения (ro) - защита от случайного изменения
- Логи и данные доступны для записи (rw)
- Лучшая безопасность и контроль

---

### 5. Systemd unit ReadWritePaths

**Файл**: `deploy/neirobot-lit@.service`

**Изменение**: Добавлен путь к рабочей директории моделей.

**Что было**:
```ini
ReadWritePaths=/var/log/neirobot/%i /var/lib/neirobot/%i
```

**Что стало**:
```ini
ReadWritePaths=/var/log/neirobot/%i /var/lib/neirobot/%i /opt/neirobot-lit/bots/%i
```

**Преимущества**:
- Сервис может читать модели из рабочей директории
- Соответствует ProtectSystem=strict

---

### 6. Тестирование

**Файл**: `tests/deploy_script_test.sh`

**Изменения**:
- Добавлена проверка абсолютного пути в конфиге
- Добавлен тест для валидации метаданных
- Обновлены проверки путей

**Новые тесты**:
- `test_metadata_validation()` - проверка метаданных модели

---

### 7. Документация

**Новые файлы**:
- `deploy/DIRECTORY_STRUCTURE.md` - полная структура директорий
- `deploy/CHANGES.md` - этот файл

**Обновленные файлы**:
- `deploy/README.md` - добавлена информация о валидации моделей
- `deploy/EXAMPLES.md` - добавлены примеры работы с метаданными

---

## Структура директорий после исправлений

### Native режим

```
/usr/bin/neirobot-lit                                    # Бинарник
/etc/neirobot/bots/SYMBOL/config.toml                   # Конфигурация
/opt/neirobot-lit/bots/SYMBOL/model/model.onnx          # Модель (рабочая)
/var/log/neirobot/SYMBOL/bot.log                        # Логи
/var/lib/neirobot/SYMBOL/                               # Данные
```

### Docker режим

```
bots/SYMBOL/model/model.onnx                            # Модель (ro volume)
bots/SYMBOL/logs/                                       # Логи (rw volume)
bots/SYMBOL/data/                                       # Данные (rw volume)
```

---

## Проверка исправлений

Все исправления соответствуют требованиям задачи 181:

✅ **Валидация моделей**: Проверка метаданных согласно задаче 056
✅ **Пути в native режиме**: Модель в `/opt/neirobot-lit/bots/SYMBOL/model/`
✅ **Абсолютные пути**: Конфиг использует `/opt/neirobot-lit/bots/SYMBOL/model/model.onnx`
✅ **Docker volume mapping**: Правильное разделение прав доступа
✅ **Systemd security**: ReadWritePaths включает рабочую директорию
✅ **Тестирование**: Добавлены тесты для новой функциональности

---

## Миграция существующих развертываний

Если у вас уже есть развернутые боты, выполните:

```bash
# 1. Остановите боты
sudo systemctl stop 'neirobot-lit@*.service'

# 2. Скопируйте модели в новую директорию
sudo mkdir -p /opt/neirobot-lit/bots
for symbol in BTCUSDT ETHUSDT; do
    sudo mkdir -p /opt/neirobot-lit/bots/$symbol/model
    sudo cp bots/$symbol/model/model.onnx /opt/neirobot-lit/bots/$symbol/model/
    sudo chown -R neirobot:neirobot /opt/neirobot-lit
done

# 3. Обновите конфиги (они будут пересозданы при следующем deploy)

# 4. Перезагрузите systemd
sudo systemctl daemon-reload

# 5. Запустите боты
sudo systemctl start 'neirobot-lit@*.service'
```


---

## Финальное исправление: Копирование глобальных конфигов

**Файл**: `deploy/manage.sh` (функция `deploy_native()`)

**Проблема**: Бот не запускался в native режиме, так как ему не хватало файлов `global.toml` и `exchange.toml` в рабочей директории `/opt/neirobot-lit`.

**Решение**: Добавлено копирование глобальных конфигов перед запуском сервиса.

**Что было добавлено**:
```bash
# Copy global and exchange configs (required for load_full_config)
log_info "Copying global and exchange configs to /opt/neirobot-lit..."
cp "${PROJECT_ROOT}/global.toml" "/opt/neirobot-lit/"
cp "${PROJECT_ROOT}/exchange.toml" "/opt/neirobot-lit/"
```

**Структура после исправления**:
```
/opt/neirobot-lit/
├── global.toml                      # Глобальные настройки
├── exchange.toml                    # Настройки биржи
└── bots/
    ├── BTCUSDT/
    │   └── model/
    │       └── model.onnx
    └── ETHUSDT/
        └── ...
```

**Преимущества**:
- Бот может загрузить полную конфигурацию через `load_full_config()`
- Все необходимые файлы находятся в рабочей директории
- Соответствует структуре Docker режима
- Гарантирует корректный запуск сервиса

**Проверка**:
```bash
# После развертывания проверьте наличие файлов
ls -la /opt/neirobot-lit/
# Должны быть: global.toml, exchange.toml, bots/

# Проверьте, что сервис запустился
sudo systemctl status neirobot-lit@BTCUSDT.service
```
