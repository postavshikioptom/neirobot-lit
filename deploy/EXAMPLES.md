# Примеры использования Deployment Manager

## Базовые сценарии

### 1. Первоначальная настройка и сборка

```bash
# Установка musl target (если не установлен)
rustup target add x86_64-unknown-linux-musl

# Опционально: установка cargo-zigbuild для лучшей кросс-компиляции
cargo install cargo-zigbuild

# Сборка проекта
cd /path/to/neirobot-lit
./deploy/manage.sh build
```

### 2. Развертывание в Native режиме (Production)

```bash
# Убедитесь, что модель обучена
ls -lh python_lab/models/BTCUSDT.onnx

# Настройка переменных окружения (опционально)
export THRESHOLD_UP=0.75
export THRESHOLD_DOWN=0.75
export MAX_POSITION_SIZE=1000.0
export MAX_DRAWDOWN=0.02
export INTRA_THREADS=8

# Развертывание (требует sudo)
sudo ./deploy/manage.sh --mode native deploy BTCUSDT

# Проверка статуса
sudo systemctl status neirobot-lit@BTCUSDT.service

# Просмотр логов
sudo journalctl -u neirobot-lit@BTCUSDT.service -f
```

### 3. Развертывание в Docker режиме (Development/Testing)

```bash
# Настройка API ключей
cat > .env << EOF
API_KEY=your_api_key_here
API_SECRET=your_api_secret_here
EOF

# Развертывание
./deploy/manage.sh --mode docker deploy ETHUSDT

# Проверка статуса
docker ps | grep neirobot-lit-ETHUSDT

# Просмотр логов
docker logs -f neirobot-lit-ETHUSDT
```

## Продвинутые сценарии

### 4. Развертывание нескольких ботов

```bash
# Обучите модели для нескольких символов
cd python_lab
python src/train.py --symbol BTCUSDT
python src/train.py --symbol ETHUSDT
python src/train.py --symbol SOLUSDT

# Разверните ботов
cd ..
for symbol in BTCUSDT ETHUSDT SOLUSDT; do
    sudo ./deploy/manage.sh --mode native deploy $symbol
done

# Проверьте статус всех ботов
./deploy/manage.sh status
```

### 5. Развертывание с кастомными параметрами

```bash
# Агрессивная стратегия для волатильного актива
export THRESHOLD_UP=0.6
export THRESHOLD_DOWN=0.6
export MAX_POSITION_SIZE=50.0
export MAX_DRAWDOWN=0.10

./deploy/manage.sh --mode docker deploy DOGEUSDT

# Консервативная стратегия для стабильного актива
export THRESHOLD_UP=0.85
export THRESHOLD_DOWN=0.85
export MAX_POSITION_SIZE=5000.0
export MAX_DRAWDOWN=0.01

./deploy/manage.sh --mode docker deploy USDCUSDT
```

### 6. Развертывание с GPU (CUDA)

```bash
# Убедитесь, что CUDA доступна
nvidia-smi

# Настройка для GPU
export EXECUTION_PROVIDER=cuda
export DEVICE_ID=0

# Развертывание
./deploy/manage.sh --mode docker deploy BTCUSDT

# Проверка использования GPU
docker exec neirobot-lit-BTCUSDT nvidia-smi
```

### 7. Мониторинг и управление

```bash
# Просмотр статуса всех ботов
./deploy/manage.sh status

# Просмотр логов конкретного бота
./deploy/manage.sh logs BTCUSDT

# Остановка бота
./deploy/manage.sh stop BTCUSDT

# Перезапуск бота (например, после обновления модели)
./deploy/manage.sh restart BTCUSDT

# Остановка всех ботов (native)
sudo systemctl stop 'neirobot-lit@*.service'

# Остановка всех ботов (docker)
docker stop $(docker ps -q --filter "name=neirobot-lit-")
```

### 8. Обновление модели без простоя

```bash
# Обучите новую модель
cd python_lab
python src/train.py --symbol BTCUSDT

# Скопируйте новую модель
cp models/BTCUSDT.onnx ../bots/BTCUSDT/model/model.onnx

# Перезапустите бота
cd ..
./deploy/manage.sh restart BTCUSDT

# Проверьте, что бот работает с новой моделью
./deploy/manage.sh logs BTCUSDT | grep "Model loaded"
```

### 9. Миграция с Docker на Native

```bash
# Остановите Docker контейнер
./deploy/manage.sh stop BTCUSDT
docker rm neirobot-lit-BTCUSDT

# Разверните в native режиме
sudo ./deploy/manage.sh --mode native deploy BTCUSDT

# Проверьте статус
sudo systemctl status neirobot-lit@BTCUSDT.service
```

### 10. Отладка проблем

```bash
# Проверка наличия модели
ls -lh python_lab/models/BTCUSDT.onnx

# Проверка конфигурации
cat bots/BTCUSDT/config.toml

# Проверка логов (native)
sudo journalctl -u neirobot-lit@BTCUSDT.service -n 100

# Проверка логов (docker)
docker logs neirobot-lit-BTCUSDT --tail 100

# Проверка ресурсов (docker)
docker stats neirobot-lit-BTCUSDT

# Проверка сети (docker)
docker exec neirobot-lit-BTCUSDT ping -c 3 api.bybit.com

# Ручной запуск для отладки
./target/x86_64-unknown-linux-musl/release/neirobot-lit \
    --config bots/BTCUSDT/config.toml
```

## Автоматизация

### 11. Cron job для автоматического перезапуска

```bash
# Добавьте в crontab
crontab -e

# Перезапуск каждый день в 3:00 AM
0 3 * * * /path/to/neirobot-lit/deploy/manage.sh restart BTCUSDT

# Проверка статуса каждый час
0 * * * * /path/to/neirobot-lit/deploy/manage.sh status >> /var/log/neirobot-status.log
```

### 12. Скрипт для массового развертывания

```bash
#!/bin/bash
# deploy_all.sh

SYMBOLS=(
    "BTCUSDT"
    "ETHUSDT"
    "SOLUSDT"
    "BNBUSDT"
    "ADAUSDT"
)

for symbol in "${SYMBOLS[@]}"; do
    echo "Deploying $symbol..."
    
    # Проверка наличия модели
    if [[ ! -f "python_lab/models/${symbol}.onnx" ]]; then
        echo "Model not found for $symbol, skipping..."
        continue
    fi
    
    # Развертывание
    sudo ./deploy/manage.sh --mode native deploy "$symbol"
    
    # Пауза между развертываниями
    sleep 5
done

echo "All bots deployed!"
./deploy/manage.sh status
```

### 13. Health check скрипт

```bash
#!/bin/bash
# health_check.sh

./deploy/manage.sh status | while read -r line; do
    if echo "$line" | grep -q "inactive\|failed\|exited"; then
        symbol=$(echo "$line" | awk '{print $1}')
        echo "Bot $symbol is down, restarting..."
        ./deploy/manage.sh restart "$symbol"
    fi
done
```

## Производственные best practices

### 14. Production deployment checklist

```bash
# 1. Проверка окружения
rustc --version
cargo --version
docker --version
systemctl --version

# 2. Сборка оптимизированного бинарника
RUSTFLAGS="-C target-cpu=native" ./deploy/manage.sh build

# 3. Настройка безопасности
chmod 600 .env
chown neirobot:neirobot .env

# 4. Настройка firewall
sudo ufw allow 443/tcp  # HTTPS для API
sudo ufw enable

# 5. Настройка мониторинга
# (интеграция с Prometheus/Grafana)

# 6. Развертывание
sudo ./deploy/manage.sh --mode native deploy BTCUSDT

# 7. Проверка
./deploy/manage.sh status
./deploy/manage.sh logs BTCUSDT | head -n 50

# 8. Настройка автоматического перезапуска
sudo systemctl enable neirobot-lit@BTCUSDT.service
```

## Troubleshooting

### Проблема: "Permission denied"

```bash
# Сделайте скрипт исполняемым
chmod +x deploy/manage.sh

# Для native режима используйте sudo
sudo ./deploy/manage.sh --mode native deploy BTCUSDT
```

### Проблема: "Model not found"

```bash
# Проверьте наличие модели
ls -lh python_lab/models/

# Обучите модель
cd python_lab
python src/train.py --symbol BTCUSDT
```

### Проблема: "envsubst: command not found"

```bash
# Ubuntu/Debian
sudo apt-get install gettext

# macOS
brew install gettext
brew link --force gettext

# Arch Linux
sudo pacman -S gettext
```

### Проблема: Docker контейнер не запускается

```bash
# Проверьте логи
docker logs neirobot-lit-BTCUSDT

# Проверьте образ
docker images | grep neirobot-lit

# Пересоберите образ
docker build -t neirobot-lit:latest -f Dockerfile .

# Проверьте переменные окружения
docker exec neirobot-lit-BTCUSDT env
```


## Работа с метаданными моделей

### 15. Создание и проверка метаданных

```bash
# Создайте файл метаданных для модели
cat > python_lab/models/metadata.json << EOF
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
    "samples": 50000
  },
  "performance": {
    "accuracy": 0.72,
    "precision": 0.68,
    "recall": 0.75
  }
}
EOF

# Развертывание с проверкой метаданных
./deploy/manage.sh --mode docker deploy BTCUSDT

# Скрипт проверит соответствие символа в метаданных
# Если символ не совпадает, развертывание будет отменено
```

### 16. Обновление метаданных при переобучении

```bash
# После переобучения модели обновите метаданные
python_lab/src/train.py --symbol BTCUSDT

# Обновите metadata.json
cat > python_lab/models/metadata.json << EOF
{
  "symbol": "BTCUSDT",
  "version": "2.0",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "model_type": "lit",
  "training_data": {
    "start_date": "2023-01-01",
    "end_date": "$(date +%Y-%m-%d)",
    "samples": 60000
  }
}
EOF

# Перезапустите бота с новой моделью
./deploy/manage.sh restart BTCUSDT
```

### 17. Валидация моделей перед развертыванием

```bash
# Проверьте наличие модели и метаданных
ls -lh python_lab/models/BTCUSDT.onnx
cat python_lab/models/metadata.json | jq .

# Проверьте, что символ в метаданных совпадает
grep -o '"symbol": "[^"]*"' python_lab/models/metadata.json

# Если все в порядке, разверните
./deploy/manage.sh deploy BTCUSDT
```
