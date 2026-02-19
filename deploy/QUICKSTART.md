# Быстрый старт

## За 5 минут до первого запущенного бота

### Шаг 1: Подготовка

```bash
# Клонируйте репозиторий (если еще не сделали)
cd /path/to/neirobot-lit

# Убедитесь, что модель обучена
ls python_lab/models/BTCUSDT.onnx
```

### Шаг 2: Настройка API ключей

```bash
# Создайте .env файл с вашими ключами
cat > .env << EOF
API_KEY=your_bybit_api_key
API_SECRET=your_bybit_api_secret
EOF

# Защитите файл
chmod 600 .env
```

### Шаг 3: Сборка

```bash
# Сделайте скрипт исполняемым
chmod +x deploy/manage.sh

# Соберите проект
./deploy/manage.sh build
```

### Шаг 4: Развертывание

#### Вариант A: Docker (рекомендуется для начала)

```bash
./deploy/manage.sh --mode docker deploy BTCUSDT
```

#### Вариант B: Native (для production)

```bash
sudo ./deploy/manage.sh --mode native deploy BTCUSDT
```

### Шаг 5: Проверка

```bash
# Проверьте статус
./deploy/manage.sh status

# Посмотрите логи
./deploy/manage.sh logs BTCUSDT
```

## Готово! 🚀

Ваш бот запущен и торгует на Bybit.

## Что дальше?

- Прочитайте [README.md](README.md) для полной документации
- Изучите [EXAMPLES.md](EXAMPLES.md) для продвинутых сценариев
- Настройте мониторинг и алерты
- Разверните ботов для других символов

## Остановка бота

```bash
./deploy/manage.sh stop BTCUSDT
```

## Помощь

```bash
./deploy/manage.sh --help
```
