#!/bin/bash
# =============================================
# upload_to_kagglehub.sh
# Загружает ВСЁ содержимое текущей папки через kagglehub
# Использует только системную переменную KAGGLE_API_TOKEN (Windows)
# =============================================

set -e

# ================= НАСТРОЙКИ =================
DATASET_HANDLE="postavshikioptom11/neirobot-lit-data"
VERSION_NOTES="auto-update $(date '+%Y-%m-%d %H:%M:%S') — full folder via kagglehub"
# =============================================

echo "🚀 Запуск загрузки через kagglehub"
echo "📁 Датасет: $DATASET_HANDLE"

# Проверяем, что переменная KAGGLE_API_TOKEN установлена
if [ -z "$KAGGLE_API_TOKEN" ]; then
    echo "❌ Ошибка: Переменная KAGGLE_API_TOKEN не найдена."
    echo "   Убедись, что ты установил её в Переменных среды Windows и перезапустил терминал."
    exit 1
fi

echo "✅ KAGGLE_API_TOKEN обнаружен (системная переменная)"

# Переходим в папку, где лежит скрипт
cd "$(dirname "$0")"

# Проверяем наличие kagglehub
if ! python -c "import kagglehub" &> /dev/null; then
    echo "❌ kagglehub не установлен. Выполни: pip install kagglehub"
    exit 1
fi

echo "📋 Содержимое папки для загрузки:"
ls -la

echo "📤 Загружаю новую версию..."

python - <<EOF
import kagglehub
import os

handle = "$DATASET_HANDLE"
local_dir = "."

print(f"Загрузка в датасет: {handle}")
print(f"Из папки: {os.getcwd()}")

try:
    kagglehub.dataset_upload(
        handle=handle,
        local_dataset_dir=local_dir,
        version_notes="$VERSION_NOTES"
        # ignore_patterns=['__pycache__/', '*.pyc', '.git/']  # раскомментировать при необходимости
    )
    print("✅ Успешно! Новая версия загружена.")
except Exception as e:
    print(f"❌ Ошибка загрузки: {e}")
    exit(1)
EOF

echo "🔗 Посмотреть датасет: https://www.kaggle.com/datasets/$DATASET_HANDLE"

sleep 2