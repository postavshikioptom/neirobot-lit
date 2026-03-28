#!/bin/bash
# =============================================
# upload_to_kaggle.sh
# Загружает ВСЁ содержимое текущей папки (файлы + все подпапки)
# Скрипт должен лежать внутри папки, которую ты хочешь загружать
# Те файлы и папки, которых нет здесь, удаляться и не сервере Kaggle
# =============================================

set -e  # остановка при любой ошибке

# ================= НАСТРОЙКИ =================
DATASET_SLUG="postavshikioptom11/neirobot-lit-data"
VERSION_MESSAGE="auto-update $(date '+%Y-%m-%d %H:%M:%S') — full folder upload"
# =============================================

echo "🚀 Запуск загрузки в Kaggle dataset: $DATASET_SLUG"
echo "📁 Загружается всё содержимое текущей папки (файлы + подпапки)"

# Переходим точно в папку, где лежит скрипт
cd "$(dirname "$0")"

# Проверяем наличие kaggle CLI
if ! command -v kaggle &> /dev/null; then
    echo "❌ Ошибка: kaggle CLI не найден. Установи командой: pip install kaggle"
    exit 1
fi

# Создаём dataset-metadata.json (обязательно для версии датасета)
cat > "dataset-metadata.json" << EOF
{
  "title": "neirobot-lit-data",
  "id": "$DATASET_SLUG",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}
EOF

echo "✅ dataset-metadata.json создан"

# Показываем, что будет загружено (для удобства)
echo "📋 Содержимое папки для загрузки:"
ls -la

# Загружаем новую версию датасета
# Kaggle автоматически берёт все файлы и подпапки из указанной папки
echo "📤 Загружаю новую версию на Kaggle..."
kaggle datasets version -p . -m "$VERSION_MESSAGE" --quiet

echo "🎉 Готово! Новая версия загружена: $VERSION_MESSAGE"
echo "🔗 Ссылка на датасет: https://www.kaggle.com/datasets/$DATASET_SLUG"

sleep 2