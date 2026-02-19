# 029 - Python Lab Early Stopping & Schedulers
Цель задачи: Оптимизировать процесс обучения, переключив Early Stopping на мониторинг MCC (вместо Loss) и добавив планировщик скорости обучения (Learning Rate Scheduler). Это позволит модели не просто минимизировать ошибку, а максимизировать реальную прогностическую точность, а также «дожимать» результат на плато.

Файлы: python_lab/src/train.py (обновить)

Инструкции для Gemini:

Обновить EarlyStopping в Trainer: Переключить мониторинг на val_mcc (так как это ключевая метрика для торгового сигнала).
early_stop_callback = EarlyStopping(
    monitor="val_mcc", 
    patience=15,        # Увеличить терпение для MCC, так как метрика шумнее лосса
    mode="max"          # Максимизируем корреляцию
)
Обновить ModelCheckpoint: Сохранять веса на основе лучшего значения MCC.
checkpoint_callback = ModelCheckpoint(
    monitor="val_mcc",
    filename="lit-{epoch:02d}-{val_mcc:.4f}",
    save_top_k=3,
    mode="max"
)
Добавить планировщик в configure_optimizers: Использовать ReduceLROnPlateau для снижения LR при замедлении роста MCC.
def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = {
        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        ),
        "monitor": "val_mcc",
    }
    return [optimizer], [scheduler]
Технические требования:

Monitor: Использовать val_mcc как основной индикатор качества (mode="max").
Scheduler: Коэффициент снижения (factor) 0.5, терпение (patience) меньше, чем у EarlyStopping.
Логирование: Убедиться, что текущий learning_rate логируется в TensorBoard.
Почему это важно: Минимизация CrossEntropyLoss не всегда означает рост прибыли или точности сигналов. Переход на Early Stopping по MCC гарантирует, что мы остановим обучение именно тогда, когда модель перестанет улучшать качество торговых сигналов. Планировщик LR позволяет модели точнее настроиться в конце обучения.

Ожидаемый результат: Обучение останавливается по достижении максимума MCC, LR снижается при стагнации метрики.