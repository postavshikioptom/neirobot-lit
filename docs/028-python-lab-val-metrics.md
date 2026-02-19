# 028 - Python Lab Validation Metrics
Цель задачи: Внедрить систему глубокой оценки модели на несбалансированных данных. Вместо простой Accuracy использовать MCC (Matthews Correlation Coefficient) как основную метрику, а также добавить метрики по каждому классу (Precision, Recall, F1) и визуализацию матрицы ошибок в TensorBoard.

Файлы: python_lab/src/train.py (обновить)

Инструкции для Gemini:

Добавить метрики в LiTModule.__init__:
from torchmetrics.classification import (
    MulticlassMatthewsCorrCoef, 
    MulticlassPrecision, 
    MulticlassRecall, 
    MulticlassF1Score,
    MulticlassConfusionMatrix
)

# В __init__:
self.mcc = MulticlassMatthewsCorrCoef(num_classes=3)
self.f1_macro = MulticlassF1Score(num_classes=3, average="macro")

# Метрики по классам (0=Flat, 1=Up, 2=Down)
self.precision_per_class = MulticlassPrecision(num_classes=3, average=None)
self.recall_per_class = MulticlassRecall(num_classes=3, average=None)
self.conf_matrix = MulticlassConfusionMatrix(num_classes=3)
Обновить validation_step: Использовать автоматическое логирование torchmetrics.
def validation_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)
    loss = self.criterion(logits, y)
    
    # Логируем основные метрики на каждой эпохе
    self.log("val_loss", loss, prog_bar=True)
    self.log("val_mcc", self.mcc(logits, y), prog_bar=True, on_epoch=True)
    self.log("val_f1_macro", self.f1_macro(logits, y), on_epoch=True)
    
    # Обновляем матрицу ошибок (без логирования на каждом шаге)
    self.conf_matrix.update(logits, y)
    
    # Обновляем метрики по классам
    self.precision_per_class.update(logits, y)
    self.recall_per_class.update(logits, y)
Добавить on_validation_epoch_end: Логировать детализированные отчеты по классам в TensorBoard.
def on_validation_epoch_end(self):
    # Логируем Precision/Recall для Up и Down классов отдельно
    prec = self.precision_per_class.compute()
    rec = self.recall_per_class.compute()
    
    self.log("val_prec_up", prec[1])
    self.log("val_rec_up", rec[1])
    self.log("val_prec_down", prec[2])
    self.log("val_rec_down", rec[2])
    
    # Сброс метрик
    self.precision_per_class.reset()
    self.recall_per_class.reset()
    # Матрицу ошибок можно выводить в логи как текст или через кастомный плоттер
Технические требования:

MCC: Основной ориентир для выбора лучшей модели (вместо Acc).
Macro F1: Позволяет оценить качество предсказания редких классов (Up/Down).
Per-class metrics: Обязательно выводить Precision и Recall для классов 1 и 2.
Reset: Не забывать сбрасывать метрики в конце эпохи.
Почему это важно: В стакане 90% времени цена стоит (Flat). Модель может получить 90% Accuracy, никогда не предсказывая движение. MCC и Precision/Recall для классов Up/Down — единственный способ понять, действительно ли нейросеть находит торговые сигналы.