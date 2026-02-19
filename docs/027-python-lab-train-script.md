# 027 - Python Lab Train Script
Цель задачи: Реализовать полный цикл обучения модели с использованием PyTorch Lightning. Скрипт должен включать эффективный Dataset со скользящим окном, разделение данных на train/val, расчет метрик (Accuracy, F1) и использование Callbacks (EarlyStopping, Checkpoint).

Файлы: python_lab/src/dataset.py (создать), python_lab/src/train.py (создать)

Инструкции для Gemini:

python_lab/src/dataset.py: Реализовать LOBDataset с использованием срезов numpy для экономии памяти (zero-copy).
import torch
import numpy as np
from torch.utils.data import Dataset

class LOBDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, seq_len: int = 100):
        # features: (N, 200), labels: (N,)
        self.features = features
        self.labels = labels
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        # Возвращаем окно и метку для следующего за окном шага (future horizon)
        x = self.features[idx : idx + self.seq_len]
        y = self.labels[idx + self.seq_len]
        return torch.from_numpy(x).float(), torch.tensor(y).long()
python_lab/src/train.py: Создать LiTModule и настроить Trainer.
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from .lit_model import LiTModel
from .dataset import LOBDataset

class LiTModule(pl.LightningModule):
    def __init__(self, seq_len=100, **model_params):
        super().__init__()
        self.save_hyperparameters()
        self.model = LiTModel(seq_len=seq_len, **model_params)
        self.criterion = nn.CrossEntropyLoss()
        self.acc = MulticlassAccuracy(num_classes=3)
        self.f1 = MulticlassF1Score(num_classes=3)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.acc(logits, y), prog_bar=True)
        self.log("val_f1", self.f1(logits, y), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-5)

# В main: загрузка данных через polars, pl.seed_everything(42), random_split(80/20), Trainer.fit
Технические требования:

Разделение: Строго 80% train / 20% validation через random_split.
Метрики: Accuracy и F1-score (multiclass) обязательны для мониторинга дисбаланса.
Оптимизация: Использовать AdamW, num_workers в DataLoader и pin_memory=True для GPU.
Callbacks: EarlyStopping (patience=10) и ModelCheckpoint для сохранения лучших весов.
Почему это важно: PyTorch Lightning стандартизирует процесс обучения, делая его воспроизводимым. Скользящее окно позволяет модели видеть динамику стакана, а метрика F1 критична для оценки качества на несбалансированных финансовых данных.

Ожидаемый результат: Скрипт успешно запускает обучение, логи отображаются в TensorBoard.