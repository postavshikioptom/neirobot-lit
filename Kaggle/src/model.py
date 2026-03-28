import torch
import torch.nn as nn


class MultiTaskTransformer(nn.Module):
    """
    Архитектура Multi-Task Learning для одновременного предсказания сигнала и волатильности.
    
    Использует Bottleneck слои для изоляции специфических шумов задач и Global Average Pooling
    для агрегации информации со всех патчей.
    
    Согласно плану задачи 130.
    """
    
    def __init__(self, backbone, hidden_dim, num_classes=3):
        """
        Args:
            backbone: модель-базис (например, LiTModel из lit_model.py)
            hidden_dim: размерность скрытого слоя
            num_classes: количество классов для классификации (по умолчанию 3: [UP, DOWN, NEUTRAL])
        """
        super().__init__()
        self.backbone = backbone
        
        # Bottleneck слои для изоляции специфических шумов задач
        self.class_bottleneck = nn.Linear(hidden_dim, hidden_dim)
        self.vol_bottleneck = nn.Linear(hidden_dim, hidden_dim)
        
        # Головы для двух задач
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.vol_regressor = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        """
        Args:
            x: входные данные (Batch, Seq_Len, Features)
            
        Returns:
            logits: предсказания класса (Batch, num_classes)
            vol: предсказания волатильности (Batch, 1)
        """
        # Получаем признаки от backbone (Batch, Patches, Hidden_Dim)
        features = self.backbone(x)
        
        # Global Average Pooling по патчам
        pooled = features.mean(dim=1)
        
        # Разделение на ветки
        logits = self.classifier(torch.relu(self.class_bottleneck(pooled)))
        vol = self.vol_regressor(torch.relu(self.vol_bottleneck(pooled)))
        
        return logits, vol
