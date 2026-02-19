# 025 - Python Lab LiT Architecture
Цель задачи: Реализовать архитектуру нейросети LiT (Lightweight Transformer). Модель использует вертикальный патчинг (Conv1d) для объединения пар цена/объем в токены уровней и CLS-токен для финальной классификации временной последовательности.

Файлы: python_lab/src/lit_model.py (создать)

Инструкции для Gemini: Реализовать класс LiTModel на PyTorch. Основная идея: сначала «схлопываем» 200 фич одного среза стакана в 100 токенов (по 2 фичи на уровень), затем агрегируем их во времени через Transformer.

import torch
import torch.nn as nn

class LiTModel(nn.Module):
    def __init__(self, seq_len=100, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        
        # 1. Vertical Patching: Группируем (price, volume) каждого уровня
        # Из (batch*seq, 1, 200) получаем (batch*seq, d_model, 100)
        self.patch_conv = nn.Conv1d(1, d_model, kernel_size=2, stride=2)
        
        # 2. Level Positional Embedding (для 100 уровней: 50 asks + 50 bids)
        self.level_pos_emb = nn.Parameter(torch.zeros(1, 100, d_model))
        
        # 3. [CLS] Токен для агрегации временной последовательности
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            batch_first=True, dropout=0.1, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5. Head (Ternary classification: 0=Flat, 1=Up, 2=Down)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, 200)
        b, s, f = x.shape
        
        # Патчинг уровней: (B*S, 1, 200) -> (B*S, 100, D)
        x_flat = x.view(b * s, 1, f)
        x_levels = self.patch_conv(x_flat).permute(0, 2, 1)
        x_levels = x_levels + self.level_pos_emb
        
        # Агрегация уровней в один вектор на шаг времени (Mean Pooling)
        # (B, S, D)
        x_temporal = x_levels.mean(dim=1).view(b, s, self.d_model)
        
        # Добавляем CLS токен: (B, S+1, D)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x_combined = torch.cat((cls_tokens, x_temporal), dim=1)
        
        # Transformer + Head
        x_trans = self.transformer(x_combined)
        return self.head(self.norm(x_trans[:, 0, :])) # Прогноз по CLS токену
Технические требования:
Вертикальный патчинг: Использовать Conv1d с kernel=2, stride=2 для обработки пар (price, volume).
CLS Токен: Обязателен для классификации всей последовательности.
Активация: Строго GELU (лучше для трансформеров).
Параметры: d_model=64 (баланс между скоростью Rust и точностью).
Почему это важно: Такая структура эффективно извлекает микро-паттерны стакана через свертки и учитывает временную динамику через механизм внимания. GELU и LayerNorm обеспечивают стабильность обучения.