# 026 - Python Lab Patching Layer
Цель задачи: Реализовать продвинутый слой патчинга LOBPatching. Слой должен не просто «схлопывать» уровни, а сохранять их структуру для трансформера, добавлять временное (temporal) и уровневое (level) позиционное кодирование. Это позволит модели различать не только время, но и глубину стакана (best price vs deep levels).

Файлы: python_lab/src/layers.py (создать), python_lab/src/lit_model.py (обновить)

Инструкции для Gemini:

python_lab/src/layers.py: Реализовать слой, который превращает сырой снимок стакана в последовательность информативных токенов.
import torch
import torch.nn as nn

class LOBPatching(nn.Module):
    def __init__(self, seq_len=100, n_levels=100, d_model=64):
        super().__init__()
        # 1. Свертка по парам (price, volume). kernel=2, stride=2
        # Из 200 фич получаем 100 токенов (50 asks + 50 bids)
        self.patch_conv = nn.Conv1d(1, d_model, kernel_size=2, stride=2)
        
        # 2. Level Positional Embedding (различаем уровни 0..99)
        self.level_pos_emb = nn.Parameter(torch.randn(1, n_levels, d_model) * 0.02)
        
        # 3. Temporal Positional Embedding (различаем шаги времени 0..seq_len-1)
        self.time_pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (Batch, Seq, 200)
        b, s, f = x.shape
        
        # Шаг 1: Патчинг уровней. (B*S, 1, 200) -> (B*S, 100, D)
        x = x.view(b * s, 1, f)
        x = self.patch_conv(x).permute(0, 2, 1)
        
        # Шаг 2: Добавляем уровневые позиции
        x = x + self.level_pos_emb # (B*S, 100, D)
        
        # Шаг 3: Агрегация уровней (вместо mean используем взвешенную сумму или проекцию)
        # Чтобы не раздувать последовательность до 10,000 токенов, 
        # сжимаем 100 уровней в 1 компактный "Snapshot Token"
        x = x.mean(dim=1).view(b, s, -1) # (B, S, D)
        
        # Шаг 4: Добавляем временные позиции
        x = x + self.time_pos_emb
        
        return self.norm(x)
python_lab/src/lit_model.py: Интегрировать LOBPatching и добавить CLS-токен перед подачей в Transformer.
Технические требования:

Изоляция позиций: Раздельное кодирование для уровней (внутри snapshot) и для времени (между snapshots).
Схема: Conv1d должен строго соответствовать interleaved схеме из задачи 012 (200 фич -> 100 патчей).
Стабильность: LayerNorm обязателен после всех сложений с pos_emb.
CLS Token: Модель должна конкатенировать обучаемый cls_token к последовательности в forward основной модели.
Почему это важно: Grok прав: простая агрегация уровней без понимания их «глубины» (level pos) теряет важную информацию о форме стакана. Добавление temporal pos позволяет трансформеру понимать порядок событий, а не просто видеть набор снимков.

Ожидаемый результат: Слой патчинга готов, модель использует CLS-токен для предсказания. Проект компилируется/запускается в Python.