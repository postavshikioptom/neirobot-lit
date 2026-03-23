import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activation(activation_type: str):
    """
    Возвращает функцию активации по имени.
    Поддерживаемые типы:
    - 'relu': Стандартная ReLU
    - 'gelu_exact': GELU с точным вычислением через erf
    - 'gelu_tanh': GELU с аппроксимацией через tanh (быстрее)
    - 'silu': SiLU (Swish) активация
    """
    activation_map = {
        'relu': nn.ReLU(),
        'gelu_exact': nn.GELU(approximate='none'),
        'gelu_tanh': nn.GELU(approximate='tanh'),
        'silu': nn.SiLU()
    }
    
    if activation_type not in activation_map:
        raise ValueError(f"Unsupported activation type: {activation_type}. "
                        f"Supported: {list(activation_map.keys())}")
    
    return activation_map[activation_type]

class LOBPatching(nn.Module):
    """
    Продвинутый слой патчинга для стакана (LOB) — Per-Level Token Architecture (Задача 319).
    Каждый уровень стакана — отдельный токен. Трансформер видит S*50 токенов
    и учит attention между уровнями (bid-loaded vs ask-loaded и т.д.).

    Вход:  (B, S, in_channels=11, n_levels=50)
    Выход: (B, S*50, d_model=96) — 5000 токенов для S=100
    """
    def __init__(self, seq_len=100, n_levels=50, in_channels=11, d_model=96, activation='gelu_exact'):
        super().__init__()
        self.d_model = d_model
        self.in_channels = in_channels
        self.n_levels = n_levels
        self.seq_len = seq_len

        # Per-level projection: each level's in_channels features → d_model
        self.level_proj = nn.Linear(in_channels, d_model)
        self.act = get_activation(activation)

        # Level Positional Embedding: (1, 1, n_levels, d_model)
        # Информация о глубине стакана (уровень 0 vs уровень 49)
        self.level_pos_emb = nn.Parameter(torch.randn(1, 1, n_levels, d_model) * 0.02)

        # Temporal Positional Embedding: (1, seq_len, 1, d_model)
        # Информация о временнóм порядке
        self.time_pos_emb = nn.Parameter(torch.randn(1, seq_len, 1, d_model) * 0.02)

        # CLS token for classification (Задача 319: learnable [CLS] token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Финальная нормализация
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (B, S, C, L) — C=11 каналов, L=50 уровней
        Возвращает: (B, S*L, d_model) — каждый уровень стакана как отдельный токен
        """
        b, s, c, l = x.shape

        # (B, S, C, L) → (B, S, L, C) — признаки одного уровня вместе
        x_perm = x.transpose(2, 3).contiguous()  # (B, S, 50, 11)

        # Per-level projection: (B*S, 50, 11) → (B*S, 50, d_model)
        x_flat = x_perm.view(b * s, l, c)
        x_proj = self.level_proj(x_flat)
        x_proj = self.act(x_proj)

        # Reshape back: (B, S, 50, d_model)
        x_tokens = x_proj.view(b, s, l, self.d_model)

        # Add positional embeddings (broadcasting)
        x_tokens = x_tokens + self.level_pos_emb + self.time_pos_emb

        # Flatten for transformer: (B, S*50, d_model)
        x_tokens_flat = x_tokens.view(b, s * l, self.d_model)

        # Добавляем CLS токен в начало каждой последовательности (Задача 319)
        cls_tokens = self.cls_token.expand(b, -1, -1)  # (B, 1, d_model)
        x_out = torch.cat([cls_tokens, x_tokens_flat], dim=1)  # (B, 1 + S*L, d_model)

        return self.norm(x_out)
