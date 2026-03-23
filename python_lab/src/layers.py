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
    Compact Snapshot-Style Patching для LOB (Задача 321).
    Сжимает 50 уровней стакана в один snapshot token на временном шаге через
    patch_conv + level_attention weighted sum. Возвращает (B, S, d_model).
    Эффективно avoids O((S*L)^2) attention cost.
    """
    def __init__(self, seq_len=100, n_levels=50, in_channels=11, d_model=96, activation='gelu_exact'):
        super().__init__()
        self.d_model = d_model
        self.in_channels = in_channels
        self.n_levels = n_levels
        self.seq_len = seq_len
        self.num_patches = n_levels

        # Patch convolution: сжимает каждый уровень (in_channels признаков) в d_model
        # Input: (batch*n_patches, 1, in_channels * n_levels?) Actually we treat each level independently.
        # Используем Conv1d с kernel_size=in_channels, stride=in_channels
        self.patch_conv = nn.Conv1d(1, d_model, kernel_size=in_channels, stride=in_channels)

        # Level Positional Embedding: (1, n_levels, d_model)
        self.level_pos_emb = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)

        # Temporal Positional Embedding: (1, seq_len, d_model)
        self.time_pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # Level Attention: взвешивание признаков по глубине стакана
        self.level_attention = nn.Linear(d_model, 1)
        self.pre_attn_norm = nn.LayerNorm(d_model)

        self.norm = nn.LayerNorm(d_model)
        self.act = get_activation(activation)

    def forward(self, x):
        """
        x: (B, S, C, L) — C=11 каналов, L=50 уровней
        Возвращает: (B, S, d_model) — один компактный токен на временной шаг
        """
        b, s, c, l = x.shape
        # (B, S, C, L) -> (B, S, L, C)
        x_perm = x.transpose(2, 3).contiguous()  # (B, S, L, C)

        # Объединяем C и L в плоскую последовательность для conv1d
        # (B, S, L, C) -> (B*S, 1, L*C)
        x_flat = x_perm.view(b * s, 1, l * c)

        # Применяем patch_conv: (B*S, 1, L*C) -> (B*S, d_model, L)
        x_patched = self.patch_conv(x_flat)
        x_patched = self.act(x_patched).permute(0, 2, 1)  # (B*S, L, d_model)

        # Восстанавливаем форму (B, S, L, d_model)
        x_patched = x_patched.view(b, s, self.num_patches, self.d_model)

        # Добавляем level positional embedding
        x_patched = x_patched + self.level_pos_emb

        # Level attention: взвешивание по уровню (глубине стакана)
        x_patched_norm = self.pre_attn_norm(x_patched)
        attn_weights = F.softmax(self.level_attention(x_patched_norm), dim=2)  # (B, S, L, 1)
        x_snapshot = (x_patched_norm * attn_weights).sum(dim=2)  # (B, S, d_model)

        # Добавляем temporal positional embedding
        x_temporal = x_snapshot + self.time_pos_emb[:, :s, :]

        return self.norm(x_temporal)
