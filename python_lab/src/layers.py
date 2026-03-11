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
    Продвинутый слой патчинга для стакана (LOB).
    Группирует уровни стакана и добавляет уровневое и временное позиционное кодирование.
    """
    def __init__(self, seq_len=100, n_levels=50, in_channels=3, d_model=64, activation='gelu_exact'):
        super().__init__()
        self.d_model = d_model
        self.in_channels = in_channels
        self.n_levels = n_levels
        self.seq_len = seq_len
        
        # Динамический расчет размерностей на основе параметров
        # Согласно плану 026: num_features = in_channels * n_levels
        self.num_features = in_channels * n_levels
        # После Conv1d(kernel=in_channels, stride=in_channels): num_patches = num_features // in_channels = n_levels
        self.num_patches = self.num_features // in_channels
        
        # 1. Vertical Patching: Объединяем все каналы одного уровня через Conv1d
        # Вход: (Batch*Seq, 1, num_features) -> Выход: (Batch*Seq, d_model, num_patches)
        # kernel_size=in_channels, stride=in_channels объединяет все каналы одного уровня в один токен
        self.patch_conv = nn.Conv1d(1, d_model, kernel_size=in_channels, stride=in_channels)
        self.act = get_activation(activation)  # Активация после patch_conv
        
        # 2. Level Positional Embedding (динамический размер по плану 026)
        # После Conv1d(kernel=in_channels, stride=in_channels) получаем num_patches токенов
        self.level_pos_emb = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        
        # 3. Temporal Positional Embedding (строго seq_len по плану 026)
        # Различаем шаги времени 0..seq_len-1
        self.time_pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        # Шаг 3: Attention Pooling (Задача 310-3)
        # Слой для вычисления весов внимания для каждого уровня стакана
        self.level_attention = nn.Linear(d_model, 1)
        
        # 4. Финальная нормализация для стабильности
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (Batch, Seq, in_channels, n_levels) - входные данные из dataset
        Преобразуем в (Batch, Seq, 1, num_features) для патчинга.
        
        ВАЖНО (Задача 304): После перехода на блочный порядок (P, P, V, V) в features.py,
        нам нужно переставить оси, чтобы свертка по каналам объединяла признаки ОДНОГО уровня.
        
        Порядок в x: [channel_0, channel_1, channel_2] где каждый - 50 уровней.
        Нам нужно: [L0_C0, L0_C1, L0_C2, L1_C0, L1_C1, L1_C2, ...]
        """
        b, s, c, l = x.shape  # c=in_channels, l=n_levels
        
        # Шаг 0: Транспонируем (B, S, C, L) -> (B, S, L, C)
        # Это гарантирует, что признаки одного уровня идут подряд
        x_permuted = x.transpose(2, 3).contiguous() # (B, S, L, C)
        
        # Flatten в (B, S, 1, L*C)
        x_flat_seq = x_permuted.view(b, s, 1, l * c) # (B, S, 1, num_features)
        
        # Шаг 1: Vertical Patching - объединяем каналы одного уровня
        # Reshape в (B*S, 1, num_features) для применения Conv1d
        x_flat = x_flat_seq.view(b * s, 1, self.num_features)  # (B*S, 1, num_features)
        x_patched = self.patch_conv(x_flat)  # (B*S, d_model, num_patches)
        x_patched = self.act(x_patched)  # Применяем активацию
        x_patched = x_patched.permute(0, 2, 1)  # (B*S, num_patches, d_model)
        
        # Reshape обратно в батч: (B, S, num_patches, d_model)
        x_patched = x_patched.view(b, s, self.num_patches, self.d_model)
        
        # Шаг 2: Добавляем уровневые позиции (информация о глубине стакана)
        x_patched = x_patched + self.level_pos_emb  # Broadcasting: (B, S, num_patches, d_model)
        
        # Шаг 3: Агрегация уровней в один "Snapshot Token" на каждый шаг времени.
        # Сжимаем num_patches уровней в один вектор размерности D через Attention Pooling (Задача 310-3).
        # (B, S, num_patches, d_model) -> (B, S, d_model)
        attn_weights = F.softmax(self.level_attention(x_patched), dim=2)  # (B, S, num_patches, 1)
        x_snapshot = (x_patched * attn_weights).sum(dim=2)  # (B, S, d_model)
        
        # Шаг 4: Добавляем временные позиции (информация о порядке событий)
        # Динамически нарезаем позиционное кодирование под текущую длину последовательности
        x_temporal = x_snapshot + self.time_pos_emb[:, :s, :]
        
        return self.norm(x_temporal)
