import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from .layers import LOBPatching

# Глобальная конфигурация входных данных (по умолчанию)
DEFAULT_INPUT_CHANNELS = 3
N_LEVELS = 50


@dataclass
class LiTConfig:
    """
    Конфигурация для модели LiT.
    Используется для удобной инициализации моделей с разными параметрами,
    особенно полезно для Knowledge Distillation (Teacher vs Student).
    """
    seq_len: int = 100
    in_channels: int = 3
    d_model: int = 64
    embed_dim: int = None  # Алиас для d_model (для совместимости с задачей 237)
    nhead: int = 4
    num_heads: int = None  # Алиас для nhead (для совместимости с задачей 237)
    num_layers: int = 2
    dropout: float = 0.1
    activation: str = 'gelu_exact'
    multi_task: bool = True
    num_regimes: int = 0  # Количество режимов рынка (0 = отключено)
    regime_embedding_dim: int = 16  # Размерность embedding для режимов
    num_horizons: int = 1  # Количество горизонтов предсказания (Задача 160)
    use_horizon_embedding: bool = False  # Использовать Horizon Embedding вместо отдельных голов
    use_gqa: bool = False  # Использовать Grouped Query Attention (опционально, задача 237)

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

def get_activation_for_transformer(activation_type: str):
    """
    Возвращает активацию для TransformerEncoderLayer.
    TransformerEncoderLayer поддерживает строки ('relu', 'gelu') или unary callable.
    Для SiLU и вариантов GELU используем callable для полной консистентности.
    """
    if activation_type == 'relu':
        return 'relu'
    elif activation_type == 'gelu_exact':
        # Используем callable для точного контроля
        return nn.GELU(approximate='none')
    elif activation_type == 'gelu_tanh':
        # Используем callable для точного контроля
        return nn.GELU(approximate='tanh')
    elif activation_type == 'silu':
        # SiLU через callable - полная поддержка
        return nn.SiLU()
    else:
        raise ValueError(f"Unsupported activation type: {activation_type}")

def compute_curvature_penalty(model, inputs, outputs, lambda_=1e-4, epsilon=1e-3, regime_id=None, **kwargs):
    """
    Вычисляет штраф за кривизну (Curvature Penalty) через конечные разности.
    
    Аппроксимирует кривизну поверхности функции потерь путем возмущения входов
    в случайном направлении и измерения изменения предсказаний модели.
    Это вычислительно эффективнее прямого вычисления Гессиана.
    
    Args:
        model: модель PyTorch (должна быть в режиме training)
        inputs: входные данные (Batch, Seq, Channels, Levels)
        outputs: текущие предсказания модели (logits)
        lambda_: коэффициент регуляризации (рекомендуется 1e-4 - 1e-3)
        epsilon: величина возмущения для конечных разностей
        regime_id: идентификатор режима рынка (опционально)
    
    Returns:
        torch.Tensor: скалярное значение штрафа за кривизну
    
    Задача 238: Регуляризация кривизны и устойчивость к шуму
    """
    if not model.training:
        return torch.tensor(0.0, device=inputs.device)

    # Генерируем случайное направление шума
    v = torch.randn_like(inputs)
    v = v / (torch.norm(v, p=2) + 1e-6)  # Нормализация вектора
    
    # Инференс с возмущенными входами
    perturbed_inputs = inputs + epsilon * v
    perturbed_outputs = model(perturbed_inputs, regime_id=regime_id, **kwargs)
    
    # Обрабатываем случай когда модель возвращает кортеж (logits, vol)
    if isinstance(perturbed_outputs, tuple):
        perturbed_outputs = perturbed_outputs[0]  # Берем только logits
    
    # Обрабатываем случай когда outputs - кортеж
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    
    # Штраф за разницу предсказаний (L2)
    diff = perturbed_outputs - outputs
    return lambda_ * (diff ** 2).mean()

def apply_input_noise(x, std=0.01):
    """
    Применяет гауссов шум к входным данным для аугментации.
    
    Добавляет случайный гауссов шум с заданным стандартным отклонением
    к нормализованным признакам. Используется во время обучения для
    повышения устойчивости модели к шуму в данных стакана.
    
    Args:
        x: входные данные (Batch, Seq, Channels, Levels)
        std: стандартное отклонение гауссова шума (рекомендуется 0.005)
    
    Returns:
        torch.Tensor: зашумленные входные данные той же формы что и x
    
    Задача 238: Регуляризация кривизны и устойчивость к шуму
    """
    noise = torch.randn_like(x) * std
    return x + noise

class CustomTransformerEncoderLayer(nn.Module):
    """
    Кастомный Transformer Encoder Layer с явным вызовом scaled_dot_product_attention.
    Поддерживает как Multi-Head Attention (MHA), так и Grouped Query Attention (GQA).
    
    Задача 237: Требование А.2 - использование F.scaled_dot_product_attention
    """
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1, activation='gelu_exact', use_gqa=False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_gqa = use_gqa
        
        # GQA: уменьшаем количество KV heads
        if use_gqa:
            self.num_kv_heads = max(1, num_heads // 4)  # Группировка 4:1
        else:
            self.num_kv_heads = num_heads
        
        # Проекции Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Feedforward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Dropout и LayerNorm
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Активация
        self.activation = get_activation(activation)
        
    def forward(self, x, src_key_padding_mask=None):
        """
        x: (batch, seq_len, d_model)
        src_key_padding_mask: (batch, seq_len) - True для padding позиций
        """
        # Self-Attention с residual connection
        x = x + self._sa_block(x, src_key_padding_mask)
        
        # Feedforward с residual connection
        x = x + self._ff_block(x)
        
        return x
    
    def _sa_block(self, x, attn_mask=None):
        """Self-Attention block с явным вызовом scaled_dot_product_attention"""
        batch_size, seq_len, _ = x.shape
        
        # Проецируем в Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, d_model)
        k = self.k_proj(x)  # (batch, seq_len, num_kv_heads * head_dim)
        v = self.v_proj(x)  # (batch, seq_len, num_kv_heads * head_dim)
        
        # Reshape для multi-head attention
        # Q: (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # K, V: (batch, seq_len, num_kv_heads, head_dim) -> (batch, num_kv_heads, seq_len, head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # GQA: расширяем KV heads для каждой группы Query heads
        if self.use_gqa and self.num_kv_heads < self.num_heads:
            # Повторяем каждый KV head для группы Query heads
            # (batch, num_kv_heads, seq_len, head_dim) -> (batch, num_heads, seq_len, head_dim)
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        
        # Задача 237 А.2: Явный вызов scaled_dot_product_attention
        # Это гарантирует использование Flash Attention на поддерживаемом железе
        
        # Обработка маски для SDPA
        # LiTModel передает маску где True=padding (игнорировать)
        # SDPA ожидает маску где True=valid (учитывать)
        # Поэтому инвертируем и добавляем размерности для broadcasting
        if attn_mask is not None:
            # Инвертируем: True -> False, False -> True
            # Добавляем размерности: (B, S) -> (B, 1, 1, S) для broadcasting
            attn_mask = (~attn_mask).unsqueeze(1).unsqueeze(2)
        
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0
        )
        
        # Reshape обратно: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout1(attn_output)
        
        # LayerNorm
        return self.norm1(attn_output)
    
    def _ff_block(self, x):
        """Feedforward block"""
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x2 = self.dropout2(x2)
        return self.norm2(x2)

class LiTModel(nn.Module):
    """
    Lightweight Transformer (LiT) для анализа данных стакана (LOB).
    Использует LOBPatching для предобработки и Transformer для классификации последовательности.
    
    Поддерживает Multi-Horizon Prediction (Задача 160):
    - Может предсказывать на нескольких временных масштабах одновременно
    - Выход формы (batch, num_horizons, 3) для multi-horizon
    - Опциональный Horizon Embedding для кондиционирования
    """
    def __init__(self, seq_len=100, in_channels=3, d_model=64, embed_dim=None, nhead=4, num_heads=None, num_layers=2, dropout=0.1, activation='gelu_exact', multi_task=True, num_regimes=0, regime_embedding_dim=16, num_horizons=1, use_horizon_embedding=False, use_gqa=False):
        super().__init__()
        
        # Поддержка алиасов для совместимости с задачей 237
        if embed_dim is not None:
            d_model = embed_dim
        if num_heads is not None:
            nhead = num_heads
        
        # Валидация архитектуры
        assert d_model % nhead == 0, f"embed_dim ({d_model}) must be divisible by num_heads ({nhead})"
        
        self.d_model = d_model
        self.in_channels = in_channels
        self.activation_type = activation
        self.multi_task = multi_task
        self.num_regimes = num_regimes
        self.regime_embedding_dim = regime_embedding_dim
        self.num_horizons = num_horizons
        self.use_horizon_embedding = use_horizon_embedding
        self.num_heads = nhead
        self.head_dim = d_model // nhead
        self.use_gqa = use_gqa
        self.seq_len = seq_len
        
        # 1. Продвинутый слой патчинга (с уровневым и временным позиционированием)
        self.patching = LOBPatching(
            seq_len=seq_len, 
            n_levels=N_LEVELS, 
            in_channels=in_channels, 
            d_model=d_model,
            activation=activation
        )
        
        # 1.5. Positional Encoding (Задача 055)
        # Поддерживает динамическое слicing до текущей seq_len
        # Используем sinusoidal PE (не требует обучения)
        max_seq_len = seq_len + 1  # +1 для [CLS] токена
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        # Регистрируем как буфер (не параметр, не требует градиентов)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_seq_len, d_model)
        
        # 2. [CLS] Токен (сохраняем для совместимости, но будем использовать GAP)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # 3. Regime Embedding (опционально, если num_regimes > 0)
        if num_regimes > 0:
            self.regime_embedding = nn.Embedding(num_regimes, regime_embedding_dim)
            # Проекция для добавления regime embedding к патчам
            self.regime_projection = nn.Linear(d_model + regime_embedding_dim, d_model)
            nn.init.xavier_uniform_(self.regime_projection.weight)
        else:
            self.regime_embedding = None
            self.regime_projection = None
        
        # 3.5. Horizon Embedding (опционально, если use_horizon_embedding=True)
        if use_horizon_embedding and num_horizons > 1:
            self.horizon_embedding = nn.Embedding(num_horizons, d_model)
            nn.init.trunc_normal_(self.horizon_embedding.weight, std=0.02)
        else:
            self.horizon_embedding = None
        
        # 4. Transformer Encoder
        # Задача 237: Кастомная реализация с явным вызовом scaled_dot_product_attention
        # Используем CustomTransformerEncoderLayer для поддержки GQA и Flash Attention
        self.transformer_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model=d_model,
                num_heads=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation=activation,
                use_gqa=use_gqa
            )
            for _ in range(num_layers)
        ])

        # 5. Multi-Task Heads
        self.norm = nn.LayerNorm(d_model)
        
        # Bottleneck слои для изоляции специфических шумов задач
        self.class_bottleneck = nn.Linear(d_model, d_model)
        self.vol_bottleneck = nn.Linear(d_model, d_model)
        
        # Активация для bottleneck
        self.bottleneck_act = get_activation(activation)
        
        # Головы классификации (Multi-Horizon)
        if use_horizon_embedding and num_horizons > 1:
            # Одна голова, кондиционированная на horizon embedding
            self.classifier = nn.Linear(d_model, 3)
        else:
            # Отдельные головы для каждого горизонта (более эффективно через один Linear)
            self.classifier = nn.Linear(d_model, 3 * num_horizons)
        
        self.vol_regressor = nn.Linear(d_model, 1) # Предсказание Log-Vol

        # Инициализация параметров
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.regime_embedding is not None:
            nn.init.trunc_normal_(self.regime_embedding.weight, std=0.02)
        
        # Инициализация bottleneck слоёв
        nn.init.xavier_uniform_(self.class_bottleneck.weight)
        nn.init.zeros_(self.class_bottleneck.bias)
        nn.init.xavier_uniform_(self.vol_bottleneck.weight)
        nn.init.zeros_(self.vol_bottleneck.bias)
        
        # Инициализация голов
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        nn.init.xavier_uniform_(self.vol_regressor.weight)
        nn.init.zeros_(self.vol_regressor.bias)
    
    @classmethod
    def from_config(cls, config: LiTConfig, **kwargs):
        """
        Создает модель из объекта конфигурации.
        Полезно для Knowledge Distillation, когда нужно создать
        Teacher и Student с разными конфигурациями.
        
        Args:
            config: объект LiTConfig с параметрами модели
            **kwargs: дополнительные параметры, переопределяющие config
        
        Returns:
            LiTModel: инициализированная модель
        
        Example:
            >>> teacher_config = LiTConfig(d_model=256, nhead=8, num_layers=8)
            >>> teacher = LiTModel.from_config(teacher_config)
            >>> 
            >>> student_config = LiTConfig(d_model=64, nhead=4, num_layers=2)
            >>> student = LiTModel.from_config(student_config)
        """
        return cls(
            seq_len=kwargs.get('seq_len', config.seq_len),
            in_channels=kwargs.get('in_channels', config.in_channels),
            d_model=kwargs.get('d_model', config.d_model),
            nhead=kwargs.get('nhead', config.nhead),
            num_layers=kwargs.get('num_layers', config.num_layers),
            dropout=kwargs.get('dropout', config.dropout),
            activation=kwargs.get('activation', config.activation),
            multi_task=kwargs.get('multi_task', config.multi_task),
            num_regimes=kwargs.get('num_regimes', config.num_regimes),
            regime_embedding_dim=kwargs.get('regime_embedding_dim', config.regime_embedding_dim),
            num_horizons=kwargs.get('num_horizons', config.num_horizons),
            use_horizon_embedding=kwargs.get('use_horizon_embedding', config.use_horizon_embedding),
            use_gqa=kwargs.get('use_gqa', config.use_gqa)
        )

    def forward(self, x, mask=None, regime_id=None):
        """
        x: (Batch, Seq, in_channels, n_levels) - входные данные из dataset
           По умолчанию: (Batch, Seq, 3, 50) - 3 канала × 50 уровней
        mask: (Batch, Seq)
        regime_id: (Batch,) - индексы режимов рынка (опционально)
        
        Returns:
            logits: (Batch, num_horizons, 3) для multi-horizon или (Batch, 3) для single horizon
            vol: (Batch, 1) - предсказание волатильности (если multi_task=True)
        """
        b, s, c, l = x.shape
        
        # Шаг 1: Патчинг и позиционное кодирование
        x = self.patching(x)  # (Batch, num_patches, d_model)
        
        # Шаг 1.5: Добавляем Positional Encoding (Задача 055)
        # Срезаем PE до текущей seq_len и добавляем к патчам
        pe_slice = self.pe[:, :x.shape[1], :]  # (1, num_patches, d_model)
        x = x + pe_slice  # (Batch, num_patches, d_model)
        
        # Шаг 2: Добавляем Regime Embedding (если включено)
        if self.regime_embedding is not None and regime_id is not None:
            # Получаем regime embedding для каждого семпла в батче
            regime_emb = self.regime_embedding(regime_id)  # (Batch, regime_embedding_dim)
            
            # Расширяем на все патчи: (Batch, 1, regime_embedding_dim) -> (Batch, num_patches, regime_embedding_dim)
            regime_emb = regime_emb.unsqueeze(1).expand(-1, x.shape[1], -1)
            
            # Конкатенируем с патчами и проецируем обратно в d_model
            x = torch.cat([x, regime_emb], dim=-1)  # (Batch, num_patches, d_model + regime_embedding_dim)
            x = self.regime_projection(x)  # (Batch, num_patches, d_model)
        
        # Шаг 3: Добавляем [CLS] токен
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Шаг 4: Подготовка маски для Transformer (Задача 055)
        src_key_padding_mask = None
        if mask is not None:
            # mask: (Batch, Seq) - True для padding, False для valid
            # Добавляем False для [CLS] токена (он всегда valid)
            cls_mask = torch.zeros((b, 1), dtype=torch.bool, device=x.device)
            src_key_padding_mask = torch.cat([cls_mask, mask], dim=1)  # (Batch, Seq+1)
        
        # Шаг 5: Transformer Encoder с поддержкой src_key_padding_mask
        # src_key_padding_mask: True для позиций, которые нужно игнорировать
        x_trans = x
        for layer in self.transformer_layers:
            x_trans = layer(x_trans, src_key_padding_mask=src_key_padding_mask)
        
        # Шаг 6: Global Average Pooling по патчам (исключая CLS токен) согласно плану 130
        pooled = x_trans[:, 1:, :].mean(dim=1)
        pooled = self.norm(pooled)
        
        # Шаг 7: Разделение на ветки (Multi-Task)
        # Классификация с Multi-Horizon поддержкой
        if self.use_horizon_embedding and self.num_horizons > 1:
            # Horizon Embedding подход: одна голова, кондиционированная на horizon
            # Создаем предсказания для каждого горизонта
            horizon_ids = torch.arange(self.num_horizons, device=x.device)  # (num_horizons,)
            horizon_embs = self.horizon_embedding(horizon_ids)  # (num_horizons, d_model)
            
            # Расширяем pooled для батча: (batch, d_model) -> (batch, 1, d_model)
            pooled_expanded = pooled.unsqueeze(1)  # (batch, 1, d_model)
            
            # Добавляем horizon embedding: (batch, 1, d_model) + (1, num_horizons, d_model)
            horizon_embs_expanded = horizon_embs.unsqueeze(0)  # (1, num_horizons, d_model)
            pooled_with_horizon = pooled_expanded + horizon_embs_expanded  # (batch, num_horizons, d_model)
            
            # Применяем bottleneck и classifier
            bottleneck_out = self.bottleneck_act(self.class_bottleneck(pooled_with_horizon))  # (batch, num_horizons, d_model)
            logits = self.classifier(bottleneck_out)  # (batch, num_horizons, 3)
        else:
            # Стандартный подход: отдельные головы (через один Linear)
            bottleneck_out = self.bottleneck_act(self.class_bottleneck(pooled))  # (batch, d_model)
            logits_flat = self.classifier(bottleneck_out)  # (batch, 3 * num_horizons)
            
            if self.num_horizons > 1:
                # Reshape в (batch, num_horizons, 3)
                logits = logits_flat.view(b, self.num_horizons, 3)
            else:
                # Single horizon: (batch, 3)
                logits = logits_flat
        
        if not self.training and not self.multi_task:
            return logits
            
        vol = self.vol_regressor(self.bottleneck_act(self.vol_bottleneck(pooled)))
        
        if self.multi_task:
            return logits, vol
        else:
            return logits

if __name__ == "__main__":
    # Тест архитектуры с разным количеством каналов и активаций
    print("Testing with 3 channels (baseline) and GELU (exact):")
    model = LiTModel(seq_len=100, in_channels=3, activation='gelu_exact')
    dummy_input = torch.randn(8, 100, 3, 50)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # Ожидаем (8, 3)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("\nTesting with 6 channels (3 baseline + 3 past returns) and SiLU:")
    model_with_returns = LiTModel(seq_len=100, in_channels=6, activation='silu')
    dummy_input_6ch = torch.randn(8, 100, 6, 50)
    output_6ch = model_with_returns(dummy_input_6ch)
    
    print(f"Input shape: {dummy_input_6ch.shape}")
    print(f"Output shape: {output_6ch.shape}") # Ожидаем (8, 3)
    
    total_params_6ch = sum(p.numel() for p in model_with_returns.parameters())
    print(f"Total parameters: {total_params_6ch:,}")
    
    print("\nTesting Multi-Horizon (3 horizons) with separate heads:")
    model_multi = LiTModel(seq_len=100, in_channels=3, num_horizons=3, use_horizon_embedding=False)
    output_multi = model_multi(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output_multi.shape}") # Ожидаем (8, 3, 3)
    
    total_params_multi = sum(p.numel() for p in model_multi.parameters())
    print(f"Total parameters: {total_params_multi:,}")
    
    print("\nTesting Multi-Horizon (3 horizons) with Horizon Embedding:")
    model_multi_emb = LiTModel(seq_len=100, in_channels=3, num_horizons=3, use_horizon_embedding=True)
    output_multi_emb = model_multi_emb(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output_multi_emb.shape}") # Ожидаем (8, 3, 3)
    
    total_params_multi_emb = sum(p.numel() for p in model_multi_emb.parameters())
    print(f"Total parameters: {total_params_multi_emb:,}")
    
    print("\nTesting all activation types:")
    for act_type in ['relu', 'gelu_exact', 'gelu_tanh', 'silu']:
        model_test = LiTModel(seq_len=100, in_channels=3, activation=act_type)
        out_test = model_test(dummy_input)
        print(f"  {act_type}: Output shape {out_test.shape}, Mean: {out_test.mean().item():.4f}")
