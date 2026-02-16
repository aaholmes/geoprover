"""GeoTransformer: Text-based transformer for geometry theorem proving.

Architecture:
  - Input: text sequence (proof state as relation list + goal)
  - Backbone: 6-layer transformer encoder, d_model=256, 8 heads
  - Value head: MLP on [CLS] embedding → (v_logit, k)
  - Policy head: dot(state_emb, construction_emb) for each candidate

Replaces the CNN (GeoNet) which suffered from permutation sensitivity.
The old GeoNet CNN is kept as GeoNetCNN for ablation.

~5-8M parameters.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Tokenizer: tiny vocabulary for geometry DSL
# ============================================================

# Relation keywords
RELATION_KEYWORDS = [
    "coll", "para", "perp", "cong", "eqangle", "mid",
    "oncirc", "cyclic", "eqratio",
]

# Construction keywords
CONSTRUCTION_KEYWORDS = [
    "mid", "alt", "circumcenter", "orthocenter", "incenter",
    "pthrough", "tthrough", "bisect", "pbisect", "circumcirc",
    "intersectll", "intersectlc", "reflect", "extend", "tangent",
    "centroid",
]

# Point names: a-z + aux_0..aux_31
POINT_NAMES = [chr(ord("a") + i) for i in range(26)]
POINT_NAMES += [f"aux_{i}" for i in range(32)]

# Special tokens
PAD_TOKEN = "[PAD]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = ";"
GOAL_TOKEN = "?"

# Build vocabulary
SPECIAL_TOKENS = [PAD_TOKEN, CLS_TOKEN, SEP_TOKEN, GOAL_TOKEN]
ALL_KEYWORDS = sorted(set(RELATION_KEYWORDS + CONSTRUCTION_KEYWORDS))
VOCAB = SPECIAL_TOKENS + ALL_KEYWORDS + POINT_NAMES
TOKEN_TO_ID = {tok: i for i, tok in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)

# Token IDs for convenience
PAD_ID = TOKEN_TO_ID[PAD_TOKEN]
CLS_ID = TOKEN_TO_ID[CLS_TOKEN]
SEP_ID = TOKEN_TO_ID[SEP_TOKEN]
GOAL_ID = TOKEN_TO_ID[GOAL_TOKEN]

# Maximum sequence length
MAX_SEQ_LEN = 512


def tokenize(text: str) -> list[int]:
    """Tokenize a geometry text string into token IDs.

    Handles the format: "coll a b c ; para a b c d ; ? perp a h b c"
    """
    ids = [CLS_ID]
    for token in text.split():
        if token in TOKEN_TO_ID:
            ids.append(TOKEN_TO_ID[token])
        else:
            # Unknown token — skip (shouldn't happen with valid input)
            pass
    return ids[:MAX_SEQ_LEN]


def pad_sequence(ids: list[int], max_len: int = MAX_SEQ_LEN) -> list[int]:
    """Pad a token ID sequence to max_len."""
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [PAD_ID] * (max_len - len(ids))


def tokenize_and_pad(text: str, max_len: int = MAX_SEQ_LEN) -> torch.Tensor:
    """Tokenize and pad a text string, returning a (max_len,) LongTensor."""
    ids = tokenize(text)
    padded = pad_sequence(ids, max_len)
    return torch.tensor(padded, dtype=torch.long)


# ============================================================
# Construction types and indexing (kept for MCTS compatibility)
# ============================================================

CONSTRUCTION_TYPES = [
    "Midpoint",             # 0
    "Altitude",             # 1
    "Circumcenter",         # 2
    "Orthocenter",          # 3
    "Incenter",             # 4
    "ParallelThrough",      # 5
    "PerpendicularThrough", # 6
]
NUM_CONSTRUCTION_TYPES = len(CONSTRUCTION_TYPES)
SLOTS_PER_TYPE = 292
POLICY_SIZE = 2048

# Tensor dimensions from encoding.rs (for legacy CNN)
NUM_CHANNELS = 20
GRID_SIZE = 32


def construction_to_index(type_name: str, args: list[int]) -> int:
    """Map a construction (type, args) to a policy index in [0, 2048).

    For 2-arg types (Midpoint): sort both args (order doesn't matter).
    For 3-arg types: preserve first arg (the point), sort last two (the line).
    """
    type_id = CONSTRUCTION_TYPES.index(type_name)
    if len(args) == 2:
        canonical = tuple(sorted(args))
    elif len(args) >= 3:
        canonical = (args[0], *sorted(args[1:]))
    else:
        canonical = tuple(args)
    h = 0
    for a in canonical:
        h = h * 37 + a + 1
    return type_id * SLOTS_PER_TYPE + (h % SLOTS_PER_TYPE)


# ============================================================
# GeoTransformer: text-based transformer model
# ============================================================

class GeoTransformer(nn.Module):
    """Transformer-based dual-head network for geometry theorem proving.

    Input: tokenized text sequence of proof state + goal
    Output:
      - v_logit: (B,) raw value before tanh
      - k: (B,) confidence scalar for delta_D weighting
      - policy_logits: (B, N) scores for each candidate construction

    For MCTS integration, use score_constructions() to get per-candidate scores.
    For training with fixed policy size, use forward() with policy_size output.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        max_seq_len: int = MAX_SEQ_LEN,
        dropout: float = 0.1,
        policy_size: int = POLICY_SIZE,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embedding + positional encoding
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        self.emb_norm = nn.LayerNorm(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Value head: [CLS] embedding → (v_logit, k)
        self.value_fc1 = nn.Linear(d_model, 128)
        self.value_fc2 = nn.Linear(128, 2)  # (v_logit, k)

        # Policy head: state embedding → construction scores
        # Two options:
        # 1. Fixed-size policy (for training with index-based targets)
        self.policy_fc1 = nn.Linear(d_model, 256)
        self.policy_fc2 = nn.Linear(256, policy_size)

        # 2. Construction scoring (for MCTS with text-based constructions)
        # Score = dot(state_proj, construction_proj)
        self.state_proj = nn.Linear(d_model, d_model)
        self.construction_proj = nn.Linear(d_model, d_model)

        self._init_weights()

    def _init_weights(self):
        # Embedding init
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        # Zero out padding embedding
        with torch.no_grad():
            self.token_emb.weight[PAD_ID].zero_()

        for m in [self.value_fc1, self.value_fc2, self.policy_fc1, self.policy_fc2,
                  self.state_proj, self.construction_proj]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Initialize k bias to 0.5
        with torch.no_grad():
            self.value_fc2.bias[1] = 0.5

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode a batch of token sequences into embeddings.

        Args:
            token_ids: (B, L) long tensor of token IDs

        Returns:
            cls_emb: (B, d_model) the [CLS] token embedding
        """
        B, L = token_ids.shape
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, L)
        x = self.token_emb(token_ids) + self.pos_emb(positions)
        x = self.emb_norm(self.emb_dropout(x))

        # Padding mask: True where padded
        padding_mask = (token_ids == PAD_ID)

        x = self.encoder(x, src_key_padding_mask=padding_mask)

        # Return [CLS] embedding (first token)
        return x[:, 0, :]

    def forward(
        self,
        token_ids: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with fixed-size policy output.

        Args:
            token_ids: (B, L) token IDs
            valid_mask: (B, POLICY_SIZE) boolean mask of valid constructions

        Returns:
            v_logit: (B,) raw value
            k: (B,) delta_D confidence scalar
            policy_logits: (B, POLICY_SIZE) construction logits
        """
        cls_emb = self.encode(token_ids)

        # Value head
        v = F.relu(self.value_fc1(cls_emb))
        v_out = self.value_fc2(v)
        v_logit = v_out[:, 0]
        k = v_out[:, 1]

        # Policy head (fixed-size)
        p = F.relu(self.policy_fc1(cls_emb))
        policy_logits = self.policy_fc2(p)

        if valid_mask is not None:
            policy_logits = policy_logits.masked_fill(~valid_mask, float("-inf"))

        return v_logit, k, policy_logits

    def score_constructions(
        self,
        state_ids: torch.Tensor,
        construction_ids_list: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Score candidate constructions via dot-product attention.

        Args:
            state_ids: (B, L_state) token IDs for proof states
            construction_ids_list: list of B tensors, each (N_i, L_constr)

        Returns:
            v_logit: (B,)
            k: (B,)
            scores: list of B tensors, each (N_i,)
        """
        cls_emb = self.encode(state_ids)

        # Value head
        v = F.relu(self.value_fc1(cls_emb))
        v_out = self.value_fc2(v)
        v_logit = v_out[:, 0]
        k = v_out[:, 1]

        # State projection for scoring
        state_vec = self.state_proj(cls_emb)  # (B, d_model)

        scores = []
        for i in range(len(construction_ids_list)):
            constr_ids = construction_ids_list[i]  # (N_i, L_constr)
            if constr_ids.numel() == 0:
                scores.append(torch.zeros(0, device=state_ids.device))
                continue
            # Encode each construction
            c_emb = self.encode(constr_ids)  # (N_i, d_model)
            c_proj = self.construction_proj(c_emb)  # (N_i, d_model)
            # Dot product scores
            s = (state_vec[i].unsqueeze(0) * c_proj).sum(dim=-1)  # (N_i,)
            scores.append(s)

        return v_logit, k, scores

    def predict(
        self,
        token_ids: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> tuple[float, float, torch.Tensor]:
        """Single-state prediction for inference."""
        self.eval()
        with torch.no_grad():
            v_logit, k, logits = self.forward(
                token_ids.unsqueeze(0),
                valid_mask.unsqueeze(0) if valid_mask is not None else None,
            )
            policy = F.softmax(logits[0], dim=0)
            return v_logit.item(), k.item(), policy

    def compute_value(self, v_logit: float, k: float, delta_d: float) -> float:
        """Compute final value: V = tanh(v_logit + k * delta_D)."""
        return math.tanh(v_logit + k * delta_d)


# ============================================================
# Legacy CNN model (kept for ablation)
# ============================================================

class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = x.mean(dim=[2, 3])
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.view(b, c, 1, 1)


class SEResBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcite(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + residual)


class GeoNetCNN(nn.Module):
    """Legacy CNN model. Kept for ablation comparison."""

    def __init__(
        self,
        in_channels: int = NUM_CHANNELS,
        backbone_channels: int = 128,
        num_blocks: int = 6,
        se_reduction: int = 8,
        policy_size: int = POLICY_SIZE,
    ):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, backbone_channels, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(backbone_channels)
        self.blocks = nn.Sequential(
            *[SEResBlock(backbone_channels, se_reduction) for _ in range(num_blocks)]
        )
        self.value_conv = nn.Conv2d(backbone_channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(GRID_SIZE * GRID_SIZE, 128)
        self.value_fc2 = nn.Linear(128, 2)
        self.policy_conv = nn.Conv2d(backbone_channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc1 = nn.Linear(2 * GRID_SIZE * GRID_SIZE, 256)
        self.policy_fc2 = nn.Linear(256, policy_size)

    def forward(self, x, valid_mask=None):
        h = F.relu(self.input_bn(self.input_conv(x)))
        h = self.blocks(h)
        v = F.relu(self.value_bn(self.value_conv(h)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v_out = self.value_fc2(v)
        v_logit = v_out[:, 0]
        k = v_out[:, 1]
        p = F.relu(self.policy_bn(self.policy_conv(h)))
        p = p.view(p.size(0), -1)
        p = F.relu(self.policy_fc1(p))
        policy_logits = self.policy_fc2(p)
        if valid_mask is not None:
            policy_logits = policy_logits.masked_fill(~valid_mask, float("-inf"))
        return v_logit, k, policy_logits


# Alias for backward compatibility
GeoNet = GeoTransformer


# ============================================================
# Utility functions
# ============================================================

def tensor_from_flat(flat: list[float], device: str = "cpu") -> torch.Tensor:
    """Convert flat 20480-element list from encode_state() to (20, 32, 32) tensor."""
    return torch.tensor(flat, dtype=torch.float32, device=device).view(
        NUM_CHANNELS, GRID_SIZE, GRID_SIZE
    )


def build_valid_mask(constructions, device: str = "cpu") -> torch.Tensor:
    """Build a boolean mask of valid policy indices from a list of PyConstruction objects."""
    mask = torch.zeros(POLICY_SIZE, dtype=torch.bool, device=device)
    for c in constructions:
        idx = construction_to_index(c.construction_type(), c.args())
        mask[idx] = True
    return mask


def constructions_to_policy_target(
    constructions,
    visit_counts: list[int],
    device: str = "cpu",
) -> torch.Tensor:
    """Convert MCTS visit counts to a policy target distribution."""
    target = torch.zeros(POLICY_SIZE, dtype=torch.float32, device=device)
    total = sum(visit_counts)
    if total == 0:
        return target
    for c, v in zip(constructions, visit_counts):
        idx = construction_to_index(c.construction_type(), c.args())
        target[idx] += v / total
    return target


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
