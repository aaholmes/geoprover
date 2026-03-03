"""GeoTransformer: Text-based transformer for geometry theorem proving.

Architecture:
  - Input: text sequence (proof state as relation list + goal)
  - Backbone: 6-layer transformer encoder, d_model=256, 8 heads
  - Value head: MLP on [CLS] embedding → sigmoid(v_logit) ∈ [0, 1]
  - Policy head: dot(state_emb, construction_emb) for each candidate

Replaces the CNN (GeoNet) which suffered from permutation sensitivity.
The old GeoNet CNN is kept as GeoNetCNN for ablation.

~5-8M parameters.
"""

import random
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
# Data augmentation: label permutation + fact shuffling
# ============================================================

POINT_NAME_SET = set(POINT_NAMES)

# Reverse lookup: name -> sequential ID (a=0, b=1, ..., aux_0=26, ...)
_NAME_TO_ID = {name: i for i, name in enumerate(POINT_NAMES)}


def make_augmentation_perm(epoch: int, idx: int, dataset_len: int) -> dict[str, str]:
    """Create a deterministic label permutation for a given (epoch, idx).

    Returns a dict mapping original point name -> permuted point name.
    Uses seed = epoch * dataset_len + idx for reproducibility.
    """
    seed = epoch * dataset_len + idx
    rng = random.Random(seed)
    shuffled = list(POINT_NAMES)
    rng.shuffle(shuffled)
    return dict(zip(POINT_NAMES, shuffled))


def permute_text(text: str, perm: dict[str, str]) -> str:
    """Apply a label permutation to a geometry text string.

    Point-name tokens are remapped via perm; keywords and special tokens pass through.
    """
    tokens = text.split()
    out = []
    for tok in tokens:
        if tok in POINT_NAME_SET:
            out.append(perm.get(tok, tok))
        else:
            out.append(tok)
    return " ".join(out)


def shuffle_facts(text: str, rng: random.Random) -> str:
    """Shuffle the order of facts in a geometry text, keeping goal at end.

    Input: "fact1 ; fact2 ; ? goal"
    Output: "fact2 ; fact1 ; ? goal"  (random order, goal always last)
    """
    # Split off goal — handle both "fact ; ? goal" and "fact ? goal" formats
    if " ; ? " in text:
        facts_part, goal_part = text.split(" ; ? ", 1)
    elif " ? " in text:
        facts_part, goal_part = text.split(" ? ", 1)
    else:
        return text  # no goal marker, nothing to shuffle

    facts = [f for f in facts_part.split(" ; ") if f.strip()]
    rng.shuffle(facts)
    return " ; ".join(facts) + " ; ? " + goal_part


def augment_state_text(
    text: str, epoch: int, idx: int, dataset_len: int
) -> tuple[str, dict[str, str]]:
    """Apply label permutation + fact shuffling to state text.

    Returns (augmented_text, perm) so caller can reuse perm for policy remapping.
    """
    seed = epoch * dataset_len + idx
    rng = random.Random(seed)

    perm = make_augmentation_perm(epoch, idx, dataset_len)
    result = permute_text(text, perm)
    result = shuffle_facts(result, rng)
    return result, perm


def permute_point_ids(args: list[int], perm: dict[str, str]) -> list[int]:
    """Remap integer point IDs through a name permutation.

    ID -> name -> perm[name] -> ID
    """
    out = []
    for pid in args:
        if 0 <= pid < len(POINT_NAMES):
            orig_name = POINT_NAMES[pid]
            new_name = perm.get(orig_name, orig_name)
            out.append(_NAME_TO_ID.get(new_name, pid))
        else:
            out.append(pid)
    return out


# ============================================================
# GeoTransformer: text-based transformer model
# ============================================================

class GeoTransformer(nn.Module):
    """Transformer-based dual-head network for geometry theorem proving.

    Input: tokenized text sequence of proof state + goal
    Output:
      - value: (B,) sigmoid(v_logit) ∈ [0, 1]
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

        # Value head: [CLS] embedding → sigmoid(v_logit)
        self.value_fc1 = nn.Linear(d_model, 128)
        self.value_fc2 = nn.Linear(128, 1)

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with fixed-size policy output.

        Args:
            token_ids: (B, L) token IDs
            valid_mask: (B, POLICY_SIZE) boolean mask of valid constructions

        Returns:
            value: (B,) sigmoid(v_logit) ∈ [0, 1]
            policy_logits: (B, POLICY_SIZE) construction logits
        """
        cls_emb = self.encode(token_ids)

        # Value head
        v = F.relu(self.value_fc1(cls_emb))
        value = torch.sigmoid(self.value_fc2(v).squeeze(-1))

        # Policy head (fixed-size)
        p = F.relu(self.policy_fc1(cls_emb))
        policy_logits = self.policy_fc2(p)

        if valid_mask is not None:
            policy_logits = policy_logits.masked_fill(~valid_mask, float("-inf"))

        return value, policy_logits

    def score_constructions(
        self,
        state_ids: torch.Tensor,
        construction_ids_list: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Score candidate constructions via dot-product attention.

        Args:
            state_ids: (B, L_state) token IDs for proof states
            construction_ids_list: list of B tensors, each (N_i, L_constr)

        Returns:
            value: (B,) sigmoid(v_logit) ∈ [0, 1]
            scores: list of B tensors, each (N_i,)
        """
        cls_emb = self.encode(state_ids)

        # Value head
        v = F.relu(self.value_fc1(cls_emb))
        value = torch.sigmoid(self.value_fc2(v).squeeze(-1))

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

        return value, scores

    def predict(
        self,
        token_ids: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> tuple[float, torch.Tensor]:
        """Single-state prediction for inference."""
        self.eval()
        with torch.no_grad():
            value, logits = self.forward(
                token_ids.unsqueeze(0),
                valid_mask.unsqueeze(0) if valid_mask is not None else None,
            )
            policy = F.softmax(logits[0], dim=0)
            return value.item(), policy


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
# SetGeoTransformer: set-equivariant architecture
# ============================================================

# Maximum tokens per fact/goal statement (e.g., "eqangle a b c d e f" = 7 tokens + CLS)
SET_MAX_TOKENS = 16
# Maximum number of facts per state
SET_MAX_FACTS = 256


def split_state_text(state_text: str) -> tuple[list[str], str]:
    """Parse "fact ; fact ; ? goal" format into (list[str], str).

    Returns (fact_texts, goal_text). If no goal marker, goal_text is empty.
    """
    if " ; ? " in state_text:
        facts_part, goal_part = state_text.split(" ; ? ", 1)
    elif " ? " in state_text:
        facts_part, goal_part = state_text.split(" ? ", 1)
    else:
        return [f.strip() for f in state_text.split(" ; ") if f.strip()], ""

    facts = [f.strip() for f in facts_part.split(" ; ") if f.strip()]
    return facts, goal_part.strip()


def tokenize_statement(text: str, max_tokens: int = SET_MAX_TOKENS) -> list[int]:
    """Tokenize a single fact/goal statement (no CLS prefix, just tokens + padding)."""
    ids = [CLS_ID]
    for token in text.split():
        if token in TOKEN_TO_ID:
            ids.append(TOKEN_TO_ID[token])
    ids = ids[:max_tokens]
    while len(ids) < max_tokens:
        ids.append(PAD_ID)
    return ids


def encode_state_as_set(
    fact_texts: list[str],
    goal_text: str,
    max_facts: int = SET_MAX_FACTS,
    max_tokens: int = SET_MAX_TOKENS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode a proof state as a set of per-fact token sequences.

    Args:
        fact_texts: list of fact strings (e.g., ["coll a b c", "para a b c d"])
        goal_text: goal string (e.g., "perp a h b c")
        max_facts: maximum number of facts (truncate if more)
        max_tokens: maximum tokens per fact/goal

    Returns:
        fact_ids: (N, max_tokens) LongTensor of per-fact token IDs
        goal_ids: (max_tokens,) LongTensor of goal token IDs
        fact_mask: (N,) BoolTensor, True for real facts, False for padding
    """
    facts = fact_texts[:max_facts]
    n = len(facts)

    # Tokenize each fact
    fact_id_list = [tokenize_statement(f, max_tokens) for f in facts]
    if not fact_id_list:
        # At least one dummy fact to avoid empty tensors
        fact_id_list = [[PAD_ID] * max_tokens]
        n = 1
        fact_mask = torch.zeros(1, dtype=torch.bool)
    else:
        fact_mask = torch.ones(n, dtype=torch.bool)

    fact_ids = torch.tensor(fact_id_list, dtype=torch.long)
    goal_ids = torch.tensor(tokenize_statement(goal_text, max_tokens), dtype=torch.long)

    return fact_ids, goal_ids, fact_mask


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention + FFN block.

    query attends to key/value from a different source.
    """

    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, Nq, D)
            kv: (B, Nkv, D)
            key_padding_mask: (B, Nkv) bool, True = ignore
        """
        q_normed = self.norm_q(query)
        kv_normed = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(
            q_normed, kv_normed, kv_normed,
            key_padding_mask=key_padding_mask,
        )
        x = query + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x


class SetGeoTransformerV1(nn.Module):
    """Set-equivariant transformer for geometry theorem proving (V1).

    Treats facts as a permutable set with a distinguished goal element.
    Permutation-invariant by construction (no cross-fact positional embeddings).

    Four stages:
      1. Per-Statement Encoding: shared 2-layer transformer with intra-fact positions
      2. Fact-to-Fact Self-Attention: 2-layer transformer, NO positional embeddings
      3. Goal-Conditioned Aggregation: 2-layer cross-attention (goal queries facts)
      4. Task Heads: value + policy (same as GeoTransformer)
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 512,
        max_tokens: int = SET_MAX_TOKENS,
        dropout: float = 0.1,
        policy_size: int = POLICY_SIZE,
        num_intra_layers: int = 2,
        num_fact_layers: int = 2,
        num_cross_layers: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_tokens = max_tokens

        # Stage 1: Per-statement encoding (shared)
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.intra_pos_emb = nn.Embedding(max_tokens, d_model)
        self.intra_norm = nn.LayerNorm(d_model)
        self.intra_dropout = nn.Dropout(dropout)

        intra_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.intra_encoder = nn.TransformerEncoder(intra_layer, num_layers=num_intra_layers)

        # Stage 2: Fact-to-fact self-attention (NO positional embeddings)
        fact_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.fact_encoder = nn.TransformerEncoder(fact_layer, num_layers=num_fact_layers)

        # Stage 3: Goal-conditioned cross-attention
        self.cross_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_cross_layers)
        ])
        self.state_norm = nn.LayerNorm(d_model)

        # Stage 4: Task heads
        # Value head
        self.value_fc1 = nn.Linear(d_model, 128)
        self.value_fc2 = nn.Linear(128, 1)

        # Fixed-size policy head
        self.policy_fc1 = nn.Linear(d_model, 256)
        self.policy_fc2 = nn.Linear(256, policy_size)

        # Dot-product policy head
        self.state_proj = nn.Linear(d_model, d_model)
        self.construction_proj = nn.Linear(d_model, d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.intra_pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.token_emb.weight[PAD_ID].zero_()

        for m in [self.value_fc1, self.value_fc2, self.policy_fc1, self.policy_fc2,
                  self.state_proj, self.construction_proj]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _encode_statements(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode a batch of statements into [CLS] embeddings.

        Args:
            token_ids: (M, L) where M is total statements, L is max_tokens

        Returns:
            cls_embs: (M, d_model)
        """
        M, L = token_ids.shape
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(M, L)
        x = self.token_emb(token_ids) + self.intra_pos_emb(positions)
        x = self.intra_norm(self.intra_dropout(x))
        padding_mask = (token_ids == PAD_ID)
        x = self.intra_encoder(x, src_key_padding_mask=padding_mask)
        return x[:, 0, :]  # [CLS] at position 0

    def encode_state(
        self,
        fact_ids: torch.Tensor,
        goal_ids: torch.Tensor,
        fact_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of states into state representations.

        Args:
            fact_ids: (B, N, L) per-fact token IDs
            goal_ids: (B, L) goal token IDs
            fact_mask: (B, N) bool, True for real facts

        Returns:
            state_repr: (B, d_model)
        """
        B, N, L = fact_ids.shape

        # Stage 1: encode all facts + goal through shared intra-encoder
        # Flatten facts for batch encoding — only encode real facts to avoid NaN
        flat_mask = fact_mask.reshape(B * N)  # (B*N,)
        flat_facts = fact_ids.reshape(B * N, L)  # (B*N, L)

        # Initialize embeddings to zeros (padding facts stay zero)
        flat_cls = torch.zeros(B * N, self.d_model, device=fact_ids.device)
        if flat_mask.any():
            real_facts = flat_facts[flat_mask]  # (M, L) where M = num real facts
            real_cls = self._encode_statements(real_facts)  # (M, D)
            flat_cls[flat_mask] = real_cls

        fact_embs = flat_cls.reshape(B, N, self.d_model)  # (B, N, D)

        goal_cls = self._encode_statements(goal_ids)  # (B, D)
        goal_emb = goal_cls.unsqueeze(1)  # (B, 1, D)

        # Stage 2: fact-to-fact self-attention (permutation equivariant)
        # Mask: True = padding (to ignore)
        fact_pad_mask = ~fact_mask  # (B, N)
        fact_embs = self.fact_encoder(
            fact_embs, src_key_padding_mask=fact_pad_mask,
        )

        # Stage 3: goal-conditioned cross-attention
        for cross_layer in self.cross_layers:
            goal_emb = cross_layer(
                query=goal_emb, kv=fact_embs,
                key_padding_mask=fact_pad_mask,
            )

        goal_repr = goal_emb.squeeze(1)  # (B, D)

        # Mean pool over valid facts as residual
        fact_mask_expanded = fact_mask.unsqueeze(-1).float()  # (B, N, 1)
        fact_sum = (fact_embs * fact_mask_expanded).sum(dim=1)  # (B, D)
        fact_count = fact_mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
        mean_facts = fact_sum / fact_count

        state_repr = self.state_norm(goal_repr + mean_facts)
        return state_repr

    def forward(
        self,
        fact_ids: torch.Tensor,
        goal_ids: torch.Tensor,
        fact_mask: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with fixed-size policy output.

        Args:
            fact_ids: (B, N, L) per-fact token IDs
            goal_ids: (B, L) goal token IDs
            fact_mask: (B, N) bool, True for real facts
            valid_mask: (B, POLICY_SIZE) bool mask of valid constructions

        Returns:
            value: (B,) sigmoid(v_logit) in [0, 1]
            policy_logits: (B, POLICY_SIZE)
        """
        state_repr = self.encode_state(fact_ids, goal_ids, fact_mask)

        # Value head
        v = F.relu(self.value_fc1(state_repr))
        value = torch.sigmoid(self.value_fc2(v).squeeze(-1))

        # Policy head
        p = F.relu(self.policy_fc1(state_repr))
        policy_logits = self.policy_fc2(p)

        if valid_mask is not None:
            policy_logits = policy_logits.masked_fill(~valid_mask, float("-inf"))

        return value, policy_logits

    def score_constructions(
        self,
        fact_ids: torch.Tensor,
        goal_ids: torch.Tensor,
        fact_mask: torch.Tensor,
        construction_ids_list: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Score candidate constructions via dot-product attention.

        Args:
            fact_ids: (B, N, L) per-fact token IDs
            goal_ids: (B, L) goal token IDs
            fact_mask: (B, N) bool
            construction_ids_list: list of B tensors, each (N_i, L_constr)

        Returns:
            value: (B,)
            scores: list of B tensors, each (N_i,)
        """
        state_repr = self.encode_state(fact_ids, goal_ids, fact_mask)

        v = F.relu(self.value_fc1(state_repr))
        value = torch.sigmoid(self.value_fc2(v).squeeze(-1))

        state_vec = self.state_proj(state_repr)  # (B, D)

        scores = []
        for i in range(len(construction_ids_list)):
            constr_ids = construction_ids_list[i]
            if constr_ids.numel() == 0:
                scores.append(torch.zeros(0, device=fact_ids.device))
                continue
            c_emb = self._encode_statements(constr_ids)
            c_proj = self.construction_proj(c_emb)
            s = (state_vec[i].unsqueeze(0) * c_proj).sum(dim=-1)
            scores.append(s)

        return value, scores

    def predict(
        self,
        fact_ids: torch.Tensor,
        goal_ids: torch.Tensor,
        fact_mask: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[float, torch.Tensor]:
        """Single-state prediction for inference."""
        self.eval()
        with torch.no_grad():
            value, logits = self.forward(
                fact_ids.unsqueeze(0),
                goal_ids.unsqueeze(0),
                fact_mask.unsqueeze(0),
                valid_mask.unsqueeze(0) if valid_mask is not None else None,
            )
            policy = F.softmax(logits[0], dim=0)
            return value.item(), policy


# Backward compatibility alias
SetGeoTransformer = SetGeoTransformerV1


# ============================================================
# SetGeoTransformerV2: 3-way attention with deferred saturation
# ============================================================

class SetGeoTransformerV2(nn.Module):
    """V2 set-equivariant transformer with 3-way attention (goal × construction × facts).

    Eliminates O(N²) fact-to-fact self-attention. Facts are encoded independently
    and cached. A 3-way attention mechanism scores constructions in O(N) each.

    Four stages:
      1. Per-Statement Encoding: shared 2-layer transformer (CACHEABLE)
         Each fact/goal/construction → intra_encoder → embedding
      2. Goal-Construction Fusion: 2-layer cross-attention
         goal_emb attends to construction tokens → joint_query
      3. Joint-Query-to-Facts Cross-Attention: 2-layer cross-attention
         joint_query attends to fact embeddings → state_repr (O(N) per query)
      4. Task Heads: policy (scalar per construction) + value

    ~2.5M params.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 512,
        max_tokens: int = SET_MAX_TOKENS,
        dropout: float = 0.1,
        num_intra_layers: int = 2,
        num_fusion_layers: int = 2,
        num_query_layers: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_tokens = max_tokens

        # Stage 1: Per-statement encoding (shared)
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.intra_pos_emb = nn.Embedding(max_tokens, d_model)
        self.intra_norm = nn.LayerNorm(d_model)
        self.intra_dropout = nn.Dropout(dropout)

        intra_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.intra_encoder = nn.TransformerEncoder(intra_layer, num_layers=num_intra_layers)

        # Stage 2: Goal-Construction Fusion (cross-attention)
        self.fusion_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_fusion_layers)
        ])

        # Stage 3a: Policy Query-to-Facts Cross-Attention
        # Used by policy: fused goal+construction queries facts
        self.policy_query_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_query_layers)
        ])

        # Stage 3b: Value Query-to-Facts Cross-Attention (separate weights)
        # Used by value: raw goal queries facts — different question, different weights
        self.value_query_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_query_layers)
        ])

        # Stage 4: Task heads
        # Policy: state_repr → scalar logit (one per construction)
        self.policy_fc1 = nn.Linear(d_model, 128)
        self.policy_fc2 = nn.Linear(128, 1)
        # Value head
        self.value_fc1 = nn.Linear(d_model, 128)
        self.value_fc2 = nn.Linear(128, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.intra_pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.token_emb.weight[PAD_ID].zero_()

        for m in [self.policy_fc1, self.policy_fc2, self.value_fc1, self.value_fc2]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _encode_statements_cls(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode statements, returning [CLS] embeddings.

        Args:
            token_ids: (M, L)
        Returns:
            cls_embs: (M, d_model)
        """
        M, L = token_ids.shape
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(M, L)
        x = self.token_emb(token_ids) + self.intra_pos_emb(positions)
        x = self.intra_norm(self.intra_dropout(x))
        padding_mask = (token_ids == PAD_ID)
        x = self.intra_encoder(x, src_key_padding_mask=padding_mask)
        return x[:, 0, :]  # [CLS] at position 0

    def _encode_statements_full(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode statements, returning full token sequence.

        Args:
            token_ids: (M, L)
        Returns:
            full_embs: (M, L, d_model)
        """
        M, L = token_ids.shape
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(M, L)
        x = self.token_emb(token_ids) + self.intra_pos_emb(positions)
        x = self.intra_norm(self.intra_dropout(x))
        padding_mask = (token_ids == PAD_ID)
        x = self.intra_encoder(x, src_key_padding_mask=padding_mask)
        return x

    def encode_facts(self, fact_ids: torch.Tensor, fact_mask: torch.Tensor) -> torch.Tensor:
        """Encode facts into [CLS] embeddings (cacheable).

        Args:
            fact_ids: (B, N, L) or (M, L) flat
            fact_mask: (B, N) or (M,) flat bool

        Returns:
            fact_embs: same leading dims as input, d_model last dim
        """
        if fact_ids.dim() == 3:
            B, N, L = fact_ids.shape
            flat_mask = fact_mask.reshape(B * N)
            flat_facts = fact_ids.reshape(B * N, L)
            flat_cls = torch.zeros(B * N, self.d_model, device=fact_ids.device)
            if flat_mask.any():
                real_facts = flat_facts[flat_mask]
                real_cls = self._encode_statements_cls(real_facts)
                flat_cls[flat_mask] = real_cls
            return flat_cls.reshape(B, N, self.d_model)
        else:
            # (M, L) flat input
            return self._encode_statements_cls(fact_ids)

    def encode_goal(self, goal_ids: torch.Tensor) -> torch.Tensor:
        """Encode goal into [CLS] embedding.

        Args:
            goal_ids: (B, L)
        Returns:
            goal_emb: (B, d_model)
        """
        return self._encode_statements_cls(goal_ids)

    def encode_construction_seq(self, constr_ids: torch.Tensor) -> torch.Tensor:
        """Encode constructions, returning full token sequences for fusion.

        Args:
            constr_ids: (M, L)
        Returns:
            constr_seq: (M, L, d_model)
        """
        return self._encode_statements_full(constr_ids)

    def fuse_goal_construction(
        self,
        goal_emb: torch.Tensor,
        constr_seq: torch.Tensor,
        constr_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Stage 2: Goal attends to construction tokens.

        Args:
            goal_emb: (B, 1, D) or (B, D)
            constr_seq: (B, L_c, D) full construction token sequence
            constr_pad_mask: (B, L_c) bool, True = padding to ignore

        Returns:
            joint_query: (B, 1, D)
        """
        if goal_emb.dim() == 2:
            goal_emb = goal_emb.unsqueeze(1)  # (B, 1, D)

        for layer in self.fusion_layers:
            goal_emb = layer(
                query=goal_emb, kv=constr_seq,
                key_padding_mask=constr_pad_mask,
            )
        return goal_emb  # (B, 1, D)

    def query_facts(
        self,
        joint_q: torch.Tensor,
        fact_kv: torch.Tensor,
        fact_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Stage 3: Joint query attends to fact embeddings.

        Args:
            joint_q: (B, 1, D)
            fact_kv: (B, N, D) fact [CLS] embeddings
            fact_mask: (B, N) bool, True = real facts

        Returns:
            state_repr: (B, D)
        """
        # fact_mask: True = real, need to invert for key_padding_mask (True = ignore)
        pad_mask = ~fact_mask if fact_mask is not None else None

        for layer in self.policy_query_layers:
            joint_q = layer(
                query=joint_q, kv=fact_kv,
                key_padding_mask=pad_mask,
            )
        return joint_q.squeeze(1)  # (B, D)

    def forward(
        self,
        fact_ids: torch.Tensor,
        goal_ids: torch.Tensor,
        fact_mask: torch.Tensor,
        constr_ids: torch.Tensor,
        constr_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full training forward pass.

        Args:
            fact_ids: (B, N, L) per-fact token IDs
            goal_ids: (B, L) goal token IDs
            fact_mask: (B, N) bool, True = real facts
            constr_ids: (B, K, L) per-construction token IDs
            constr_mask: (B, K) bool, True = real constructions

        Returns:
            value: (B,) sigmoid(v_logit) in [0, 1]
            logits: (B, K) per-construction logits
        """
        B, N, L = fact_ids.shape
        K = constr_ids.shape[1]

        # Stage 1: encode facts, goal, constructions
        fact_embs = self.encode_facts(fact_ids, fact_mask)  # (B, N, D)
        goal_emb = self.encode_goal(goal_ids)  # (B, D)

        # Encode all constructions (flatten for batch efficiency)
        flat_constr = constr_ids.reshape(B * K, L)  # (B*K, L)
        flat_constr_seq = self._encode_statements_full(flat_constr)  # (B*K, L, D)
        constr_seqs = flat_constr_seq.reshape(B, K, L, self.d_model)  # (B, K, L, D)

        # Score each construction
        logits = torch.zeros(B, K, device=fact_ids.device)

        # Process constructions: for each, fuse with goal and query facts
        # Efficient batched version: expand goal for all K constructions
        goal_expanded = goal_emb.unsqueeze(1).expand(B, K, self.d_model)  # (B, K, D)
        goal_flat = goal_expanded.reshape(B * K, 1, self.d_model)  # (B*K, 1, D)

        constr_seqs_flat = constr_seqs.reshape(B * K, L, self.d_model)  # (B*K, L, D)

        # Construct padding mask for construction tokens
        flat_constr_pad = (flat_constr == PAD_ID)  # (B*K, L)

        # Stage 2: fuse goal with each construction
        joint_q = goal_flat
        for layer in self.fusion_layers:
            joint_q = layer(query=joint_q, kv=constr_seqs_flat, key_padding_mask=flat_constr_pad)
        # joint_q: (B*K, 1, D)

        # Stage 3: query facts with joint query
        fact_embs_expanded = fact_embs.unsqueeze(1).expand(B, K, N, self.d_model)
        fact_embs_flat = fact_embs_expanded.reshape(B * K, N, self.d_model)

        fact_mask_expanded = fact_mask.unsqueeze(1).expand(B, K, N)
        fact_pad_flat = ~fact_mask_expanded.reshape(B * K, N)

        state_repr = joint_q
        for layer in self.policy_query_layers:
            state_repr = layer(query=state_repr, kv=fact_embs_flat, key_padding_mask=fact_pad_flat)
        state_repr = state_repr.squeeze(1)  # (B*K, D)

        # Policy logits
        p = F.relu(self.policy_fc1(state_repr))
        logits = self.policy_fc2(p).squeeze(-1)  # (B*K,)
        logits = logits.reshape(B, K)  # (B, K)

        # Mask invalid constructions
        logits = logits.masked_fill(~constr_mask, float("-inf"))

        # Value: use goal querying facts directly (no construction)
        goal_q = goal_emb.unsqueeze(1)  # (B, 1, D)
        fact_pad = ~fact_mask
        for layer in self.value_query_layers:
            goal_q = layer(query=goal_q, kv=fact_embs, key_padding_mask=fact_pad)
        value_repr = goal_q.squeeze(1)  # (B, D)

        v = F.relu(self.value_fc1(value_repr))
        value = torch.sigmoid(self.value_fc2(v).squeeze(-1))

        return value, logits

    def score_constructions_cached(
        self,
        fact_kv: torch.Tensor,
        fact_mask: torch.Tensor,
        goal_emb: torch.Tensor,
        constr_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Score constructions using pre-cached fact embeddings.

        Args:
            fact_kv: (N, D) cached fact [CLS] embeddings
            fact_mask: (N,) bool
            goal_emb: (D,) cached goal embedding
            constr_ids: (K, L) construction token IDs

        Returns:
            logits: (K,) per-construction logits
        """
        K, L = constr_ids.shape
        # Encode construction sequences
        constr_seqs = self._encode_statements_full(constr_ids)  # (K, L, D)
        constr_pad = (constr_ids == PAD_ID)  # (K, L)

        # Expand goal for K constructions
        goal_q = goal_emb.unsqueeze(0).unsqueeze(0).expand(K, 1, self.d_model)  # (K, 1, D)

        # Stage 2: fuse
        for layer in self.fusion_layers:
            goal_q = layer(query=goal_q, kv=constr_seqs, key_padding_mask=constr_pad)

        # Stage 3: query facts
        fact_kv_exp = fact_kv.unsqueeze(0).expand(K, -1, self.d_model)  # (K, N, D)
        fact_pad = (~fact_mask).unsqueeze(0).expand(K, -1)  # (K, N)

        state_repr = goal_q
        for layer in self.policy_query_layers:
            state_repr = layer(query=state_repr, kv=fact_kv_exp, key_padding_mask=fact_pad)
        state_repr = state_repr.squeeze(1)  # (K, D)

        p = F.relu(self.policy_fc1(state_repr))
        logits = self.policy_fc2(p).squeeze(-1)  # (K,)
        return logits

    def evaluate_cached(
        self,
        fact_kv: torch.Tensor,
        fact_mask: torch.Tensor,
        goal_emb: torch.Tensor,
    ) -> float:
        """Get value from cached fact embeddings (post-saturation).

        Args:
            fact_kv: (N, D) fact embeddings
            fact_mask: (N,) bool
            goal_emb: (D,) goal embedding

        Returns:
            value: float in [0, 1]
        """
        # Goal queries facts directly (no construction)
        goal_q = goal_emb.unsqueeze(0).unsqueeze(0)  # (1, 1, D)
        fkv = fact_kv.unsqueeze(0)  # (1, N, D)
        fpad = (~fact_mask).unsqueeze(0)  # (1, N)

        for layer in self.value_query_layers:
            goal_q = layer(query=goal_q, kv=fkv, key_padding_mask=fpad)

        value_repr = goal_q.squeeze(1).squeeze(0)  # (D,)
        v = F.relu(self.value_fc1(value_repr))
        return torch.sigmoid(self.value_fc2(v)).item()

    def predict_value(
        self,
        fact_kv: torch.Tensor,
        fact_mask: torch.Tensor,
        goal_emb: torch.Tensor,
    ) -> float:
        """Root value: goal queries facts, no construction. Alias for evaluate_cached."""
        return self.evaluate_cached(fact_kv, fact_mask, goal_emb)


def create_model(model_type: str = "set", **kwargs) -> nn.Module:
    """Factory function to create a model by type.

    Args:
        model_type: "set" for SetGeoTransformer, "transformer" for GeoTransformer

    Returns:
        Model instance
    """
    if model_type == "set":
        return SetGeoTransformerV1(**kwargs)
    elif model_type == "set_v2":
        return SetGeoTransformerV2(**kwargs)
    elif model_type == "transformer":
        return GeoTransformer(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


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
