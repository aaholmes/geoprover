"""FactSummarizer: NNUE-style learned filter for post-saturation deduced facts.

Architecture (two-stage NNUE decomposition):
  - Context encoder: 2-layer transformer encodes initial facts + constructions + goal (expensive, once)
  - Fact encoder: 1-layer transformer encodes each candidate deduced fact (cheap, N times)
  - Score: dot(context_embedding, fact_embedding) → relevance score per fact

Training:
  - Labels from proof traces: 1 = fact on proof path, 0 = not
  - BCE loss on per-fact scores
  - Data: deduction-solved JGEX problems provide clean proof-path labels

Usage at MCTS time:
  1. Parse problem → initial facts (pre-saturation snapshot)
  2. Saturate → all facts (post-saturation)
  3. Deduced facts = post - pre
  4. Score each deduced fact with Summarizer
  5. Keep top-K deduced facts (K = |initial| + |constructions| + 1)
  6. Build compact text = initial facts + top-K deduced + goal → feed to GeoTransformer

~1.5M parameters (context encoder dominates).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import (
    CLS_ID,
    PAD_ID,
    SEP_ID,
    VOCAB_SIZE,
    tokenize,
    pad_sequence,
)

# Summarizer defaults
SUMMARIZER_D_MODEL = 128
SUMMARIZER_NHEAD = 4
SUMMARIZER_CONTEXT_LAYERS = 2
SUMMARIZER_FACT_LAYERS = 1
SUMMARIZER_DIM_FF = 256
SUMMARIZER_MAX_SEQ_LEN = 256
SUMMARIZER_DROPOUT = 0.1


class FactSummarizer(nn.Module):
    """NNUE-style fact relevance scorer.

    Encodes context (initial facts + goal) once with a deep encoder,
    then scores each candidate deduced fact cheaply via dot product.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = SUMMARIZER_D_MODEL,
        nhead: int = SUMMARIZER_NHEAD,
        context_layers: int = SUMMARIZER_CONTEXT_LAYERS,
        fact_layers: int = SUMMARIZER_FACT_LAYERS,
        dim_feedforward: int = SUMMARIZER_DIM_FF,
        max_seq_len: int = SUMMARIZER_MAX_SEQ_LEN,
        dropout: float = SUMMARIZER_DROPOUT,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Shared token + position embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        self.emb_norm = nn.LayerNorm(d_model)

        # Context encoder: deeper (encodes initial facts + goal once)
        context_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            context_layer, num_layers=context_layers
        )

        # Fact encoder: shallow (encodes each candidate fact cheaply)
        fact_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.fact_encoder = nn.TransformerEncoder(
            fact_layer, num_layers=fact_layers
        )

        # Projection heads for dot-product scoring
        self.context_proj = nn.Linear(d_model, d_model)
        self.fact_proj = nn.Linear(d_model, d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.token_emb.weight[PAD_ID].zero_()
        for m in [self.context_proj, self.fact_proj]:
            nn.init.xavier_uniform_(m.weight, gain=0.01)
            nn.init.constant_(m.bias, 0)

    def _embed(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared embedding: tokens + positions → embeddings + padding mask."""
        B, L = token_ids.shape
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, L)
        x = self.token_emb(token_ids) + self.pos_emb(positions)
        x = self.emb_norm(self.emb_dropout(x))
        padding_mask = token_ids == PAD_ID
        return x, padding_mask

    def encode_context(self, context_ids: torch.Tensor) -> torch.Tensor:
        """Encode context (initial facts + goal) → (B, d_model) embedding.

        Args:
            context_ids: (B, L_ctx) token IDs for "[CLS] fact1 ; fact2 ; ? goal"
        Returns:
            (B, d_model) context embedding from [CLS] position
        """
        x, mask = self._embed(context_ids)
        x = self.context_encoder(x, src_key_padding_mask=mask)
        return self.context_proj(x[:, 0, :])  # [CLS] embedding

    def encode_facts(self, fact_ids: torch.Tensor) -> torch.Tensor:
        """Encode candidate facts → (B, d_model) embeddings.

        Args:
            fact_ids: (N, L_fact) token IDs for individual facts (each prepended with [CLS])
        Returns:
            (N, d_model) fact embeddings from [CLS] position
        """
        x, mask = self._embed(fact_ids)
        x = self.fact_encoder(x, src_key_padding_mask=mask)
        return self.fact_proj(x[:, 0, :])  # [CLS] embedding

    def score_facts(
        self,
        context_ids: torch.Tensor,
        fact_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Score each candidate fact against the context.

        Args:
            context_ids: (1, L_ctx) or (B, L_ctx) context token IDs
            fact_ids: (N, L_fact) candidate fact token IDs

        Returns:
            (N,) relevance scores (logits, not sigmoided)
        """
        ctx_emb = self.encode_context(context_ids)  # (B, d_model)
        fact_emb = self.encode_facts(fact_ids)  # (N, d_model)

        # If batch context, expand to match facts
        if ctx_emb.shape[0] == 1 and fact_emb.shape[0] > 1:
            ctx_emb = ctx_emb.expand(fact_emb.shape[0], -1)

        # Dot product scores
        scores = (ctx_emb * fact_emb).sum(dim=-1)  # (N,)
        return scores

    def forward(
        self,
        context_ids: torch.Tensor,
        fact_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: score facts against context.

        Args:
            context_ids: (B, L_ctx) context token IDs
            fact_ids: (B, N, L_fact) padded fact token IDs per example
                      OR (N, L_fact) for single-context scoring

        Returns:
            scores: same shape as fact_ids batch dims
        """
        return self.score_facts(context_ids, fact_ids)


# ============================================================
# Tokenization helpers for Summarizer
# ============================================================

def build_context_tokens(
    initial_facts: list[str],
    goal_text: str | None,
    max_len: int = SUMMARIZER_MAX_SEQ_LEN,
) -> torch.Tensor:
    """Build context token sequence: "[CLS] fact1 ; fact2 ; ... ; ? goal"

    Args:
        initial_facts: list of fact text strings (pre-saturation)
        goal_text: goal text string, or None
        max_len: maximum sequence length

    Returns:
        (max_len,) LongTensor of token IDs
    """
    text_parts = " ; ".join(initial_facts)
    if goal_text:
        text = f"{text_parts} ; ? {goal_text}" if text_parts else f"? {goal_text}"
    else:
        text = text_parts
    ids = tokenize(text)
    padded = pad_sequence(ids, max_len)
    return torch.tensor(padded, dtype=torch.long)


def build_fact_tokens(
    facts: list[str],
    max_len: int = 32,
) -> torch.Tensor:
    """Tokenize individual facts, each prepended with [CLS].

    Args:
        facts: list of fact text strings
        max_len: max tokens per fact (facts are short, 32 is plenty)

    Returns:
        (N, max_len) LongTensor of token IDs
    """
    all_ids = []
    for fact in facts:
        ids = tokenize(fact)  # tokenize already prepends [CLS]
        padded = pad_sequence(ids, max_len)
        all_ids.append(padded)
    if not all_ids:
        return torch.zeros(0, max_len, dtype=torch.long)
    return torch.tensor(all_ids, dtype=torch.long)


# ============================================================
# Training data generation
# ============================================================

@dataclass
class SummarizerSample:
    """A single training sample for the Summarizer.

    initial_facts: pre-saturation fact texts
    goal_text: goal as text
    deduced_facts: post-saturation minus pre-saturation fact texts
    labels: 1.0 if fact is on proof path, 0.0 otherwise (parallel to deduced_facts)
    """
    initial_facts: list[str]
    goal_text: str
    deduced_facts: list[str]
    labels: list[float]


def generate_summarizer_data(problems_file: str) -> list[SummarizerSample]:
    """Generate Summarizer training data from deduction-solved JGEX problems.

    For each problem that deduction solves:
    1. Parse → snapshot initial facts
    2. saturate_with_trace → get proof path facts
    3. Deduced = post - pre
    4. Label each deduced fact: 1 if on proof path, 0 otherwise
    """
    import geoprover
    from orchestrate import load_problems

    problems = load_problems(problems_file)
    samples = []
    solved = 0
    skipped = 0

    for name, definition in problems:
        problem_text = f"{name}\n{definition}"
        try:
            state = geoprover.parse_problem(problem_text)
            initial_facts = set(state.facts_as_text_list())
            goal_text = state.goal_as_text()

            proved, trace = geoprover.saturate_with_trace(state)
            if not proved:
                continue
            solved += 1

            post_facts = set(state.facts_as_text_list())
            deduced_facts = sorted(post_facts - initial_facts)

            if not deduced_facts:
                skipped += 1
                continue

            # Get proof path facts (non-axiom derived facts)
            proof_path = trace.proof_path_facts()
            if proof_path is None:
                skipped += 1
                continue
            proof_path_set = set(proof_path)

            labels = [1.0 if f in proof_path_set else 0.0 for f in deduced_facts]

            # Skip if all labels are 0 or all are 1 (no signal)
            if sum(labels) == 0 or sum(labels) == len(labels):
                skipped += 1
                continue

            samples.append(SummarizerSample(
                initial_facts=sorted(initial_facts),
                goal_text=goal_text or "",
                deduced_facts=deduced_facts,
                labels=labels,
            ))
        except Exception as e:
            skipped += 1

    print(f"Summarizer data: {len(samples)} samples from {solved} solved problems "
          f"({skipped} skipped)")
    return samples


# ============================================================
# Fact filtering at inference time
# ============================================================

@torch.no_grad()
def filter_facts(
    summarizer: "FactSummarizer",
    initial_facts: list[str],
    deduced_facts: list[str],
    goal_text: str | None,
    k: int | None = None,
    device: str = "cpu",
    fact_max_len: int = 32,
    context_max_len: int = SUMMARIZER_MAX_SEQ_LEN,
) -> list[str]:
    """Filter deduced facts to top-K most relevant using the Summarizer.

    Args:
        summarizer: trained FactSummarizer model
        initial_facts: pre-saturation fact texts
        deduced_facts: post-saturation minus pre-saturation fact texts
        goal_text: goal as text
        k: number of facts to keep (default: len(initial_facts) + 1)
        device: torch device
        fact_max_len: max tokens per fact
        context_max_len: max tokens for context

    Returns:
        Top-K deduced fact texts sorted by relevance score (highest first)
    """
    if not deduced_facts:
        return []

    if k is None:
        k = len(initial_facts) + 1
    k = min(k, len(deduced_facts))

    summarizer.eval()

    # Encode context
    ctx_ids = build_context_tokens(initial_facts, goal_text, context_max_len).to(device)
    ctx_ids = ctx_ids.unsqueeze(0)  # (1, L_ctx)

    # Encode facts in batches to avoid OOM on huge fact sets
    batch_size = 256
    all_scores = []
    for i in range(0, len(deduced_facts), batch_size):
        batch_facts = deduced_facts[i : i + batch_size]
        fact_ids = build_fact_tokens(batch_facts, fact_max_len).to(device)
        scores = summarizer.score_facts(ctx_ids, fact_ids)
        all_scores.append(scores)

    scores = torch.cat(all_scores, dim=0)  # (N,)

    # Top-K
    topk_indices = torch.topk(scores, k).indices.cpu().tolist()
    return [deduced_facts[i] for i in sorted(topk_indices)]


def build_summarized_text(
    initial_facts: list[str],
    filtered_deduced: list[str],
    goal_text: str | None,
) -> str:
    """Build compact text from initial + filtered deduced facts + goal.

    Format: "fact1 ; fact2 ; ... ; ? goal"
    Same format as state_to_text() but with filtered deduced facts.
    """
    all_facts = initial_facts + filtered_deduced
    text = " ; ".join(all_facts)
    if goal_text:
        text += " ; ? " + goal_text
    return text


# ============================================================
# Utility
# ============================================================

def count_summarizer_parameters(model: FactSummarizer) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
