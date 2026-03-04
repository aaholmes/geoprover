"""Training loop for GeoTransformer: synthetic pre-training + expert iteration.

Phases:
  A. Synthetic pre-training: train on Rust-generated synthetic data (100K+ examples)
  B. JGEX supervised fine-tuning on deduction-solvable problems
  C. Expert iteration: MCTS self-play -> collect data -> train -> repeat

Loss: L = KL(policy || target) + c_value * MSE(value, target)
"""

import argparse
import json
import os
import random
import threading
import time
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import geoprover
from model import (
    POLICY_SIZE,
    GeoNet,
    GeoTransformer,
    SetGeoTransformer,
    SetGeoTransformerV2,
    build_valid_mask,
    construction_to_index,
    count_parameters,
    create_model,
    tokenize_and_pad,
    tokenize_statement,
    augment_state_text,
    permute_text,
    permute_point_ids,
    make_augmentation_perm,
    encode_state_as_set,
    split_state_text,
    MAX_SEQ_LEN,
    SET_MAX_TOKENS,
    SET_MAX_FACTS,
    PAD_ID,
)
from orchestrate import (
    GAMMA,
    MctsConfig,
    TrainingSample,
    load_problems,
    mcts_search,
    self_play_episode,
    solve_problem,
)
# Training defaults
DEFAULT_LR = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS_PER_ITER = 5
DEFAULT_REPLAY_SIZE = 100000
DEFAULT_VALUE_WEIGHT = 1.0
DEFAULT_NUM_ITERATIONS = 10
DEFAULT_PROBLEMS_FILE = "problems/jgex_ag_231.txt"
DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_SYNTHETIC_SIZE = 50000
DEFAULT_SYNTHETIC_SEED = 42
DEFAULT_MAX_SEQ_LEN = 128
DEFAULT_SELFPLAY_MCTS_ITERS = 200


def _resolve_device(requested: str) -> str:
    """Resolve device string: 'auto' detects best available device."""
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            return "cpu"
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_default_batch_size(device: str) -> int:
    """Return default batch size based on device."""
    if device == "cuda":
        return 256
    return DEFAULT_BATCH_SIZE  # 128


def _make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    collate_fn=None,
    *,
    num_workers: int = 0,
    device: str = "cpu",
) -> DataLoader:
    """Create a DataLoader with device-appropriate settings."""
    use_cuda = device == "cuda"
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


class ReplayBuffer:
    """Fixed-size replay buffer for training samples."""

    def __init__(self, max_size: int = DEFAULT_REPLAY_SIZE):
        self.buffer: deque[TrainingSample] = deque(maxlen=max_size)

    def add(self, samples: list[TrainingSample]) -> None:
        self.buffer.extend(samples)

    def __len__(self) -> int:
        return len(self.buffer)

    def sample(self, n: int) -> list[TrainingSample]:
        n = min(n, len(self.buffer))
        return random.sample(list(self.buffer), n)

    def all(self) -> list[TrainingSample]:
        return list(self.buffer)


class TextGeometryDataset(Dataset):
    """PyTorch dataset from text-based training samples."""

    def __init__(
        self,
        samples: list[TrainingSample],
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        augment: bool = False,
        epoch: int = 0,
    ):
        self.samples = samples
        self.max_seq_len = max_seq_len
        self.augment = augment
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        if self.augment:
            aug_text, perm = augment_state_text(
                s.state_text, self.epoch, idx, len(self.samples)
            )
            token_ids = tokenize_and_pad(aug_text, max_len=self.max_seq_len)

            # Rebuild policy from sparse constructions if available
            if s.policy_constructions is not None:
                policy = torch.zeros(POLICY_SIZE, dtype=torch.float32)
                for type_name, args, weight in s.policy_constructions:
                    new_args = permute_point_ids(args, perm)
                    new_idx = construction_to_index(type_name, new_args)
                    policy[new_idx] += weight
            else:
                # Value-only sample, keep zero policy
                policy = torch.tensor(s.policy_target, dtype=torch.float32)
        else:
            token_ids = tokenize_and_pad(s.state_text, max_len=self.max_seq_len)
            policy = torch.tensor(s.policy_target, dtype=torch.float32)

        value = torch.tensor(s.value_target, dtype=torch.float32)
        return token_ids, policy, value


class SyntheticDataset(Dataset):
    """Dataset from Rust-generated synthetic (state_text, construction_text, goal_text) tuples.

    Handles both positive examples (construction makes goal provable, value=1.0)
    and negative examples (construction_text prefixed with "NEG:", value=0.0).
    """

    def __init__(
        self,
        examples: list[tuple[str, str, str]],
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        augment: bool = False,
        epoch: int = 0,
    ):
        self.examples = examples
        self.max_seq_len = max_seq_len
        self.augment = augment
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        state_text, construction_text, goal_text = self.examples[idx]

        # Detect negative examples (prefixed with "NEG:")
        is_negative = construction_text.startswith("NEG:")
        clean_construction = construction_text[4:] if is_negative else construction_text

        if self.augment:
            perm = make_augmentation_perm(self.epoch, idx, len(self.examples))
            state_text = permute_text(state_text, perm)
            goal_text = permute_text(goal_text, perm)
            clean_construction = permute_text(clean_construction, perm)

        # State + goal as input: "coll a b c ; cong a b c d ; ? perp a h b c"
        input_text = f"{state_text} ; ? {goal_text}" if state_text else f"? {goal_text}"

        if self.augment:
            import random as _random
            rng = _random.Random(self.epoch * len(self.examples) + idx)
            from model import shuffle_facts
            input_text = shuffle_facts(input_text, rng)

        token_ids = tokenize_and_pad(input_text, max_len=self.max_seq_len)

        # Policy target: the construction that makes the goal provable
        policy = torch.zeros(POLICY_SIZE, dtype=torch.float32)
        if not is_negative:
            c_type, c_args = _parse_construction_text(clean_construction)
            if c_type is not None:
                idx_val = construction_to_index(c_type, c_args)
                policy[idx_val] = 1.0

        # Value: 1.0 for positive, 0.0 for negative
        value = torch.tensor(0.0 if is_negative else GAMMA, dtype=torch.float32)

        return token_ids, policy, value


# ============================================================
# Set-encoding datasets for SetGeoTransformer
# ============================================================

class SetSyntheticDataset(Dataset):
    """Dataset from Rust-generated synthetic data for SetGeoTransformer.

    Each example is split into per-fact token sequences instead of one flat sequence.
    """

    def __init__(
        self,
        examples: list[tuple[str, str, str]],
        max_facts: int = SET_MAX_FACTS,
        max_tokens: int = SET_MAX_TOKENS,
        augment: bool = False,
        epoch: int = 0,
    ):
        self.examples = examples
        self.max_facts = max_facts
        self.max_tokens = max_tokens
        self.augment = augment
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        state_text, construction_text, goal_text = self.examples[idx]

        is_negative = construction_text.startswith("NEG:")
        clean_construction = construction_text[4:] if is_negative else construction_text

        if self.augment:
            perm = make_augmentation_perm(self.epoch, idx, len(self.examples))
            state_text = permute_text(state_text, perm)
            goal_text = permute_text(goal_text, perm)
            clean_construction = permute_text(clean_construction, perm)

        # Split state into individual facts
        facts = [f.strip() for f in state_text.split(" ; ") if f.strip()]
        if self.augment:
            rng = random.Random(self.epoch * len(self.examples) + idx)
            rng.shuffle(facts)

        fact_ids, goal_ids, fact_mask = encode_state_as_set(
            facts, goal_text, self.max_facts, self.max_tokens,
        )

        # Policy target
        policy = torch.zeros(POLICY_SIZE, dtype=torch.float32)
        if not is_negative:
            c_type, c_args = _parse_construction_text(clean_construction)
            if c_type is not None:
                idx_val = construction_to_index(c_type, c_args)
                policy[idx_val] = 1.0

        value = torch.tensor(0.0 if is_negative else GAMMA, dtype=torch.float32)
        return fact_ids, goal_ids, fact_mask, policy, value


class SetGeometryDataset(Dataset):
    """PyTorch dataset from TrainingSample for SetGeoTransformer."""

    def __init__(
        self,
        samples: list[TrainingSample],
        max_facts: int = SET_MAX_FACTS,
        max_tokens: int = SET_MAX_TOKENS,
        augment: bool = False,
        epoch: int = 0,
    ):
        self.samples = samples
        self.max_facts = max_facts
        self.max_tokens = max_tokens
        self.augment = augment
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        state_text = s.state_text

        if self.augment:
            aug_text, perm = augment_state_text(
                state_text, self.epoch, idx, len(self.samples)
            )
            state_text = aug_text

            if s.policy_constructions is not None:
                policy = torch.zeros(POLICY_SIZE, dtype=torch.float32)
                for type_name, args, weight in s.policy_constructions:
                    new_args = permute_point_ids(args, perm)
                    new_idx = construction_to_index(type_name, new_args)
                    policy[new_idx] += weight
            else:
                policy = torch.tensor(s.policy_target, dtype=torch.float32)
        else:
            policy = torch.tensor(s.policy_target, dtype=torch.float32)

        facts, goal_text = split_state_text(state_text)
        fact_ids, goal_ids, fact_mask = encode_state_as_set(
            facts, goal_text, self.max_facts, self.max_tokens,
        )

        value = torch.tensor(s.value_target, dtype=torch.float32)
        return fact_ids, goal_ids, fact_mask, policy, value


def set_collate_fn(batch):
    """Custom collate for set-encoding datasets.

    Pads the fact dimension (N) to the max in the batch.
    """
    fact_ids_list, goal_ids_list, fact_mask_list, policy_list, value_list = zip(*batch)

    max_n = max(f.shape[0] for f in fact_ids_list)
    L = fact_ids_list[0].shape[1]
    B = len(batch)

    padded_facts = torch.zeros(B, max_n, L, dtype=torch.long)
    padded_masks = torch.zeros(B, max_n, dtype=torch.bool)

    for i, (fi, fm) in enumerate(zip(fact_ids_list, fact_mask_list)):
        n = fi.shape[0]
        padded_facts[i, :n] = fi
        padded_masks[i, :n] = fm

    goal_ids = torch.stack(goal_ids_list)
    policy = torch.stack(policy_list)
    value = torch.stack(value_list)

    return padded_facts, goal_ids, padded_masks, policy, value


# ============================================================
# V2 datasets for SetGeoTransformerV2
# ============================================================

class V2SyntheticDataset(Dataset):
    """Synthetic dataset for SetGeoTransformerV2.

    Each sample: (fact_ids, goal_ids, fact_mask, constr_ids, value_target, is_positive).
    The construction is the single target from synthetic data.
    """

    def __init__(
        self,
        examples: list[tuple[str, str, str]],
        max_facts: int = SET_MAX_FACTS,
        max_tokens: int = SET_MAX_TOKENS,
        augment: bool = False,
        epoch: int = 0,
    ):
        self.examples = examples
        self.max_facts = max_facts
        self.max_tokens = max_tokens
        self.augment = augment
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        state_text, construction_text, goal_text = self.examples[idx]

        is_negative = construction_text.startswith("NEG:")
        clean_construction = construction_text[4:] if is_negative else construction_text

        if self.augment:
            perm = make_augmentation_perm(self.epoch, idx, len(self.examples))
            state_text = permute_text(state_text, perm)
            goal_text = permute_text(goal_text, perm)
            clean_construction = permute_text(clean_construction, perm)

        facts = [f.strip() for f in state_text.split(" ; ") if f.strip()]
        if self.augment:
            rng = random.Random(self.epoch * len(self.examples) + idx)
            rng.shuffle(facts)

        fact_ids, goal_ids, fact_mask = encode_state_as_set(
            facts, goal_text, self.max_facts, self.max_tokens,
        )

        # Single construction
        constr_ids = torch.tensor(
            tokenize_statement(clean_construction, self.max_tokens),
            dtype=torch.long,
        )

        value = torch.tensor(0.0 if is_negative else GAMMA, dtype=torch.float32)
        is_pos = torch.tensor(0.0 if is_negative else 1.0, dtype=torch.float32)

        return fact_ids, goal_ids, fact_mask, constr_ids, value, is_pos


class V2GeometryDataset(Dataset):
    """Training dataset from TrainingSample for SetGeoTransformerV2.

    Each sample uses policy_constructions for variable-count constructions.
    """

    def __init__(
        self,
        samples: list,
        max_facts: int = SET_MAX_FACTS,
        max_tokens: int = SET_MAX_TOKENS,
        augment: bool = False,
        epoch: int = 0,
    ):
        self.samples = samples
        self.max_facts = max_facts
        self.max_tokens = max_tokens
        self.augment = augment
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        state_text = s.state_text

        if self.augment:
            aug_text, perm = augment_state_text(
                state_text, self.epoch, idx, len(self.samples)
            )
            state_text = aug_text
        else:
            perm = None

        facts, goal_text = split_state_text(state_text)
        fact_ids, goal_ids, fact_mask = encode_state_as_set(
            facts, goal_text, self.max_facts, self.max_tokens,
        )

        # Build construction IDs and weights from policy_constructions
        if s.policy_constructions:
            constr_id_list = []
            weights = []
            for type_name, args, weight in s.policy_constructions:
                if perm is not None:
                    args = permute_point_ids(args, perm)
                # Reconstruct construction text
                rev_type = {v: k for k, v in _KEYWORD_TO_TYPE.items()}
                keyword = rev_type.get(type_name, type_name.lower())
                from model import POINT_NAMES as _PN
                arg_names = [_PN[a] if 0 <= a < len(_PN) else f"aux_{a-26}" for a in args]
                ctext = keyword + " " + " ".join(arg_names)
                constr_id_list.append(
                    torch.tensor(tokenize_statement(ctext, self.max_tokens), dtype=torch.long)
                )
                weights.append(weight)

            constr_ids = torch.stack(constr_id_list)  # (K, L)
            constr_weights = torch.tensor(weights, dtype=torch.float32)
            # Normalize weights to form distribution
            if constr_weights.sum() > 0:
                constr_weights = constr_weights / constr_weights.sum()
        else:
            # Value-only sample: single dummy construction
            constr_ids = torch.zeros(1, self.max_tokens, dtype=torch.long)
            constr_weights = torch.zeros(1, dtype=torch.float32)

        constr_mask = torch.ones(constr_ids.shape[0], dtype=torch.bool)
        value = torch.tensor(s.value_target, dtype=torch.float32)

        return fact_ids, goal_ids, fact_mask, constr_ids, constr_mask, constr_weights, value


def v2_collate_fn(batch):
    """Custom collate for V2 datasets.

    Pads both fact dimension N and construction dimension K per batch.
    Handles both V2SyntheticDataset (6 items) and V2GeometryDataset (7 items).
    """
    n_items = len(batch[0])

    if n_items == 6:
        # V2SyntheticDataset: (fact_ids, goal_ids, fact_mask, constr_ids, value, is_pos)
        fact_ids_list, goal_ids_list, fact_mask_list, constr_ids_list, value_list, is_pos_list = zip(*batch)

        B = len(batch)
        max_n = max(f.shape[0] for f in fact_ids_list)
        L = fact_ids_list[0].shape[1]

        padded_facts = torch.zeros(B, max_n, L, dtype=torch.long)
        padded_masks = torch.zeros(B, max_n, dtype=torch.bool)
        for i, (fi, fm) in enumerate(zip(fact_ids_list, fact_mask_list)):
            n = fi.shape[0]
            padded_facts[i, :n] = fi
            padded_masks[i, :n] = fm

        goal_ids = torch.stack(goal_ids_list)
        # Constructions: each is (L,), pad to (B, 1, L) for single construction
        constr_ids = torch.stack(constr_ids_list).unsqueeze(1)  # (B, 1, L)
        constr_mask = torch.ones(B, 1, dtype=torch.bool)
        # Policy: target construction gets weight 1.0 for positive, 0.0 for negative
        constr_weights = torch.stack(is_pos_list).unsqueeze(1)  # (B, 1)
        values = torch.stack(value_list)

        return padded_facts, goal_ids, padded_masks, constr_ids, constr_mask, constr_weights, values

    else:
        # V2GeometryDataset: (fact_ids, goal_ids, fact_mask, constr_ids, constr_mask, constr_weights, value)
        fact_ids_list, goal_ids_list, fact_mask_list, constr_ids_list, constr_mask_list, constr_weights_list, value_list = zip(*batch)

        B = len(batch)
        max_n = max(f.shape[0] for f in fact_ids_list)
        max_k = max(c.shape[0] for c in constr_ids_list)
        L = fact_ids_list[0].shape[1]
        Lc = constr_ids_list[0].shape[1]

        padded_facts = torch.zeros(B, max_n, L, dtype=torch.long)
        padded_masks = torch.zeros(B, max_n, dtype=torch.bool)
        padded_constrs = torch.zeros(B, max_k, Lc, dtype=torch.long)
        padded_cmasks = torch.zeros(B, max_k, dtype=torch.bool)
        padded_cweights = torch.zeros(B, max_k, dtype=torch.float32)

        for i in range(B):
            fi = fact_ids_list[i]
            fm = fact_mask_list[i]
            padded_facts[i, :fi.shape[0]] = fi
            padded_masks[i, :fi.shape[0]] = fm

            ci = constr_ids_list[i]
            cm = constr_mask_list[i]
            cw = constr_weights_list[i]
            padded_constrs[i, :ci.shape[0]] = ci
            padded_cmasks[i, :ci.shape[0]] = cm
            padded_cweights[i, :cw.shape[0]] = cw

        goal_ids = torch.stack(goal_ids_list)
        values = torch.stack(value_list)

        return padded_facts, goal_ids, padded_masks, padded_constrs, padded_cmasks, padded_cweights, values


# Map from text keywords to ConstructionType names
_KEYWORD_TO_TYPE = {
    "mid": "Midpoint",
    "alt": "Altitude",
    "circumcenter": "Circumcenter",
    "orthocenter": "Orthocenter",
    "incenter": "Incenter",
    "pthrough": "ParallelThrough",
    "tthrough": "PerpendicularThrough",
}


def _parse_construction_text(text: str) -> tuple[str | None, list[int]]:
    """Parse 'mid a b' -> ('Midpoint', [id_a, id_b]).

    Since we don't have the actual state, we use point name -> sequential ID mapping.
    """
    parts = text.split()
    if len(parts) < 2:
        return None, []
    keyword = parts[0]
    c_type = _KEYWORD_TO_TYPE.get(keyword)
    if c_type is None:
        return None, []
    # Map point names to sequential IDs (a=0, b=1, etc.)
    args = []
    for name in parts[1:]:
        if name.startswith("aux_"):
            try:
                args.append(26 + int(name[4:]))
            except ValueError:
                args.append(0)
        elif len(name) == 1 and name.isalpha():
            args.append(ord(name) - ord("a"))
        else:
            args.append(0)
    return c_type, args


def compute_loss(
    model,
    token_ids: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
    value_weight: float = DEFAULT_VALUE_WEIGHT,
) -> tuple[torch.Tensor, dict]:
    """Compute combined policy + value loss (for GeoTransformer)."""
    value_pred, policy_logits = model(token_ids)

    # Policy loss: KL(target || predicted)
    log_probs = F.log_softmax(policy_logits, dim=1)
    policy_mask = policy_targets > 0
    if policy_mask.any():
        policy_loss = F.kl_div(
            log_probs, policy_targets, reduction="batchmean", log_target=False
        )
    else:
        policy_loss = torch.tensor(0.0, device=token_ids.device)

    # Value loss: MSE between sigmoid value and target
    value_loss = F.mse_loss(value_pred, value_targets)

    total_loss = policy_loss + value_weight * value_loss

    metrics = {
        "loss": total_loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "mean_pred_value": value_pred.mean().item(),
    }
    return total_loss, metrics


def compute_set_loss(
    model: SetGeoTransformer,
    fact_ids: torch.Tensor,
    goal_ids: torch.Tensor,
    fact_mask: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
    value_weight: float = DEFAULT_VALUE_WEIGHT,
) -> tuple[torch.Tensor, dict]:
    """Compute combined policy + value loss (for SetGeoTransformer)."""
    value_pred, policy_logits = model(fact_ids, goal_ids, fact_mask)

    log_probs = F.log_softmax(policy_logits, dim=1)
    policy_mask = policy_targets > 0
    if policy_mask.any():
        policy_loss = F.kl_div(
            log_probs, policy_targets, reduction="batchmean", log_target=False
        )
    else:
        policy_loss = torch.tensor(0.0, device=fact_ids.device)

    value_loss = F.mse_loss(value_pred, value_targets)
    total_loss = policy_loss + value_weight * value_loss

    metrics = {
        "loss": total_loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "mean_pred_value": value_pred.mean().item(),
    }
    return total_loss, metrics


def compute_v2_loss(
    model: SetGeoTransformerV2,
    fact_ids: torch.Tensor,
    goal_ids: torch.Tensor,
    fact_mask: torch.Tensor,
    constr_ids: torch.Tensor,
    constr_mask: torch.Tensor,
    constr_weights: torch.Tensor,
    value_targets: torch.Tensor,
    value_weight: float = DEFAULT_VALUE_WEIGHT,
) -> tuple[torch.Tensor, dict]:
    """Compute combined policy + value loss for SetGeoTransformerV2.

    Args:
        fact_ids: (B, N, L)
        goal_ids: (B, L)
        fact_mask: (B, N)
        constr_ids: (B, K, L)
        constr_mask: (B, K)
        constr_weights: (B, K) target distribution over constructions
        value_targets: (B,)
    """
    value_pred, logits = model(fact_ids, goal_ids, fact_mask, constr_ids, constr_mask)

    # Policy loss: KL-div over per-construction logits vs target distribution
    policy_has_target = constr_weights.sum(dim=1) > 0  # (B,)
    if policy_has_target.any():
        # Only compute policy loss on samples with valid targets
        valid_logits = logits[policy_has_target]  # (B', K)
        valid_targets = constr_weights[policy_has_target]  # (B', K)
        valid_cmask = constr_mask[policy_has_target]  # (B', K)

        # Mask invalid constructions before softmax
        valid_logits = valid_logits.masked_fill(~valid_cmask, float("-inf"))
        log_probs = F.log_softmax(valid_logits, dim=1)

        # Replace -inf in log_probs with 0 to avoid NaN in KL div
        log_probs = log_probs.masked_fill(~valid_cmask, 0.0)

        policy_loss = F.kl_div(
            log_probs, valid_targets, reduction="batchmean", log_target=False,
        )
    else:
        policy_loss = torch.tensor(0.0, device=fact_ids.device)

    value_loss = F.mse_loss(value_pred, value_targets)
    total_loss = policy_loss + value_weight * value_loss

    metrics = {
        "loss": total_loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "mean_pred_value": value_pred.mean().item(),
    }
    return total_loss, metrics


def generate_supervised_data(
    problems_file: str,
    model: GeoTransformer | None = None,
    max_problems: int | None = None,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    device: str = "cpu",
) -> list[TrainingSample]:
    """Generate supervised training data from JGEX problems.

    For deduction-solvable problems: run short MCTS on the pre-saturated state
    to get visit distributions as policy targets (instead of zero vectors).
    For MCTS-solvable problems: collect all-node samples from the search tree.
    """
    problems = load_problems(problems_file)
    if max_problems:
        problems = problems[:max_problems]

    samples = []
    deduction_solved = 0
    mcts_solved = 0

    # Short MCTS config for generating policy targets
    mcts_config = MctsConfig(
        num_iterations=50,
        max_children=20,
        max_depth=3,
        max_seq_len=max_seq_len,
    )

    print(f"Generating supervised data from {len(problems)} problems...")
    for name, definition in problems:
        problem_text = f"{name}\n{definition}"
        try:
            state = geoprover.parse_problem(problem_text)
            state_text = geoprover.state_to_text(state)

            proved = geoprover.saturate(state)
            if proved:
                deduction_solved += 1

                if model is not None:
                    # Run short MCTS on the pre-saturated state to get policy targets
                    pre_state = geoprover.parse_problem(problem_text)
                    result = mcts_search(pre_state, model, mcts_config, device)
                    if result.samples:
                        samples.extend(result.samples)
                    else:
                        # Fallback: add value-only sample
                        samples.append(TrainingSample(
                            state_text=state_text,
                            policy_target=[0.0] * POLICY_SIZE,
                            value_target=1.0,
                        ))
                else:
                    # No model available - generate constructions and uniform policy
                    pre_state = geoprover.parse_problem(problem_text)
                    constructions = geoprover.generate_constructions(pre_state)
                    if constructions:
                        policy_target = [0.0] * POLICY_SIZE
                        weight = 1.0 / len(constructions)
                        pc_list = []
                        for c in constructions:
                            idx = construction_to_index(c.construction_type(), c.args())
                            policy_target[idx] = weight
                            pc_list.append((c.construction_type(), c.args(), weight))
                    else:
                        policy_target = [0.0] * POLICY_SIZE
                        pc_list = None
                    samples.append(TrainingSample(
                        state_text=state_text,
                        policy_target=policy_target,
                        value_target=1.0,
                        policy_constructions=pc_list,
                    ))
            elif model is not None:
                # Try MCTS on unsolved problems to find MCTS-solvable ones
                pre_state = geoprover.parse_problem(problem_text)
                result = mcts_search(pre_state, model, mcts_config, device)
                if result.solved:
                    mcts_solved += 1
                    samples.extend(result.samples)
        except Exception as e:
            print(f"  Skip {name}: {e}")

    total_solved = deduction_solved + mcts_solved
    print(f"  Generated {len(samples)} samples from {total_solved}/{len(problems)} solved "
          f"(deduction: {deduction_solved}, MCTS: {mcts_solved})")
    return samples


def train_epoch(
    model,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    value_weight: float = DEFAULT_VALUE_WEIGHT,
    epoch: int = 0,
    model_type: str = "transformer",
    scaler: torch.amp.GradScaler | None = None,
) -> dict:
    """Train for one epoch. Returns average metrics."""
    # Update dataset epoch for augmentation
    if hasattr(dataloader.dataset, 'epoch'):
        dataloader.dataset.epoch = epoch
    model.train()
    total_metrics = {}
    n_batches = 0

    is_set = model_type == "set"
    is_v2 = model_type == "set_v2"
    use_amp = scaler is not None
    amp_device_type = "cuda" if device == "cuda" else "cpu"

    for batch in dataloader:
        if is_v2:
            fact_ids, goal_ids, fact_mask, constr_ids, constr_mask, constr_weights, value_targets = batch
            fact_ids = fact_ids.to(device)
            goal_ids = goal_ids.to(device)
            fact_mask = fact_mask.to(device)
            constr_ids = constr_ids.to(device)
            constr_mask = constr_mask.to(device)
            constr_weights = constr_weights.to(device)
            value_targets = value_targets.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(amp_device_type, enabled=use_amp):
                loss, metrics = compute_v2_loss(
                    model, fact_ids, goal_ids, fact_mask,
                    constr_ids, constr_mask, constr_weights,
                    value_targets, value_weight,
                )
        elif is_set:
            fact_ids, goal_ids, fact_mask, policy_targets, value_targets = batch
            fact_ids = fact_ids.to(device)
            goal_ids = goal_ids.to(device)
            fact_mask = fact_mask.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(amp_device_type, enabled=use_amp):
                loss, metrics = compute_set_loss(
                    model, fact_ids, goal_ids, fact_mask,
                    policy_targets, value_targets, value_weight,
                )
        else:
            token_ids, policy_targets, value_targets = batch
            token_ids = token_ids.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(amp_device_type, enabled=use_amp):
                loss, metrics = compute_loss(
                    model, token_ids, policy_targets, value_targets, value_weight
                )

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        for k_name, v in metrics.items():
            total_metrics[k_name] = total_metrics.get(k_name, 0.0) + v
        n_batches += 1

    if n_batches > 0:
        for k_name in total_metrics:
            total_metrics[k_name] /= n_batches
    return total_metrics


def save_checkpoint(
    model: GeoTransformer,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    metrics: dict,
    path: str,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
        "metrics": metrics,
    }, path)
    print(f"  Saved checkpoint: {path}")


def load_checkpoint(
    model: GeoTransformer,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: str,
) -> int:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("iteration", 0)


def _make_typed_loader(
    samples,
    model_type: str,
    batch_size: int,
    num_workers: int,
    device: str,
    augment: bool = True,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    is_synthetic: bool = False,
) -> DataLoader:
    """Create a DataLoader for the given model type and sample source."""
    if model_type == "set_v2":
        if is_synthetic:
            dataset = V2SyntheticDataset(samples, augment=augment)
        else:
            dataset = V2GeometryDataset(samples, augment=augment)
        collate_fn = v2_collate_fn
    elif model_type == "set":
        if is_synthetic:
            dataset = SetSyntheticDataset(samples, augment=augment)
        else:
            dataset = SetGeometryDataset(samples, augment=augment)
        collate_fn = set_collate_fn
    else:
        if is_synthetic:
            dataset = SyntheticDataset(samples, max_seq_len=max_seq_len, augment=augment)
        else:
            dataset = TextGeometryDataset(samples, max_seq_len=max_seq_len, augment=augment)
        collate_fn = None
    return _make_loader(
        dataset, batch_size, shuffle=True, collate_fn=collate_fn,
        num_workers=num_workers, device=device,
    )


def expert_iteration(
    model,
    problems_file: str = DEFAULT_PROBLEMS_FILE,
    num_iterations: int = DEFAULT_NUM_ITERATIONS,
    epochs_per_iter: int = DEFAULT_EPOCHS_PER_ITER,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    value_weight: float = DEFAULT_VALUE_WEIGHT,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    device: str = "cpu",
    resume_from: str | None = None,
    synthetic_size: int = DEFAULT_SYNTHETIC_SIZE,
    synthetic_seed: int = DEFAULT_SYNTHETIC_SEED,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    augment: bool = True,
    model_type: str = "set_v2",
    num_workers: int = 0,
):
    """Run expert iteration training loop.

    1. Generate synthetic pre-training data (from Rust engine)
    2. Pre-train on synthetic data
    3. Fine-tune on JGEX supervised data
    4. Expert iteration: MCTS self-play -> train -> repeat
    """
    problems = load_problems(problems_file)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=DEFAULT_WEIGHT_DECAY
    )
    # Total training steps: synthetic (2x) + supervised (1x) + expert (num_iterations)
    total_epochs = epochs_per_iter * 2 + epochs_per_iter + num_iterations * epochs_per_iter
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=lr * 0.01,
    )
    replay = ReplayBuffer()
    start_iter = 0
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    if resume_from and os.path.exists(resume_from):
        start_iter = load_checkpoint(model, optimizer, resume_from, device)
        print(f"Resumed from iteration {start_iter}")

    # Phase A: Synthetic pre-training
    print("=" * 60)
    print("Phase A: Synthetic pre-training")
    print("=" * 60)
    print(f"Generating {synthetic_size} synthetic examples (seed={synthetic_seed})...")
    t0 = time.time()
    # Generate in batches for progress reporting
    batch_gen_size = min(synthetic_size, 2000)
    synthetic_data = []
    generated = 0
    while generated < synthetic_size:
        batch = min(batch_gen_size, synthetic_size - generated)
        chunk = geoprover.generate_synthetic_data(batch, synthetic_seed + generated)
        synthetic_data.extend(chunk)
        generated += batch
        elapsed = time.time() - t0
        rate = len(synthetic_data) / elapsed if elapsed > 0 else 0
        print(f"  {len(synthetic_data)}/{synthetic_size} ({rate:.0f}/s, {elapsed:.0f}s)")
    elapsed = time.time() - t0
    print(f"  Generated {len(synthetic_data)} examples in {elapsed:.1f}s")

    if synthetic_data:
        loader = _make_typed_loader(
            synthetic_data, model_type, batch_size, num_workers, device,
            augment=augment, max_seq_len=max_seq_len, is_synthetic=True,
        )
        for epoch in range(epochs_per_iter * 2):  # more epochs for synthetic
            metrics = train_epoch(model, loader, optimizer, device, value_weight, epoch=epoch, model_type=model_type, scaler=scaler)
            scheduler.step()
            print(f"  Synthetic epoch {epoch+1}: "
                  f"loss={metrics['loss']:.4f} "
                  f"policy={metrics['policy_loss']:.4f} "
                  f"value={metrics['value_loss']:.4f}")
        save_checkpoint(
            model, optimizer, 0, metrics,
            os.path.join(checkpoint_dir, "synthetic.pt"),
        )

    # Phase B: JGEX supervised fine-tuning
    print("=" * 60)
    print("Phase B: JGEX supervised fine-tuning")
    print("=" * 60)
    supervised_samples = generate_supervised_data(
        problems_file, model=model, max_seq_len=max_seq_len, device=device,
    )
    if supervised_samples:
        replay.add(supervised_samples)
        loader = _make_typed_loader(
            supervised_samples, model_type, batch_size, num_workers, device,
            augment=augment, max_seq_len=max_seq_len,
        )
        for epoch in range(epochs_per_iter):
            metrics = train_epoch(model, loader, optimizer, device, value_weight, epoch=epoch, model_type=model_type, scaler=scaler)
            scheduler.step()
            print(f"  Supervised epoch {epoch+1}: "
                  f"loss={metrics['loss']:.4f} "
                  f"policy={metrics['policy_loss']:.4f} "
                  f"value={metrics['value_loss']:.4f}")
        save_checkpoint(
            model, optimizer, 0, metrics,
            os.path.join(checkpoint_dir, "pretrained.pt"),
        )

    # Phase C: Expert iteration
    print("=" * 60)
    print("Phase C: Expert iteration")
    print("=" * 60)
    mcts_config = MctsConfig(
        num_iterations=DEFAULT_SELFPLAY_MCTS_ITERS,
        max_children=20,
        max_depth=3,
        max_seq_len=max_seq_len,
        model_type=model_type,
    )

    for iteration in range(start_iter, num_iterations):
        print(f"\n--- Iteration {iteration+1}/{num_iterations} ---")
        t0 = time.time()

        print("  Self-play...")
        new_samples = self_play_episode(problems, model, mcts_config, device)
        replay.add(new_samples)
        print(f"  Collected {len(new_samples)} samples (buffer: {len(replay)})")

        if len(replay) >= batch_size:
            train_samples = replay.all()
            loader = _make_typed_loader(
                train_samples, model_type, batch_size, num_workers, device,
                augment=augment, max_seq_len=max_seq_len,
            )
            for epoch in range(epochs_per_iter):
                metrics = train_epoch(model, loader, optimizer, device, value_weight, epoch=epoch, model_type=model_type, scaler=scaler)
                scheduler.step()
            print(f"  Train: loss={metrics['loss']:.4f} "
                  f"policy={metrics['policy_loss']:.4f} "
                  f"value={metrics['value_loss']:.4f} "
                  f"mean_v={metrics['mean_pred_value']:.3f}")

        save_checkpoint(
            model, optimizer, iteration + 1, metrics,
            os.path.join(checkpoint_dir, f"iter_{iteration+1:03d}.pt"),
        )
        elapsed = time.time() - t0
        print(f"  Iteration time: {elapsed:.1f}s")

    print("\nTraining complete!")
    return model


def _generate_synthetic_background(
    total_size: int,
    seed: int,
    initial_size: int = 10000,
) -> tuple[list, threading.Thread | None]:
    """Generate synthetic data, returning initial batch and background thread for the rest.

    Returns (initial_data, bg_thread). If total_size <= initial_size, bg_thread is None.
    Call bg_thread.join() then access bg_thread.result for remaining data.
    """
    batch_gen_size = min(total_size, 2000)

    # Generate initial batch synchronously
    initial_data = []
    generated = 0
    t0 = time.time()
    while generated < min(total_size, initial_size):
        batch = min(batch_gen_size, min(total_size, initial_size) - generated)
        chunk = geoprover.generate_synthetic_data(batch, seed + generated)
        initial_data.extend(chunk)
        generated += batch
        elapsed = time.time() - t0
        rate = len(initial_data) / elapsed if elapsed > 0 else 0
        print(f"  {len(initial_data)}/{total_size} ({rate:.0f}/s, {elapsed:.0f}s)")

    if generated >= total_size:
        return initial_data, None

    # Spawn background thread for the rest (GIL released in Rust)
    remaining = total_size - generated
    bg_seed = seed + generated

    class BgThread(threading.Thread):
        def __init__(self):
            super().__init__(daemon=True)
            self.result: list = []

        def run(self):
            data = []
            gen = 0
            while gen < remaining:
                batch = min(batch_gen_size, remaining - gen)
                chunk = geoprover.generate_synthetic_data(batch, bg_seed + gen)
                data.extend(chunk)
                gen += batch
            self.result = data

    bg = BgThread()
    bg.start()
    print(f"  Background thread generating {remaining} more examples...")
    return initial_data, bg


def main():
    parser = argparse.ArgumentParser(description="Train GeoTransformer")
    parser.add_argument("--problems", default=DEFAULT_PROBLEMS_FILE, help="Problem file path")
    parser.add_argument("--iterations", type=int, default=DEFAULT_NUM_ITERATIONS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS_PER_ITER)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (default: 256 for CUDA, 128 for CPU)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: auto-scaled with batch size)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--supervised-only", action="store_true",
                        help="Only run supervised pre-training, no self-play")
    parser.add_argument("--synthetic-size", type=int, default=DEFAULT_SYNTHETIC_SIZE)
    parser.add_argument("--synthetic-seed", type=int, default=DEFAULT_SYNTHETIC_SEED)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable data augmentation (label permutation + fact shuffling)")
    parser.add_argument("--model-type", default="set_v2",
                        choices=["set", "set_v2", "transformer"],
                        help="Model architecture: 'set' for SetGeoTransformerV1, 'set_v2' for V2, 'transformer' for GeoTransformer")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="DataLoader workers (default: 4 for CUDA, 0 for CPU)")
    parser.add_argument("--synthetic-cache", default=None,
                        help="Path to cache synthetic data (saves/loads .json)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile on CUDA")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    print(f"Device: {device}")

    # Auto-tune batch size and learning rate
    batch_size = args.batch_size if args.batch_size is not None else _get_default_batch_size(device)
    lr = args.lr if args.lr is not None else DEFAULT_LR * (batch_size / DEFAULT_BATCH_SIZE)
    num_workers = args.num_workers if args.num_workers is not None else (4 if device == "cuda" else 0)
    print(f"Batch size: {batch_size}, LR: {lr:.6f}, Workers: {num_workers}")

    model = create_model(args.model_type).to(device)
    model_name = type(model).__name__
    print(f"{model_name} parameters: {count_parameters(model):,}")

    # torch.compile for CUDA
    if device == "cuda" and not args.no_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile enabled (reduce-overhead)")
        except Exception as e:
            print(f"torch.compile failed, continuing without: {e}")

    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
    if scaler:
        print("AMP enabled (float16)")

    if args.supervised_only:
        # Generate synthetic + supervised data and train
        print("Generating synthetic data...")
        t0 = time.time()

        # Load from cache or generate
        bg_thread = None
        if args.synthetic_cache and os.path.exists(args.synthetic_cache):
            with open(args.synthetic_cache) as f:
                synthetic_data = json.load(f)
            # Convert lists back to tuples
            synthetic_data = [tuple(x) for x in synthetic_data]
            print(f"Loaded {len(synthetic_data)} cached synthetic examples from {args.synthetic_cache}")
        else:
            synthetic_data, bg_thread = _generate_synthetic_background(
                args.synthetic_size, args.synthetic_seed,
            )

        num_synthetic_epochs = args.epochs * 3
        num_bg_epochs = args.epochs if bg_thread is not None else 0
        num_supervised_epochs = args.epochs * 5
        total_epochs = num_synthetic_epochs + num_bg_epochs + num_supervised_epochs
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=DEFAULT_WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=lr * 0.01,
        )

        augment = not args.no_augment
        mt = args.model_type
        if synthetic_data:
            loader = _make_typed_loader(
                synthetic_data, mt, batch_size, num_workers, device,
                augment=augment, max_seq_len=args.max_seq_len, is_synthetic=True,
            )
            for epoch in range(num_synthetic_epochs):
                metrics = train_epoch(model, loader, optimizer, device, epoch=epoch, model_type=mt, scaler=scaler)
                scheduler.step()
                print(f"Synthetic epoch {epoch+1}/{num_synthetic_epochs}: "
                      f"loss={metrics['loss']:.4f} "
                      f"policy={metrics['policy_loss']:.4f} "
                      f"value={metrics['value_loss']:.4f}")

            # Collect background data and train on full dataset
            if bg_thread is not None:
                print("Waiting for background synthetic generation...")
                bg_thread.join()
                synthetic_data.extend(bg_thread.result)
                print(f"Total synthetic examples: {len(synthetic_data)}")
                if bg_thread.result:
                    loader = _make_typed_loader(
                        synthetic_data, mt, batch_size, num_workers, device,
                        augment=augment, max_seq_len=args.max_seq_len, is_synthetic=True,
                    )
                    for epoch in range(num_bg_epochs):
                        metrics = train_epoch(model, loader, optimizer, device, epoch=epoch, model_type=mt, scaler=scaler)
                        scheduler.step()
                        print(f"Synthetic+bg epoch {epoch+1}/{num_bg_epochs}: "
                              f"loss={metrics['loss']:.4f} "
                              f"policy={metrics['policy_loss']:.4f} "
                              f"value={metrics['value_loss']:.4f}")

            # Save cache if requested
            if args.synthetic_cache and not os.path.exists(args.synthetic_cache):
                os.makedirs(os.path.dirname(args.synthetic_cache) or ".", exist_ok=True)
                with open(args.synthetic_cache, "w") as f:
                    json.dump(synthetic_data, f)
                print(f"Cached {len(synthetic_data)} synthetic examples to {args.synthetic_cache}")

            save_checkpoint(
                model, optimizer, 0, metrics,
                os.path.join(args.checkpoint_dir, "synthetic.pt"),
            )

        samples = generate_supervised_data(
            args.problems, model=model, max_seq_len=args.max_seq_len, device=device,
        )
        if samples:
            loader = _make_typed_loader(
                samples, mt, batch_size, num_workers, device,
                augment=augment, max_seq_len=args.max_seq_len,
            )
            for epoch in range(num_supervised_epochs):
                metrics = train_epoch(model, loader, optimizer, device, epoch=epoch, model_type=mt, scaler=scaler)
                scheduler.step()
                print(f"Supervised epoch {epoch+1}/{num_supervised_epochs}: "
                      f"loss={metrics['loss']:.4f} "
                      f"policy={metrics['policy_loss']:.4f} "
                      f"value={metrics['value_loss']:.4f}")

        save_checkpoint(
            model, optimizer, 0, metrics,
            os.path.join(args.checkpoint_dir, "supervised.pt"),
        )
    else:
        expert_iteration(
            model,
            problems_file=args.problems,
            num_iterations=args.iterations,
            epochs_per_iter=args.epochs,
            batch_size=batch_size,
            lr=lr,
            checkpoint_dir=args.checkpoint_dir,
            device=device,
            resume_from=args.resume,
            synthetic_size=args.synthetic_size,
            synthetic_seed=args.synthetic_seed,
            max_seq_len=args.max_seq_len,
            augment=not args.no_augment,
            model_type=args.model_type,
            num_workers=num_workers,
        )


if __name__ == "__main__":
    main()
