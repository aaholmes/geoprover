"""Training loop for GeoTransformer: synthetic pre-training + expert iteration.

Phases:
  0. (Optional) Summarizer pre-training: train FactSummarizer on proof traces
  A. Synthetic pre-training: train on Rust-generated synthetic data (100K+ examples)
  B. JGEX supervised fine-tuning on deduction-solvable problems
  C. Expert iteration: MCTS self-play -> collect data -> train -> repeat

Loss: L = KL(policy || target) + c_value * MSE(value, target)
"""

import argparse
import json
import os
import random
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import geoprover
from model import (
    POLICY_SIZE,
    GeoNet,
    GeoTransformer,
    build_valid_mask,
    construction_to_index,
    count_parameters,
    tokenize_and_pad,
    augment_state_text,
    permute_text,
    permute_point_ids,
    make_augmentation_perm,
    MAX_SEQ_LEN,
)
from orchestrate import (
    MctsConfig,
    TrainingSample,
    load_problems,
    mcts_search,
    self_play_episode,
    solve_problem,
)
from summarizer import (
    FactSummarizer,
    SummarizerSample,
    build_context_tokens,
    build_fact_tokens,
    count_summarizer_parameters,
    generate_summarizer_data,
    SUMMARIZER_MAX_SEQ_LEN,
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
        value = torch.tensor(0.0 if is_negative else 1.0, dtype=torch.float32)

        return token_ids, policy, value


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


# ============================================================
# Summarizer training
# ============================================================

class SummarizerDataset(Dataset):
    """PyTorch dataset for Summarizer training.

    Each item is one *problem* (not one fact), returning all deduced facts at once:
      - context_ids: (L_ctx,) token IDs for initial facts + goal
      - fact_ids: (N_facts, L_fact) token IDs for all deduced facts
      - labels: (N_facts,) 1.0 if on proof path, 0.0 otherwise

    This allows the training loop to encode the context once per problem
    and score all facts against it, avoiding redundant context encoding.

    With augment=True, applies label permutation + fact shuffling per epoch.
    """

    def __init__(
        self,
        samples: list[SummarizerSample],
        context_max_len: int = SUMMARIZER_MAX_SEQ_LEN,
        fact_max_len: int = 32,
        augment: bool = False,
        epoch: int = 0,
    ):
        self.samples = samples
        self.context_max_len = context_max_len
        self.fact_max_len = fact_max_len
        self.augment = augment
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        initial_facts = list(s.initial_facts)
        goal_text = s.goal_text
        deduced_facts = list(s.deduced_facts)
        labels = list(s.labels)

        if self.augment:
            perm = make_augmentation_perm(self.epoch, idx, len(self.samples))
            initial_facts = [permute_text(f, perm) for f in initial_facts]
            if goal_text:
                goal_text = permute_text(goal_text, perm)
            deduced_facts = [permute_text(f, perm) for f in deduced_facts]
            # Shuffle initial facts order
            rng = random.Random(self.epoch * len(self.samples) + idx)
            rng.shuffle(initial_facts)

        ctx_ids = build_context_tokens(initial_facts, goal_text, self.context_max_len)
        fact_ids = build_fact_tokens(deduced_facts, self.fact_max_len)
        return ctx_ids, fact_ids, torch.tensor(labels, dtype=torch.float32)


def train_summarizer(
    summarizer: FactSummarizer,
    samples: list[SummarizerSample],
    num_epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    augment: bool = True,
) -> FactSummarizer:
    """Train the Summarizer on proof trace labels with BCE loss.

    Each problem's context is encoded once; all its deduced facts are scored
    against that single context encoding. Facts from multiple problems are
    accumulated into mini-batches of ~batch_size fact-level examples.
    """
    dataset = SummarizerDataset(samples, augment=augment)
    total_facts = sum(len(s.deduced_facts) for s in samples)
    print(f"  Summarizer dataset: {len(dataset)} problems, {total_facts} fact-level examples "
          f"(augment={augment})")

    if len(dataset) == 0:
        print("  No training data, skipping Summarizer training")
        return summarizer

    optimizer = torch.optim.Adam(summarizer.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01,
    )

    summarizer.to(device)
    for epoch in range(num_epochs):
        dataset.epoch = epoch
        summarizer.train()
        total_loss = 0.0
        total_correct = 0
        total_items = 0
        n_steps = 0

        # Shuffle problem order each epoch
        indices = list(range(len(dataset)))
        random.Random(epoch).shuffle(indices)

        # Process per-problem: encode context ONCE, score all facts against it.
        # Accumulate gradients across problems, step every ~batch_size facts.
        optimizer.zero_grad()
        acc_facts_since_step = 0

        for problem_idx in indices:
            ctx_ids, fact_ids, labels = dataset[problem_idx]
            # ctx_ids: (L_ctx,), fact_ids: (N_facts, L_fact), labels: (N_facts,)
            n_facts = fact_ids.shape[0]
            if n_facts == 0:
                continue

            ctx_ids = ctx_ids.unsqueeze(0).to(device)  # (1, L_ctx)
            fact_ids = fact_ids.to(device)              # (N_facts, L_fact)
            labels = labels.to(device)                  # (N_facts,)

            # NNUE trick: encode context once (1, d), score all N facts
            scores = summarizer.score_facts(ctx_ids, fact_ids)  # (N_facts,)
            loss = F.binary_cross_entropy_with_logits(scores, labels)
            # Scale loss by number of facts so gradient magnitude is consistent
            (loss * n_facts / batch_size).backward()

            total_loss += loss.item() * n_facts
            preds = (scores > 0).float()
            total_correct += (preds == labels).sum().item()
            total_items += n_facts
            acc_facts_since_step += n_facts

            if acc_facts_since_step >= batch_size:
                torch.nn.utils.clip_grad_norm_(summarizer.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                acc_facts_since_step = 0
                n_steps += 1

        # Flush remaining gradients
        if acc_facts_since_step > 0:
            torch.nn.utils.clip_grad_norm_(summarizer.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            n_steps += 1

        scheduler.step()
        avg_loss = total_loss / max(total_items, 1)
        accuracy = total_correct / max(total_items, 1)
        print(f"  Summarizer epoch {epoch+1}/{num_epochs}: "
              f"loss={avg_loss:.4f} acc={accuracy:.3f}")

    # Save checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, "summarizer.pt")
    torch.save({
        "model_state_dict": summarizer.state_dict(),
        "num_epochs": num_epochs,
    }, path)
    print(f"  Saved Summarizer checkpoint: {path}")

    return summarizer


def load_summarizer_checkpoint(
    summarizer: FactSummarizer,
    path: str,
    device: str,
) -> None:
    """Load a Summarizer checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    summarizer.load_state_dict(checkpoint["model_state_dict"])


def compute_loss(
    model: GeoTransformer,
    token_ids: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
    value_weight: float = DEFAULT_VALUE_WEIGHT,
) -> tuple[torch.Tensor, dict]:
    """Compute combined policy + value loss."""
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
    model: GeoTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    value_weight: float = DEFAULT_VALUE_WEIGHT,
    epoch: int = 0,
) -> dict:
    """Train for one epoch. Returns average metrics."""
    # Update dataset epoch for augmentation
    if hasattr(dataloader.dataset, 'epoch'):
        dataloader.dataset.epoch = epoch
    model.train()
    total_metrics = {}
    n_batches = 0

    for token_ids, policy_targets, value_targets in dataloader:
        token_ids = token_ids.to(device)
        policy_targets = policy_targets.to(device)
        value_targets = value_targets.to(device)

        optimizer.zero_grad()
        loss, metrics = compute_loss(
            model, token_ids, policy_targets, value_targets, value_weight
        )
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


def expert_iteration(
    model: GeoTransformer,
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
    train_summarizer_flag: bool = False,
    summarizer_epochs: int = 10,
) -> GeoTransformer:
    """Run expert iteration training loop.

    0. (Optional) Train Summarizer on proof traces
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

    if resume_from and os.path.exists(resume_from):
        start_iter = load_checkpoint(model, optimizer, resume_from, device)
        print(f"Resumed from iteration {start_iter}")

    # Phase 0: Summarizer pre-training (optional)
    summarizer = None
    if train_summarizer_flag:
        print("=" * 60)
        print("Phase 0: Summarizer pre-training")
        print("=" * 60)
        summarizer_path = os.path.join(checkpoint_dir, "summarizer.pt")
        summarizer = FactSummarizer().to(device)
        print(f"FactSummarizer parameters: {count_summarizer_parameters(summarizer):,}")

        if os.path.exists(summarizer_path):
            load_summarizer_checkpoint(summarizer, summarizer_path, device)
            print(f"  Loaded existing Summarizer from {summarizer_path}")
        else:
            print("  Generating Summarizer training data from proof traces...")
            summ_samples = generate_summarizer_data(problems_file)
            if summ_samples:
                summarizer = train_summarizer(
                    summarizer, summ_samples,
                    num_epochs=summarizer_epochs,
                    batch_size=256,
                    device=device,
                    checkpoint_dir=checkpoint_dir,
                )
            else:
                print("  No Summarizer data, skipping")
                summarizer = None

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
        dataset = SyntheticDataset(synthetic_data, max_seq_len=max_seq_len, augment=augment)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        for epoch in range(epochs_per_iter * 2):  # more epochs for synthetic
            metrics = train_epoch(model, loader, optimizer, device, value_weight, epoch=epoch)
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
        dataset = TextGeometryDataset(supervised_samples, max_seq_len=max_seq_len, augment=augment)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        for epoch in range(epochs_per_iter):
            metrics = train_epoch(model, loader, optimizer, device, value_weight, epoch=epoch)
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
        use_summarizer=summarizer is not None,
    )

    for iteration in range(start_iter, num_iterations):
        print(f"\n--- Iteration {iteration+1}/{num_iterations} ---")
        t0 = time.time()

        print("  Self-play...")
        new_samples = self_play_episode(problems, model, mcts_config, device, summarizer)
        replay.add(new_samples)
        print(f"  Collected {len(new_samples)} samples (buffer: {len(replay)})")

        if len(replay) >= batch_size:
            train_samples = replay.all()
            dataset = TextGeometryDataset(train_samples, max_seq_len=max_seq_len, augment=augment)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            for epoch in range(epochs_per_iter):
                metrics = train_epoch(model, loader, optimizer, device, value_weight, epoch=epoch)
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


def main():
    parser = argparse.ArgumentParser(description="Train GeoTransformer")
    parser.add_argument("--problems", default=DEFAULT_PROBLEMS_FILE, help="Problem file path")
    parser.add_argument("--iterations", type=int, default=DEFAULT_NUM_ITERATIONS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS_PER_ITER)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--supervised-only", action="store_true",
                        help="Only run supervised pre-training, no self-play")
    parser.add_argument("--synthetic-size", type=int, default=DEFAULT_SYNTHETIC_SIZE)
    parser.add_argument("--synthetic-seed", type=int, default=DEFAULT_SYNTHETIC_SEED)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable data augmentation (label permutation + fact shuffling)")
    parser.add_argument("--train-summarizer", action="store_true",
                        help="Train FactSummarizer on proof traces (Phase 0)")
    parser.add_argument("--summarizer-epochs", type=int, default=10,
                        help="Number of epochs for Summarizer training")
    parser.add_argument("--summarizer-only", action="store_true",
                        help="Only train the Summarizer, skip GeoTransformer training")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"Device: {device}")

    # Summarizer-only mode: train just the Summarizer and exit
    if args.summarizer_only:
        summarizer = FactSummarizer().to(device)
        print(f"FactSummarizer parameters: {count_summarizer_parameters(summarizer):,}")
        print("Generating Summarizer training data from proof traces...")
        summ_samples = generate_summarizer_data(args.problems)
        if summ_samples:
            train_summarizer(
                summarizer, summ_samples,
                num_epochs=args.summarizer_epochs,
                batch_size=args.batch_size,
                device=device,
                checkpoint_dir=args.checkpoint_dir,
            )
        else:
            print("No Summarizer data available")
        return

    model = GeoTransformer().to(device)
    print(f"GeoTransformer parameters: {count_parameters(model):,}")

    if args.supervised_only:
        # Generate synthetic + supervised data and train
        print("Generating synthetic data...")
        t0 = time.time()
        synthetic_data = []
        generated = 0
        batch_gen_size = min(args.synthetic_size, 2000)
        while generated < args.synthetic_size:
            batch = min(batch_gen_size, args.synthetic_size - generated)
            chunk = geoprover.generate_synthetic_data(batch, args.synthetic_seed + generated)
            synthetic_data.extend(chunk)
            generated += batch
            elapsed = time.time() - t0
            rate = len(synthetic_data) / elapsed if elapsed > 0 else 0
            print(f"  {len(synthetic_data)}/{args.synthetic_size} ({rate:.0f}/s, {elapsed:.0f}s)")
        print(f"Generated {len(synthetic_data)} synthetic examples in {time.time()-t0:.1f}s")

        num_synthetic_epochs = args.epochs * 3
        num_supervised_epochs = args.epochs * 5
        total_epochs = num_synthetic_epochs + num_supervised_epochs
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=DEFAULT_WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=args.lr * 0.01,
        )

        augment = not args.no_augment
        if synthetic_data:
            dataset = SyntheticDataset(synthetic_data, max_seq_len=args.max_seq_len, augment=augment)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
            for epoch in range(num_synthetic_epochs):
                metrics = train_epoch(model, loader, optimizer, device, epoch=epoch)
                scheduler.step()
                print(f"Synthetic epoch {epoch+1}/{num_synthetic_epochs}: "
                      f"loss={metrics['loss']:.4f} "
                      f"policy={metrics['policy_loss']:.4f} "
                      f"value={metrics['value_loss']:.4f}")
            save_checkpoint(
                model, optimizer, 0, metrics,
                os.path.join(args.checkpoint_dir, "synthetic.pt"),
            )

        samples = generate_supervised_data(
            args.problems, model=model, max_seq_len=args.max_seq_len, device=device,
        )
        if samples:
            dataset = TextGeometryDataset(samples, max_seq_len=args.max_seq_len, augment=augment)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
            for epoch in range(num_supervised_epochs):
                metrics = train_epoch(model, loader, optimizer, device, epoch=epoch)
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
            batch_size=args.batch_size,
            lr=args.lr,
            checkpoint_dir=args.checkpoint_dir,
            device=device,
            resume_from=args.resume,
            synthetic_size=args.synthetic_size,
            synthetic_seed=args.synthetic_seed,
            max_seq_len=args.max_seq_len,
            augment=not args.no_augment,
            train_summarizer_flag=args.train_summarizer,
            summarizer_epochs=args.summarizer_epochs,
        )


if __name__ == "__main__":
    main()
