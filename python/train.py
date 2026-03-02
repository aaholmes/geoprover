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

    def __init__(self, samples: list[TrainingSample], max_seq_len: int = DEFAULT_MAX_SEQ_LEN):
        self.samples = samples
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        token_ids = tokenize_and_pad(s.state_text, max_len=self.max_seq_len)
        policy = torch.tensor(s.policy_target, dtype=torch.float32)
        value = torch.tensor(s.value_target, dtype=torch.float32)
        return token_ids, policy, value


class SyntheticDataset(Dataset):
    """Dataset from Rust-generated synthetic (state_text, construction_text, goal_text) tuples.

    Handles both positive examples (construction makes goal provable, value=1.0)
    and negative examples (construction_text prefixed with "NEG:", value=0.0).
    """

    def __init__(self, examples: list[tuple[str, str, str]], max_seq_len: int = DEFAULT_MAX_SEQ_LEN):
        self.examples = examples
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        state_text, construction_text, goal_text = self.examples[idx]

        # Detect negative examples (prefixed with "NEG:")
        is_negative = construction_text.startswith("NEG:")
        clean_construction = construction_text[4:] if is_negative else construction_text

        # State + goal as input: "coll a b c ; cong a b c d ; ? perp a h b c"
        input_text = f"{state_text} ; ? {goal_text}" if state_text else f"? {goal_text}"
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
                        for c in constructions:
                            idx = construction_to_index(c.construction_type(), c.args())
                            policy_target[idx] = weight
                    else:
                        policy_target = [0.0] * POLICY_SIZE
                    samples.append(TrainingSample(
                        state_text=state_text,
                        policy_target=policy_target,
                        value_target=1.0,
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
) -> dict:
    """Train for one epoch. Returns average metrics."""
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
) -> GeoTransformer:
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
        dataset = SyntheticDataset(synthetic_data, max_seq_len=max_seq_len)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        for epoch in range(epochs_per_iter * 2):  # more epochs for synthetic
            metrics = train_epoch(model, loader, optimizer, device, value_weight)
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
        dataset = TextGeometryDataset(supervised_samples, max_seq_len=max_seq_len)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        for epoch in range(epochs_per_iter):
            metrics = train_epoch(model, loader, optimizer, device, value_weight)
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
            dataset = TextGeometryDataset(train_samples, max_seq_len=max_seq_len)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            for epoch in range(epochs_per_iter):
                metrics = train_epoch(model, loader, optimizer, device, value_weight)
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
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"Device: {device}")

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

        if synthetic_data:
            dataset = SyntheticDataset(synthetic_data, max_seq_len=args.max_seq_len)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
            for epoch in range(num_synthetic_epochs):
                metrics = train_epoch(model, loader, optimizer, device)
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
            dataset = TextGeometryDataset(samples, max_seq_len=args.max_seq_len)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
            for epoch in range(num_supervised_epochs):
                metrics = train_epoch(model, loader, optimizer, device)
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
        )


if __name__ == "__main__":
    main()
