"""Training loop for GeoNet: supervised pre-training + self-play expert iteration.

Phases:
  A. Supervised pre-training on deduction-solvable problems
  B. Self-play expert iteration: MCTS with NN → collect data → train → repeat

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
    build_valid_mask,
    construction_to_index,
    count_parameters,
    tensor_from_flat,
)
from orchestrate import (
    MctsConfig,
    TrainingSample,
    load_problems,
    self_play_episode,
    solve_problem,
)

# Training defaults
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS_PER_ITER = 5
DEFAULT_REPLAY_SIZE = 50000
DEFAULT_VALUE_WEIGHT = 1.0
DEFAULT_NUM_ITERATIONS = 20
DEFAULT_PROBLEMS_FILE = "problems/jgex_ag_231.txt"
DEFAULT_CHECKPOINT_DIR = "checkpoints"


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


class GeometryDataset(Dataset):
    """PyTorch dataset from training samples."""

    def __init__(self, samples: list[TrainingSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        tensor = torch.tensor(s.tensor, dtype=torch.float32).view(20, 32, 32)
        policy = torch.tensor(s.policy_target, dtype=torch.float32)
        value = torch.tensor(s.value_target, dtype=torch.float32)
        delta_d = torch.tensor(s.delta_d, dtype=torch.float32)
        return tensor, policy, value, delta_d


def compute_loss(
    model: GeoNet,
    tensors: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
    delta_d: torch.Tensor,
    value_weight: float = DEFAULT_VALUE_WEIGHT,
) -> tuple[torch.Tensor, dict]:
    """Compute combined policy + value loss.

    Policy loss: KL divergence between predicted policy and target distribution.
    Value loss: MSE between predicted combined value and target value.
    """
    v_logit, k, policy_logits = model(tensors)

    # Policy loss: KL(target || predicted)
    # Only compute over positions that have nonzero target probability
    log_probs = F.log_softmax(policy_logits, dim=1)
    # Mask: only where target > 0 to avoid log(0) in target
    policy_mask = policy_targets > 0
    if policy_mask.any():
        policy_loss = F.kl_div(
            log_probs, policy_targets, reduction="batchmean", log_target=False
        )
    else:
        policy_loss = torch.tensor(0.0, device=tensors.device)

    # Value loss: MSE between tanh(v_logit + k * delta_d) and target
    predicted_value = torch.tanh(v_logit + k * delta_d)
    value_loss = F.mse_loss(predicted_value, value_targets)

    total_loss = policy_loss + value_weight * value_loss

    metrics = {
        "loss": total_loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "mean_v_logit": v_logit.mean().item(),
        "mean_k": k.mean().item(),
        "mean_pred_value": predicted_value.mean().item(),
    }
    return total_loss, metrics


def generate_supervised_data(
    problems_file: str,
    max_problems: int | None = None,
) -> list[TrainingSample]:
    """Generate supervised training data from deduction-solvable problems.

    For each problem solvable by deduction alone:
      - Encode the initial state (before saturation) as the tensor
      - Policy target: uniform over all candidate constructions (deduction solved it,
        so any construction is fine — teaches exploration)
      - Value target: 1.0 (provable)
    """
    problems = load_problems(problems_file)
    if max_problems:
        problems = problems[:max_problems]

    samples = []
    solved = 0

    print(f"Generating supervised data from {len(problems)} problems...")
    for name, definition in problems:
        problem_text = f"{name}\n{definition}"
        try:
            state = geoprover.parse_problem(problem_text)

            # Encode state BEFORE saturation (this is what the NN sees at decision time)
            flat = geoprover.encode_state(state)
            delta_d_before = geoprover.compute_delta_d(state)

            # Check if deduction solves it
            proved = geoprover.saturate(state)
            if not proved:
                continue

            solved += 1
            delta_d_after = geoprover.compute_delta_d(state)

            # Create a sample. Policy target: uniform (no construction needed).
            # Value target: 1.0 (provable).
            policy_target = [0.0] * POLICY_SIZE
            # Don't set any policy slots — this teaches the value head that
            # deduction-solvable states have high value regardless of construction
            samples.append(TrainingSample(
                tensor=flat,
                policy_target=policy_target,
                value_target=1.0,
                delta_d=delta_d_before,
            ))

            # Also create a negative sample from an unsolvable state?
            # Not needed — the MCTS samples will provide negative examples.

        except Exception as e:
            print(f"  Skip {name}: {e}")

    print(f"  Generated {len(samples)} samples from {solved}/{len(problems)} solved problems")
    return samples


def train_epoch(
    model: GeoNet,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    value_weight: float = DEFAULT_VALUE_WEIGHT,
) -> dict:
    """Train for one epoch. Returns average metrics."""
    model.train()
    total_metrics = {}
    n_batches = 0

    for tensors, policy_targets, value_targets, delta_d in dataloader:
        tensors = tensors.to(device)
        policy_targets = policy_targets.to(device)
        value_targets = value_targets.to(device)
        delta_d = delta_d.to(device)

        optimizer.zero_grad()
        loss, metrics = compute_loss(
            model, tensors, policy_targets, value_targets, delta_d, value_weight
        )
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v
        n_batches += 1

    # Average
    if n_batches > 0:
        for k in total_metrics:
            total_metrics[k] /= n_batches
    return total_metrics


def save_checkpoint(
    model: GeoNet,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    metrics: dict,
    path: str,
) -> None:
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
        "metrics": metrics,
    }, path)
    print(f"  Saved checkpoint: {path}")


def load_checkpoint(
    model: GeoNet,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: str,
) -> int:
    """Load model checkpoint. Returns the iteration number."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("iteration", 0)


def expert_iteration(
    model: GeoNet,
    problems_file: str = DEFAULT_PROBLEMS_FILE,
    num_iterations: int = DEFAULT_NUM_ITERATIONS,
    epochs_per_iter: int = DEFAULT_EPOCHS_PER_ITER,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    value_weight: float = DEFAULT_VALUE_WEIGHT,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    device: str = "cpu",
    resume_from: str | None = None,
) -> GeoNet:
    """Run expert iteration training loop.

    1. Generate supervised pre-training data
    2. Pre-train model
    3. For each iteration:
       a. Self-play: run MCTS on problems with current model
       b. Add samples to replay buffer
       c. Train on replay buffer
       d. Evaluate and checkpoint
    """
    problems = load_problems(problems_file)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=DEFAULT_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_iterations * epochs_per_iter
    )
    replay = ReplayBuffer()
    start_iter = 0

    if resume_from and os.path.exists(resume_from):
        start_iter = load_checkpoint(model, optimizer, resume_from, device)
        print(f"Resumed from iteration {start_iter}")

    # Phase A: supervised pre-training
    print("=" * 60)
    print("Phase A: Supervised pre-training")
    print("=" * 60)
    supervised_samples = generate_supervised_data(problems_file)
    if supervised_samples:
        replay.add(supervised_samples)
        dataset = GeometryDataset(supervised_samples)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs_per_iter):
            metrics = train_epoch(model, loader, optimizer, device, value_weight)
            scheduler.step()
            print(f"  Pre-train epoch {epoch+1}: "
                  f"loss={metrics['loss']:.4f} "
                  f"policy={metrics['policy_loss']:.4f} "
                  f"value={metrics['value_loss']:.4f}")
        save_checkpoint(
            model, optimizer, 0, metrics,
            os.path.join(checkpoint_dir, "pretrained.pt"),
        )

    # Phase B: expert iteration
    print("=" * 60)
    print("Phase B: Expert iteration")
    print("=" * 60)
    mcts_config = MctsConfig(
        num_iterations=100,  # fewer iterations per problem for speed
        max_children=30,
        max_depth=3,
    )

    for iteration in range(start_iter, num_iterations):
        print(f"\n--- Iteration {iteration+1}/{num_iterations} ---")
        t0 = time.time()

        # Self-play: collect training data
        print("  Self-play...")
        new_samples = self_play_episode(problems, model, mcts_config, device)
        replay.add(new_samples)
        print(f"  Collected {len(new_samples)} samples "
              f"(buffer: {len(replay)})")

        # Train on replay buffer
        if len(replay) >= batch_size:
            train_samples = replay.all()
            dataset = GeometryDataset(train_samples)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for epoch in range(epochs_per_iter):
                metrics = train_epoch(model, loader, optimizer, device, value_weight)
                scheduler.step()
            print(f"  Train: loss={metrics['loss']:.4f} "
                  f"policy={metrics['policy_loss']:.4f} "
                  f"value={metrics['value_loss']:.4f} "
                  f"mean_v={metrics['mean_pred_value']:.3f}")

        # Save checkpoint
        save_checkpoint(
            model, optimizer, iteration + 1, metrics,
            os.path.join(checkpoint_dir, f"iter_{iteration+1:03d}.pt"),
        )
        elapsed = time.time() - t0
        print(f"  Iteration time: {elapsed:.1f}s")

    print("\nTraining complete!")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train GeoNet")
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
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"Device: {device}")

    model = GeoNet().to(device)
    print(f"GeoNet parameters: {count_parameters(model):,}")

    if args.supervised_only:
        samples = generate_supervised_data(args.problems)
        if samples:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            dataset = GeometryDataset(samples)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
            for epoch in range(args.epochs * 5):  # more epochs for supervised-only
                metrics = train_epoch(model, loader, optimizer, device)
                print(f"Epoch {epoch+1}: loss={metrics['loss']:.4f}")
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
        )


if __name__ == "__main__":
    main()
