"""Evaluation suite: benchmark GeoTransformer on JGEX-AG-231 and IMO-AG-30.

Compares three modes:
  1. Deduction only (no MCTS, no NN)
  2. MCTS with NN policy/value (full system)

Reports solve rates, proof lengths, and timing.
"""

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass

import torch

import geoprover
from model import GeoNet, GeoTransformer, SetGeoTransformer, SetGeoTransformerV2, count_parameters, create_model
from orchestrate import MctsConfig, SearchResult, load_problems, solve_problem
from train import load_checkpoint


@dataclass
class ProblemResult:
    """Result for a single problem."""
    name: str
    solved: bool
    mode: str  # "deduction", "mcts_nn"
    elapsed_ms: float
    num_facts: int
    num_objects: int
    proof_length: int
    value: float


@dataclass
class BenchmarkResult:
    """Aggregate benchmark results."""
    mode: str
    total: int
    solved: int
    solve_rate: float
    mean_time_ms: float
    median_time_ms: float
    total_time_s: float


def evaluate_deduction(problems: list[tuple[str, str]]) -> list[ProblemResult]:
    """Evaluate deduction-only solving."""
    results = []
    for i, (name, definition) in enumerate(problems):
        problem_text = f"{name}\n{definition}"
        t0 = time.time()
        try:
            state = geoprover.parse_problem(problem_text)
            proved = geoprover.saturate(state)
            elapsed = (time.time() - t0) * 1000
            results.append(ProblemResult(
                name=name, solved=proved, mode="deduction",
                elapsed_ms=elapsed, num_facts=state.num_facts(),
                num_objects=state.num_objects(), proof_length=0,
                value=1.0 if proved else 0.0,
            ))
        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            proved = False
            results.append(ProblemResult(
                name=name, solved=False, mode="deduction",
                elapsed_ms=elapsed, num_facts=0, num_objects=0,
                proof_length=0, value=0.0,
            ))
        status = "SOLVED" if proved else "failed"
        n_solved = sum(1 for r in results if r.solved)
        print(f"  [{i+1}/{len(problems)}] {name}: {status} ({elapsed:.0f}ms) [{n_solved}/{i+1} so far]", flush=True)
    return results


def evaluate_mcts_nn(
    problems: list[tuple[str, str]],
    model: GeoTransformer,
    config: MctsConfig,
    device: str = "cpu",
) -> list[ProblemResult]:
    """Evaluate NN-guided MCTS solving."""
    results = []
    for i, (name, definition) in enumerate(problems):
        problem_text = f"{name}\n{definition}"
        t0 = time.time()
        try:
            result = solve_problem(problem_text, model, config, device)
            elapsed = (time.time() - t0) * 1000
            state = geoprover.parse_problem(problem_text)
            geoprover.saturate(state)
            solved = result.solved
            results.append(ProblemResult(
                name=name, solved=solved, mode="mcts_nn",
                elapsed_ms=elapsed, num_facts=state.num_facts(),
                num_objects=state.num_objects(),
                proof_length=len(result.actions),
                value=result.best_value,
            ))
        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            solved = False
            results.append(ProblemResult(
                name=name, solved=False, mode="mcts_nn",
                elapsed_ms=elapsed, num_facts=0, num_objects=0,
                proof_length=0, value=0.0,
            ))
        status = "SOLVED" if solved else "failed"
        n_solved = sum(1 for r in results if r.solved)
        print(f"  [{i+1}/{len(problems)}] {name}: {status} ({elapsed:.0f}ms) [{n_solved}/{i+1} so far]", flush=True)
    return results


def summarize(results: list[ProblemResult], mode: str) -> BenchmarkResult:
    total = len(results)
    solved = sum(1 for r in results if r.solved)
    times = [r.elapsed_ms for r in results]
    times_sorted = sorted(times)
    mean_time = sum(times) / total if total else 0
    median_time = times_sorted[total // 2] if total else 0
    return BenchmarkResult(
        mode=mode, total=total, solved=solved,
        solve_rate=solved / total if total else 0,
        mean_time_ms=mean_time, median_time_ms=median_time,
        total_time_s=sum(times) / 1000,
    )


def print_summary(bench: BenchmarkResult) -> None:
    print(f"\n{'=' * 50}")
    print(f"  Mode: {bench.mode}")
    print(f"  Solved: {bench.solved}/{bench.total} ({bench.solve_rate:.1%})")
    print(f"  Mean time: {bench.mean_time_ms:.1f}ms")
    print(f"  Median time: {bench.median_time_ms:.1f}ms")
    print(f"  Total time: {bench.total_time_s:.1f}s")
    print(f"{'=' * 50}")


def compare_results(
    deduction: list[ProblemResult],
    mcts_nn: list[ProblemResult],
) -> None:
    deduction_solved = {r.name for r in deduction if r.solved}
    mcts_solved = {r.name for r in mcts_nn if r.solved}

    mcts_only = mcts_solved - deduction_solved
    deduction_only = deduction_solved - mcts_solved

    print(f"\nSolved by deduction only: {len(deduction_only)}")
    print(f"Solved by MCTS+NN only:   {len(mcts_only)}")
    print(f"Solved by both:           {len(deduction_solved & mcts_solved)}")
    print(f"Unsolved by both:         {len(set(r.name for r in deduction) - deduction_solved - mcts_solved)}")

    if mcts_only:
        print("\nProblems solved by MCTS+NN but not deduction:")
        for name in sorted(mcts_only):
            r = next(r for r in mcts_nn if r.name == name)
            print(f"  {name} (steps={r.proof_length}, time={r.elapsed_ms:.0f}ms)")


def save_results(
    results: dict[str, list[ProblemResult]],
    output_path: str,
) -> None:
    data = {}
    for mode, problem_results in results.items():
        data[mode] = [asdict(r) for r in problem_results]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nDetailed results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate GeoTransformer")
    parser.add_argument("--problems", default="problems/jgex_ag_231.txt")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint path")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--mcts-iterations", type=int, default=200)
    parser.add_argument("--max-children", type=int, default=30)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--output", default="results/evaluation.json")
    parser.add_argument("--deduction-only", action="store_true",
                        help="Only run deduction evaluation")
    parser.add_argument("--model-type", default="set_v2",
                        choices=["set", "set_v2", "transformer"],
                        help="Model architecture: 'set' for SetGeoTransformerV1, 'set_v2' for V2, 'transformer' for GeoTransformer")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    problems = load_problems(args.problems)
    print(f"Loaded {len(problems)} problems from {args.problems}")

    all_results = {}

    # 1. Deduction only
    print("\nEvaluating: Deduction only")
    deduction_results = evaluate_deduction(problems)
    all_results["deduction"] = deduction_results
    ded_bench = summarize(deduction_results, "deduction")
    print_summary(ded_bench)

    if args.deduction_only:
        save_results(all_results, args.output)
        return

    # 2. MCTS + NN
    model = create_model(args.model_type).to(device)
    model_name = type(model).__name__
    if args.checkpoint and os.path.exists(args.checkpoint):
        load_checkpoint(model, None, args.checkpoint, device)
        print(f"Loaded model from {args.checkpoint}")
    else:
        print("No checkpoint - using random NN weights (baseline)")

    print(f"{model_name} parameters: {count_parameters(model):,}")

    config = MctsConfig(
        num_iterations=args.mcts_iterations,
        max_children=args.max_children,
        max_depth=args.max_depth,
        model_type=args.model_type,
    )

    print(f"\nEvaluating: MCTS + NN (iterations={config.num_iterations}, "
          f"depth={config.max_depth})")
    mcts_results = evaluate_mcts_nn(problems, model, config, device)
    all_results["mcts_nn"] = mcts_results
    mcts_bench = summarize(mcts_results, "mcts_nn")
    print_summary(mcts_bench)

    compare_results(deduction_results, mcts_results)
    save_results(all_results, args.output)


if __name__ == "__main__":
    main()
