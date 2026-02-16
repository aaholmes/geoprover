"""Orchestration: NN-guided MCTS self-play for geometry theorem proving.

This module wires the GeoTransformer neural network into the MCTS search loop.
Uses text-based state encoding instead of tensor grids.

Flow:
  1. Parse problem -> ProofState
  2. Saturate (deduction)
  3. If not solved, run NN-guided MCTS:
     a. Generate candidate constructions
     b. Encode state as text, score with NN policy head
     c. Expand best candidates
     d. Evaluate with NN value head + delta_D
     e. Backpropagate
  4. Collect (state_text, visit_distribution, outcome) training data
"""

import math
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

import geoprover
from model import (
    POLICY_SIZE,
    GeoNet,
    build_valid_mask,
    construction_to_index,
    tokenize_and_pad,
)


@dataclass
class MctsConfig:
    """Configuration for NN-guided MCTS."""
    num_iterations: int = 200
    max_children: int = 30
    c_puct: float = 1.4
    max_depth: int = 3
    # Deduction config for child nodes
    saturate_max_iterations: int = 50
    saturate_max_facts: int = 5000
    # Sequence length for tokenizer
    max_seq_len: int = 256


@dataclass
class MctsNode:
    """A node in the NN-guided MCTS tree."""
    state: object  # PyProofState
    action: object = None  # PyConstruction that led here
    action_index: int = -1  # Policy index of the action
    parent: object = None  # Parent MctsNode (or None for root)
    children: list = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0
    prior: float = 0.0  # NN policy prior
    terminal_value: float | None = None
    expanded: bool = False
    depth: int = 0


@dataclass
class TrainingSample:
    """A single training sample from self-play."""
    state_text: str  # text encoding of the proof state
    policy_target: list[float]  # POLICY_SIZE-element distribution
    value_target: float  # outcome: 1.0 if proved, 0.0 otherwise
    delta_d: float  # proof-distance at this state


@dataclass
class SearchResult:
    """Result of an MCTS search."""
    solved: bool
    best_value: float
    root_visits: int
    samples: list[TrainingSample]
    actions: list[str]  # construction descriptions for proof trace
    elapsed_ms: float


def _q_value(node: MctsNode) -> float:
    if node.visits == 0:
        return 0.0
    return node.total_value / node.visits


def _ucb_score(child: MctsNode, parent_visits: int, c_puct: float) -> float:
    q = _q_value(child)
    u = c_puct * child.prior * math.sqrt(parent_visits) / (1 + child.visits)
    return q + u


def _select_leaf(root: MctsNode, c_puct: float) -> MctsNode:
    node = root
    while node.expanded and node.children:
        unvisited = [c for c in node.children if c.visits == 0]
        if unvisited:
            node = unvisited[0]
            continue
        best_score = float("-inf")
        best_child = node.children[0]
        for child in node.children:
            score = _ucb_score(child, node.visits, c_puct)
            if score > best_score:
                best_score = score
                best_child = child
        node = best_child
    return node


def _expand(node: MctsNode, config: MctsConfig, model: GeoNet, device: str) -> None:
    """Expand a leaf node: generate children with NN priors."""
    if node.depth >= config.max_depth:
        return

    constructions = geoprover.generate_constructions(node.state)
    if not constructions:
        return

    constructions = constructions[:config.max_children]

    # Encode state as text and tokenize
    state_text = geoprover.state_to_text(node.state)
    token_ids = tokenize_and_pad(state_text, max_len=config.max_seq_len).to(device)
    mask = build_valid_mask(constructions, device=device).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        _, _, logits = model(token_ids.unsqueeze(0), mask)
        priors = F.softmax(logits[0], dim=0)

    for c in constructions:
        idx = construction_to_index(c.construction_type(), c.args())
        child_state = geoprover.apply_construction(node.state, c)
        child = MctsNode(
            state=child_state,
            action=c,
            action_index=idx,
            parent=node,
            prior=priors[idx].item(),
            depth=node.depth + 1,
        )
        node.children.append(child)

    node.expanded = True


def _evaluate(node: MctsNode, model: GeoNet, config: MctsConfig, device: str) -> float:
    """Evaluate a node: run deduction, then NN value if not terminal."""
    proved = geoprover.saturate_with_config(
        node.state,
        max_iterations=config.saturate_max_iterations,
        max_facts=config.saturate_max_facts,
    )

    if proved:
        node.terminal_value = 1.0
        return 1.0

    delta_d = geoprover.compute_delta_d(node.state)

    # Encode state as text
    state_text = geoprover.state_to_text(node.state)
    token_ids = tokenize_and_pad(state_text, max_len=config.max_seq_len).to(device)

    model.eval()
    with torch.no_grad():
        v_logit, k, _ = model(token_ids.unsqueeze(0))
        value = math.tanh(v_logit.item() + k.item() * delta_d)

    return max(0.0, min(1.0, value))


def _backprop(node: MctsNode, value: float) -> None:
    current = node
    while current is not None:
        current.visits += 1
        current.total_value += value
        current = current.parent


def mcts_search(
    state: object,
    model: GeoNet,
    config: MctsConfig | None = None,
    device: str = "cpu",
) -> SearchResult:
    """Run NN-guided MCTS on a proof state."""
    if config is None:
        config = MctsConfig()

    start = time.time()
    samples: list[TrainingSample] = []

    root = MctsNode(state=state)
    root_value = _evaluate(root, model, config, device)

    if root.terminal_value == 1.0:
        elapsed = (time.time() - start) * 1000
        return SearchResult(
            solved=True, best_value=1.0, root_visits=1,
            samples=[], actions=[], elapsed_ms=elapsed,
        )

    _expand(root, config, model, device)

    for _ in range(config.num_iterations):
        leaf = _select_leaf(root, config.c_puct)

        if leaf.terminal_value is not None:
            _backprop(leaf, leaf.terminal_value)
            if leaf.terminal_value == 1.0:
                break
            continue

        value = _evaluate(leaf, model, config, device)

        if leaf.terminal_value == 1.0:
            _backprop(leaf, 1.0)
            break

        if leaf.depth < config.max_depth:
            _expand(leaf, config, model, device)

        _backprop(leaf, value)

    solved = _is_solved(root)

    # Collect training sample from root
    if root.children:
        visit_counts = [c.visits for c in root.children]
        total = sum(visit_counts)
        if total > 0:
            policy_target = [0.0] * POLICY_SIZE
            for child, vc in zip(root.children, visit_counts):
                policy_target[child.action_index] = vc / total

            root_text = geoprover.state_to_text(root.state)
            root_delta = geoprover.compute_delta_d(root.state)
            samples.append(TrainingSample(
                state_text=root_text,
                policy_target=policy_target,
                value_target=1.0 if solved else _q_value(root),
                delta_d=root_delta,
            ))

    actions = []
    if solved:
        actions = _extract_proof_path(root)

    elapsed = (time.time() - start) * 1000
    return SearchResult(
        solved=solved, best_value=_q_value(root), root_visits=root.visits,
        samples=samples, actions=actions, elapsed_ms=elapsed,
    )


def _is_solved(node: MctsNode) -> bool:
    if node.terminal_value == 1.0:
        return True
    for child in node.children:
        if _is_solved(child):
            return True
    return False


def _extract_proof_path(node: MctsNode) -> list[str]:
    if node.terminal_value == 1.0:
        return []
    for child in node.children:
        if _is_solved(child):
            desc = repr(child.action) if child.action else "root"
            return [desc] + _extract_proof_path(child)
    return []


def solve_problem(
    problem_text: str,
    model: GeoNet,
    config: MctsConfig | None = None,
    device: str = "cpu",
) -> SearchResult:
    """Parse, saturate, and optionally MCTS-search a problem."""
    state = geoprover.parse_problem(problem_text)

    proved = geoprover.saturate(state)
    if proved:
        return SearchResult(
            solved=True, best_value=1.0, root_visits=0,
            samples=[], actions=["deduction"], elapsed_ms=0.0,
        )

    return mcts_search(state, model, config, device)


def load_problems(path: str) -> list[tuple[str, str]]:
    """Load problems from a JGEX file (alternating name/definition lines)."""
    problems = []
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    for i in range(0, len(lines) - 1, 2):
        name = lines[i]
        definition = lines[i + 1]
        problems.append((name, definition))
    return problems


def self_play_episode(
    problems: list[tuple[str, str]],
    model: GeoNet,
    config: MctsConfig | None = None,
    device: str = "cpu",
) -> list[TrainingSample]:
    """Run one self-play episode over all problems, collecting training data."""
    all_samples = []
    solved_count = 0
    deduction_count = 0

    for name, definition in problems:
        problem_text = f"{name}\n{definition}"
        try:
            result = solve_problem(problem_text, model, config, device)
            if result.solved:
                solved_count += 1
                if result.actions == ["deduction"]:
                    deduction_count += 1
            all_samples.extend(result.samples)
        except Exception as e:
            print(f"  Error on {name}: {e}")

    print(f"  Solved: {solved_count}/{len(problems)} "
          f"(deduction: {deduction_count}, MCTS: {solved_count - deduction_count})")
    return all_samples
