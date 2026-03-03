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
     d. Evaluate with NN value head (sigmoid)
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
    encode_state_as_set,
    split_state_text,
    tokenize_and_pad,
)
from summarizer import FactSummarizer, filter_facts, build_summarized_text


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
    # Summarizer: if set, filter deduced facts before feeding to GeoTransformer
    use_summarizer: bool = False
    summarizer_k: int | None = None  # None = adaptive (|initial| + |constructions| + 1)
    # Model type: "set" for SetGeoTransformer, "transformer" for GeoTransformer
    model_type: str = "transformer"


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
    pre_facts: set = field(default_factory=set)  # Facts before saturation (for Summarizer)


@dataclass
class TrainingSample:
    """A single training sample from self-play."""
    state_text: str  # text encoding of the proof state
    policy_target: list[float]  # POLICY_SIZE-element distribution
    value_target: float  # outcome: 1.0 if proved, 0.0 otherwise
    # Sparse construction info for augmentation: list of (type_name, args, weight)
    policy_constructions: list[tuple[str, list[int], float]] | None = None


@dataclass
class SearchResult:
    """Result of an MCTS search."""
    solved: bool
    best_value: float
    root_visits: int
    samples: list[TrainingSample]
    actions: list[str]  # construction descriptions for proof trace
    elapsed_ms: float


def _get_state_text(
    state: object,
    config: MctsConfig,
    summarizer: FactSummarizer | None,
    pre_facts: set[str] | None,
    num_constructions: int,
    device: str,
) -> str:
    """Get state text, optionally filtering deduced facts through the Summarizer."""
    if summarizer is None or not config.use_summarizer or pre_facts is None:
        return geoprover.state_to_text(state)

    post_facts = set(state.facts_as_text_list())
    initial_facts = sorted(pre_facts)
    deduced_facts = sorted(post_facts - pre_facts)

    if not deduced_facts:
        return geoprover.state_to_text(state)

    goal_text = state.goal_as_text()
    k = config.summarizer_k

    filtered = filter_facts(
        summarizer, initial_facts, deduced_facts, goal_text,
        num_constructions=num_constructions, k=k, device=device,
    )
    return build_summarized_text(initial_facts, filtered, goal_text)


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


def _expand(
    node: MctsNode,
    config: MctsConfig,
    model: GeoNet,
    device: str,
    summarizer: FactSummarizer | None = None,
) -> None:
    """Expand a leaf node: generate children with NN priors."""
    if node.depth >= config.max_depth:
        return

    constructions = geoprover.generate_constructions(node.state)
    if not constructions:
        return

    constructions = constructions[:config.max_children]
    mask = build_valid_mask(constructions, device=device).unsqueeze(0)

    model.eval()
    if config.model_type == "set":
        fact_texts = list(node.state.facts_as_text_list())
        goal_text = node.state.goal_as_text()
        fact_ids, goal_ids, fact_mask = encode_state_as_set(fact_texts, goal_text)
        with torch.no_grad():
            _, logits = model(
                fact_ids.unsqueeze(0).to(device),
                goal_ids.unsqueeze(0).to(device),
                fact_mask.unsqueeze(0).to(device),
                mask,
            )
            priors = F.softmax(logits[0], dim=0)
    else:
        state_text = _get_state_text(node.state, config, summarizer, node.pre_facts, node.depth, device)
        token_ids = tokenize_and_pad(state_text, max_len=config.max_seq_len).to(device)
        with torch.no_grad():
            _, logits = model(token_ids.unsqueeze(0), mask)
            priors = F.softmax(logits[0], dim=0)

    for c in constructions:
        idx = construction_to_index(c.construction_type(), c.args())
        child_state = geoprover.apply_construction(node.state, c)
        # Snapshot pre-saturation facts for the child
        child_pre_facts = set(child_state.facts_as_text_list()) if config.use_summarizer else set()
        child = MctsNode(
            state=child_state,
            action=c,
            action_index=idx,
            parent=node,
            prior=priors[idx].item(),
            depth=node.depth + 1,
            pre_facts=child_pre_facts,
        )
        node.children.append(child)

    node.expanded = True


def _evaluate(
    node: MctsNode,
    model: GeoNet,
    config: MctsConfig,
    device: str,
    summarizer: FactSummarizer | None = None,
) -> float:
    """Evaluate a node: run deduction, then NN value if not terminal."""
    proved = geoprover.saturate_with_config(
        node.state,
        max_iterations=config.saturate_max_iterations,
        max_facts=config.saturate_max_facts,
    )

    if proved:
        node.terminal_value = 1.0
        return 1.0

    model.eval()
    if config.model_type == "set":
        fact_texts = list(node.state.facts_as_text_list())
        goal_text = node.state.goal_as_text()
        fact_ids, goal_ids, fact_mask = encode_state_as_set(fact_texts, goal_text)
        with torch.no_grad():
            value, _ = model(
                fact_ids.unsqueeze(0).to(device),
                goal_ids.unsqueeze(0).to(device),
                fact_mask.unsqueeze(0).to(device),
            )
    else:
        state_text = _get_state_text(node.state, config, summarizer, node.pre_facts, node.depth, device)
        token_ids = tokenize_and_pad(state_text, max_len=config.max_seq_len).to(device)
        with torch.no_grad():
            value, _ = model(token_ids.unsqueeze(0))

    return value.item()


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
    summarizer: FactSummarizer | None = None,
) -> SearchResult:
    """Run NN-guided MCTS on a proof state."""
    if config is None:
        config = MctsConfig()

    start = time.time()
    samples: list[TrainingSample] = []

    # Snapshot pre-saturation facts for root (if Summarizer enabled)
    root_pre_facts = set(state.facts_as_text_list()) if config.use_summarizer else set()
    root = MctsNode(state=state, pre_facts=root_pre_facts)
    root_value = _evaluate(root, model, config, device, summarizer)

    if root.terminal_value == 1.0:
        elapsed = (time.time() - start) * 1000
        return SearchResult(
            solved=True, best_value=1.0, root_visits=1,
            samples=[], actions=[], elapsed_ms=elapsed,
        )

    _expand(root, config, model, device, summarizer)

    for _ in range(config.num_iterations):
        leaf = _select_leaf(root, config.c_puct)

        if leaf.terminal_value is not None:
            _backprop(leaf, leaf.terminal_value)
            if leaf.terminal_value == 1.0:
                break
            continue

        value = _evaluate(leaf, model, config, device, summarizer)

        if leaf.terminal_value == 1.0:
            _backprop(leaf, 1.0)
            break

        if leaf.depth < config.max_depth:
            _expand(leaf, config, model, device, summarizer)

        _backprop(leaf, value)

    solved = _is_solved(root)

    # Collect training samples from ALL nodes with sufficient visits
    samples = _collect_all_node_samples(root, solved, min_visits=5)

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


def _collect_all_node_samples(
    root: MctsNode,
    solved: bool,
    min_visits: int = 10,
) -> list[TrainingSample]:
    """Walk the MCTS tree and collect training samples from all nodes with sufficient visits.

    This extracts signal from every internal node, not just the root,
    providing 10-20x more training data per search.
    """
    samples = []
    stack = [root]
    while stack:
        node = stack.pop()
        if not node.children or node.visits < min_visits:
            continue
        visit_counts = [c.visits for c in node.children]
        total = sum(visit_counts)
        if total == 0:
            continue

        policy_target = [0.0] * POLICY_SIZE
        policy_constructions = []
        for child, vc in zip(node.children, visit_counts):
            policy_target[child.action_index] = vc / total
            if vc > 0 and child.action is not None:
                weight = vc / total
                policy_constructions.append((
                    child.action.construction_type(),
                    child.action.args(),
                    weight,
                ))

        try:
            state_text = geoprover.state_to_text(node.state)
        except Exception:
            continue

        # Value target: 1.0 if this subtree contains a solution, else Q-value
        subtree_solved = _is_solved(node)
        value = 1.0 if subtree_solved else _q_value(node)

        samples.append(TrainingSample(
            state_text=state_text,
            policy_target=policy_target,
            value_target=value,
            policy_constructions=policy_constructions if policy_constructions else None,
        ))

        # Continue walking into children
        for child in node.children:
            if child.visits >= min_visits:
                stack.append(child)

    return samples


def solve_problem(
    problem_text: str,
    model: GeoNet,
    config: MctsConfig | None = None,
    device: str = "cpu",
    summarizer: FactSummarizer | None = None,
) -> SearchResult:
    """Parse, saturate, and optionally MCTS-search a problem."""
    state = geoprover.parse_problem(problem_text)

    proved = geoprover.saturate(state)
    if proved:
        return SearchResult(
            solved=True, best_value=1.0, root_visits=0,
            samples=[], actions=["deduction"], elapsed_ms=0.0,
        )

    return mcts_search(state, model, config, device, summarizer)


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
    summarizer: FactSummarizer | None = None,
) -> list[TrainingSample]:
    """Run one self-play episode over all problems, collecting training data."""
    all_samples = []
    solved_count = 0
    deduction_count = 0
    mcts_count = 0
    t0 = time.time()

    for i, (name, definition) in enumerate(problems):
        problem_text = f"{name}\n{definition}"
        try:
            result = solve_problem(problem_text, model, config, device, summarizer)
            if result.solved:
                solved_count += 1
                if result.actions == ["deduction"]:
                    deduction_count += 1
                else:
                    mcts_count += 1
            all_samples.extend(result.samples)
        except Exception as e:
            print(f"  Error on {name}: {e}")

        # Progress every 50 problems
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"    [{i+1}/{len(problems)}] solved={solved_count} "
                  f"(ded={deduction_count}, mcts={mcts_count}) "
                  f"samples={len(all_samples)} {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"  Solved: {solved_count}/{len(problems)} "
          f"(deduction: {deduction_count}, MCTS: {mcts_count}) "
          f"in {elapsed:.0f}s, {len(all_samples)} samples")
    return all_samples
