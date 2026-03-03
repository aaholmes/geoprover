# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Geoprover is a neurosymbolic geometry theorem prover using a three-tier MCTS architecture adapted from the Caissawary chess engine. It targets Euclidean geometry (triangle geometry MVP) and benchmarks against AlphaGeometry's JGEX-AG-231 and IMO-AG-30 problem sets.

## Build & Development Commands

```bash
maturin develop          # Build Rust extension, importable as `import geoprover` in Python
cargo test               # Run Rust unit tests (fast only, ~10s)
cargo clippy             # Lint Rust code
cargo fmt                # Format Rust code
```

## Testing

**Always run the fast suite after every change:**
```bash
./scripts/test-fast.sh   # Fast suite (~20s): cargo test + Python tests
```

Slow tests are tagged `#[ignore]` and excluded from `cargo test` by default. To run them:
```bash
cargo test -- --ignored          # Only slow benchmarks (~12 min)
cargo test -- --include-ignored  # Everything
./scripts/test-extended.sh       # Full suite + clippy
```

A background watchdog auto-runs the extended suite after 1 hour idle:
```bash
./scripts/test-watchdog.sh &     # Start idle auto-runner
```

## Architecture

Hybrid Rust/Python via PyO3 (in-process, no subprocess overhead):

```
Python (orchestration, NN training, evaluation)
  │  PyO3 calls
  ▼
Rust extension module (MCTS, deduction engine, state encoding)
```

**Three-tier search (adapted from chess engine):**

| Tier | Role | Geometry equivalent | Chess equivalent |
|------|------|-------------------|-----------------|
| 1 | Symbolic deduction | `DeductionEngine::saturate()` — 21 rules to fixed point | `mate_search()` |
| 2 | MCTS tree search | Search over 16 auxiliary construction types (~30-50 candidates/step) | Tactical MCTS |
| 3 | Neural oracle | SE-ResNet (~2M params), dual-head: policy + value | OracleNet |

**Key insight:** `saturate()` runs exhaustive deduction before every MCTS expansion. The NN only suggests *auxiliary constructions* — the creative step deduction can't do.

## Planned Source Layout

- `src/proof_state.rs` — ProofState, GeoObject, Relation (objects have u16 IDs, facts stored as relations, XOR Zobrist hashing)
- `src/deduction.rs` — 21 forward-chaining rules: angle(8), congruence(4), length(4), circle(3), parallel/perp(2)
- `src/construction.rs` — 16 construction types (Midpoint, AngleBisector, Circumcenter, etc.), priority: GoalRelevant > RecentlyActive > Exploratory
- `src/mcts/` — MctsNode, MCTS loop, UCB/PUCT selection (ported from caissawary)
- `src/encoding.rs` — `state_to_tensor()` producing C×32×32 relation adjacency grid (cap 32 objects, ~12 relation channels + 8 object-type features)
- `src/parser.rs` — Parse AlphaGeometry's JGEX DSL format
- `python/model.py` — GeoNet: SE-ResNet backbone, policy head (~512 max constructions), value head with proof-distance heuristic
- `python/train.py` — Replay buffer, KL-div policy loss + MSE value loss
- `python/orchestrate.py` — Self-play training loop (expert iteration)
- `python/evaluate.py` — Benchmark on JGEX-AG-231 and IMO-AG-30

## Ported Patterns from Caissawary

Reference files are in `../hybrid-chess-engine/`. These are architectural patterns, not importable modules.

### Backpropagation (single-player simplification)

Chess alternates sign during backprop (`reward = -value` at each level). Geometry is single-player — value is always "progress toward proof" from the same perspective. Backprop simplifies to `total_value += value` at every ancestor. No sign flip.

### evaluate_leaf_node Three-Tier Cascade

```
1. Check cached terminal_value → return if set (Tier 1 cache)
2. Run saturate(). If goal ∈ facts → terminal, value = 1.0
3. If no NN value yet:
   a. Batch predict via inference server
   b. Get (value, policy) from NN — value = sigmoid(v_logit) ∈ [0, 1]
   c. Classical fallback (no NN): value = tanh(0.5 * delta_D) (Rust-only MCTS)
4. Return nn_value
```

### UCB/PUCT Selection

```
Q = child.total_value / child.visits  (0.0 if unvisited)
U = c_puct * prior_prob * sqrt(parent_visits) / (1 + child_visits)
score = Q + U
```

Q is always positive (0.0 to 1.0, progress toward proof). `prior_prob` from NN policy head. Two-phase selection: (1) visit unexplored priority constructions first (GoalRelevant → RecentlyActive → Exploratory), (2) then PUCT with NN priors. At root: force every child to get at least 1 visit before PUCT.

### MctsNode Key Fields

- `state: ProofState`, `action: Option<Construction>`
- `visits: u32`, `total_value: f64`
- `terminal_value: Option<f64>` — 1.0 = proved, 0.0 = impossible
- `nn_value: Option<f64>`, `v_logit: Option<f64>`, `k_val: f32` (Rust-side MCTS, kept for classical fallback)
- `children: Vec<Rc<RefCell<MctsNode>>>`, `parent: Option<Weak<...>>`
- `priority_constructions` / `priority_explored` (GoalRelevant first)
- `move_priorities: HashMap<Construction, f64>` — NN policy priors

### Value Function

Python NN: `V = sigmoid(v_logit)` — single scalar output in [0, 1]. Rust-only MCTS classical fallback still uses `V = tanh(0.5 * delta_D)`.

### Inference Server

Batched async predictions: caller sends `predict_async(state)` → gets a `Receiver`. Server collects until batch_size or timeout, sends batch to GPU, distributes results via channels. Same pattern, different tensor shapes: `encode_state()` → `Vec<f32>` → batch → NN → (value, policy).

## Implementation Phases

1. **Rust core** — ProofState, DeductionEngine, ConstructionGen, unit tests → deduction solves Level 1
2. **MCTS port** — MctsNode, search loop, selection with construction categories → random MCTS solves Level 2
3. **PyO3 bridge** — Maturin build, expose search/saturate/encode_state/gen_constructions
4. **NN + training** — GeoNet, tensor encoding, synthetic data from AlphaGeometry, self-play
5. **Evaluation** — Ablation study, proof quality metrics

## Development Process

- **Always use TDD**: write failing tests before implementation. Red → Green → Refactor.
- Run `./scripts/test-fast.sh` after every change (or at minimum `cargo test`).

## Key References

- Architecture adapted from Caissawary chess engine (separate repo, patterns ported not imported)
- Problem format: AlphaGeometry's JGEX DSL (construction steps → goal predicate)
- Training data: AlphaGeometry's synthetic data generator for Phase A (supervised), MCTS self-improvement for Phase B
