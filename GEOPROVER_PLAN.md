# Neurosymbolic Geometry Theorem Prover — Feasibility & Architecture Plan

## Context

The Caissawary chess engine's three-tier MCTS architecture (symbolic gates → tactical heuristics → neural oracle) is domain-agnostic as a *pattern*. The goal is to adapt it for Euclidean geometry theorem proving — specifically triangle geometry as an MVP. This would be a new crate/repo, not a refactor of the chess engine.

## What Transfers from the Chess Engine

| Subsystem | Reusability | Adaptation needed |
|-----------|------------|-------------------|
| MCTS tree search (select/expand/backprop) | ~90% | Replace `Board`/`Move` with `ProofState`/`Construction` |
| Tiered symbolic gates | Pattern transfers perfectly | Replace mate_search with deduction-to-fixed-point |
| Inference server (batched NN) | ~95% | Change tensor shapes |
| Training loop (self-play → train → gate) | ~90% | Replace adversarial self-play with curriculum generation |
| SE-ResNet dual-head NN | Backbone reusable | Change input from 17×8×8 to C×32×32 relation grid |

**Key architectural analog:** `saturate()` (run all deduction rules to fixed point) **is** the geometry equivalent of `mate_search` — provably correct symbolic reasoning that short-circuits the NN. The NN's job is only to suggest *auxiliary constructions* (the creative step that deduction alone can't do).

## Training Data Strategy

Chess uses adversarial self-play. Geometry doesn't have an adversary, but **we don't need to invent problems** — AlphaGeometry's open-source pipeline provides:

1. **Supervised pre-training**: AG's synthetic data generator creates millions of (random_config → exhaustive_deduction → discovered_facts) pairs. We use these to bootstrap the NN: the "policy target" is which construction leads to the proof, the "value target" is whether the goal was proved.
2. **MCTS self-improvement**: Once the NN has a baseline, run MCTS search on JGEX-AG-231 problems. The MCTS visit distribution becomes the policy target (expert iteration), and proof success becomes the value target. This is the same loop as chess self-play, just single-player.

## Architecture (Hybrid Rust/Python via PyO3)

```
Python (orchestration, NN, training)
  │
  │ PyO3 calls (same process, no subprocess overhead)
  ▼
Rust extension module (fast MCTS, deduction engine, encoding)
  ├── ProofState / GeoObject / Relation  (state representation)
  ├── DeductionEngine::saturate()        (Tier 1 — 21 rules, forward chaining)
  ├── ConstructionGen                    (action generation, 16 types)
  ├── MCTS loop                          (ported from tactical_mcts.rs)
  └── state_to_tensor()                  (C×32×32 relation adjacency encoding)
```

### State Representation (Rust)

- **Objects**: Points, Lines, Segments, Circles, Angles, Triangles — each with a unique u16 ID
- **Facts**: Relations between objects (Congruent, Parallel, Perpendicular, Collinear, EqualAngle, EqualLength, Similar, Midpoint, OnCircle, etc.)
- **Goal**: A target `Relation` to prove
- **Hash**: XOR-based incremental hashing on facts (same pattern as Zobrist in `make_move.rs`) for transposition detection

### Action Space

**Tier 1 (deductive, exhaustive, not MCTS actions):** 21 rules applied to fixed point before expansion:
- Angle rules (8): vertical angles, supplementary, triangle sum, isosceles base angles, inscribed angle, alternate/corresponding angles, exterior angle
- Congruence (4): SAS, ASA, SSS, CPCTC
- Length/segment (4): midpoint definition, transitive equality, midpoint theorem, angle bisector property
- Circle (3): equal radii, tangent⊥radius, cyclic opposite angles
- Parallel/perp (2): transitive parallel, perp-to-parallel

**Tier 3 (creative, MCTS search space):** 16 auxiliary construction types:
Midpoint, AngleBisector, PerpendicularBisector, Altitude, ParallelThrough, PerpendicularThrough, Circumcenter, Incenter, Centroid, Orthocenter, CircumscribedCircle, IntersectLines, IntersectLineCircle, ReflectPoint, ExtendSegment, TangentLine

Branching factor: ~30-50 candidates per step (comparable to chess's ~35 legal moves). Categorized as GoalRelevant > RecentlyActive > Exploratory (maps to Check > Capture > Quiet in selection priority).

### NN Architecture (Python)

- **Input**: C×32×32 relation adjacency grid (C≈12 relation channels + 8 object-type features). Cap at 32 objects with zero-padding (triangle problems rarely exceed 15).
- **Backbone**: SE-ResNet (direct port from OracleNet in `python/model.py`)
- **Policy head**: logits over enumerated legal constructions (max ~512)
- **Value head**: `v_logit` (unbounded), combined as `V = tanh(v_logit + k * δ_D)` where `δ_D` is a proof-distance heuristic (how many goal sub-conditions are met)

### Training Pipeline (Python)

- **Phase A — Supervised pre-training**: Use AlphaGeometry's synthetic data generator to produce (state, construction, proof_success) triples. Train NN on these directly.
- **Phase B — MCTS self-improvement**: Run MCTS on JGEX-AG-231 problems. Policy target = visit distribution, value target = proof found (1.0/0.0). Same expert iteration loop as chess.
- **Training loop**: Reuse `train.py` structure (replay buffer, KL-div policy loss + MSE value loss)

## Existing Problem Sets & Benchmarks (No Need to Create Our Own)

### Geometry (direct fit for our architecture)

1. **AlphaGeometry's JGEX-AG-231** — 231 problems in a declarative geometry DSL, open source at [google-deepmind/alphageometry](https://github.com/google-deepmind/alphageometry). Format: construction steps → goal predicate (e.g., `cyclic`, `cong`, `perp`, `eqangle`). These range from textbook-level to competition-level. We'd parse the same DSL.

2. **AlphaGeometry's IMO-AG-30** — 30 formalized IMO geometry problems (2000-2022) in the same DSL. The gold standard. AG1 solved 25/30, AG2 solves 84% of 2000-2024 IMO geometry. We'd benchmark against the same set.

3. **AlphaGeometry's synthetic data generator** — their open-source code generates random geometric configurations + exhaustive deduction → produces millions of (problem, proof) pairs. We could use this directly or adapt it.

### Formal theorem proving (different approach, but existing benchmarks)

4. **miniF2F** — 488 problems (AMC, AIME, IMO, textbook) formalized in Lean 4 / Isabelle / Metamath. Covers algebra, number theory, geometry. Available at [openai/miniF2F](https://github.com/openai/miniF2F). This requires working in a formal proof assistant (Lean tactics as the action space).

5. **PutnamBench** — 1724 formalized Putnam competition problems in Lean 4 / Isabelle / Coq. Very hard — existing systems solve only a handful. At [trishullab.github.io/PutnamBench](https://trishullab.github.io/PutnamBench/).

### Which to target?

**AlphaGeometry's geometry problems (options 1-3) are the natural fit.** They use a custom geometry DSL (not a general proof assistant), and the problem structure maps directly onto our architecture: constructions = MCTS actions, deduction rules = Tier 1 gates, NN suggests auxiliary points = Tier 3. We can parse their exact problem format, reuse their synthetic data generator, and benchmark on their problem sets.

The Lean-based benchmarks (miniF2F, PutnamBench) are a different paradigm — the action space is Lean tactics, not geometric constructions. That's more like "LLM generates proof steps" and less like "MCTS searches over constructions with symbolic pruning." Still possible but doesn't leverage our architecture's strengths.

## MVP Scope: Triangle Center Concurrence

**Level 1 (Tier 1 only, no MCTS):** Isosceles base angles, alternate interior angles, triangle angle sum
**Level 2 (1 construction):** Circumcenter concurrence, centroid concurrence, midpoint theorem
**Level 3 (2 constructions):** Incenter concurrence, orthocenter concurrence, Thales' theorem

**Success criteria:** Solve all L1-L2 via deduction + shallow search. Solve ≥2/3 L3 with trained NN.

## Implementation Phases

| Phase | Scope | Key deliverable |
|-------|-------|-----------------|
| 1. Rust core | `ProofState`, `DeductionEngine` (21 rules), `ConstructionGen` (16 types), unit tests | Deduction alone solves Level 1 problems |
| 2. MCTS port | Port `MctsNode`, `tactical_mcts_search`, selection with construction categories | Random-rollout MCTS solves Level 2 |
| 3. PyO3 bridge | Maturin build, expose `search`/`saturate`/`encode_state`/`gen_constructions` | Python can call Rust round-trip |
| 4. NN + training | `GeoNet` model, tensor encoding, problem generator, self-play orchestrator | Trained model attempts Level 3 |
| 5. Evaluation | Ablation (with/without NN, with/without Tier 1), proof quality metrics | Publishable results |

## Key Files to Port From

| Chess engine file | What to port |
|-------------------|-------------|
| `src/mcts/tactical_mcts.rs` | Three-tier MCTS loop, `evaluate_leaf_node` flow |
| `src/mcts/node.rs` | `MctsNode` structure (visits, value, children, backprop) |
| `src/mcts/selection.rs` | UCB/PUCT with tactical priority categories |
| `src/search/mate_search.rs` | Tier 1 gate pattern (symbolic solver before NN) |
| `python/model.py` | SE-ResNet + dual head architecture |
| `python/orchestrate.py` | Self-play → train → evaluate → gate loop |
| `python/train.py` | Training pipeline, loss functions, replay buffer integration |

## Caissawary Code Patterns to Port (for geoprover CLAUDE.md)

These are the exact conventions from the chess engine that must be preserved in the geometry prover. Reference files are in `../hybrid-chess-engine/`.

### Backpropagation Sign Convention (node.rs:384-405)

Value starts as STM's perspective at the leaf. During backprop, it alternates sign at each level:
```
// value is relative to side to move at current_node
// node.total_value is relative to side that just moved (parent's side)
let reward = -value;
current_node.total_value += reward;
value = -value; // flip for parent
```

**For geometry (single-player):** No sign flip needed. Value is always "progress toward proof" from the same perspective. Backprop simplifies to `total_value += value` at every ancestor. This is a significant simplification.

### UCB/PUCT Formula (selection.rs:226-246)

```
Q = child.total_value / child.visits  (0.0 if unvisited)
U = c_puct * prior_prob * sqrt(parent_visits) / (1 + child_visits)
score = Q + U
```

For geometry: same formula, but Q is always positive (progress toward proof, 0.0 to 1.0). The `prior_prob` comes from the NN policy head. `c_puct` (exploration_constant) is a tunable hyperparameter (chess uses ~1.41).

### evaluate_leaf_node Flow (tactical_mcts.rs:351-511)

The three-tier evaluation cascade:
```
1. Check cached terminal_or_mate_value → return if set (Tier 1 cache)
2. Run domain-specific symbolic gates:
   - Chess: KOTH check, mate_search (with transposition table)
   - Geometry: saturate() (deduction to fixed point)
   If gate resolves: set terminal_or_mate_value, return ±1.0
3. If nn_value not yet computed:
   a. Call inference server (async batched prediction)
   b. Get (policy, v_logit, k) from NN
   c. Compute domain-specific heuristic delta (chess: forced_material_balance)
   d. final_value = tanh(v_logit + k * delta)
   e. Store v_logit, nn_value, k_val on node
4. Return nn_value
```

**For geometry:**
- Step 2 becomes: run `saturate()`. If goal is in facts → terminal, value = 1.0
- Step 3c: `delta_D` = proof distance heuristic (e.g., fraction of goal sub-conditions proved)
- Classical fallback (no NN): v_logit=0, k=0.5, value=tanh(0.5 * delta_D)

### Node Structure (node.rs:50-113)

Key fields to port:
- `state: ProofState` (replaces `Board`)
- `action: Option<Construction>` (replaces `Option<Move>`)
- `visits: u32`, `total_value: f64` — identical
- `terminal_or_mate_value: Option<f64>` — becomes `terminal_value: Option<f64>` (1.0 = proved, 0.0 = impossible)
- `nn_value: Option<f64>`, `v_logit: Option<f64>`, `k_val: f32` — identical
- `children: Vec<Rc<RefCell<MctsNode>>>`, `parent: Option<Weak<...>>` — identical
- `tactical_moves` / `tactical_moves_explored` → `priority_constructions` / `priority_explored` (GoalRelevant first)
- `move_priorities: HashMap<Construction, f64>` — NN policy priors
- `raw_nn_policy: Option<Vec<f32>>` — cached NN output
- `is_terminal: bool` — set when saturate() proves the goal or no constructions remain

### Selection Priority (selection.rs:18-69)

Two-phase selection:
1. **Priority phase**: Visit unexplored priority actions first (chess: checks/captures, geometry: goal-relevant/recently-active constructions)
2. **UCB phase**: Once all priority actions explored, use PUCT with NN policy priors

At root depth=0: force every child to get at least 1 visit before PUCT kicks in.

### Inference Server Pattern (inference_server.rs)

Batched async predictions:
- Caller sends `predict_async(state)` → gets a `Receiver`
- Server collects requests until batch_size or timeout
- Sends batch to GPU, distributes results back via channels
- For geometry: same pattern, different tensor shapes. state → `encode_state()` → `Vec<f32>` → batch → NN → (policy, v_logit, k)

### Value Function: V = tanh(v_logit + k * delta)

- `v_logit`: raw NN output (unbounded), represents "positional assessment"
- `k`: NN-predicted confidence scalar (how much to trust the heuristic delta)
- `delta`: domain heuristic (chess: material balance from Q-search; geometry: proof distance)
- Combined via `tanh()` to squash into [-1, 1]
- Classical fallback: v_logit=0.0, k=0.5

## Strategy: MCTS vs AlphaGeometry's LLM Approach

AlphaGeometry uses an LLM (Gemini in AG2) to suggest auxiliary constructions, with DDAR as the symbolic engine. Our approach replaces the LLM with MCTS + a small trained NN. This has potential advantages:
- **Search over constructions** rather than one-shot generation — MCTS can explore multiple construction paths and backtrack
- **Much smaller model** — a ~2M param CNN vs a billion-param LLM
- **Self-play training** rather than requiring massive pre-training corpora
- **Interpretable search tree** — can visualize which constructions were considered and why

We'd parse AlphaGeometry's problem format, use their JGEX-AG-231 and IMO-AG-30 as evaluation benchmarks, and potentially adapt their synthetic data generator for training.

## Repo Setup

Separate repo (no dependency on caissawary — the code we're "porting" is architectural patterns, not importable modules, since the chess MCTS is deeply coupled to Board/Move/MoveGen).

```
geoprover/
├── Cargo.toml              # [lib] crate-type = ["cdylib"] for PyO3
├── pyproject.toml           # maturin build config
├── src/
│   ├── lib.rs               # PyO3 module exports
│   ├── proof_state.rs       # ProofState, GeoObject, Relation
│   ├── deduction.rs         # 21 deduction rules, saturate()
│   ├── construction.rs      # 16 construction types, ConstructionGen
│   ├── mcts/
│   │   ├── mod.rs
│   │   ├── node.rs          # MctsNode (ported from caissawary)
│   │   ├── search.rs        # MCTS loop (ported from tactical_mcts.rs)
│   │   └── selection.rs     # UCB + construction priority (ported)
│   ├── encoding.rs          # state_to_tensor(), construction_to_index()
│   └── parser.rs            # Parse AlphaGeometry's JGEX problem format
├── python/
│   ├── model.py             # GeoNet (adapted from OracleNet)
│   ├── train.py             # Training loop (adapted from caissawary)
│   ├── orchestrate.py       # Self-play + train loop
│   └── evaluate.py          # Run on JGEX-AG-231, IMO-AG-30
├── problems/
│   ├── jgex_ag_231.txt      # From AlphaGeometry repo
│   └── imo_ag_30.txt        # From AlphaGeometry repo
└── tests/
    ├── test_deduction.rs
    ├── test_construction.rs
    ├── test_mcts.rs
    └── test_parser.rs
```

Setup: `maturin develop` builds the Rust extension, importable as `import geoprover` in Python.

## References

- AlphaGeometry code/data: [google-deepmind/alphageometry](https://github.com/google-deepmind/alphageometry)
- JGEX-AG-231 benchmark: 231 graded geometry problems in AG's DSL
- IMO-AG-30 benchmark: 30 formalized IMO geometry problems (2000-2022)
- AlphaGeometry2 paper: [arxiv.org/abs/2502.03544](https://arxiv.org/abs/2502.03544)
