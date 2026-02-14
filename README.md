# Geoprover

Neurosymbolic geometry theorem prover using a three-tier MCTS architecture adapted from the Caissawary chess engine. `saturate()` (forward-chaining deduction to fixed point) runs as the Tier 1 symbolic gate before every MCTS expansion. The SE-ResNet neural network suggests auxiliary constructions — the creative step deduction can't do. Benchmarks against AlphaGeometry's JGEX-AG-231 (231 problems) and IMO-AG-30 (30 formalized IMO problems).

## Current Status

| Phase | Status | Details |
|-------|--------|---------|
| 1. Rust core | **Done** | ProofState, parser (231/231 JGEX), deduction (49 rules), construction (16 types) |
| 2. MCTS | **Done** | MctsNode tree, UCB/PUCT selection, expand/evaluate/backprop, classical fallback |
| 3. PyO3 bridge | **Done** | Tensor encoding (20x32x32), full PyO3 API: parse, saturate, encode, construct |
| 4. NN + training | **Done** | GeoNet SE-ResNet (~3M params), NN-guided MCTS, expert iteration training loop |
| 5. Evaluation | **Done** | Benchmark suite comparing deduction-only vs MCTS+NN, JSON result export |

**JGEX-AG-231: ~180/231 by deduction (~78%), 185/231 with MCTS (80.1%).**

**388 Rust tests** (371 unit + 17 integration) + **13 Python NN tests**, clippy clean.

## Architecture

Hybrid Rust/Python via PyO3 (in-process, no subprocess overhead):

```
Python (orchestration, NN training, evaluation)
  │  PyO3 calls
  ▼
Rust extension module (MCTS, deduction engine, state encoding)
```

**Three-tier search (adapted from chess engine):**

| Tier | Role | Geometry equivalent |
|------|------|-------------------|
| 1 | Symbolic deduction | `saturate()` — 49 rules to fixed point |
| 2 | MCTS tree search | Search over auxiliary constructions (~30-50 candidates/step) |
| 3 | Neural oracle | GeoNet SE-ResNet (~3M params), dual-head: policy + value |

**GeoNet architecture:**
- Input: 20x32x32 tensor (12 relation channels + 3 goal channels + 1 reserved + 4 object-type channels)
- Backbone: 128-channel SE-ResNet with 6 residual blocks
- Policy head: 2048 logits over construction index space (7 types x 292 slots)
- Value head: `V = tanh(v_logit + k * delta_D)` where k is a learned confidence scalar

## Source Layout

```
src/
  proof_state.rs    ProofState, GeoObject, Relation (Zobrist hashing)
  deduction.rs      49 forward-chaining rules + degenerate-fact filtering
  construction.rs   16 construction types with priority classification
  parser.rs         JGEX DSL parser (40+ predicates, 231/231 coverage)
  encoding.rs       state_to_tensor() — 20x32x32 relation adjacency grid
  lib.rs            PyO3 bridge (PyProofState, PyConstruction, 7 exposed functions)
  mcts/
    mod.rs          Module re-exports
    node.rs         MctsNode (Rc<RefCell> tree), expand, evaluate, backprop, UCB/PUCT
    search.rs       mcts_search() loop, select_leaf, proof path extraction
python/
  model.py          GeoNet SE-ResNet, construction indexing, tensor conversion
  orchestrate.py    NN-guided MCTS (Python-side tree), self-play data collection
  train.py          Supervised pre-training + expert iteration training loop
  evaluate.py       Benchmark suite: deduction vs MCTS+NN, comparison reports
  test_nn.py        13 end-to-end tests for NN modules
  test_bridge.py    PyO3 bridge smoke tests
```

## The Geometry Domain

### ProofState

- **Objects**: points, lines, circles — each with a `u16` ID
- **Facts**: `HashSet<Relation>` — `Parallel`, `Congruent`, `Collinear`, `EqualAngle`, `Midpoint`, `Perpendicular`, `OnCircle`, `Cyclic`, `EqualRatio`
- **Goal**: a single `Relation` to prove
- **Proved** when `goal ∈ facts`. Any construction sequence that gets the goal into the fact set is a valid proof.
- **Zobrist hash** on facts for transposition detection

### Auxiliary Constructions (Action Space)

7 generated types: Midpoint, Altitude, Circumcenter, Orthocenter, Incenter, ParallelThrough, PerpendicularThrough. Priority: GoalRelevant > RecentlyActive > Exploratory. Capped at 30 children per MCTS node.

### MCTS Search

- **Select**: Two-phase — visit unvisited children first (by priority), then UCB/PUCT with NN priors
- **Expand**: Generate constructions, score with NN policy head, create child nodes
- **Evaluate**: Run `saturate()`. If proved, value=1.0. Otherwise `V = tanh(v_logit + k * delta_D)`
- **Backprop**: Single-player — `total_value += value` at every ancestor (no sign flip)

### Deduction Rules (49 active)

**Parallel/perpendicular**: transitive parallel, perp-to-parallel, perp+parallel transfer, parallel+collinear extension, perp+collinear extension, parallel shared point collinear, two equidistant points perpendicular, equidistant+cyclic perpendicular (AG25), eqangle+perp transfer (AG31)

**Congruence/midpoint**: transitive congruent, midpoint definition, midpoint converse, equidistant midpoint, perpendicular bisector, isosceles converse, perp+midpoint congruent, midpoint diagonal parallelogram, cyclic equal angle congruent, midpoint+parallelogram (AG27)

**Triangle congruence**: SAS (side-angle-side), ASA (angle-side-angle), SSS (side-side-side) — all with non-collinear guards

**Angles**: isosceles base angles, alternate interior angles, corresponding angles (AG9, two perps), transitive equal angle, perpendicular right angles, equal angles to parallel, cyclic inscribed angles, inscribed angle converse, cyclic+parallel base angles (AG22)

**Ratios**: transitive ratio, ratio=1 to congruence, midpoint to ratio, Thales (parallel+collinear to ratio), congruent to ratio, converse Thales (ratio+collinear to parallel), parallel base ratio (|AB|/|CD| from transversals)

**Quadrilaterals**: parallelogram opposite angles, isosceles trapezoid base angles, trapezoid midsegment

**Circles**: circle-point equidistance, congruent to OnCircle, cyclic from OnCircle, Thales' theorem, equal tangent lengths, tangent-chord angle

**Angle bisector**: angle bisector ratio (bisector theorem), incenter equal inradii

**Collinearity**: collinear transitivity, midline parallel

## Problem Format (AlphaGeometry JGEX DSL)

```
problem_name
premises ? goal_predicate
```

Example — orthocenter concurrence:
```
orthocenter
a b c = triangle; h = on_tline b a c, on_tline c a b ? perp a h b c
```

## Build & Test

```bash
cargo test                                          # 388 Rust tests
cargo clippy                                        # lint
cargo test --test test_integration -- --nocapture   # integration tests with output
maturin develop                                     # build PyO3 extension
python python/test_bridge.py                        # PyO3 bridge smoke tests
python python/test_nn.py                            # NN module tests
```

## Training

```bash
# Supervised pre-training on deduction-solvable problems
python python/train.py --supervised-only

# Full expert iteration (supervised + self-play)
python python/train.py --iterations 20 --device cuda

# Evaluate against JGEX-AG-231
python python/evaluate.py --checkpoint checkpoints/iter_020.pt

# Deduction-only baseline
python python/evaluate.py --deduction-only
```
