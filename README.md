# Geoprover

Neurosymbolic geometry theorem prover using a three-tier MCTS architecture adapted from the Caissawary chess engine. `saturate()` (forward-chaining deduction to fixed point) runs as the Tier 1 symbolic gate before every MCTS expansion. A text-based GeoTransformer neural network suggests auxiliary constructions — the creative step deduction can't do. Benchmarks against AlphaGeometry's JGEX-AG-231 (231 problems) and IMO-AG-30 (30 IMO competition problems).

## Results

### JGEX-AG-231 (standard benchmark)

| Method | Solved | Rate | Mean Time | Notes |
|--------|--------|------|-----------|-------|
| Deduction only | 181/231 | 78.4% | 428ms | 49 rules, pure symbolic |
| MCTS + random NN | 187/231 | 81.0% | 3.6s | Untrained transformer baseline |
| **MCTS + trained NN** | **189/231** | **81.8%** | **2.7s** | Synthetic + supervised pre-training |

The trained NN solves **9 additional problems** that pure deduction cannot, including **Morley's theorem** and the **9-point circle**. The trained model is 25% faster than the random baseline because it prioritizes promising constructions.

Solve counts vary ±2 between runs due to `HashSet` iteration non-determinism in Rust.

### IMO-AG-30 (competition problems)

| Method | Solved | Rate | Mean Time | Notes |
|--------|--------|------|-----------|-------|
| Deduction only | 5/30 | 16.7% | 751ms | Pure symbolic |
| **MCTS + trained NN** | **7/30** | **23.3%** | **15.5s** | +2 problems over deduction |

MCTS+NN solves 2 additional IMO problems that deduction cannot: **IMO 2012 P1** (excircle tangent congruence, 44s) and **IMO 2019 P2** (cyclic quadrilateral, 73s). Both required a single auxiliary construction to unlock the proof.

### Comparison with AlphaGeometry

| System | Model Size | JGEX-AG-231 | IMO-AG-30 |
|--------|-----------|-------------|-----------|
| AlphaGeometry DD (deduction only) | — | ~75% | — |
| AlphaGeometry (DD + LLM) | ~7B params | — | 25/30 |
| **Geoprover (deduction only)** | **—** | **78.4%** | **5/30** |
| **Geoprover (MCTS + trained NN)** | **~4M params** | **81.8%** | **7/30** |

Geoprover's deduction engine exceeds AlphaGeometry's DD baseline on JGEX-AG-231. The MCTS+NN system achieves this with a **1750x smaller** neural component (~4M vs ~7B parameters). The IMO gap (7/30 vs 25/30) reflects the difficulty of multi-step auxiliary construction chains — IMO problems often require 3-5 constructions, while most JGEX problems need 0-1.

### Problems solved by MCTS+NN that deduction cannot

**JGEX-AG-231:**

| Problem | Steps | Time |
|---------|-------|------|
| L046-16 (parallel line concurrence) | 1 | 236ms |
| 9-point circle | 1 | 1.3s |
| L058-9 (angle bisector) | 1 | 35ms |
| **Morley's theorem** | 1 | 2.1s |
| GDD_FULL 21-40 #40 | 1 | 100ms |
| Auxiliary construction #22 | 1 | 143ms |
| Ye's auxiliary thinking | 2 | 19.1s |
| E076-31 | 1 | 9.5s |
| E051-28 | 1 | 5.9s |

**IMO-AG-30:**

| Problem | Steps | Time |
|---------|-------|------|
| IMO 2012 P1 (excircle tangent) | 1 | 43.7s |
| IMO 2019 P2 (cyclic quadrilateral) | 1 | 73.2s |

## Current Status

| Phase | Status | Details |
|-------|--------|---------|
| 1. Rust core | **Done** | ProofState, parser (231/231 JGEX), deduction (49 rules), construction (7 types) |
| 2. MCTS | **Done** | MctsNode tree, UCB/PUCT selection, expand/evaluate/backprop, classical fallback |
| 3. PyO3 bridge | **Done** | Text + tensor encoding, full PyO3 API: parse, saturate, encode, construct |
| 4. NN + training | **Done** | GeoTransformer (~4M params), NN-guided MCTS, 3-phase training pipeline |
| 5. Evaluation | **Done** | Benchmark suite, 3-way comparison (deduction vs random vs trained NN) |

**390 Rust tests** (373 unit + 17 integration) + **17 Python NN tests**, clippy clean.

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
| 2 | MCTS tree search | Search over auxiliary constructions (~20-30 candidates/step) |
| 3 | Neural oracle | GeoTransformer (~4M params), dual-head: policy + value |

**GeoTransformer architecture:**
- Input: tokenized text sequence (proof state as relation list + goal)
- Backbone: 6-layer transformer encoder, d_model=256, 8 heads, dim_ff=512
- Custom tokenizer: ~86-token geometry vocabulary (point names, relation/construction keywords)
- Policy head: 2048 logits over construction index space (7 types x 292 slots)
- Value head: `V = tanh(v_logit + k * delta_D)` where k is a learned confidence scalar
- ~4M trainable parameters

**Training pipeline (3-phase):**
1. Synthetic pre-training on Rust-generated random geometry configurations (10K+ examples)
2. Supervised fine-tuning on deduction-solvable JGEX problems
3. Expert iteration: MCTS self-play → collect visit distributions → train → repeat

## Source Layout

```
src/
  proof_state.rs    ProofState, GeoObject, Relation (Zobrist hashing, text serialization)
  deduction.rs      49 forward-chaining rules + degenerate-fact filtering
  construction.rs   7 auxiliary construction types with priority classification
  parser.rs         JGEX DSL parser (40+ predicates, 231/231 coverage)
  encoding.rs       state_to_tensor() — 20x32x32 relation adjacency grid (legacy)
  synthetic.rs      Random geometry data generator for pre-training
  lib.rs            PyO3 bridge (PyProofState, PyConstruction, 10 exposed functions)
  mcts/
    mod.rs          Module re-exports
    node.rs         MctsNode (Rc<RefCell> tree), expand, evaluate, backprop, UCB/PUCT
    search.rs       mcts_search() loop, select_leaf, proof path extraction
python/
  model.py          GeoTransformer (text-based), GeoNetCNN (legacy), tokenizer
  orchestrate.py    NN-guided MCTS (Python-side tree), self-play data collection
  train.py          3-phase training: synthetic → supervised → expert iteration
  evaluate.py       Benchmark suite: deduction vs MCTS+NN, comparison reports
  test_nn.py        17 end-to-end tests for NN modules
  test_bridge.py    11 PyO3 bridge smoke tests
```

## The Geometry Domain

### ProofState

- **Objects**: points, lines, circles — each with a `u16` ID
- **Facts**: `HashSet<Relation>` — `Parallel`, `Congruent`, `Collinear`, `EqualAngle`, `Midpoint`, `Perpendicular`, `OnCircle`, `Cyclic`, `EqualRatio`
- **Goal**: a single `Relation` to prove
- **Proved** when `goal ∈ facts`. Any construction sequence that gets the goal into the fact set is a valid proof.
- **Zobrist hash** on facts for transposition detection
- **Text encoding**: `to_text()` serializes state as `"coll a b c ; para a b c d ; ? perp a h b c"`

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
cargo test                                          # 390 Rust tests
cargo clippy                                        # lint
cargo test --test test_integration -- --nocapture   # integration tests with output
maturin develop                                     # build PyO3 extension
python python/test_bridge.py                        # PyO3 bridge smoke tests
python python/test_nn.py                            # NN module tests
```

## Training

```bash
# Phase A+B: Synthetic pre-training + supervised fine-tuning
python python/train.py --supervised-only --synthetic-size 10000

# Full pipeline: synthetic + supervised + expert iteration
python python/train.py --iterations 5 --synthetic-size 10000

# Resume expert iteration from checkpoint
python python/train.py --resume checkpoints/supervised.pt --iterations 10

# Evaluate
python python/evaluate.py --checkpoint checkpoints/iter_005.pt
python python/evaluate.py --deduction-only
```
