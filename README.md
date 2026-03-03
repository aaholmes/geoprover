# Geoprover

Neurosymbolic geometry theorem prover using a three-tier MCTS architecture adapted from the Caissawary chess engine. `saturate()` (forward-chaining deduction to fixed point) runs as the Tier 1 symbolic gate before every MCTS expansion. A text-based GeoTransformer neural network suggests auxiliary constructions — the creative step deduction can't do. Benchmarks against AlphaGeometry's JGEX-AG-231 (231 problems) and IMO-AG-30 (30 IMO competition problems).

## Results

### JGEX-AG-231 (standard benchmark)

| Method | Solved | Rate | Mean Time | Notes |
|--------|--------|------|-----------|-------|
| Deduction only | 181/231 | 78.4% | 428ms | 54 rules, pure symbolic |
| MCTS + random NN | 187/231 | 81.0% | 3.6s | Untrained transformer baseline |
| **MCTS + trained NN** | **189/231** | **81.8%** | **2.7s** | Synthetic + supervised pre-training |

The trained NN solves **9 additional problems** that pure deduction cannot, including **Morley's theorem** and the **9-point circle**. The trained model is 25% faster than the random baseline because it prioritizes promising constructions.

Solve counts vary +/-2 between runs due to `HashSet` iteration non-determinism in Rust.

### IMO-AG-30 (competition problems)

| Method | Solved | Rate | Mean Time | Notes |
|--------|--------|------|-----------|-------|
| Deduction only | 5/30 | 16.7% | 751ms | Pure symbolic |
| **MCTS + trained NN** | **7/30** | **23.3%** | **15.5s** | +2 problems over deduction |

MCTS+NN solves 2 additional IMO problems that deduction cannot: **IMO 2012 P1** (excircle tangent congruence, 44s) and **IMO 2019 P2** (cyclic quadrilateral, 73s). Both required a single auxiliary construction to unlock the proof.

### Comparison with AlphaGeometry

| System | Model Size | JGEX-AG-231 | IMO-AG-30 |
|--------|-----------|-------------|-----------|
| AlphaGeometry DD (deduction only) | -- | ~75% | -- |
| AlphaGeometry (DD + LLM) | ~7B params | -- | 25/30 |
| **Geoprover (deduction only)** | **--** | **78.4%** | **5/30** |
| **Geoprover (MCTS + trained NN)** | **~5M params** | **81.8%** | **7/30** |

Geoprover's deduction engine exceeds AlphaGeometry's DD baseline on JGEX-AG-231. The MCTS+NN system achieves this with a **1400x smaller** neural component (~5M vs ~7B parameters). The IMO gap (7/30 vs 25/30) reflects the difficulty of multi-step auxiliary construction chains — IMO problems often require 3-5 constructions, while most JGEX problems need 0-1.

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

## Proof Output

`saturate_with_trace()` records derivation provenance during deduction, producing human-readable proofs that trace each step from axioms to the goal.

**Circumcenter equidistance** — circumcenter O of triangle ABC is equidistant from A and C:
```
Proof (3 steps):
  1. cong a o b o [axiom]
  2. cong b o c o [axiom]
  3. cong a o c o [TransitiveCongruent from 1, 2]
```

**Isosceles base angles** — if |AB| = |AC|, then angle ABC = angle ACB:
```
Proof (2 steps):
  1. cong a b a c [axiom]
  2. eqangle a b c a c b [IsoscelesBaseAngles from 1]
```

**Two altitudes imply the third** — perpendicular from A to BC via the orthocenter:
```
Proof (3 steps):
  1. perp b c b e [ThalesTheorem]
  2. perp a d b c [axiom]
  3. para a d b e [PerpToParallel from 2, 1]
```

Proofs are extracted by backward BFS from the goal through the derivation DAG, then topologically sorted so axioms appear first.

```python
import geoprover

state = geoprover.parse_problem("iso\na b c = iso_triangle a b c ? eqangle b a b c c a c b")
proved, trace = geoprover.saturate_with_trace(state)
print(trace.format_proof())
```

## Architecture

Hybrid Rust/Python via PyO3 (in-process, no subprocess overhead):

```
Python (orchestration, NN training, evaluation, visualization)
  |  PyO3 calls
  v
Rust extension module (MCTS, deduction engine, state encoding, synthetic data)
```

**Three-tier search (adapted from chess engine):**

| Tier | Role | Geometry equivalent |
|------|------|-------------------|
| 1 | Symbolic deduction | `saturate()` — 54 rules to fixed point |
| 2 | MCTS tree search | Search over auxiliary constructions (~20-30 candidates/step) |
| 3 | Neural oracle | GeoTransformer (~5M params), dual-head: policy + value |

**Two neural architectures** (selectable via `--model-type`):

**SetGeoTransformer** (`--model-type set`, ~3.9M params) — set-equivariant, permutation invariant by construction:

```
                    ┌─────────────────────────────────────────────┐
                    │           Stage 1: Per-Statement Encoding   │
                    │         (shared 2-layer transformer)        │
                    │         intra-fact positional embeddings     │
                    └──────────────┬──────────────────┬───────────┘
                                   │                  │
                         fact [CLS] embs        goal [CLS] emb
                          (B, N, 256)             (B, 1, 256)
                                   │                  │
                    ┌──────────────┴──────────────┐   │
                    │  Stage 2: Fact Self-Attention│   │
                    │  (2-layer transformer)       │   │
                    │  NO positional embeddings    │   │
                    │  → permutation equivariant   │   │
                    └──────────────┬───────────────┘   │
                                   │                   │
                         enriched facts          goal query
                          (B, N, 256)           (B, 1, 256)
                                   │                   │
                    ┌──────────────┴───────────────────┴──────────┐
                    │    Stage 3: Goal-Conditioned Aggregation    │
                    │    (2-layer cross-attention: goal → facts)  │
                    │    + mean-pool residual                     │
                    └──────────────────────┬──────────────────────┘
                                           │
                                     state_repr
                                      (B, 256)
                                      ┌────┴────┐
                              ┌───────┴──┐  ┌───┴────────┐
                              │  Value   │  │   Policy   │
                              │  FC→sig  │  │  FC→2048   │
                              │  [0, 1]  │  │  logits    │
                              └──────────┘  └────────────┘
```

Key properties:
- **Permutation invariant**: No cross-fact positional embeddings; reordering facts produces identical output
- **Goal isolation**: Goal uses same encoder but separate cross-attention projections as the "query" into facts
- **Subsumes FactSummarizer**: Stage 3 attention weights implicitly learn fact relevance (no separate model needed)
- **Eliminates fact-order sensitivity**: The +/-2 solve-count variation from `HashSet` iteration order is architecturally eliminated

**GeoTransformer** (`--model-type transformer`, ~4M params) — the original flat-sequence model:
- Input: tokenized text sequence (proof state as relation list + goal)
- Backbone: 6-layer transformer encoder, d_model=256, 8 heads, dim_ff=512
- Custom tokenizer: ~86-token geometry vocabulary (point names, relation/construction keywords)
- Policy head: 2048 logits over construction index space (7 types x 292 slots)
- Value head: `V = sigmoid(v_logit)` — single scalar output in [0, 1]

**Training pipeline (3-phase, end-to-end for SetGeoTransformer):**
1. Synthetic pre-training on Rust-generated random geometry configurations (50K examples, mixed difficulty, with negative examples)
2. Supervised fine-tuning on JGEX problems with MCTS-derived policy targets (not zeros) + all-node sample collection
3. Expert iteration: MCTS self-play -> collect visit distributions from all tree nodes -> train -> repeat

The SetGeoTransformer trains fully end-to-end — Stage 3 cross-attention learns fact relevance implicitly from the policy+value loss, eliminating the separate FactSummarizer pre-training step. The GeoTransformer optionally uses a pretrained FactSummarizer (frozen during RL) to filter post-saturation facts.

## Visualization

Static proof diagrams and animated walkthroughs are generated with matplotlib:

```bash
# Render a specific problem
python python/visualize.py --problem "9point" --output diagrams/

# Animated proof walkthrough (GIF)
python python/animate.py --problem "9point" --mode steps --output diagrams/

# MCTS tree visualization
python python/animate.py --problem "morley" --mode mcts --output diagrams/
```

An interactive web dashboard is available at `web/index.html` — open with any HTTP server:

```bash
python -m http.server 8080
# Then visit http://localhost:8080/web/
```

Features: problem browser with search/filter, ablation comparison charts (Plotly), IMO results vs AlphaGeometry, architecture overview.

## Source Layout

```
src/
  proof_state.rs    ProofState, GeoObject, Relation (Zobrist hashing, text serialization)
  deduction.rs      54 forward-chaining rules + saturate_with_trace() for proof output
  construction.rs   7 auxiliary construction types with priority classification
  parser.rs         JGEX DSL parser (40+ predicates, 231/231 coverage)
  proof_trace.rs    RuleName, Derivation, ProofTrace, identify_premises (provenance tracking)
  encoding.rs       state_to_tensor() — 20x32x32 relation adjacency grid (legacy)
  synthetic.rs      Random geometry data generator: multi-point, multi-step, negative examples
  lib.rs            PyO3 bridge (PyProofState, PyConstruction, PyProofTrace, 12 functions)
  mcts/
    mod.rs          Module re-exports
    node.rs         MctsNode (Rc<RefCell> tree), expand, evaluate, backprop, UCB/PUCT
    search.rs       mcts_search() loop, select_leaf, proof path extraction
python/
  model.py          SetGeoTransformer, GeoTransformer, GeoNetCNN (legacy), tokenizer
  orchestrate.py    NN-guided MCTS (Python-side tree), all-node sample collection
  train.py          3-phase training: synthetic -> supervised -> expert iteration
  evaluate.py       Benchmark suite: deduction vs MCTS+NN, comparison reports
  summarizer.py     FactSummarizer: cross-attention fact relevance scoring
  visualize.py      Coordinate synthesis + static proof diagram rendering
  animate.py        Animated proof walkthroughs + MCTS tree visualization
  test_nn.py        36 end-to-end tests for NN modules
  test_bridge.py    14 PyO3 bridge smoke tests
web/
  index.html        Interactive dashboard: problem browser, charts, IMO comparison
```

## The Geometry Domain

### ProofState

- **Objects**: points, lines, circles — each with a `u16` ID
- **Facts**: `HashSet<Relation>` — `Parallel`, `Congruent`, `Collinear`, `EqualAngle`, `Midpoint`, `Perpendicular`, `OnCircle`, `Cyclic`, `EqualRatio`
- **Goal**: a single `Relation` to prove
- **Proved** when `goal in facts`. Any construction sequence that gets the goal into the fact set is a valid proof.
- **Zobrist hash** on facts for transposition detection
- **Text encoding**: `to_text()` serializes state as `"coll a b c ; para a b c d ; ? perp a h b c"`

### Auxiliary Constructions (Action Space)

7 generated types: Midpoint, Altitude, Circumcenter, Orthocenter, Incenter, ParallelThrough, PerpendicularThrough. Priority: GoalRelevant > Exploratory. Capped at 30 children per MCTS node.

### MCTS Search

- **Select**: Two-phase — visit unvisited children first (by priority), then UCB/PUCT with NN priors
- **Expand**: Generate constructions, score with NN policy head, create child nodes
- **Evaluate**: Run `saturate()`. If proved, value=1.0. Otherwise `V = sigmoid(v_logit)`
- **Backprop**: Single-player — `total_value += value` at every ancestor (no sign flip)

### Deduction Rules (54 active)

**Parallel/perpendicular**: transitive parallel, perp-to-parallel, perp+parallel transfer, parallel+collinear extension, perp+collinear extension, parallel shared point collinear, two equidistant points perpendicular, equidistant+cyclic perpendicular (AG25), eqangle+perp transfer (AG31)

**Congruence/midpoint**: transitive congruent, midpoint definition, midpoint converse, equidistant midpoint, perpendicular bisector, isosceles converse, perp+midpoint congruent, midpoint diagonal parallelogram, cyclic equal angle congruent, midpoint+parallelogram (AG27)

**Triangle congruence**: SAS (side-angle-side), ASA (angle-side-angle), SSS (side-side-side) — all with non-collinear guards

**Angles**: isosceles base angles, alternate interior angles, corresponding angles (AG9, two perps), transitive equal angle, perpendicular right angles, equal angles to parallel, cyclic inscribed angles, inscribed angle converse, cyclic+parallel base angles (AG22), AA similarity

**Ratios**: transitive ratio, ratio=1 to congruence, midpoint to ratio, Thales (parallel+collinear to ratio), congruent to ratio, converse Thales (ratio+collinear to parallel), parallel base ratio (|AB|/|CD| from transversals)

**Quadrilaterals**: parallelogram opposite angles, isosceles trapezoid base angles, trapezoid midsegment

**Circles**: circle-point equidistance, congruent to OnCircle, cyclic from OnCircle, Thales' theorem, equal tangent lengths, tangent-chord angle, opposite angles to cyclic

**Angle bisector**: angle bisector ratio (bisector theorem), incenter equal inradii

**Concurrence**: orthocenter concurrence — two altitudes imply the third

**Collinearity**: collinear transitivity, midline parallel

**Projection**: parallel projection (two parallel pencils through concurrent collinear lines)

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
cargo test                                          # 399 Rust tests
cargo clippy                                        # lint
cargo test --test test_integration -- --nocapture   # integration tests with output
maturin develop                                     # build PyO3 extension
python python/test_bridge.py                        # PyO3 bridge smoke tests (14 tests)
python python/test_nn.py                            # NN module tests (36 tests)
./scripts/test-fast.sh                              # Full fast suite (~20s)
```

## Training

```bash
# Train SetGeoTransformer (set-equivariant, recommended)
python python/train.py --model-type set --supervised-only --synthetic-size 50000

# Train GeoTransformer (original flat-sequence model)
python python/train.py --model-type transformer --supervised-only --synthetic-size 50000

# Full pipeline: synthetic + supervised + expert iteration
python python/train.py --model-type set --iterations 10 --synthetic-size 50000

# Resume expert iteration from checkpoint
python python/train.py --model-type set --resume checkpoints/supervised.pt --iterations 10

# Evaluate
python python/evaluate.py --model-type set --checkpoint checkpoints/iter_005.pt
python python/evaluate.py --deduction-only
```
