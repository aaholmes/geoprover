# Geoprover

Neurosymbolic geometry theorem prover using a three-tier MCTS architecture adapted from the Caissawary chess engine. `saturate()` (forward-chaining deduction to fixed point) runs as the Tier 1 symbolic gate before every MCTS expansion. The NN (Phase 4, not yet implemented) will only suggest auxiliary constructions — the creative step deduction can't do. Benchmarks against AlphaGeometry's JGEX-AG-231 (231 problems) and IMO-AG-30 (30 formalized IMO problems).

## Current Status

| Phase | Status | Details |
|-------|--------|---------|
| 1. Rust core | **Done** | ProofState, parser (231/231 JGEX), deduction (49 rules), construction (16 types) |
| 2. MCTS | **Done** | MctsNode tree, UCB/PUCT selection, expand/evaluate/backprop, classical fallback |
| 3. PyO3 bridge | Boilerplate | `lib.rs` has module init, no exposed functions yet |
| 4. NN + training | Not started | `encoding.rs` is a stub, no Python code |
| 5. Evaluation | Not started | No benchmark harness beyond integration tests |

**JGEX-AG-231: 179/231 by deduction (77.5%), 185/231 with MCTS (80.1%).**

**353 tests passing** (336 unit + 17 integration), clippy clean.

## Architecture

Hybrid Rust/Python via PyO3 (in-process, no subprocess overhead):

```
Python (orchestration, NN training, evaluation)
  |  PyO3 calls
  v
Rust extension module (MCTS, deduction engine, state encoding)
```

**Three-tier search (adapted from chess engine):**

| Tier | Role | Geometry equivalent |
|------|------|-------------------|
| 1 | Symbolic deduction | `saturate()` — 49 rules to fixed point |
| 2 | MCTS tree search | Search over auxiliary constructions (~30-50 candidates/step) |
| 3 | Neural oracle | SE-ResNet (~2M params), dual-head: policy + value (Phase 4) |

## Source Layout

```
src/
  proof_state.rs    ProofState, GeoObject, Relation (Zobrist hashing)
  deduction.rs      49 forward-chaining rules + degenerate-fact filtering
  construction.rs   16 construction types with priority classification
  parser.rs         JGEX DSL parser (40+ predicates, 231/231 coverage)
  mcts/
    mod.rs          Module re-exports
    node.rs         MctsNode (Rc<RefCell> tree), expand, evaluate, backprop, UCB/PUCT
    search.rs       mcts_search() loop, select_leaf, proof path extraction
  encoding.rs       Stub for state-to-tensor encoding
  lib.rs            PyO3 module init
```

## The Geometry Domain

### ProofState

- **Objects**: points, lines, circles — each with a `u16` ID
- **Facts**: `HashSet<Relation>` — `Parallel`, `Congruent`, `Collinear`, `EqualAngle`, `Midpoint`, `Perpendicular`, `OnCircle`, `Cyclic`, `EqualRatio`
- **Goal**: a single `Relation` to prove
- **Proved** when `goal in facts`. Any construction sequence that gets the goal into the fact set is a valid proof.
- **Zobrist hash** on facts for transposition detection

### Auxiliary Constructions (Action Space)

16 types: Midpoint, AngleBisector, PerpendicularBisector, Altitude, ParallelThrough, PerpendicularThrough, Circumcenter, Incenter, Centroid, Orthocenter, CircumscribedCircle, IntersectLines, IntersectLineCircle, ReflectPoint, ExtendSegment, TangentLine. Priority: GoalRelevant > RecentlyActive > Exploratory.

### MCTS Search

- **Select**: Two-phase — visit unvisited children first (by priority), then UCB/PUCT
- **Expand**: Generate constructions, create child nodes (capped branching factor)
- **Evaluate**: Run `saturate()`. If proved, value=1.0. Otherwise classical fallback: `value = tanh(0.5 * delta_D)`
- **Backprop**: Single-player — `total_value += value` at every ancestor (no sign flip)
- **UCB**: `Q + c_puct * prior * sqrt(parent_visits) / (1 + child_visits)`, uniform priors in Phase 2

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
cargo test                                          # all 353 tests
cargo clippy                                        # lint
cargo test --test test_integration -- --nocapture   # integration tests with output
maturin develop                                     # build PyO3 extension (Phase 3+)
```
