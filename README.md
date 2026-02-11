# Geoprover

Neurosymbolic geometry theorem prover using a three-tier MCTS architecture adapted from the Caissawary chess engine. `saturate()` (forward-chaining deduction to fixed point) runs as the Tier 1 symbolic gate before every MCTS expansion. The NN (Phase 4, not yet implemented) will only suggest auxiliary constructions — the creative step deduction can't do. Benchmarks against AlphaGeometry's JGEX-AG-231 (231 problems) and IMO-AG-30 (30 formalized IMO problems).

## Current Status

| Phase | Status | Details |
|-------|--------|---------|
| 1. Rust core | **Done** | ProofState, parser (228/231 JGEX), deduction (9 rules), construction (8/16 types) |
| 2. MCTS | **Done** | MctsNode tree, UCB/PUCT selection, expand/evaluate/backprop, classical fallback |
| 3. PyO3 bridge | Boilerplate | `lib.rs` has module init, no exposed functions yet |
| 4. NN + training | Not started | `encoding.rs` is a stub, no Python code |
| 5. Evaluation | Not started | No benchmark harness beyond integration tests |

**206 tests passing** (189 unit + 17 integration), clippy clean, ~3,800 LOC Rust.

MCTS solves synthetic problems (midpoint congruence, circumcenter equidistance) in 1-2 iterations. JGEX-AG-231 problems require additional deduction rules (inscribed angle theorem, Thales' theorem, cyclic quadrilateral properties) before MCTS search becomes effective.

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
| 1 | Symbolic deduction | `saturate()` — 9 rules to fixed point (21 planned) |
| 2 | MCTS tree search | Search over auxiliary constructions (~30-50 candidates/step) |
| 3 | Neural oracle | SE-ResNet (~2M params), dual-head: policy + value (Phase 4) |

## Source Layout

```
src/
  proof_state.rs    ProofState, GeoObject, Relation (Zobrist hashing)
  deduction.rs      9 forward-chaining rules (transitive parallel/congruent, isosceles, etc.)
  construction.rs   8 construction types with priority classification
  parser.rs         JGEX DSL parser (30+ predicates, 228/231 coverage)
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
- **Facts**: `HashSet<Relation>` — `Parallel`, `Congruent`, `Collinear`, `EqualAngle`, `Midpoint`, `Perpendicular`, `OnCircle`, `Cyclic`
- **Goal**: a single `Relation` to prove
- **Proved** when `goal in facts`. Any construction sequence that gets the goal into the fact set is a valid proof.
- **Zobrist hash** on facts for transposition detection

### Auxiliary Constructions (Action Space)

16 types defined, 8 implemented: Midpoint, Altitude, Circumcenter, Orthocenter, Incenter, ParallelThrough, PerpendicularThrough, + fallback. Priority: GoalRelevant > RecentlyActive > Exploratory.

### MCTS Search

- **Select**: Two-phase — visit unvisited children first (by priority), then UCB/PUCT
- **Expand**: Generate constructions, create child nodes (capped branching factor)
- **Evaluate**: Run `saturate()`. If proved, value=1.0. Otherwise classical fallback: `value = tanh(0.5 * delta_D)`
- **Backprop**: Single-player — `total_value += value` at every ancestor (no sign flip)
- **UCB**: `Q + c_puct * prior * sqrt(parent_visits) / (1 + child_visits)`, uniform priors in Phase 2

### Deduction Rules (9/21 implemented)

- Transitive parallel, perp-to-parallel
- Midpoint definition (collinear + congruent)
- Transitive congruent
- Isosceles base angles
- Alternate interior angles
- Transitive equal angle
- Stubs: corresponding angles, perpendicular angles

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
cargo test                                          # all 206 tests
cargo clippy                                        # lint
cargo test --test test_integration -- --nocapture   # integration tests with output
maturin develop                                     # build PyO3 extension (Phase 3+)
```
