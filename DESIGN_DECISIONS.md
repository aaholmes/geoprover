# Design Decisions

A journal of what we tried, what worked, what didn't, and why.

## Phase 1: Rust Core — ProofState and Deduction Engine

### Relation representation: canonical forms everywhere

The first big decision was how to represent geometric facts. We store relations in a `HashSet<Relation>` so we need equality to work correctly. The problem: `Parallel(A,B,C,D)` and `Parallel(B,A,D,C)` and `Parallel(C,D,A,B)` all mean the same thing.

**Decision**: Every `Relation` variant has a canonical-form constructor (`Relation::parallel()`, `Relation::congruent()`, etc.) that sorts endpoints and pairs before storing. This means `HashSet` deduplication works for free, and we never have to worry about "is this the same fact stated differently?"

**Trade-off**: The sorting logic is fiddly — `EqualAngle` normalizes each triple so the first ray point < last ray point, then sorts the two triples lexicographically. Getting this wrong causes silent duplication or missed matches. We caught several bugs here during testing.

### Zobrist hashing for state identity

Ported from the chess engine. Each fact gets a random 64-bit hash (lazily generated, seeded RNG for determinism). The state hash is the XOR of all fact hashes. This gives O(1) state comparison for MCTS transposition detection.

**Why lazy**: We don't know in advance which facts will appear, so we generate Zobrist values on demand. Seeded RNG ensures the same fact always gets the same hash across runs (important for reproducibility, though `HashSet` iteration order still causes +-2 variation in solve counts).

### Starting with 21 rules, then growing to 54

We started with AlphaGeometry's documented deduction rules (DD) — the 21 "core" rules. These got us to ~155/231 on JGEX. Then we added rules in waves:

1. **Phase 2 rules (+7)**: Thales' theorem, inscribed angle converse, isosceles converse, etc. Got us to ~165/231.
2. **Phase 3 rules (+4)**: AG22, AG25, AG27, AG31 — specific rules from the AlphaGeometry paper that we'd initially skipped. +10 problems.
3. **Triangle congruence (+3)**: SAS, ASA, SSS. These are expensive (O(n^2) or O(n^3) over congruent/angle facts) but unlock whole categories of problems. +5 problems.
4. **Ratio rules (+6)**: Equal ratios, Thales' theorem for ratios, converse Thales. Required adding the `EqualRatio` relation type. +3 problems.
5. **Quadrilateral rules (+4)**: Parallelogram, trapezoid properties.
6. **Tangent and angle bisector rules (+4)**: For circle tangent problems and incenter-related goals.
7. **Similarity and concurrence (+3)**: AA similarity, orthocenter concurrence, opposite angles to cyclic.
8. **Parallel projection (+1)**: A single complex rule for when two parallel pencils pass through concurrent collinear points.

Each wave was TDD: write a test for a specific unsolved JGEX problem, see it fail, implement the rule, see it pass.

### The degenerate-fact filter

Early on, rules would produce garbage like `Parallel(A,A,B,C)` (a "line" with zero length) or `EqualAngle(A,A,B,C,D,E)` (a degenerate angle). These pollute the fact set and cause downstream rules to generate even more garbage, leading to exponential blowup.

**Fix**: A single `retain()` filter after all rules run, rejecting any fact with degenerate geometry. This one filter prevents thousands of junk derivations per saturation cycle.

### SaturateConfig: controlling the explosion

For MCTS, we need saturation to be fast — it runs at every node expansion. The default config (`max_iterations=50, max_facts=5000`) is fine for standalone use, but MCTS child nodes often have much larger fact sets.

**MCTS fast config**: `stall_limit=8` (stop if 8 consecutive iterations produce nothing goal-relevant) + `max_new_per_iteration=1000` (batch cap, prioritizing goal-relevant facts). This typically cuts saturation time by 3-5x on complex states without missing proofs.

**Goal-relevance heuristic**: A new fact is "goal-relevant" if it shares any point IDs with the goal relation. This is a cheap check that correlates well with actual usefulness. Goal-relevant facts are always kept; non-relevant facts get truncated first when hitting the batch cap.

## Phase 2: MCTS Port

### Single-player simplification

Chess MCTS alternates sign during backpropagation because it's adversarial. Geometry is single-player — value is always "progress toward proof." So backprop simplifies to `total_value += value` at every ancestor. No sign flip. This made the port much cleaner.

### Two-phase selection

Pure PUCT from the chess engine doesn't work well for geometry. The problem: with 20-30 candidate constructions, PUCT with uniform priors wastes many iterations exploring all children once before the value signal kicks in.

**Solution**: Two-phase selection. Phase 1: visit every unvisited child once, in priority order (GoalRelevant first, then Exploratory). Phase 2: standard PUCT with NN priors. This ensures the most promising constructions get evaluated first, before we start using the NN to guide search.

### Construction priority categories

Not all auxiliary constructions are equal. A midpoint of two goal-relevant points is far more likely to be useful than a circumcenter of three random points.

**Priority scheme**: `GoalRelevant` (involves points from the goal) > `RecentlyActive` (unused) > `Exploratory` (everything else). GoalRelevant constructions are visited first in the two-phase selection.

### delta_D heuristic for classical fallback

When running without a neural network (pure MCTS), we need some value signal. `delta_D` measures the fraction of "sub-conditions" of the goal that are already satisfied. For example, if the goal is `Congruent(A,B,C,D)` and we already have facts involving A,B and C,D separately, delta_D is nonzero.

**Formula**: `V = tanh(0.5 * delta_D)`. This gives a value in [0,1] that correlates with how close we are to the goal. It's weak — just a heuristic — but enough to guide MCTS in the right direction for simple problems.

## Phase 3: PyO3 Bridge

### In-process vs subprocess

We considered three options:
1. **Pure Python**: Too slow for saturation (hundreds of rule firings per iteration).
2. **Rust subprocess**: Shell out to a Rust binary. High overhead per call, hard to share state.
3. **PyO3 in-process**: Rust code compiled as a Python extension module. Zero-copy where possible, native Python objects.

**Decision**: PyO3. The `maturin develop` workflow is smooth, and we get to call Rust functions with ~microsecond overhead. The downside: `#[pyclass(unsendable)]` required because `ProofState` contains `StdRng` which is `!Send`. This means Python can't share `PyProofState` across threads, but that's fine for our single-threaded training loop.

### What to expose

We expose the minimum needed for Python orchestration:
- `parse_problem`, `saturate`, `saturate_with_config`, `saturate_with_trace` — the core proving pipeline
- `generate_constructions`, `apply_construction` — for MCTS tree building in Python
- `encode_state`, `state_to_text`, `construction_to_text` — for NN input encoding
- `compute_delta_d` — for classical MCTS fallback
- `generate_synthetic_data` — for training data generation

Everything else (individual rules, internal state manipulation) stays Rust-private.

## Phase 4: Neural Network — From CNN to Transformer

### The CNN that didn't work

Our first NN was a CNN operating on a 20x32x32 tensor encoding. Each channel represents a relation type, and the 32x32 grid is the adjacency matrix of object pairs.

**Problem**: The encoding is sensitive to point labeling. If you rename point A to point D, the tensor changes completely even though the geometry is identical. This meant the CNN had to memorize specific point orderings rather than learning geometric structure.

**Result**: The CNN barely outperformed random. Policy loss plateaued at ~6.0 (out of log(2048) ≈ 7.6). Value predictions were noisy.

### Text-based transformer (GeoTransformer)

**Key insight**: If we serialize the proof state as text (`"coll a b c ; para a b c d ; ? perp a h b c"`), the model can attend to relationships between facts regardless of point ordering.

**Architecture**: 6-layer transformer encoder, d_model=256, 8 attention heads, dim_ff=512. Custom tokenizer with ~86 tokens (point names a-z, aux_0..aux_31, relation keywords, special tokens). ~5M parameters total.

**Why this worked**: The transformer can learn patterns like "if there's a `para` and a `coll` sharing a point, `eqangle` might follow" — structural patterns that generalize across different point labelings.

### Data augmentation via label permutation

Even with text encoding, the model could overfit to specific point name patterns (e.g., always seeing "a" as a triangle vertex). We added label permutation: randomly shuffle point names while preserving the geometric structure. This was surprisingly effective — a ~2% improvement on JGEX.

### Value head: just sigmoid

We tried several value head designs:
1. `V = sigmoid(v_logit)` — simple, single scalar
2. `V = sigmoid(v_logit) * (1 - exp(-k * delta_D))` — modulated by classical heuristic
3. `V = sigmoid(v_logit + alpha * delta_D)` — additive combination

**Winner**: The simple sigmoid. The delta_D modulation added complexity without improving results. The NN learns its own value function that subsumes the classical heuristic.

### Policy head: fixed-size vs dot-product

We implemented two policy head variants:
1. **Fixed-size**: CLS token → FC(256) → FC(2048). Each of 2048 output positions corresponds to a specific construction type + argument combination.
2. **Dot-product**: Encode each candidate construction as text, project both state and construction to the same space, compute dot-product scores.

**Decision**: Fixed-size for training/evaluation, dot-product available for future work. The fixed-size head is simpler and faster, and the 2048-slot indexing scheme (7 types x 292 slots per type) covers all constructions we generate.

## Phase 5: Training Pipeline

### Three-phase training

1. **Synthetic pre-training** (Phase A): Generate random geometry configurations in Rust, apply a random construction, find what new facts appear, use those as training targets. 50K examples, ~4 minutes to generate. This gives the model basic geometric intuition.

2. **Supervised fine-tuning** (Phase B): Run deduction on all 231 JGEX problems. For the ~181 that deduction solves, use the final state as a positive value example. For the ~50 that fail, use them as negative examples. This calibrates the value head.

3. **Expert iteration** (Phase C): Run MCTS with the current model on unsolved problems. Collect visit distributions from all tree nodes (not just the root — this was a key improvement). Train on the collected samples. Repeat.

### Why synthetic data matters

Without synthetic pre-training, supervised fine-tuning on 181 examples is not enough — the model memorizes rather than generalizes. Synthetic pre-training provides diverse geometric patterns at scale.

**Generator design**: Random 3-4 points → 1-2 setup constructions → saturate → apply key construction → saturate again → diff the fact sets. Filter for "interesting" new facts (no trivial collinear or same-point congruences). This produces examples where the construction genuinely enables new deductions.

### Expert iteration plateau

Expert iteration improves from 187/231 (random) to 189/231 (trained) but then plateaus. Each round generates ~50 new training samples from MCTS self-play, but the marginal problems require longer construction chains or more sophisticated constructions.

**Diagnosis**: The remaining ~42 unsolved problems mostly need 2+ construction steps, but our MCTS config (`max_depth=1-2`) doesn't search deep enough. The problems that MCTS does solve almost always need exactly 1 construction.

### Training on all tree nodes

Initially we only collected training samples from the MCTS root node. This wastes information — child nodes also have meaningful value estimates and visit distributions.

**Fix**: Walk the entire MCTS tree after each episode, collecting (state, value, policy) from every node with >= 2 visits. This 5-10x'd our training data per episode and improved the value head significantly.

## Proof Trace Infrastructure

### Why proof traces?

The system could answer "proved" or "not proved" but couldn't explain why. For a theorem prover, this is a significant gap. Users need to see the logical chain from axioms to the goal.

### Design: lazy premise identification

The naive approach — identify premises for every derived fact during saturation — is too slow. Rules like SAS congruence require O(n^3) search through congruent and angle facts. With ~2000 facts derived per saturation, this blows up.

**Solution**: During saturation, record only `(fact, rule_name)` with empty premises. In `extract_proof()`, walk backward from the goal through the derivation DAG, and only then call `identify_premises()` for the ~10-20 facts on the proof path. This makes saturation overhead negligible (~2x) while keeping proof extraction fast.

**Trade-off**: Premise identification uses the final fact set, not the fact set at derivation time. This means we might attribute premises that were actually derived later. In practice this doesn't matter for proof readability, and it avoids storing fact-set snapshots.

### identify_premises: pattern matching per rule

Each of the 54 rules has a characteristic "shape" — the types and point-relationship patterns of its inputs. For example:
- **TransitiveParallel**: Two `Parallel` facts sharing a line
- **MidpointDefinition**: One `Midpoint` fact containing the derived fact's points
- **SAS**: Two `Congruent` + one `EqualAngle` whose points form two triangles

The `identify_premises` function matches on `RuleName` and searches the fact set for inputs matching the expected pattern. It's essentially running the rule logic in reverse.

**Fallback**: For complex rules where precise identification is hard, we use a "point overlap" heuristic — find facts of the right type whose points overlap with the derived fact's points. This is occasionally imprecise (may return extra premises) but never misses the real ones.

## What Didn't Work

### Grid-based tensor encoding

The 20x32x32 CNN encoding was our first attempt. It suffered from:
- Point-labeling sensitivity (renaming points changes the tensor)
- Fixed 32-object cap (some problems have more)
- Sparse tensors (most cells are zero, wasting compute)
- No way to encode the goal distinctly from facts

We kept it as `encoding.rs` for ablation studies but it's not used in the final system.

### Deep MCTS search (max_depth > 2)

We tried `max_depth=3` and `max_depth=4` hoping to solve multi-step problems. The combinatorial explosion was too severe — with ~30 constructions per step, depth 3 means ~27,000 leaf states, each requiring saturation. A single problem could take 10+ minutes.

**Conclusion**: For the current system, depth 1-2 is the sweet spot. Going deeper requires better pruning (e.g., the NN accurately predicting which branches are hopeless) or a fundamentally different search strategy.

### Pure supervised learning on JGEX

We tried training directly on the 231 JGEX problems without synthetic pre-training. With only 181 positive examples, the model badly overfits. Validation loss diverged after 2-3 epochs.

### Directed angle encoding (eqangle with 8 args)

AlphaGeometry uses directed angles with 8 arguments (angle between two lines, not vertex angles). We initially tried to match this encoding but it caused confusion in our deduction rules, which think in terms of vertex angles (6 arguments). We converted to vertex-form in the parser and stored 6-arg `EqualAngle` relations internally. The `find_vertex()` function in the parser detects the shared point between two line pairs to convert 8-arg directed angles to 6-arg vertex angles.

### Hash-based deduplication of MCTS states

We tried using Zobrist hashes to detect duplicate MCTS states (transpositions). In theory, this avoids re-evaluating the same geometric configuration reached via different construction orders. In practice, transpositions are rare in geometry (unlike chess where transpositions are common), and the overhead of maintaining a transposition table wasn't worth it.

## Parser Challenges

The JGEX DSL is underdocumented. We reverse-engineered most predicate semantics from AlphaGeometry's source code. Some discoveries:

- `circle` means circumcenter (a point), not a circle object
- `on_tline x a b c` means x is on the line through a perpendicular to bc (not tangent line)
- `on_dia x a b` creates a point on the circle with diameter AB, which requires creating an implicit midpoint
- `lc_tangent x p o` means tangent from external point p to circle centered at o, with x as the tangent point
- `shift` creates a full parallelogram (both parallel pairs and congruent sides)
- `intersection_lt`, `intersection_tt`, `intersection_lp` all have different argument counts and semantics

We got to 228/231 parseable (3 problems use predicates we haven't implemented: `incenter2`, `excenter2`, `eqangle3`).

## Performance Characteristics

- **Saturation** (default config): ~400ms mean on JGEX problems, up to ~2s for complex ones
- **MCTS** (10 iterations, depth 1): ~2-3s per problem
- **Synthetic data generation**: ~42 examples/sec in Rust
- **Training**: ~3 min/epoch for 10K examples at seq_len=128 (CPU-only)
- **Test suite**: 399 Rust tests in ~90s (dominated by one MCTS timeout test)

The main bottleneck is CPU-only training. GPU training would likely 10x throughput and enable larger synthetic datasets.
