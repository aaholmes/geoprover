# Geoprover

MCTS + small neural network for Euclidean geometry theorem proving, adapted from the Caissawary chess engine's three-tier architecture. `saturate()` (forward-chaining deduction to fixed point) replaces `mate_search` as the Tier 1 symbolic gate. The NN only suggests auxiliary constructions — the creative step deduction can't do. Benchmarks against AlphaGeometry's JGEX-AG-231 (231 problems) and IMO-AG-30 (30 formalized IMO geometry problems).

## The Geometry Domain

### ProofState — the state representation

- **Objects**: points, lines, circles — each with a `u16` ID
- **Facts**: a `HashSet<Relation>` of known truths — `Parallel(A,B,C,D)`, `Congruent(A,B,C,D)`, `Collinear(A,B,C)`, `EqualAngle(...)`, `Midpoint(M,A,B)`, `Cyclic(A,B,C,D)`, etc.
- **Goal**: a single `Relation` to prove
- **Proved** when `goal in facts`. Binary — no partial credit, no depth optimization. Any construction sequence that gets the goal into the fact set is a valid proof.
- **Zobrist hash** on facts (same XOR pattern as board positions) for transposition detection

### Auxiliary constructions — the action space (replaces chess moves)

16 types: Midpoint, AngleBisector, PerpendicularBisector, Altitude, ParallelThrough, PerpendicularThrough, Circumcenter, Incenter, Centroid, Orthocenter, CircumscribedCircle, IntersectLines, IntersectLineCircle, ReflectPoint, ExtendSegment, TangentLine.

Each construction adds new objects and seed facts to the state. ~30-50 candidates per step. Priority: GoalRelevant > RecentlyActive > Exploratory (maps directly to Check > Capture > Quiet in chess).

### saturate() — the Tier 1 gate (replaces mate_search)

Forward-chaining rule engine. Applies 21 deduction rules to the fact set until fixed point. Runs after every construction. If the goal appears in facts, the node is terminal (value = 1.0). The 21 rules:

- **Angle (8)**: vertical, supplementary, triangle sum, isosceles base, inscribed, alternate interior, corresponding, exterior
- **Congruence (4)**: SAS, ASA, SSS, CPCTC
- **Length (4)**: midpoint definition, transitive equality, midpoint theorem, angle bisector
- **Circle (3)**: equal radii, tangent perpendicular to radius, cyclic opposite angles
- **Parallel/perp (2)**: transitive parallel, perp-to-parallel

## AlphaGeometry DSL Format (Problem Input)

Problems come from AlphaGeometry's open-source datasets. Format:

```
problem_name
premises ? goal_predicate
```

Premises are `;`-separated construction clauses: `output_points = action1, action2`

Example — orthocenter concurrence:

```
orthocenter
a b c = triangle; h = on_tline b a c, on_tline c a b ? perp a h b c
```

Parses to: 4 points, initial facts from constructions (`on_tline` -> perpendicularity), goal = `Perpendicular(A,H,B,C)`.

This is a Level 1 problem — `saturate()` alone resolves it (no MCTS needed). Level 2 problems require 1 auxiliary construction + deduction. Level 3 require 2+.

## Domain Simplifications vs Chess

- **Single-player**: backprop is `total_value += value` at every ancestor (no sign flip)
- **Value range** is [0, 1] not [-1, 1] — "proof progress" from one perspective
- **Tree is shallow** (0-3 constructions for MVP problems) but correct construction choice is critical
- **delta_D** (proof distance heuristic for the value function) = fraction of goal sub-conditions satisfied
