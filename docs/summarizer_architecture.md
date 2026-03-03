# Summarizer + Value/Policy Architecture

## Motivation

After `saturate()`, a proof state may contain hundreds of deduced facts, but only
a handful are relevant to the goal. Feeding all facts to the Value/Policy network
is wasteful (attention cost, truncation risk) and noisy. Feeding none of them
starves the value head of progress signal.

Key observation: saturated facts are a deterministic function of initial facts +
constructions. They don't enable new constructions (only new points do). Their
only role is informing the neural network about which constructions to try next
and how close we are to a proof.

## Design Overview

Two-stage architecture with a shared context encoder:

```
Stage 1 — Summarizer (pretrained, frozen during RL)
  Input:  initial facts + constructions + goal
  Output: relevance score per deduced fact
  Keep:   top-K deduced facts, where K = len(initial) + len(constructions) + 1

Stage 2 — Value/Policy (trained during RL)
  Input:  initial facts + constructions + summarized deductions + goal
  Output: value ∈ [0, 1], policy over constructions
```

Both stages share the same first-layer context encoder. During RL, only the
Value/Policy layers (2-N) are trained; the shared encoder and Summarizer head
remain frozen.

## Architecture Details

All encoder components are **bidirectional transformer encoders** (BERT-style,
full self-attention with no causal mask). The model reads a proof state and makes
a judgment — there is no autoregressive generation.

### encode(): Context Encoder (2 layers)

Bidirectional encoder over a multi-fact token sequence. Produces per-token
representations used as the KV cache for cross-attention scoring.

```
Input:  [CLS] fact₁ ; fact₂ ; ... ; ? goal     (tokenized)
Output: context_tokens ∈ R^(L_ctx × d)
Depth:  2 bidirectional transformer layers
```

The encoder treats all facts uniformly — it does not distinguish whether a fact
was given initially, deduced by saturation, or selected by the Summarizer. Every
fact is simply "something known that is relevant to the goal."

This component is trained during Summarizer pretraining, then frozen for RL.
Token embeddings are shared with embed().

### embed(): Single-Fact Encoder (1 layer)

Bidirectional encoder over a single fact's tokens. Same architecture as encode()
but shallower — individual facts have simple structure (a relation keyword plus
arguments), so one layer suffices. Shares the token embedding table with encode().

```
Input:  [CLS] perp a h b c                      (single fact, tokenized)
Output: fact_tokens ∈ R^(L_fact × d)
Depth:  1 bidirectional transformer layer
```

### Summarizer Head (cross-attention, default)

The fact encoder's output tokens cross-attend into the context's KV cache,
producing **context-dependent** fact representations. This allows the model
to learn that a fact is *necessary* for the proof (e.g., an angle equality
that bridges two disconnected subgraphs toward the goal), not just that it
is semantically similar to the context.

```
# Computed once (expensive):
context_tokens = encode(initial facts + constructions + goal)  # (L_ctx, d)
K = W_k · context_tokens                                       # (L_ctx, d)
V = W_v · context_tokens                                       # (L_ctx, d)

# Per candidate fact (cheap):
for each candidate fact f in saturated_facts - initial_facts:
    fact_tokens = embed(f)                             # (L_fact, d)
    Q = fact_tokens                                    # queries from fact
    attended = softmax(Q · K^T / √d) · V              # cross-attention
    fact_tokens = LayerNorm(fact_tokens + attended)    # residual + norm
    score_f = MLP(fact_tokens[CLS])                    # scalar from [CLS]

keep top-K facts by score, where K = |initial| + |constructions| + 1
```

The KV cache from the context is computed once. Each candidate fact does one
small cross-attention operation (L_fact is 3-8 tokens against L_ctx of ~50-100
context tokens) followed by an MLP on the [CLS] token. This is more expensive
per fact than the dot-product variant, but the cost is dominated by the
one-time context encoding. The key advantage: the same fact gets different
scores depending on the context, since its representation is shaped by what
it attends to in the KV cache.

### Legacy: Dot-Product Scoring (DotProductSummarizer)

Available as a faster but less expressive alternative. Context and facts are
encoded independently, scored via projected dot product:

```
h = W_ctx · encode(context)[CLS]    # (d,) computed once
f = W_fact · embed(fact)[CLS]       # (d,) per candidate
score = dot(h, f)                    # scalar
```

Fact embeddings are context-independent, limiting the model to semantic
similarity rather than logical necessity. Use via `DotProductSummarizer`.

### Value/Policy Network (Layers 2-N)

Takes the enriched input (initial + constructions + summarized deductions + goal)
and produces value and policy outputs.

```
H' = encode(initial + constructions + summarized + goal)  # shared encoder (frozen)
z  = layers_2_to_N(H')                                    # trainable, 2-3 layers

value  = sigmoid(W_v · z + b_v)                           # ∈ [0, 1]
policy = softmax(W_p · z + b_p)                           # over constructions
```

During RL, gradients flow through layers 2-N only. The shared encoder (Layer 1)
is frozen, providing a stable geometric representation learned during Summarizer
pretraining.

## Adaptive Summary Size

The number of summarized deductions K scales with problem complexity:

```
K = len(initial_facts) + len(constructions) + 1 (goal)
```

Examples:
- Simple triangle, no constructions: 5 initial + 0 + 1 = 6 deductions kept
- After 2 constructions: 5 + 2 + 1 = 8 deductions kept
- Complex setup, deep in MCTS: 10 + 4 + 1 = 15 deductions kept

This means harder problems with more constructions automatically get a richer
summary, while simple problems stay compact.

## Training Pipeline

### Phase 1: Summarizer Pretraining (supervised)

Training data comes from proof traces on solved problems:

1. Run `saturate_with_trace()` on problems solvable by deduction
2. Extract proof via backward BFS from goal
3. Label each deduced fact: 1 if on the proof path, 0 otherwise
4. Train shared encoder + Summarizer head with binary cross-entropy

This phase trains the shared context encoder to understand geometric structure
and the Summarizer head to identify proof-relevant facts.

Data augmentation (label permutation + fact shuffling) applies here.

### Phase 2: Value/Policy Training (RL via expert iteration)

1. Freeze shared encoder (Layer 1) and Summarizer head
2. Initialize Value/Policy layers (2-N) randomly
3. For each MCTS episode:
   a. At each node: saturate, run Summarizer to select top-K facts
   b. Feed enriched state to Value/Policy network
   c. Use policy for MCTS priors, value for leaf evaluation
4. Train Value/Policy layers on MCTS outcomes

### Optional Phase 3: End-to-end fine-tuning

Once Value/Policy training stabilizes, optionally unfreeze the shared encoder
with a small learning rate to allow end-to-end adaptation. This lets the
Value/Policy network influence what the Summarizer considers "relevant."

## Integration with MCTS

At each MCTS node expansion:

```
1. state = apply_construction(parent_state, construction)
2. saturate(state)                              # exhaustive deduction
3. if goal ∈ state.facts → terminal, value = 1.0
4. H = encode(initial + constructions + goal)   # shared encoder
5. summary = summarizer_select(H, state.facts, K)
6. value, policy = value_policy(initial + constructions + summary + goal)
7. expand children using policy priors
8. backpropagate value
```

## Open Questions

- **Threshold vs top-K:** Current design uses top-K. An alternative is a
  probability threshold (e.g., keep all facts with score > 0.5). Top-K gives
  fixed-size input; threshold gives variable-size but might be more principled.
  Starting with top-K for simplicity.

- **Multiple proof paths:** Proof traces yield one proof path, but alternatives
  exist. Facts labeled 0 might be relevant to alternative proofs. For a first
  pass this noise is acceptable. Future work: enumerate paths within 1-2 steps
  of the shortest proof to get richer positive labels.

- **Summarizer for MCTS-solved problems:** Phase 1 pretrains on deduction-solved
  problems. Once MCTS solves additional problems with proof traces, those can
  augment the Summarizer's training set in later iterations.
