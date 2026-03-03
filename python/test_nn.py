"""End-to-end tests for the neural network modules.

Tests: model.py (GeoTransformer), orchestrate.py, train.py, evaluate.py
"""

import sys
import os
import time

# Ensure python/ is on the path
sys.path.insert(0, os.path.dirname(__file__))

import torch
import geoprover
from model import (
    GeoNet, GeoTransformer, GeoNetCNN, SetGeoTransformer, SetGeoTransformerV2,
    construction_to_index, tokenize, tokenize_and_pad,
    build_valid_mask, constructions_to_policy_target,
    count_parameters, create_model, POLICY_SIZE, VOCAB_SIZE, MAX_SEQ_LEN,
    PAD_ID, CLS_ID, GOAL_ID, SEP_ID,
    CONSTRUCTION_TYPES, TOKEN_TO_ID, POINT_NAMES, POINT_NAME_SET,
    make_augmentation_perm, permute_text, shuffle_facts,
    augment_state_text, permute_point_ids,
    encode_state_as_set, split_state_text, SET_MAX_TOKENS,
)


def test_tokenizer():
    """Test the custom tokenizer."""
    text = "coll a b c ; para a b c d ; ? perp a h b c"
    ids = tokenize(text)
    assert ids[0] == CLS_ID, f"First token should be CLS, got {ids[0]}"
    assert len(ids) > 5, f"Should have multiple tokens, got {len(ids)}"
    # Check known tokens are in vocabulary
    for token in ["coll", "para", "perp", "a", "b", "c", "d", "h", ";", "?"]:
        assert token in TOKEN_TO_ID, f"Token '{token}' not in vocabulary"
    print(f"  Tokenized {len(text.split())} words -> {len(ids)} token IDs")
    print(f"  VOCAB_SIZE = {VOCAB_SIZE}")
    print("  PASS: tokenizer")


def test_tokenize_and_pad():
    """Test tokenize + padding."""
    text = "coll a b c"
    t = tokenize_and_pad(text, max_len=32)
    assert t.shape == (32,), f"Shape should be (32,), got {t.shape}"
    assert t.dtype == torch.long
    assert t[0].item() == CLS_ID
    # Rest should be padded
    ids = tokenize(text)
    for i in range(len(ids), 32):
        assert t[i].item() == PAD_ID, f"Position {i} should be PAD"
    print("  PASS: tokenize_and_pad")


def test_model_architecture():
    """Test GeoTransformer can be instantiated and has ~5-10M params."""
    model = GeoTransformer()
    params = count_parameters(model)
    print(f"  GeoTransformer parameters: {params:,}")
    assert 3_000_000 < params < 15_000_000, f"Expected ~5-10M params, got {params:,}"
    # GeoNet should be an alias
    assert GeoNet is GeoTransformer
    print("  PASS: model architecture")


def test_forward_pass():
    """Test forward pass with tokenized input."""
    model = GeoTransformer()
    # Create a batch of 2 token sequences
    text1 = "coll a b c ; para a b c d"
    text2 = "cong a b c d ; ? perp a h b c"
    t1 = tokenize_and_pad(text1, max_len=64)
    t2 = tokenize_and_pad(text2, max_len=64)
    batch = torch.stack([t1, t2])

    value, policy = model(batch)
    assert value.shape == (2,), f"value shape: {value.shape}"
    assert policy.shape == (2, POLICY_SIZE), f"policy shape: {policy.shape}"
    # Value should be in [0, 1] (sigmoid output)
    assert (value >= 0).all() and (value <= 1).all(), f"Value out of [0,1]: {value}"
    print(f"  value={value[0].item():.4f}")
    print("  PASS: forward pass")


def test_forward_with_mask():
    """Test forward pass with valid_mask."""
    model = GeoTransformer()
    text = "coll a b c"
    t = tokenize_and_pad(text, max_len=64).unsqueeze(0)
    mask = torch.zeros(1, POLICY_SIZE, dtype=torch.bool)
    mask[0, [0, 10, 73, 146]] = True
    value, policy = model(t, mask)
    assert policy[0, 1].item() == float("-inf"), "Masked position should be -inf"
    assert policy[0, 0].item() != float("-inf"), "Valid position should not be -inf"
    print("  PASS: forward with mask")


def test_predict_single():
    """Test single-state prediction."""
    model = GeoTransformer()
    text = "coll a b c ; cong a b c d"
    t = tokenize_and_pad(text, max_len=64)
    mask = torch.zeros(POLICY_SIZE, dtype=torch.bool)
    mask[[0, 73, 146]] = True
    value, policy = model.predict(t, mask)
    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0, f"Value should be in [0,1], got {value}"
    assert policy.shape == (POLICY_SIZE,)
    valid_sum = policy[[0, 73, 146]].sum().item()
    assert abs(valid_sum - 1.0) < 0.01, f"Policy sum over valid: {valid_sum}"
    print(f"  value={value:.4f}, policy_sum={valid_sum:.4f}")
    print("  PASS: predict single")


def test_value_range():
    """Test that value output is in [0, 1] via sigmoid."""
    model = GeoTransformer()
    texts = [
        "coll a b c",
        "cong a b c d ; para a b c d ; perp a c b d ; ? eqangle a b c d e f",
    ]
    for text in texts:
        t = tokenize_and_pad(text, max_len=64)
        value, _ = model.predict(t)
        assert 0.0 <= value <= 1.0, f"Value should be in [0,1], got {value}"
    print(f"  All values in [0, 1]")
    print("  PASS: value range")


def test_construction_to_index():
    """Test construction -> policy index mapping."""
    idx = construction_to_index("Midpoint", [0, 1])
    assert 0 <= idx < POLICY_SIZE
    idx2 = construction_to_index("Midpoint", [1, 0])
    assert idx == idx2, f"Order shouldn't matter: {idx} != {idx2}"
    idx_alt = construction_to_index("Altitude", [0, 1, 2])
    assert idx_alt != idx, "Different types should map to different indices"
    print(f"  Midpoint(0,1) -> {idx}, Altitude(0,1,2) -> {idx_alt}")
    print("  PASS: construction_to_index")


def test_state_to_text():
    """Test PyO3 state_to_text function."""
    state = geoprover.parse_problem(
        "orthocenter\n"
        "a b c = triangle; h = on_tline b a c, on_tline c a b ? perp a h b c"
    )
    text = geoprover.state_to_text(state)
    assert "perp" in text, f"Should contain 'perp': {text}"
    assert "?" in text, f"Should contain goal marker: {text}"
    # Should be tokenizable
    ids = tokenize(text)
    assert len(ids) > 3, f"Should produce multiple tokens: {ids}"
    print(f"  State text: {text[:80]}...")
    print(f"  Tokens: {len(ids)}")
    print("  PASS: state_to_text")


def test_construction_to_text():
    """Test PyO3 construction_to_text function."""
    state = geoprover.parse_problem("test\na b c = triangle ? perp a b b c")
    cs = geoprover.generate_constructions(state)
    assert len(cs) > 0
    ct = geoprover.construction_to_text(cs[0], state)
    assert len(ct) > 0, "Construction text should not be empty"
    # Should be tokenizable
    ids = tokenize(ct)
    assert len(ids) > 1
    print(f"  Construction text: {ct}")
    print("  PASS: construction_to_text")


def test_synthetic_data():
    """Test Rust synthetic data generation."""
    data = geoprover.generate_synthetic_data(10, 42)
    assert len(data) > 0, "Should generate some examples"
    for state, construction, goal in data:
        assert len(state) > 0, "State should not be empty"
        assert len(construction) > 0, "Construction should not be empty"
        assert len(goal) > 0, "Goal should not be empty"
    print(f"  Generated {len(data)} synthetic examples")
    print(f"  Example: state={data[0][0][:60]}..., constr={data[0][1]}, goal={data[0][2]}")
    print("  PASS: synthetic_data")


def test_integration_with_geoprover():
    """Test full integration: parse -> text -> model -> predict."""
    model = GeoTransformer()

    state = geoprover.parse_problem(
        "orthocenter\n"
        "a b c = triangle; h = on_tline b a c, on_tline c a b ? perp a h b c"
    )

    # Encode as text and tokenize
    state_text = geoprover.state_to_text(state)
    token_ids = tokenize_and_pad(state_text, max_len=128)

    # Generate constructions and build mask
    constructions = geoprover.generate_constructions(state)
    mask = build_valid_mask(constructions)

    # Predict
    value, policy = model.predict(token_ids, mask)
    assert 0.0 <= value <= 1.0, f"Value should be in [0,1], got {value}"

    print(f"  Problem: orthocenter")
    print(f"  Objects: {state.num_objects()}, Facts: {state.num_facts()}")
    print(f"  Text tokens: {(token_ids != PAD_ID).sum().item()}")
    print(f"  Constructions: {len(constructions)}")
    print(f"  Value: {value:.4f}")
    print("  PASS: integration with geoprover")


def test_training_step():
    """Test a single training step with text input."""
    model = GeoTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 4
    max_len = 64
    # Create text inputs
    texts = [
        "coll a b c ; cong a b c d",
        "para a b c d ; perp a c b d",
        "mid m a b ; cong a m m b",
        "eqangle a b c d e f",
    ]
    token_ids = torch.stack([tokenize_and_pad(t, max_len=max_len) for t in texts])
    policy_targets = torch.zeros(batch_size, POLICY_SIZE)
    for i in range(batch_size):
        active = torch.randperm(POLICY_SIZE)[:5]
        policy_targets[i, active] = torch.rand(5)
        policy_targets[i] /= policy_targets[i].sum()
    value_targets = torch.rand(batch_size)

    from train import compute_loss
    optimizer.zero_grad()
    loss, metrics = compute_loss(model, token_ids, policy_targets, value_targets)
    loss.backward()
    optimizer.step()

    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Policy loss: {metrics['policy_loss']:.4f}")
    print(f"  Value loss: {metrics['value_loss']:.4f}")
    assert metrics['loss'] > 0
    print("  PASS: training step")


def test_supervised_data_small():
    """Test supervised data generation on a small set."""
    from train import generate_supervised_data
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("orthocenter\n")
        f.write("a b c = triangle; h = on_tline b a c, on_tline c a b ? perp a h b c\n")
        f.write("midpoint_test\n")
        f.write("a b = segment; m = midpoint m a b ? cong a m m b\n")
        path = f.name
    try:
        samples = generate_supervised_data(path)
        print(f"  Generated {len(samples)} samples")
        assert len(samples) >= 1, "Should generate at least 1 sample"
        s = samples[0]
        assert isinstance(s.state_text, str)
        assert len(s.policy_target) == POLICY_SIZE
        assert 0.0 <= s.value_target <= 1.0
        print("  PASS: supervised data generation")
    finally:
        os.unlink(path)


def test_mcts_search_basic():
    """Test NN-guided MCTS search on a simple problem."""
    from orchestrate import MctsConfig, mcts_search

    model = GeoTransformer()
    state = geoprover.parse_problem(
        "orthocenter\n"
        "a b c = triangle; h = on_tline b a c, on_tline c a b ? perp a h b c"
    )
    proved = geoprover.saturate(state)
    if proved:
        print("  Problem solved by deduction (expected)")

    # Test MCTS on a harder problem
    state2 = geoprover.parse_problem("test\na b c = triangle ? perp a b b c")
    config = MctsConfig(num_iterations=5, max_children=10, max_depth=2, max_seq_len=128)
    result = mcts_search(state2, model, config)
    print(f"  MCTS result: solved={result.solved}, value={result.best_value:.4f}, "
          f"visits={result.root_visits}, samples={len(result.samples)}, "
          f"time={result.elapsed_ms:.0f}ms")
    assert result.root_visits > 0
    print("  PASS: MCTS search basic")


def test_synthetic_dataset():
    """Test SyntheticDataset for training."""
    from train import SyntheticDataset

    data = geoprover.generate_synthetic_data(5, 42)
    dataset = SyntheticDataset(data, max_seq_len=64)
    assert len(dataset) == len(data)

    token_ids, policy, value = dataset[0]
    assert token_ids.shape == (64,)
    assert token_ids.dtype == torch.long
    assert policy.shape == (POLICY_SIZE,)
    assert policy.sum().item() > 0, "Policy should have at least one non-zero entry"
    from orchestrate import GAMMA
    assert abs(value.item() - GAMMA) < 1e-6, f"Value should be {GAMMA} (1 step from proof), got {value.item()}"
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Token IDs shape: {token_ids.shape}")
    print(f"  Policy nonzero: {(policy > 0).sum().item()}")
    print("  PASS: synthetic dataset")


def test_legacy_cnn():
    """Test that the legacy CNN model still works."""
    model = GeoNetCNN()
    params = count_parameters(model)
    print(f"  GeoNetCNN parameters: {params:,}")
    x = torch.randn(1, 20, 32, 32)
    # Legacy CNN still outputs (v_logit, k, policy) — kept for ablation
    v_logit, k, policy = model(x)
    assert v_logit.shape == (1,)
    assert k.shape == (1,)
    assert policy.shape == (1, POLICY_SIZE)
    print("  PASS: legacy CNN")


def test_permute_text():
    """Test that permute_text changes point names but preserves keywords and structure."""
    perm = {"a": "c", "b": "d", "c": "a", "d": "b", "h": "h"}
    text = "coll a b c ; para a b c d ; ? perp a h b c"
    result = permute_text(text, perm)
    tokens = result.split()
    # Keywords should be unchanged
    assert "coll" in tokens
    assert "para" in tokens
    assert "perp" in tokens
    assert ";" in tokens
    assert "?" in tokens
    # Point names should be permuted
    assert result == "coll c d a ; para c d a b ; ? perp c h d a"
    print(f"  Original: {text}")
    print(f"  Permuted: {result}")
    print("  PASS: permute_text")


def test_shuffle_facts():
    """Test that shuffle_facts reorders facts but preserves goal at end."""
    text = "coll a b c ; para a b c d ; cong a b c d ; ? perp a h b c"
    import random
    rng = random.Random(42)
    result = shuffle_facts(text, rng)
    # Goal should still be at the end
    assert "; ? perp a h b c" in result or "? perp a h b c" in result, \
        f"Goal missing from: {result}"
    # All facts should still be present (as a set)
    orig_facts = {"coll a b c", "para a b c d", "cong a b c d"}
    # Extract facts from result (everything before " ; ? ")
    facts_part = result.split(" ; ? ")[0]
    result_facts = set(facts_part.split(" ; "))
    assert result_facts == orig_facts, f"Facts differ: {result_facts} != {orig_facts}"
    print(f"  Original: {text}")
    print(f"  Shuffled: {result}")
    print("  PASS: shuffle_facts")


def test_permute_policy_reconstruction():
    """Test that policy index changes consistently with text permutation."""
    # Original construction: Midpoint of a(0) and b(1)
    orig_args = [0, 1]  # a=0, b=1
    orig_idx = construction_to_index("Midpoint", orig_args)

    # Permutation swaps a<->c: a→c(2), b→d(3)
    perm = {"a": "c", "b": "d", "c": "a", "d": "b"}
    permuted_args = permute_point_ids(orig_args, perm)
    permuted_idx = construction_to_index("Midpoint", permuted_args)

    # After permutation, args should be [2, 3] (c=2, d=3)
    assert permuted_args == [2, 3], f"Expected [2, 3], got {permuted_args}"
    # Index should be different (different args)
    assert orig_idx != permuted_idx, f"Indices should differ: {orig_idx} == {permuted_idx}"
    # But if we manually compute Midpoint([2,3]), we should get permuted_idx
    manual_idx = construction_to_index("Midpoint", [2, 3])
    assert permuted_idx == manual_idx, f"Mismatch: {permuted_idx} != {manual_idx}"
    print(f"  Original: Midpoint([0,1]) -> idx {orig_idx}")
    print(f"  Permuted: Midpoint([2,3]) -> idx {permuted_idx}")
    print("  PASS: permute_policy_reconstruction")


def test_augmentation_deterministic():
    """Test that same (epoch, idx) gives same result; different epoch gives different."""
    text = "coll a b c ; para a b c d ; ? perp a h b c"
    aug1, perm1 = augment_state_text(text, epoch=0, idx=5, dataset_len=100)
    aug2, perm2 = augment_state_text(text, epoch=0, idx=5, dataset_len=100)
    assert aug1 == aug2, "Same seed should give same result"
    assert perm1 == perm2, "Same seed should give same perm"

    aug3, perm3 = augment_state_text(text, epoch=1, idx=5, dataset_len=100)
    assert aug3 != aug1, "Different epoch should give different result"
    print(f"  epoch=0: {aug1}")
    print(f"  epoch=1: {aug3}")
    print("  PASS: augmentation deterministic")


def test_augmented_synthetic_dataset():
    """Test that augmented SyntheticDataset produces different tokens per epoch."""
    from train import SyntheticDataset
    data = geoprover.generate_synthetic_data(5, 42)
    if len(data) == 0:
        print("  SKIP: no synthetic data generated")
        return

    ds0 = SyntheticDataset(data, max_seq_len=64, augment=True, epoch=0)
    ds1 = SyntheticDataset(data, max_seq_len=64, augment=True, epoch=1)

    t0, p0, v0 = ds0[0]
    t1, p1, v1 = ds1[0]

    # Value should be the same (same underlying example)
    assert v0.item() == v1.item(), f"Values differ: {v0} vs {v1}"
    # Token IDs should differ (different permutation)
    assert not torch.equal(t0, t1), "Token IDs should differ between epochs"
    print(f"  epoch 0 tokens[:8]: {t0[:8].tolist()}")
    print(f"  epoch 1 tokens[:8]: {t1[:8].tolist()}")
    print("  PASS: augmented synthetic dataset")


def test_summarizer_architecture():
    """Test FactSummarizer model creation and parameter count."""
    from summarizer import FactSummarizer, count_summarizer_parameters
    model = FactSummarizer()
    n_params = count_summarizer_parameters(model)
    assert n_params > 0, "Model should have parameters"
    assert n_params < 5_000_000, f"Model too large: {n_params}"
    print(f"  FactSummarizer params: {n_params:,}")
    print("  PASS: summarizer architecture")


def test_summarizer_forward():
    """Test FactSummarizer forward pass and scoring."""
    from summarizer import (
        FactSummarizer, build_context_tokens, build_fact_tokens,
    )
    model = FactSummarizer()
    model.eval()

    ctx = build_context_tokens(["coll a b c", "para a b c d"], "perp a h b c", max_len=64)
    facts = build_fact_tokens(["cong a b c d", "eqangle a b c d e f"], max_len=16)

    with torch.no_grad():
        scores = model.score_facts(ctx.unsqueeze(0), facts)

    assert scores.shape == (2,), f"Expected (2,), got {scores.shape}"
    assert scores.dtype == torch.float32
    print(f"  Scores: {scores.tolist()}")
    print("  PASS: summarizer forward")


def test_summarizer_training_step():
    """Test one training step of the Summarizer."""
    from summarizer import FactSummarizer, build_context_tokens, build_fact_tokens
    model = FactSummarizer()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    ctx = build_context_tokens(["coll a b c"], "perp a h b c", max_len=64)
    facts = build_fact_tokens(["cong a b c d", "eqangle a b c d e f"], max_len=16)
    labels = torch.tensor([1.0, 0.0])

    model.train()
    scores = model.score_facts(ctx.unsqueeze(0).expand(2, -1), facts)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels)
    loss.backward()
    optimizer.step()

    assert loss.item() > 0, "Loss should be positive"
    print(f"  Loss: {loss.item():.4f}")
    print("  PASS: summarizer training step")


def test_summarizer_filter_facts():
    """Test fact filtering with FactSummarizer."""
    from summarizer import FactSummarizer, filter_facts
    model = FactSummarizer()
    model.eval()

    initial_facts = ["coll a b c", "para a b c d"]
    deduced_facts = ["cong a b c d", "eqangle a b c d e f", "perp a b c d",
                     "mid a b c", "oncirc a b c"]
    goal_text = "perp a h b c"

    # K=3: keep top 3 of 5
    result = filter_facts(model, initial_facts, deduced_facts, goal_text, k=3)
    assert len(result) == 3, f"Expected 3 filtered facts, got {len(result)}"
    assert all(f in deduced_facts for f in result), "Filtered facts should be from deduced"
    print(f"  Filtered: {result}")
    print("  PASS: summarizer filter facts")


def test_summarizer_data_generation():
    """Test Summarizer training data generation on a real problem."""
    # Use facts_as_text_list and proof_path_facts directly
    state = geoprover.parse_problem(
        'test\na b c = triangle a b c; o = circle o a b c; '
        'h = midpoint h c b; d = on_line d o h, on_line d a b; '
        'e = on_tline e c c o, on_tline e a a o ? cyclic a o e d'
    )
    pre_facts = set(state.facts_as_text_list())
    assert len(pre_facts) > 0, "Should have initial facts"

    proved, trace = geoprover.saturate_with_trace(state)
    assert proved, "Problem should be solvable"

    post_facts = set(state.facts_as_text_list())
    deduced = post_facts - pre_facts
    assert len(deduced) > 0, "Should have deduced facts"

    proof_path = trace.proof_path_facts()
    assert proof_path is not None, "Should have proof path"
    assert len(proof_path) > 0, "Proof path should be non-empty"

    axioms = trace.axiom_facts()
    assert len(axioms) > 0, "Should have axioms"

    print(f"  Initial: {len(pre_facts)}, Deduced: {len(deduced)}, "
          f"Proof path: {len(proof_path)}, Axioms: {len(axioms)}")
    print("  PASS: summarizer data generation")


def test_build_summarized_text():
    """Test compact text building from filtered facts."""
    from summarizer import build_summarized_text
    result = build_summarized_text(
        initial_facts=["coll a b c", "para a b c d"],
        filtered_deduced=["cong a b c d"],
        goal_text="perp a h b c",
    )
    assert "coll a b c" in result
    assert "para a b c d" in result
    assert "cong a b c d" in result
    assert "? perp a h b c" in result
    print(f"  Text: {result}")
    print("  PASS: build summarized text")


def test_split_state_text():
    """Test split_state_text parses facts and goal correctly."""
    facts, goal = split_state_text("coll a b c ; para a b c d ; ? perp a h b c")
    assert facts == ["coll a b c", "para a b c d"], f"Facts: {facts}"
    assert goal == "perp a h b c", f"Goal: {goal}"

    # No goal marker
    facts2, goal2 = split_state_text("coll a b c ; para a b c d")
    assert facts2 == ["coll a b c", "para a b c d"]
    assert goal2 == ""

    # Single fact with goal
    facts3, goal3 = split_state_text("coll a b c ; ? perp a b c d")
    assert len(facts3) == 1
    assert goal3 == "perp a b c d"
    print("  PASS: split_state_text")


def test_encode_state_as_set():
    """Test encode_state_as_set produces correct shapes."""
    facts = ["coll a b c", "para a b c d", "cong a b c d"]
    goal = "perp a h b c"
    fact_ids, goal_ids, fact_mask = encode_state_as_set(facts, goal)

    assert fact_ids.shape == (3, SET_MAX_TOKENS), f"fact_ids shape: {fact_ids.shape}"
    assert goal_ids.shape == (SET_MAX_TOKENS,), f"goal_ids shape: {goal_ids.shape}"
    assert fact_mask.shape == (3,), f"fact_mask shape: {fact_mask.shape}"
    assert fact_mask.all(), "All facts should be valid"
    assert fact_ids.dtype == torch.long
    assert goal_ids.dtype == torch.long

    # Empty facts should produce dummy
    fact_ids_e, goal_ids_e, fact_mask_e = encode_state_as_set([], goal)
    assert fact_ids_e.shape[0] == 1, "Should have 1 dummy fact"
    assert not fact_mask_e[0], "Dummy fact should be masked out"
    print(f"  fact_ids shape: {fact_ids.shape}")
    print("  PASS: encode_state_as_set")


def test_set_model_architecture():
    """Test SetGeoTransformer instantiation and parameter count."""
    model = SetGeoTransformer()
    params = count_parameters(model)
    print(f"  SetGeoTransformer parameters: {params:,}")
    assert 1_000_000 < params < 15_000_000, f"Expected ~3-8M params, got {params:,}"
    print("  PASS: set model architecture")


def test_set_forward_pass():
    """Test SetGeoTransformer forward pass with variable fact counts."""
    model = SetGeoTransformer()
    model.eval()

    # Batch of 2 states with different fact counts (padded to max)
    facts1 = ["coll a b c", "para a b c d"]
    facts2 = ["cong a b c d", "perp a c b d", "mid m a b"]
    goal1 = "perp a h b c"
    goal2 = "cong a b c d"

    fi1, gi1, fm1 = encode_state_as_set(facts1, goal1)
    fi2, gi2, fm2 = encode_state_as_set(facts2, goal2)

    # Pad to same N
    max_n = max(fi1.shape[0], fi2.shape[0])
    L = fi1.shape[1]

    padded_facts = torch.zeros(2, max_n, L, dtype=torch.long)
    padded_masks = torch.zeros(2, max_n, dtype=torch.bool)

    padded_facts[0, :fi1.shape[0]] = fi1
    padded_masks[0, :fi1.shape[0]] = fm1
    padded_facts[1, :fi2.shape[0]] = fi2
    padded_masks[1, :fi2.shape[0]] = fm2

    goal_ids = torch.stack([gi1, gi2])

    with torch.no_grad():
        value, policy = model(padded_facts, goal_ids, padded_masks)

    assert value.shape == (2,), f"value shape: {value.shape}"
    assert policy.shape == (2, POLICY_SIZE), f"policy shape: {policy.shape}"
    assert (value >= 0).all() and (value <= 1).all(), f"Value out of [0,1]: {value}"
    print(f"  value={value[0].item():.4f}, {value[1].item():.4f}")
    print("  PASS: set forward pass")


def test_set_permutation_invariance():
    """Test that SetGeoTransformer produces same output regardless of fact order."""
    model = SetGeoTransformer()
    model.eval()

    facts = ["coll a b c", "para a b c d", "cong a b c d", "perp a c b d"]
    goal = "eqangle a b c d e f"

    # Order 1: original
    fi1, gi1, fm1 = encode_state_as_set(facts, goal)

    # Order 2: reversed
    fi2, gi2, fm2 = encode_state_as_set(list(reversed(facts)), goal)

    # Order 3: shuffled
    import random
    rng = random.Random(42)
    shuffled = list(facts)
    rng.shuffle(shuffled)
    fi3, gi3, fm3 = encode_state_as_set(shuffled, goal)

    with torch.no_grad():
        v1, p1 = model(fi1.unsqueeze(0), gi1.unsqueeze(0), fm1.unsqueeze(0))
        v2, p2 = model(fi2.unsqueeze(0), gi2.unsqueeze(0), fm2.unsqueeze(0))
        v3, p3 = model(fi3.unsqueeze(0), gi3.unsqueeze(0), fm3.unsqueeze(0))

    # Values should be identical (permutation invariant)
    assert abs(v1.item() - v2.item()) < 1e-5, \
        f"Values differ: {v1.item()} vs {v2.item()}"
    assert abs(v1.item() - v3.item()) < 1e-5, \
        f"Values differ: {v1.item()} vs {v3.item()}"

    # Policy logits should be identical
    assert torch.allclose(p1, p2, atol=1e-4), \
        f"Policies differ (orig vs reversed): max diff={torch.max(torch.abs(p1 - p2)).item()}"
    assert torch.allclose(p1, p3, atol=1e-4), \
        f"Policies differ (orig vs shuffled): max diff={torch.max(torch.abs(p1 - p3)).item()}"

    print(f"  v_orig={v1.item():.6f}, v_rev={v2.item():.6f}, v_shuf={v3.item():.6f}")
    print(f"  max policy diff (orig vs rev): {torch.max(torch.abs(p1 - p2)).item():.8f}")
    print("  PASS: set permutation invariance")


def test_set_predict_single():
    """Test SetGeoTransformer single-state prediction."""
    model = SetGeoTransformer()
    facts = ["coll a b c", "cong a b c d"]
    goal = "perp a h b c"
    fact_ids, goal_ids, fact_mask = encode_state_as_set(facts, goal)
    mask = torch.zeros(POLICY_SIZE, dtype=torch.bool)
    mask[[0, 73, 146]] = True

    value, policy = model.predict(fact_ids, goal_ids, fact_mask, mask)
    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0
    assert policy.shape == (POLICY_SIZE,)
    valid_sum = policy[[0, 73, 146]].sum().item()
    assert abs(valid_sum - 1.0) < 0.01, f"Policy sum over valid: {valid_sum}"
    print(f"  value={value:.4f}, policy_sum={valid_sum:.4f}")
    print("  PASS: set predict single")


def test_create_model_factory():
    """Test create_model factory function."""
    m1 = create_model("transformer")
    assert isinstance(m1, GeoTransformer)
    m2 = create_model("set")
    assert isinstance(m2, SetGeoTransformer)
    m3 = create_model("set_v2")
    assert isinstance(m3, SetGeoTransformerV2)
    try:
        create_model("invalid")
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    print("  PASS: create_model factory")


def test_set_training_step():
    """Test a single training step with SetGeoTransformer."""
    from train import compute_set_loss
    model = SetGeoTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    facts = [["coll a b c", "cong a b c d"], ["para a b c d", "perp a c b d"]]
    goals = ["perp a h b c", "cong a b c d"]
    B = len(facts)

    # Encode and pad
    encoded = [encode_state_as_set(f, g) for f, g in zip(facts, goals)]
    max_n = max(e[0].shape[0] for e in encoded)
    L = encoded[0][0].shape[1]

    padded_facts = torch.zeros(B, max_n, L, dtype=torch.long)
    padded_masks = torch.zeros(B, max_n, dtype=torch.bool)
    for i, (fi, gi, fm) in enumerate(encoded):
        padded_facts[i, :fi.shape[0]] = fi
        padded_masks[i, :fi.shape[0]] = fm
    goal_ids = torch.stack([e[1] for e in encoded])

    policy_targets = torch.zeros(B, POLICY_SIZE)
    for i in range(B):
        active = torch.randperm(POLICY_SIZE)[:5]
        policy_targets[i, active] = torch.rand(5)
        policy_targets[i] /= policy_targets[i].sum()
    value_targets = torch.rand(B)

    optimizer.zero_grad()
    loss, metrics = compute_set_loss(
        model, padded_facts, goal_ids, padded_masks,
        policy_targets, value_targets,
    )
    loss.backward()
    optimizer.step()

    assert metrics['loss'] > 0
    print(f"  Loss: {metrics['loss']:.4f}")
    print("  PASS: set training step")


def test_v2_architecture():
    """Test SetGeoTransformerV2 instantiation and parameter count."""
    model = SetGeoTransformerV2()
    params = count_parameters(model)
    print(f"  SetGeoTransformerV2 parameters: {params:,}")
    assert 1_000_000 < params < 5_000_000, f"Expected ~2.5M params, got {params:,}"
    print("  PASS: v2 architecture")


def test_v2_forward_pass():
    """Test V2 forward pass with variable fact counts and construction counts."""
    model = SetGeoTransformerV2()
    model.eval()

    B = 2
    # Facts
    facts1 = ["coll a b c", "para a b c d"]
    facts2 = ["cong a b c d", "perp a c b d", "mid m a b"]
    goal1 = "perp a h b c"
    goal2 = "cong a b c d"

    fi1, gi1, fm1 = encode_state_as_set(facts1, goal1)
    fi2, gi2, fm2 = encode_state_as_set(facts2, goal2)

    max_n = max(fi1.shape[0], fi2.shape[0])
    L = fi1.shape[1]

    padded_facts = torch.zeros(B, max_n, L, dtype=torch.long)
    padded_masks = torch.zeros(B, max_n, dtype=torch.bool)
    padded_facts[0, :fi1.shape[0]] = fi1
    padded_masks[0, :fi1.shape[0]] = fm1
    padded_facts[1, :fi2.shape[0]] = fi2
    padded_masks[1, :fi2.shape[0]] = fm2
    goal_ids = torch.stack([gi1, gi2])

    # Constructions: 2 per sample
    from model import tokenize_statement, SET_MAX_TOKENS
    c1a = torch.tensor([tokenize_statement("mid a b")], dtype=torch.long)
    c1b = torch.tensor([tokenize_statement("alt c a b")], dtype=torch.long)
    c2a = torch.tensor([tokenize_statement("circumcenter a b c")], dtype=torch.long)
    c2b = torch.tensor([tokenize_statement("mid b c")], dtype=torch.long)

    K = 2
    constr_ids = torch.zeros(B, K, SET_MAX_TOKENS, dtype=torch.long)
    constr_ids[0, 0] = c1a
    constr_ids[0, 1] = c1b
    constr_ids[1, 0] = c2a
    constr_ids[1, 1] = c2b
    constr_mask = torch.ones(B, K, dtype=torch.bool)

    with torch.no_grad():
        value, logits = model(padded_facts, goal_ids, padded_masks, constr_ids, constr_mask)

    assert value.shape == (B,), f"value shape: {value.shape}"
    assert logits.shape == (B, K), f"logits shape: {logits.shape}"
    assert (value >= 0).all() and (value <= 1).all(), f"Value out of [0,1]: {value}"
    assert not torch.isnan(logits).any(), "Logits contain NaN"
    print(f"  value={value[0].item():.4f}, {value[1].item():.4f}")
    print(f"  logits={logits[0].tolist()}")
    print("  PASS: v2 forward pass")


def test_v2_permutation_invariance():
    """Test V2 produces same output regardless of fact order."""
    model = SetGeoTransformerV2()
    model.eval()

    facts = ["coll a b c", "para a b c d", "cong a b c d", "perp a c b d"]
    goal = "eqangle a b c d e f"
    from model import tokenize_statement, SET_MAX_TOKENS

    # Two constructions
    K = 2
    constr_ids = torch.zeros(1, K, SET_MAX_TOKENS, dtype=torch.long)
    constr_ids[0, 0] = torch.tensor(tokenize_statement("mid a b"), dtype=torch.long)
    constr_ids[0, 1] = torch.tensor(tokenize_statement("alt c a b"), dtype=torch.long)
    constr_mask = torch.ones(1, K, dtype=torch.bool)

    # Order 1: original
    fi1, gi1, fm1 = encode_state_as_set(facts, goal)
    # Order 2: reversed
    fi2, gi2, fm2 = encode_state_as_set(list(reversed(facts)), goal)
    # Order 3: shuffled
    import random
    rng = random.Random(42)
    shuffled = list(facts)
    rng.shuffle(shuffled)
    fi3, gi3, fm3 = encode_state_as_set(shuffled, goal)

    with torch.no_grad():
        v1, p1 = model(fi1.unsqueeze(0), gi1.unsqueeze(0), fm1.unsqueeze(0), constr_ids, constr_mask)
        v2, p2 = model(fi2.unsqueeze(0), gi2.unsqueeze(0), fm2.unsqueeze(0), constr_ids, constr_mask)
        v3, p3 = model(fi3.unsqueeze(0), gi3.unsqueeze(0), fm3.unsqueeze(0), constr_ids, constr_mask)

    assert abs(v1.item() - v2.item()) < 1e-5, \
        f"Values differ: {v1.item()} vs {v2.item()}"
    assert abs(v1.item() - v3.item()) < 1e-5, \
        f"Values differ: {v1.item()} vs {v3.item()}"
    assert torch.allclose(p1, p2, atol=1e-4), \
        f"Logits differ (orig vs rev): max diff={torch.max(torch.abs(p1 - p2)).item()}"
    assert torch.allclose(p1, p3, atol=1e-4), \
        f"Logits differ (orig vs shuf): max diff={torch.max(torch.abs(p1 - p3)).item()}"

    print(f"  v_orig={v1.item():.6f}, v_rev={v2.item():.6f}, v_shuf={v3.item():.6f}")
    print(f"  max logit diff (orig vs rev): {torch.max(torch.abs(p1 - p2)).item():.8f}")
    print("  PASS: v2 permutation invariance")


def test_v2_kv_cache_consistency():
    """Test that cached expand/evaluate matches full forward."""
    model = SetGeoTransformerV2()
    model.eval()

    facts = ["coll a b c", "para a b c d", "cong a b c d"]
    goal = "perp a h b c"
    from model import tokenize_statement, SET_MAX_TOKENS

    fi, gi, fm = encode_state_as_set(facts, goal)

    constr_texts = ["mid a b", "alt c a b"]
    K = len(constr_texts)
    constr_ids = torch.zeros(1, K, SET_MAX_TOKENS, dtype=torch.long)
    for i, ct in enumerate(constr_texts):
        constr_ids[0, i] = torch.tensor(tokenize_statement(ct), dtype=torch.long)
    constr_mask = torch.ones(1, K, dtype=torch.bool)

    with torch.no_grad():
        # Full forward
        value_full, logits_full = model(
            fi.unsqueeze(0), gi.unsqueeze(0), fm.unsqueeze(0),
            constr_ids, constr_mask,
        )

        # Cached path
        fact_kv = model.encode_facts(fi.unsqueeze(0), fm.unsqueeze(0)).squeeze(0)  # (N, D)
        goal_emb = model.encode_goal(gi.unsqueeze(0)).squeeze(0)  # (D,)

        constr_ids_flat = constr_ids.squeeze(0)  # (K, L)
        logits_cached = model.score_constructions_cached(fact_kv, fm, goal_emb, constr_ids_flat)

        value_cached = model.evaluate_cached(fact_kv, fm, goal_emb)

    # Logits should match
    assert torch.allclose(logits_full.squeeze(0), logits_cached, atol=1e-4), \
        f"Logits differ: {logits_full.squeeze(0)} vs {logits_cached}"
    # Values should match
    assert abs(value_full.item() - value_cached) < 1e-4, \
        f"Values differ: {value_full.item()} vs {value_cached}"

    print(f"  full logits: {logits_full.squeeze(0).tolist()}")
    print(f"  cached logits: {logits_cached.tolist()}")
    print(f"  full value: {value_full.item():.6f}, cached value: {value_cached:.6f}")
    print("  PASS: v2 kv cache consistency")


def test_v2_training_step():
    """Test loss backward works for V2."""
    from train import compute_v2_loss
    model = SetGeoTransformerV2()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    from model import tokenize_statement, SET_MAX_TOKENS
    B = 2
    facts = [["coll a b c", "cong a b c d"], ["para a b c d", "perp a c b d"]]
    goals = ["perp a h b c", "cong a b c d"]
    constr_texts = [["mid a b", "alt c a b"], ["circumcenter a b c", "mid b c"]]

    encoded = [encode_state_as_set(f, g) for f, g in zip(facts, goals)]
    max_n = max(e[0].shape[0] for e in encoded)
    L = encoded[0][0].shape[1]

    padded_facts = torch.zeros(B, max_n, L, dtype=torch.long)
    padded_masks = torch.zeros(B, max_n, dtype=torch.bool)
    for i, (fi, gi, fm) in enumerate(encoded):
        padded_facts[i, :fi.shape[0]] = fi
        padded_masks[i, :fi.shape[0]] = fm
    goal_ids = torch.stack([e[1] for e in encoded])

    K = 2
    constr_ids = torch.zeros(B, K, SET_MAX_TOKENS, dtype=torch.long)
    constr_mask = torch.ones(B, K, dtype=torch.bool)
    for i, cts in enumerate(constr_texts):
        for j, ct in enumerate(cts):
            constr_ids[i, j] = torch.tensor(tokenize_statement(ct), dtype=torch.long)

    # Policy targets: distribution over K constructions
    policy_targets = torch.zeros(B, K)
    policy_targets[0] = torch.tensor([0.7, 0.3])
    policy_targets[1] = torch.tensor([0.4, 0.6])
    value_targets = torch.tensor([0.8, 0.3])

    optimizer.zero_grad()
    loss, metrics = compute_v2_loss(
        model, padded_facts, goal_ids, padded_masks,
        constr_ids, constr_mask, policy_targets, value_targets,
    )
    loss.backward()
    optimizer.step()

    assert metrics['loss'] > 0
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Policy loss: {metrics['policy_loss']:.4f}")
    print(f"  Value loss: {metrics['value_loss']:.4f}")
    print("  PASS: v2 training step")


def test_backprop_discount():
    """Test that _backprop applies gamma discount per level."""
    from orchestrate import MctsNode, _backprop, GAMMA

    # Build 3-node chain: root -> parent -> leaf
    root = MctsNode(state=None)
    parent = MctsNode(state=None, parent=root, depth=1)
    root.children.append(parent)
    leaf = MctsNode(state=None, parent=parent, depth=2)
    parent.children.append(leaf)

    _backprop(leaf, 1.0)

    assert leaf.best_value == 1.0, f"leaf.best_value={leaf.best_value}"
    assert abs(parent.best_value - GAMMA) < 1e-9, f"parent.best_value={parent.best_value}"
    assert abs(root.best_value - GAMMA * GAMMA) < 1e-9, f"root.best_value={root.best_value}"
    assert leaf.visits == 1
    assert parent.visits == 1
    assert root.visits == 1
    print(f"  leaf={leaf.best_value}, parent={parent.best_value}, root={root.best_value}")
    print("  PASS: backprop discount")


def test_backprop_max_semantics():
    """Test that _backprop uses max (not sum) for best_value."""
    from orchestrate import MctsNode, _backprop, GAMMA

    root = MctsNode(state=None)
    leaf = MctsNode(state=None, parent=root, depth=1)
    root.children.append(leaf)

    # First backprop: 0.8 from leaf -> root gets 0.8 * GAMMA
    _backprop(leaf, 0.8)
    assert abs(root.best_value - 0.8 * GAMMA) < 1e-9

    # Second backprop: 0.3 from leaf -> root should keep max
    _backprop(leaf, 0.3)
    assert abs(root.best_value - 0.8 * GAMMA) < 1e-9, \
        f"Should be max, got {root.best_value}"
    # leaf.best_value should be max(0.8, 0.3) = 0.8
    assert abs(leaf.best_value - 0.8) < 1e-9
    assert root.visits == 2
    print("  PASS: backprop max semantics")


def test_find_proof_path_nodes():
    """Test _find_proof_path_nodes returns correct depth mapping."""
    from orchestrate import MctsNode, _find_proof_path_nodes

    root = MctsNode(state=None)
    root.expanded = True
    child_a = MctsNode(state=None, parent=root, depth=1)
    child_b = MctsNode(state=None, parent=root, depth=1)
    root.children = [child_a, child_b]

    # child_b is solved
    child_b.terminal_value = 1.0

    mapping = _find_proof_path_nodes(root)
    assert id(child_b) in mapping, "Solved child should be in mapping"
    assert mapping[id(child_b)] == 0, "Terminal node depth_to_terminal=0"
    assert id(root) in mapping, "Root should be in mapping"
    assert mapping[id(root)] == 1, "Root is 1 step from terminal"
    assert id(child_a) not in mapping, "Non-solving child should not be in mapping"
    print(f"  Mapping: {len(mapping)} nodes")
    print("  PASS: find proof path nodes")


def test_td_value_targets():
    """Test _collect_all_node_samples produces TD-style value targets."""
    from orchestrate import MctsNode, _collect_all_node_samples, GAMMA

    # Build tree: root -> [child_a (solved), child_b (not solved)]
    root = MctsNode(state=None)
    root.visits = 10
    root.expanded = True
    root.best_value = GAMMA  # from solved child

    child_a = MctsNode(state=None, parent=root, depth=1, action_index=0)
    child_a.terminal_value = 1.0
    child_a.visits = 5
    child_a.best_value = 1.0

    child_b = MctsNode(state=None, parent=root, depth=1, action_index=1)
    child_b.visits = 5
    child_b.nn_value = 0.3  # NN estimate for off-path
    child_b.best_value = 0.3

    root.children = [child_a, child_b]

    # Need state_to_text to work - use a real state
    state = geoprover.parse_problem("test\na b c = triangle ? perp a b b c")
    root.state = state
    child_a.state = state
    child_b.state = state

    samples = _collect_all_node_samples(root, solved=True, min_visits=1)
    assert len(samples) >= 1, f"Expected >=1 samples, got {len(samples)}"

    # Root sample: on winning path, 1 step from terminal -> gamma^1
    root_sample = samples[0]
    expected = GAMMA ** 1  # root is 1 step from terminal
    assert abs(root_sample.value_target - expected) < 1e-9, \
        f"Root value_target={root_sample.value_target}, expected {expected}"
    print(f"  Root value_target={root_sample.value_target}")
    print("  PASS: td value targets")


def test_synthetic_value_discount():
    """Test V2SyntheticDataset returns GAMMA for positive, 0.0 for negative."""
    from train import V2SyntheticDataset
    from orchestrate import GAMMA

    data = geoprover.generate_synthetic_data(5, 42)
    if len(data) == 0:
        print("  SKIP: no synthetic data generated")
        return
    dataset = V2SyntheticDataset(data)

    _, _, _, _, value, is_pos = dataset[0]
    assert abs(value.item() - GAMMA) < 1e-6, \
        f"Positive value should be {GAMMA}, got {value.item()}"
    print(f"  Positive value: {value.item()}")
    print("  PASS: synthetic value discount")


def test_v2_construction_scoring():
    """Test individual construction scores are sensible."""
    model = SetGeoTransformerV2()
    model.eval()

    facts = ["coll a b c", "para a b c d"]
    goal = "perp a h b c"
    from model import tokenize_statement, SET_MAX_TOKENS

    fi, gi, fm = encode_state_as_set(facts, goal)

    with torch.no_grad():
        fact_kv = model.encode_facts(fi.unsqueeze(0), fm.unsqueeze(0)).squeeze(0)
        goal_emb = model.encode_goal(gi.unsqueeze(0)).squeeze(0)

        # Score 3 constructions
        constr_texts = ["mid a b", "alt c a b", "circumcenter a b c"]
        constr_ids = torch.stack([
            torch.tensor(tokenize_statement(ct), dtype=torch.long)
            for ct in constr_texts
        ])
        logits = model.score_constructions_cached(fact_kv, fm, goal_emb, constr_ids)

    assert logits.shape == (3,), f"Expected (3,), got {logits.shape}"
    assert not torch.isnan(logits).any(), "Logits contain NaN"
    # Scores should respond to different constructions (not all identical)
    assert not torch.allclose(logits, logits[0].expand(3), atol=1e-6), \
        f"All logits identical: {logits.tolist()}"
    print(f"  Logits: {logits.tolist()}")
    print("  PASS: v2 construction scoring")


# Tests organized by speed.
# Fast tests run in <2s each, slow tests may take minutes.
FAST_TESTS = [
    test_tokenizer,
    test_tokenize_and_pad,
    test_model_architecture,
    test_forward_pass,
    test_forward_with_mask,
    test_predict_single,
    test_value_range,
    test_construction_to_index,
    test_state_to_text,
    test_construction_to_text,
    test_synthetic_data,
    test_integration_with_geoprover,
    test_training_step,
    test_supervised_data_small,
    test_mcts_search_basic,
    test_synthetic_dataset,
    test_legacy_cnn,
    test_permute_text,
    test_shuffle_facts,
    test_permute_policy_reconstruction,
    test_augmentation_deterministic,
    test_augmented_synthetic_dataset,
    test_summarizer_architecture,
    test_summarizer_forward,
    test_summarizer_training_step,
    test_summarizer_filter_facts,
    test_build_summarized_text,
    # SetGeoTransformer tests
    test_split_state_text,
    test_encode_state_as_set,
    test_set_model_architecture,
    test_set_forward_pass,
    test_set_permutation_invariance,
    test_set_predict_single,
    test_create_model_factory,
    test_set_training_step,
    # TD-style value learning tests
    test_backprop_discount,
    test_backprop_max_semantics,
    test_find_proof_path_nodes,
    test_td_value_targets,
    test_synthetic_value_discount,
    # SetGeoTransformerV2 tests
    test_v2_architecture,
    test_v2_forward_pass,
    test_v2_permutation_invariance,
    test_v2_kv_cache_consistency,
    test_v2_training_step,
    test_v2_construction_scoring,
]

SLOW_TESTS = [
    test_summarizer_data_generation,  # ~15 min: saturates complex problem with 8000+ facts
]


def run_tests(tests, label=""):
    passed = 0
    failed = 0
    for test in tests:
        print(f"\n{test.__name__}:")
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Results{label}: {passed} passed, {failed} failed out of {len(tests)}")
    return failed


if __name__ == "__main__":
    include_slow = "--include-slow" in sys.argv

    failed = run_tests(FAST_TESTS)

    if include_slow:
        print(f"\n{'=' * 50}")
        print("Running slow tests...")
        failed += run_tests(SLOW_TESTS, " (slow)")

    if failed == 0:
        print("All tests passed!")
    else:
        sys.exit(1)
