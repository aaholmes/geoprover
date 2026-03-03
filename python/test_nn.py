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
    GeoNet, GeoTransformer, GeoNetCNN,
    construction_to_index, tokenize, tokenize_and_pad,
    build_valid_mask, constructions_to_policy_target,
    count_parameters, POLICY_SIZE, VOCAB_SIZE, MAX_SEQ_LEN,
    PAD_ID, CLS_ID, GOAL_ID, SEP_ID,
    CONSTRUCTION_TYPES, TOKEN_TO_ID, POINT_NAMES, POINT_NAME_SET,
    make_augmentation_perm, permute_text, shuffle_facts,
    augment_state_text, permute_point_ids,
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
    assert value.item() == 1.0, "Value should be 1.0 (provable)"
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


if __name__ == "__main__":
    tests = [
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
        test_summarizer_data_generation,
        test_build_summarized_text,
    ]

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
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed == 0:
        print("All tests passed!")
    else:
        sys.exit(1)
