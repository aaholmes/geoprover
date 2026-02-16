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
    CONSTRUCTION_TYPES, TOKEN_TO_ID,
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

    v_logit, k, policy = model(batch)
    assert v_logit.shape == (2,), f"v_logit shape: {v_logit.shape}"
    assert k.shape == (2,), f"k shape: {k.shape}"
    assert policy.shape == (2, POLICY_SIZE), f"policy shape: {policy.shape}"
    print(f"  v_logit={v_logit[0].item():.4f}, k={k[0].item():.4f}")
    print("  PASS: forward pass")


def test_forward_with_mask():
    """Test forward pass with valid_mask."""
    model = GeoTransformer()
    text = "coll a b c"
    t = tokenize_and_pad(text, max_len=64).unsqueeze(0)
    mask = torch.zeros(1, POLICY_SIZE, dtype=torch.bool)
    mask[0, [0, 10, 73, 146]] = True
    v_logit, k, policy = model(t, mask)
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
    v_logit, k, policy = model.predict(t, mask)
    assert isinstance(v_logit, float)
    assert isinstance(k, float)
    assert policy.shape == (POLICY_SIZE,)
    valid_sum = policy[[0, 73, 146]].sum().item()
    assert abs(valid_sum - 1.0) < 0.01, f"Policy sum over valid: {valid_sum}"
    print(f"  v_logit={v_logit:.4f}, k={k:.4f}, policy_sum={valid_sum:.4f}")
    print("  PASS: predict single")


def test_compute_value():
    """Test value computation."""
    model = GeoTransformer()
    val = model.compute_value(v_logit=0.0, k=0.5, delta_d=0.5)
    expected = 0.24491866  # tanh(0.25)
    assert abs(val - expected) < 0.01, f"Expected ~{expected}, got {val}"
    val_proved = model.compute_value(v_logit=0.5, k=0.5, delta_d=1.0)
    assert val_proved > 0.7, f"Proved state value: {val_proved}"
    print(f"  V(0.0, 0.5, 0.5) = {val:.4f}")
    print(f"  V(0.5, 0.5, 1.0) = {val_proved:.4f}")
    print("  PASS: compute value")


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
    v_logit, k, policy = model.predict(token_ids, mask)
    delta_d = geoprover.compute_delta_d(state)
    value = model.compute_value(v_logit, k, delta_d)

    print(f"  Problem: orthocenter")
    print(f"  Objects: {state.num_objects()}, Facts: {state.num_facts()}")
    print(f"  Text tokens: {(token_ids != PAD_ID).sum().item()}")
    print(f"  Constructions: {len(constructions)}")
    print(f"  v_logit={v_logit:.4f}, k={k:.4f}, delta_d={delta_d:.4f}")
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
    delta_d = torch.rand(batch_size)

    from train import compute_loss
    optimizer.zero_grad()
    loss, metrics = compute_loss(model, token_ids, policy_targets, value_targets, delta_d)
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

    token_ids, policy, value, delta_d = dataset[0]
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
    v_logit, k, policy = model(x)
    assert v_logit.shape == (1,)
    assert k.shape == (1,)
    assert policy.shape == (1, POLICY_SIZE)
    print("  PASS: legacy CNN")


if __name__ == "__main__":
    tests = [
        test_tokenizer,
        test_tokenize_and_pad,
        test_model_architecture,
        test_forward_pass,
        test_forward_with_mask,
        test_predict_single,
        test_compute_value,
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
