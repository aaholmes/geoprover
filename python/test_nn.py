"""End-to-end tests for the neural network modules.

Tests: model.py, orchestrate.py, train.py, evaluate.py
"""

import sys
import os
import time

# Ensure python/ is on the path
sys.path.insert(0, os.path.dirname(__file__))

import torch
import geoprover
from model import (
    GeoNet, construction_to_index, tensor_from_flat,
    build_valid_mask, constructions_to_policy_target,
    count_parameters, POLICY_SIZE, NUM_CHANNELS, GRID_SIZE,
    CONSTRUCTION_TYPES,
)


def test_model_architecture():
    """Test GeoNet can be instantiated and has ~2M params."""
    model = GeoNet()
    params = count_parameters(model)
    print(f"GeoNet parameters: {params:,}")
    assert 1_500_000 < params < 4_000_000, f"Expected ~2-3M params, got {params:,}"
    print("  PASS: model architecture")


def test_forward_pass():
    """Test forward pass with a random tensor."""
    model = GeoNet()
    x = torch.randn(2, NUM_CHANNELS, GRID_SIZE, GRID_SIZE)
    v_logit, k, policy = model(x)
    assert v_logit.shape == (2,), f"v_logit shape: {v_logit.shape}"
    assert k.shape == (2,), f"k shape: {k.shape}"
    assert policy.shape == (2, POLICY_SIZE), f"policy shape: {policy.shape}"
    print(f"  v_logit={v_logit[0].item():.4f}, k={k[0].item():.4f}")
    print("  PASS: forward pass")


def test_forward_with_mask():
    """Test forward pass with valid_mask."""
    model = GeoNet()
    x = torch.randn(1, NUM_CHANNELS, GRID_SIZE, GRID_SIZE)
    mask = torch.zeros(1, POLICY_SIZE, dtype=torch.bool)
    mask[0, [0, 10, 73, 146]] = True  # 4 valid actions
    v_logit, k, policy = model(x, mask)
    # Masked positions should be -inf
    assert policy[0, 1].item() == float("-inf"), "Masked position should be -inf"
    assert policy[0, 0].item() != float("-inf"), "Valid position should not be -inf"
    print("  PASS: forward with mask")


def test_predict_single():
    """Test single-state prediction."""
    model = GeoNet()
    x = torch.randn(NUM_CHANNELS, GRID_SIZE, GRID_SIZE)
    mask = torch.zeros(POLICY_SIZE, dtype=torch.bool)
    mask[[0, 73, 146]] = True
    v_logit, k, policy = model.predict(x, mask)
    assert isinstance(v_logit, float)
    assert isinstance(k, float)
    assert policy.shape == (POLICY_SIZE,)
    # Policy should sum to ~1.0 over valid entries
    valid_sum = policy[[0, 73, 146]].sum().item()
    assert abs(valid_sum - 1.0) < 0.01, f"Policy sum over valid: {valid_sum}"
    print(f"  v_logit={v_logit:.4f}, k={k:.4f}, policy_sum={valid_sum:.4f}")
    print("  PASS: predict single")


def test_compute_value():
    """Test value computation."""
    model = GeoNet()
    val = model.compute_value(v_logit=0.0, k=0.5, delta_d=0.5)
    expected = 0.24491866  # tanh(0.0 + 0.5 * 0.5) = tanh(0.25)
    assert abs(val - expected) < 0.01, f"Expected ~{expected}, got {val}"
    # Proved state: delta_d = 1.0 should give high value
    val_proved = model.compute_value(v_logit=0.5, k=0.5, delta_d=1.0)
    assert val_proved > 0.7, f"Proved state value: {val_proved}"
    print(f"  V(0.0, 0.5, 0.5) = {val:.4f}")
    print(f"  V(0.5, 0.5, 1.0) = {val_proved:.4f}")
    print("  PASS: compute value")


def test_construction_to_index():
    """Test construction → policy index mapping."""
    idx = construction_to_index("Midpoint", [0, 1])
    assert 0 <= idx < POLICY_SIZE
    # Same args in different order should give same index
    idx2 = construction_to_index("Midpoint", [1, 0])
    assert idx == idx2, f"Order shouldn't matter: {idx} != {idx2}"
    # Different types should give different ranges
    idx_alt = construction_to_index("Altitude", [0, 1, 2])
    assert idx_alt != idx, "Different types should map to different indices"
    assert idx_alt >= 73, "Altitude should be in second slot range"
    print(f"  Midpoint(0,1) → {idx}, Altitude(0,1,2) → {idx_alt}")
    print("  PASS: construction_to_index")


def test_tensor_from_flat():
    """Test converting flat list to tensor."""
    # From a real state
    state = geoprover.parse_problem("test\na b c = triangle ? perp a b b c")
    flat = geoprover.encode_state(state)
    t = tensor_from_flat(flat)
    assert t.shape == (NUM_CHANNELS, GRID_SIZE, GRID_SIZE)
    assert t.dtype == torch.float32
    print(f"  nonzero: {(t != 0).sum().item()}")
    print("  PASS: tensor_from_flat")


def test_build_valid_mask():
    """Test building valid mask from constructions."""
    state = geoprover.parse_problem("test\na b c = triangle ? perp a b b c")
    constructions = geoprover.generate_constructions(state)
    mask = build_valid_mask(constructions)
    assert mask.shape == (POLICY_SIZE,)
    assert mask.sum().item() > 0
    assert mask.sum().item() <= len(constructions)
    print(f"  {len(constructions)} constructions → {mask.sum().item()} valid indices")
    print("  PASS: build_valid_mask")


def test_policy_target():
    """Test converting visit counts to policy target."""
    state = geoprover.parse_problem("test\na b c = triangle ? perp a b b c")
    constructions = geoprover.generate_constructions(state)[:5]
    visits = [10, 5, 3, 1, 1]
    target = constructions_to_policy_target(constructions, visits)
    assert target.shape == (POLICY_SIZE,)
    assert abs(target.sum().item() - 1.0) < 0.01, f"Target sum: {target.sum().item()}"
    print(f"  Target sum: {target.sum().item():.4f}, max: {target.max().item():.4f}")
    print("  PASS: policy target")


def test_integration_with_geoprover():
    """Test full integration: parse → encode → model → predict."""
    model = GeoNet()

    state = geoprover.parse_problem(
        "orthocenter\n"
        "a b c = triangle; h = on_tline b a c, on_tline c a b ? perp a h b c"
    )

    # Encode state
    flat = geoprover.encode_state(state)
    tensor = tensor_from_flat(flat)

    # Generate constructions and build mask
    constructions = geoprover.generate_constructions(state)
    mask = build_valid_mask(constructions)

    # Predict
    v_logit, k, policy = model.predict(tensor, mask)
    delta_d = geoprover.compute_delta_d(state)
    value = model.compute_value(v_logit, k, delta_d)

    print(f"  Problem: orthocenter")
    print(f"  Objects: {state.num_objects()}, Facts: {state.num_facts()}")
    print(f"  Constructions: {len(constructions)}")
    print(f"  v_logit={v_logit:.4f}, k={k:.4f}, delta_d={delta_d:.4f}")
    print(f"  Value: {value:.4f}")

    # Find best construction
    best_idx = policy.argmax().item()
    for c in constructions:
        if construction_to_index(c.construction_type(), c.args()) == best_idx:
            print(f"  Best construction: {c.construction_type()} args={c.args()}")
            break
    print("  PASS: integration with geoprover")


def test_training_step():
    """Test a single training step."""
    model = GeoNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Create a small batch
    batch_size = 4
    tensors = torch.randn(batch_size, NUM_CHANNELS, GRID_SIZE, GRID_SIZE)
    policy_targets = torch.zeros(batch_size, POLICY_SIZE)
    for i in range(batch_size):
        # Random valid policy targets
        active = torch.randperm(POLICY_SIZE)[:5]
        policy_targets[i, active] = torch.rand(5)
        policy_targets[i] /= policy_targets[i].sum()
    value_targets = torch.rand(batch_size)
    delta_d = torch.rand(batch_size)

    # Forward + backward
    import torch.nn.functional as F
    from train import compute_loss
    optimizer.zero_grad()
    loss, metrics = compute_loss(model, tensors, policy_targets, value_targets, delta_d)
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
    # Write a tiny problem file
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
        assert len(s.tensor) == 20 * 32 * 32
        assert len(s.policy_target) == POLICY_SIZE
        assert 0.0 <= s.value_target <= 1.0
        print("  PASS: supervised data generation")
    finally:
        os.unlink(path)


def test_mcts_search_basic():
    """Test NN-guided MCTS search on a simple problem."""
    from orchestrate import MctsConfig, mcts_search

    model = GeoNet()
    state = geoprover.parse_problem(
        "orthocenter\n"
        "a b c = triangle; h = on_tline b a c, on_tline c a b ? perp a h b c"
    )
    # Saturate first (deduction should solve this)
    proved = geoprover.saturate(state)
    if proved:
        print("  Problem solved by deduction (expected)")

    # Test MCTS on a harder problem
    state2 = geoprover.parse_problem("test\na b c = triangle ? perp a b b c")
    config = MctsConfig(num_iterations=5, max_children=10, max_depth=2)
    result = mcts_search(state2, model, config)
    print(f"  MCTS result: solved={result.solved}, value={result.best_value:.4f}, "
          f"visits={result.root_visits}, samples={len(result.samples)}, "
          f"time={result.elapsed_ms:.0f}ms")
    assert result.root_visits > 0
    print("  PASS: MCTS search basic")


if __name__ == "__main__":
    tests = [
        test_model_architecture,
        test_forward_pass,
        test_forward_with_mask,
        test_predict_single,
        test_compute_value,
        test_construction_to_index,
        test_tensor_from_flat,
        test_build_valid_mask,
        test_policy_target,
        test_integration_with_geoprover,
        test_training_step,
        test_supervised_data_small,
        test_mcts_search_basic,
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
