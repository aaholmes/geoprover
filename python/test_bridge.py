"""Smoke test for the geoprover PyO3 bridge.

Run with: python python/test_bridge.py
"""

import geoprover


def test_version():
    assert geoprover.__version__ == "0.1.0"
    print(f"Version: {geoprover.__version__}")


def test_parse_simple():
    state = geoprover.parse_problem("test\na b c = triangle ? perp a b b c")
    assert state.num_objects() >= 3
    assert not state.is_proved()
    names = state.object_names()
    assert "a" in names
    assert "b" in names
    assert "c" in names
    print(f"Parse simple: objects={state.num_objects()}, facts={state.num_facts()}")
    print(f"  Goal: {state.goal_description()}")
    print(f"  Repr: {repr(state)}")


def test_saturate_simple():
    state = geoprover.parse_problem("test\na b c = triangle ? perp a b b c")
    proved = geoprover.saturate(state)
    # This particular goal (perp a b b c) is not provable from just triangle
    print(f"Simple saturate: proved={proved}, facts={state.num_facts()}")


def test_encode_state():
    state = geoprover.parse_problem("test\na b c = triangle ? perp a b b c")
    tensor = geoprover.encode_state(state)
    assert len(tensor) == 20 * 32 * 32, f"Expected 20480, got {len(tensor)}"
    assert all(isinstance(x, float) for x in tensor[:100])
    nonzero = sum(1 for x in tensor if x > 0)
    print(f"Encode state: tensor_len={len(tensor)}, nonzero={nonzero}")


def test_generate_constructions():
    state = geoprover.parse_problem("test\na b c = triangle ? perp a b b c")
    constructions = geoprover.generate_constructions(state)
    assert len(constructions) > 0
    print(f"Generated {len(constructions)} constructions")
    for c in constructions[:5]:
        print(f"  {repr(c)}")


def test_apply_construction():
    state = geoprover.parse_problem("test\na b c = triangle ? perp a b b c")
    constructions = geoprover.generate_constructions(state)
    assert len(constructions) > 0
    new_state = geoprover.apply_construction(state, constructions[0])
    assert new_state.num_objects() > state.num_objects()
    print(f"Apply construction: {state.num_objects()} -> {new_state.num_objects()} objects")


def test_orthocenter_solvable():
    state = geoprover.parse_problem(
        "orthocenter\n"
        "a b c = triangle; h = on_tline b a c, on_tline c a b ? perp a h b c"
    )
    proved = geoprover.saturate(state)
    assert proved, "Orthocenter problem should be solvable"
    print(f"Orthocenter: proved={proved}, facts={state.num_facts()}")


def test_encode_solved_state():
    state = geoprover.parse_problem(
        "orthocenter\n"
        "a b c = triangle; h = on_tline b a c, on_tline c a b ? perp a h b c"
    )
    geoprover.saturate(state)
    tensor = geoprover.encode_state(state)
    nonzero = sum(1 for x in tensor if x > 0)
    assert nonzero > 0, "Solved state should have nonzero tensor"
    print(f"Solved state encoding: nonzero={nonzero}")


def test_compute_delta_d():
    state = geoprover.parse_problem("test\na b c = triangle ? perp a b b c")
    delta = geoprover.compute_delta_d(state)
    assert isinstance(delta, float)
    assert 0.0 <= delta <= 1.0
    print(f"Delta D (unsolved): {delta:.4f}")

    # After saturation
    geoprover.saturate(state)
    delta2 = geoprover.compute_delta_d(state)
    print(f"Delta D (after saturate): {delta2:.4f}")


def test_saturate_with_config():
    state = geoprover.parse_problem(
        "orthocenter\n"
        "a b c = triangle; h = on_tline b a c, on_tline c a b ? perp a h b c"
    )
    proved = geoprover.saturate_with_config(state, max_iterations=50, max_facts=5000)
    assert proved, "Orthocenter should be solvable with config"
    print(f"Saturate with config: proved={proved}, facts={state.num_facts()}")


def test_parse_error():
    try:
        geoprover.parse_problem("bad input")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"Parse error (expected): {e}")


if __name__ == "__main__":
    test_version()
    test_parse_simple()
    test_saturate_simple()
    test_encode_state()
    test_generate_constructions()
    test_apply_construction()
    test_orthocenter_solvable()
    test_encode_solved_state()
    test_compute_delta_d()
    test_saturate_with_config()
    test_parse_error()
    print("\nAll bridge tests passed!")
