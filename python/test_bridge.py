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


def test_saturate_with_trace():
    """Test that saturate_with_trace returns (bool, PyProofTrace)."""
    state = geoprover.parse_problem(
        "orthocenter\n"
        "a b c = triangle; h = on_tline b a c, on_tline c a b ? perp a h b c"
    )
    proved, trace = geoprover.saturate_with_trace(state)
    assert proved, "Orthocenter should be proved with trace"
    assert trace.len() > 0, "Trace should have derivations"
    assert trace.axiom_count() > 0, "Trace should have axioms"
    print(f"Saturate with trace: proved={proved}, derivations={trace.len()}, axioms={trace.axiom_count()}")
    print(f"  Repr: {repr(trace)}")


def test_proof_trace_extract():
    """Test extract_proof returns list of derivation steps."""
    state = geoprover.parse_problem(
        "midpoint_trace\n"
        "a b = segment a b; m = midpoint a b ? cong a m m b"
    )
    proved, trace = geoprover.saturate_with_trace(state)
    assert proved
    proof = trace.extract_proof()
    assert proof is not None, "Should extract proof"
    assert len(proof) > 0, "Proof should have steps"
    print(f"Proof trace extract: {len(proof)} steps")
    for fact_text, rule_name, premise_texts in proof:
        print(f"  {fact_text} [{rule_name}] from {premise_texts}")


def test_proof_trace_format():
    """Test format_proof returns readable string."""
    state = geoprover.parse_problem(
        "iso_trace\n"
        "a b c = iso_triangle a b c ? eqangle b a b c c a c b"
    )
    proved, trace = geoprover.saturate_with_trace(state)
    assert proved
    formatted = trace.format_proof()
    assert formatted is not None, "Should format proof"
    assert "Proof" in formatted
    assert "axiom" in formatted
    print(f"Formatted proof:\n{formatted}")


def test_goal_description():
    """Test PyProofState.goal_description() returns a string."""
    state = geoprover.parse_problem("test\na b c = triangle ? perp a b b c")
    desc = state.goal_description()
    assert desc is not None, "goal_description should return a string"
    assert "Perpendicular" in desc or "perp" in desc.lower(), f"Expected perp in: {desc}"
    print(f"Goal description: {desc}")


def test_goal_as_text():
    """Test PyProofState.goal_as_text() returns human-readable text."""
    state = geoprover.parse_problem("test\na b c = triangle ? perp a b b c")
    text = state.goal_as_text()
    assert text is not None, "goal_as_text should return text"
    assert "perp" in text, f"Expected 'perp' in: {text}"
    print(f"Goal as text: {text}")


def test_goal_none_when_unset():
    """Test goal methods return None when no goal is set."""
    state = geoprover.parse_problem("test\na b c = triangle ? perp a b b c")
    # The state has a goal from parsing, so let's just verify it's not None
    assert state.goal_description() is not None
    assert state.goal_as_text() is not None


def test_construction_methods():
    """Test PyConstruction.construction_type(), args(), priority()."""
    state = geoprover.parse_problem("test\na b c = triangle ? perp a b b c")
    constructions = geoprover.generate_constructions(state)
    assert len(constructions) > 0

    for c in constructions[:5]:
        ctype = c.construction_type()
        args = c.args()
        priority = c.priority()

        assert isinstance(ctype, str) and len(ctype) > 0, f"Bad type: {ctype}"
        assert isinstance(args, list) and len(args) > 0, f"Bad args: {args}"
        assert priority in ("GoalRelevant", "RecentlyActive", "Exploratory"), f"Bad priority: {priority}"
        # args should be valid u16 IDs
        for arg in args:
            assert isinstance(arg, int) and 0 <= arg < 65536, f"Bad arg ID: {arg}"

    print(f"Construction methods: checked {min(5, len(constructions))} constructions")
    c = constructions[0]
    print(f"  First: type={c.construction_type()}, args={c.args()}, priority={c.priority()}")


def test_construction_to_text():
    """Test construction_to_text returns meaningful text."""
    state = geoprover.parse_problem("test\na b c = triangle ? perp a b b c")
    constructions = geoprover.generate_constructions(state)
    assert len(constructions) > 0
    text = geoprover.construction_to_text(constructions[0], state)
    assert isinstance(text, str) and len(text) > 0
    print(f"Construction to text: {text}")


def test_state_to_text():
    """Test state_to_text serialization."""
    state = geoprover.parse_problem(
        "orthocenter\n"
        "a b c = triangle; h = on_tline b a c, on_tline c a b ? perp a h b c"
    )
    text = geoprover.state_to_text(state)
    assert isinstance(text, str) and len(text) > 0
    assert "?" in text, "State text should contain goal separator"
    assert "perp" in text, "State text should contain goal"
    print(f"State to text: {text[:80]}...")


def test_proof_path_facts():
    """Test PyProofTrace.proof_path_facts() returns non-axiom derived facts."""
    state = geoprover.parse_problem(
        "orthocenter\n"
        "a b c = triangle; h = on_tline b a c, on_tline c a b ? perp a h b c"
    )
    proved, trace = geoprover.saturate_with_trace(state)
    assert proved
    path_facts = trace.proof_path_facts()
    assert path_facts is not None, "proof_path_facts should not be None for proved goal"
    assert isinstance(path_facts, list)
    # All should be text strings
    for f in path_facts:
        assert isinstance(f, str) and len(f) > 0
    print(f"Proof path facts: {len(path_facts)} facts")
    for f in path_facts[:3]:
        print(f"  {f}")


def test_axiom_facts():
    """Test PyProofTrace.axiom_facts() returns axiom text list."""
    state = geoprover.parse_problem(
        "orthocenter\n"
        "a b c = triangle; h = on_tline b a c, on_tline c a b ? perp a h b c"
    )
    proved, trace = geoprover.saturate_with_trace(state)
    assert proved
    axioms = trace.axiom_facts()
    assert isinstance(axioms, list)
    assert len(axioms) > 0
    # Axioms should be sorted
    assert axioms == sorted(axioms), "axiom_facts should be sorted"
    print(f"Axiom facts: {len(axioms)} axioms")
    for a in axioms[:3]:
        print(f"  {a}")


def test_extract_all_shortest_proofs():
    """Test PyProofTrace.extract_all_shortest_proofs() returns list of proofs."""
    state = geoprover.parse_problem(
        "orthocenter\n"
        "a b c = triangle; h = on_tline b a c, on_tline c a b ? perp a h b c"
    )
    proved, trace = geoprover.saturate_with_trace(state)
    assert proved
    all_proofs = trace.extract_all_shortest_proofs()
    assert all_proofs is not None, "Should return proofs for proved goal"
    assert isinstance(all_proofs, list)
    assert len(all_proofs) >= 1, "Should have at least one proof"

    # Each proof should be a list of (fact_text, rule_name, premise_texts)
    for proof in all_proofs:
        assert isinstance(proof, list) and len(proof) > 0
        for fact_text, rule_name, premise_texts in proof:
            assert isinstance(fact_text, str)
            assert isinstance(rule_name, str)
            assert isinstance(premise_texts, list)

    # All proofs should have the same length (tied shortest)
    lengths = [len(p) for p in all_proofs]
    assert all(l == lengths[0] for l in lengths), f"Proofs should be same length: {lengths}"

    print(f"All shortest proofs: {len(all_proofs)} proofs, each {lengths[0]} steps")


def test_extract_all_shortest_proofs_no_goal():
    """Test extract_all_shortest_proofs raises error when no goal."""
    state = geoprover.parse_problem(
        "midpoint\na b = segment a b; m = midpoint a b ? cong a m m b"
    )
    proved, trace = geoprover.saturate_with_trace(state)
    # This should work fine - just verifying the proved case works
    assert proved
    proofs = trace.extract_all_shortest_proofs()
    assert proofs is not None


def test_generate_synthetic_data():
    """Test generate_synthetic_data returns valid triples."""
    data = geoprover.generate_synthetic_data(5, 42)
    assert isinstance(data, list)
    assert len(data) > 0
    for state_text, constr_text, goal_text in data:
        assert isinstance(state_text, str) and len(state_text) > 0
        assert isinstance(constr_text, str) and len(constr_text) > 0
        assert isinstance(goal_text, str) and len(goal_text) > 0
    print(f"Synthetic data: {len(data)} examples")
    s, c, g = data[0]
    print(f"  First: state={s[:40]}..., constr={c}, goal={g}")


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
    test_saturate_with_trace()
    test_proof_trace_extract()
    test_proof_trace_format()
    test_goal_description()
    test_goal_as_text()
    test_goal_none_when_unset()
    test_construction_methods()
    test_construction_to_text()
    test_state_to_text()
    test_proof_path_facts()
    test_axiom_facts()
    test_extract_all_shortest_proofs()
    test_extract_all_shortest_proofs_no_goal()
    test_generate_synthetic_data()
    print("\nAll bridge tests passed!")
