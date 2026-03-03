use std::collections::HashSet;
use geoprover::parser::parse_problem;
use geoprover::deduction::{saturate, saturate_with_trace};
use geoprover::construction::{generate_constructions, apply_construction};
use geoprover::mcts::{mcts_search, MctsConfig};
use geoprover::proof_trace::RuleName;

// ============================
// Level 1 — saturate() alone
// ============================

#[test]
fn test_level1_isosceles_base_angles() {
    // Isosceles triangle with |AB| = |AC| → base angles equal
    // angle(ABC) = angle(ACB), i.e., angle between line(BA),line(BC) = angle between line(CA),line(CB)
    // eqangle b a b c c a c b
    let input = "isosceles_base\na b c = iso_triangle a b c ? eqangle b a b c c a c b";
    let mut state = parse_problem(input).unwrap();
    let proved = saturate(&mut state);
    assert!(proved, "Isosceles base angles should be proved by deduction");
}

#[test]
fn test_level1_perpendicular_direct() {
    // If we directly encode perpendicularity, goal should be trivially proved
    let input = "perp_direct\na b c = triangle a b c; d = foot a b c ? perp a d b c";
    let mut state = parse_problem(input).unwrap();
    let proved = saturate(&mut state);
    assert!(proved, "Direct perpendicularity from foot should be proved");
}

#[test]
fn test_level1_midpoint_congruent() {
    // M is midpoint of AB → |AM| = |MB|
    let input = "midpoint_cong\na b = segment a b; m = midpoint a b ? cong a m m b";
    let mut state = parse_problem(input).unwrap();
    let proved = saturate(&mut state);
    assert!(proved, "Midpoint definition should give congruent segments");
}

#[test]
fn test_level1_transitive_parallel() {
    // AB ∥ CD, CD ∥ EF → AB ∥ EF
    // Construct manually since JGEX format is awkward for this case
    let mut state = geoprover::proof_state::ProofState::new();
    let a = state.add_object("a", geoprover::proof_state::ObjectType::Point);
    let b = state.add_object("b", geoprover::proof_state::ObjectType::Point);
    let c = state.add_object("c", geoprover::proof_state::ObjectType::Point);
    let d = state.add_object("d", geoprover::proof_state::ObjectType::Point);
    let e = state.add_object("e", geoprover::proof_state::ObjectType::Point);
    let f = state.add_object("f", geoprover::proof_state::ObjectType::Point);
    state.add_fact(geoprover::proof_state::Relation::parallel(a, b, c, d));
    state.add_fact(geoprover::proof_state::Relation::parallel(c, d, e, f));
    state.set_goal(geoprover::proof_state::Relation::parallel(a, b, e, f));
    assert!(saturate(&mut state), "Transitive parallel should be proved");
}

#[test]
fn test_level1_perp_to_parallel() {
    // AB ⊥ CD, EF ⊥ CD → AB ∥ EF
    let mut state = geoprover::proof_state::ProofState::new();
    let a = state.add_object("a", geoprover::proof_state::ObjectType::Point);
    let b = state.add_object("b", geoprover::proof_state::ObjectType::Point);
    let c = state.add_object("c", geoprover::proof_state::ObjectType::Point);
    let d = state.add_object("d", geoprover::proof_state::ObjectType::Point);
    let e = state.add_object("e", geoprover::proof_state::ObjectType::Point);
    let f = state.add_object("f", geoprover::proof_state::ObjectType::Point);
    state.add_fact(geoprover::proof_state::Relation::perpendicular(a, b, c, d));
    state.add_fact(geoprover::proof_state::Relation::perpendicular(e, f, c, d));
    state.set_goal(geoprover::proof_state::Relation::parallel(a, b, e, f));
    assert!(saturate(&mut state), "Perp to parallel should be proved");
}

#[test]
fn test_level1_circumcenter_equidistant() {
    // Circumcenter equidistance: |OA|=|OB|, |OB|=|OC| → |OA|=|OC|
    let input = "circumcenter_eq\na b c = triangle a b c; o = circumcenter a b c ? cong o a o c";
    let mut state = parse_problem(input).unwrap();
    let proved = saturate(&mut state);
    assert!(proved, "Circumcenter equidistance should be proved");
}

#[test]
fn test_level1_alternate_interior_angles() {
    // Parallel lines + transversal → alternate interior angles equal
    use geoprover::proof_state::{ProofState, ObjectType, Relation};
    let mut state = ProofState::new();
    let a = state.add_object("a", ObjectType::Point);
    let b = state.add_object("b", ObjectType::Point);
    let c = state.add_object("c", ObjectType::Point);
    let d = state.add_object("d", ObjectType::Point);
    let t = state.add_object("t", ObjectType::Point);
    // AB ∥ CD, transversal A-T-C
    state.add_fact(Relation::parallel(a, b, c, d));
    state.add_fact(Relation::collinear(a, t, c));
    // Goal: angle(B,A,C) = angle(D,C,A)
    state.set_goal(Relation::equal_angle(b, a, c, d, c, a));
    assert!(saturate(&mut state), "Alternate interior angles should be proved");
}

// ============================
// Level 1 — parsed from JGEX
// ============================

#[test]
fn test_parse_jgex_problems_no_panic() {
    // Parse a subset of jgex_ag_231.txt and ensure no panics
    let content = std::fs::read_to_string("problems/jgex_ag_231.txt").unwrap();
    let lines: Vec<&str> = content.lines().collect();
    let mut parsed = 0;
    let mut failed = 0;
    // Each problem is 2 lines: name, definition
    for chunk in lines.chunks(2) {
        if chunk.len() == 2 {
            let problem = format!("{}\n{}", chunk[0], chunk[1]);
            match parse_problem(&problem) {
                Ok(_) => parsed += 1,
                Err(_) => failed += 1,
            }
        }
    }
    println!("Parsed {}/{} problems ({} failed)", parsed, parsed + failed, failed);
    // At least 50% should parse without error
    assert!(parsed > (parsed + failed) / 2,
        "Less than 50% of problems parsed: {}/{}", parsed, parsed + failed);
}

// ============================
// Construction + deduction
// ============================

#[test]
fn test_construction_then_saturate() {
    // Triangle ABC, apply midpoint construction on AB, then saturate
    // Should get Midpoint, Collinear, and Congruent facts
    use geoprover::proof_state::{ProofState, ObjectType, Relation};
    let mut state = ProofState::new();
    let a = state.add_object("a", ObjectType::Point);
    let b = state.add_object("b", ObjectType::Point);
    let _c = state.add_object("c", ObjectType::Point);
    state.set_goal(Relation::congruent(a, 3, 3, b)); // goal: |AM| = |MB| where M is id 3

    let constructions = generate_constructions(&state);
    // Find the midpoint construction for a,b
    let midpoint_ab = constructions.iter().find(|c| {
        c.ctype == geoprover::construction::ConstructionType::Midpoint
            && c.args == vec![a, b]
    });
    assert!(midpoint_ab.is_some(), "Should have midpoint(a,b) construction");

    let new_state = apply_construction(&state, midpoint_ab.unwrap());
    let mut new_state = new_state;
    new_state.set_goal(Relation::congruent(a, 3, 3, b));
    let proved = saturate(&mut new_state);
    assert!(proved, "After midpoint construction + saturate, congruence should be proved");
}

// ============================
// Smoke test: count JGEX solvable by saturate()
// ============================

#[test]
fn test_count_jgex_solvable_by_deduction() {
    let content = std::fs::read_to_string("problems/jgex_ag_231.txt").unwrap();
    let lines: Vec<&str> = content.lines().collect();
    let mut total = 0;
    let mut solved = 0;
    let mut solved_names = Vec::new();

    for chunk in lines.chunks(2) {
        if chunk.len() == 2 {
            let problem = format!("{}\n{}", chunk[0], chunk[1]);
            if let Ok(mut state) = parse_problem(&problem) {
                total += 1;
                let facts_before = state.facts.len();
                if saturate(&mut state) {
                    solved += 1;
                    solved_names.push(chunk[0].to_string());
                } else {
                    let facts_after = state.facts.len();
                    let new_facts = facts_after - facts_before;
                    if new_facts >= 3 {
                        let goal_str = chunk[1].split("? ").nth(1).unwrap_or("?");
                        let goal_pred = goal_str.split_whitespace().next().unwrap_or("?");
                        println!("PROGRESS: {} goal={} (+{} facts)", chunk[0], goal_pred, new_facts);
                    }
                }
            }
        }
    }
    println!("\nSolved {}/{} parseable JGEX problems by deduction alone", solved, total);
    for name in &solved_names {
        println!("  SOLVED: {}", name);
    }
}

// ============================
// IMO-AG-30 parse smoke test
// ============================

#[test]
fn test_parse_imo_ag_30_no_panic() {
    let content = std::fs::read_to_string("problems/imo_ag_30.txt").unwrap();
    let lines: Vec<&str> = content.lines().collect();
    let mut parsed = 0;
    let mut failed = 0;
    let mut fail_names = Vec::new();
    for chunk in lines.chunks(2) {
        if chunk.len() == 2 {
            let problem = format!("{}\n{}", chunk[0], chunk[1]);
            match parse_problem(&problem) {
                Ok(_) => parsed += 1,
                Err(_) => {
                    failed += 1;
                    fail_names.push(chunk[0].to_string());
                }
            }
        }
    }
    println!("IMO-AG-30: Parsed {}/{} ({} failed)", parsed, parsed + failed, failed);
    for name in &fail_names {
        println!("  Failed: {}", name);
    }
    // IMO problems may have complex constructions; we just want no panics
    assert!(parsed > 0, "Should parse at least some IMO problems");
}

// ============================
// Multi-step: construction → saturate → verify new facts
// ============================

#[test]
fn test_construction_saturate_new_facts() {
    use geoprover::proof_state::{ProofState, ObjectType, Relation};
    let mut state = ProofState::new();
    let a = state.add_object("a", ObjectType::Point);
    let b = state.add_object("b", ObjectType::Point);
    let c = state.add_object("c", ObjectType::Point);

    // Apply circumcenter construction
    let construction = geoprover::construction::Construction {
        ctype: geoprover::construction::ConstructionType::Circumcenter,
        args: vec![a, b, c],
        priority: geoprover::construction::Priority::GoalRelevant,
    };
    let mut new_state = apply_construction(&state, &construction);
    let o = 3u16;

    // After saturate, transitive congruence should give |OA| = |OC|
    new_state.set_goal(Relation::congruent(o, a, o, c));
    let proved = saturate(&mut new_state);
    assert!(proved, "Circumcenter construction + saturate should prove |OA| = |OC|");
}

// ============================
// Unsolvable goal: saturate returns false
// ============================

#[test]
fn test_unsolvable_goal() {
    use geoprover::proof_state::{ProofState, ObjectType, Relation};
    let mut state = ProofState::new();
    let a = state.add_object("a", ObjectType::Point);
    let b = state.add_object("b", ObjectType::Point);
    let c = state.add_object("c", ObjectType::Point);
    // Set an impossible goal: collinear from just triangle points with no collinear facts
    state.set_goal(Relation::collinear(a, b, c));
    assert!(!saturate(&mut state), "Should not prove collinearity of triangle vertices");
}

// ============================
// Identify failing JGEX parse problems
// ============================

#[test]
fn test_identify_jgex_parse_failures() {
    let content = std::fs::read_to_string("problems/jgex_ag_231.txt").unwrap();
    let lines: Vec<&str> = content.lines().collect();
    let mut fail_names = Vec::new();
    let mut fail_errors = Vec::new();
    for chunk in lines.chunks(2) {
        if chunk.len() == 2 {
            let problem = format!("{}\n{}", chunk[0], chunk[1]);
            if let Err(e) = parse_problem(&problem) {
                fail_names.push(chunk[0].to_string());
                fail_errors.push(e.0);
            }
        }
    }
    println!("JGEX parse failures ({}):", fail_names.len());
    for (name, err) in fail_names.iter().zip(fail_errors.iter()) {
        println!("  {}: {}", name, err);
    }
    // We know from memory that 228/231 parse, so at most 3 failures
    assert!(fail_names.len() <= 5, "Expected at most 5 parse failures, got {}", fail_names.len());
}

// ============================
// MCTS tests
// ============================

#[test]
fn test_mcts_solves_midpoint_congruence() {
    // Triangle ABC, goal: |AM| = |MB| where M is aux point
    // MCTS should find midpoint construction
    use geoprover::proof_state::{ProofState, ObjectType, Relation};
    let mut state = ProofState::new();
    let a = state.add_object("a", ObjectType::Point);
    let b = state.add_object("b", ObjectType::Point);
    state.add_object("c", ObjectType::Point);
    state.set_goal(Relation::congruent(a, 3, 3, b));

    let config = MctsConfig {
        num_iterations: 300,
        max_children: 30,
        c_puct: 1.4,
        max_depth: 2,
    };
    let result = mcts_search(state, &config);
    assert!(result.solved, "MCTS should solve midpoint congruence");
    assert!(!result.proof_actions.is_empty());
    println!("Midpoint solved in {} iterations with {} constructions",
        result.iterations, result.proof_actions.len());
}

#[test]
fn test_mcts_solves_circumcenter_equidistance() {
    // Triangle ABC, goal: |OA| = |OC| where O = circumcenter
    // MCTS should find circumcenter construction → transitive congruence
    use geoprover::proof_state::{ProofState, ObjectType, Relation};
    let mut state = ProofState::new();
    let a = state.add_object("a", ObjectType::Point);
    state.add_object("b", ObjectType::Point);
    let c = state.add_object("c", ObjectType::Point);
    let o = 3u16; // circumcenter will be aux_3
    state.set_goal(Relation::congruent(o, a, o, c));

    let config = MctsConfig {
        num_iterations: 300,
        max_children: 30,
        c_puct: 1.4,
        max_depth: 2,
    };
    let result = mcts_search(state, &config);
    assert!(result.solved, "MCTS should solve circumcenter equidistance");
    println!("Circumcenter solved in {} iterations with {} constructions",
        result.iterations, result.proof_actions.len());
}

#[test]
fn test_count_jgex_solvable_by_mcts() {
    let content = std::fs::read_to_string("problems/jgex_ag_231.txt").unwrap();
    let lines: Vec<&str> = content.lines().collect();
    let mut total = 0;
    let mut solved_deduction = 0;
    let mut solved_mcts = 0;
    let mut solved_names = Vec::new();

    let config = MctsConfig {
        num_iterations: 10,
        max_children: 5,
        c_puct: 1.4,
        max_depth: 1,
    };

    for chunk in lines.chunks(2) {
        if chunk.len() == 2 {
            let problem = format!("{}\n{}", chunk[0], chunk[1]);
            if let Ok(state) = parse_problem(&problem) {
                total += 1;

                // First try deduction alone
                let mut deduction_state = state.clone();
                if saturate(&mut deduction_state) {
                    solved_deduction += 1;
                    solved_mcts += 1;
                    solved_names.push(format!("{} (deduction)", chunk[0]));
                    continue;
                }

                // Then try MCTS (lightweight config for benchmark speed)
                let result = mcts_search(state, &config);
                if result.solved {
                    solved_mcts += 1;
                    solved_names.push(format!("{} (mcts, {} iters, {} steps)",
                        chunk[0], result.iterations, result.proof_actions.len()));
                }
            }
        }
    }

    println!("\n=== JGEX-AG-231 Results ===");
    println!("Total parseable: {}", total);
    println!("Solved by deduction: {}", solved_deduction);
    println!("Solved by MCTS: {}", solved_mcts - solved_deduction);
    println!("Total solved: {}", solved_mcts);
    println!("\nSolved problems:");
    for name in &solved_names {
        println!("  {}", name);
    }
}

// ============================
// Proof Trace integration tests
// ============================

#[test]
fn test_trace_midpoint_proof() {
    // Midpoint(m,a,b) → saturate → proves Cong(a,m,m,b)
    let input = "midpoint_trace\na b = segment a b; m = midpoint a b ? cong a m m b";
    let mut state = parse_problem(input).unwrap();
    let (proved, trace) = saturate_with_trace(&mut state);
    assert!(proved, "Midpoint congruence should be proved with trace");
    assert!(trace.axiom_count() > 0, "Should have axioms");
    assert!(trace.len() > trace.axiom_count(), "Should have derived facts beyond axioms");

    // Extract proof for the goal
    let goal = state.goal.as_ref().unwrap();
    let proof = trace.extract_proof(goal);
    assert!(proof.is_some(), "Should extract proof for goal");
    let steps = proof.unwrap();
    assert!(!steps.is_empty(), "Proof should have steps");
    // Goal should appear in the proof
    assert!(steps.iter().any(|d| &d.fact == goal), "Goal should be in proof");
}

#[test]
fn test_trace_matches_saturate() {
    // For several JGEX problems, both functions should agree on proved/not-proved
    let content = std::fs::read_to_string("problems/jgex_ag_231.txt").unwrap();
    let lines: Vec<&str> = content.lines().collect();
    let mut checked = 0;
    let mut mismatches = 0;

    for chunk in lines.chunks(2) {
        if chunk.len() == 2 && checked < 10 {
            let problem = format!("{}\n{}", chunk[0], chunk[1]);
            if let Ok(state) = parse_problem(&problem) {
                checked += 1;
                let mut state1 = state.clone();
                let mut state2 = state;
                let proved_normal = saturate(&mut state1);
                let (proved_trace, _trace) = saturate_with_trace(&mut state2);
                if proved_normal != proved_trace {
                    mismatches += 1;
                    println!("MISMATCH: {} normal={} trace={}", chunk[0], proved_normal, proved_trace);
                }
            }
        }
    }
    println!("Checked {} problems, {} mismatches", checked, mismatches);
    assert_eq!(mismatches, 0, "saturate_with_trace should agree with saturate");
}

#[test]
fn test_trace_orthocenter() {
    // Full problem: triangle orthocenter
    let input = "ortho_trace\na b c = triangle a b c; h = orthocenter a b c ? perp b h a c";
    let mut state = parse_problem(input).unwrap();
    let (proved, trace) = saturate_with_trace(&mut state);
    assert!(proved, "Orthocenter perpendicularity should be proved");

    let goal = state.goal.as_ref().unwrap();
    let proof = trace.extract_proof(goal).unwrap();
    assert!(!proof.is_empty());
    // Verify chain: all premises should exist in the proof or be axioms
    for step in &proof {
        for premise in &step.premises {
            assert!(
                proof.iter().any(|s| &s.fact == premise),
                "Premise {:?} not found in proof", premise
            );
        }
    }
}

#[test]
fn test_trace_isosceles() {
    let input = "iso_trace\na b c = iso_triangle a b c ? eqangle b a b c c a c b";
    let mut state = parse_problem(input).unwrap();
    let (proved, trace) = saturate_with_trace(&mut state);
    assert!(proved, "Isosceles base angles should be proved");

    let goal = state.goal.as_ref().unwrap();
    let proof = trace.extract_proof(goal).unwrap();
    // All premises should exist in the proof
    for step in &proof {
        for premise in &step.premises {
            assert!(
                proof.iter().any(|s| &s.fact == premise),
                "Premise {:?} not found in proof", premise
            );
        }
    }
}

#[test]
fn test_trace_format_readable() {
    let input = "format_trace\na b c = triangle a b c; h = orthocenter a b c ? perp b h a c";
    let mut state = parse_problem(input).unwrap();
    let (proved, trace) = saturate_with_trace(&mut state);
    assert!(proved);

    let goal = state.goal.as_ref().unwrap();
    let formatted = trace.format_proof(goal, &state);
    assert!(formatted.is_some(), "Should produce formatted proof");
    let text = formatted.unwrap();
    println!("{}", text);
    assert!(text.contains("Proof"), "Should start with Proof header");
    assert!(text.contains("axiom"), "Should mention axioms");
    assert!(text.lines().count() >= 2, "Should have at least header + steps");
}

#[test]
fn test_trace_transitive_parallel() {
    // Simple: AB ∥ CD, CD ∥ EF → AB ∥ EF
    use geoprover::proof_state::{ProofState, ObjectType, Relation};
    let mut state = ProofState::new();
    let a = state.add_object("a", ObjectType::Point);
    let b = state.add_object("b", ObjectType::Point);
    let c = state.add_object("c", ObjectType::Point);
    let d = state.add_object("d", ObjectType::Point);
    let e = state.add_object("e", ObjectType::Point);
    let f = state.add_object("f", ObjectType::Point);
    state.add_fact(Relation::parallel(a, b, c, d));
    state.add_fact(Relation::parallel(c, d, e, f));
    state.set_goal(Relation::parallel(a, b, e, f));

    let (proved, trace) = saturate_with_trace(&mut state);
    assert!(proved);
    assert_eq!(trace.axiom_count(), 2);

    let goal = state.goal.as_ref().unwrap();
    let proof = trace.extract_proof(goal).unwrap();
    assert_eq!(proof.len(), 3); // 2 axioms + 1 derived

    let formatted = trace.format_proof(goal, &state).unwrap();
    println!("{}", formatted);
    assert!(formatted.contains("TransitiveParallel"));
}

// ============================
// Proof trace: no circular deps
// ============================

/// Helper: check that no fact in the proof depends on itself (directly or transitively).
fn assert_no_circular_deps(proof: &[geoprover::proof_trace::Derivation]) {
    // Build dependency graph: fact -> set of premise facts
    let fact_set: HashSet<_> = proof.iter().map(|d| &d.fact).collect();
    for d in proof {
        if d.rule == RuleName::Axiom {
            assert!(d.premises.is_empty(), "Axiom {:?} should have no premises", d.fact);
            continue;
        }
        // No self-referencing: a fact should not be its own premise
        assert!(
            !d.premises.contains(&d.fact),
            "Self-reference: {:?} lists itself as a premise (rule: {:?})",
            d.fact, d.rule,
        );
        // All premises should appear earlier in the proof (topological order)
        let my_pos = proof.iter().position(|x| x.fact == d.fact).unwrap();
        for premise in &d.premises {
            if let Some(prem_pos) = proof.iter().position(|x| &x.fact == premise) {
                assert!(
                    prem_pos < my_pos,
                    "Circular dep: {:?} (pos {}) depends on {:?} (pos {})",
                    d.fact, my_pos, premise, prem_pos,
                );
            }
        }
    }
}

#[test]
fn test_proof_trace_no_circular_deps_circumcenter() {
    // This problem previously had circular deps in TransitiveCongruent premises
    let input = "test\na b c = triangle a b c; o = circle o a b c; \
        h = midpoint h c b; d = on_line d o h, on_line d a b; \
        e = on_tline e c c o, on_tline e a a o ? cyclic a o e d";
    let mut state = parse_problem(input).unwrap();
    let (proved, trace) = saturate_with_trace(&mut state);
    assert!(proved, "Problem should be solved by deduction");

    let goal = state.goal.as_ref().unwrap();
    let proof = trace.extract_proof(goal).unwrap();
    println!("Proof steps: {}", proof.len());
    for (i, d) in proof.iter().enumerate() {
        let prem_strs: Vec<String> = d.premises.iter().map(|p| format!("{:?}", p)).collect();
        println!("  {}: {:?} [{:?}] from {:?}", i + 1, d.fact, d.rule,
                 prem_strs.join(", "));
    }
    assert_no_circular_deps(&proof);
}

#[test]
fn test_proof_trace_no_circular_deps_isquare() {
    // The E061-65 problem that showed circular TransitiveCongruent deps
    let input = "test\na b c d = isquare a b c d; \
        e = s_angle c d e 15, s_angle d c e -15; \
        f = reflect f e a c ? contri e a b a b e";
    let mut state = parse_problem(input).unwrap();
    let (proved, trace) = saturate_with_trace(&mut state);
    assert!(proved, "Problem should be solved by deduction");

    let goal = state.goal.as_ref().unwrap();
    let proof = trace.extract_proof(goal).unwrap();
    println!("Proof steps: {}", proof.len());
    for (i, d) in proof.iter().enumerate() {
        println!("  {}: {:?} [{:?}]", i + 1, d.fact, d.rule);
    }
    assert_no_circular_deps(&proof);
}

#[test]
fn test_proof_trace_no_circular_deps_angle_bisector() {
    // Circumcenter + angle bisector problem
    let input = "test\na b c = triangle a b c; d = circumcenter d a c b; \
        e = on_line e b c; \
        f = on_circle f d a, angle_bisector f a c e ? cong a f f b";
    let mut state = parse_problem(input).unwrap();
    let (proved, trace) = saturate_with_trace(&mut state);
    assert!(proved, "Problem should be solved by deduction");

    let goal = state.goal.as_ref().unwrap();
    let proof = trace.extract_proof(goal).unwrap();
    println!("Proof steps: {}", proof.len());
    for (i, d) in proof.iter().enumerate() {
        println!("  {}: {:?} [{:?}]", i + 1, d.fact, d.rule);
    }
    assert_no_circular_deps(&proof);
}

#[test]
fn test_proof_path_subset_of_deduced() {
    // Verify that proof path facts are a subset of post-saturation facts
    let input = "test\na b c = triangle a b c; d = circumcenter d a c b; \
        e = on_line e b c; \
        f = on_circle f d a, angle_bisector f a c e ? cong a f f b";
    let mut state = parse_problem(input).unwrap();
    let pre_facts: HashSet<_> = state.facts.iter().cloned().collect();
    let (proved, trace) = saturate_with_trace(&mut state);
    assert!(proved);

    let goal = state.goal.as_ref().unwrap();
    let proof = trace.extract_proof(goal).unwrap();
    let post_facts: HashSet<_> = state.facts.iter().cloned().collect();
    let deduced: HashSet<_> = post_facts.difference(&pre_facts).cloned().collect();

    for d in &proof {
        if d.rule != RuleName::Axiom {
            assert!(
                post_facts.contains(&d.fact),
                "Proof step {:?} not in post-saturation facts",
                d.fact,
            );
        }
    }
}

#[test]
fn test_proof_premises_exist_in_proof() {
    // Every premise of a proof step should also be in the proof
    let input = "test\na b c = triangle a b c; d = circumcenter d a c b; \
        e = on_line e b c; \
        f = on_circle f d a, angle_bisector f a c e ? cong a f f b";
    let mut state = parse_problem(input).unwrap();
    let (proved, trace) = saturate_with_trace(&mut state);
    assert!(proved);

    let goal = state.goal.as_ref().unwrap();
    let proof = trace.extract_proof(goal).unwrap();
    let proof_facts: HashSet<_> = proof.iter().map(|d| &d.fact).collect();

    for d in &proof {
        for premise in &d.premises {
            assert!(
                proof_facts.contains(premise),
                "Premise {:?} of step {:?} [{:?}] not found in proof",
                premise, d.fact, d.rule,
            );
        }
    }
}
