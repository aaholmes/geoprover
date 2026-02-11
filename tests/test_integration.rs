use geoprover::parser::parse_problem;
use geoprover::deduction::saturate;
use geoprover::construction::{generate_constructions, apply_construction};
use geoprover::mcts::{mcts_search, MctsConfig};

// ============================
// Level 1 — saturate() alone
// ============================

#[test]
fn test_level1_isosceles_base_angles() {
    // Isosceles triangle with |AB| = |AC| → base angles equal
    // angle(A,B,C) = angle(A,C,B)
    let input = "isosceles_base\na b c = iso_triangle a b c ? eqangle a b a c a c a b";
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

    for chunk in lines.chunks(2) {
        if chunk.len() == 2 {
            let problem = format!("{}\n{}", chunk[0], chunk[1]);
            if let Ok(mut state) = parse_problem(&problem) {
                total += 1;
                if saturate(&mut state) {
                    solved += 1;
                }
            }
        }
    }
    println!("Solved {}/{} parseable JGEX problems by deduction alone", solved, total);
    // We expect at least some to be solvable (Level 1 problems)
    // Don't assert a specific count since it depends on rule completeness
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
        num_iterations: 100,
        max_children: 20,
        c_puct: 1.4,
        max_depth: 2,
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

                // Then try MCTS
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
