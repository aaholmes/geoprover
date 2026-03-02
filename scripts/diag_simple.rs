use geoprover::parser::parse_problem;
use geoprover::deduction::saturate;
use geoprover::proof_state::Relation;

fn diagnose(name: &str, def: &str) {
    let problem = format!("{}\n{}", name, def);
    let Ok(mut state) = parse_problem(&problem) else {
        println!("{}: PARSE ERROR\n", name);
        return;
    };
    let goal = state.goal.as_ref().unwrap().clone();
    let initial = state.facts.len();
    let solved = saturate(&mut state);
    if solved { return; } // skip solved

    println!("=== {} ===", name);
    println!("Goal: {:?}", goal);
    println!("Facts: {} → {}", initial, state.facts.len());

    // Print facts of same type as goal
    match &goal {
        Relation::Perpendicular(..) => {
            println!("Perp facts:");
            for f in &state.facts {
                if matches!(f, Relation::Perpendicular(..)) {
                    println!("  {:?}", f);
                }
            }
        }
        Relation::Parallel(..) => {
            println!("Para facts:");
            for f in &state.facts {
                if matches!(f, Relation::Parallel(..)) {
                    println!("  {:?}", f);
                }
            }
        }
        Relation::Congruent(..) => {
            println!("Cong facts:");
            for f in &state.facts {
                if matches!(f, Relation::Congruent(..)) {
                    println!("  {:?}", f);
                }
            }
        }
        Relation::Collinear(..) => {
            println!("Coll facts:");
            for f in &state.facts {
                if matches!(f, Relation::Collinear(..)) {
                    println!("  {:?}", f);
                }
            }
        }
        Relation::Midpoint(..) => {
            println!("Midp facts:");
            for f in &state.facts {
                if matches!(f, Relation::Midpoint(..)) {
                    println!("  {:?}", f);
                }
            }
            println!("Cong facts:");
            for f in &state.facts {
                if matches!(f, Relation::Congruent(..)) {
                    println!("  {:?}", f);
                }
            }
            println!("Coll facts:");
            for f in &state.facts {
                if matches!(f, Relation::Collinear(..)) {
                    println!("  {:?}", f);
                }
            }
        }
        _ => {}
    }
    println!();
}

fn main() {
    // Simple "level 0" problems — 1-3 construction steps, few points
    // From jgex_ag_231.txt, pick some of the simplest looking ones

    // Midpoint problems
    diagnose("e04f", "a b c = triangle a b c; d = midpoint d a b; e = midpoint e a c ? midp d e a");
    diagnose("trapezoid_midp", "a b c d = trapezoid a b c d; e = midpoint e a d; f = midpoint f b c ? midp e f b");

    // Simple perp problems
    diagnose("perp_01-20_02", "a b c = triangle a b c; d = foot d a b c; e = foot e b a c ? perp a d b e");
    diagnose("perp_E037-20", "a b c = triangle a b c; d = foot d a b c; e = midpoint e b c ? perp a d d e");
    diagnose("perp_81-109_96", "a b c = triangle a b c; d = foot d a b c; e = foot e b a c; f = foot f c a b ? perp d e a b");
    diagnose("perp_E037-21", "a b c = triangle a b c; d = foot d a b c; e = midpoint e a d ? perp b e a c");

    // Simple para problems
    diagnose("para_E022-9", "a b c = triangle a b c; d = midpoint d b c; e = midpoint e a c ? para d e a b");
    diagnose("para_E023-15", "a b c = triangle a b c; d = midpoint d a b; e = midpoint e a c ? para d e b c");
    diagnose("para_ndgs_01", "a b c = triangle a b c; d = midpoint d b c; e = midpoint e a c; f = midpoint f a b ? para e f b c");

    // Simple cong problems
    diagnose("cong_E022-12", "a b c = triangle a b c; d = midpoint d a c; e = midpoint e a b ? cong d e d e");
    diagnose("cong_L057-2", "a b c = triangle a b c; d = midpoint d a b; e = midpoint e b c; f = midpoint f a c ? cong d e d f");
    diagnose("cong_L046-16", "a b = segment a b; c = midpoint c a b; d = on_bline d a b ? cong a d b d");
    diagnose("cong_E051-26", "a b c = triangle a b c; d = midpoint d b c; e = midpoint e a c; f = midpoint f a b ? cong a d f e");

    // Simple collinear problems
    diagnose("coll_M024-94", "a b c = triangle a b c; d = midpoint d b c; e = midpoint e a d ? coll b e c");
}
