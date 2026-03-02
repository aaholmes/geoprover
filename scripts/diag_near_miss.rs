use geoprover::parser::parse_problem;
use geoprover::deduction::saturate;
use geoprover::proof_state::Relation;

fn diagnose(name: &str, def: &str) {
    let problem = format!("{}\n{}", name, def);
    let Ok(mut state) = parse_problem(&problem) else {
        println!("{}: PARSE ERROR", name);
        return;
    };

    let goal = state.goal.as_ref().unwrap().clone();
    println!("=== {} ===", name);
    println!("Goal: {:?}", goal);

    let solved = saturate(&mut state);
    println!("Solved: {}, Facts: {}", solved, state.facts.len());

    if !solved {
        match &goal {
            Relation::Perpendicular(a, b, c, d) => {
                println!("\nPerp facts with 3+ point overlap:");
                let goal_pts = [*a, *b, *c, *d];
                for f in &state.facts {
                    if let Relation::Perpendicular(pa, pb, pc, pd) = f {
                        let fact_pts = [*pa, *pb, *pc, *pd];
                        let overlap = fact_pts.iter().filter(|p| goal_pts.contains(p)).count();
                        if overlap >= 3 {
                            println!("  {:?} (overlap={})", f, overlap);
                        }
                    }
                }
                // Show collinear facts involving goal points
                println!("\nCollinear with goal points:");
                for f in &state.facts {
                    if let Relation::Collinear(ca, cb, cc) = f {
                        let pts = [*ca, *cb, *cc];
                        if goal_pts.iter().filter(|p| pts.contains(p)).count() >= 2 {
                            println!("  {:?}", f);
                        }
                    }
                }
            }
            Relation::Parallel(a, b, c, d) => {
                println!("\nParallel facts with 3+ point overlap:");
                let goal_pts = [*a, *b, *c, *d];
                for f in &state.facts {
                    if let Relation::Parallel(pa, pb, pc, pd) = f {
                        let fact_pts = [*pa, *pb, *pc, *pd];
                        let overlap = fact_pts.iter().filter(|p| goal_pts.contains(p)).count();
                        if overlap >= 3 {
                            println!("  {:?} (overlap={})", f, overlap);
                        }
                    }
                }
                println!("\nCollinear with goal points:");
                for f in &state.facts {
                    if let Relation::Collinear(ca, cb, cc) = f {
                        let pts = [*ca, *cb, *cc];
                        if goal_pts.iter().filter(|p| pts.contains(p)).count() >= 2 {
                            println!("  {:?}", f);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    println!();
}

fn main() {
    // Perp with overlap=4 (should be solved!?)
    diagnose("ndgs_02",
        "a b c = triangle a b c; d = midpoint d b c; e = midpoint e a c; f = midpoint f a b; g = orthocenter g d e f ? perp a g b c");

    // Another perp overlap=4
    diagnose("01-20_16",
        "a b c d = quadrangle a b c d; e = midpoint e a c; f = midpoint f b d; g = midpoint g a b; h = midpoint h c d; i = midpoint i a d; j = midpoint j b c; k = intersection_ll k e f g h; l = intersection_ll l e f i j ? perp e k e l");

    // Para with overlap=4
    diagnose("E092-5",
        "a b c = triangle a b c; d = midpoint d b c; e = midpoint e a c; f = midpoint f a b; g = parallelogram d a e g ? para f g d a");
}
