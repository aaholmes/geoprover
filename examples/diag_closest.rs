use geoprover::parser::parse_problem;
use geoprover::deduction::saturate;
use geoprover::proof_state::Relation;

fn diagnose(name: &str, def: &str) {
    let problem = format!("{}\n{}", name, def);
    let Ok(mut state) = parse_problem(&problem) else {
        println!("{}: PARSE ERROR", name);
        return;
    };

    println!("=== {} ===", name);
    println!("Points: {:?}", state.objects);
    println!("Initial facts: {}", state.facts.len());
    println!("Goal: {:?}", state.goal);

    let solved = saturate(&mut state);
    println!("Solved: {}", solved);
    println!("Total facts: {}", state.facts.len());

    let goal = state.goal.as_ref().unwrap().clone();

    // Show relevant facts
    match &goal {
        Relation::Parallel(a, b, c, d) => {
            println!("\nParallel facts with overlapping points:");
            let goal_pts = [*a, *b, *c, *d];
            for f in &state.facts {
                if let Relation::Parallel(pa, pb, pc, pd) = f {
                    let fact_pts = [*pa, *pb, *pc, *pd];
                    let overlap = fact_pts.iter().filter(|p| goal_pts.contains(p)).count();
                    if overlap >= 2 {
                        println!("  {:?} (overlap={})", f, overlap);
                    }
                }
            }
            // Also show collinear facts involving goal points
            println!("\nCollinear facts with goal points:");
            for f in &state.facts {
                if let Relation::Collinear(ca, cb, cc) = f {
                    let col_pts = [*ca, *cb, *cc];
                    if goal_pts.iter().any(|p| col_pts.contains(p)) {
                        println!("  {:?}", f);
                    }
                }
            }
        }
        Relation::Perpendicular(a, b, c, d) => {
            println!("\nPerp facts with overlapping points:");
            let goal_pts = [*a, *b, *c, *d];
            for f in &state.facts {
                if let Relation::Perpendicular(pa, pb, pc, pd) = f {
                    let fact_pts = [*pa, *pb, *pc, *pd];
                    let overlap = fact_pts.iter().filter(|p| goal_pts.contains(p)).count();
                    if overlap >= 2 {
                        println!("  {:?} (overlap={})", f, overlap);
                    }
                }
            }
        }
        _ => {
            println!("\nAll facts:");
            for f in &state.facts {
                println!("  {:?}", f);
            }
        }
    }
    println!();
}

fn main() {
    // Simplest CLOSE para problems
    diagnose("21-40_34",
        "a b c = triangle a b c; h = orthocenter h a b c; o = circle o h b c; p = on_tline p h c h, on_circle p o b ? para a h b p");

    diagnose("41-60_51",
        "a b c = triangle a b c; o = circle o a b c; d = on_tline d b a c, on_circle d o a; e = on_circle e o d, on_line e d o ? para b e a c");
}
