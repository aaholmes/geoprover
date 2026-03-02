use geoprover::parser::parse_problem;
use geoprover::deduction::saturate;
use geoprover::proof_state::Relation;
use std::collections::HashMap;

fn goal_type(r: &Relation) -> &'static str {
    match r {
        Relation::Parallel(..) => "para",
        Relation::Perpendicular(..) => "perp",
        Relation::Congruent(..) => "cong",
        Relation::EqualAngle(..) => "eqangle",
        Relation::Collinear(..) => "coll",
        Relation::Cyclic(..) => "cyclic",
        Relation::Midpoint(..) => "midp",
        Relation::OnCircle(..) => "oncircle",
        Relation::EqualRatio(..) => "eqratio",
    }
}

fn goal_points(r: &Relation) -> Vec<u16> {
    match r {
        Relation::Parallel(a, b, c, d) | Relation::Perpendicular(a, b, c, d) |
        Relation::Congruent(a, b, c, d) => vec![*a, *b, *c, *d],
        Relation::EqualAngle(a, b, c, d, e, f) => vec![*a, *b, *c, *d, *e, *f],
        Relation::Collinear(a, b, c) | Relation::Midpoint(a, b, c) => vec![*a, *b, *c],
        Relation::Cyclic(a, b, c, d) => vec![*a, *b, *c, *d],
        Relation::OnCircle(a, b) => vec![*a, *b],
        Relation::EqualRatio(a, b, c, d, e, f, g, h) => vec![*a, *b, *c, *d, *e, *f, *g, *h],
    }
}

fn main() {
    let content = std::fs::read_to_string("problems/jgex_ag_231.txt").unwrap();
    let lines: Vec<&str> = content.lines().collect();

    let mut by_type: HashMap<&str, Vec<(String, usize, String)>> = HashMap::new();

    for chunk in lines.chunks(2) {
        if chunk.len() < 2 { continue; }
        let problem = format!("{}\n{}", chunk[0], chunk[1]);
        let Ok(mut state) = parse_problem(&problem) else { continue; };

        let goal = match &state.goal {
            Some(g) => g.clone(),
            None => continue,
        };

        let gtype = goal_type(&goal);
        let gpts = goal_points(&goal);

        let solved = saturate(&mut state);
        if solved { continue; }

        // Check how close we are
        let mut closeness = String::new();
        match &goal {
            Relation::Parallel(a, b, c, d) => {
                let mut best_overlap = 0;
                for f in &state.facts {
                    if let Relation::Parallel(pa, pb, pc, pd) = f {
                        let fact_pts = [*pa, *pb, *pc, *pd];
                        let overlap = gpts.iter().filter(|p| fact_pts.contains(p)).count();
                        best_overlap = best_overlap.max(overlap);
                    }
                }
                // Also check if we have lines through goal points as collinear
                let has_col_ab = state.facts.iter().any(|f| matches!(f, Relation::Collinear(x,y,z) if
                    [*x,*y,*z].contains(a) && [*x,*y,*z].contains(b)));
                let has_col_cd = state.facts.iter().any(|f| matches!(f, Relation::Collinear(x,y,z) if
                    [*x,*y,*z].contains(c) && [*x,*y,*z].contains(d)));
                closeness = format!("best_overlap={}, colAB={}, colCD={}", best_overlap, has_col_ab, has_col_cd);
            }
            Relation::Perpendicular(a, b, c, d) => {
                let mut best_overlap = 0;
                for f in &state.facts {
                    if let Relation::Perpendicular(pa, pb, pc, pd) = f {
                        let fact_pts = [*pa, *pb, *pc, *pd];
                        let overlap = gpts.iter().filter(|p| fact_pts.contains(p)).count();
                        best_overlap = best_overlap.max(overlap);
                    }
                }
                closeness = format!("best_perp_overlap={}", best_overlap);
            }
            Relation::Congruent(a, b, c, d) => {
                let has_ab = state.facts.iter().any(|f| if let Relation::Congruent(p,q,r,s) = f {
                    (*p==*a && *q==*b) || (*r==*a && *s==*b) || (*p==*b && *q==*a) || (*r==*b && *s==*a)
                } else { false });
                let has_cd = state.facts.iter().any(|f| if let Relation::Congruent(p,q,r,s) = f {
                    (*p==*c && *q==*d) || (*r==*c && *s==*d) || (*p==*d && *q==*c) || (*r==*d && *s==*c)
                } else { false });
                closeness = format!("has_seg_ab={}, has_seg_cd={}", has_ab, has_cd);
            }
            Relation::EqualAngle(..) => {
                closeness = format!("eqangle_facts={}", state.facts.iter().filter(|f| matches!(f, Relation::EqualAngle(..))).count());
            }
            Relation::Cyclic(a, b, c, d) => {
                let pts = [*a, *b, *c, *d];
                let mut on_same_circle = 0;
                for f in &state.facts {
                    if let Relation::Cyclic(pa, pb, pc, pd) = f {
                        let overlap = pts.iter().filter(|&&p| [*pa,*pb,*pc,*pd].contains(&p)).count();
                        on_same_circle = on_same_circle.max(overlap);
                    }
                }
                // Count how many pairs share a circle via OnCircle
                let mut oncircle_centers: HashMap<u16, Vec<u16>> = HashMap::new();
                for f in &state.facts {
                    if let Relation::OnCircle(pt, center) = f {
                        oncircle_centers.entry(*center).or_default().push(*pt);
                    }
                }
                let mut max_on_circle = 0;
                for (_center, circle_pts) in &oncircle_centers {
                    let count = pts.iter().filter(|p| circle_pts.contains(p)).count();
                    max_on_circle = max_on_circle.max(count);
                }
                closeness = format!("best_cyclic_overlap={}, max_oncircle={}", on_same_circle, max_on_circle);
            }
            Relation::Collinear(a, b, c) => {
                let pts = [*a, *b, *c];
                let mut best_overlap = 0;
                for f in &state.facts {
                    if let Relation::Collinear(pa, pb, pc) = f {
                        let overlap = pts.iter().filter(|&&p| [*pa,*pb,*pc].contains(&p)).count();
                        best_overlap = best_overlap.max(overlap);
                    }
                }
                closeness = format!("best_coll_overlap={}", best_overlap);
            }
            _ => { closeness = "other".to_string(); }
        }

        by_type.entry(gtype).or_default().push((chunk[0].to_string(), state.facts.len(), closeness));
    }

    // Print summary
    println!("=== Goal type distribution (unsolved) ===");
    let mut types: Vec<_> = by_type.iter().collect();
    types.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
    for (gtype, problems) in &types {
        println!("  {}: {} problems", gtype, problems.len());
    }

    // For each type, show closest problems
    println!("\n=== Closest to solved by type ===");
    for (gtype, problems) in &types {
        println!("\n--- {} ({} unsolved) ---", gtype, problems.len());
        // Sort by fact count (more facts = more progress usually)
        let mut sorted: Vec<_> = problems.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        for (name, nfacts, close) in sorted.iter().take(8) {
            println!("  {} ({} facts) — {}", name, nfacts, close);
        }
    }
}
