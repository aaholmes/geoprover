use geoprover::parser::parse_problem;
use geoprover::deduction::saturate;
use geoprover::proof_state::Relation;

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
    }
}

fn main() {
    let content = std::fs::read_to_string("problems/jgex_ag_231.txt").unwrap();
    let lines: Vec<&str> = content.lines().collect();

    let mut unsolved: Vec<(String, String, usize, usize, String)> = Vec::new(); // (name, def, initial_facts, total_facts, goal_type)

    for chunk in lines.chunks(2) {
        if chunk.len() < 2 { continue; }
        let problem = format!("{}\n{}", chunk[0], chunk[1]);
        let Ok(mut state) = parse_problem(&problem) else { continue; };
        let goal = match &state.goal { Some(g) => g.clone(), None => continue };
        let gtype = goal_type(&goal).to_string();
        let initial = state.facts.len();
        if saturate(&mut state) { continue; }
        let num_points = state.objects.len();
        unsolved.push((chunk[0].to_string(), chunk[1].to_string(), num_points, state.facts.len(), gtype));
    }

    // Sort by number of points (simplest first), then by number of facts
    unsolved.sort_by_key(|p| (p.2, p.3));

    println!("=== Simplest unsolved problems (by number of points) ===\n");
    for (name, def, npts, nfacts, gtype) in unsolved.iter().take(30) {
        println!("[{} pts, {} facts, {}] {}", npts, nfacts, gtype, name);
        println!("  {}\n", def);
    }
}
