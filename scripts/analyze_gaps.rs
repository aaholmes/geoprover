use geoprover::parser::parse_problem;
use geoprover::deduction::saturate;
use geoprover::proof_state::Relation;
use std::collections::HashMap;

/// Classify a goal relation into a string tag
fn goal_type(goal: &Relation) -> &'static str {
    match goal {
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

/// Compute a "closeness" score for an unsolved problem.
/// Higher = closer to solved.
/// Returns (score, details_string)
fn closeness_score(
    state: &geoprover::proof_state::ProofState,
) -> (f64, String) {
    let goal = match &state.goal {
        Some(g) => g.clone(),
        None => return (0.0, "no goal".to_string()),
    };

    let mut details = Vec::new();
    let total_facts = state.facts.len();

    match &goal {
        Relation::Parallel(a, b, c, d) => {
            let goal_pts = [*a, *b, *c, *d];
            // Count parallel facts with overlapping points
            let mut max_overlap = 0usize;
            let mut best_parallel = None;
            let mut para_count = 0;
            let mut perp_count = 0;
            let mut collinear_goal_pts = 0;

            for f in &state.facts {
                match f {
                    Relation::Parallel(pa, pb, pc, pd) => {
                        para_count += 1;
                        let fact_pts = [*pa, *pb, *pc, *pd];
                        let overlap = fact_pts.iter().filter(|p| goal_pts.contains(p)).count();
                        if overlap > max_overlap {
                            max_overlap = overlap;
                            best_parallel = Some(f.clone());
                        }
                    }
                    Relation::Perpendicular(pa, pb, pc, pd) => {
                        let fact_pts = [*pa, *pb, *pc, *pd];
                        let overlap = fact_pts.iter().filter(|p| goal_pts.contains(p)).count();
                        if overlap >= 2 {
                            perp_count += 1;
                        }
                    }
                    Relation::Collinear(ca, cb, cc) => {
                        let col_pts = [*ca, *cb, *cc];
                        if goal_pts.iter().any(|p| col_pts.contains(p)) {
                            collinear_goal_pts += 1;
                        }
                    }
                    _ => {}
                }
            }
            // Score: overlap with closest parallel + bonuses
            let mut score = max_overlap as f64 * 2.0;
            if perp_count >= 2 { score += 1.5; } // Two perp to same line -> parallel
            if collinear_goal_pts >= 2 { score += 0.5; }
            score += (total_facts as f64).log2() * 0.1;

            details.push(format!("parallel_facts={}, max_overlap={}", para_count, max_overlap));
            if let Some(bp) = &best_parallel {
                details.push(format!("best_match={:?}", bp));
            }
            if perp_count >= 2 {
                details.push(format!("perp_involving_goal_pts={} (potential perp-to-para)", perp_count));
            }
            details.push(format!("collinear_with_goal_pts={}", collinear_goal_pts));
            (score, details.join("; "))
        }

        Relation::Perpendicular(a, b, c, d) => {
            let goal_pts = [*a, *b, *c, *d];
            let mut max_overlap = 0usize;
            let mut perp_count = 0;
            let mut para_count_relevant = 0;

            for f in &state.facts {
                match f {
                    Relation::Perpendicular(pa, pb, pc, pd) => {
                        perp_count += 1;
                        let fact_pts = [*pa, *pb, *pc, *pd];
                        let overlap = fact_pts.iter().filter(|p| goal_pts.contains(p)).count();
                        if overlap > max_overlap { max_overlap = overlap; }
                    }
                    Relation::Parallel(pa, pb, pc, pd) => {
                        let fact_pts = [*pa, *pb, *pc, *pd];
                        let overlap = fact_pts.iter().filter(|p| goal_pts.contains(p)).count();
                        if overlap >= 2 { para_count_relevant += 1; }
                    }
                    _ => {}
                }
            }
            let mut score = max_overlap as f64 * 2.0;
            if para_count_relevant >= 1 && perp_count >= 1 { score += 1.5; } // perp+para transfer
            score += (total_facts as f64).log2() * 0.1;

            details.push(format!("perp_facts={}, max_overlap={}, para_relevant={}", perp_count, max_overlap, para_count_relevant));
            (score, details.join("; "))
        }

        Relation::Congruent(a, b, c, d) => {
            let goal_pts = [*a, *b, *c, *d];
            let mut max_overlap = 0usize;
            let mut cong_count = 0;
            let mut has_midpoint_relevant = false;
            let mut oncircle_relevant = 0;

            for f in &state.facts {
                match f {
                    Relation::Congruent(ca, cb, cc, cd) => {
                        cong_count += 1;
                        let fact_pts = [*ca, *cb, *cc, *cd];
                        let overlap = fact_pts.iter().filter(|p| goal_pts.contains(p)).count();
                        if overlap > max_overlap { max_overlap = overlap; }
                    }
                    Relation::Midpoint(m, ma, mb) => {
                        let mid_pts = [*m, *ma, *mb];
                        if goal_pts.iter().any(|p| mid_pts.contains(p)) {
                            has_midpoint_relevant = true;
                        }
                    }
                    Relation::OnCircle(pt, center) => {
                        if goal_pts.contains(pt) || goal_pts.contains(center) {
                            oncircle_relevant += 1;
                        }
                    }
                    _ => {}
                }
            }
            let mut score = max_overlap as f64 * 2.0;
            if has_midpoint_relevant { score += 1.0; }
            if oncircle_relevant >= 2 { score += 1.0; }
            score += (total_facts as f64).log2() * 0.1;

            details.push(format!("cong_facts={}, max_overlap={}, midpoint_relevant={}, oncircle_relevant={}",
                cong_count, max_overlap, has_midpoint_relevant, oncircle_relevant));
            (score, details.join("; "))
        }

        Relation::EqualAngle(a, b, c, d, e, f) => {
            let goal_pts = [*a, *b, *c, *d, *e, *f];
            let mut max_overlap = 0usize;
            let mut eqangle_count = 0;
            let mut parallel_relevant = 0;
            let mut perp_relevant = 0;
            let mut cyclic_relevant = 0;

            for fact in &state.facts {
                match fact {
                    Relation::EqualAngle(ea, eb, ec, ed, ee, ef) => {
                        eqangle_count += 1;
                        let fact_pts = [*ea, *eb, *ec, *ed, *ee, *ef];
                        let overlap = fact_pts.iter().filter(|p| goal_pts.contains(p)).count();
                        if overlap > max_overlap { max_overlap = overlap; }
                    }
                    Relation::Parallel(pa, pb, pc, pd) => {
                        let fact_pts = [*pa, *pb, *pc, *pd];
                        let overlap = fact_pts.iter().filter(|p| goal_pts.contains(p)).count();
                        if overlap >= 2 { parallel_relevant += 1; }
                    }
                    Relation::Perpendicular(pa, pb, pc, pd) => {
                        let fact_pts = [*pa, *pb, *pc, *pd];
                        let overlap = fact_pts.iter().filter(|p| goal_pts.contains(p)).count();
                        if overlap >= 2 { perp_relevant += 1; }
                    }
                    Relation::Cyclic(ca, cb, cc, cd) => {
                        let fact_pts = [*ca, *cb, *cc, *cd];
                        let overlap = fact_pts.iter().filter(|p| goal_pts.contains(p)).count();
                        if overlap >= 3 { cyclic_relevant += 1; }
                    }
                    _ => {}
                }
            }
            let mut score = max_overlap as f64 * 1.5;
            if parallel_relevant >= 1 { score += 1.0; } // parallel -> alternate interior
            if perp_relevant >= 1 { score += 0.8; } // perp -> right angle equality
            if cyclic_relevant >= 1 { score += 1.5; } // cyclic -> inscribed angle
            score += (total_facts as f64).log2() * 0.1;

            details.push(format!("eqangle_facts={}, max_overlap={}, parallel_relevant={}, perp_relevant={}, cyclic_relevant={}",
                eqangle_count, max_overlap, parallel_relevant, perp_relevant, cyclic_relevant));
            (score, details.join("; "))
        }

        Relation::Cyclic(a, b, c, d) => {
            let goal_pts = [*a, *b, *c, *d];
            let mut oncircle_matching = 0;
            let mut cong_equidistant = 0;

            for f in &state.facts {
                match f {
                    Relation::OnCircle(pt, _center) => {
                        if goal_pts.contains(pt) { oncircle_matching += 1; }
                    }
                    Relation::Congruent(ca, cb, cc, cd) => {
                        // Check for equidistant-from-center patterns
                        let fact_pts = [*ca, *cb, *cc, *cd];
                        let overlap = fact_pts.iter().filter(|p| goal_pts.contains(p)).count();
                        if overlap >= 2 { cong_equidistant += 1; }
                    }
                    _ => {}
                }
            }
            // How many of the 4 cyclic points are on some circle?
            let mut score = oncircle_matching as f64 * 2.0;
            if cong_equidistant >= 2 { score += 1.5; }
            score += (total_facts as f64).log2() * 0.1;

            details.push(format!("oncircle_matching={}/4, cong_equidistant_patterns={}",
                oncircle_matching, cong_equidistant));
            (score, details.join("; "))
        }

        Relation::Collinear(..) => {
            let mut collinear_count = 0;
            for f in &state.facts {
                if matches!(f, Relation::Collinear(..)) {
                    collinear_count += 1;
                }
            }
            let score = collinear_count as f64 * 0.5 + (total_facts as f64).log2() * 0.1;
            details.push(format!("collinear_facts={}", collinear_count));
            (score, details.join("; "))
        }

        Relation::Midpoint(..) => {
            let mut midpoint_count = 0;
            for f in &state.facts {
                if matches!(f, Relation::Midpoint(..)) {
                    midpoint_count += 1;
                }
            }
            let score = midpoint_count as f64 * 1.0 + (total_facts as f64).log2() * 0.1;
            details.push(format!("midpoint_facts={}", midpoint_count));
            (score, details.join("; "))
        }

        _ => {
            (0.0, "unknown goal type".to_string())
        }
    }
}

/// Detailed analysis of what facts exist and what's missing for a specific problem
fn detailed_analysis(name: &str, def: &str, state: &geoprover::proof_state::ProofState) -> String {
    let mut out = Vec::new();
    let goal = state.goal.as_ref().unwrap();
    let id_to_name: HashMap<u16, String> = state.name_to_id.iter().map(|(k, v)| (*v, k.clone())).collect();

    let name_pt = |id: u16| -> String {
        id_to_name.get(&id).cloned().unwrap_or_else(|| format!("#{}", id))
    };

    out.push(format!("  Problem: {}", name));
    out.push(format!("  Def: {}", def.split(" ? ").next().unwrap_or("?")));
    out.push(format!("  Goal: {} ({})", goal_type(goal), format_relation(goal, &id_to_name)));
    out.push(format!("  Points: {} | Facts after saturate: {}", state.objects.len(), state.facts.len()));

    // Show facts by type
    let mut fact_counts: HashMap<&str, usize> = HashMap::new();
    for f in &state.facts {
        let t = match f {
            Relation::Parallel(..) => "para",
            Relation::Perpendicular(..) => "perp",
            Relation::Congruent(..) => "cong",
            Relation::EqualAngle(..) => "eqangle",
            Relation::Collinear(..) => "coll",
            Relation::Cyclic(..) => "cyclic",
            Relation::Midpoint(..) => "midp",
            Relation::OnCircle(..) => "oncircle",
            Relation::EqualRatio(..) => "eqratio",
        };
        *fact_counts.entry(t).or_insert(0) += 1;
    }
    let mut counts_str: Vec<String> = fact_counts.iter()
        .map(|(k, v)| format!("{}={}", k, v))
        .collect();
    counts_str.sort();
    out.push(format!("  Fact breakdown: {}", counts_str.join(", ")));

    // Show facts most relevant to the goal
    match goal {
        Relation::Parallel(a, b, c, d) => {
            let goal_pts = [*a, *b, *c, *d];
            out.push(format!("  Goal: para({}, {}, {}, {})", name_pt(*a), name_pt(*b), name_pt(*c), name_pt(*d)));
            out.push("  Relevant parallel facts:".to_string());
            for f in &state.facts {
                if let Relation::Parallel(pa, pb, pc, pd) = f {
                    let fact_pts = [*pa, *pb, *pc, *pd];
                    let overlap = fact_pts.iter().filter(|p| goal_pts.contains(p)).count();
                    if overlap >= 2 {
                        out.push(format!("    para({}, {}, {}, {}) overlap={}", name_pt(*pa), name_pt(*pb), name_pt(*pc), name_pt(*pd), overlap));
                    }
                }
            }
            out.push("  Relevant perp facts:".to_string());
            for f in &state.facts {
                if let Relation::Perpendicular(pa, pb, pc, pd) = f {
                    let fact_pts = [*pa, *pb, *pc, *pd];
                    let overlap = fact_pts.iter().filter(|p| goal_pts.contains(p)).count();
                    if overlap >= 2 {
                        out.push(format!("    perp({}, {}, {}, {}) overlap={}", name_pt(*pa), name_pt(*pb), name_pt(*pc), name_pt(*pd), overlap));
                    }
                }
            }
            out.push("  Relevant equal angle facts:".to_string());
            for f in &state.facts {
                if let Relation::EqualAngle(ea, eb, ec, ed, ee, ef) = f {
                    let angle_pts = [*ea, *eb, *ec, *ed, *ee, *ef];
                    let overlap = angle_pts.iter().filter(|p| goal_pts.contains(p)).count();
                    if overlap >= 3 {
                        out.push(format!("    eqangle({},{},{}, {},{},{}) overlap={}",
                            name_pt(*ea), name_pt(*eb), name_pt(*ec), name_pt(*ed), name_pt(*ee), name_pt(*ef), overlap));
                    }
                }
            }
        }
        Relation::Perpendicular(a, b, c, d) => {
            let goal_pts = [*a, *b, *c, *d];
            out.push(format!("  Goal: perp({}, {}, {}, {})", name_pt(*a), name_pt(*b), name_pt(*c), name_pt(*d)));
            out.push("  Relevant perp facts:".to_string());
            for f in &state.facts {
                if let Relation::Perpendicular(pa, pb, pc, pd) = f {
                    let fact_pts = [*pa, *pb, *pc, *pd];
                    let overlap = fact_pts.iter().filter(|p| goal_pts.contains(p)).count();
                    if overlap >= 2 {
                        out.push(format!("    perp({}, {}, {}, {}) overlap={}", name_pt(*pa), name_pt(*pb), name_pt(*pc), name_pt(*pd), overlap));
                    }
                }
            }
            out.push("  Relevant parallel facts:".to_string());
            for f in &state.facts {
                if let Relation::Parallel(pa, pb, pc, pd) = f {
                    let fact_pts = [*pa, *pb, *pc, *pd];
                    let overlap = fact_pts.iter().filter(|p| goal_pts.contains(p)).count();
                    if overlap >= 2 {
                        out.push(format!("    para({}, {}, {}, {}) overlap={}", name_pt(*pa), name_pt(*pb), name_pt(*pc), name_pt(*pd), overlap));
                    }
                }
            }
        }
        Relation::Congruent(a, b, c, d) => {
            let goal_pts = [*a, *b, *c, *d];
            out.push(format!("  Goal: cong({}, {}, {}, {})", name_pt(*a), name_pt(*b), name_pt(*c), name_pt(*d)));
            out.push("  Relevant cong facts:".to_string());
            for f in &state.facts {
                if let Relation::Congruent(ca, cb, cc, cd) = f {
                    let fact_pts = [*ca, *cb, *cc, *cd];
                    let overlap = fact_pts.iter().filter(|p| goal_pts.contains(p)).count();
                    if overlap >= 2 {
                        out.push(format!("    cong({}, {}, {}, {}) overlap={}", name_pt(*ca), name_pt(*cb), name_pt(*cc), name_pt(*cd), overlap));
                    }
                }
            }
            out.push("  Relevant midpoint facts:".to_string());
            for f in &state.facts {
                if let Relation::Midpoint(m, ma, mb) = f {
                    if goal_pts.contains(m) || goal_pts.contains(ma) || goal_pts.contains(mb) {
                        out.push(format!("    midp({}, {}, {})", name_pt(*m), name_pt(*ma), name_pt(*mb)));
                    }
                }
            }
        }
        Relation::EqualAngle(a, b, c, d, e, f) => {
            let goal_pts = [*a, *b, *c, *d, *e, *f];
            out.push(format!("  Goal: eqangle({},{},{}, {},{},{})", name_pt(*a), name_pt(*b), name_pt(*c), name_pt(*d), name_pt(*e), name_pt(*f)));
            out.push("  Relevant eqangle facts:".to_string());
            for fact in &state.facts {
                if let Relation::EqualAngle(ea, eb, ec, ed, ee, ef) = fact {
                    let fact_pts = [*ea, *eb, *ec, *ed, *ee, *ef];
                    let overlap = fact_pts.iter().filter(|p| goal_pts.contains(p)).count();
                    if overlap >= 3 {
                        out.push(format!("    eqangle({},{},{}, {},{},{}) overlap={}",
                            name_pt(*ea), name_pt(*eb), name_pt(*ec), name_pt(*ed), name_pt(*ee), name_pt(*ef), overlap));
                    }
                }
            }
            out.push("  Relevant parallel/perp facts:".to_string());
            for fact in &state.facts {
                match fact {
                    Relation::Parallel(pa, pb, pc, pd) => {
                        let fact_pts = [*pa, *pb, *pc, *pd];
                        let overlap = fact_pts.iter().filter(|p| goal_pts.contains(p)).count();
                        if overlap >= 2 {
                            out.push(format!("    para({},{},{},{}) overlap={}", name_pt(*pa), name_pt(*pb), name_pt(*pc), name_pt(*pd), overlap));
                        }
                    }
                    Relation::Perpendicular(pa, pb, pc, pd) => {
                        let fact_pts = [*pa, *pb, *pc, *pd];
                        let overlap = fact_pts.iter().filter(|p| goal_pts.contains(p)).count();
                        if overlap >= 2 {
                            out.push(format!("    perp({},{},{},{}) overlap={}", name_pt(*pa), name_pt(*pb), name_pt(*pc), name_pt(*pd), overlap));
                        }
                    }
                    _ => {}
                }
            }
            out.push("  Relevant cyclic facts:".to_string());
            for fact in &state.facts {
                if let Relation::Cyclic(ca, cb, cc, cd) = fact {
                    let fact_pts = [*ca, *cb, *cc, *cd];
                    let overlap = fact_pts.iter().filter(|p| goal_pts.contains(p)).count();
                    if overlap >= 2 {
                        out.push(format!("    cyclic({},{},{},{}) overlap={}", name_pt(*ca), name_pt(*cb), name_pt(*cc), name_pt(*cd), overlap));
                    }
                }
            }
        }
        Relation::Cyclic(a, b, c, d) => {
            let goal_pts = [*a, *b, *c, *d];
            out.push(format!("  Goal: cyclic({}, {}, {}, {})", name_pt(*a), name_pt(*b), name_pt(*c), name_pt(*d)));
            out.push("  OnCircle facts for goal points:".to_string());
            for f in &state.facts {
                if let Relation::OnCircle(pt, center) = f {
                    if goal_pts.contains(pt) {
                        out.push(format!("    oncircle({}, {})", name_pt(*pt), name_pt(*center)));
                    }
                }
            }
            out.push("  Congruent facts involving goal points:".to_string());
            for f in &state.facts {
                if let Relation::Congruent(ca, cb, cc, cd) = f {
                    let fact_pts = [*ca, *cb, *cc, *cd];
                    let overlap = fact_pts.iter().filter(|p| goal_pts.contains(p)).count();
                    if overlap >= 2 {
                        out.push(format!("    cong({},{},{},{}) overlap={}", name_pt(*ca), name_pt(*cb), name_pt(*cc), name_pt(*cd), overlap));
                    }
                }
            }
        }
        Relation::Collinear(a, b, c) => {
            out.push(format!("  Goal: coll({}, {}, {})", name_pt(*a), name_pt(*b), name_pt(*c)));
            out.push("  Relevant collinear facts:".to_string());
            for f in &state.facts {
                if let Relation::Collinear(ca, cb, cc) = f {
                    if [*ca, *cb, *cc].iter().any(|p| [*a, *b, *c].contains(p)) {
                        out.push(format!("    coll({},{},{})", name_pt(*ca), name_pt(*cb), name_pt(*cc)));
                    }
                }
            }
        }
        Relation::Midpoint(m, a, b) => {
            out.push(format!("  Goal: midp({}, {}, {})", name_pt(*m), name_pt(*a), name_pt(*b)));
            out.push("  Relevant midpoint facts:".to_string());
            for f in &state.facts {
                if let Relation::Midpoint(fm, fa, fb) = f {
                    if [*fm, *fa, *fb].iter().any(|p| [*m, *a, *b].contains(p)) {
                        out.push(format!("    midp({},{},{})", name_pt(*fm), name_pt(*fa), name_pt(*fb)));
                    }
                }
            }
            out.push("  Relevant cong facts:".to_string());
            for f in &state.facts {
                if let Relation::Congruent(ca, cb, cc, cd) = f {
                    if [*ca, *cb, *cc, *cd].iter().any(|p| [*m, *a, *b].contains(p)) {
                        out.push(format!("    cong({},{},{},{})", name_pt(*ca), name_pt(*cb), name_pt(*cc), name_pt(*cd)));
                    }
                }
            }
            out.push("  Relevant collinear facts:".to_string());
            for f in &state.facts {
                if let Relation::Collinear(ca, cb, cc) = f {
                    if [*ca, *cb, *cc].iter().any(|p| [*m, *a, *b].contains(p)) {
                        out.push(format!("    coll({},{},{})", name_pt(*ca), name_pt(*cb), name_pt(*cc)));
                    }
                }
            }
        }
        _ => {}
    }

    out.join("\n")
}

fn format_relation(r: &Relation, id_to_name: &HashMap<u16, String>) -> String {
    let n = |id: u16| -> String {
        id_to_name.get(&id).cloned().unwrap_or_else(|| format!("#{}", id))
    };
    match r {
        Relation::Parallel(a, b, c, d) => format!("para({},{},{},{})", n(*a), n(*b), n(*c), n(*d)),
        Relation::Perpendicular(a, b, c, d) => format!("perp({},{},{},{})", n(*a), n(*b), n(*c), n(*d)),
        Relation::Congruent(a, b, c, d) => format!("cong({},{},{},{})", n(*a), n(*b), n(*c), n(*d)),
        Relation::EqualAngle(a, b, c, d, e, f) => format!("eqangle({},{},{},{},{},{})", n(*a), n(*b), n(*c), n(*d), n(*e), n(*f)),
        Relation::Collinear(a, b, c) => format!("coll({},{},{})", n(*a), n(*b), n(*c)),
        Relation::Cyclic(a, b, c, d) => format!("cyclic({},{},{},{})", n(*a), n(*b), n(*c), n(*d)),
        Relation::Midpoint(m, a, b) => format!("midp({},{},{})", n(*m), n(*a), n(*b)),
        Relation::OnCircle(p, c) => format!("oncircle({},{})", n(*p), n(*c)),
        Relation::EqualRatio(a, b, c, d, e, f, g, h) => format!("eqratio({},{},{},{},{},{},{},{})", n(*a), n(*b), n(*c), n(*d), n(*e), n(*f), n(*g), n(*h)),
    }
}

/// Analyze what missing inference patterns would help
fn analyze_missing_patterns(unsolved: &[(String, String, geoprover::proof_state::ProofState, f64, String)]) {
    println!("\n{}", "=".repeat(80));
    println!("========== MISSING INFERENCE PATTERN ANALYSIS ==========");
    println!("{}\n", "=".repeat(80));

    // Category 1: Has perps on goal lines but can't derive goal perp (orthocenter-like)
    let mut needs_orthocenter_theorem = Vec::new();
    // Category 2: Has parallel + eqangle but can't connect (directed angle issues)
    let mut needs_directed_angles = Vec::new();
    // Category 3: Has cong facts involving goal points but can't connect transitively
    let mut needs_triangle_congruence = Vec::new();
    // Category 4: Has cyclic/oncircle but missing some points
    let mut needs_cyclic_inference = Vec::new();
    // Category 5: Missing construction semantics (lc_tangent, angle_mirror, etc.)
    let mut needs_construction_semantics = Vec::new();

    for (name, def, state, _score, _details) in unsolved {
        let goal = state.goal.as_ref().unwrap();

        // Check for missing construction semantics
        let def_lower = def.to_lowercase();
        let has_stub_construction = def_lower.contains("lc_tangent")
            || def_lower.contains("cc_tangent")
            || def_lower.contains("angle_mirror")
            || def_lower.contains("intersection_lp")
            || def_lower.contains("intersection_lt")
            || def_lower.contains("intersection_tt")
            || def_lower.contains("2l1c")
            || def_lower.contains("3peq")
            || def_lower.contains("trisect")
            || def_lower.contains("trisegment")
            || def_lower.contains("risos")
            || def_lower.contains("pentagon")
            || def_lower.contains("e5128")
            || def_lower.contains("s_angle")
            || def_lower.contains("eqangle2");
        if has_stub_construction {
            needs_construction_semantics.push((name.clone(), def.clone()));
            continue; // Don't analyze these further since their facts are incomplete
        }

        match goal {
            Relation::Perpendicular(..) => {
                // Check if we have related perps that an orthocenter-like theorem could combine
                let perp_count = state.facts.iter().filter(|f| matches!(f, Relation::Perpendicular(..))).count();
                if perp_count >= 2 {
                    needs_orthocenter_theorem.push((name.clone(), def.clone()));
                }
            }
            Relation::EqualAngle(..) => {
                let has_parallel = state.facts.iter().any(|f| matches!(f, Relation::Parallel(..)));
                let has_perp = state.facts.iter().any(|f| matches!(f, Relation::Perpendicular(..)));
                if has_parallel || has_perp {
                    needs_directed_angles.push((name.clone(), def.clone()));
                }
            }
            Relation::Congruent(..) => {
                let cong_count = state.facts.iter().filter(|f| matches!(f, Relation::Congruent(..))).count();
                if cong_count >= 3 {
                    needs_triangle_congruence.push((name.clone(), def.clone()));
                }
            }
            Relation::Cyclic(..) => {
                let oncircle_count = state.facts.iter().filter(|f| matches!(f, Relation::OnCircle(..))).count();
                if oncircle_count >= 2 {
                    needs_cyclic_inference.push((name.clone(), def.clone()));
                }
            }
            _ => {}
        }
    }

    println!("--- Problems needing STUB CONSTRUCTION SEMANTICS ({}) ---", needs_construction_semantics.len());
    println!("These use constructions that generate no/incomplete facts:");
    // Count which stubs appear
    let mut stub_counts: HashMap<&str, usize> = HashMap::new();
    for (_name, def) in &needs_construction_semantics {
        for stub in &["lc_tangent", "cc_tangent", "angle_mirror", "intersection_lp", "intersection_lt",
                       "intersection_tt", "2l1c", "3peq", "trisect", "trisegment", "risos", "pentagon",
                       "e5128", "s_angle", "eqangle2"] {
            if def.contains(stub) {
                *stub_counts.entry(stub).or_insert(0) += 1;
            }
        }
    }
    let mut stub_list: Vec<(&&str, &usize)> = stub_counts.iter().collect();
    stub_list.sort_by(|a, b| b.1.cmp(a.1));
    for (stub, count) in &stub_list {
        println!("  {} — {} problems", stub, count);
    }

    println!("\n--- Problems needing ORTHOCENTER/ALTITUDE THEOREM ({}) ---", needs_orthocenter_theorem.len());
    println!("Pattern: have 2+ perpendiculars, need to derive another (e.g., two altitudes -> third)");
    for (name, _def) in &needs_orthocenter_theorem {
        println!("  {}", name);
    }

    println!("\n--- Problems needing DIRECTED ANGLE reasoning ({}) ---", needs_directed_angles.len());
    println!("Pattern: have parallel/perp facts, goal is eqangle, but current angle rules can't bridge");
    for (name, _def) in &needs_directed_angles {
        println!("  {}", name);
    }

    println!("\n--- Problems needing TRIANGLE CONGRUENCE/SAS/SSS ({}) ---", needs_triangle_congruence.len());
    println!("Pattern: have multiple congruent segments, goal is another congruence (need SAS/SSS/ASA)");
    for (name, _def) in &needs_triangle_congruence {
        println!("  {}", name);
    }

    println!("\n--- Problems needing CYCLIC QUADRILATERAL reasoning ({}) ---", needs_cyclic_inference.len());
    println!("Pattern: have OnCircle facts, goal is cyclic, but not all 4 points are on same circle");
    for (name, _def) in &needs_cyclic_inference {
        println!("  {}", name);
    }
}

fn main() {
    let content = std::fs::read_to_string("problems/jgex_ag_231.txt").unwrap();
    let lines: Vec<&str> = content.lines().collect();

    let mut solved_names = Vec::new();
    let mut unsolved_by_type: HashMap<String, Vec<(String, String, geoprover::proof_state::ProofState, f64, String)>> = HashMap::new();
    let mut parse_errors = Vec::new();
    let mut total = 0;
    let mut solved = 0;

    for chunk in lines.chunks(2) {
        if chunk.len() < 2 { continue; }
        let name = chunk[0].trim().to_string();
        let def = chunk[1].trim().to_string();
        let problem = format!("{}\n{}", name, def);

        match parse_problem(&problem) {
            Ok(mut state) => {
                total += 1;
                let _facts_before = state.facts.len();
                if saturate(&mut state) {
                    solved += 1;
                    solved_names.push(name.clone());
                } else {
                    let gt = goal_type(state.goal.as_ref().unwrap()).to_string();
                    let (score, details) = closeness_score(&state);
                    unsolved_by_type.entry(gt).or_default().push((name, def, state, score, details));
                }
            }
            Err(e) => {
                parse_errors.push((name, e.to_string()));
            }
        }
    }

    // Sort each group by closeness score (descending)
    for group in unsolved_by_type.values_mut() {
        group.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
    }

    println!("{}", "=".repeat(80));
    println!("========== JGEX-AG-231 GAP ANALYSIS ==========");
    println!("{}\n", "=".repeat(80));
    println!("Total parseable: {}", total);
    println!("Solved by deduction: {}", solved);
    println!("Parse errors: {}", parse_errors.len());
    println!("Unsolved: {}\n", total - solved);

    // Print solved problems
    println!("--- SOLVED ({}) ---", solved);
    for name in &solved_names {
        println!("  {}", name);
    }

    // Summary by goal type
    println!("\n--- UNSOLVED BY GOAL TYPE ---");
    let mut type_counts: Vec<(String, usize)> = unsolved_by_type.iter()
        .map(|(k, v)| (k.clone(), v.len()))
        .collect();
    type_counts.sort_by(|a, b| b.1.cmp(&a.1));
    for (gt, count) in &type_counts {
        println!("  {}: {} problems", gt, count);
    }

    // Parse errors
    if !parse_errors.is_empty() {
        println!("\n--- PARSE ERRORS ({}) ---", parse_errors.len());
        for (name, err) in &parse_errors {
            println!("  {}: {}", name, err);
        }
    }

    // Top 5 closest for each goal type
    println!("\n{}", "=".repeat(80));
    println!("========== TOP 5 CLOSEST TO SOLVED PER GOAL TYPE ==========");
    println!("{}", "=".repeat(80));

    for (gt, _count) in &type_counts {
        let group = &unsolved_by_type[gt];
        let top_n = group.len().min(5);
        println!("\n--- {} ({} unsolved, showing top {}) ---", gt.to_uppercase(), group.len(), top_n);

        for (name, def, state, score, closeness_details) in group.iter().take(top_n) {
            println!("\n  [score={:.1}] {}", score, name);
            println!("  closeness: {}", closeness_details);
            let analysis = detailed_analysis(name, def, state);
            println!("{}", analysis);
        }
    }

    // Collect all unsolved into a flat list for pattern analysis
    let all_unsolved: Vec<(String, String, geoprover::proof_state::ProofState, f64, String)> =
        unsolved_by_type.into_values().flatten().collect();
    analyze_missing_patterns(&all_unsolved);

    // Summary of what to implement
    println!("\n{}", "=".repeat(80));
    println!("========== RECOMMENDED DEDUCTION RULES TO IMPLEMENT ==========");
    println!("{}\n", "=".repeat(80));
    println!("Based on the gap analysis, here are the most impactful rules to add:\n");
    println!("1. CONSTRUCTION SEMANTICS (highest impact, ~30+ problems):");
    println!("   - lc_tangent: tangent line from point to circle");
    println!("   - risos (right isosceles): congruent legs + right angle");
    println!("   - angle_mirror: reflection preserving angles");
    println!("   - intersection_lp, intersection_lt, intersection_tt: line-perp, line-tangent, tangent-tangent intersections");
    println!("   - s_angle: numerical angle specification (needed for 60/30/15 degree problems)");
    println!("   - eqangle2: equal angle construction\n");
    println!("2. SAS/SSS/ASA TRIANGLE CONGRUENCE (medium-high impact):");
    println!("   - If |AB|=|DE|, |BC|=|EF|, angle(ABC)=angle(DEF) => |AC|=|DF| etc.");
    println!("   - This unlocks many cong goals that have partial segment congruences\n");
    println!("3. DIRECTED ANGLE SYSTEM (medium impact):");
    println!("   - Current eqangle uses vertex-form angles");
    println!("   - JGEX uses directed line angles: angle(line AB, line CD)");
    println!("   - Need proper directed angle storage and 8-arg eqangle reasoning\n");
    println!("4. PERPENDICULAR-FROM-ALTITUDES (specific theorem):");
    println!("   - If BH perp AC and CH perp AB, then AH perp BC (orthocenter theorem)");
    println!("   - Several problems need this specific deduction\n");
    println!("5. CYCLIC QUADRILATERAL CONVERSE:");
    println!("   - If angle(A,C,B) = angle(A,D,B) then A,B,C,D are cyclic");
    println!("   - Several cyclic goals could be proved this way\n");
    println!("6. SIMILAR TRIANGLES:");
    println!("   - AA similarity: two equal angles => triangles similar => proportional sides");
    println!("   - Enables many cong goals via ratio=1 cases");
}
