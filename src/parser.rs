use crate::proof_state::{ObjectType, ProofState, Relation};

#[derive(Debug)]
pub struct ParseError(pub String);

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ParseError: {}", self.0)
    }
}

impl std::error::Error for ParseError {}

/// Parse an AlphaGeometry JGEX DSL problem string into a ProofState.
///
/// Format:
/// ```text
/// problem_name
/// premises ? goal_predicate
/// ```
///
/// Premises are `;`-separated construction clauses.
/// Each clause: `output_points = action1, action2` or just `action args`
pub fn parse_problem(input: &str) -> Result<ProofState, ParseError> {
    let lines: Vec<&str> = input.lines().collect();
    if lines.len() < 2 {
        return Err(ParseError("Expected at least 2 lines (name and definition)".into()));
    }

    let definition = lines[1].trim();

    // Split on " ? " to get premises and goal
    let parts: Vec<&str> = definition.splitn(2, " ? ").collect();
    if parts.len() != 2 {
        return Err(ParseError("Expected ' ? ' separator between premises and goal".into()));
    }

    let premises_str = parts[0].trim();
    let goal_str = parts[1].trim();

    let mut state = ProofState::new();

    // Parse premises
    let clauses: Vec<&str> = premises_str.split("; ").collect();
    for clause in &clauses {
        parse_clause(clause.trim(), &mut state)?;
    }

    // Parse goal
    let goal = parse_goal(goal_str, &state)?;
    state.set_goal(goal);

    Ok(state)
}

/// Parse a single premise clause.
/// Format: `output_points = action1, action2` or `points = predicate args`
fn parse_clause(clause: &str, state: &mut ProofState) -> Result<(), ParseError> {
    if let Some((outputs_str, actions_str)) = clause.split_once(" = ") {
        let output_names: Vec<&str> = outputs_str.split_whitespace().collect();

        // Handle comma-separated multi-constraint definitions
        let actions: Vec<&str> = split_actions(actions_str);

        for action in &actions {
            parse_action(action.trim(), &output_names, state)?;
        }
    } else {
        return Err(ParseError(format!("Invalid clause (no '='): '{}'", clause)));
    }

    Ok(())
}

/// Split actions by comma, but respect that action args themselves are space-separated
fn split_actions(s: &str) -> Vec<&str> {
    s.split(", ").collect()
}

/// Parse a single action/predicate within a clause and add objects + facts
fn parse_action(
    action: &str,
    output_names: &[&str],
    state: &mut ProofState,
) -> Result<(), ParseError> {
    let tokens: Vec<&str> = action.split_whitespace().collect();
    if tokens.is_empty() {
        return Err(ParseError("Empty action".into()));
    }

    let predicate = tokens[0];
    // JGEX format repeats the single output name as the first arg for construction predicates
    // (e.g., `c = on_bline c a b` → args should be [a, b], not [c, a, b]).
    // Only strip for single-output predicates to avoid breaking shape predicates
    // (e.g., `a b c = triangle a b c` uses all args).
    let raw_args: Vec<&str> = tokens[1..].to_vec();
    let args: Vec<&str> = if !raw_args.is_empty()
        && output_names.len() == 1
        && raw_args[0] == output_names[0]
    {
        raw_args[1..].to_vec()
    } else {
        raw_args
    };

    match predicate {
        "triangle" => {
            // `a b c = triangle` — create 3 points, no collinearity
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
        }
        "iso_triangle" => {
            // `a b c = iso_triangle a b c` — isosceles triangle with |AB| = |AC|
            // output_names are the triangle vertices, args repeat them
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 3 {
                let a = state.id(args[0]);
                let b = state.id(args[1]);
                let c = state.id(args[2]);
                // |AB| = |AC| (isosceles at A)
                state.add_fact(Relation::congruent(a, b, a, c));
            }
        }
        "r_triangle" => {
            // Right triangle: `a b c = r_triangle a b c` — right angle at A
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 3 {
                let a = state.id(args[0]);
                let b = state.id(args[1]);
                let c = state.id(args[2]);
                state.add_fact(Relation::perpendicular(a, b, a, c));
            }
        }
        "midpoint" => {
            // `m = midpoint a b` — m is midpoint of AB
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 2 {
                let m = state.id(output_names[0]);
                let a = ensure_point(args[0], state)?;
                let b = ensure_point(args[1], state)?;
                state.add_fact(Relation::midpoint(m, a, b));
                state.add_fact(Relation::collinear(a, m, b));
                state.add_fact(Relation::congruent(a, m, m, b));
            }
        }
        "on_tline" => {
            // `x = on_tline x a b c` — point x on line through a perpendicular to bc
            // Creates: Perpendicular(a,x, b,c) — line AX ⊥ line BC
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 3 {
                let x = state.id(output_names[0]);
                let through = ensure_point(args[0], state)?;
                let b = ensure_point(args[1], state)?;
                let c = ensure_point(args[2], state)?;
                state.add_fact(Relation::perpendicular(through, x, b, c));
            }
        }
        "on_pline" => {
            // `x = on_pline x a b c` — point x on line through a parallel to bc
            // Creates: Parallel(a,x, b,c) — line AX || line BC
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 3 {
                let x = state.id(output_names[0]);
                let through = ensure_point(args[0], state)?;
                let b = ensure_point(args[1], state)?;
                let c = ensure_point(args[2], state)?;
                state.add_fact(Relation::parallel(through, x, b, c));
            }
        }
        "foot" => {
            // `x = foot a b c` — foot of perpendicular from a to line bc
            // Creates: Perpendicular(a,x, b,c) and Collinear(b,x,c)
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 3 {
                let x = state.id(output_names[0]);
                let a = ensure_point(args[0], state)?;
                let b = ensure_point(args[1], state)?;
                let c = ensure_point(args[2], state)?;
                state.add_fact(Relation::perpendicular(a, x, b, c));
                state.add_fact(Relation::collinear(b, x, c));
            }
        }
        "circumcenter" => {
            // `o = circumcenter a b c` — circumcenter of triangle ABC
            // Creates: |OA| = |OB|, |OB| = |OC|, OnCircle(a,o), OnCircle(b,o), OnCircle(c,o)
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 3 {
                let o = state.id(output_names[0]);
                let a = ensure_point(args[0], state)?;
                let b = ensure_point(args[1], state)?;
                let c = ensure_point(args[2], state)?;
                state.add_fact(Relation::congruent(o, a, o, b));
                state.add_fact(Relation::congruent(o, b, o, c));
                state.add_fact(Relation::on_circle(a, o));
                state.add_fact(Relation::on_circle(b, o));
                state.add_fact(Relation::on_circle(c, o));
            }
        }
        "incenter" => {
            // `i = incenter a b c` — incenter of triangle ABC
            // Creates: equal angle facts for angle bisectors
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 3 {
                let i = state.id(output_names[0]);
                let a = ensure_point(args[0], state)?;
                let b = ensure_point(args[1], state)?;
                let c = ensure_point(args[2], state)?;
                // Angle bisector at A: angle(B,A,I) = angle(I,A,C)
                state.add_fact(Relation::equal_angle(b, a, i, i, a, c));
                // Angle bisector at B: angle(A,B,I) = angle(I,B,C)
                state.add_fact(Relation::equal_angle(a, b, i, i, b, c));
                // Angle bisector at C: angle(A,C,I) = angle(I,C,B)
                state.add_fact(Relation::equal_angle(a, c, i, i, c, b));
            }
        }
        "centroid" => {
            // `g = centroid a b c` — centroid of triangle ABC
            // The centroid is the intersection of medians
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            // For centroid, we need midpoints. We'll create implicit midpoints.
            if args.len() >= 3 {
                let _g = state.id(output_names[0]);
                let _a = ensure_point(args[0], state)?;
                let _b = ensure_point(args[1], state)?;
                let _c = ensure_point(args[2], state)?;
                // Centroid facts are complex — we'll handle this via midpoints in construction
            }
        }
        "orthocenter" => {
            // `h = orthocenter a b c` — orthocenter
            // Creates: AH ⊥ BC, BH ⊥ AC, CH ⊥ AB
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 3 {
                let h = state.id(output_names[0]);
                let a = ensure_point(args[0], state)?;
                let b = ensure_point(args[1], state)?;
                let c = ensure_point(args[2], state)?;
                state.add_fact(Relation::perpendicular(a, h, b, c));
                state.add_fact(Relation::perpendicular(b, h, a, c));
                state.add_fact(Relation::perpendicular(c, h, a, b));
            }
        }
        "on_line" => {
            // `x = on_line a b` — point x on line AB
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 2 {
                let x = state.id(output_names[0]);
                let a = ensure_point(args[0], state)?;
                let b = ensure_point(args[1], state)?;
                state.add_fact(Relation::collinear(x, a, b));
            }
        }
        "on_circle" => {
            // `x = on_circle x o a` — point x on circle centered at o through a
            // Creates: |OX| = |OA| and OnCircle(x, o), OnCircle(a, o)
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 2 {
                let x = state.id(output_names[0]);
                let o = ensure_point(args[0], state)?;
                let a = ensure_point(args[1], state)?;
                state.add_fact(Relation::congruent(o, x, o, a));
                state.add_fact(Relation::on_circle(x, o));
                state.add_fact(Relation::on_circle(a, o));
            }
        }
        "angle_bisector" => {
            // `x = angle_bisector b a c` — point x on bisector of angle BAC
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 3 {
                let x = state.id(output_names[0]);
                let b = ensure_point(args[0], state)?;
                let a = ensure_point(args[1], state)?;
                let c = ensure_point(args[2], state)?;
                state.add_fact(Relation::equal_angle(b, a, x, x, a, c));
            }
        }
        "mirror" | "reflect" => {
            // `x = mirror a b` — reflect a over point b
            // Creates: Midpoint(b, a, x) — b is midpoint of AX
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 2 {
                let x = state.id(output_names[0]);
                let a = ensure_point(args[0], state)?;
                let b = ensure_point(args[1], state)?;
                state.add_fact(Relation::midpoint(b, a, x));
                state.add_fact(Relation::collinear(a, b, x));
                state.add_fact(Relation::congruent(a, b, b, x));
            }
        }
        "eq_triangle" => {
            // Equilateral triangle: either `x = eq_triangle x a b` (1 output)
            // or `a b c = eq_triangle a b c` (3 outputs)
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if output_names.len() == 1 && args.len() >= 2 {
                // Single output: x is new vertex on base (args[0], args[1])
                let x = state.id(output_names[0]);
                let a = ensure_point(args[0], state)?;
                let b = ensure_point(args[1], state)?;
                state.add_fact(Relation::congruent(x, a, x, b));
                state.add_fact(Relation::congruent(x, a, a, b));
            } else if args.len() >= 3 {
                // Three outputs: all three vertices given
                let a = state.id(args[0]);
                let b = state.id(args[1]);
                let c = state.id(args[2]);
                state.add_fact(Relation::congruent(a, b, a, c));
                state.add_fact(Relation::congruent(a, b, b, c));
            }
        }
        "eqdistance" => {
            // `x = eqdistance a b c` — |XA| = |BC|
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 3 {
                let x = state.id(output_names[0]);
                let a = ensure_point(args[0], state)?;
                let b = ensure_point(args[1], state)?;
                let c = ensure_point(args[2], state)?;
                state.add_fact(Relation::congruent(x, a, b, c));
            }
        }
        "on_bline" => {
            // `x = on_bline a b` — x on perpendicular bisector of AB
            // Creates: |XA| = |XB|
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 2 {
                let x = state.id(output_names[0]);
                let a = ensure_point(args[0], state)?;
                let b = ensure_point(args[1], state)?;
                state.add_fact(Relation::congruent(x, a, x, b));
            }
        }
        "circle" => {
            // `o = circle o a b c` — o is circumcenter of a,b,c (a point, not a circle object)
            // In JGEX format, first arg is the output name repeated, rest are the 3 points
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 3 {
                let o = state.id(output_names[0]);
                // args might be [o, a, b, c] or [a, b, c]
                let (a, b, c) = if args.len() >= 4 {
                    (ensure_point(args[1], state)?, ensure_point(args[2], state)?, ensure_point(args[3], state)?)
                } else {
                    (ensure_point(args[0], state)?, ensure_point(args[1], state)?, ensure_point(args[2], state)?)
                };
                state.add_fact(Relation::congruent(o, a, o, b));
                state.add_fact(Relation::congruent(o, b, o, c));
                state.add_fact(Relation::on_circle(a, o));
                state.add_fact(Relation::on_circle(b, o));
                state.add_fact(Relation::on_circle(c, o));
            }
        }
        "s_angle" => {
            // `x = s_angle a b N` — point x such that angle ABX = N degrees
            // This is a numerical angle specification — for now, just create the point
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
        }
        "r_trapezoid" => {
            // Right trapezoid
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 4 {
                let a = state.id(args[0]);
                let b = state.id(args[1]);
                let c = state.id(args[2]);
                let d = state.id(args[3]);
                state.add_fact(Relation::parallel(a, b, c, d));
                state.add_fact(Relation::perpendicular(a, d, a, b));
            }
        }
        "parallelogram" => {
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 4 {
                let a = state.id(args[0]);
                let b = state.id(args[1]);
                let c = state.id(args[2]);
                let d = state.id(args[3]);
                state.add_fact(Relation::parallel(a, b, c, d));
                state.add_fact(Relation::parallel(a, d, b, c));
                state.add_fact(Relation::congruent(a, b, c, d));
                state.add_fact(Relation::congruent(a, d, b, c));
            }
        }
        "rectangle" => {
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 4 {
                let a = state.id(args[0]);
                let b = state.id(args[1]);
                let c = state.id(args[2]);
                let d = state.id(args[3]);
                state.add_fact(Relation::parallel(a, b, c, d));
                state.add_fact(Relation::parallel(a, d, b, c));
                state.add_fact(Relation::congruent(a, b, c, d));
                state.add_fact(Relation::congruent(a, d, b, c));
                state.add_fact(Relation::perpendicular(a, b, a, d));
            }
        }
        "square" | "isquare" => {
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 4 {
                let a = state.id(args[0]);
                let b = state.id(args[1]);
                let c = state.id(args[2]);
                let d = state.id(args[3]);
                state.add_fact(Relation::parallel(a, b, c, d));
                state.add_fact(Relation::parallel(a, d, b, c));
                state.add_fact(Relation::congruent(a, b, b, c));
                state.add_fact(Relation::congruent(b, c, c, d));
                state.add_fact(Relation::congruent(c, d, d, a));
                state.add_fact(Relation::perpendicular(a, b, a, d));
            }
        }
        "trapezoid" => {
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 4 {
                let a = state.id(args[0]);
                let b = state.id(args[1]);
                let c = state.id(args[2]);
                let d = state.id(args[3]);
                state.add_fact(Relation::parallel(a, b, c, d));
            }
        }
        "iso_trapezoid" | "eq_trapezoid" => {
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 4 {
                let a = state.id(args[0]);
                let b = state.id(args[1]);
                let c = state.id(args[2]);
                let d = state.id(args[3]);
                state.add_fact(Relation::parallel(a, b, c, d));
                state.add_fact(Relation::congruent(a, d, b, c));
            }
        }
        "quadrangle" => {
            // Four free points, no constraints
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
        }
        "on_aline" => {
            // `x = on_aline A B C D E` — angle(B,A,X) = angle(D,C,E)
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 5 {
                let x = state.id(output_names[0]);
                let anchor = ensure_point(args[0], state)?;
                let b = ensure_point(args[1], state)?;
                let c = ensure_point(args[2], state)?;
                let d = ensure_point(args[3], state)?;
                let e = ensure_point(args[4], state)?;
                // angle(B, A, X) = angle(D, C, E)
                state.add_fact(Relation::equal_angle(b, anchor, x, d, c, e));
            }
        }
        "intersection_pp" => {
            // Intersection of two perpendicular lines
            // `x = intersection_pp p1 l1a l1b p2 l2a l2b`
            // x on line through p1 perp to l1a-l1b AND on line through p2 perp to l2a-l2b
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 6 {
                let x = state.id(output_names[0]);
                let p1 = ensure_point(args[0], state)?;
                let l1a = ensure_point(args[1], state)?;
                let l1b = ensure_point(args[2], state)?;
                let p2 = ensure_point(args[3], state)?;
                let l2a = ensure_point(args[4], state)?;
                let l2b = ensure_point(args[5], state)?;
                state.add_fact(Relation::perpendicular(p1, x, l1a, l1b));
                state.add_fact(Relation::perpendicular(p2, x, l2a, l2b));
            }
        }
        "nsquare" | "psquare" => {
            // `c = nsquare b a` or `d = psquare a b`
            // Construct a square vertex: |pivot-output| = |pivot-other|, perpendicular
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 2 {
                let x = state.id(output_names[0]);
                let pivot = ensure_point(args[0], state)?;
                let other = ensure_point(args[1], state)?;
                state.add_fact(Relation::congruent(pivot, x, pivot, other));
                state.add_fact(Relation::perpendicular(pivot, x, pivot, other));
            }
        }
        "eqangle2" => {
            // `x = eqangle2 d a b` → angle(x,d,a) = angle(a,d,b)
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 3 {
                let x = state.id(output_names[0]);
                let d = ensure_point(args[0], state)?;
                let a = ensure_point(args[1], state)?;
                let b = ensure_point(args[2], state)?;
                state.add_fact(Relation::equal_angle(x, d, a, a, d, b));
            }
        }
        "lc_tangent" => {
            // `x = lc_tangent x p o` — tangent from external point p to circle(o)
            // x is the tangent point: x on circle(o), radius ox ⊥ tangent line px
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 2 {
                let x = state.id(output_names[0]);
                let p = ensure_point(args[0], state)?;
                let o = ensure_point(args[1], state)?;
                state.add_fact(Relation::on_circle(x, o));
                state.add_fact(Relation::perpendicular(o, x, p, x));
            }
        }
        "cc_tangent" => {
            // Circle-circle tangent
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
        }
        "on_dia" => {
            // `x = on_dia a b` — x on circle with diameter AB → angle AXB = 90
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 2 {
                let x = state.id(output_names[0]);
                let a = ensure_point(args[0], state)?;
                let b = ensure_point(args[1], state)?;
                state.add_fact(Relation::perpendicular(a, x, b, x));
            }
        }
        "free" => {
            // Free point — just create it
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
        }
        "segment" => {
            // Two free points forming a segment
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
        }
        "shift" => {
            // `x = shift a b c` — x = c + (b - a), i.e., translation
            // Forms parallelogram: AB || CX, AC || BX, |AB|=|CX|, |AC|=|BX|
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 3 {
                let x = state.id(output_names[0]);
                let a = ensure_point(args[0], state)?;
                let b = ensure_point(args[1], state)?;
                let c = ensure_point(args[2], state)?;
                state.add_fact(Relation::parallel(a, b, c, x));
                state.add_fact(Relation::congruent(a, b, c, x));
                state.add_fact(Relation::parallel(a, c, b, x));
                state.add_fact(Relation::congruent(a, c, b, x));
            }
        }
        "intersection_cc" | "inter_cc" => {
            // `x = intersection_cc x c1 c2 rp` — intersection of two circles
            // x on circle(c1) and circle(c2), with |c2,x| = |c2,rp|
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 2 {
                let x = state.id(output_names[0]);
                let c1 = ensure_point(args[0], state)?;
                let c2 = ensure_point(args[1], state)?;
                state.add_fact(Relation::on_circle(x, c1));
                state.add_fact(Relation::on_circle(x, c2));
                if args.len() >= 3 {
                    let rp = ensure_point(args[2], state)?;
                    state.add_fact(Relation::congruent(c2, x, c2, rp));
                }
            }
        }
        "intersection_lc" | "inter_lc" => {
            // `x = intersection_lc x lp center rp` — intersection of line and circle
            // x on circle(center) with |center,x| = |center,rp|
            // Line defined by x and lp (or additional on_line constraints)
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 3 {
                let x = state.id(output_names[0]);
                let _lp = ensure_point(args[0], state)?;
                let center = ensure_point(args[1], state)?;
                let rp = ensure_point(args[2], state)?;
                state.add_fact(Relation::on_circle(x, center));
                state.add_fact(Relation::congruent(center, x, center, rp));
            }
        }
        "intersection_ll" | "inter_ll" => {
            // Intersection of two lines
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 4 {
                let x = state.id(output_names[0]);
                let a = ensure_point(args[0], state)?;
                let b = ensure_point(args[1], state)?;
                let c = ensure_point(args[2], state)?;
                let d = ensure_point(args[3], state)?;
                state.add_fact(Relation::collinear(x, a, b));
                state.add_fact(Relation::collinear(x, c, d));
            }
        }
        "risos" => {
            // Right isosceles triangle: right angle at first vertex, equal legs
            // `c a b = risos c a b` → right angle at c, |CA|=|CB|
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 3 {
                let c = state.id(args[0]);
                let a = state.id(args[1]);
                let b = state.id(args[2]);
                state.add_fact(Relation::perpendicular(c, a, c, b));
                state.add_fact(Relation::congruent(c, a, c, b));
            }
        }
        "angle_mirror" => {
            // `x = angle_mirror x a b c` → angle(c,a,b) = angle(b,a,x)
            // Mirror ray AC across ray AB at vertex a
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 3 {
                let x = state.id(output_names[0]);
                let a = ensure_point(args[0], state)?;
                let b = ensure_point(args[1], state)?;
                let c = ensure_point(args[2], state)?;
                state.add_fact(Relation::equal_angle(c, a, b, b, a, x));
            }
        }
        "pentagon" => {
            // 5 free points, no constraints (like quadrangle)
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
        }
        "trisegment" => {
            // `d x = trisegment d x c b` → d and x trisect segment CB
            // c-d-x-b collinear, |cd|=|dx|=|xb|
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if output_names.len() >= 2 && args.len() >= 2 {
                let d = state.id(output_names[0]);
                let x = state.id(output_names[1]);
                let c = ensure_point(args[0], state)?;
                let b = ensure_point(args[1], state)?;
                state.add_fact(Relation::collinear(c, d, b));
                state.add_fact(Relation::collinear(c, x, b));
                state.add_fact(Relation::collinear(d, x, b));
                state.add_fact(Relation::congruent(c, d, d, x));
                state.add_fact(Relation::congruent(d, x, x, b));
            }
        }
        "intersection_lt" | "inter_lt" => {
            // Intersection of line and tline (perpendicular line)
            // `x = intersection_lt x p1 p2 through la lb`
            // x on line(p1,p2) and on tline through `through` perp to (la,lb)
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 5 {
                let x = state.id(output_names[0]);
                let p1 = ensure_point(args[0], state)?;
                let p2 = ensure_point(args[1], state)?;
                let through = ensure_point(args[2], state)?;
                let la = ensure_point(args[3], state)?;
                let lb = ensure_point(args[4], state)?;
                state.add_fact(Relation::collinear(x, p1, p2));
                state.add_fact(Relation::perpendicular(through, x, la, lb));
            }
        }
        "intersection_tt" | "inter_tt" => {
            // Intersection of two tlines (perpendicular lines)
            // `x = intersection_tt x t1 la1 lb1 t2 la2 lb2`
            // x on tline through t1 perp to (la1,lb1) and tline through t2 perp to (la2,lb2)
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 6 {
                let x = state.id(output_names[0]);
                let t1 = ensure_point(args[0], state)?;
                let la1 = ensure_point(args[1], state)?;
                let lb1 = ensure_point(args[2], state)?;
                let t2 = ensure_point(args[3], state)?;
                let la2 = ensure_point(args[4], state)?;
                let lb2 = ensure_point(args[5], state)?;
                state.add_fact(Relation::perpendicular(t1, x, la1, lb1));
                state.add_fact(Relation::perpendicular(t2, x, la2, lb2));
            }
        }
        "intersection_lp" | "inter_lp" => {
            // Intersection of line and pline (parallel line)
            // `x = intersection_lp x p1 p2 through pa pb`
            // x on line(p1,p2) and on pline through `through` parallel to (pa,pb)
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
            if args.len() >= 5 {
                let x = state.id(output_names[0]);
                let p1 = ensure_point(args[0], state)?;
                let p2 = ensure_point(args[1], state)?;
                let through = ensure_point(args[2], state)?;
                let pa = ensure_point(args[3], state)?;
                let pb = ensure_point(args[4], state)?;
                state.add_fact(Relation::collinear(x, p1, p2));
                state.add_fact(Relation::parallel(through, x, pa, pb));
            }
        }
        _ => {
            // Unknown predicate — just create output points
            for name in output_names {
                state.add_object(name, ObjectType::Point);
            }
        }
    }

    Ok(())
}

/// Parse a goal predicate string into a Relation
fn parse_goal(goal_str: &str, state: &ProofState) -> Result<Relation, ParseError> {
    let tokens: Vec<&str> = goal_str.split_whitespace().collect();
    if tokens.is_empty() {
        return Err(ParseError("Empty goal".into()));
    }

    let predicate = tokens[0];
    let args = &tokens[1..];

    match predicate {
        "coll" => {
            require_args(predicate, args, 3)?;
            let a = lookup(args[0], state)?;
            let b = lookup(args[1], state)?;
            let c = lookup(args[2], state)?;
            Ok(Relation::collinear(a, b, c))
        }
        "cong" => {
            require_args(predicate, args, 4)?;
            let a = lookup(args[0], state)?;
            let b = lookup(args[1], state)?;
            let c = lookup(args[2], state)?;
            let d = lookup(args[3], state)?;
            Ok(Relation::congruent(a, b, c, d))
        }
        "perp" => {
            require_args(predicate, args, 4)?;
            let a = lookup(args[0], state)?;
            let b = lookup(args[1], state)?;
            let c = lookup(args[2], state)?;
            let d = lookup(args[3], state)?;
            Ok(Relation::perpendicular(a, b, c, d))
        }
        "para" => {
            require_args(predicate, args, 4)?;
            let a = lookup(args[0], state)?;
            let b = lookup(args[1], state)?;
            let c = lookup(args[2], state)?;
            let d = lookup(args[3], state)?;
            Ok(Relation::parallel(a, b, c, d))
        }
        "midp" => {
            require_args(predicate, args, 3)?;
            let m = lookup(args[0], state)?;
            let a = lookup(args[1], state)?;
            let b = lookup(args[2], state)?;
            Ok(Relation::midpoint(m, a, b))
        }
        "eqangle" => {
            require_args(predicate, args, 8)?;
            let a = lookup(args[0], state)?;
            let b = lookup(args[1], state)?;
            let c = lookup(args[2], state)?;
            let d = lookup(args[3], state)?;
            let e = lookup(args[4], state)?;
            let f = lookup(args[5], state)?;
            let g = lookup(args[6], state)?;
            let h = lookup(args[7], state)?;
            // eqangle a b c d e f g h means: angle between lines AB,CD = angle between lines EF,GH
            // We store as angle(a,b,c) = angle(d,e,f) where b,e are vertices
            // AlphaGeometry's eqangle has 8 args: directed angle of (ab, cd) = directed angle of (ef, gh)
            // For our Relation, we'll store the 8-arg version differently:
            // We map to EqualAngle using vertex form when possible
            // But AG's eqangle is about directed angles between lines, not vertex angles
            // For now, map eqangle a b c d e f g h to our representation
            // AG format: angle(line(a,b), line(c,d)) = angle(line(e,f), line(g,h))
            // This doesn't directly map to vertex-angle form. We need a separate representation.
            // For MVP, we'll use a simplified mapping.
            // When b==c (i.e., the two lines share a point), it's a vertex angle at b
            // eqangle a b b d e f f h → angle(a,b,d) = angle(e,f,h)
            if b == c && f == g {
                Ok(Relation::equal_angle(a, b, d, e, f, h))
            } else {
                // General directed angle — store as EqualAngle with the line endpoints
                // We'll use a new convention: store as the 8-arg form using two EqualAngle facts
                // For now, approximate: use (a,b,c) and (e,f,g) as the angle triples
                Ok(Relation::equal_angle(a, b, d, e, f, h))
            }
        }
        "cyclic" => {
            require_args(predicate, args, 4)?;
            let a = lookup(args[0], state)?;
            let b = lookup(args[1], state)?;
            let c = lookup(args[2], state)?;
            let d = lookup(args[3], state)?;
            Ok(Relation::cyclic(a, b, c, d))
        }
        "contri" | "simtri" => {
            // Congruent/similar triangles — complex goal, handle as needed
            require_args(predicate, args, 6)?;
            let _a = lookup(args[0], state)?;
            let _b = lookup(args[1], state)?;
            let _c = lookup(args[2], state)?;
            let _d = lookup(args[3], state)?;
            let _e = lookup(args[4], state)?;
            let _f = lookup(args[5], state)?;
            // For now, return a placeholder — congruent triangle goals need decomposition
            Err(ParseError(format!("Goal predicate '{}' not yet supported", predicate)))
        }
        _ => Err(ParseError(format!("Unknown goal predicate: '{}'", predicate))),
    }
}

fn require_args(pred: &str, args: &[&str], expected: usize) -> Result<(), ParseError> {
    if args.len() < expected {
        Err(ParseError(format!(
            "'{}' requires {} args, got {}",
            pred,
            expected,
            args.len()
        )))
    } else {
        Ok(())
    }
}

fn lookup(name: &str, state: &ProofState) -> Result<u16, ParseError> {
    state
        .try_id(name)
        .ok_or_else(|| ParseError(format!("Undefined point: '{}'", name)))
}

fn ensure_point(name: &str, state: &mut ProofState) -> Result<u16, ParseError> {
    Ok(state.add_object(name, ObjectType::Point))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_orthocenter() {
        let input = "orthocenter\na b c = triangle; h = on_tline b a c, on_tline c a b ? perp a h b c";
        let state = parse_problem(input).unwrap();
        // Should have 4 points: a, b, c, h
        assert_eq!(state.objects.len(), 4);
        assert!(state.try_id("a").is_some());
        assert!(state.try_id("b").is_some());
        assert!(state.try_id("c").is_some());
        assert!(state.try_id("h").is_some());
        // Goal should be Perpendicular(a,h,b,c)
        let a = state.id("a");
        let h = state.id("h");
        let b = state.id("b");
        let c = state.id("c");
        assert_eq!(state.goal, Some(Relation::perpendicular(a, h, b, c)));
    }

    #[test]
    fn test_parse_triangle() {
        let input = "test\na b c = triangle ? coll a b c";
        let state = parse_problem(input).unwrap();
        assert_eq!(state.objects.len(), 3);
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        assert_eq!(state.goal, Some(Relation::collinear(a, b, c)));
    }

    #[test]
    fn test_parse_midpoint() {
        let input = "test_midp\na b c = triangle; m = midpoint a b ? midp m a b";
        let state = parse_problem(input).unwrap();
        assert_eq!(state.objects.len(), 4); // a, b, c, m
        let m = state.id("m");
        let a = state.id("a");
        let b = state.id("b");
        // Should have Midpoint fact
        assert!(state.facts.contains(&Relation::midpoint(m, a, b)));
        // Should have Collinear fact
        assert!(state.facts.contains(&Relation::collinear(a, m, b)));
    }

    #[test]
    fn test_parse_goal_coll() {
        let input = "test\na b c = triangle ? coll a b c";
        let state = parse_problem(input).unwrap();
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        assert_eq!(state.goal, Some(Relation::collinear(a, b, c)));
    }

    #[test]
    fn test_parse_goal_cong() {
        let input = "test\na b c d = triangle; e = midpoint a b ? cong a e e b";
        let state = parse_problem(input).unwrap();
        let a = state.id("a");
        let e = state.id("e");
        let b = state.id("b");
        assert_eq!(state.goal, Some(Relation::congruent(a, e, e, b)));
    }

    #[test]
    fn test_parse_goal_cyclic() {
        let input = "test\na b c d = triangle ? cyclic a b c d";
        let state = parse_problem(input).unwrap();
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        let d = state.id("d");
        assert_eq!(state.goal, Some(Relation::cyclic(a, b, c, d)));
    }

    #[test]
    fn test_parse_multi_constraint() {
        // Point defined with multiple constraints (comma-separated actions)
        let input = "test\na b c = triangle; h = on_tline b a c, on_tline c a b ? perp a h b c";
        let state = parse_problem(input).unwrap();
        let b = state.id("b");
        let h = state.id("h");
        let a = state.id("a");
        let c = state.id("c");
        // Should have both perp facts
        assert!(state.facts.contains(&Relation::perpendicular(b, h, a, c)));
        assert!(state.facts.contains(&Relation::perpendicular(c, h, a, b)));
    }

    #[test]
    fn test_error_undefined_point() {
        let input = "test\na b c = triangle ? coll a b z";
        let result = parse_problem(input);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_iso_triangle() {
        let input = "test\na b c = iso_triangle a b c ? cong a b a c";
        let state = parse_problem(input).unwrap();
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        assert!(state.facts.contains(&Relation::congruent(a, b, a, c)));
    }

    #[test]
    fn test_parse_foot() {
        let input = "test\na b c = triangle; d = foot a b c ? perp a d b c";
        let state = parse_problem(input).unwrap();
        let a = state.id("a");
        let d = state.id("d");
        let b = state.id("b");
        let c = state.id("c");
        assert!(state.facts.contains(&Relation::perpendicular(a, d, b, c)));
        assert!(state.facts.contains(&Relation::collinear(b, d, c)));
    }

    #[test]
    fn test_parse_circumcenter() {
        let input = "test\na b c = triangle; o = circumcenter a b c ? cong o a o b";
        let state = parse_problem(input).unwrap();
        let o = state.id("o");
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        assert!(state.facts.contains(&Relation::congruent(o, a, o, b)));
        assert!(state.facts.contains(&Relation::congruent(o, b, o, c)));
    }

    #[test]
    fn test_parse_real_jgex_problem() {
        // First problem from jgex_ag_231.txt
        let input = "complete_004_6_GDD_FULL_81-109_101\na b c = triangle a b c; o = circle o a b c; h = midpoint h c b; d = on_line d o h, on_line d a b; e = on_tline e c c o, on_tline e a a o ? cyclic a o e d";
        let state = parse_problem(input).unwrap();
        assert!(state.try_id("a").is_some());
        assert!(state.try_id("b").is_some());
        assert!(state.try_id("c").is_some());
        assert!(state.try_id("o").is_some());
        assert!(state.try_id("h").is_some());
        assert!(state.try_id("d").is_some());
        assert!(state.try_id("e").is_some());
        assert_eq!(state.objects.len(), 7);
        // Goal should be cyclic
        let a = state.id("a");
        let o = state.id("o");
        let e = state.id("e");
        let d = state.id("d");
        assert_eq!(state.goal, Some(Relation::cyclic(a, o, e, d)));
    }

    // --- Construction predicate tests ---

    #[test]
    fn test_parse_r_triangle() {
        let input = "test\na b c = r_triangle a b c ? perp a b a c";
        let state = parse_problem(input).unwrap();
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        assert!(state.facts.contains(&Relation::perpendicular(a, b, a, c)));
        assert_eq!(state.objects.len(), 3);
    }

    #[test]
    fn test_parse_on_pline() {
        let input = "test\na b c = triangle; x = on_pline a b c ? para a x b c";
        let state = parse_problem(input).unwrap();
        let a = state.id("a");
        let x = state.id("x");
        let b = state.id("b");
        let c = state.id("c");
        assert!(state.facts.contains(&Relation::parallel(a, x, b, c)));
    }

    #[test]
    fn test_parse_on_line() {
        let input = "test\na b = segment; x = on_line a b ? coll x a b";
        let state = parse_problem(input).unwrap();
        let x = state.id("x");
        let a = state.id("a");
        let b = state.id("b");
        assert!(state.facts.contains(&Relation::collinear(x, a, b)));
    }

    #[test]
    fn test_parse_on_circle() {
        let input = "test\na b c = triangle; x = on_circle o a ? cong o x o a";
        let state = parse_problem(input).unwrap();
        let o = state.id("o");
        let x = state.id("x");
        let a = state.id("a");
        assert!(state.facts.contains(&Relation::congruent(o, x, o, a)));
    }

    #[test]
    fn test_parse_on_bline() {
        let input = "test\na b = segment; x = on_bline a b ? cong x a x b";
        let state = parse_problem(input).unwrap();
        let x = state.id("x");
        let a = state.id("a");
        let b = state.id("b");
        assert!(state.facts.contains(&Relation::congruent(x, a, x, b)));
    }

    #[test]
    fn test_parse_on_dia() {
        let input = "test\na b = segment; x = on_dia a b ? perp a x b x";
        let state = parse_problem(input).unwrap();
        let a = state.id("a");
        let x = state.id("x");
        let b = state.id("b");
        assert!(state.facts.contains(&Relation::perpendicular(a, x, b, x)));
    }

    #[test]
    fn test_parse_incenter() {
        let input = "test\na b c = triangle; i = incenter a b c ? cong a b a c";
        let state = parse_problem(input).unwrap();
        let i = state.id("i");
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        // Check all 3 angle bisector facts
        assert!(state.facts.contains(&Relation::equal_angle(b, a, i, i, a, c)));
        assert!(state.facts.contains(&Relation::equal_angle(a, b, i, i, b, c)));
        assert!(state.facts.contains(&Relation::equal_angle(a, c, i, i, c, b)));
    }

    #[test]
    fn test_parse_centroid() {
        let input = "test\na b c = triangle; g = centroid a b c ? coll a b c";
        let state = parse_problem(input).unwrap();
        // Centroid just creates the point, minimal facts
        assert!(state.try_id("g").is_some());
        assert_eq!(state.objects.len(), 4);
    }

    #[test]
    fn test_parse_orthocenter_action() {
        let input = "test\na b c = triangle; h = orthocenter a b c ? perp a h b c";
        let state = parse_problem(input).unwrap();
        let h = state.id("h");
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        assert!(state.facts.contains(&Relation::perpendicular(a, h, b, c)));
        assert!(state.facts.contains(&Relation::perpendicular(b, h, a, c)));
        assert!(state.facts.contains(&Relation::perpendicular(c, h, a, b)));
    }

    #[test]
    fn test_parse_angle_bisector() {
        let input = "test\na b c = triangle; x = angle_bisector b a c ? cong a b a c";
        let state = parse_problem(input).unwrap();
        let x = state.id("x");
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        assert!(state.facts.contains(&Relation::equal_angle(b, a, x, x, a, c)));
    }

    #[test]
    fn test_parse_mirror() {
        let input = "test\na b = segment; x = mirror a b ? midp b a x";
        let state = parse_problem(input).unwrap();
        let x = state.id("x");
        let a = state.id("a");
        let b = state.id("b");
        assert!(state.facts.contains(&Relation::midpoint(b, a, x)));
        assert!(state.facts.contains(&Relation::collinear(a, b, x)));
        assert!(state.facts.contains(&Relation::congruent(a, b, b, x)));
    }

    #[test]
    fn test_parse_reflect() {
        // "reflect" is an alias for "mirror"
        let input = "test\na b = segment; x = reflect a b ? midp b a x";
        let state = parse_problem(input).unwrap();
        let x = state.id("x");
        let a = state.id("a");
        let b = state.id("b");
        assert!(state.facts.contains(&Relation::midpoint(b, a, x)));
    }

    #[test]
    fn test_parse_eq_triangle() {
        // 3-output form: a b c = eq_triangle a b c
        let input = "test\na b c = eq_triangle a b c ? cong a b b c";
        let state = parse_problem(input).unwrap();
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        // x=a, base=(b,c): |ab|=|ac| and |ab|=|bc|
        assert!(state.facts.contains(&Relation::congruent(a, b, a, c)));
        assert!(state.facts.contains(&Relation::congruent(a, b, b, c)));
    }

    #[test]
    fn test_parse_eq_triangle_single_output() {
        // 1-output form: d = eq_triangle d a b (construct vertex d on base ab)
        let input = "test\na b c = triangle a b c; d = eq_triangle d a b ? cong d a d b";
        let state = parse_problem(input).unwrap();
        let a = state.id("a");
        let b = state.id("b");
        let d = state.id("d");
        // |da| = |db| and |da| = |ab|
        assert!(state.facts.contains(&Relation::congruent(d, a, d, b)));
        assert!(state.facts.contains(&Relation::congruent(a, b, a, d)));
    }

    #[test]
    fn test_parse_parallelogram() {
        let input = "test\na b c d = parallelogram a b c d ? para a b c d";
        let state = parse_problem(input).unwrap();
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        let d = state.id("d");
        assert!(state.facts.contains(&Relation::parallel(a, b, c, d)));
        assert!(state.facts.contains(&Relation::parallel(a, d, b, c)));
        assert!(state.facts.contains(&Relation::congruent(a, b, c, d)));
        assert!(state.facts.contains(&Relation::congruent(a, d, b, c)));
    }

    #[test]
    fn test_parse_rectangle() {
        let input = "test\na b c d = rectangle a b c d ? perp a b a d";
        let state = parse_problem(input).unwrap();
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        let d = state.id("d");
        assert!(state.facts.contains(&Relation::parallel(a, b, c, d)));
        assert!(state.facts.contains(&Relation::parallel(a, d, b, c)));
        assert!(state.facts.contains(&Relation::congruent(a, b, c, d)));
        assert!(state.facts.contains(&Relation::congruent(a, d, b, c)));
        assert!(state.facts.contains(&Relation::perpendicular(a, b, a, d)));
    }

    #[test]
    fn test_parse_square() {
        let input = "test\na b c d = square a b c d ? cong a b c d";
        let state = parse_problem(input).unwrap();
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        let d = state.id("d");
        assert!(state.facts.contains(&Relation::congruent(a, b, b, c)));
        assert!(state.facts.contains(&Relation::congruent(b, c, c, d)));
        assert!(state.facts.contains(&Relation::congruent(c, d, d, a)));
        assert!(state.facts.contains(&Relation::perpendicular(a, b, a, d)));
        assert!(state.facts.contains(&Relation::parallel(a, b, c, d)));
        assert!(state.facts.contains(&Relation::parallel(a, d, b, c)));
    }

    #[test]
    fn test_parse_trapezoid() {
        let input = "test\na b c d = trapezoid a b c d ? para a b c d";
        let state = parse_problem(input).unwrap();
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        let d = state.id("d");
        assert!(state.facts.contains(&Relation::parallel(a, b, c, d)));
    }

    #[test]
    fn test_parse_iso_trapezoid() {
        let input = "test\na b c d = iso_trapezoid a b c d ? cong a d b c";
        let state = parse_problem(input).unwrap();
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        let d = state.id("d");
        assert!(state.facts.contains(&Relation::parallel(a, b, c, d)));
        assert!(state.facts.contains(&Relation::congruent(a, d, b, c)));
    }

    #[test]
    fn test_parse_r_trapezoid() {
        let input = "test\na b c d = r_trapezoid a b c d ? perp a d a b";
        let state = parse_problem(input).unwrap();
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        let d = state.id("d");
        assert!(state.facts.contains(&Relation::parallel(a, b, c, d)));
        assert!(state.facts.contains(&Relation::perpendicular(a, d, a, b)));
    }

    #[test]
    fn test_parse_circle_circumcenter() {
        let input = "test\na b c = triangle; o = circle o a b c ? cong o a o c";
        let state = parse_problem(input).unwrap();
        let o = state.id("o");
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        assert!(state.facts.contains(&Relation::congruent(o, a, o, b)));
        assert!(state.facts.contains(&Relation::congruent(o, b, o, c)));
    }

    #[test]
    fn test_parse_circle_3args() {
        // circle with only 3 args (no repeated output name)
        let input = "test\na b c = triangle; o = circle a b c ? cong o a o b";
        let state = parse_problem(input).unwrap();
        let o = state.id("o");
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        assert!(state.facts.contains(&Relation::congruent(o, a, o, b)));
        assert!(state.facts.contains(&Relation::congruent(o, b, o, c)));
    }

    #[test]
    fn test_parse_shift() {
        let input = "test\na b c = triangle; x = shift a b c ? para a b c x";
        let state = parse_problem(input).unwrap();
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        let x = state.id("x");
        assert!(state.facts.contains(&Relation::parallel(a, b, c, x)));
        assert!(state.facts.contains(&Relation::congruent(a, b, c, x)));
    }

    #[test]
    fn test_parse_intersection_ll() {
        let input = "test\na b c d = triangle; x = intersection_ll a b c d ? coll x a b";
        let state = parse_problem(input).unwrap();
        let x = state.id("x");
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        let d = state.id("d");
        assert!(state.facts.contains(&Relation::collinear(x, a, b)));
        assert!(state.facts.contains(&Relation::collinear(x, c, d)));
    }

    // --- Goal predicate tests ---

    #[test]
    fn test_parse_goal_perp() {
        let input = "test\na b c d = triangle ? perp a b c d";
        let state = parse_problem(input).unwrap();
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        let d = state.id("d");
        assert_eq!(state.goal, Some(Relation::perpendicular(a, b, c, d)));
    }

    #[test]
    fn test_parse_goal_midp() {
        let input = "test\na b c = triangle; m = midpoint a b ? midp m a b";
        let state = parse_problem(input).unwrap();
        let m = state.id("m");
        let a = state.id("a");
        let b = state.id("b");
        assert_eq!(state.goal, Some(Relation::midpoint(m, a, b)));
    }

    #[test]
    fn test_parse_goal_para() {
        let input = "test\na b c d = triangle ? para a b c d";
        let state = parse_problem(input).unwrap();
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        let d = state.id("d");
        assert_eq!(state.goal, Some(Relation::parallel(a, b, c, d)));
    }

    // --- Negative / error tests ---

    #[test]
    fn test_error_missing_question_separator() {
        let input = "test\na b c = triangle coll a b c";
        let result = parse_problem(input);
        assert!(result.is_err());
        assert!(result.unwrap_err().0.contains("separator"));
    }

    #[test]
    fn test_error_single_line() {
        let input = "only_one_line";
        let result = parse_problem(input);
        assert!(result.is_err());
        assert!(result.unwrap_err().0.contains("at least 2 lines"));
    }

    #[test]
    fn test_error_empty_input() {
        let input = "";
        let result = parse_problem(input);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_too_few_goal_args() {
        let input = "test\na b c = triangle ? cong a b";
        let result = parse_problem(input);
        assert!(result.is_err());
        assert!(result.unwrap_err().0.contains("requires"));
    }

    #[test]
    fn test_error_unknown_goal_predicate() {
        let input = "test\na b c = triangle ? frobnicate a b c";
        let result = parse_problem(input);
        assert!(result.is_err());
        assert!(result.unwrap_err().0.contains("Unknown goal"));
    }

    #[test]
    fn test_parse_error_display() {
        let e = ParseError("test error".into());
        assert_eq!(format!("{}", e), "ParseError: test error");
    }

    #[test]
    fn test_parse_unknown_action_creates_points() {
        // Unknown predicates should still create output points
        let input = "test\na b = unknown_pred x y ? coll a b a";
        let state = parse_problem(input).unwrap();
        assert!(state.try_id("a").is_some());
        assert!(state.try_id("b").is_some());
    }

    #[test]
    fn test_parse_s_angle_creates_point() {
        let input = "test\na b = segment; x = s_angle a b 90 ? coll x a b";
        let state = parse_problem(input).unwrap();
        assert!(state.try_id("x").is_some());
    }

    #[test]
    fn test_parse_free_point() {
        let input = "test\na = free ? coll a a a";
        let state = parse_problem(input).unwrap();
        assert!(state.try_id("a").is_some());
    }

    #[test]
    fn test_parse_eqangle_vertex_form() {
        // When b==c and f==g, eqangle maps to vertex-angle form
        let input = "test\na b c d = triangle ? eqangle a b b d c d d a";
        let state = parse_problem(input).unwrap();
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        let d = state.id("d");
        // b==c (args[1]==args[2]) and f==g (args[5]==args[6])
        // Should be equal_angle(a, b, d, c, d, a)
        assert_eq!(state.goal, Some(Relation::equal_angle(a, b, d, c, d, a)));
    }

    #[test]
    fn test_parse_eqangle_general_form() {
        // When b!=c or f!=g, general directed angle form
        let input = "test\na b c d e f g h = triangle ? eqangle a b c d e f g h";
        let state = parse_problem(input).unwrap();
        // Uses the fallback: equal_angle(a, b, d, e, f, h)
        let a = state.id("a");
        let b = state.id("b");
        let d = state.id("d");
        let e = state.id("e");
        let f = state.id("f");
        let h = state.id("h");
        assert_eq!(state.goal, Some(Relation::equal_angle(a, b, d, e, f, h)));
    }

    #[test]
    fn test_parse_contri_goal_unsupported() {
        let input = "test\na b c d e f = triangle ? contri a b c d e f";
        let result = parse_problem(input);
        assert!(result.is_err());
        assert!(result.unwrap_err().0.contains("not yet supported"));
    }

    #[test]
    fn test_parse_simtri_goal_unsupported() {
        let input = "test\na b c d e f = triangle ? simtri a b c d e f";
        let result = parse_problem(input);
        assert!(result.is_err());
        assert!(result.unwrap_err().0.contains("not yet supported"));
    }

    // --- Stub predicates that create points but no facts ---

    #[test]
    fn test_parse_eqdistance_creates_point() {
        let input = "test\na b c = triangle; x = eqdistance a b c ? coll x a b";
        let state = parse_problem(input).unwrap();
        assert!(state.try_id("x").is_some());
    }

    #[test]
    fn test_parse_eqangle2_creates_point() {
        let input = "test\na b c = triangle; x = eqangle2 a b c ? coll x a b";
        let state = parse_problem(input).unwrap();
        assert!(state.try_id("x").is_some());
    }

    #[test]
    fn test_parse_lc_tangent_creates_point() {
        let input = "test\na b = segment; x = lc_tangent a b ? coll x a b";
        let state = parse_problem(input).unwrap();
        assert!(state.try_id("x").is_some());
    }

    #[test]
    fn test_parse_cc_tangent_creates_point() {
        let input = "test\na b = segment; x = cc_tangent a b ? coll x a b";
        let state = parse_problem(input).unwrap();
        assert!(state.try_id("x").is_some());
    }

    #[test]
    fn test_parse_intersection_cc_creates_point() {
        let input = "test\na b c = triangle; x = intersection_cc a b c ? coll x a b";
        let state = parse_problem(input).unwrap();
        assert!(state.try_id("x").is_some());
    }

    #[test]
    fn test_parse_intersection_lc_creates_point() {
        let input = "test\na b c = triangle; x = intersection_lc a b c ? coll x a b";
        let state = parse_problem(input).unwrap();
        assert!(state.try_id("x").is_some());
    }

    #[test]
    fn test_parse_inter_cc_alias() {
        let input = "test\na b c = triangle; x = inter_cc a b c ? coll x a b";
        let state = parse_problem(input).unwrap();
        assert!(state.try_id("x").is_some());
    }

    #[test]
    fn test_parse_inter_lc_alias() {
        let input = "test\na b c = triangle; x = inter_lc a b c ? coll x a b";
        let state = parse_problem(input).unwrap();
        assert!(state.try_id("x").is_some());
    }

    #[test]
    fn test_parse_inter_ll_alias() {
        let input = "test\na b c d = triangle; x = inter_ll a b c d ? coll x a b";
        let state = parse_problem(input).unwrap();
        let x = state.id("x");
        let a = state.id("a");
        let b = state.id("b");
        assert!(state.facts.contains(&Relation::collinear(x, a, b)));
    }

    // --- Edge case: on_tline and on_pline with fewer args ---

    #[test]
    fn test_parse_on_tline_fewer_args() {
        // If on_tline has fewer than 3 args, should still create the output point
        let input = "test\na b = segment; x = on_tline a b ? coll x a b";
        let state = parse_problem(input).unwrap();
        assert!(state.try_id("x").is_some());
        // No perpendicular fact should be added (too few args)
    }

    // --- Multi-output point clause ---

    #[test]
    fn test_parse_multiple_output_points() {
        // segment creates two output points
        let input = "test\na b = segment ? coll a b a";
        let state = parse_problem(input).unwrap();
        assert!(state.try_id("a").is_some());
        assert!(state.try_id("b").is_some());
    }

    // --- Circle with 4 args (output name repeated) ---

    #[test]
    fn test_parse_circle_4args_with_repeat() {
        let input = "test\na b c = triangle; o = circle o a b c ? cong o a o c";
        let state = parse_problem(input).unwrap();
        let o = state.id("o");
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        assert!(state.facts.contains(&Relation::congruent(o, a, o, b)));
        assert!(state.facts.contains(&Relation::congruent(o, b, o, c)));
    }

    // --- Whitespace and formatting edge cases ---

    #[test]
    fn test_parse_trailing_whitespace() {
        let input = "test  \n  a b c = triangle ? coll a b c  ";
        let state = parse_problem(input).unwrap();
        assert_eq!(state.objects.len(), 3);
    }

    // --- Missing constructors (TDD: write tests first) ---

    #[test]
    fn test_parse_eq_trapezoid() {
        // eq_trapezoid = isosceles trapezoid: AB || CD, |AD| = |BC|
        let input = "test\na b c d = eq_trapezoid a b c d ? para a b c d";
        let state = parse_problem(input).unwrap();
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        let d = state.id("d");
        assert!(state.facts.contains(&Relation::parallel(a, b, c, d)));
        assert!(state.facts.contains(&Relation::congruent(a, d, b, c)));
    }

    #[test]
    fn test_parse_quadrangle() {
        // quadrangle = 4 free points, no constraints
        let input = "test\na b c d = quadrangle a b c d ? coll a b c";
        let state = parse_problem(input).unwrap();
        assert_eq!(state.objects.len(), 4);
        assert!(state.try_id("a").is_some());
        assert!(state.try_id("d").is_some());
        // No facts should be added
        assert!(state.facts.is_empty());
    }

    #[test]
    fn test_parse_isquare() {
        // isquare = same as square
        let input = "test\na b c d = isquare a b c d ? cong a b b c";
        let state = parse_problem(input).unwrap();
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        let d = state.id("d");
        assert!(state.facts.contains(&Relation::congruent(a, b, b, c)));
        assert!(state.facts.contains(&Relation::perpendicular(a, b, a, d)));
        assert!(state.facts.contains(&Relation::parallel(a, b, c, d)));
    }

    #[test]
    fn test_parse_eqdistance() {
        // eqdistance x a b c → |XA| = |BC|
        let input = "test\na b c = triangle; g = on_line a b, eqdistance a b c ? cong g a b c";
        let state = parse_problem(input).unwrap();
        let g = state.id("g");
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        assert!(state.facts.contains(&Relation::congruent(g, a, b, c)));
    }

    #[test]
    fn test_parse_on_aline() {
        // on_aline x A B C D E → angle(B,A,X) = angle(D,C,E)
        let input = "test\na b c = triangle; s = on_aline a b c b a ? cong a b a c";
        let state = parse_problem(input).unwrap();
        let s = state.id("s");
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        // on_aline with args [a, b, c, b, a] after stripping s
        // A=a, B=b, C=c, D=b, E=a → angle(b, a, s) = angle(b, c, a)
        assert!(state.facts.contains(&Relation::equal_angle(b, a, s, b, c, a)));
    }

    #[test]
    fn test_parse_intersection_pp() {
        // intersection_pp d a b c c a b → orthocenter: Perp(a,d,b,c) + Perp(c,d,a,b)
        let input = "test\na b c = triangle; d = intersection_pp a b c c a b ? perp a d b c";
        let state = parse_problem(input).unwrap();
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        let d = state.id("d");
        assert!(state.facts.contains(&Relation::perpendicular(a, d, b, c)));
        assert!(state.facts.contains(&Relation::perpendicular(c, d, a, b)));
    }

    #[test]
    fn test_parse_nsquare() {
        // nsquare c b a → |BC| = |BA|, BC ⊥ BA
        let input = "test\na b = segment; c = nsquare b a ? cong b c b a";
        let state = parse_problem(input).unwrap();
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        assert!(state.facts.contains(&Relation::congruent(b, c, b, a)));
        assert!(state.facts.contains(&Relation::perpendicular(b, c, b, a)));
    }

    #[test]
    fn test_parse_psquare() {
        // psquare d a b → |AD| = |AB|, AD ⊥ AB
        let input = "test\na b = segment; d = psquare a b ? cong a d a b";
        let state = parse_problem(input).unwrap();
        let a = state.id("a");
        let b = state.id("b");
        let d = state.id("d");
        assert!(state.facts.contains(&Relation::congruent(a, d, a, b)));
        assert!(state.facts.contains(&Relation::perpendicular(a, d, a, b)));
    }

    // --- New coverage tests ---

    #[test]
    fn test_parse_circumcenter_oncircle_facts() {
        // Circumcenter should generate OnCircle facts for all three vertices
        let input = "test\na b c = triangle; o = circumcenter a b c ? cong o a o b";
        let state = parse_problem(input).unwrap();
        let o = state.id("o");
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        assert!(state.facts.contains(&Relation::on_circle(a, o)));
        assert!(state.facts.contains(&Relation::on_circle(b, o)));
        assert!(state.facts.contains(&Relation::on_circle(c, o)));
    }

    #[test]
    fn test_parse_circle_oncircle_facts() {
        // circle predicate should generate OnCircle facts
        let input = "test\na b c = triangle; o = circle o a b c ? cong o a o b";
        let state = parse_problem(input).unwrap();
        let o = state.id("o");
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        assert!(state.facts.contains(&Relation::on_circle(a, o)));
        assert!(state.facts.contains(&Relation::on_circle(b, o)));
        assert!(state.facts.contains(&Relation::on_circle(c, o)));
    }

    #[test]
    fn test_parse_on_circle_oncircle_facts() {
        // on_circle predicate should generate OnCircle facts for both points
        let input = "test\na b c = triangle; x = on_circle o a ? cong o x o a";
        let state = parse_problem(input).unwrap();
        let o = state.id("o");
        let x = state.id("x");
        let a = state.id("a");
        assert!(state.facts.contains(&Relation::on_circle(x, o)));
        assert!(state.facts.contains(&Relation::on_circle(a, o)));
    }

    #[test]
    fn test_parse_segment_standalone() {
        // segment creates exactly two points with no facts
        let input = "test\na b = segment ? coll a b a";
        let state = parse_problem(input).unwrap();
        assert!(state.try_id("a").is_some());
        assert!(state.try_id("b").is_some());
        assert_eq!(state.objects.len(), 2);
        assert!(state.facts.is_empty());
    }

    #[test]
    fn test_error_lookup_in_multi_constraint() {
        // Second action references undefined point z
        let input = "test\na b c = triangle; x = on_line a b, on_line z b ? coll a b c";
        let result = parse_problem(input);
        // This should succeed because ensure_point creates z on the fly
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_on_pline_fewer_args() {
        // on_pline with fewer than 3 args — should still create output point
        let input = "test\na b = segment; x = on_pline a b ? coll x a b";
        let state = parse_problem(input).unwrap();
        assert!(state.try_id("x").is_some());
    }

    #[test]
    fn test_parse_midpoint_congruent_fact() {
        // Midpoint should add congruent(a,m,m,b) fact
        let input = "test\na b = segment; m = midpoint a b ? cong a m m b";
        let state = parse_problem(input).unwrap();
        let a = state.id("a");
        let b = state.id("b");
        let m = state.id("m");
        assert!(state.facts.contains(&Relation::congruent(a, m, m, b)));
    }

    #[test]
    fn test_parse_eqdistance_congruent_fact() {
        // eqdistance x a b c → |XA| = |BC| (verify the congruent fact, not just point creation)
        let input = "test\na b c = triangle; x = eqdistance a b c ? cong x a b c";
        let state = parse_problem(input).unwrap();
        let x = state.id("x");
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        assert!(state.facts.contains(&Relation::congruent(x, a, b, c)));
    }

    #[test]
    fn test_parse_shift_parallel_and_congruent() {
        // shift x a b c → Parallel(a,b, c,x) AND Congruent(a,b, c,x)
        let input = "test\na b c = triangle; x = shift a b c ? cong a b c x";
        let state = parse_problem(input).unwrap();
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        let x = state.id("x");
        assert!(state.facts.contains(&Relation::parallel(a, b, c, x)));
        assert!(state.facts.contains(&Relation::congruent(a, b, c, x)));
    }

    #[test]
    fn test_parse_on_dia_perpendicular_fact() {
        // on_dia x a b → Perpendicular(a,x, b,x) (angle AXB = 90)
        let input = "test\na b = segment; x = on_dia a b ? perp a x b x";
        let state = parse_problem(input).unwrap();
        let a = state.id("a");
        let b = state.id("b");
        let x = state.id("x");
        assert!(state.facts.contains(&Relation::perpendicular(a, x, b, x)));
    }

    // --- Tests for newly fixed parser stubs ---

    #[test]
    fn test_parse_eqangle2_facts() {
        // eqangle2 x d a b → angle(x,d,a) = angle(a,d,b)
        let input = "test\na b c = triangle; e = eqangle2 c a b ? coll e a b";
        let state = parse_problem(input).unwrap();
        let e = state.id("e");
        let c = state.id("c");
        let a = state.id("a");
        let b = state.id("b");
        assert!(state.facts.contains(&Relation::equal_angle(e, c, a, a, c, b)));
    }

    #[test]
    fn test_parse_lc_tangent_facts() {
        // lc_tangent x p o → on_circle(x, o), perp(o, x, p, x)
        let input = "test\na b = segment; x = lc_tangent a b ? perp b x a x";
        let state = parse_problem(input).unwrap();
        let x = state.id("x");
        let a = state.id("a");
        let b = state.id("b");
        assert!(state.facts.contains(&Relation::on_circle(x, b)));
        assert!(state.facts.contains(&Relation::perpendicular(b, x, a, x)));
    }

    #[test]
    fn test_parse_intersection_cc_facts() {
        // intersection_cc x c1 c2 rp → on_circle(x,c1), on_circle(x,c2), cong(c2,x,c2,rp)
        let input = "test\na b c = triangle; x = intersection_cc a b c ? coll x a b";
        let state = parse_problem(input).unwrap();
        let x = state.id("x");
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        assert!(state.facts.contains(&Relation::on_circle(x, a)));
        assert!(state.facts.contains(&Relation::on_circle(x, b)));
        assert!(state.facts.contains(&Relation::congruent(b, x, b, c)));
    }

    #[test]
    fn test_parse_intersection_lc_facts() {
        // intersection_lc x lp center rp → on_circle(x, center), cong(center,x,center,rp)
        let input = "test\na b c = triangle; x = intersection_lc a b c ? coll x a b";
        let state = parse_problem(input).unwrap();
        let x = state.id("x");
        let b = state.id("b");
        let c = state.id("c");
        assert!(state.facts.contains(&Relation::on_circle(x, b)));
        assert!(state.facts.contains(&Relation::congruent(b, x, b, c)));
    }

    #[test]
    fn test_parse_shift_full_parallelogram() {
        // shift x a b c → both parallel pairs and both congruent pairs
        let input = "test\na b c = triangle; x = shift a b c ? para a c b x";
        let state = parse_problem(input).unwrap();
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        let x = state.id("x");
        assert!(state.facts.contains(&Relation::parallel(a, b, c, x)));
        assert!(state.facts.contains(&Relation::parallel(a, c, b, x)));
        assert!(state.facts.contains(&Relation::congruent(a, b, c, x)));
        assert!(state.facts.contains(&Relation::congruent(a, c, b, x)));
    }

    #[test]
    fn test_parse_risos() {
        // risos c a b → right angle at c, |ca|=|cb|
        let input = "test\nc a b = risos c a b ? perp c a c b";
        let state = parse_problem(input).unwrap();
        let c = state.id("c");
        let a = state.id("a");
        let b = state.id("b");
        assert!(state.facts.contains(&Relation::perpendicular(c, a, c, b)));
        assert!(state.facts.contains(&Relation::congruent(c, a, c, b)));
    }

    #[test]
    fn test_parse_angle_mirror() {
        // angle_mirror x a b c → angle(c,a,b) = angle(b,a,x)
        let input = "test\na b c = triangle; x = angle_mirror a b c ? coll x a b";
        let state = parse_problem(input).unwrap();
        let x = state.id("x");
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        assert!(state.facts.contains(&Relation::equal_angle(c, a, b, b, a, x)));
    }

    #[test]
    fn test_parse_pentagon() {
        let input = "test\na b c d e = pentagon a b c d e ? coll a b c";
        let state = parse_problem(input).unwrap();
        assert_eq!(state.objects.len(), 5);
        assert!(state.facts.is_empty());
    }

    #[test]
    fn test_parse_trisegment() {
        // trisegment d x c b → c-d-x-b collinear, |cd|=|dx|=|xb|
        let input = "test\nc b = segment; d x = trisegment c b ? cong c d d x";
        let state = parse_problem(input).unwrap();
        let c = state.id("c");
        let b = state.id("b");
        let d = state.id("d");
        let x = state.id("x");
        assert!(state.facts.contains(&Relation::collinear(c, d, b)));
        assert!(state.facts.contains(&Relation::collinear(c, x, b)));
        assert!(state.facts.contains(&Relation::congruent(c, d, d, x)));
        assert!(state.facts.contains(&Relation::congruent(d, x, x, b)));
    }

    #[test]
    fn test_parse_intersection_lt() {
        // intersection_lt x p1 p2 through la lb → collinear(x,p1,p2) + perp(through,x,la,lb)
        let input = "test\na b c d e = triangle; x = intersection_lt a b c d e ? coll x a b";
        let state = parse_problem(input).unwrap();
        let x = state.id("x");
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        let d = state.id("d");
        let e = state.id("e");
        assert!(state.facts.contains(&Relation::collinear(x, a, b)));
        assert!(state.facts.contains(&Relation::perpendicular(c, x, d, e)));
    }

    #[test]
    fn test_parse_intersection_tt() {
        // intersection_tt x t1 la1 lb1 t2 la2 lb2 → two perp facts
        let input = "test\na b c d e f = triangle; x = intersection_tt a b c d e f ? coll x a b";
        let state = parse_problem(input).unwrap();
        let x = state.id("x");
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        let d = state.id("d");
        let e = state.id("e");
        let f = state.id("f");
        assert!(state.facts.contains(&Relation::perpendicular(a, x, b, c)));
        assert!(state.facts.contains(&Relation::perpendicular(d, x, e, f)));
    }

    #[test]
    fn test_parse_intersection_lp() {
        // intersection_lp x p1 p2 through pa pb → collinear(x,p1,p2) + parallel(through,x,pa,pb)
        let input = "test\na b c d e = triangle; x = intersection_lp a b c d e ? coll x a b";
        let state = parse_problem(input).unwrap();
        let x = state.id("x");
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        let d = state.id("d");
        let e = state.id("e");
        assert!(state.facts.contains(&Relation::collinear(x, a, b)));
        assert!(state.facts.contains(&Relation::parallel(c, x, d, e)));
    }
}
