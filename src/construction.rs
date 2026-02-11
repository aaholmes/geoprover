use crate::proof_state::{ObjectType, ProofState, Relation};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ConstructionType {
    Midpoint,
    AngleBisector,
    PerpendicularBisector,
    Altitude,
    ParallelThrough,
    PerpendicularThrough,
    Circumcenter,
    Incenter,
    Centroid,
    Orthocenter,
    CircumscribedCircle,
    IntersectLines,
    IntersectLineCircle,
    ReflectPoint,
    ExtendSegment,
    TangentLine,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Priority {
    GoalRelevant,
    RecentlyActive,
    Exploratory,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Construction {
    pub ctype: ConstructionType,
    pub args: Vec<u16>,
    pub priority: Priority,
}

/// Generate all candidate auxiliary constructions for the current state.
pub fn generate_constructions(state: &ProofState) -> Vec<Construction> {
    let mut constructions = Vec::new();
    let points = get_points(state);

    // Midpoint constructions — for each pair of points
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let a = points[i];
            let b = points[j];
            // Don't generate midpoint if it already exists
            if !has_midpoint(state, a, b) {
                let priority = classify_priority(state, &[a, b]);
                constructions.push(Construction {
                    ctype: ConstructionType::Midpoint,
                    args: vec![a, b],
                    priority,
                });
            }
        }
    }

    // Altitude — for each point to each line defined by two other points
    for i in 0..points.len() {
        for j in 0..points.len() {
            for k in (j + 1)..points.len() {
                if i != j && i != k {
                    let a = points[i];
                    let b = points[j];
                    let c = points[k];
                    if !has_foot(state, a, b, c) {
                        let priority = classify_priority(state, &[a, b, c]);
                        constructions.push(Construction {
                            ctype: ConstructionType::Altitude,
                            args: vec![a, b, c],
                            priority,
                        });
                    }
                }
            }
        }
    }

    // Circumcenter — for each triple of points
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            for k in (j + 1)..points.len() {
                let a = points[i];
                let b = points[j];
                let c = points[k];
                if !has_circumcenter(state, a, b, c) {
                    let priority = classify_priority(state, &[a, b, c]);
                    constructions.push(Construction {
                        ctype: ConstructionType::Circumcenter,
                        args: vec![a, b, c],
                        priority,
                    });
                }
            }
        }
    }

    // Orthocenter — for each triple
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            for k in (j + 1)..points.len() {
                let a = points[i];
                let b = points[j];
                let c = points[k];
                let priority = classify_priority(state, &[a, b, c]);
                constructions.push(Construction {
                    ctype: ConstructionType::Orthocenter,
                    args: vec![a, b, c],
                    priority,
                });
            }
        }
    }

    // Incenter — for each triple
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            for k in (j + 1)..points.len() {
                let a = points[i];
                let b = points[j];
                let c = points[k];
                let priority = classify_priority(state, &[a, b, c]);
                constructions.push(Construction {
                    ctype: ConstructionType::Incenter,
                    args: vec![a, b, c],
                    priority,
                });
            }
        }
    }

    // ParallelThrough — for each point and each line
    for &p in &points {
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                let a = points[i];
                let b = points[j];
                if p != a && p != b {
                    let priority = classify_priority(state, &[p, a, b]);
                    constructions.push(Construction {
                        ctype: ConstructionType::ParallelThrough,
                        args: vec![p, a, b],
                        priority,
                    });
                }
            }
        }
    }

    // PerpendicularThrough — for each point and each line
    for &p in &points {
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                let a = points[i];
                let b = points[j];
                if p != a && p != b {
                    let priority = classify_priority(state, &[p, a, b]);
                    constructions.push(Construction {
                        ctype: ConstructionType::PerpendicularThrough,
                        args: vec![p, a, b],
                        priority,
                    });
                }
            }
        }
    }

    // Sort by priority: GoalRelevant first, then RecentlyActive, then Exploratory
    constructions.sort_by_key(|c| match c.priority {
        Priority::GoalRelevant => 0,
        Priority::RecentlyActive => 1,
        Priority::Exploratory => 2,
    });

    constructions
}

/// Apply a construction to a state, creating a new state with the construction applied.
pub fn apply_construction(state: &ProofState, construction: &Construction) -> ProofState {
    let mut new_state = state.clone();
    let new_id = new_state.objects.len() as u16;
    let name = format!("aux_{}", new_id);

    match construction.ctype {
        ConstructionType::Midpoint => {
            let (a, b) = (construction.args[0], construction.args[1]);
            let m = new_state.add_object(&name, ObjectType::Point);
            new_state.add_fact(Relation::midpoint(m, a, b));
            new_state.add_fact(Relation::collinear(a, m, b));
            new_state.add_fact(Relation::congruent(a, m, m, b));
        }
        ConstructionType::Altitude => {
            let (a, b, c) = (construction.args[0], construction.args[1], construction.args[2]);
            let f = new_state.add_object(&name, ObjectType::Point);
            new_state.add_fact(Relation::perpendicular(a, f, b, c));
            new_state.add_fact(Relation::collinear(b, f, c));
        }
        ConstructionType::Circumcenter => {
            let (a, b, c) = (construction.args[0], construction.args[1], construction.args[2]);
            let o = new_state.add_object(&name, ObjectType::Point);
            new_state.add_fact(Relation::congruent(o, a, o, b));
            new_state.add_fact(Relation::congruent(o, b, o, c));
        }
        ConstructionType::Orthocenter => {
            let (a, b, c) = (construction.args[0], construction.args[1], construction.args[2]);
            let h = new_state.add_object(&name, ObjectType::Point);
            new_state.add_fact(Relation::perpendicular(a, h, b, c));
            new_state.add_fact(Relation::perpendicular(b, h, a, c));
            new_state.add_fact(Relation::perpendicular(c, h, a, b));
        }
        ConstructionType::Incenter => {
            let (a, b, c) = (construction.args[0], construction.args[1], construction.args[2]);
            let i = new_state.add_object(&name, ObjectType::Point);
            new_state.add_fact(Relation::equal_angle(b, a, i, i, a, c));
            new_state.add_fact(Relation::equal_angle(a, b, i, i, b, c));
            new_state.add_fact(Relation::equal_angle(a, c, i, i, c, b));
        }
        ConstructionType::ParallelThrough => {
            let (p, a, b) = (construction.args[0], construction.args[1], construction.args[2]);
            let x = new_state.add_object(&name, ObjectType::Point);
            new_state.add_fact(Relation::parallel(p, x, a, b));
        }
        ConstructionType::PerpendicularThrough => {
            let (p, a, b) = (construction.args[0], construction.args[1], construction.args[2]);
            let x = new_state.add_object(&name, ObjectType::Point);
            new_state.add_fact(Relation::perpendicular(p, x, a, b));
        }
        _ => {
            // Other construction types — add a generic point for now
            new_state.add_object(&name, ObjectType::Point);
        }
    }

    new_state
}

fn get_points(state: &ProofState) -> Vec<u16> {
    state
        .objects
        .iter()
        .filter(|o| o.otype == ObjectType::Point)
        .map(|o| o.id)
        .collect()
}

fn has_midpoint(state: &ProofState, a: u16, b: u16) -> bool {
    state.facts.iter().any(|f| matches!(f, Relation::Midpoint(_, x, y) if
        (*x == a && *y == b) || (*x == b && *y == a)))
}

fn has_foot(state: &ProofState, a: u16, b: u16, c: u16) -> bool {
    // Check if there's already a foot of a onto line bc
    state.facts.iter().any(|f| {
        if let Relation::Perpendicular(p, q, r, s) = f {
            // foot pattern: perp(a, foot, b, c)
            (*p == a || *q == a)
                && ((*r == b && *s == c) || (*r == c && *s == b) || (*p == b && *q == c) || (*p == c && *q == b))
        } else {
            false
        }
    })
}

fn has_circumcenter(state: &ProofState, a: u16, b: u16, _c: u16) -> bool {
    // Check if there's a point equidistant from a, b, and c
    state.facts.iter().any(|f| {
        if let Relation::Congruent(p, q, r, s) = f {
            // Pattern: |oa| = |ob|
            (*q == a && *s == b && p == r)
                || (*q == b && *s == a && p == r)
                || (*p == a && *r == b && q == s)
        } else {
            false
        }
    })
}

/// Classify a construction's priority based on goal relevance
fn classify_priority(state: &ProofState, involved_points: &[u16]) -> Priority {
    if let Some(goal) = &state.goal {
        let goal_points = get_relation_points(goal);
        // If any of the construction's involved points appear in the goal, it's GoalRelevant
        if involved_points.iter().any(|p| goal_points.contains(p)) {
            return Priority::GoalRelevant;
        }
    }
    Priority::Exploratory
}

fn get_relation_points(relation: &Relation) -> Vec<u16> {
    match relation {
        Relation::Collinear(a, b, c) => vec![*a, *b, *c],
        Relation::Parallel(a, b, c, d) => vec![*a, *b, *c, *d],
        Relation::Perpendicular(a, b, c, d) => vec![*a, *b, *c, *d],
        Relation::Congruent(a, b, c, d) => vec![*a, *b, *c, *d],
        Relation::EqualAngle(a, b, c, d, e, f) => vec![*a, *b, *c, *d, *e, *f],
        Relation::Midpoint(m, a, b) => vec![*m, *a, *b],
        Relation::OnCircle(a, b) => vec![*a, *b],
        Relation::Cyclic(a, b, c, d) => vec![*a, *b, *c, *d],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_triangle() -> ProofState {
        let mut state = ProofState::new();
        state.add_object("a", ObjectType::Point);
        state.add_object("b", ObjectType::Point);
        state.add_object("c", ObjectType::Point);
        state
    }

    #[test]
    fn test_generates_midpoint_constructions() {
        let state = make_triangle();
        let constructions = generate_constructions(&state);
        let midpoints: Vec<_> = constructions
            .iter()
            .filter(|c| c.ctype == ConstructionType::Midpoint)
            .collect();
        // 3 pairs: (a,b), (a,c), (b,c)
        assert_eq!(midpoints.len(), 3);
    }

    #[test]
    fn test_generates_circumcenter() {
        let state = make_triangle();
        let constructions = generate_constructions(&state);
        let circumcenters: Vec<_> = constructions
            .iter()
            .filter(|c| c.ctype == ConstructionType::Circumcenter)
            .collect();
        assert_eq!(circumcenters.len(), 1); // only one triple
    }

    #[test]
    fn test_generates_orthocenter_incenter() {
        let state = make_triangle();
        let constructions = generate_constructions(&state);
        let ortho: Vec<_> = constructions
            .iter()
            .filter(|c| c.ctype == ConstructionType::Orthocenter)
            .collect();
        let incenter: Vec<_> = constructions
            .iter()
            .filter(|c| c.ctype == ConstructionType::Incenter)
            .collect();
        assert_eq!(ortho.len(), 1);
        assert_eq!(incenter.len(), 1);
    }

    #[test]
    fn test_priority_goal_relevant() {
        let mut state = make_triangle();
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        state.set_goal(Relation::congruent(a, b, a, c));
        let constructions = generate_constructions(&state);
        // All constructions involving a, b, or c should be GoalRelevant
        assert!(constructions.iter().all(|c| c.priority == Priority::GoalRelevant));
    }

    #[test]
    fn test_no_duplicate_constructions() {
        let state = make_triangle();
        let constructions = generate_constructions(&state);
        let unique: std::collections::HashSet<_> = constructions.iter().collect();
        assert_eq!(unique.len(), constructions.len());
    }

    #[test]
    fn test_constructions_reference_existing_objects() {
        let state = make_triangle();
        let constructions = generate_constructions(&state);
        let max_id = state.objects.len() as u16;
        for c in &constructions {
            for &arg in &c.args {
                assert!(arg < max_id, "Construction references non-existent object {}", arg);
            }
        }
    }

    #[test]
    fn test_apply_midpoint_construction() {
        let state = make_triangle();
        let a = state.id("a");
        let b = state.id("b");
        let construction = Construction {
            ctype: ConstructionType::Midpoint,
            args: vec![a, b],
            priority: Priority::Exploratory,
        };
        let new_state = apply_construction(&state, &construction);
        assert_eq!(new_state.objects.len(), 4); // 3 + 1 new point
        // Should have midpoint and collinear facts
        let m = 3u16; // the new point
        assert!(new_state.facts.contains(&Relation::midpoint(m, a, b)));
        assert!(new_state.facts.contains(&Relation::collinear(a, m, b)));
    }

    #[test]
    fn test_apply_altitude_construction() {
        let state = make_triangle();
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        let construction = Construction {
            ctype: ConstructionType::Altitude,
            args: vec![a, b, c],
            priority: Priority::Exploratory,
        };
        let new_state = apply_construction(&state, &construction);
        let f = 3u16;
        assert!(new_state.facts.contains(&Relation::perpendicular(a, f, b, c)));
        assert!(new_state.facts.contains(&Relation::collinear(b, f, c)));
    }
}
