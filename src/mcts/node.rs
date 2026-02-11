use std::cell::RefCell;
use std::rc::{Rc, Weak};

use crate::construction::{apply_construction, generate_constructions, Construction};
use crate::deduction::saturate;
use crate::proof_state::ProofState;

pub type NodeRef = Rc<RefCell<MctsNode>>;
pub type WeakNodeRef = Weak<RefCell<MctsNode>>;

pub struct MctsNode {
    pub state: ProofState,
    pub action: Option<Construction>,
    pub visits: u32,
    pub total_value: f64,
    pub terminal_value: Option<f64>,
    pub children: Vec<NodeRef>,
    pub parent: Option<WeakNodeRef>,
    /// Whether this node has been expanded (children generated)
    pub expanded: bool,
}

impl MctsNode {
    /// Create a root node from a proof state.
    pub fn new_root(state: ProofState) -> NodeRef {
        Rc::new(RefCell::new(MctsNode {
            state,
            action: None,
            visits: 0,
            total_value: 0.0,
            terminal_value: None,
            children: Vec::new(),
            parent: None,
            expanded: false,
        }))
    }

    /// Create a child node from applying a construction to a parent's state.
    pub fn new_child(
        parent: &NodeRef,
        action: Construction,
        state: ProofState,
    ) -> NodeRef {
        let child = Rc::new(RefCell::new(MctsNode {
            state,
            action: Some(action),
            visits: 0,
            total_value: 0.0,
            terminal_value: None,
            children: Vec::new(),
            parent: Some(Rc::downgrade(parent)),
            expanded: false,
        }));
        parent.borrow_mut().children.push(Rc::clone(&child));
        child
    }

    /// Expand this node: generate constructions, create child nodes.
    /// Returns the number of children created.
    pub fn expand(node: &NodeRef, max_children: usize) -> usize {
        let constructions = {
            let n = node.borrow();
            if n.expanded || n.terminal_value.is_some() {
                return 0;
            }
            let mut cs = generate_constructions(&n.state);
            // Limit branching factor
            cs.truncate(max_children);
            cs
        };

        let count = constructions.len();
        for construction in constructions {
            let child_state = {
                let n = node.borrow();
                apply_construction(&n.state, &construction)
            };
            MctsNode::new_child(node, construction, child_state);
        }
        node.borrow_mut().expanded = true;
        count
    }

    /// Evaluate a leaf node: run saturate() and compute value.
    /// Returns value in [0.0, 1.0].
    pub fn evaluate(node: &NodeRef) -> f64 {
        let mut n = node.borrow_mut();

        // Already evaluated as terminal
        if let Some(v) = n.terminal_value {
            return v;
        }

        // Run deduction to fixed point
        if saturate(&mut n.state) {
            n.terminal_value = Some(1.0);
            return 1.0;
        }

        // Classical fallback: value = tanh(0.5 * delta_D)
        let delta_d = compute_delta_d(&n.state);
        (0.5 * delta_d).tanh()
    }

    /// Backpropagate a value up to the root.
    pub fn backprop(node: &NodeRef, value: f64) {
        let mut current = Some(Rc::clone(node));
        while let Some(n) = current {
            {
                let mut borrowed = n.borrow_mut();
                borrowed.visits += 1;
                borrowed.total_value += value;
            }
            let parent = n.borrow().parent.as_ref().and_then(|w| w.upgrade());
            current = parent;
        }
    }

    /// UCB/PUCT score for child selection.
    pub fn ucb_score(child: &NodeRef, parent_visits: u32, c_puct: f64, num_siblings: usize) -> f64 {
        let c = child.borrow();

        // Unvisited children get infinite score (explore first)
        if c.visits == 0 {
            return f64::INFINITY;
        }

        let q = c.total_value / c.visits as f64;
        // Uniform prior (no NN in Phase 2)
        let prior = 1.0 / num_siblings as f64;
        let u = c_puct * prior * (parent_visits as f64).sqrt() / (1.0 + c.visits as f64);
        q + u
    }

    /// Check if this node represents a proved state.
    pub fn is_terminal(&self) -> bool {
        self.terminal_value == Some(1.0)
    }
}

/// Compute delta_D: fraction of goal sub-conditions present in the state.
/// Simple heuristic for proof distance.
fn compute_delta_d(state: &ProofState) -> f64 {
    let goal = match &state.goal {
        Some(g) => g,
        None => return 0.0,
    };

    // Check if the goal is directly proved
    if state.facts.contains(goal) {
        return 1.0;
    }

    // Heuristic: check how many "related" facts exist
    use crate::proof_state::Relation;
    match goal {
        Relation::Congruent(a, b, c, d) => {
            // Check if we have any congruence involving these points
            let mut score: f64 = 0.0;
            let total: f64 = 2.0; // two segments to match
            for fact in &state.facts {
                if let Relation::Congruent(p, q, r, s) = fact {
                    // One segment matches
                    if segments_match(*p, *q, *a, *b) || segments_match(*r, *s, *a, *b) {
                        score += 0.5;
                    }
                    if segments_match(*p, *q, *c, *d) || segments_match(*r, *s, *c, *d) {
                        score += 0.5;
                    }
                }
            }
            (score / total).min(0.9) // cap below 1.0 since not actually proved
        }
        Relation::Parallel(a, b, c, d) => {
            let mut score: f64 = 0.0;
            for fact in &state.facts {
                if let Relation::Parallel(p, q, r, s) = fact {
                    if lines_match(*p, *q, *a, *b) || lines_match(*r, *s, *a, *b) {
                        score += 0.25;
                    }
                    if lines_match(*p, *q, *c, *d) || lines_match(*r, *s, *c, *d) {
                        score += 0.25;
                    }
                }
            }
            score.min(0.9)
        }
        Relation::Perpendicular(a, b, c, d) => {
            let mut score: f64 = 0.0;
            for fact in &state.facts {
                if let Relation::Perpendicular(p, q, r, s) = fact {
                    if lines_match(*p, *q, *a, *b) || lines_match(*r, *s, *a, *b) {
                        score += 0.25;
                    }
                    if lines_match(*p, *q, *c, *d) || lines_match(*r, *s, *c, *d) {
                        score += 0.25;
                    }
                }
            }
            score.min(0.9)
        }
        Relation::EqualAngle(a, b, c, d, e, f) => {
            let mut score: f64 = 0.0;
            for fact in &state.facts {
                if let Relation::EqualAngle(p, q, r, s, t, u) = fact {
                    if *p == *a && *q == *b && *r == *c {
                        score += 0.25;
                    }
                    if *s == *d && *t == *e && *u == *f {
                        score += 0.25;
                    }
                    if *p == *d && *q == *e && *r == *f {
                        score += 0.25;
                    }
                    if *s == *a && *t == *b && *u == *c {
                        score += 0.25;
                    }
                }
            }
            score.min(0.9)
        }
        Relation::Collinear(a, b, c) => {
            let mut score: f64 = 0.0;
            for fact in &state.facts {
                if let Relation::Collinear(p, q, r) = fact {
                    let goal_pts = [a, b, c];
                    let fact_pts = [p, q, r];
                    let overlap = goal_pts.iter().filter(|&&g| fact_pts.iter().any(|&&f| f == *g)).count();
                    if overlap >= 2 {
                        score += 0.3;
                    }
                }
            }
            score.min(0.9)
        }
        _ => 0.0,
    }
}

fn segments_match(a1: u16, b1: u16, a2: u16, b2: u16) -> bool {
    (a1 == a2 && b1 == b2) || (a1 == b2 && b1 == a2)
}

fn lines_match(a1: u16, b1: u16, a2: u16, b2: u16) -> bool {
    (a1 == a2 && b1 == b2) || (a1 == b2 && b1 == a2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::construction::{Construction, ConstructionType, Priority};
    use crate::proof_state::{ObjectType, ProofState, Relation};

    fn make_triangle_state() -> ProofState {
        let mut state = ProofState::new();
        state.add_object("a", ObjectType::Point);
        state.add_object("b", ObjectType::Point);
        state.add_object("c", ObjectType::Point);
        state
    }

    #[test]
    fn test_new_root() {
        let state = make_triangle_state();
        let root = MctsNode::new_root(state);
        let n = root.borrow();
        assert_eq!(n.visits, 0);
        assert_eq!(n.total_value, 0.0);
        assert!(n.action.is_none());
        assert!(n.parent.is_none());
        assert!(n.children.is_empty());
        assert!(!n.expanded);
    }

    #[test]
    fn test_expand_creates_children() {
        let state = make_triangle_state();
        let root = MctsNode::new_root(state);
        let count = MctsNode::expand(&root, 50);
        assert!(count > 0);
        assert_eq!(root.borrow().children.len(), count);
        assert!(root.borrow().expanded);
    }

    #[test]
    fn test_expand_respects_max_children() {
        let state = make_triangle_state();
        let root = MctsNode::new_root(state);
        let count = MctsNode::expand(&root, 3);
        assert!(count <= 3);
        assert_eq!(root.borrow().children.len(), count);
    }

    #[test]
    fn test_expand_idempotent() {
        let state = make_triangle_state();
        let root = MctsNode::new_root(state);
        let count1 = MctsNode::expand(&root, 50);
        let count2 = MctsNode::expand(&root, 50);
        assert!(count1 > 0);
        assert_eq!(count2, 0); // second expand does nothing
    }

    #[test]
    fn test_backprop_updates_ancestors() {
        let state = make_triangle_state();
        let root = MctsNode::new_root(state);
        MctsNode::expand(&root, 50);

        let child = Rc::clone(&root.borrow().children[0]);
        MctsNode::backprop(&child, 0.5);

        assert_eq!(child.borrow().visits, 1);
        assert_eq!(child.borrow().total_value, 0.5);
        assert_eq!(root.borrow().visits, 1);
        assert_eq!(root.borrow().total_value, 0.5);
    }

    #[test]
    fn test_backprop_multiple() {
        let state = make_triangle_state();
        let root = MctsNode::new_root(state);
        MctsNode::expand(&root, 50);

        let child0 = Rc::clone(&root.borrow().children[0]);
        let child1 = Rc::clone(&root.borrow().children[1]);
        MctsNode::backprop(&child0, 0.5);
        MctsNode::backprop(&child1, 0.8);

        assert_eq!(root.borrow().visits, 2);
        assert!((root.borrow().total_value - 1.3).abs() < 1e-10);
    }

    #[test]
    fn test_ucb_unvisited_is_infinity() {
        let state = make_triangle_state();
        let root = MctsNode::new_root(state);
        MctsNode::expand(&root, 10);

        let child = Rc::clone(&root.borrow().children[0]);
        let score = MctsNode::ucb_score(&child, 1, 1.4, 10);
        assert!(score.is_infinite());
    }

    #[test]
    fn test_ucb_visited() {
        let state = make_triangle_state();
        let root = MctsNode::new_root(state);
        MctsNode::expand(&root, 10);

        let child = Rc::clone(&root.borrow().children[0]);
        MctsNode::backprop(&child, 0.5);
        let num_children = root.borrow().children.len();
        let score = MctsNode::ucb_score(&child, 1, 1.4, num_children);
        assert!(score.is_finite());
        assert!(score > 0.0);
    }

    #[test]
    fn test_evaluate_proved_state() {
        let mut state = make_triangle_state();
        let a = state.id("a");
        let b = state.id("b");
        let m = state.add_object("m", ObjectType::Point);
        state.add_fact(Relation::midpoint(m, a, b));
        state.set_goal(Relation::congruent(a, m, m, b));

        let root = MctsNode::new_root(state);
        let value = MctsNode::evaluate(&root);
        assert_eq!(value, 1.0);
        assert_eq!(root.borrow().terminal_value, Some(1.0));
    }

    #[test]
    fn test_evaluate_unproved_state() {
        let mut state = make_triangle_state();
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        state.set_goal(Relation::collinear(a, b, c));

        let root = MctsNode::new_root(state);
        let value = MctsNode::evaluate(&root);
        assert!(value >= 0.0 && value < 1.0);
    }

    #[test]
    fn test_delta_d_congruent_partial() {
        let mut state = make_triangle_state();
        let a = state.id("a");
        let b = state.id("b");
        let c = state.id("c");
        // Goal: |AB| = |AC|, we have |AB| = |XY| (some related congruence)
        state.add_fact(Relation::congruent(a, b, 10, 11));
        state.set_goal(Relation::congruent(a, b, a, c));
        let d = compute_delta_d(&state);
        assert!(d > 0.0, "Should have partial progress");
        assert!(d < 1.0);
    }

    #[test]
    fn test_delta_d_no_goal() {
        let state = make_triangle_state();
        let d = compute_delta_d(&state);
        assert_eq!(d, 0.0);
    }

    #[test]
    fn test_is_terminal() {
        let node = MctsNode {
            state: ProofState::new(),
            action: None,
            visits: 0,
            total_value: 0.0,
            terminal_value: Some(1.0),
            children: Vec::new(),
            parent: None,
            expanded: false,
        };
        assert!(node.is_terminal());
    }

    #[test]
    fn test_child_has_parent_ref() {
        let state = make_triangle_state();
        let root = MctsNode::new_root(state);
        MctsNode::expand(&root, 5);

        let child = Rc::clone(&root.borrow().children[0]);
        let parent = child.borrow().parent.as_ref().unwrap().upgrade().unwrap();
        assert!(Rc::ptr_eq(&parent, &root));
    }

    #[test]
    fn test_expand_terminal_node_noop() {
        let state = make_triangle_state();
        let root = MctsNode::new_root(state);
        root.borrow_mut().terminal_value = Some(1.0);
        let count = MctsNode::expand(&root, 50);
        assert_eq!(count, 0);
        assert!(root.borrow().children.is_empty());
    }

    #[test]
    fn test_evaluate_returns_cached_terminal() {
        let state = make_triangle_state();
        let root = MctsNode::new_root(state);
        root.borrow_mut().terminal_value = Some(0.5);
        let value = MctsNode::evaluate(&root);
        assert_eq!(value, 0.5);
    }

    #[test]
    fn test_is_terminal_false_for_none() {
        let node = MctsNode {
            state: ProofState::new(),
            action: None,
            visits: 0,
            total_value: 0.0,
            terminal_value: None,
            children: Vec::new(),
            parent: None,
            expanded: false,
        };
        assert!(!node.is_terminal());
    }

    #[test]
    fn test_is_terminal_false_for_zero() {
        let node = MctsNode {
            state: ProofState::new(),
            action: None,
            visits: 0,
            total_value: 0.0,
            terminal_value: Some(0.0),
            children: Vec::new(),
            parent: None,
            expanded: false,
        };
        assert!(!node.is_terminal());
    }

    #[test]
    fn test_delta_d_parallel() {
        let mut state = ProofState::new();
        state.add_object("a", ObjectType::Point);
        state.add_object("b", ObjectType::Point);
        state.add_object("c", ObjectType::Point);
        state.add_object("d", ObjectType::Point);
        let (a, b, c, d) = (0, 1, 2, 3);
        // Goal: AB || CD, we have some parallel fact involving AB
        state.add_fact(Relation::parallel(a, b, 10, 11));
        state.set_goal(Relation::parallel(a, b, c, d));
        let delta = compute_delta_d(&state);
        assert!(delta > 0.0);
        assert!(delta <= 0.9);
    }

    #[test]
    fn test_delta_d_perpendicular() {
        let mut state = ProofState::new();
        state.add_object("a", ObjectType::Point);
        state.add_object("b", ObjectType::Point);
        state.add_object("c", ObjectType::Point);
        state.add_object("d", ObjectType::Point);
        let (a, b, c, d) = (0, 1, 2, 3);
        state.add_fact(Relation::perpendicular(a, b, 10, 11));
        state.set_goal(Relation::perpendicular(a, b, c, d));
        let delta = compute_delta_d(&state);
        assert!(delta > 0.0);
        assert!(delta <= 0.9);
    }

    #[test]
    fn test_delta_d_equal_angle() {
        let mut state = ProofState::new();
        for i in 0..6 {
            state.add_object(&format!("p{}", i), ObjectType::Point);
        }
        // Goal: angle(0,1,2) = angle(3,4,5)
        // Have a fact matching first triple
        state.add_fact(Relation::equal_angle(0, 1, 2, 10, 11, 12));
        state.set_goal(Relation::equal_angle(0, 1, 2, 3, 4, 5));
        let delta = compute_delta_d(&state);
        assert!(delta > 0.0);
        assert!(delta <= 0.9);
    }

    #[test]
    fn test_delta_d_collinear() {
        let mut state = ProofState::new();
        state.add_object("a", ObjectType::Point);
        state.add_object("b", ObjectType::Point);
        state.add_object("c", ObjectType::Point);
        let (a, b, c) = (0, 1, 2);
        // Have collinear(a, b, x) — shares 2 points with goal collinear(a, b, c)
        state.add_object("x", ObjectType::Point);
        state.add_fact(Relation::collinear(a, b, 3));
        state.set_goal(Relation::collinear(a, b, c));
        let delta = compute_delta_d(&state);
        assert!(delta > 0.0);
        assert!(delta <= 0.9);
    }

    #[test]
    fn test_delta_d_cyclic_returns_zero() {
        let mut state = ProofState::new();
        for i in 0..4 {
            state.add_object(&format!("p{}", i), ObjectType::Point);
        }
        state.set_goal(Relation::cyclic(0, 1, 2, 3));
        let delta = compute_delta_d(&state);
        assert_eq!(delta, 0.0); // Cyclic hits the `_ => 0.0` branch
    }

    #[test]
    fn test_delta_d_proved_returns_one() {
        let mut state = ProofState::new();
        state.add_object("a", ObjectType::Point);
        state.add_object("b", ObjectType::Point);
        state.add_object("c", ObjectType::Point);
        let goal = Relation::collinear(0, 1, 2);
        state.add_fact(goal.clone());
        state.set_goal(goal);
        let delta = compute_delta_d(&state);
        assert_eq!(delta, 1.0);
    }

    #[test]
    fn test_backprop_three_levels() {
        let state = ProofState::new();
        let root = MctsNode::new_root(state);
        let child = MctsNode::new_child(
            &root,
            Construction { ctype: ConstructionType::Midpoint, args: vec![0, 1], priority: Priority::Exploratory },
            ProofState::new(),
        );
        let grandchild = MctsNode::new_child(
            &child,
            Construction { ctype: ConstructionType::Altitude, args: vec![0, 1, 2], priority: Priority::Exploratory },
            ProofState::new(),
        );
        MctsNode::backprop(&grandchild, 0.7);

        assert_eq!(grandchild.borrow().visits, 1);
        assert!((grandchild.borrow().total_value - 0.7).abs() < 1e-10);
        assert_eq!(child.borrow().visits, 1);
        assert!((child.borrow().total_value - 0.7).abs() < 1e-10);
        assert_eq!(root.borrow().visits, 1);
        assert!((root.borrow().total_value - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_ucb_high_q_wins() {
        let state = make_triangle_state();
        let root = MctsNode::new_root(state);
        MctsNode::expand(&root, 10);

        let child0 = Rc::clone(&root.borrow().children[0]);
        let child1 = Rc::clone(&root.borrow().children[1]);
        // Give child0 high value, child1 low value
        MctsNode::backprop(&child0, 0.9);
        MctsNode::backprop(&child1, 0.1);
        let n = root.borrow().children.len();
        let score0 = MctsNode::ucb_score(&child0, 2, 1.4, n);
        let score1 = MctsNode::ucb_score(&child1, 2, 1.4, n);
        assert!(score0 > score1, "Higher Q child should have higher UCB score");
    }
}
