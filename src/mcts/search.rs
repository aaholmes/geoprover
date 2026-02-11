use std::rc::Rc;

use super::node::{MctsNode, NodeRef};

/// Configuration for MCTS search.
pub struct MctsConfig {
    /// Number of MCTS iterations (select-expand-evaluate-backprop cycles).
    pub num_iterations: u32,
    /// Maximum children per node (branching factor cap).
    pub max_children: usize,
    /// Exploration constant for UCB/PUCT.
    pub c_puct: f64,
    /// Maximum tree depth (construction steps).
    pub max_depth: u32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig {
            num_iterations: 200,
            max_children: 30,
            c_puct: 1.4,
            max_depth: 3,
        }
    }
}

/// Result of an MCTS search.
pub struct MctsResult {
    /// Whether the goal was proved.
    pub solved: bool,
    /// Best value found.
    pub best_value: f64,
    /// Sequence of constructions leading to the proof (if solved).
    pub proof_actions: Vec<crate::construction::Construction>,
    /// Total iterations run.
    pub iterations: u32,
}

/// Run MCTS search from a root state.
pub fn mcts_search(
    state: crate::proof_state::ProofState,
    config: &MctsConfig,
) -> MctsResult {
    let root = MctsNode::new_root(state);

    // First, check if the root state is already proved
    let root_value = MctsNode::evaluate(&root);
    if root.borrow().terminal_value == Some(1.0) {
        return MctsResult {
            solved: true,
            best_value: 1.0,
            proof_actions: Vec::new(),
            iterations: 0,
        };
    }
    MctsNode::backprop(&root, root_value);

    for iter in 0..config.num_iterations {
        // 1. Select a leaf node
        let leaf = select_leaf(&root, config);

        // Check if we've already found a proof
        if leaf.borrow().terminal_value == Some(1.0) {
            let actions = extract_proof_path(&leaf);
            return MctsResult {
                solved: true,
                best_value: 1.0,
                proof_actions: actions,
                iterations: iter + 1,
            };
        }

        // 2. Check depth — don't expand beyond max_depth
        let depth = node_depth(&leaf);
        if depth >= config.max_depth {
            let value = MctsNode::evaluate(&leaf);
            MctsNode::backprop(&leaf, value);
            continue;
        }

        // 3. Expand
        let num_children = MctsNode::expand(&leaf, config.max_children);
        if num_children == 0 {
            // No constructions available — evaluate leaf
            let value = MctsNode::evaluate(&leaf);
            MctsNode::backprop(&leaf, value);
            continue;
        }

        // 4. Evaluate first child (or a random unvisited one)
        let child = Rc::clone(&leaf.borrow().children[0]);
        let value = MctsNode::evaluate(&child);

        // Check if proof found
        if child.borrow().terminal_value == Some(1.0) {
            MctsNode::backprop(&child, 1.0);
            let actions = extract_proof_path(&child);
            return MctsResult {
                solved: true,
                best_value: 1.0,
                proof_actions: actions,
                iterations: iter + 1,
            };
        }

        MctsNode::backprop(&child, value);
    }

    // Search exhausted — return best result
    let best_value = if root.borrow().visits > 0 {
        root.borrow().total_value / root.borrow().visits as f64
    } else {
        0.0
    };

    MctsResult {
        solved: false,
        best_value,
        proof_actions: Vec::new(),
        iterations: config.num_iterations,
    }
}

/// Select a leaf node using UCB/PUCT, with two-phase selection:
/// 1. Visit unexplored children first (by priority order — they're already sorted)
/// 2. Then use PUCT among visited children
fn select_leaf(root: &NodeRef, config: &MctsConfig) -> NodeRef {
    let mut current = Rc::clone(root);

    loop {
        let is_leaf = {
            let n = current.borrow();
            !n.expanded || n.children.is_empty() || n.terminal_value.is_some()
        };

        if is_leaf {
            return current;
        }

        let next = {
            let n = current.borrow();
            let parent_visits = n.visits;
            let num_children = n.children.len();

            // Phase 1: find first unvisited child
            let unvisited = n.children.iter().find(|c| c.borrow().visits == 0);
            if let Some(child) = unvisited {
                Rc::clone(child)
            } else {
                // Phase 2: PUCT selection among all children
                let mut best_score = f64::NEG_INFINITY;
                let mut best_child = Rc::clone(&n.children[0]);
                for child in &n.children {
                    let score = MctsNode::ucb_score(child, parent_visits, config.c_puct, num_children);
                    if score > best_score {
                        best_score = score;
                        best_child = Rc::clone(child);
                    }
                }
                best_child
            }
        };

        current = next;
    }
}

/// Compute the depth of a node (0 for root).
fn node_depth(node: &NodeRef) -> u32 {
    let mut depth = 0u32;
    let mut current = Rc::clone(node);
    loop {
        let parent = current.borrow().parent.as_ref().and_then(|w| w.upgrade());
        match parent {
            Some(p) => {
                depth += 1;
                current = p;
            }
            None => return depth,
        }
    }
}

/// Extract the sequence of constructions from root to this node.
fn extract_proof_path(node: &NodeRef) -> Vec<crate::construction::Construction> {
    let mut actions = Vec::new();
    let mut current = Rc::clone(node);
    loop {
        let action = current.borrow().action.clone();
        let parent = current.borrow().parent.as_ref().and_then(|w| w.upgrade());
        if let Some(a) = action {
            actions.push(a);
        }
        match parent {
            Some(p) => current = p,
            None => break,
        }
    }
    actions.reverse();
    actions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof_state::{ObjectType, ProofState, Relation};

    #[test]
    fn test_mcts_already_proved() {
        let mut state = ProofState::new();
        let a = state.add_object("a", ObjectType::Point);
        let b = state.add_object("b", ObjectType::Point);
        let m = state.add_object("m", ObjectType::Point);
        state.add_fact(Relation::midpoint(m, a, b));
        state.set_goal(Relation::congruent(a, m, m, b));

        let config = MctsConfig::default();
        let result = mcts_search(state, &config);
        assert!(result.solved);
        assert_eq!(result.best_value, 1.0);
        assert!(result.proof_actions.is_empty()); // no constructions needed
    }

    #[test]
    fn test_mcts_unsolvable() {
        let mut state = ProofState::new();
        let a = state.add_object("a", ObjectType::Point);
        let b = state.add_object("b", ObjectType::Point);
        let c = state.add_object("c", ObjectType::Point);
        state.set_goal(Relation::collinear(a, b, c));

        let config = MctsConfig {
            num_iterations: 20,
            max_children: 5,
            c_puct: 1.4,
            max_depth: 2,
        };
        let result = mcts_search(state, &config);
        assert!(!result.solved);
    }

    #[test]
    fn test_mcts_one_step_midpoint() {
        // Triangle, goal: |AM| = |MB| where M = midpoint(A,B)
        // MCTS should find the midpoint construction
        let mut state = ProofState::new();
        let a = state.add_object("a", ObjectType::Point);
        let b = state.add_object("b", ObjectType::Point);
        state.add_object("c", ObjectType::Point);
        // The midpoint will be object ID 3
        state.set_goal(Relation::congruent(a, 3, 3, b));

        let config = MctsConfig {
            num_iterations: 200,
            max_children: 30,
            c_puct: 1.4,
            max_depth: 2,
        };
        let result = mcts_search(state, &config);
        assert!(result.solved, "MCTS should solve midpoint problem");
        assert!(!result.proof_actions.is_empty());
    }

    #[test]
    fn test_mcts_config_default() {
        let config = MctsConfig::default();
        assert_eq!(config.num_iterations, 200);
        assert_eq!(config.max_children, 30);
        assert!((config.c_puct - 1.4).abs() < 1e-10);
        assert_eq!(config.max_depth, 3);
    }

    #[test]
    fn test_node_depth() {
        let state = ProofState::new();
        let root = MctsNode::new_root(state);
        assert_eq!(node_depth(&root), 0);

        let child_state = ProofState::new();
        let child = MctsNode::new_child(
            &root,
            crate::construction::Construction {
                ctype: crate::construction::ConstructionType::Midpoint,
                args: vec![0, 1],
                priority: crate::construction::Priority::Exploratory,
            },
            child_state,
        );
        assert_eq!(node_depth(&child), 1);
    }

    #[test]
    fn test_extract_proof_path() {
        let state = ProofState::new();
        let root = MctsNode::new_root(state);

        let action1 = crate::construction::Construction {
            ctype: crate::construction::ConstructionType::Midpoint,
            args: vec![0, 1],
            priority: crate::construction::Priority::GoalRelevant,
        };
        let child = MctsNode::new_child(&root, action1.clone(), ProofState::new());

        let action2 = crate::construction::Construction {
            ctype: crate::construction::ConstructionType::Altitude,
            args: vec![0, 1, 2],
            priority: crate::construction::Priority::Exploratory,
        };
        let grandchild = MctsNode::new_child(&child, action2.clone(), ProofState::new());

        let path = extract_proof_path(&grandchild);
        assert_eq!(path.len(), 2);
        assert_eq!(path[0], action1);
        assert_eq!(path[1], action2);
    }

    #[test]
    fn test_select_leaf_returns_root_when_unexpanded() {
        let state = ProofState::new();
        let root = MctsNode::new_root(state);
        let config = MctsConfig::default();
        let leaf = select_leaf(&root, &config);
        assert!(Rc::ptr_eq(&leaf, &root));
    }

    #[test]
    fn test_select_leaf_returns_child_when_expanded() {
        let mut state = ProofState::new();
        state.add_object("a", ObjectType::Point);
        state.add_object("b", ObjectType::Point);
        state.add_object("c", ObjectType::Point);
        let root = MctsNode::new_root(state);
        MctsNode::expand(&root, 5);
        let config = MctsConfig::default();
        let leaf = select_leaf(&root, &config);
        // Should select one of the children (an unvisited one)
        assert!(!Rc::ptr_eq(&leaf, &root));
        assert_eq!(leaf.borrow().visits, 0);
    }

    #[test]
    fn test_select_leaf_terminal_node_stops() {
        let state = ProofState::new();
        let root = MctsNode::new_root(state);
        root.borrow_mut().terminal_value = Some(1.0);
        root.borrow_mut().expanded = true;
        let config = MctsConfig::default();
        let leaf = select_leaf(&root, &config);
        assert!(Rc::ptr_eq(&leaf, &root));
    }

    #[test]
    fn test_mcts_zero_iterations() {
        let mut state = ProofState::new();
        let a = state.add_object("a", ObjectType::Point);
        let b = state.add_object("b", ObjectType::Point);
        let c = state.add_object("c", ObjectType::Point);
        state.set_goal(Relation::collinear(a, b, c));

        let config = MctsConfig {
            num_iterations: 0,
            max_children: 5,
            c_puct: 1.4,
            max_depth: 2,
        };
        let result = mcts_search(state, &config);
        assert!(!result.solved);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_mcts_depth_one() {
        // With max_depth=1, MCTS should still be able to solve single-step problems
        let mut state = ProofState::new();
        let a = state.add_object("a", ObjectType::Point);
        let b = state.add_object("b", ObjectType::Point);
        state.add_object("c", ObjectType::Point);
        state.set_goal(Relation::congruent(a, 3, 3, b));

        let config = MctsConfig {
            num_iterations: 200,
            max_children: 30,
            c_puct: 1.4,
            max_depth: 1,
        };
        let result = mcts_search(state, &config);
        assert!(result.solved, "Should solve at depth 1");
    }

    #[test]
    fn test_extract_proof_path_root_is_empty() {
        let state = ProofState::new();
        let root = MctsNode::new_root(state);
        let path = extract_proof_path(&root);
        assert!(path.is_empty());
    }

    #[test]
    fn test_mcts_result_iterations_count() {
        let mut state = ProofState::new();
        let a = state.add_object("a", ObjectType::Point);
        let b = state.add_object("b", ObjectType::Point);
        state.add_object("c", ObjectType::Point);
        state.set_goal(Relation::congruent(a, 3, 3, b));

        let config = MctsConfig {
            num_iterations: 500,
            max_children: 30,
            c_puct: 1.4,
            max_depth: 2,
        };
        let result = mcts_search(state, &config);
        assert!(result.solved);
        // Should solve in fewer iterations than the max
        assert!(result.iterations < 500);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_node_depth_two_levels() {
        let state = ProofState::new();
        let root = MctsNode::new_root(state);
        let child = MctsNode::new_child(
            &root,
            crate::construction::Construction {
                ctype: crate::construction::ConstructionType::Midpoint,
                args: vec![0, 1],
                priority: crate::construction::Priority::Exploratory,
            },
            ProofState::new(),
        );
        let grandchild = MctsNode::new_child(
            &child,
            crate::construction::Construction {
                ctype: crate::construction::ConstructionType::Altitude,
                args: vec![0, 1, 2],
                priority: crate::construction::Priority::Exploratory,
            },
            ProofState::new(),
        );
        assert_eq!(node_depth(&grandchild), 2);
    }
}
