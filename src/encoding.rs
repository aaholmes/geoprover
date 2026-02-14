// State-to-tensor encoding for neural network input

use crate::proof_state::{ObjectType, ProofState, Relation};

pub const NUM_CHANNELS: usize = 20;
pub const GRID_SIZE: usize = 32;
pub const TENSOR_SIZE: usize = NUM_CHANNELS * GRID_SIZE * GRID_SIZE;

/// Index into the flat tensor: tensor[channel][row][col]
#[inline]
fn idx(ch: usize, r: usize, c: usize) -> usize {
    ch * GRID_SIZE * GRID_SIZE + r * GRID_SIZE + c
}

/// Set tensor[ch][r][c] = val, only if r and c are within GRID_SIZE.
#[inline]
fn set(tensor: &mut [f32], ch: usize, r: u16, c: u16, val: f32) {
    let r = r as usize;
    let c = c as usize;
    if r < GRID_SIZE && c < GRID_SIZE {
        tensor[idx(ch, r, c)] = val;
    }
}

/// Set symmetric pair: tensor[ch][r][c] = tensor[ch][c][r] = val.
#[inline]
fn set_sym(tensor: &mut [f32], ch: usize, r: u16, c: u16, val: f32) {
    set(tensor, ch, r, c, val);
    set(tensor, ch, c, r, val);
}

/// Encode a ProofState as a flat f32 tensor of shape (20, 32, 32).
/// Channel-first layout: index = channel * 32 * 32 + row * 32 + col.
/// Objects with IDs >= 32 are silently ignored.
pub fn state_to_tensor(state: &ProofState) -> Vec<f32> {
    let mut tensor = vec![0.0f32; TENSOR_SIZE];

    // Channels 0-11: Relation encoding
    for fact in &state.facts {
        encode_relation(&mut tensor, fact);
    }

    // Channels 12-15: Goal encoding
    if let Some(goal) = &state.goal {
        encode_goal(&mut tensor, goal, state);
    }

    // Channels 16-19: Object-type encoding (on diagonal)
    for obj in &state.objects {
        let i = obj.id;
        if (i as usize) >= GRID_SIZE {
            continue;
        }
        // Ch 16: Is Point
        if obj.otype == ObjectType::Point {
            set(&mut tensor, 16, i, i, 1.0);
        }
        // Ch 17: Is Circle center
        if obj.otype == ObjectType::Circle {
            set(&mut tensor, 17, i, i, 1.0);
        }
        // Ch 18: Is auxiliary (name starts with "aux_")
        if obj.name.starts_with("aux_") {
            set(&mut tensor, 18, i, i, 1.0);
        }
        // Ch 19: Positional encoding
        set(&mut tensor, 19, i, i, i as f32 / 32.0);
    }

    tensor
}

/// Encode a single relation into the appropriate channels.
fn encode_relation(tensor: &mut [f32], fact: &Relation) {
    match fact {
        // Ch 0: Collinear — mark all pairs
        Relation::Collinear(a, b, c) => {
            set_sym(tensor, 0, *a, *b, 1.0);
            set_sym(tensor, 0, *a, *c, 1.0);
            set_sym(tensor, 0, *b, *c, 1.0);
        }
        // Ch 1: Parallel — cross-line pairs
        Relation::Parallel(a, b, c, d) => {
            set_sym(tensor, 1, *a, *c, 1.0);
            set_sym(tensor, 1, *a, *d, 1.0);
            set_sym(tensor, 1, *b, *c, 1.0);
            set_sym(tensor, 1, *b, *d, 1.0);
        }
        // Ch 2: Perpendicular — cross-line pairs
        Relation::Perpendicular(a, b, c, d) => {
            set_sym(tensor, 2, *a, *c, 1.0);
            set_sym(tensor, 2, *a, *d, 1.0);
            set_sym(tensor, 2, *b, *c, 1.0);
            set_sym(tensor, 2, *b, *d, 1.0);
        }
        // Ch 3: Congruent-seg — mark segments
        // Ch 4: Congruent-cross — cross pairs
        Relation::Congruent(a, b, c, d) => {
            set_sym(tensor, 3, *a, *b, 1.0);
            set_sym(tensor, 3, *c, *d, 1.0);
            set_sym(tensor, 4, *a, *c, 1.0);
            set_sym(tensor, 4, *a, *d, 1.0);
            set_sym(tensor, 4, *b, *c, 1.0);
            set_sym(tensor, 4, *b, *d, 1.0);
        }
        // Ch 5: EqualAngle-vertex — vertex pair
        // Ch 6: EqualAngle-ray — ray endpoint pairs
        Relation::EqualAngle(a, b, c, d, e, f) => {
            // b,e are vertices
            set_sym(tensor, 5, *b, *e, 1.0);
            // a,d and c,f are ray endpoints
            set_sym(tensor, 6, *a, *d, 1.0);
            set_sym(tensor, 6, *c, *f, 1.0);
        }
        // Ch 7: Midpoint
        Relation::Midpoint(m, a, b) => {
            set_sym(tensor, 7, *m, *a, 1.0);
            set_sym(tensor, 7, *m, *b, 1.0);
        }
        // Ch 8: OnCircle
        Relation::OnCircle(p, c) => {
            set_sym(tensor, 8, *p, *c, 1.0);
        }
        // Ch 9: Cyclic — all ordered pairs
        Relation::Cyclic(a, b, c, d) => {
            let pts = [*a, *b, *c, *d];
            for i in 0..4 {
                for j in (i + 1)..4 {
                    set_sym(tensor, 9, pts[i], pts[j], 1.0);
                }
            }
        }
        // Ch 10: EqualRatio-seg — mark all 4 segment pairs
        // Ch 11: EqualRatio-cross — cross pairs between ratio sides
        Relation::EqualRatio(a, b, c, d, e, f, g, h) => {
            // Segments: AB, CD, EF, GH
            set_sym(tensor, 10, *a, *b, 1.0);
            set_sym(tensor, 10, *c, *d, 1.0);
            set_sym(tensor, 10, *e, *f, 1.0);
            set_sym(tensor, 10, *g, *h, 1.0);
            // Cross: left side (AB,CD) × right side (EF,GH)
            set_sym(tensor, 11, *a, *e, 1.0);
            set_sym(tensor, 11, *a, *f, 1.0);
            set_sym(tensor, 11, *b, *e, 1.0);
            set_sym(tensor, 11, *b, *f, 1.0);
            set_sym(tensor, 11, *c, *g, 1.0);
            set_sym(tensor, 11, *c, *h, 1.0);
            set_sym(tensor, 11, *d, *g, 1.0);
            set_sym(tensor, 11, *d, *h, 1.0);
        }
    }
}

/// Encode goal information into channels 12-14.
fn encode_goal(tensor: &mut [f32], goal: &Relation, state: &ProofState) {
    let goal_points = goal.point_ids();

    // Ch 12: Goal point pair mask — [i,j] = 1 if both i and j are goal points
    for i in 0..goal_points.len() {
        for j in (i + 1)..goal_points.len() {
            set_sym(tensor, 12, goal_points[i], goal_points[j], 1.0);
        }
    }

    // Ch 13: Goal type on diagonal — [i,i] = type_index/9.0 for each goal point
    let type_index = match goal {
        Relation::Collinear(..) => 0,
        Relation::Parallel(..) => 1,
        Relation::Perpendicular(..) => 2,
        Relation::Congruent(..) => 3,
        Relation::EqualAngle(..) => 4,
        Relation::Midpoint(..) => 5,
        Relation::OnCircle(..) => 6,
        Relation::Cyclic(..) => 7,
        Relation::EqualRatio(..) => 8,
    };
    let type_val = type_index as f32 / 9.0;
    // Set on diagonal for all objects (not just goal points) so the NN knows the goal type
    for obj in &state.objects {
        if (obj.id as usize) < GRID_SIZE {
            set(tensor, 13, obj.id, obj.id, type_val);
        }
    }

    // Ch 14: Goal same-relation channel — encode goal using relation channels scheme
    encode_goal_relation(tensor, goal);
}

/// Encode the goal relation into channel 14 using the same pattern as relation channels.
fn encode_goal_relation(tensor: &mut [f32], goal: &Relation) {
    match goal {
        Relation::Collinear(a, b, c) => {
            set_sym(tensor, 14, *a, *b, 1.0);
            set_sym(tensor, 14, *a, *c, 1.0);
            set_sym(tensor, 14, *b, *c, 1.0);
        }
        Relation::Parallel(a, b, c, d) | Relation::Perpendicular(a, b, c, d) => {
            set_sym(tensor, 14, *a, *c, 1.0);
            set_sym(tensor, 14, *a, *d, 1.0);
            set_sym(tensor, 14, *b, *c, 1.0);
            set_sym(tensor, 14, *b, *d, 1.0);
        }
        Relation::Congruent(a, b, c, d) => {
            set_sym(tensor, 14, *a, *b, 1.0);
            set_sym(tensor, 14, *c, *d, 1.0);
            set_sym(tensor, 14, *a, *c, 1.0);
            set_sym(tensor, 14, *b, *d, 1.0);
        }
        Relation::EqualAngle(a, b, c, d, e, f) => {
            set_sym(tensor, 14, *b, *e, 1.0);
            set_sym(tensor, 14, *a, *d, 1.0);
            set_sym(tensor, 14, *c, *f, 1.0);
        }
        Relation::Midpoint(m, a, b) => {
            set_sym(tensor, 14, *m, *a, 1.0);
            set_sym(tensor, 14, *m, *b, 1.0);
        }
        Relation::OnCircle(p, c) => {
            set_sym(tensor, 14, *p, *c, 1.0);
        }
        Relation::Cyclic(a, b, c, d) => {
            let pts = [*a, *b, *c, *d];
            for i in 0..4 {
                for j in (i + 1)..4 {
                    set_sym(tensor, 14, pts[i], pts[j], 1.0);
                }
            }
        }
        Relation::EqualRatio(a, b, c, d, e, f, g, h) => {
            set_sym(tensor, 14, *a, *b, 1.0);
            set_sym(tensor, 14, *c, *d, 1.0);
            set_sym(tensor, 14, *e, *f, 1.0);
            set_sym(tensor, 14, *g, *h, 1.0);
        }
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
    fn test_tensor_size() {
        assert_eq!(TENSOR_SIZE, 20 * 32 * 32);
        assert_eq!(TENSOR_SIZE, 20480);
    }

    #[test]
    fn test_empty_state_all_zeros() {
        let state = ProofState::new();
        let tensor = state_to_tensor(&state);
        assert_eq!(tensor.len(), TENSOR_SIZE);
        assert!(tensor.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_triangle_object_channels() {
        let state = make_triangle();
        let tensor = state_to_tensor(&state);

        // Ch 16: all three are Points → diagonal [0,0], [1,1], [2,2] = 1.0
        assert_eq!(tensor[idx(16, 0, 0)], 1.0);
        assert_eq!(tensor[idx(16, 1, 1)], 1.0);
        assert_eq!(tensor[idx(16, 2, 2)], 1.0);
        assert_eq!(tensor[idx(16, 3, 3)], 0.0);

        // Ch 17: none are circles
        assert_eq!(tensor[idx(17, 0, 0)], 0.0);

        // Ch 18: none are auxiliary
        assert_eq!(tensor[idx(18, 0, 0)], 0.0);

        // Ch 19: positional encoding
        assert_eq!(tensor[idx(19, 0, 0)], 0.0 / 32.0);
        assert_eq!(tensor[idx(19, 1, 1)], 1.0 / 32.0);
        assert_eq!(tensor[idx(19, 2, 2)], 2.0 / 32.0);
    }

    #[test]
    fn test_collinear_channel() {
        let mut state = make_triangle();
        state.add_fact(Relation::collinear(0, 1, 2));
        let tensor = state_to_tensor(&state);

        // Ch 0: all 6 directed pairs set
        assert_eq!(tensor[idx(0, 0, 1)], 1.0);
        assert_eq!(tensor[idx(0, 1, 0)], 1.0);
        assert_eq!(tensor[idx(0, 0, 2)], 1.0);
        assert_eq!(tensor[idx(0, 2, 0)], 1.0);
        assert_eq!(tensor[idx(0, 1, 2)], 1.0);
        assert_eq!(tensor[idx(0, 2, 1)], 1.0);

        // Other channels should be zero at these positions
        assert_eq!(tensor[idx(1, 0, 1)], 0.0);
    }

    #[test]
    fn test_parallel_channel() {
        let mut state = make_triangle();
        let d = state.add_object("d", ObjectType::Point);
        // para(a,b,c,d) → cross-line pairs
        state.add_fact(Relation::parallel(0, 1, 2, d));
        let tensor = state_to_tensor(&state);

        // Ch 1: [a,c], [a,d], [b,c], [b,d] + symmetric
        assert_eq!(tensor[idx(1, 0, 2)], 1.0);
        assert_eq!(tensor[idx(1, 2, 0)], 1.0);
        assert_eq!(tensor[idx(1, 0, 3)], 1.0);
        assert_eq!(tensor[idx(1, 1, 2)], 1.0);
        assert_eq!(tensor[idx(1, 1, 3)], 1.0);
    }

    #[test]
    fn test_perpendicular_channel() {
        let mut state = make_triangle();
        let d = state.add_object("d", ObjectType::Point);
        state.add_fact(Relation::perpendicular(0, 1, 2, d));
        let tensor = state_to_tensor(&state);

        // Ch 2: cross-line pairs
        assert_eq!(tensor[idx(2, 0, 2)], 1.0);
        assert_eq!(tensor[idx(2, 2, 0)], 1.0);
        assert_eq!(tensor[idx(2, 1, 3)], 1.0);
        assert_eq!(tensor[idx(2, 3, 1)], 1.0);
    }

    #[test]
    fn test_congruent_channels() {
        let mut state = make_triangle();
        let d = state.add_object("d", ObjectType::Point);
        state.add_fact(Relation::congruent(0, 1, 2, d));
        let tensor = state_to_tensor(&state);

        // Ch 3: segment pairs
        assert_eq!(tensor[idx(3, 0, 1)], 1.0);
        assert_eq!(tensor[idx(3, 1, 0)], 1.0);
        assert_eq!(tensor[idx(3, 2, 3)], 1.0);
        assert_eq!(tensor[idx(3, 3, 2)], 1.0);

        // Ch 4: cross pairs
        assert_eq!(tensor[idx(4, 0, 2)], 1.0);
        assert_eq!(tensor[idx(4, 0, 3)], 1.0);
        assert_eq!(tensor[idx(4, 1, 2)], 1.0);
        assert_eq!(tensor[idx(4, 1, 3)], 1.0);
    }

    #[test]
    fn test_equal_angle_channels() {
        let mut state = ProofState::new();
        for i in 0..6 {
            state.add_object(&format!("p{}", i), ObjectType::Point);
        }
        // eqangle(0,1,2, 3,4,5) → vertices 1,4; rays 0,3 and 2,5
        state.add_fact(Relation::equal_angle(0, 1, 2, 3, 4, 5));
        let tensor = state_to_tensor(&state);

        // Ch 5: vertex pair [1,4]
        assert_eq!(tensor[idx(5, 1, 4)], 1.0);
        assert_eq!(tensor[idx(5, 4, 1)], 1.0);

        // Ch 6: ray pairs [0,3] and [2,5]
        assert_eq!(tensor[idx(6, 0, 3)], 1.0);
        assert_eq!(tensor[idx(6, 3, 0)], 1.0);
        assert_eq!(tensor[idx(6, 2, 5)], 1.0);
        assert_eq!(tensor[idx(6, 5, 2)], 1.0);
    }

    #[test]
    fn test_midpoint_channel() {
        let mut state = make_triangle();
        // mid(0, 1, 2) — point 0 is midpoint of segment 1-2
        state.add_fact(Relation::midpoint(0, 1, 2));
        let tensor = state_to_tensor(&state);

        // Ch 7: [m,a], [m,b] + symmetric
        assert_eq!(tensor[idx(7, 0, 1)], 1.0);
        assert_eq!(tensor[idx(7, 1, 0)], 1.0);
        assert_eq!(tensor[idx(7, 0, 2)], 1.0);
        assert_eq!(tensor[idx(7, 2, 0)], 1.0);
    }

    #[test]
    fn test_oncircle_channel() {
        let mut state = ProofState::new();
        state.add_object("p", ObjectType::Point);
        state.add_object("c", ObjectType::Circle);
        state.add_fact(Relation::on_circle(0, 1));
        let tensor = state_to_tensor(&state);

        // Ch 8: [p,c] + symmetric
        assert_eq!(tensor[idx(8, 0, 1)], 1.0);
        assert_eq!(tensor[idx(8, 1, 0)], 1.0);
    }

    #[test]
    fn test_cyclic_channel() {
        let mut state = ProofState::new();
        for i in 0..4 {
            state.add_object(&format!("p{}", i), ObjectType::Point);
        }
        state.add_fact(Relation::cyclic(0, 1, 2, 3));
        let tensor = state_to_tensor(&state);

        // Ch 9: all 6 unordered pairs × 2 directions = 12 cells
        assert_eq!(tensor[idx(9, 0, 1)], 1.0);
        assert_eq!(tensor[idx(9, 1, 0)], 1.0);
        assert_eq!(tensor[idx(9, 0, 2)], 1.0);
        assert_eq!(tensor[idx(9, 2, 3)], 1.0);
        assert_eq!(tensor[idx(9, 1, 3)], 1.0);
    }

    #[test]
    fn test_equal_ratio_channels() {
        let mut state = ProofState::new();
        for i in 0..8 {
            state.add_object(&format!("p{}", i), ObjectType::Point);
        }
        state.add_fact(Relation::equal_ratio(0, 1, 2, 3, 4, 5, 6, 7));
        let tensor = state_to_tensor(&state);

        // Ch 10: all 4 segment pairs
        assert_eq!(tensor[idx(10, 0, 1)], 1.0);
        assert_eq!(tensor[idx(10, 2, 3)], 1.0);
        assert_eq!(tensor[idx(10, 4, 5)], 1.0);
        assert_eq!(tensor[idx(10, 6, 7)], 1.0);

        // Ch 11: cross pairs
        assert_eq!(tensor[idx(11, 0, 4)], 1.0);
        assert_eq!(tensor[idx(11, 1, 5)], 1.0);
        assert_eq!(tensor[idx(11, 2, 6)], 1.0);
        assert_eq!(tensor[idx(11, 3, 7)], 1.0);
    }

    #[test]
    fn test_goal_channels() {
        let mut state = make_triangle();
        state.set_goal(Relation::congruent(0, 1, 0, 2));
        let tensor = state_to_tensor(&state);

        // Ch 12: goal point pair mask — points 0,1,2 are all goal points
        assert_eq!(tensor[idx(12, 0, 1)], 1.0);
        assert_eq!(tensor[idx(12, 1, 0)], 1.0);
        assert_eq!(tensor[idx(12, 0, 2)], 1.0);
        assert_eq!(tensor[idx(12, 2, 0)], 1.0);
        assert_eq!(tensor[idx(12, 1, 2)], 1.0);

        // Ch 13: goal type on diagonal — congruent = 3, so 3/9 = 0.333...
        let expected_type = 3.0 / 9.0;
        assert!((tensor[idx(13, 0, 0)] - expected_type).abs() < 1e-6);
        assert!((tensor[idx(13, 1, 1)] - expected_type).abs() < 1e-6);
        assert!((tensor[idx(13, 2, 2)] - expected_type).abs() < 1e-6);

        // Ch 14: goal relation encoding — congruent segments + cross
        assert_eq!(tensor[idx(14, 0, 1)], 1.0);
        assert_eq!(tensor[idx(14, 0, 2)], 1.0);
    }

    #[test]
    fn test_goal_type_indices() {
        // Verify each goal type produces a distinct type_index on ch13
        let mut state = make_triangle();

        let goals_and_expected = vec![
            (Relation::collinear(0, 1, 2), 0.0 / 9.0),
            (Relation::parallel(0, 1, 0, 2), 1.0 / 9.0),
            (Relation::perpendicular(0, 1, 0, 2), 2.0 / 9.0),
            (Relation::congruent(0, 1, 0, 2), 3.0 / 9.0),
            (Relation::equal_angle(0, 1, 2, 0, 2, 1), 4.0 / 9.0),
            (Relation::midpoint(0, 1, 2), 5.0 / 9.0),
            (Relation::on_circle(0, 1), 6.0 / 9.0),
        ];

        for (goal, expected_val) in goals_and_expected {
            state.set_goal(goal);
            let tensor = state_to_tensor(&state);
            assert!(
                (tensor[idx(13, 0, 0)] - expected_val).abs() < 1e-6,
                "Goal type mismatch for expected {}", expected_val
            );
        }
    }

    #[test]
    fn test_objects_beyond_32_ignored() {
        let mut state = ProofState::new();
        for i in 0..35 {
            state.add_object(&format!("p{}", i), ObjectType::Point);
        }
        // Add a collinear fact involving objects beyond GRID_SIZE
        state.add_fact(Relation::collinear(0, 33, 34));
        let tensor = state_to_tensor(&state);

        // Should not panic, tensor size correct
        assert_eq!(tensor.len(), TENSOR_SIZE);

        // Object 0 should be encoded, 33/34 should be silently ignored
        assert_eq!(tensor[idx(16, 0, 0)], 1.0); // point 0 exists
        // The collinear entries involving 33/34 should not set anything in-bounds
        // but [0,1] for the collinear encoding of (0,33,34) should be 0
        // since 33 is out of range
    }

    #[test]
    fn test_mixed_facts() {
        let mut state = ProofState::new();
        for i in 0..5 {
            state.add_object(&format!("p{}", i), ObjectType::Point);
        }
        state.add_fact(Relation::collinear(0, 1, 2));
        state.add_fact(Relation::parallel(0, 1, 3, 4));
        state.add_fact(Relation::congruent(0, 1, 2, 3));
        let tensor = state_to_tensor(&state);

        // Verify each channel has the right data
        assert_eq!(tensor[idx(0, 0, 1)], 1.0); // collinear
        assert_eq!(tensor[idx(1, 0, 3)], 1.0); // parallel
        assert_eq!(tensor[idx(3, 0, 1)], 1.0); // congruent-seg
        assert_eq!(tensor[idx(4, 0, 2)], 1.0); // congruent-cross
    }

    #[test]
    fn test_auxiliary_object_channel() {
        let mut state = ProofState::new();
        state.add_object("a", ObjectType::Point);
        state.add_object("aux_3", ObjectType::Point);
        let tensor = state_to_tensor(&state);

        assert_eq!(tensor[idx(18, 0, 0)], 0.0); // "a" is not auxiliary
        assert_eq!(tensor[idx(18, 1, 1)], 1.0); // "aux_3" is auxiliary
    }

    #[test]
    fn test_circle_object_channel() {
        let mut state = ProofState::new();
        state.add_object("center", ObjectType::Circle);
        state.add_object("p", ObjectType::Point);
        let tensor = state_to_tensor(&state);

        assert_eq!(tensor[idx(17, 0, 0)], 1.0); // circle
        assert_eq!(tensor[idx(17, 1, 1)], 0.0); // point
        assert_eq!(tensor[idx(16, 0, 0)], 0.0); // circle is not a point
        assert_eq!(tensor[idx(16, 1, 1)], 1.0); // point
    }

    #[test]
    fn test_no_goal_channels_empty() {
        let state = make_triangle();
        let tensor = state_to_tensor(&state);

        // Channels 12-14 should be all zeros when no goal
        for ch in 12..15 {
            for r in 0..GRID_SIZE {
                for c in 0..GRID_SIZE {
                    assert_eq!(tensor[idx(ch, r, c)], 0.0,
                        "ch={} r={} c={} should be 0 with no goal", ch, r, c);
                }
            }
        }
    }

    #[test]
    fn test_channel_15_reserved_zeros() {
        let mut state = make_triangle();
        state.add_fact(Relation::collinear(0, 1, 2));
        state.set_goal(Relation::congruent(0, 1, 0, 2));
        let tensor = state_to_tensor(&state);

        // Channel 15 is reserved — always zeros
        for r in 0..GRID_SIZE {
            for c in 0..GRID_SIZE {
                assert_eq!(tensor[idx(15, r, c)], 0.0);
            }
        }
    }
}
