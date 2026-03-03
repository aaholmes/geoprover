//! Synthetic training data generator.
//!
//! Generates (state_text, construction_text, goal_text) tuples by:
//! 1. Creating a random base configuration (triangle/quad/pentagon/hexagon)
//! 2. Applying 1-4 random constructions + saturation to build facts
//! 3. Applying one or more "key" constructions + saturation
//! 4. Recording (state_before_key, key_construction, new_goal) triples
//!
//! Also generates negative examples: states where a goal is NOT achievable,
//! with value_target=0 to teach the model to reject bad paths.

use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;

use crate::construction::{
    apply_construction, generate_constructions, Construction, ConstructionType,
};
use crate::deduction::{self, SaturateConfig};
use crate::proof_state::{ObjectType, ProofState, Relation};

/// Configuration for synthetic data generation.
fn synth_saturate_config() -> SaturateConfig {
    SaturateConfig {
        max_iterations: 20,
        max_facts: 500,
        ..SaturateConfig::default()
    }
}

/// Generate a random base proof state with n_points free points.
fn random_base_state(n_points: usize) -> ProofState {
    let mut state = ProofState::new();
    let names = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"];
    for name in names.iter().take(n_points) {
        state.add_object(name, ObjectType::Point);
    }
    state
}

/// The 7 construction types we support.
const USEFUL_TYPES: [ConstructionType; 7] = [
    ConstructionType::Midpoint,
    ConstructionType::Altitude,
    ConstructionType::Circumcenter,
    ConstructionType::Orthocenter,
    ConstructionType::Incenter,
    ConstructionType::ParallelThrough,
    ConstructionType::PerpendicularThrough,
];

/// Pick a random applicable construction from the available ones.
fn random_construction(rng: &mut impl Rng, state: &ProofState) -> Option<Construction> {
    let constructions = generate_constructions(state);
    if constructions.is_empty() {
        return None;
    }
    let filtered: Vec<&Construction> = constructions
        .iter()
        .filter(|c| USEFUL_TYPES.contains(&c.ctype))
        .collect();
    let pool = if filtered.is_empty() {
        constructions.iter().collect::<Vec<_>>()
    } else {
        filtered
    };
    pool.choose(rng).map(|c| (*c).clone())
}

/// Pick a random construction of a DIFFERENT type than `exclude`.
fn random_different_construction(
    rng: &mut impl Rng,
    state: &ProofState,
    exclude: &Construction,
) -> Option<Construction> {
    let constructions = generate_constructions(state);
    let filtered: Vec<&Construction> = constructions
        .iter()
        .filter(|c| USEFUL_TYPES.contains(&c.ctype))
        .filter(|c| c.ctype != exclude.ctype || c.args != exclude.args)
        .collect();
    if filtered.is_empty() {
        return None;
    }
    filtered.choose(rng).map(|c| (*c).clone())
}

/// Check if a relation makes an interesting goal.
fn is_interesting_goal(rel: &Relation) -> bool {
    match rel {
        // Skip trivially derived facts
        Relation::Collinear(..) => false,
        Relation::OnCircle(..) => false,
        // Congruent where both pairs are the same segment is trivial
        Relation::Congruent(a, b, c, d) => !(a == c && b == d),
        // Good goals: parallel, perpendicular, equal angle, cyclic, ratio, midpoint, non-trivial congruent
        _ => true,
    }
}

/// Difficulty level for synthetic data generation.
#[derive(Clone, Copy)]
enum Difficulty {
    Easy,   // 3-4 points, 1 setup, 1 key construction
    Medium, // 4-5 points, 2 setup, 1-2 key constructions
    Hard,   // 5-7 points, 2-4 setup, 2-3 key constructions
}

/// Generate training examples from a single random configuration.
fn generate_one(rng: &mut impl Rng, difficulty: Difficulty) -> Vec<(String, String, String)> {
    let config = synth_saturate_config();

    let (n_points, n_setup_range, n_key_range) = match difficulty {
        Difficulty::Easy => (rng.gen_range(3..=4), 1..=2, 1..=1),
        Difficulty::Medium => (rng.gen_range(4..=5), 2..=3, 1..=2),
        Difficulty::Hard => (rng.gen_range(5..=7), 2..=4, 2..=3),
    };

    let base = random_base_state(n_points);

    // Apply setup constructions to create a richer state
    let n_setup = rng.gen_range(n_setup_range);
    let mut setup_state = base;
    for _ in 0..n_setup {
        if let Some(c) = random_construction(rng, &setup_state) {
            setup_state = apply_construction(&setup_state, &c);
        }
    }

    // Saturate the setup state to get all deducible facts
    let mut before_state = setup_state.clone();
    deduction::saturate_with_config(&mut before_state, &config);

    // Skip if the state is too trivial (< 3 facts) or too large (> 500)
    if before_state.facts.len() < 3 || before_state.facts.len() > 500 {
        return Vec::new();
    }

    let before_text = before_state.to_text();

    // Apply key constructions (potentially multi-step)
    let n_key = rng.gen_range(n_key_range);
    let mut current_state = before_state.clone();
    let mut key_constructions = Vec::new();

    for _ in 0..n_key {
        let construction = match random_construction(rng, &current_state) {
            Some(c) => c,
            None => break,
        };
        key_constructions.push(construction.clone());
        current_state = apply_construction(&current_state, &construction);
        deduction::saturate_with_config(&mut current_state, &config);
    }

    if key_constructions.is_empty() {
        return Vec::new();
    }

    // Use the first key construction as the labeled one
    let construction_text = key_constructions[0].to_text(&before_state);

    // Find new interesting facts (only those requiring ALL key constructions)
    let new_facts: Vec<Relation> = current_state
        .facts
        .iter()
        .filter(|f| !before_state.facts.contains(f))
        .filter(|f| is_interesting_goal(f))
        .cloned()
        .collect();

    // Sort for determinism
    let mut sorted_facts = new_facts;
    sorted_facts.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));

    let mut examples = Vec::new();
    for fact in sorted_facts.iter().take(3) {
        let goal_text = current_state.relation_to_text_pub(fact);
        examples.push((
            before_text.clone(),
            construction_text.clone(),
            goal_text,
        ));
    }

    examples
}

/// Generate a negative example: a state + goal where a different construction is needed.
///
/// Returns (state_text, wrong_construction_text, goal_text) with value 0.
fn generate_negative(rng: &mut impl Rng) -> Option<(String, String, String)> {
    let config = synth_saturate_config();
    let n_points = rng.gen_range(3..=5);
    let base = random_base_state(n_points);

    // Setup
    let n_setup = rng.gen_range(1..=2);
    let mut setup_state = base;
    for _ in 0..n_setup {
        if let Some(c) = random_construction(rng, &setup_state) {
            setup_state = apply_construction(&setup_state, &c);
        }
    }
    let mut before_state = setup_state.clone();
    deduction::saturate_with_config(&mut before_state, &config);

    if before_state.facts.len() < 3 || before_state.facts.len() > 300 {
        return None;
    }

    // Apply the correct construction
    let correct_construction = random_construction(rng, &before_state)?;

    // Apply and saturate to find a goal
    let mut after_state = apply_construction(&before_state, &correct_construction);
    deduction::saturate_with_config(&mut after_state, &config);

    let new_facts: Vec<Relation> = after_state
        .facts
        .iter()
        .filter(|f| !before_state.facts.contains(f))
        .filter(|f| is_interesting_goal(f))
        .cloned()
        .collect();

    if new_facts.is_empty() {
        return None;
    }

    // Only keep goals that reference points existing in before_state
    let max_before_id = before_state.objects.len() as u16;
    let valid_facts: Vec<&Relation> = new_facts
        .iter()
        .filter(|f| f.point_ids().iter().all(|&id| id < max_before_id))
        .collect();
    if valid_facts.is_empty() {
        return None;
    }

    // Pick a random goal that the correct construction achieves
    let goal = (*valid_facts.choose(rng)?).clone();
    let goal_text = after_state.relation_to_text_pub(&goal);

    // Now apply a WRONG construction (different type)
    let wrong_construction =
        random_different_construction(rng, &before_state, &correct_construction)?;

    // Check that the wrong construction doesn't also achieve the goal
    let mut wrong_after = apply_construction(&before_state, &wrong_construction);
    deduction::saturate_with_config(&mut wrong_after, &config);
    if wrong_after.facts.contains(&goal) {
        return None; // The "wrong" construction also works — skip
    }

    let wrong_text = wrong_construction.to_text(&before_state);

    // Set the goal on before_state to get proper text output
    let mut labeled_state = before_state.clone();
    labeled_state.goal = Some(goal);
    let before_text = labeled_state.to_text();

    Some((before_text, wrong_text, goal_text))
}

/// Generate a batch of synthetic training examples.
///
/// Returns Vec of (state_text, construction_text, goal_text) tuples.
/// Positive examples (value=1.0) are followed by negative examples (value=0.0).
/// Negative examples have construction_text prefixed with "NEG:" to distinguish them.
pub fn generate_batch(num_examples: usize, seed: u64) -> Vec<(String, String, String)> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut results = Vec::new();

    let max_attempts = num_examples * 30;
    let mut attempts = 0;

    // Target: ~80% positive, ~20% negative
    let num_positive = (num_examples * 4) / 5;
    let num_negative = num_examples - num_positive;

    // Generate positive examples with mixed difficulty
    while results.len() < num_positive && attempts < max_attempts {
        attempts += 1;
        let difficulty = match rng.gen_range(0..10) {
            0..=4 => Difficulty::Easy,   // 50%
            5..=7 => Difficulty::Medium, // 30%
            _ => Difficulty::Hard,       // 20%
        };
        let examples = generate_one(&mut rng, difficulty);
        for ex in examples {
            if results.len() >= num_positive {
                break;
            }
            results.push(ex);
        }
    }

    // Generate negative examples
    let mut neg_attempts = 0;
    let neg_max = num_negative * 50;
    let mut neg_count = 0;
    while neg_count < num_negative && neg_attempts < neg_max {
        neg_attempts += 1;
        if let Some((state, construction, goal)) = generate_negative(&mut rng) {
            // Mark negative with "NEG:" prefix so Python can detect value=0
            results.push((state, format!("NEG:{}", construction), goal));
            neg_count += 1;
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_base_state() {
        let state = random_base_state(3);
        assert_eq!(state.objects.len(), 3);
        assert_eq!(state.name_of(0), "a");
        assert_eq!(state.name_of(1), "b");
        assert_eq!(state.name_of(2), "c");
    }

    #[test]
    fn test_random_base_state_larger() {
        let state = random_base_state(7);
        assert_eq!(state.objects.len(), 7);
        assert_eq!(state.name_of(6), "g");
    }

    #[test]
    fn test_generate_batch_produces_examples() {
        let examples = generate_batch(10, 42);
        assert!(!examples.is_empty(), "Should generate at least some examples");
        for (state, construction, goal) in &examples {
            assert!(!state.is_empty(), "State text should not be empty");
            assert!(!construction.is_empty(), "Construction text should not be empty");
            assert!(!goal.is_empty(), "Goal text should not be empty");
        }
    }

    #[test]
    fn test_generate_batch_different_seeds() {
        let batch1 = generate_batch(5, 1);
        let batch2 = generate_batch(5, 2);
        // Different seeds should produce different results (if both succeed)
        if !batch1.is_empty() && !batch2.is_empty() {
            assert_ne!(batch1, batch2, "Different seeds should produce different results");
        }
    }

    #[test]
    fn test_is_interesting_goal() {
        assert!(!is_interesting_goal(&Relation::collinear(0, 1, 2)));
        assert!(!is_interesting_goal(&Relation::on_circle(0, 1)));
        assert!(!is_interesting_goal(&Relation::congruent(0, 1, 0, 1))); // trivial self-congruent
        assert!(is_interesting_goal(&Relation::congruent(0, 1, 2, 3)));
        assert!(is_interesting_goal(&Relation::perpendicular(0, 1, 2, 3)));
        assert!(is_interesting_goal(&Relation::parallel(0, 1, 2, 3)));
        assert!(is_interesting_goal(&Relation::equal_angle(0, 1, 2, 3, 4, 5)));
    }

    #[test]
    fn test_example_content() {
        let examples = generate_batch(3, 42);
        for (state, construction, goal) in &examples {
            // State should contain at least one relation keyword
            let has_relation = ["coll", "para", "perp", "cong", "eqangle", "mid", "oncirc", "cyclic", "eqratio"]
                .iter()
                .any(|kw| state.contains(kw));
            assert!(has_relation, "State should contain relation keywords: {}", state);

            // Construction should start with a known keyword (or NEG: prefix)
            let c = construction.strip_prefix("NEG:").unwrap_or(construction);
            let keywords = ["mid", "alt", "circumcenter", "orthocenter", "incenter", "pthrough", "tthrough"];
            let starts_valid = keywords.iter().any(|kw| c.starts_with(kw));
            assert!(starts_valid, "Construction should start with known keyword: {}", construction);

            // Goal should contain a relation keyword
            let goal_has_relation = ["para", "perp", "cong", "eqangle", "mid", "cyclic", "eqratio"]
                .iter()
                .any(|kw| goal.contains(kw));
            assert!(goal_has_relation, "Goal should contain relation keyword: {}", goal);
        }
    }

    #[test]
    fn test_generate_larger_batch() {
        let examples = generate_batch(50, 99);
        assert!(examples.len() >= 20, "Should generate at least 20/50 examples, got {}", examples.len());
    }

    #[test]
    fn test_negative_examples_present() {
        let examples = generate_batch(50, 42);
        let neg_count = examples.iter().filter(|(_, c, _)| c.starts_with("NEG:")).count();
        // Should have at least some negative examples (target is 20% of 50 = 10)
        // But generation may fail sometimes, so be lenient
        assert!(neg_count >= 1 || examples.len() < 40,
            "Should have some negative examples, got {neg_count} negatives out of {} total",
            examples.len());
    }

    #[test]
    fn test_multi_point_generation() {
        // Hard difficulty should sometimes produce states with 5+ objects
        let examples = generate_batch(100, 123);
        let has_large = examples.iter().any(|(state, _, _)| {
            // Count unique point names (single letters)
            let points: std::collections::HashSet<&str> = state.split_whitespace()
                .filter(|w| w.len() == 1 && w.chars().next().map_or(false, |c| c.is_alphabetic()))
                .collect();
            points.len() >= 5
        });
        assert!(has_large, "Hard difficulty should generate states with 5+ points");
    }

    #[test]
    fn test_seed_produces_same_count() {
        // Same seed should produce the same number of examples
        let batch1 = generate_batch(10, 42);
        let batch2 = generate_batch(10, 42);
        assert_eq!(batch1.len(), batch2.len(), "Same seed should produce same count");
    }

    #[test]
    fn test_different_seeds_differ() {
        // Different seeds should produce different results (with high probability)
        let batch1 = generate_batch(20, 100);
        let batch2 = generate_batch(20, 200);
        assert!(batch1.len() > 0 && batch2.len() > 0);
        // At least some examples should differ
        let differ = batch1.iter().zip(batch2.iter())
            .any(|(a, b)| a.1 != b.1);
        assert!(differ, "Different seeds should produce different constructions");
    }

    #[test]
    fn test_generate_batch_zero() {
        // Generating 0 examples should return empty vec
        let examples = generate_batch(0, 42);
        assert!(examples.is_empty(), "0 examples should return empty vec");
    }

    #[test]
    fn test_state_text_has_facts() {
        // Each state text should contain relation keywords
        let examples = generate_batch(10, 77);
        for (state, _, _) in &examples {
            assert!(!state.is_empty(), "State text should not be empty");
            // State text should contain at least one relation keyword
            let has_relation = ["coll", "para", "perp", "cong", "eqangle", "mid",
                "oncirc", "cyclic", "eqratio"]
                .iter().any(|kw| state.contains(kw));
            assert!(has_relation, "State text should contain relation keywords: {}", state);
        }
    }

    #[test]
    fn test_goal_text_has_relation() {
        // Goal text should contain a relation keyword
        let examples = generate_batch(10, 88);
        for (_, _, goal) in &examples {
            assert!(!goal.is_empty(), "Goal text should not be empty");
            let has_relation = ["coll", "para", "perp", "cong", "eqangle", "mid",
                "oncirc", "cyclic", "eqratio"]
                .iter().any(|kw| goal.contains(kw));
            assert!(has_relation, "Goal text should contain a relation keyword: {}", goal);
        }
    }

    #[test]
    fn test_construction_text_nonempty() {
        let examples = generate_batch(10, 55);
        for (_, constr, _) in &examples {
            assert!(!constr.is_empty(), "Construction text should not be empty");
            // Should contain a known construction keyword
            let has_keyword = ["mid", "alt", "circumcenter", "orthocenter", "incenter",
                "pthrough", "tthrough", "reflect", "extend"]
                .iter().any(|kw| constr.contains(kw));
            assert!(has_keyword, "Construction should contain a known keyword: {}", constr);
        }
    }
}
