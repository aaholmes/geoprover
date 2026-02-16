//! Synthetic training data generator.
//!
//! Generates (state_text, construction_text, goal_text) tuples by:
//! 1. Creating a random base configuration (triangle/quad)
//! 2. Applying 1-3 random constructions + saturation to build facts
//! 3. Applying one more "key" construction + saturation
//! 4. Recording (state_before_key, key_construction, new_goal) triples

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

/// Pick a random applicable construction from the available ones.
fn random_construction(rng: &mut impl Rng, state: &ProofState) -> Option<Construction> {
    let constructions = generate_constructions(state);
    if constructions.is_empty() {
        return None;
    }
    // Only use the 7 types we actually implement
    let useful_types = [
        ConstructionType::Midpoint,
        ConstructionType::Altitude,
        ConstructionType::Circumcenter,
        ConstructionType::Orthocenter,
        ConstructionType::Incenter,
        ConstructionType::ParallelThrough,
        ConstructionType::PerpendicularThrough,
    ];
    let filtered: Vec<&Construction> = constructions
        .iter()
        .filter(|c| useful_types.contains(&c.ctype))
        .collect();
    let pool = if filtered.is_empty() {
        constructions.iter().collect::<Vec<_>>()
    } else {
        filtered
    };
    pool.choose(rng).map(|c| (*c).clone())
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

/// Generate training examples from a single random configuration.
fn generate_one(rng: &mut impl Rng) -> Vec<(String, String, String)> {
    let config = synth_saturate_config();
    let n_points = rng.gen_range(3..=4);
    let base = random_base_state(n_points);

    // Apply 1-2 "setup" constructions to create a richer state
    let n_setup = rng.gen_range(1..=2);
    let mut setup_state = base;
    for _ in 0..n_setup {
        if let Some(c) = random_construction(rng, &setup_state) {
            setup_state = apply_construction(&setup_state, &c);
        }
    }

    // Saturate the setup state to get all deducible facts
    let mut before_state = setup_state.clone();
    deduction::saturate_with_config(&mut before_state, &config);

    // Skip if the state is too trivial (< 3 facts) or too large (> 200)
    if before_state.facts.len() < 3 || before_state.facts.len() > 200 {
        return Vec::new();
    }

    let before_text = before_state.to_text();

    // Apply one more construction (the "key" construction)
    let construction = match random_construction(rng, &before_state) {
        Some(c) => c,
        None => return Vec::new(),
    };
    let construction_text = construction.to_text(&before_state);

    // Apply and saturate
    let mut after_state = apply_construction(&before_state, &construction);
    deduction::saturate_with_config(&mut after_state, &config);

    // Find new interesting facts
    let new_facts: Vec<Relation> = after_state
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
        let goal_text = after_state.relation_to_text_pub(fact);
        examples.push((
            before_text.clone(),
            construction_text.clone(),
            goal_text,
        ));
    }

    examples
}

/// Generate a batch of synthetic training examples.
///
/// Returns Vec of (state_text, construction_text, goal_text) tuples.
pub fn generate_batch(num_examples: usize, seed: u64) -> Vec<(String, String, String)> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut results = Vec::new();

    let max_attempts = num_examples * 30;
    let mut attempts = 0;

    while results.len() < num_examples && attempts < max_attempts {
        attempts += 1;
        let examples = generate_one(&mut rng);
        for ex in examples {
            if results.len() >= num_examples {
                break;
            }
            results.push(ex);
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

            // Construction should start with a known keyword
            let keywords = ["mid", "alt", "circumcenter", "orthocenter", "incenter", "pthrough", "tthrough"];
            let starts_valid = keywords.iter().any(|kw| construction.starts_with(kw));
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
}
