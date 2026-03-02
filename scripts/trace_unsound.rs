use geoprover::construction::{generate_constructions, apply_construction};
use geoprover::deduction::saturate;
use geoprover::proof_state::{ObjectType, ProofState, Relation};

fn main() {
    let mut state = ProofState::new();
    let a = state.add_object("a", ObjectType::Point);
    let b = state.add_object("b", ObjectType::Point);
    let c = state.add_object("c", ObjectType::Point);
    state.set_goal(Relation::collinear(a, b, c));

    let constructions = generate_constructions(&state);
    println!("Total constructions: {}", constructions.len());

    // Try first 5 (matching the test config)
    for (i, con) in constructions.iter().take(5).enumerate() {
        let mut child_state = apply_construction(&state, con);
        println!("\n=== Construction {}: {:?} ===", i, con);
        println!("Facts before saturate: {}", child_state.facts.len());
        for f in &child_state.facts {
            println!("  {:?}", f);
        }

        let proved = saturate(&mut child_state);
        println!("Facts after saturate: {}", child_state.facts.len());
        println!("Proved: {}", proved);

        if proved {
            println!("!!! UNSOUND PROOF FOUND !!!");
            println!("All facts:");
            for f in &child_state.facts {
                println!("  {:?}", f);
            }
            // Find collinear facts
            println!("\nCollinear facts:");
            for f in &child_state.facts {
                if matches!(f, Relation::Collinear(..)) {
                    println!("  {:?}", f);
                }
            }
        }

        // Also try depth 2
        if !proved {
            let child_constructions = generate_constructions(&child_state);
            for (j, con2) in child_constructions.iter().take(5).enumerate() {
                let mut grandchild = apply_construction(&child_state, con2);
                let proved2 = saturate(&mut grandchild);
                if proved2 {
                    println!("\n!!! UNSOUND PROOF at depth 2: con[{}] then con2[{}] !!!", i, j);
                    println!("con: {:?}", con);
                    println!("con2: {:?}", con2);
                    println!("\nCollinear facts:");
                    for f in &grandchild.facts {
                        if matches!(f, Relation::Collinear(..)) {
                            println!("  {:?}", f);
                        }
                    }
                    println!("\nParallel facts:");
                    for f in &grandchild.facts {
                        if matches!(f, Relation::Parallel(..)) {
                            println!("  {:?}", f);
                        }
                    }
                    println!("\nEqualAngle facts:");
                    for f in &grandchild.facts {
                        if matches!(f, Relation::EqualAngle(..)) {
                            println!("  {:?}", f);
                        }
                    }
                    return;
                }
            }
        }
    }
    println!("\nNo unsound proof found in first 5 constructions at depth ≤ 2");
}
