use geoprover::parser::parse_problem;
use geoprover::deduction::saturate;
use geoprover::proof_state::Relation;

fn main() {
    // PWW_B016x: medial triangle + parallelogram
    let problem = "PWW_B016x\na b c = triangle a b c; d = midpoint d b c; e = midpoint e c a; f = midpoint f b a; g = parallelogram d a e g ? cong c f g b";
    let mut state = parse_problem(problem).unwrap();
    let solved = saturate(&mut state);
    
    let c = state.id("c");
    let f = state.id("f");
    let g = state.id("g");
    let b = state.id("b");
    
    println!("Goal: Congruent(c,f, g,b) = {:?}", Relation::congruent(c, f, g, b));
    println!("Solved: {}", solved);
    println!("Total facts: {}", state.facts.len());
    
    // Check what congruent facts involving these points exist
    println!("\nCongruent facts involving c,f,g,b:");
    for fact in &state.facts {
        if let Relation::Congruent(a, b2, c2, d) = fact {
            let pts = [*a, *b2, *c2, *d];
            if pts.contains(&c) || pts.contains(&f) || pts.contains(&g) || pts.contains(&b) {
                println!("  {:?}", fact);
            }
        }
    }
    
    println!("\nParallel facts:");
    for fact in &state.facts {
        if let Relation::Parallel(_, _, _, _) = fact {
            println!("  {:?}", fact);
        }
    }
    
    println!("\n---\n");
    
    // intersection_pp problem (JAR_JAR02)
    let problem2 = "JAR_JAR02\na b c = triangle a b c; d = intersection_pp a b c c a b; e = intersection_ll e a c b d ? cong a e e c";
    let mut state2 = parse_problem(problem2).unwrap();
    let solved2 = saturate(&mut state2);
    println!("JAR_JAR02 solved: {}", solved2);
    println!("Facts: {}", state2.facts.len());
    
    let a = state2.id("a");
    let e = state2.id("e");
    let c = state2.id("c");
    
    println!("Goal: {:?}", Relation::congruent(a, e, e, c));
    println!("\nAll facts:");
    for fact in &state2.facts {
        println!("  {:?}", fact);
    }
}
