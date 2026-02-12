use geoprover::parser::parse_problem;
use geoprover::proof_state::Relation;

fn main() {
    let input = "test\na b c = eq_triangle a b c ? cong a b b c";
    let state = parse_problem(input).unwrap();
    let a = state.id("a");
    let b = state.id("b");
    let c = state.id("c");
    println!("a={}, b={}, c={}", a, b, c);
    println!("Facts:");
    for f in &state.facts {
        println!("  {:?}", f);
    }
    println!("Expected: {:?}", Relation::congruent(a, b, a, c));
    println!("Expected: {:?}", Relation::congruent(a, b, b, c));
}
