use std::collections::{HashMap, HashSet};
use rand::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ObjectType {
    Point,
    Line,
    Circle,
}

#[derive(Clone, Debug)]
pub struct GeoObject {
    pub id: u16,
    pub otype: ObjectType,
    pub name: String,
}

/// Relations between geometric objects. Each variant stores object IDs (u16).
/// Relations are stored in canonical form for deduplication.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Relation {
    /// Three collinear points (sorted)
    Collinear(u16, u16, u16),
    /// Lines AB and CD are parallel
    Parallel(u16, u16, u16, u16),
    /// Lines AB and CD are perpendicular
    Perpendicular(u16, u16, u16, u16),
    /// |AB| = |CD| — segment congruence (canonical: (min(a,b), max(a,b), min(c,d), max(c,d)), first pair <= second pair)
    Congruent(u16, u16, u16, u16),
    /// Angle(a,b,c) = Angle(d,e,f) — equal angles where b,e are vertices
    /// Stored as 8 IDs: a,b,c,d,e,f with canonical ordering of the two angle triples
    EqualAngle(u16, u16, u16, u16, u16, u16),
    /// M is the midpoint of segment AB
    Midpoint(u16, u16, u16),
    /// Point is on circle
    OnCircle(u16, u16),
    /// Four concyclic points (sorted)
    Cyclic(u16, u16, u16, u16),
}

impl Relation {
    /// Create a Collinear relation in canonical (sorted) form
    pub fn collinear(a: u16, b: u16, c: u16) -> Self {
        let mut pts = [a, b, c];
        pts.sort();
        Relation::Collinear(pts[0], pts[1], pts[2])
    }

    /// Create a Parallel relation in canonical form
    /// Parallel(a,b,c,d) means line AB || line CD
    /// Canonical: each pair sorted, then pairs sorted lexicographically
    pub fn parallel(a: u16, b: u16, c: u16, d: u16) -> Self {
        let p1 = if a <= b { (a, b) } else { (b, a) };
        let p2 = if c <= d { (c, d) } else { (d, c) };
        let (p1, p2) = if p1 <= p2 { (p1, p2) } else { (p2, p1) };
        Relation::Parallel(p1.0, p1.1, p2.0, p2.1)
    }

    /// Create a Perpendicular relation in canonical form
    pub fn perpendicular(a: u16, b: u16, c: u16, d: u16) -> Self {
        let p1 = if a <= b { (a, b) } else { (b, a) };
        let p2 = if c <= d { (c, d) } else { (d, c) };
        let (p1, p2) = if p1 <= p2 { (p1, p2) } else { (p2, p1) };
        Relation::Perpendicular(p1.0, p1.1, p2.0, p2.1)
    }

    /// Create a Congruent relation in canonical form: |AB| = |CD|
    /// Canonical: each pair sorted, then pairs sorted lexicographically
    pub fn congruent(a: u16, b: u16, c: u16, d: u16) -> Self {
        let p1 = if a <= b { (a, b) } else { (b, a) };
        let p2 = if c <= d { (c, d) } else { (d, c) };
        let (p1, p2) = if p1 <= p2 { (p1, p2) } else { (p2, p1) };
        Relation::Congruent(p1.0, p1.1, p2.0, p2.1)
    }

    /// Create an EqualAngle relation: angle(a,b,c) = angle(d,e,f)
    /// b and e are the vertices. Canonical: first triple <= second triple lexicographically
    pub fn equal_angle(a: u16, b: u16, c: u16, d: u16, e: u16, f: u16) -> Self {
        let t1 = (a, b, c);
        let t2 = (d, e, f);
        if t1 <= t2 {
            Relation::EqualAngle(t1.0, t1.1, t1.2, t2.0, t2.1, t2.2)
        } else {
            Relation::EqualAngle(t2.0, t2.1, t2.2, t1.0, t1.1, t1.2)
        }
    }

    /// Create a Midpoint relation: m is the midpoint of segment AB
    /// The segment endpoints are sorted
    pub fn midpoint(m: u16, a: u16, b: u16) -> Self {
        let (a, b) = if a <= b { (a, b) } else { (b, a) };
        Relation::Midpoint(m, a, b)
    }

    /// Create an OnCircle relation
    pub fn on_circle(point: u16, circle: u16) -> Self {
        Relation::OnCircle(point, circle)
    }

    /// Create a Cyclic relation for four concyclic points (sorted)
    pub fn cyclic(a: u16, b: u16, c: u16, d: u16) -> Self {
        let mut pts = [a, b, c, d];
        pts.sort();
        Relation::Cyclic(pts[0], pts[1], pts[2], pts[3])
    }
}

pub struct ProofState {
    pub objects: Vec<GeoObject>,
    pub name_to_id: HashMap<String, u16>,
    pub facts: HashSet<Relation>,
    pub goal: Option<Relation>,
    pub hash: u64,
    zobrist_table: HashMap<Relation, u64>,
    rng: rand::rngs::StdRng,
}

impl Default for ProofState {
    fn default() -> Self {
        Self::new()
    }
}

impl ProofState {
    pub fn new() -> Self {
        use rand::SeedableRng;
        ProofState {
            objects: Vec::new(),
            name_to_id: HashMap::new(),
            facts: HashSet::new(),
            goal: None,
            hash: 0,
            zobrist_table: HashMap::new(),
            rng: rand::rngs::StdRng::seed_from_u64(0x6E0_9B0FE12),
        }
    }

    /// Add an object, returning its ID. If a name collision occurs, returns existing ID.
    pub fn add_object(&mut self, name: &str, otype: ObjectType) -> u16 {
        if let Some(&id) = self.name_to_id.get(name) {
            return id;
        }
        let id = self.objects.len() as u16;
        self.objects.push(GeoObject {
            id,
            otype,
            name: name.to_string(),
        });
        self.name_to_id.insert(name.to_string(), id);
        id
    }

    /// Add a fact (relation) to the state. Updates Zobrist hash. Returns true if new.
    pub fn add_fact(&mut self, fact: Relation) -> bool {
        if self.facts.contains(&fact) {
            return false;
        }
        let zobrist = self.get_or_create_zobrist(&fact);
        self.hash ^= zobrist;
        self.facts.insert(fact);
        true
    }

    /// Check if the goal has been proved (goal is in the fact set)
    pub fn is_proved(&self) -> bool {
        match &self.goal {
            Some(goal) => self.facts.contains(goal),
            None => false,
        }
    }

    /// Set the goal relation
    pub fn set_goal(&mut self, goal: Relation) {
        self.goal = Some(goal);
    }

    fn get_or_create_zobrist(&mut self, fact: &Relation) -> u64 {
        if let Some(&z) = self.zobrist_table.get(fact) {
            return z;
        }
        let z = self.rng.gen::<u64>();
        self.zobrist_table.insert(fact.clone(), z);
        z
    }

    /// Get object ID by name
    pub fn id(&self, name: &str) -> u16 {
        self.name_to_id[name]
    }

    /// Try to get object ID by name
    pub fn try_id(&self, name: &str) -> Option<u16> {
        self.name_to_id.get(name).copied()
    }
}

impl Clone for ProofState {
    fn clone(&self) -> Self {
        ProofState {
            objects: self.objects.clone(),
            name_to_id: self.name_to_id.clone(),
            facts: self.facts.clone(),
            goal: self.goal.clone(),
            hash: self.hash,
            zobrist_table: self.zobrist_table.clone(),
            rng: self.rng.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_triangle_objects() {
        let mut state = ProofState::new();
        let a = state.add_object("a", ObjectType::Point);
        let b = state.add_object("b", ObjectType::Point);
        let c = state.add_object("c", ObjectType::Point);
        assert_eq!(state.objects.len(), 3);
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(c, 2);
    }

    #[test]
    fn test_add_fact_and_dedup() {
        let mut state = ProofState::new();
        let a = state.add_object("a", ObjectType::Point);
        let b = state.add_object("b", ObjectType::Point);
        let c = state.add_object("c", ObjectType::Point);
        assert!(state.add_fact(Relation::collinear(a, b, c)));
        assert!(!state.add_fact(Relation::collinear(a, b, c)));
        assert_eq!(state.facts.len(), 1);
    }

    #[test]
    fn test_canonical_congruent() {
        // Congruent(a,b,c,d) should equal Congruent(c,d,a,b)
        let r1 = Relation::congruent(0, 1, 2, 3);
        let r2 = Relation::congruent(2, 3, 0, 1);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_canonical_collinear() {
        let r1 = Relation::collinear(2, 0, 1);
        let r2 = Relation::collinear(0, 1, 2);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_canonical_parallel() {
        let r1 = Relation::parallel(1, 0, 3, 2);
        let r2 = Relation::parallel(0, 1, 2, 3);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_canonical_equal_angle() {
        // angle(a,b,c) = angle(d,e,f) should equal angle(d,e,f) = angle(a,b,c)
        let r1 = Relation::equal_angle(0, 1, 2, 3, 4, 5);
        let r2 = Relation::equal_angle(3, 4, 5, 0, 1, 2);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_is_proved_true() {
        let mut state = ProofState::new();
        let a = state.add_object("a", ObjectType::Point);
        let b = state.add_object("b", ObjectType::Point);
        let c = state.add_object("c", ObjectType::Point);
        let goal = Relation::collinear(a, b, c);
        state.set_goal(goal.clone());
        state.add_fact(goal);
        assert!(state.is_proved());
    }

    #[test]
    fn test_is_proved_false() {
        let mut state = ProofState::new();
        let a = state.add_object("a", ObjectType::Point);
        let b = state.add_object("b", ObjectType::Point);
        let c = state.add_object("c", ObjectType::Point);
        state.set_goal(Relation::collinear(a, b, c));
        assert!(!state.is_proved());
    }

    #[test]
    fn test_zobrist_hash_changes() {
        let mut state = ProofState::new();
        let a = state.add_object("a", ObjectType::Point);
        let b = state.add_object("b", ObjectType::Point);
        let c = state.add_object("c", ObjectType::Point);
        let hash_before = state.hash;
        state.add_fact(Relation::collinear(a, b, c));
        assert_ne!(state.hash, hash_before);
    }

    #[test]
    fn test_zobrist_hash_order_independent() {
        let mut state1 = ProofState::new();
        let a1 = state1.add_object("a", ObjectType::Point);
        let b1 = state1.add_object("b", ObjectType::Point);
        let c1 = state1.add_object("c", ObjectType::Point);
        let d1 = state1.add_object("d", ObjectType::Point);
        state1.add_fact(Relation::collinear(a1, b1, c1));
        state1.add_fact(Relation::congruent(a1, b1, c1, d1));

        let mut state2 = ProofState::new();
        let a2 = state2.add_object("a", ObjectType::Point);
        let b2 = state2.add_object("b", ObjectType::Point);
        let c2 = state2.add_object("c", ObjectType::Point);
        let d2 = state2.add_object("d", ObjectType::Point);
        // Add in reverse order
        state2.add_fact(Relation::congruent(a2, b2, c2, d2));
        state2.add_fact(Relation::collinear(a2, b2, c2));

        assert_eq!(state1.hash, state2.hash);
    }

    #[test]
    fn test_clone_independent() {
        let mut state = ProofState::new();
        let a = state.add_object("a", ObjectType::Point);
        let b = state.add_object("b", ObjectType::Point);
        let c = state.add_object("c", ObjectType::Point);
        state.add_fact(Relation::collinear(a, b, c));

        let mut cloned = state.clone();
        let d = cloned.add_object("d", ObjectType::Point);
        cloned.add_fact(Relation::congruent(a, b, c, d));

        assert_eq!(state.objects.len(), 3);
        assert_eq!(cloned.objects.len(), 4);
        assert_eq!(state.facts.len(), 1);
        assert_eq!(cloned.facts.len(), 2);
    }

    #[test]
    fn test_add_object_name_collision() {
        let mut state = ProofState::new();
        let a1 = state.add_object("a", ObjectType::Point);
        let a2 = state.add_object("a", ObjectType::Point);
        assert_eq!(a1, a2);
        assert_eq!(state.objects.len(), 1);
    }
}
