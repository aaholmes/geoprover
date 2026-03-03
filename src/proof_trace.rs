use crate::proof_state::{ProofState, Relation};
use std::collections::{HashMap, HashSet, VecDeque};

/// Names for all deduction rules, used for provenance tracking.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RuleName {
    Axiom,
    TransitiveParallel,
    PerpToParallel,
    MidpointDefinition,
    TransitiveCongruent,
    IsoscelesBaseAngles,
    AlternateInteriorAngles,
    CorrespondingAngles,
    TransitiveEqualAngle,
    PerpendicularAngles,
    CirclePointEquidistance,
    MidlineParallel,
    CyclicFromOncircle,
    EqualAnglesToParallel,
    MidpointConverse,
    CongruentOncircle,
    PerpendicularBisector,
    EquidistantMidpoint,
    PerpParallelTransfer,
    LineCollinearExtension,
    CollinearTransitivity,
    CyclicInscribedAngles,
    ParallelSharedPointCollinear,
    ThalesTheorem,
    InscribedAngleConverse,
    IsoscelesConverse,
    PerpMidpointCongruent,
    TwoEquidistantPerp,
    MidpointDiagonalParallelogram,
    CyclicEqualAngleCongruent,
    CyclicParallelEqangle,
    EquidistantCyclicPerp,
    MidpointParallelogram,
    EqanglePerpToPerp,
    SasCongruence,
    AsaCongruence,
    SssCongruence,
    TransitiveRatio,
    RatioOneCongruence,
    MidpointRatio,
    ParallelCollinearRatio,
    CongruentRatio,
    RatioCollinearParallel,
    ParallelogramOppositeAngles,
    IsoscelesTrapezoidBaseAngles,
    TrapezoidMidsegment,
    ParallelBaseRatio,
    ParallelProjection,
    EqualTangentLengths,
    TangentChordAngle,
    AngleBisectorRatio,
    IncenterEqualInradii,
    AaSimilarity,
    OrthocenterConcurrence,
    OppositeAnglesCyclic,
}

impl std::fmt::Display for RuleName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// A single derivation step: a fact derived by a rule from premises.
#[derive(Clone, Debug)]
pub struct Derivation {
    pub fact: Relation,
    pub rule: RuleName,
    pub premises: Vec<Relation>,
}

/// Proof trace: records how each fact was derived during saturation.
/// Premises are resolved lazily during extract_proof to avoid expensive
/// O(n^2+) searches for every derived fact during the hot saturation loop.
#[derive(Clone, Debug)]
pub struct ProofTrace {
    derivations: HashMap<Relation, Derivation>,
    axioms: HashSet<Relation>,
    /// All facts available for premise resolution (populated at end of saturation)
    all_facts: HashSet<Relation>,
}

impl ProofTrace {
    pub fn new() -> Self {
        ProofTrace {
            derivations: HashMap::new(),
            axioms: HashSet::new(),
            all_facts: HashSet::new(),
        }
    }

    /// Record a fact as an axiom (initial fact).
    pub fn add_axiom(&mut self, fact: Relation) {
        self.axioms.insert(fact.clone());
        // Also record a derivation entry for axioms
        self.derivations.entry(fact.clone()).or_insert(Derivation {
            fact,
            rule: RuleName::Axiom,
            premises: vec![],
        });
    }

    /// Record a derived fact. First derivation wins (no overwrite).
    pub fn add_derivation(&mut self, fact: Relation, rule: RuleName, premises: Vec<Relation>) {
        self.derivations.entry(fact.clone()).or_insert(Derivation {
            fact,
            rule,
            premises,
        });
    }

    /// Is a fact recorded as an axiom?
    pub fn is_axiom(&self, fact: &Relation) -> bool {
        self.axioms.contains(fact)
    }

    /// Number of derivation entries (axioms + derived).
    pub fn len(&self) -> usize {
        self.derivations.len()
    }

    /// Is the trace empty?
    pub fn is_empty(&self) -> bool {
        self.derivations.is_empty()
    }

    /// Set the complete fact set for lazy premise resolution.
    pub fn set_all_facts(&mut self, facts: HashSet<Relation>) {
        self.all_facts = facts;
    }

    /// Number of axioms.
    pub fn axiom_count(&self) -> usize {
        self.axioms.len()
    }

    /// Get the derivation for a fact, if it exists.
    pub fn get(&self, fact: &Relation) -> Option<&Derivation> {
        self.derivations.get(fact)
    }

    /// Iterate over axiom facts.
    pub fn axioms_iter(&self) -> impl Iterator<Item = &Relation> {
        self.axioms.iter()
    }

    /// Resolve premises for a derivation, using lazy identification if needed.
    fn resolve_premises(&self, deriv: &Derivation) -> Vec<Relation> {
        if deriv.rule == RuleName::Axiom {
            return vec![];
        }
        if !deriv.premises.is_empty() {
            return deriv.premises.clone();
        }
        // Lazy resolution: identify premises from the full fact set
        if !self.all_facts.is_empty() {
            identify_premises(&deriv.fact, &deriv.rule, &self.all_facts)
        } else {
            vec![]
        }
    }

    /// Extract a minimal proof: backward BFS from goal through premises,
    /// return topologically sorted derivations (axioms first, goal last).
    pub fn extract_proof(&self, goal: &Relation) -> Option<Vec<Derivation>> {
        // Check if goal is in our derivations
        if !self.derivations.contains_key(goal) {
            return None;
        }

        // First pass: resolve premises for facts on the proof path (lazy)
        let mut resolved: HashMap<Relation, Derivation> = HashMap::new();
        let mut queue: VecDeque<Relation> = VecDeque::new();
        let mut visited: HashSet<Relation> = HashSet::new();
        queue.push_back(goal.clone());
        visited.insert(goal.clone());

        while let Some(fact) = queue.pop_front() {
            if let Some(deriv) = self.derivations.get(&fact) {
                let premises = self.resolve_premises(deriv);
                for premise in &premises {
                    if visited.insert(premise.clone()) {
                        queue.push_back(premise.clone());
                    }
                }
                resolved.insert(
                    fact,
                    Derivation {
                        fact: deriv.fact.clone(),
                        rule: deriv.rule.clone(),
                        premises,
                    },
                );
            }
        }

        // Topological sort: axioms first, then derived facts in dependency order
        let mut result: Vec<Derivation> = Vec::new();
        let mut emitted: HashSet<Relation> = HashSet::new();

        let mut remaining: Vec<Relation> = resolved.keys().cloned().collect();
        let max_iters = remaining.len() * remaining.len() + 1;
        let mut iter_count = 0;

        while !remaining.is_empty() && iter_count < max_iters {
            iter_count += 1;
            let mut progress = false;

            remaining.retain(|fact| {
                if let Some(deriv) = resolved.get(fact) {
                    if deriv.premises.iter().all(|p| emitted.contains(p)) {
                        result.push(deriv.clone());
                        emitted.insert(fact.clone());
                        progress = true;
                        return false;
                    }
                }
                true
            });

            if !progress {
                // Cycle or missing premises — emit remaining as-is
                for fact in &remaining {
                    if let Some(deriv) = resolved.get(fact) {
                        result.push(deriv.clone());
                    }
                }
                break;
            }
        }

        Some(result)
    }

    /// Format a human-readable proof using point names from the state.
    pub fn format_proof(&self, goal: &Relation, state: &ProofState) -> Option<String> {
        let steps = self.extract_proof(goal)?;
        let mut lines = Vec::new();

        for (i, step) in steps.iter().enumerate() {
            let fact_text = state.relation_to_text_pub(&step.fact);

            if step.rule == RuleName::Axiom {
                lines.push(format!("  {}. {} [axiom]", i + 1, fact_text));
            } else {
                let premise_refs: Vec<String> = step
                    .premises
                    .iter()
                    .filter_map(|p| {
                        // Find the step number for this premise
                        steps.iter().position(|s| &s.fact == p).map(|idx| format!("{}", idx + 1))
                    })
                    .collect();
                let refs = if premise_refs.is_empty() {
                    String::new()
                } else {
                    format!(" from {}", premise_refs.join(", "))
                };
                lines.push(format!("  {}. {} [{}{}]", i + 1, fact_text, step.rule, refs));
            }
        }

        Some(format!("Proof ({} steps):\n{}", steps.len(), lines.join("\n")))
    }
}

impl Default for ProofTrace {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Premise identification
// =============================================================================

/// Given a derived fact, the rule that produced it, and the current fact set,
/// reverse-engineer which existing facts the rule consumed as premises.
pub fn identify_premises(
    fact: &Relation,
    rule: &RuleName,
    facts: &HashSet<Relation>,
) -> Vec<Relation> {
    match rule {
        RuleName::Axiom => vec![],

        // --- Single-source rules: one input fact fully determines the derived fact ---

        RuleName::MidpointDefinition => {
            // Midpoint(m,a,b) → Congruent(a,m,m,b) and Collinear(a,m,b)
            match fact {
                Relation::Congruent(a, b, c, d) => {
                    // |AM|=|MB| from Midpoint(M,A,B)
                    // In canonical form: find midpoint fact matching these points
                    let pts = [*a, *b, *c, *d];
                    for &m in &pts {
                        for &x in &pts {
                            for &y in &pts {
                                if x != y && m != x && m != y {
                                    let mid = Relation::midpoint(m, x, y);
                                    if facts.contains(&mid) {
                                        // Verify: midpoint(m,x,y) → cong(x,m,m,y)
                                        let expected = Relation::congruent(x, m, m, y);
                                        if &expected == fact {
                                            return vec![mid];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    vec![]
                }
                Relation::Collinear(a, b, c) => {
                    let pts = [*a, *b, *c];
                    for &m in &pts {
                        for &x in &pts {
                            for &y in &pts {
                                if x < y && m != x && m != y {
                                    let mid = Relation::midpoint(m, x, y);
                                    if facts.contains(&mid) {
                                        return vec![mid];
                                    }
                                }
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::IsoscelesBaseAngles => {
            // Congruent(a,b,c,d) with shared endpoint → EqualAngle
            match fact {
                Relation::EqualAngle(..) => {
                    let fpts = fact.point_ids();
                    // Find a Congruent fact whose points are a subset
                    for f in facts {
                        if let Relation::Congruent(a, b, c, d) = f {
                            // Check if the congruent segments share an endpoint (isosceles)
                            if (a == c || a == d || b == c || b == d)
                                && fpts.iter().all(|p| [*a, *b, *c, *d].contains(p))
                            {
                                return vec![f.clone()];
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::CirclePointEquidistance => {
            // OnCircle(P, C) + OnCircle(Q, C) → Congruent(C,P, C,Q)
            match fact {
                Relation::Congruent(a, b, c, d) => {
                    // center is the shared point: a==c or a==d or b==c or b==d
                    let pairs = [(*a, *b, *c, *d), (*c, *d, *a, *b)];
                    for &(p1, p2, p3, p4) in &pairs {
                        if p1 == p3 {
                            let center = p1;
                            let oc1 = Relation::on_circle(p2, center);
                            let oc2 = Relation::on_circle(p4, center);
                            if facts.contains(&oc1) && facts.contains(&oc2) {
                                return vec![oc1, oc2];
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::CyclicFromOncircle => {
            // 4 OnCircle facts with same circle → Cyclic
            match fact {
                Relation::Cyclic(a, b, c, d) => {
                    // Find common circle
                    for f in facts {
                        if let Relation::OnCircle(p, circ) = f {
                            if *p == *a {
                                let oc_b = Relation::on_circle(*b, *circ);
                                let oc_c = Relation::on_circle(*c, *circ);
                                let oc_d = Relation::on_circle(*d, *circ);
                                if facts.contains(&oc_b)
                                    && facts.contains(&oc_c)
                                    && facts.contains(&oc_d)
                                {
                                    return vec![f.clone(), oc_b, oc_c, oc_d];
                                }
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::MidpointRatio => {
            // Midpoint(m,a,b) → EqualRatio(a,m, a,b, ...)  (1:2 ratio type facts)
            match fact {
                Relation::EqualRatio(..) => {
                    let fpts = fact.point_ids();
                    for f in facts {
                        if let Relation::Midpoint(m, a, b) = f {
                            if fpts.contains(m) && fpts.contains(a) && fpts.contains(b) {
                                return vec![f.clone()];
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        // --- Transitive rules: two facts of same type sharing a component ---

        RuleName::TransitiveParallel => {
            // Para(A,B,C,D) + Para(C,D,E,F) → Para(A,B,E,F)
            match fact {
                Relation::Parallel(a, b, c, d) => {
                    find_transitive_pair(facts, *a, *b, *c, *d, |f| {
                        matches!(f, Relation::Parallel(..))
                    }, |f| {
                        if let Relation::Parallel(x, y, z, w) = f {
                            Some((*x, *y, *z, *w))
                        } else {
                            None
                        }
                    })
                }
                _ => vec![],
            }
        }

        RuleName::TransitiveCongruent => {
            match fact {
                Relation::Congruent(a, b, c, d) => {
                    find_transitive_pair(facts, *a, *b, *c, *d, |f| {
                        matches!(f, Relation::Congruent(..))
                    }, |f| {
                        if let Relation::Congruent(x, y, z, w) = f {
                            Some((*x, *y, *z, *w))
                        } else {
                            None
                        }
                    })
                }
                _ => vec![],
            }
        }

        RuleName::TransitiveEqualAngle => {
            match fact {
                Relation::EqualAngle(a, b, c, d, e, f) => {
                    // Find two EqualAngle facts sharing a triple
                    let _target_t1 = (*a, *b, *c);
                    let _target_t2 = (*d, *e, *f);
                    let eqangles: Vec<_> = facts
                        .iter()
                        .filter_map(|fact| {
                            if let Relation::EqualAngle(x, y, z, u, v, w) = fact {
                                Some(((*x, *y, *z), (*u, *v, *w), fact.clone()))
                            } else {
                                None
                            }
                        })
                        .collect();

                    for i in 0..eqangles.len() {
                        for j in (i + 1)..eqangles.len() {
                            let (t1a, t1b, ref f1) = eqangles[i];
                            let (t2a, t2b, ref f2) = eqangles[j];
                            // Check if they share a triple and the other two are our targets
                            let shared_combos = [
                                (t1a, t1b, t2a, t2b),
                                (t1a, t1b, t2b, t2a),
                                (t1b, t1a, t2a, t2b),
                                (t1b, t1a, t2b, t2a),
                            ];
                            for (s1, o1, s2, o2) in shared_combos {
                                if s1 == s2 {
                                    let result = Relation::equal_angle(
                                        o1.0, o1.1, o1.2, o2.0, o2.1, o2.2,
                                    );
                                    if &result == fact {
                                        return vec![f1.clone(), f2.clone()];
                                    }
                                }
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::TransitiveRatio => {
            // Two EqualRatio facts sharing a ratio side
            match fact {
                Relation::EqualRatio(..) => {
                    let eqratios: Vec<_> = facts
                        .iter()
                        .filter(|f| matches!(f, Relation::EqualRatio(..)))
                        .collect();
                    for i in 0..eqratios.len() {
                        for j in (i + 1)..eqratios.len() {
                            // If their combination could produce our fact, return them
                            if ratio_shares_side(eqratios[i], eqratios[j], fact) {
                                return vec![eqratios[i].clone(), eqratios[j].clone()];
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        // --- Two-perps → parallel ---

        RuleName::PerpToParallel => {
            match fact {
                Relation::Parallel(_a, _b, _c, _d) => {
                    // Find two Perp facts that share a line and whose other lines are a,b and c,d
                    let perps: Vec<_> = facts
                        .iter()
                        .filter_map(|f| {
                            if let Relation::Perpendicular(x, y, z, w) = f {
                                Some((*x, *y, *z, *w))
                            } else {
                                None
                            }
                        })
                        .collect();
                    for i in 0..perps.len() {
                        for j in (i + 1)..perps.len() {
                            let (x1, y1, z1, w1) = perps[i];
                            let (x2, y2, z2, w2) = perps[j];
                            // Check all shared-line combos
                            let combos = [
                                ((x1, y1), (z1, w1), (x2, y2), (z2, w2)),
                                ((x1, y1), (z1, w1), (z2, w2), (x2, y2)),
                                ((z1, w1), (x1, y1), (x2, y2), (z2, w2)),
                                ((z1, w1), (x1, y1), (z2, w2), (x2, y2)),
                            ];
                            for (shared1, other1, shared2, other2) in combos {
                                if lines_equal_u(shared1.0, shared1.1, shared2.0, shared2.1) {
                                    let result = Relation::parallel(
                                        other1.0, other1.1, other2.0, other2.1,
                                    );
                                    if &result == fact {
                                        return vec![
                                            Relation::perpendicular(x1, y1, z1, w1),
                                            Relation::perpendicular(x2, y2, z2, w2),
                                        ];
                                    }
                                }
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        // --- Two-type cross rules ---

        RuleName::AlternateInteriorAngles => {
            // Parallel + Collinear → EqualAngle
            match fact {
                Relation::EqualAngle(a, b, c, d, e, f) => {
                    let pts: HashSet<u16> = [*a, *b, *c, *d, *e, *f].iter().copied().collect();
                    for para in facts.iter().filter(|f| matches!(f, Relation::Parallel(..))) {
                        for coll in facts.iter().filter(|f| matches!(f, Relation::Collinear(..))) {
                            let para_pts = para.point_ids();
                            let coll_pts = coll.point_ids();
                            if para_pts.iter().any(|p| pts.contains(p))
                                && coll_pts.iter().any(|p| pts.contains(p))
                            {
                                // Verify: the parallel and collinear facts involve our angle points
                                return vec![para.clone(), coll.clone()];
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::CorrespondingAngles => {
            // Two Perp facts → EqualAngle
            match fact {
                Relation::EqualAngle(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    let perps: Vec<_> = facts
                        .iter()
                        .filter(|f| matches!(f, Relation::Perpendicular(..)))
                        .collect();
                    for i in 0..perps.len() {
                        for j in (i + 1)..perps.len() {
                            let pts1 = perps[i].point_ids();
                            let pts2 = perps[j].point_ids();
                            let all_perp_pts: HashSet<u16> =
                                pts1.iter().chain(pts2.iter()).copied().collect();
                            if fpts.is_subset(&all_perp_pts) {
                                return vec![perps[i].clone(), perps[j].clone()];
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::PerpendicularAngles => {
            // Perpendicular → EqualAngle (various patterns)
            match fact {
                Relation::EqualAngle(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    for f in facts {
                        if let Relation::Perpendicular(a, b, c, d) = f {
                            let ppts: HashSet<u16> = [*a, *b, *c, *d].iter().copied().collect();
                            if fpts.is_subset(&ppts) {
                                return vec![f.clone()];
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::MidlineParallel => {
            // Two Midpoint facts → Parallel and Congruent
            match fact {
                Relation::Parallel(..) | Relation::Congruent(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    let midpoints: Vec<_> = facts
                        .iter()
                        .filter(|f| matches!(f, Relation::Midpoint(..)))
                        .collect();
                    for i in 0..midpoints.len() {
                        for j in (i + 1)..midpoints.len() {
                            let pts: HashSet<u16> = midpoints[i]
                                .point_ids()
                                .into_iter()
                                .chain(midpoints[j].point_ids())
                                .collect();
                            if fpts.is_subset(&pts) {
                                return vec![midpoints[i].clone(), midpoints[j].clone()];
                            }
                        }
                    }
                    // Single midpoint case
                    for mid in &midpoints {
                        let mpts: HashSet<u16> = mid.point_ids().into_iter().collect();
                        if fpts.is_subset(&mpts) {
                            return vec![(*mid).clone()];
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::EqualAnglesToParallel => {
            // EqualAngle → Parallel
            match fact {
                Relation::Parallel(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    for f in facts {
                        if let Relation::EqualAngle(..) = f {
                            if f.point_ids().iter().any(|p| fpts.contains(p)) {
                                // Also need a collinear for transversal
                                for coll in
                                    facts.iter().filter(|c| matches!(c, Relation::Collinear(..)))
                                {
                                    if coll.point_ids().iter().any(|p| fpts.contains(p)) {
                                        return vec![f.clone(), coll.clone()];
                                    }
                                }
                                return vec![f.clone()];
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::MidpointConverse => {
            // Collinear + Congruent → Midpoint
            match fact {
                Relation::Midpoint(m, a, b) => {
                    let coll = Relation::collinear(*a, *m, *b);
                    let cong = Relation::congruent(*a, *m, *m, *b);
                    let mut premises = vec![];
                    if facts.contains(&coll) {
                        premises.push(coll);
                    }
                    if facts.contains(&cong) {
                        premises.push(cong);
                    }
                    premises
                }
                _ => vec![],
            }
        }

        RuleName::CongruentOncircle => {
            // Congruent + OnCircle → OnCircle
            match fact {
                Relation::OnCircle(p, circ) => {
                    // Find OnCircle(q, circ) and Congruent(circ,q, circ,p) or similar
                    for f in facts {
                        if let Relation::OnCircle(q, c2) = f {
                            if c2 == circ && q != p {
                                let cong = Relation::congruent(*circ, *q, *circ, *p);
                                if facts.contains(&cong) {
                                    return vec![f.clone(), cong];
                                }
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::PerpendicularBisector => {
            // Midpoint + Congruent → Perpendicular
            match fact {
                Relation::Perpendicular(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    for mid in facts.iter().filter(|f| matches!(f, Relation::Midpoint(..))) {
                        if mid.point_ids().iter().any(|p| fpts.contains(p)) {
                            for cong in
                                facts.iter().filter(|f| matches!(f, Relation::Congruent(..)))
                            {
                                if cong.point_ids().iter().any(|p| fpts.contains(p)) {
                                    return vec![mid.clone(), cong.clone()];
                                }
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::EquidistantMidpoint => {
            // Congruent → Midpoint (via equidistance)
            match fact {
                Relation::Midpoint(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    for f in facts {
                        if let Relation::Congruent(..) = f {
                            if f.point_ids().iter().all(|p| fpts.contains(p)) {
                                return vec![f.clone()];
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::PerpParallelTransfer => {
            // Perpendicular + Parallel → Perpendicular
            match fact {
                Relation::Perpendicular(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    for perp in
                        facts.iter().filter(|f| matches!(f, Relation::Perpendicular(..)))
                    {
                        if perp != fact {
                            for para in
                                facts.iter().filter(|f| matches!(f, Relation::Parallel(..)))
                            {
                                let all_pts: HashSet<u16> = perp
                                    .point_ids()
                                    .into_iter()
                                    .chain(para.point_ids())
                                    .collect();
                                if fpts.is_subset(&all_pts) {
                                    return vec![perp.clone(), para.clone()];
                                }
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::LineCollinearExtension | RuleName::CollinearTransitivity => {
            // Collinear + Collinear → Collinear
            match fact {
                Relation::Collinear(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    let colls: Vec<_> = facts
                        .iter()
                        .filter(|f| matches!(f, Relation::Collinear(..)) && *f != fact)
                        .collect();
                    for i in 0..colls.len() {
                        for j in (i + 1)..colls.len() {
                            let pts: HashSet<u16> = colls[i]
                                .point_ids()
                                .into_iter()
                                .chain(colls[j].point_ids())
                                .collect();
                            if fpts.is_subset(&pts)
                                && colls[i]
                                    .point_ids()
                                    .iter()
                                    .any(|p| colls[j].point_ids().contains(p))
                            {
                                return vec![colls[i].clone(), colls[j].clone()];
                            }
                        }
                    }
                    // Also check Parallel + Collinear for line extension
                    for coll in &colls {
                        for para in
                            facts.iter().filter(|f| matches!(f, Relation::Parallel(..)))
                        {
                            let pts: HashSet<u16> = coll
                                .point_ids()
                                .into_iter()
                                .chain(para.point_ids())
                                .collect();
                            if fpts.is_subset(&pts) {
                                return vec![(*coll).clone(), para.clone()];
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::CyclicInscribedAngles => {
            // Cyclic → EqualAngle
            match fact {
                Relation::EqualAngle(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    for f in facts {
                        if let Relation::Cyclic(a, b, c, d) = f {
                            let cpts: HashSet<u16> = [*a, *b, *c, *d].iter().copied().collect();
                            if fpts.is_subset(&cpts) {
                                return vec![f.clone()];
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::ParallelSharedPointCollinear => {
            // Parallel + shared point → Collinear
            match fact {
                Relation::Collinear(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    for f in facts {
                        if let Relation::Parallel(a, b, c, d) = f {
                            let ppts: HashSet<u16> = [*a, *b, *c, *d].iter().copied().collect();
                            if fpts.is_subset(&ppts) {
                                return vec![f.clone()];
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::ThalesTheorem => {
            // Parallel + Collinear → EqualRatio
            match fact {
                Relation::EqualRatio(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    for para in facts.iter().filter(|f| matches!(f, Relation::Parallel(..))) {
                        if para.point_ids().iter().any(|p| fpts.contains(p)) {
                            for coll in
                                facts.iter().filter(|f| matches!(f, Relation::Collinear(..)))
                            {
                                if coll.point_ids().iter().any(|p| fpts.contains(p)) {
                                    return vec![para.clone(), coll.clone()];
                                }
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::InscribedAngleConverse => {
            // EqualAngle → Cyclic
            match fact {
                Relation::Cyclic(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    for f in facts {
                        if let Relation::EqualAngle(..) = f {
                            if f.point_ids().iter().all(|p| fpts.contains(p)) {
                                return vec![f.clone()];
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::IsoscelesConverse => {
            // EqualAngle → Congruent
            match fact {
                Relation::Congruent(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    for f in facts {
                        if let Relation::EqualAngle(..) = f {
                            if f.point_ids().iter().all(|p| fpts.contains(p)) {
                                return vec![f.clone()];
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::PerpMidpointCongruent => {
            // Perpendicular + Midpoint → Congruent
            match fact {
                Relation::Congruent(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    for perp in
                        facts.iter().filter(|f| matches!(f, Relation::Perpendicular(..)))
                    {
                        if perp.point_ids().iter().any(|p| fpts.contains(p)) {
                            for mid in
                                facts.iter().filter(|f| matches!(f, Relation::Midpoint(..)))
                            {
                                if mid.point_ids().iter().any(|p| fpts.contains(p)) {
                                    return vec![perp.clone(), mid.clone()];
                                }
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::TwoEquidistantPerp => {
            // Two Congruent facts → Perpendicular
            match fact {
                Relation::Perpendicular(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    let congs: Vec<_> = facts
                        .iter()
                        .filter(|f| matches!(f, Relation::Congruent(..)))
                        .collect();
                    for i in 0..congs.len() {
                        for j in (i + 1)..congs.len() {
                            let pts: HashSet<u16> = congs[i]
                                .point_ids()
                                .into_iter()
                                .chain(congs[j].point_ids())
                                .collect();
                            if fpts.is_subset(&pts) {
                                return vec![congs[i].clone(), congs[j].clone()];
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::MidpointDiagonalParallelogram => {
            // Multiple Midpoint facts → Parallel or Midpoint
            match fact {
                Relation::Parallel(..) | Relation::Midpoint(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    let midpoints: Vec<_> = facts
                        .iter()
                        .filter(|f| matches!(f, Relation::Midpoint(..)) && *f != fact)
                        .collect();
                    for i in 0..midpoints.len() {
                        for j in (i + 1)..midpoints.len() {
                            let pts: HashSet<u16> = midpoints[i]
                                .point_ids()
                                .into_iter()
                                .chain(midpoints[j].point_ids())
                                .collect();
                            if fpts.is_subset(&pts) {
                                return vec![midpoints[i].clone(), midpoints[j].clone()];
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::CyclicEqualAngleCongruent => {
            // Cyclic + EqualAngle → Congruent
            match fact {
                Relation::Congruent(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    for cyc in facts.iter().filter(|f| matches!(f, Relation::Cyclic(..))) {
                        if cyc.point_ids().iter().any(|p| fpts.contains(p)) {
                            for ea in
                                facts.iter().filter(|f| matches!(f, Relation::EqualAngle(..)))
                            {
                                if ea.point_ids().iter().any(|p| fpts.contains(p)) {
                                    return vec![cyc.clone(), ea.clone()];
                                }
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::CyclicParallelEqangle => {
            // Cyclic + Parallel → EqualAngle
            match fact {
                Relation::EqualAngle(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    for cyc in facts.iter().filter(|f| matches!(f, Relation::Cyclic(..))) {
                        if cyc.point_ids().iter().any(|p| fpts.contains(p)) {
                            for para in
                                facts.iter().filter(|f| matches!(f, Relation::Parallel(..)))
                            {
                                if para.point_ids().iter().any(|p| fpts.contains(p)) {
                                    return vec![cyc.clone(), para.clone()];
                                }
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::EquidistantCyclicPerp => {
            // Congruent + Cyclic → Perpendicular
            match fact {
                Relation::Perpendicular(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    for cong in facts.iter().filter(|f| matches!(f, Relation::Congruent(..))) {
                        if cong.point_ids().iter().any(|p| fpts.contains(p)) {
                            for cyc in
                                facts.iter().filter(|f| matches!(f, Relation::Cyclic(..)))
                            {
                                if cyc.point_ids().iter().any(|p| fpts.contains(p)) {
                                    return vec![cong.clone(), cyc.clone()];
                                }
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::MidpointParallelogram => {
            // Midpoint + Parallelogram(parallel) → Midpoint
            match fact {
                Relation::Midpoint(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    for mid in facts
                        .iter()
                        .filter(|f| matches!(f, Relation::Midpoint(..)) && *f != fact)
                    {
                        if mid.point_ids().iter().any(|p| fpts.contains(p)) {
                            for para in
                                facts.iter().filter(|f| matches!(f, Relation::Parallel(..)))
                            {
                                if para.point_ids().iter().any(|p| fpts.contains(p)) {
                                    return vec![mid.clone(), para.clone()];
                                }
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::EqanglePerpToPerp => {
            // EqualAngle + Perpendicular → Perpendicular
            match fact {
                Relation::Perpendicular(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    for ea in facts.iter().filter(|f| matches!(f, Relation::EqualAngle(..))) {
                        if ea.point_ids().iter().any(|p| fpts.contains(p)) {
                            for perp in facts
                                .iter()
                                .filter(|f| matches!(f, Relation::Perpendicular(..)) && *f != fact)
                            {
                                if perp.point_ids().iter().any(|p| fpts.contains(p)) {
                                    return vec![ea.clone(), perp.clone()];
                                }
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        // --- Complex multi-fact rules ---

        RuleName::SasCongruence => {
            // 2 Congruent + 1 EqualAngle + non-collinear guard
            match fact {
                Relation::Congruent(..) | Relation::EqualAngle(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    let congs: Vec<_> = facts
                        .iter()
                        .filter(|f| matches!(f, Relation::Congruent(..)))
                        .collect();
                    let eqangles: Vec<_> = facts
                        .iter()
                        .filter(|f| matches!(f, Relation::EqualAngle(..)))
                        .collect();
                    for i in 0..congs.len() {
                        for j in (i + 1)..congs.len() {
                            for ea in &eqangles {
                                let all_pts: HashSet<u16> = congs[i]
                                    .point_ids()
                                    .into_iter()
                                    .chain(congs[j].point_ids())
                                    .chain(ea.point_ids())
                                    .collect();
                                if fpts.is_subset(&all_pts) {
                                    return vec![
                                        congs[i].clone(),
                                        congs[j].clone(),
                                        (*ea).clone(),
                                    ];
                                }
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::AsaCongruence => {
            // 1 Congruent + 2 EqualAngle
            match fact {
                Relation::Congruent(..) | Relation::EqualAngle(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    let congs: Vec<_> = facts
                        .iter()
                        .filter(|f| matches!(f, Relation::Congruent(..)))
                        .collect();
                    let eqangles: Vec<_> = facts
                        .iter()
                        .filter(|f| matches!(f, Relation::EqualAngle(..)))
                        .collect();
                    for cong in &congs {
                        for i in 0..eqangles.len() {
                            for j in (i + 1)..eqangles.len() {
                                let all_pts: HashSet<u16> = cong
                                    .point_ids()
                                    .into_iter()
                                    .chain(eqangles[i].point_ids())
                                    .chain(eqangles[j].point_ids())
                                    .collect();
                                if fpts.is_subset(&all_pts) {
                                    return vec![
                                        (*cong).clone(),
                                        eqangles[i].clone(),
                                        eqangles[j].clone(),
                                    ];
                                }
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::SssCongruence => {
            // 3 Congruent facts
            match fact {
                Relation::EqualAngle(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    let congs: Vec<_> = facts
                        .iter()
                        .filter(|f| matches!(f, Relation::Congruent(..)))
                        .collect();
                    for i in 0..congs.len() {
                        for j in (i + 1)..congs.len() {
                            for k in (j + 1)..congs.len() {
                                let all_pts: HashSet<u16> = congs[i]
                                    .point_ids()
                                    .into_iter()
                                    .chain(congs[j].point_ids())
                                    .chain(congs[k].point_ids())
                                    .collect();
                                if fpts.is_subset(&all_pts) {
                                    return vec![
                                        congs[i].clone(),
                                        congs[j].clone(),
                                        congs[k].clone(),
                                    ];
                                }
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::RatioOneCongruence => {
            // EqualRatio with unit segments → Congruent
            match fact {
                Relation::Congruent(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    for f in facts {
                        if let Relation::EqualRatio(..) = f {
                            if f.point_ids().iter().any(|p| fpts.contains(p)) {
                                return vec![f.clone()];
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::ParallelCollinearRatio => {
            // Parallel + Collinear → EqualRatio (Thales variant)
            match fact {
                Relation::EqualRatio(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    for para in facts.iter().filter(|f| matches!(f, Relation::Parallel(..))) {
                        if para.point_ids().iter().any(|p| fpts.contains(p)) {
                            let mut coll_premises = vec![para.clone()];
                            for coll in
                                facts.iter().filter(|f| matches!(f, Relation::Collinear(..)))
                            {
                                if coll.point_ids().iter().any(|p| fpts.contains(p)) {
                                    coll_premises.push(coll.clone());
                                    if coll_premises.len() >= 3 {
                                        return coll_premises;
                                    }
                                }
                            }
                            if coll_premises.len() >= 2 {
                                return coll_premises;
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::CongruentRatio => {
            // Congruent → EqualRatio (with unit ratio)
            match fact {
                Relation::EqualRatio(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    for f in facts {
                        if let Relation::Congruent(..) = f {
                            if f.point_ids().iter().all(|p| fpts.contains(p)) {
                                return vec![f.clone()];
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::RatioCollinearParallel => {
            // EqualRatio + Collinear → Parallel
            match fact {
                Relation::Parallel(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    for ratio in facts.iter().filter(|f| matches!(f, Relation::EqualRatio(..))) {
                        if ratio.point_ids().iter().any(|p| fpts.contains(p)) {
                            for coll in
                                facts.iter().filter(|f| matches!(f, Relation::Collinear(..)))
                            {
                                if coll.point_ids().iter().any(|p| fpts.contains(p)) {
                                    return vec![ratio.clone(), coll.clone()];
                                }
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        // --- Quadrilateral rules ---

        RuleName::ParallelogramOppositeAngles => {
            // Two Parallel facts → EqualAngle
            match fact {
                Relation::EqualAngle(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    let paras: Vec<_> = facts
                        .iter()
                        .filter(|f| matches!(f, Relation::Parallel(..)))
                        .collect();
                    for i in 0..paras.len() {
                        for j in (i + 1)..paras.len() {
                            let pts: HashSet<u16> = paras[i]
                                .point_ids()
                                .into_iter()
                                .chain(paras[j].point_ids())
                                .collect();
                            if fpts.is_subset(&pts) {
                                return vec![paras[i].clone(), paras[j].clone()];
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::IsoscelesTrapezoidBaseAngles => {
            // Parallel + Congruent → EqualAngle
            match fact {
                Relation::EqualAngle(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    for para in facts.iter().filter(|f| matches!(f, Relation::Parallel(..))) {
                        if para.point_ids().iter().any(|p| fpts.contains(p)) {
                            for cong in
                                facts.iter().filter(|f| matches!(f, Relation::Congruent(..)))
                            {
                                if cong.point_ids().iter().any(|p| fpts.contains(p)) {
                                    return vec![para.clone(), cong.clone()];
                                }
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::TrapezoidMidsegment => {
            // Parallel + 2 Midpoint → Parallel + Midpoint + EqualRatio
            match fact {
                Relation::Parallel(..)
                | Relation::Midpoint(..)
                | Relation::EqualRatio(..)
                | Relation::Congruent(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    let mut premises = vec![];
                    for para in facts.iter().filter(|f| matches!(f, Relation::Parallel(..))) {
                        if para.point_ids().iter().any(|p| fpts.contains(p)) {
                            premises.push(para.clone());
                            break;
                        }
                    }
                    for mid in facts.iter().filter(|f| matches!(f, Relation::Midpoint(..))) {
                        if mid.point_ids().iter().any(|p| fpts.contains(p)) {
                            premises.push(mid.clone());
                        }
                    }
                    if premises.len() >= 2 {
                        premises
                    } else {
                        vec![]
                    }
                }
                _ => vec![],
            }
        }

        RuleName::ParallelBaseRatio => {
            // Parallel + Congruent/Collinear → EqualRatio
            match fact {
                Relation::EqualRatio(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    let mut premises = vec![];
                    for para in facts.iter().filter(|f| matches!(f, Relation::Parallel(..))) {
                        if para.point_ids().iter().any(|p| fpts.contains(p)) {
                            premises.push(para.clone());
                            break;
                        }
                    }
                    for coll in facts.iter().filter(|f| matches!(f, Relation::Collinear(..))) {
                        if coll.point_ids().iter().any(|p| fpts.contains(p)) {
                            premises.push(coll.clone());
                            if premises.len() >= 3 {
                                break;
                            }
                        }
                    }
                    if premises.len() >= 2 {
                        premises
                    } else {
                        vec![]
                    }
                }
                _ => vec![],
            }
        }

        RuleName::ParallelProjection => {
            // Multiple Parallel + Collinear facts
            match fact {
                Relation::Parallel(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    let mut premises = vec![];
                    for para in facts
                        .iter()
                        .filter(|f| matches!(f, Relation::Parallel(..)) && *f != fact)
                    {
                        if para.point_ids().iter().any(|p| fpts.contains(p)) {
                            premises.push(para.clone());
                        }
                    }
                    for coll in facts.iter().filter(|f| matches!(f, Relation::Collinear(..))) {
                        if coll.point_ids().iter().any(|p| fpts.contains(p)) {
                            premises.push(coll.clone());
                        }
                    }
                    if premises.len() >= 2 {
                        premises
                    } else {
                        vec![]
                    }
                }
                _ => vec![],
            }
        }

        // --- Tangent rules ---

        RuleName::EqualTangentLengths => {
            // OnCircle + Perpendicular → Congruent
            match fact {
                Relation::Congruent(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    let mut premises = vec![];
                    for oc in facts.iter().filter(|f| matches!(f, Relation::OnCircle(..))) {
                        if oc.point_ids().iter().any(|p| fpts.contains(p)) {
                            premises.push(oc.clone());
                        }
                    }
                    for perp in
                        facts.iter().filter(|f| matches!(f, Relation::Perpendicular(..)))
                    {
                        if perp.point_ids().iter().any(|p| fpts.contains(p)) {
                            premises.push(perp.clone());
                        }
                    }
                    for cong in facts.iter().filter(|ff| {
                        matches!(ff, Relation::Congruent(..)) && ff != &fact
                    }) {
                        if cong.point_ids().iter().any(|p| fpts.contains(p)) {
                            premises.push(cong.clone());
                        }
                    }
                    if premises.len() >= 2 {
                        premises
                    } else {
                        vec![]
                    }
                }
                _ => vec![],
            }
        }

        RuleName::TangentChordAngle => {
            // OnCircle + Perpendicular + Cyclic → EqualAngle
            match fact {
                Relation::EqualAngle(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    let mut premises = vec![];
                    for oc in facts.iter().filter(|f| matches!(f, Relation::OnCircle(..))) {
                        if oc.point_ids().iter().any(|p| fpts.contains(p)) {
                            premises.push(oc.clone());
                        }
                    }
                    for perp in
                        facts.iter().filter(|f| matches!(f, Relation::Perpendicular(..)))
                    {
                        if perp.point_ids().iter().any(|p| fpts.contains(p)) {
                            premises.push(perp.clone());
                        }
                    }
                    if premises.len() >= 2 {
                        premises
                    } else {
                        vec![]
                    }
                }
                _ => vec![],
            }
        }

        // --- Angle bisector rules ---

        RuleName::AngleBisectorRatio => {
            // EqualAngle + Collinear → EqualRatio
            match fact {
                Relation::EqualRatio(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    for ea in facts.iter().filter(|f| matches!(f, Relation::EqualAngle(..))) {
                        if ea.point_ids().iter().any(|p| fpts.contains(p)) {
                            for coll in
                                facts.iter().filter(|f| matches!(f, Relation::Collinear(..)))
                            {
                                if coll.point_ids().iter().any(|p| fpts.contains(p)) {
                                    return vec![ea.clone(), coll.clone()];
                                }
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::IncenterEqualInradii => {
            // Multiple EqualAngle + Perpendicular → Congruent
            match fact {
                Relation::Congruent(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    let mut premises = vec![];
                    for ea in facts.iter().filter(|f| matches!(f, Relation::EqualAngle(..))) {
                        if ea.point_ids().iter().any(|p| fpts.contains(p)) {
                            premises.push(ea.clone());
                        }
                    }
                    for perp in
                        facts.iter().filter(|f| matches!(f, Relation::Perpendicular(..)))
                    {
                        if perp.point_ids().iter().any(|p| fpts.contains(p)) {
                            premises.push(perp.clone());
                        }
                    }
                    if premises.len() >= 2 {
                        premises
                    } else {
                        vec![]
                    }
                }
                _ => vec![],
            }
        }

        // --- Similarity and concurrence ---

        RuleName::AaSimilarity => {
            // 2 EqualAngle → EqualRatio / EqualAngle
            match fact {
                Relation::EqualAngle(..) | Relation::EqualRatio(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    let eqangles: Vec<_> = facts
                        .iter()
                        .filter(|f| matches!(f, Relation::EqualAngle(..)) && *f != fact)
                        .collect();
                    for i in 0..eqangles.len() {
                        for j in (i + 1)..eqangles.len() {
                            let pts: HashSet<u16> = eqangles[i]
                                .point_ids()
                                .into_iter()
                                .chain(eqangles[j].point_ids())
                                .collect();
                            if fpts.is_subset(&pts) {
                                return vec![eqangles[i].clone(), eqangles[j].clone()];
                            }
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }

        RuleName::OrthocenterConcurrence => {
            // 2 Perpendicular + Collinear → Perpendicular / Collinear
            match fact {
                Relation::Perpendicular(..) | Relation::Collinear(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    let mut premises = vec![];
                    for perp in facts
                        .iter()
                        .filter(|f| matches!(f, Relation::Perpendicular(..)) && *f != fact)
                    {
                        if perp.point_ids().iter().any(|p| fpts.contains(p)) {
                            premises.push(perp.clone());
                        }
                    }
                    for coll in facts
                        .iter()
                        .filter(|f| matches!(f, Relation::Collinear(..)) && *f != fact)
                    {
                        if coll.point_ids().iter().any(|p| fpts.contains(p)) {
                            premises.push(coll.clone());
                        }
                    }
                    if premises.len() >= 2 {
                        premises
                    } else {
                        vec![]
                    }
                }
                _ => vec![],
            }
        }

        RuleName::OppositeAnglesCyclic => {
            // EqualAngle (supplementary) → Cyclic
            match fact {
                Relation::Cyclic(..) => {
                    let fpts: HashSet<u16> = fact.point_ids().into_iter().collect();
                    for ea in facts.iter().filter(|f| matches!(f, Relation::EqualAngle(..))) {
                        if ea.point_ids().iter().all(|p| fpts.contains(p)) {
                            return vec![ea.clone()];
                        }
                    }
                    // Fallback: look for partial overlap
                    for ea in facts.iter().filter(|f| matches!(f, Relation::EqualAngle(..))) {
                        if ea.point_ids().iter().any(|p| fpts.contains(p)) {
                            return vec![ea.clone()];
                        }
                    }
                    vec![]
                }
                _ => vec![],
            }
        }
    }
}

// =============================================================================
// Helper functions
// =============================================================================

/// Check if two lines (as point pairs) are the same line (order-independent).
fn lines_equal_u(a1: u16, b1: u16, a2: u16, b2: u16) -> bool {
    (a1 == a2 && b1 == b2) || (a1 == b2 && b1 == a2)
}

/// Check if two segments are equal (order-independent).
fn segments_equal_u(a1: u16, b1: u16, a2: u16, b2: u16) -> bool {
    let s1 = if a1 <= b1 { (a1, b1) } else { (b1, a1) };
    let s2 = if a2 <= b2 { (a2, b2) } else { (b2, a2) };
    s1 == s2
}

/// Find two facts of the same relation type that share a component pair
/// and whose other components match the target fact's components.
fn find_transitive_pair<F, G>(
    facts: &HashSet<Relation>,
    a: u16,
    b: u16,
    c: u16,
    d: u16,
    filter: F,
    extract: G,
) -> Vec<Relation>
where
    F: Fn(&Relation) -> bool,
    G: Fn(&Relation) -> Option<(u16, u16, u16, u16)>,
{
    let matching: Vec<_> = facts.iter().filter(|f| filter(f)).collect();
    for i in 0..matching.len() {
        for j in (i + 1)..matching.len() {
            if let (Some((x1, y1, z1, w1)), Some((x2, y2, z2, w2))) =
                (extract(matching[i]), extract(matching[j]))
            {
                // Check all ways they could share a component and produce (a,b,c,d)
                let combos = [
                    ((x1, y1), (z1, w1), (x2, y2), (z2, w2)),
                    ((x1, y1), (z1, w1), (z2, w2), (x2, y2)),
                    ((z1, w1), (x1, y1), (x2, y2), (z2, w2)),
                    ((z1, w1), (x1, y1), (z2, w2), (x2, y2)),
                ];
                for (shared1, other1, shared2, other2) in combos {
                    if segments_equal_u(shared1.0, shared1.1, shared2.0, shared2.1) {
                        // other1 and other2 should produce (a,b) and (c,d)
                        if (segments_equal_u(other1.0, other1.1, a, b)
                            && segments_equal_u(other2.0, other2.1, c, d))
                            || (segments_equal_u(other1.0, other1.1, c, d)
                                && segments_equal_u(other2.0, other2.1, a, b))
                        {
                            return vec![matching[i].clone(), matching[j].clone()];
                        }
                    }
                }
            }
        }
    }
    vec![]
}

/// Check if two EqualRatio facts share a ratio side and could produce the target.
fn ratio_shares_side(r1: &Relation, r2: &Relation, target: &Relation) -> bool {
    if let (
        Relation::EqualRatio(a1, b1, c1, d1, e1, f1, g1, h1),
        Relation::EqualRatio(a2, b2, c2, d2, e2, f2, g2, h2),
        Relation::EqualRatio(at, bt, ct, dt, et, ft, gt, ht),
    ) = (r1, r2, target)
    {
        let r1_left = (*a1, *b1, *c1, *d1);
        let r1_right = (*e1, *f1, *g1, *h1);
        let r2_left = (*a2, *b2, *c2, *d2);
        let r2_right = (*e2, *f2, *g2, *h2);
        let t_left = (*at, *bt, *ct, *dt);
        let t_right = (*et, *ft, *gt, *ht);

        let sides = [
            (r1_left, r1_right, r2_left, r2_right),
            (r1_left, r1_right, r2_right, r2_left),
            (r1_right, r1_left, r2_left, r2_right),
            (r1_right, r1_left, r2_right, r2_left),
        ];

        for (s1, o1, s2, o2) in sides {
            if s1 == s2 && ((o1 == t_left && o2 == t_right) || (o1 == t_right && o2 == t_left)) {
                return true;
            }
        }
    }
    false
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof_state::{ObjectType, ProofState, Relation};

    #[test]
    fn test_add_axiom() {
        let mut trace = ProofTrace::new();
        let fact = Relation::parallel(0, 1, 2, 3);
        trace.add_axiom(fact.clone());
        assert!(trace.is_axiom(&fact));
        assert_eq!(trace.axiom_count(), 1);
        assert_eq!(trace.len(), 1);
    }

    #[test]
    fn test_add_derivation() {
        let mut trace = ProofTrace::new();
        let premise = Relation::parallel(0, 1, 2, 3);
        let derived = Relation::parallel(0, 1, 4, 5);
        trace.add_axiom(premise.clone());
        trace.add_derivation(
            derived.clone(),
            RuleName::TransitiveParallel,
            vec![premise.clone()],
        );
        assert!(!trace.is_axiom(&derived));
        assert_eq!(trace.len(), 2);
        let d = trace.get(&derived).unwrap();
        assert_eq!(d.rule, RuleName::TransitiveParallel);
        assert_eq!(d.premises.len(), 1);
    }

    #[test]
    fn test_no_overwrite() {
        let mut trace = ProofTrace::new();
        let fact = Relation::congruent(0, 1, 2, 3);
        trace.add_derivation(
            fact.clone(),
            RuleName::TransitiveCongruent,
            vec![Relation::congruent(0, 1, 4, 5)],
        );
        // Second derivation should NOT overwrite
        trace.add_derivation(
            fact.clone(),
            RuleName::MidpointDefinition,
            vec![Relation::midpoint(0, 1, 2)],
        );
        let d = trace.get(&fact).unwrap();
        assert_eq!(d.rule, RuleName::TransitiveCongruent);
    }

    #[test]
    fn test_extract_linear_proof() {
        let mut trace = ProofTrace::new();
        let a = Relation::parallel(0, 1, 2, 3);
        let b = Relation::parallel(0, 1, 4, 5);
        let c = Relation::parallel(0, 1, 6, 7);
        trace.add_axiom(a.clone());
        trace.add_derivation(b.clone(), RuleName::TransitiveParallel, vec![a.clone()]);
        trace.add_derivation(c.clone(), RuleName::TransitiveParallel, vec![b.clone()]);

        let proof = trace.extract_proof(&c).unwrap();
        assert_eq!(proof.len(), 3); // axiom a, derivation b, derivation c
        // First should be the axiom
        assert_eq!(proof[0].rule, RuleName::Axiom);
        // Last should be the goal
        assert_eq!(proof.last().unwrap().fact, c);
    }

    #[test]
    fn test_extract_diamond_proof() {
        // A (axiom) → B and A → C, then B+C → D
        let mut trace = ProofTrace::new();
        let a = Relation::congruent(0, 1, 2, 3);
        let b = Relation::congruent(0, 1, 4, 5);
        let c = Relation::congruent(0, 1, 6, 7);
        let d = Relation::congruent(4, 5, 6, 7);

        trace.add_axiom(a.clone());
        trace.add_derivation(
            b.clone(),
            RuleName::TransitiveCongruent,
            vec![a.clone()],
        );
        trace.add_derivation(
            c.clone(),
            RuleName::TransitiveCongruent,
            vec![a.clone()],
        );
        trace.add_derivation(
            d.clone(),
            RuleName::TransitiveCongruent,
            vec![b.clone(), c.clone()],
        );

        let proof = trace.extract_proof(&d).unwrap();
        assert_eq!(proof.len(), 4); // a, b, c, d
        assert_eq!(proof.last().unwrap().fact, d);
    }

    #[test]
    fn test_extract_prunes_irrelevant() {
        let mut trace = ProofTrace::new();
        let a = Relation::parallel(0, 1, 2, 3);
        let b = Relation::parallel(0, 1, 4, 5);
        let irrelevant = Relation::congruent(10, 11, 12, 13);

        trace.add_axiom(a.clone());
        trace.add_axiom(irrelevant.clone());
        trace.add_derivation(b.clone(), RuleName::TransitiveParallel, vec![a.clone()]);

        let proof = trace.extract_proof(&b).unwrap();
        // Should only contain a and b, not the irrelevant fact
        assert_eq!(proof.len(), 2);
        assert!(proof.iter().all(|d| d.fact != irrelevant));
    }

    #[test]
    fn test_format_proof() {
        let mut state = ProofState::new();
        let a = state.add_object("a", ObjectType::Point);
        let b = state.add_object("b", ObjectType::Point);
        let c = state.add_object("c", ObjectType::Point);
        let d = state.add_object("d", ObjectType::Point);
        let e = state.add_object("e", ObjectType::Point);
        let f = state.add_object("f", ObjectType::Point);

        let mut trace = ProofTrace::new();
        let fact1 = Relation::parallel(a, b, c, d);
        let fact2 = Relation::parallel(c, d, e, f);
        let goal = Relation::parallel(a, b, e, f);

        trace.add_axiom(fact1.clone());
        trace.add_axiom(fact2.clone());
        trace.add_derivation(
            goal.clone(),
            RuleName::TransitiveParallel,
            vec![fact1.clone(), fact2.clone()],
        );

        let formatted = trace.format_proof(&goal, &state).unwrap();
        assert!(formatted.contains("Proof"));
        assert!(formatted.contains("axiom"));
        assert!(formatted.contains("TransitiveParallel"));
        assert!(!formatted.is_empty());
    }

    #[test]
    fn test_extract_nonexistent_goal() {
        let trace = ProofTrace::new();
        let goal = Relation::parallel(0, 1, 2, 3);
        assert!(trace.extract_proof(&goal).is_none());
    }

    // --- Premise identification tests ---

    #[test]
    fn test_premises_transitive_parallel() {
        let mut facts = HashSet::new();
        let p1 = Relation::parallel(0, 1, 2, 3);
        let p2 = Relation::parallel(2, 3, 4, 5);
        facts.insert(p1.clone());
        facts.insert(p2.clone());

        let derived = Relation::parallel(0, 1, 4, 5);
        let premises = identify_premises(&derived, &RuleName::TransitiveParallel, &facts);
        assert_eq!(premises.len(), 2);
        assert!(premises.contains(&p1));
        assert!(premises.contains(&p2));
    }

    #[test]
    fn test_premises_midpoint_definition() {
        let mut facts = HashSet::new();
        let mid = Relation::midpoint(2, 0, 1); // M=2 is midpoint of A=0, B=1
        facts.insert(mid.clone());

        // Midpoint → Congruent(A,M, M,B) = cong(0,2, 2,1)
        let cong = Relation::congruent(0, 2, 2, 1);
        let premises = identify_premises(&cong, &RuleName::MidpointDefinition, &facts);
        assert_eq!(premises.len(), 1);
        assert_eq!(premises[0], mid);

        // Midpoint → Collinear(A,M,B)
        let coll = Relation::collinear(0, 2, 1);
        let premises = identify_premises(&coll, &RuleName::MidpointDefinition, &facts);
        assert_eq!(premises.len(), 1);
        assert_eq!(premises[0], mid);
    }

    #[test]
    fn test_premises_alternate_interior_angles() {
        let mut facts = HashSet::new();
        let para = Relation::parallel(0, 1, 2, 3);
        let coll = Relation::collinear(0, 4, 2);
        facts.insert(para.clone());
        facts.insert(coll.clone());

        let eqangle = Relation::equal_angle(1, 0, 2, 3, 2, 0);
        let premises = identify_premises(&eqangle, &RuleName::AlternateInteriorAngles, &facts);
        assert!(premises.len() >= 2);
    }

    #[test]
    fn test_premises_sas() {
        let mut facts = HashSet::new();
        let c1 = Relation::congruent(0, 1, 3, 4);
        let c2 = Relation::congruent(0, 2, 3, 5);
        let ea = Relation::equal_angle(1, 0, 2, 4, 3, 5);
        facts.insert(c1.clone());
        facts.insert(c2.clone());
        facts.insert(ea.clone());

        let derived = Relation::congruent(1, 2, 4, 5);
        let premises = identify_premises(&derived, &RuleName::SasCongruence, &facts);
        assert!(premises.len() >= 3);
    }

    #[test]
    fn test_premises_isosceles_base_angles() {
        let mut facts = HashSet::new();
        let cong = Relation::congruent(0, 1, 0, 2); // |AB| = |AC|, isosceles at A
        facts.insert(cong.clone());

        let ea = Relation::equal_angle(0, 1, 2, 0, 2, 1);
        let premises = identify_premises(&ea, &RuleName::IsoscelesBaseAngles, &facts);
        assert_eq!(premises.len(), 1);
        assert_eq!(premises[0], cong);
    }

    #[test]
    fn test_premises_circle_equidistance() {
        let mut facts = HashSet::new();
        let oc1 = Relation::on_circle(1, 0); // point 1 on circle 0
        let oc2 = Relation::on_circle(2, 0); // point 2 on circle 0
        facts.insert(oc1.clone());
        facts.insert(oc2.clone());

        let cong = Relation::congruent(0, 1, 0, 2);
        let premises = identify_premises(&cong, &RuleName::CirclePointEquidistance, &facts);
        assert_eq!(premises.len(), 2);
    }

    #[test]
    fn test_premises_perp_to_parallel() {
        let mut facts = HashSet::new();
        let perp1 = Relation::perpendicular(0, 1, 2, 3);
        let perp2 = Relation::perpendicular(4, 5, 2, 3);
        facts.insert(perp1.clone());
        facts.insert(perp2.clone());

        let derived = Relation::parallel(0, 1, 4, 5);
        let premises = identify_premises(&derived, &RuleName::PerpToParallel, &facts);
        assert_eq!(premises.len(), 2);
    }
}
