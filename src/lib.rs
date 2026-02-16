#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;

pub mod proof_state;
pub mod deduction;
pub mod construction;
pub mod parser;
pub mod encoding;
pub mod mcts;
pub mod synthetic;

// --- PyO3 wrapper types ---

/// Python-visible wrapper around ProofState.
/// Uses `unsendable` because ProofState contains StdRng which is !Send.
#[pyclass(unsendable)]
#[derive(Clone)]
struct PyProofState {
    inner: proof_state::ProofState,
}

#[pymethods]
impl PyProofState {
    /// Number of geometric objects in the state.
    fn num_objects(&self) -> usize {
        self.inner.objects.len()
    }

    /// Number of known facts (relations) in the state.
    fn num_facts(&self) -> usize {
        self.inner.facts.len()
    }

    /// Whether the goal has been proved.
    fn is_proved(&self) -> bool {
        self.inner.is_proved()
    }

    /// List of object names in ID order.
    fn object_names(&self) -> Vec<String> {
        self.inner.objects.iter().map(|o| o.name.clone()).collect()
    }

    /// Human-readable description of the goal, or None if no goal set.
    fn goal_description(&self) -> Option<String> {
        self.inner.goal.as_ref().map(|g| format!("{:?}", g))
    }

    fn __repr__(&self) -> String {
        format!(
            "PyProofState(objects={}, facts={}, proved={})",
            self.inner.objects.len(),
            self.inner.facts.len(),
            self.inner.is_proved()
        )
    }
}

/// Python-visible wrapper around Construction.
#[pyclass(unsendable)]
#[derive(Clone)]
struct PyConstruction {
    inner: construction::Construction,
}

#[pymethods]
impl PyConstruction {
    /// The type of construction (e.g. "Midpoint", "Altitude").
    fn construction_type(&self) -> String {
        format!("{:?}", self.inner.ctype)
    }

    /// The argument object IDs for this construction.
    fn args(&self) -> Vec<u16> {
        self.inner.args.clone()
    }

    /// Priority category: "GoalRelevant", "RecentlyActive", or "Exploratory".
    fn priority(&self) -> String {
        format!("{:?}", self.inner.priority)
    }

    fn __repr__(&self) -> String {
        format!(
            "PyConstruction({:?}, args={:?}, priority={:?})",
            self.inner.ctype, self.inner.args, self.inner.priority
        )
    }
}

// --- PyO3 exposed functions ---

/// Parse a JGEX DSL problem string into a PyProofState.
#[pyfunction]
fn parse_problem(problem_text: &str) -> PyResult<PyProofState> {
    let state = parser::parse_problem(problem_text)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyProofState { inner: state })
}

/// Run deduction to fixed point with default config. Returns true if goal is proved.
#[pyfunction]
fn saturate(state: &mut PyProofState) -> PyResult<bool> {
    Ok(deduction::saturate(&mut state.inner))
}

/// Run deduction with custom config. Returns true if goal is proved.
#[pyfunction]
fn saturate_with_config(
    state: &mut PyProofState,
    max_iterations: usize,
    max_facts: usize,
) -> PyResult<bool> {
    let config = deduction::SaturateConfig {
        max_iterations,
        max_facts,
        ..deduction::SaturateConfig::default()
    };
    Ok(deduction::saturate_with_config(&mut state.inner, &config))
}

/// Encode a ProofState as a flat f32 tensor of shape (20, 32, 32).
#[pyfunction]
fn encode_state(state: &PyProofState) -> PyResult<Vec<f32>> {
    Ok(encoding::state_to_tensor(&state.inner))
}

/// Generate candidate auxiliary constructions for the current state.
#[pyfunction]
fn generate_constructions(state: &PyProofState) -> PyResult<Vec<PyConstruction>> {
    let constructions = construction::generate_constructions(&state.inner);
    Ok(constructions
        .into_iter()
        .map(|c| PyConstruction { inner: c })
        .collect())
}

/// Apply a construction to a state, returning a new state with the construction applied.
#[pyfunction]
fn apply_construction(
    state: &PyProofState,
    construction: &PyConstruction,
) -> PyResult<PyProofState> {
    let new_state = construction::apply_construction(&state.inner, &construction.inner);
    Ok(PyProofState { inner: new_state })
}

/// Compute delta_D: fraction of goal sub-conditions present in the state.
#[pyfunction]
fn compute_delta_d(state: &PyProofState) -> PyResult<f64> {
    Ok(mcts::compute_delta_d(&state.inner))
}

/// Serialize a ProofState as compact text (relations + goal).
#[pyfunction]
fn state_to_text(state: &PyProofState) -> PyResult<String> {
    Ok(state.inner.to_text())
}

/// Serialize a construction as compact text using object names from the state.
#[pyfunction]
fn construction_to_text(construction: &PyConstruction, state: &PyProofState) -> PyResult<String> {
    Ok(construction.inner.to_text(&state.inner))
}

/// Generate synthetic training data: Vec of (state_text, construction_text, goal_text).
#[pyfunction]
fn generate_synthetic_data(num_examples: usize, seed: u64) -> PyResult<Vec<(String, String, String)>> {
    Ok(synthetic::generate_batch(num_examples, seed))
}

// --- Module registration ---

#[pymodule]
fn geoprover(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    m.add_class::<PyProofState>()?;
    m.add_class::<PyConstruction>()?;
    m.add_function(wrap_pyfunction!(parse_problem, m)?)?;
    m.add_function(wrap_pyfunction!(saturate, m)?)?;
    m.add_function(wrap_pyfunction!(saturate_with_config, m)?)?;
    m.add_function(wrap_pyfunction!(encode_state, m)?)?;
    m.add_function(wrap_pyfunction!(generate_constructions, m)?)?;
    m.add_function(wrap_pyfunction!(apply_construction, m)?)?;
    m.add_function(wrap_pyfunction!(compute_delta_d, m)?)?;
    m.add_function(wrap_pyfunction!(state_to_text, m)?)?;
    m.add_function(wrap_pyfunction!(construction_to_text, m)?)?;
    m.add_function(wrap_pyfunction!(generate_synthetic_data, m)?)?;
    Ok(())
}
