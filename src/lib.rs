use pyo3::prelude::*;

pub mod proof_state;
pub mod deduction;
pub mod construction;
pub mod parser;
pub mod encoding;

#[pymodule]
fn geoprover(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    Ok(())
}
