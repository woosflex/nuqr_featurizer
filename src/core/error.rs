//! Error types for the NuXplore library

use thiserror::Error;

/// Error type for all fallible NuXplore operations.
#[derive(Error, Debug)]
pub enum FeaturizerError {
    /// Input values violated semantic preconditions.
    ///
    /// Examples include too-few samples for a model fit or invalid parameter
    /// ranges.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// A mask contained no foreground pixels.
    #[error("Empty mask: nucleus has no pixels")]
    EmptyMask,

    /// A provided array shape did not match the expected dimensions.
    #[error("Invalid image dimensions: expected {expected}, got {got}")]
    InvalidDimensions {
        /// Human-readable expected shape.
        expected: String,
        /// Human-readable actual shape.
        got: String,
    },

    /// A numerical instability occurred (NaN/Inf, singular behavior, etc.).
    #[error("Numerical error: {0}")]
    NumericalError(String),

    /// GPU backend or compute-dispatch error.
    ///
    /// The historical variant name is retained for compatibility.
    #[error("CUDA error: {0}")]
    CudaError(String),

    /// Conversion between Python and Rust types failed.
    #[error("Python conversion error: {0}")]
    PythonConversionError(String),

    /// I/O failure while reading/writing filesystem resources.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Linear-algebra backend failure.
    #[error("Linear algebra error: {0}")]
    LinalgError(String),

    /// A specific feature calculation failed.
    #[error("Feature computation failed: {feature} - {reason}")]
    FeatureComputationFailed {
        /// Feature family or identifier that failed.
        feature: String,
        /// Root cause details.
        reason: String,
    },
}

/// Result type alias for the library
pub type Result<T> = std::result::Result<T, FeaturizerError>;

impl From<FeaturizerError> for pyo3::PyErr {
    fn from(err: FeaturizerError) -> pyo3::PyErr {
        use pyo3::exceptions::*;
        match err {
            FeaturizerError::InvalidInput(_) => PyValueError::new_err(err.to_string()),
            FeaturizerError::EmptyMask => PyValueError::new_err(err.to_string()),
            FeaturizerError::InvalidDimensions { .. } => PyValueError::new_err(err.to_string()),
            FeaturizerError::PythonConversionError(_) => PyTypeError::new_err(err.to_string()),
            _ => PyRuntimeError::new_err(err.to_string()),
        }
    }
}
