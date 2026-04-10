//! Stain normalization algorithms.

pub mod vahadane;

pub use vahadane::{
    normalize_staining_default, snmf_update_loop, VahadaneStainNormalizer,
    REFERENCE_MAX_CONCENTRATIONS_V, REFERENCE_STAIN_MATRIX_V,
};
