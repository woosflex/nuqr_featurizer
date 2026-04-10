//! Logging setup with Python integration

use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Initialize logging for the library
pub fn init_logging() {
    // Only initialize once
    static INIT: std::sync::Once = std::sync::Once::new();

    INIT.call_once(|| {
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

        tracing_subscriber::registry()
            .with(filter)
            .with(tracing_subscriber::fmt::layer())
            .init();
    });
}

/// Log a span for feature computation
#[macro_export]
macro_rules! feature_span {
    ($name:expr) => {
        tracing::info_span!("feature", name = $name)
    };
    ($name:expr, $nucleus_id:expr) => {
        tracing::info_span!("feature", name = $name, nucleus_id = $nucleus_id)
    };
}
