# NuQR Featurizer Developer Guide

This guide describes how to work on the Rust/PyO3 codebase and ship wheels reliably.

## Architecture summary

- `src/lib.rs`: PyO3 module entrypoint and Python API wiring.
- `src/features/`: CPU and optional WGPU feature implementations.
- `src/gpu/`: WGPU backend + shader pipelines.
- `src/stain_norm/`: Vahadane stain normalization.
- `python/nuqr_featurizer/`: Python package wrapper, stubs, and typed marker.

## Local development setup

1. Install Rust toolchain and Python.
2. Install maturin.
3. Build and install a local wheel:

```bash
maturin build --release --out dist --interpreter python
python -m pip install --force-reinstall --no-deps dist/nuqr_featurizer-*.whl
```

If you use the project micromamba environment:

```bash
eval "$(micromamba shell hook -s bash)"
micromamba activate nuqr_library
```

## Core quality commands

```bash
cargo fmt --all
cargo test --tests
cargo rustdoc --lib -- -W missing-docs -D rustdoc::invalid_html_tags
```

## Profiling and benchmarking

- CPU profile runner: `src/bin/cpu_profile.rs`
- Numerical parity runner: `src/bin/numerical_validation.rs`
- Memory profiling runner: `src/bin/memory_profile.rs`
- Criterion benchmarks: `benches/benchmark_suite.rs`

Useful commands:

```bash
cargo run --release --bin numerical_validation
cargo run --release --bin cpu_profile
cargo bench --bench benchmark_suite
```

## Python API contracts

`extract_features` expects:

- `image`: `(H, W, 3)` `uint8`
- `masks`: either `(H, W)` `uint32` instance map or a list of `(H, W)` bool masks

Dimension mismatches return structured errors from `FeaturizerError`.

## Packaging and CI

- Packaging is configured in `pyproject.toml` using maturin.
- CI workflow: `.github/workflows/CI.yml`
  - Runs Rust integration tests.
  - Builds wheel and runs Python smoke import checks.
  - Builds matrix wheels/sdist and publishes on tagged releases.
- The transitive OpenBLAS downloader path is patched to use `rustls` TLS in
  `vendor/openblas-build` (wired via `[patch.crates-io]`), avoiding
  `native-tls`/OpenSSL dependency failures in manylinux and musllinux builds.

### PyPI release trigger

- Tag-based release publish is enabled in CI.
- Manual publish is also available through `workflow_dispatch` with:
  - `publish_to_pypi = true`
- Publishing uses `UV_PUBLISH_TOKEN` from repository secrets.

## GPU notes

- This project uses WGPU for cross-platform acceleration.
- Legacy CUDA-specific paths are not part of the current runtime interface.
- Keep CPU fallback behavior correct for environments without compatible adapters.

### CUDA wheel policy

- The project does not currently ship a separate CUDA-only wheel variant.
- Standard wheels are GPU-capable through WGPU on supported adapters.

## Documentation and typing

- User-facing package docs: `README.md`
- User example notebook: `examples/quickstart.ipynb`
- Type stubs: `python/nuqr_featurizer/__init__.pyi` and `_core.pyi`
- Typed marker: `python/nuqr_featurizer/py.typed`
