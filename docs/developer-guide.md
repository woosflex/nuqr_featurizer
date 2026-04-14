# NuQR Featurizer Developer Guide

This guide describes how to work on the Rust/PyO3 codebase and ship wheels reliably.

Dependency model:
- Required runtime dependency: `numpy`
- Optional file-I/O API dependency set: `pillow` + `scipy`

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
python -m pip install -r requirements.txt
python -m pip install --force-reinstall dist/nuqr_featurizer-*.whl
```

When running from the source tree (`PYTHONPATH=python`), keep
`python/nuqr_featurizer/_core.abi3.so` synchronized with Rust source changes.
If this binary is stale, parity checks can regress even when Python wrappers are correct.

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

`extract_features_from_files` is the convenience path:

- Inputs: `image_path`, `mat_path`, optional `mat_key`, optional `use_gpu`
- Behavior: loads files via `python/nuqr_featurizer/io.py`, validates shape compatibility, then delegates to `extract_features`
- Dependency note: requires optional Python dependencies (`Pillow`, `scipy`) via `pip install "nuqr-featurizer[io]"`

Comparison script API mode:

- `scripts/compare_with_python_features.py --extractor-api direct` uses the direct array API.
- `scripts/compare_with_python_features.py --extractor-api files` exercises the file-based convenience API.

## Packaging and CI

- Packaging is configured in `pyproject.toml` using maturin.
- CI workflow: `.github/workflows/CI.yml`
  - Runs Rust integration tests.
  - Builds wheel and runs Python smoke import checks.
  - Builds matrix wheels/sdist and uploads assets to GitHub Releases on tags.
- The Rust dependency graph for wheels is kept free of `native-tls`/`openssl-sys`
  to avoid OpenSSL toolchain issues in manylinux and musllinux builds.

### PyPI release trigger

- Tag pushes upload wheel/sdist artifacts to the corresponding GitHub Release.
- PyPI publish is performed only through `workflow_dispatch` with:
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
