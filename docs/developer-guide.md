# NuXplore Developer Guide

This guide describes how to work on the Rust/PyO3 codebase and ship wheels reliably.

Dependency model:
- Required Python runtime dependencies: none for the file API.
- Optional array API dependency: `numpy` through `pip install "nuxplore[array]"`.
- Development and validation dependencies: `requirements.txt`.

## Architecture summary

- `src/lib.rs`: PyO3 module entrypoint and Python API wiring.
- `src/features/`: CPU and optional WGPU feature implementations.
- `src/gpu/`: WGPU backend + shader pipelines.
- `src/io/`: Rust image and MATLAB v5 readers used by the dependency-free file API.
- `src/stain_norm/`: Vahadane stain normalization.
- `python/nuxplore/`: Python package wrapper, stubs, and typed marker.

## Local development setup

1. Install Rust toolchain and Python.
2. Install maturin.
3. Build and install a local wheel:

```bash
python -m pip install maturin
maturin build --release --out dist --interpreter python
python -m pip install -r requirements.txt
python -m pip install --force-reinstall --no-deps dist/nuxplore-*.whl
```

When running from the source tree (`PYTHONPATH=python`), keep
`python/nuxplore/_core.abi3.so` synchronized with Rust source changes.
If this binary is stale, parity checks can regress even when Python wrappers are correct.

If you use the project micromamba environment:

```bash
eval "$(micromamba shell hook -s bash)"
micromamba activate nuxplore_test
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
- optional crop export:
  - `save_crops=True`
  - `crop_output_dir=<path>`
  - `save_pre_normalized_crops` and `save_post_normalized_crops` control which crop sets are written

Dimension mismatches return structured errors from `FeaturizerError`.

`extract_features_from_files` is the convenience path:

- Inputs: `image_path`, `mat_path`, optional `mat_key`, optional `use_gpu`
- Behavior: loads RGB image and MAT instance map in Rust through `src/io/`, validates shape compatibility, then runs the same extraction pipeline as the array API.
- Dependency note: does not require Python `numpy`, `pillow`, or `scipy` at runtime.
- Optional crop export:
  - `save_crops=True`
  - `crop_output_dir=<path>`
  - `save_pre_normalized_crops` and `save_post_normalized_crops` control which crop sets are written.

`normalize_staining` is exposed by the compiled `_core` module. The extraction pipeline enables Vahadane post-normalization when `NUQR_ENABLE_STAIN_NORMALIZATION` is set to `1`, `true`, `yes`, or `on`.

`save_cropped_nuclei_from_files` is a low-level crop export path:

- Inputs: `image_path`, `mat_path`, `output_dir`, optional `mat_key`, optional `padding`, and pre/post save flags.
- Behavior: saves masked padded nucleus PNG crops under:
  - `<output_dir>/pre_normalized_nuclei/nucleus_0001.png`
  - `<output_dir>/post_normalized_nuclei/nucleus_0001.png`
- Returns: per-nucleus records (`nucleus_id`, `bbox`, `pre_path`, `post_path`).

`nuxplore.batch.batch_extract_and_crop` is the orchestration path:

- Inputs: image/MAT roots, output roots, metadata options, worker count, GPU flag, and stain-normalization toggle.
- Behavior:
  - scans paired image/MAT files
  - computes features via `extract_features(image, instance_map, ...)`
  - writes one CSV per image
  - writes pre/post normalized crop PNGs per nucleus
- `BatchExtractor.extract_features(...)` uses the same batch pipeline but keeps crop saving opt-in through `save_crops=True`.
- `BatchExtractor.extract_and_crop(...)` is the convenience wrapper for crop-saving batch runs.
- `inst_type` compatibility:
  - best-effort loading using SciPy when available
  - if unavailable/missing, non-fatal warning(s) are returned and `nucleus_type` defaults to `Unknown`.

CLI compatibility:

- `scripts/batch_extract_and_crop.py` is intentionally a thin wrapper calling `nuxplore.batch.main()`.
- Keep reusable behavior in `python/nuxplore/batch.py`; avoid re-adding orchestration logic to the script.

Comparison script API mode:

- `scripts/compare_with_python_features.py --extractor-api direct` uses the direct array API.
- `scripts/compare_with_python_features.py --extractor-api files` exercises the file-based convenience API.

Batch/crop milestone checks:

```bash
# Milestone 1
cargo test crop_export -- --nocapture
cargo test test_full_pipeline_end_to_end_cpu

# Milestone 2/3
python -m py_compile python/nuxplore/batch.py scripts/batch_extract_and_crop.py
PYTHONPATH=python python scripts/batch_extract_and_crop.py --help
```

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
- TestPyPI publish is performed through `workflow_dispatch` with:
  - `publish_to_testpypi = true`
  - Uses repository secret: `TEST_PYPI_API_TOKEN`
- PyPI publish is performed through `workflow_dispatch` with:
  - `publish_to_pypi = true`
  - Uses repository secret: `PYPI_API_TOKEN`

TestPyPI install command for collaborators:

```bash
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
  nuxplore
```

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
- Type stubs: `python/nuxplore/__init__.pyi` and `_core.pyi`
- Typed marker: `python/nuxplore/py.typed`
