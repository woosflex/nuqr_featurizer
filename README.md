# NuXplore

`NuXplore` is a Rust + PyO3 Python package for high-throughput histopathology nucleus feature extraction.

It provides two primary input paths:

- `extract_features_from_files(...)`: preferred path for image + `.mat` files. Image and MAT loading happen in Rust and do not require Python `numpy`, `pillow`, or `scipy`.
- `extract_features(...)`: NumPy array path for in-memory pipelines. Install the optional `array` extra when using this API from a fresh environment.

Additional orchestration APIs:

- `save_cropped_nuclei_from_files(...)`: saves masked nucleus crops (`pre_normalized_nuclei` and `post_normalized_nuclei`) for one image/MAT pair.
- `extract_features(..., save_crops=True, crop_output_dir=...)`: saves masked crops while running single-image feature extraction on in-memory arrays.
- `batch_extract_features(..., save_crops=True, ...)`: runs paired image/MAT batch extraction and can save crop outputs alongside CSVs.
- `batch_extract_and_crop(...)`: runs paired image/MAT batch extraction, writes per-image CSVs, and writes cropped nucleus PNGs by default.

The package uses WGPU for optional cross-platform GPU acceleration. No separate CUDA wheel is currently produced.

## Installation

### Runtime wheel

```bash
python -m pip install nuxplore
```

This is enough for dependency-free file input:

```python
import nuxplore as nf

features = nf.extract_features_from_files(
    "/path/to/tile.png",
    "/path/to/tile.mat",
    mat_key=None,
    use_gpu=False,
)
```

### Runtime wheel with NumPy array API

```bash
python -m pip install "nuxplore[array]"
```

Use this when calling `extract_features(image, masks, ...)` with NumPy arrays.

### TestPyPI pre-release

```bash
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
  nuxplore
```

With the array API extra:

```bash
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
  "nuxplore[array]"
```

### Local wheel

```bash
python -m pip install /path/to/nuxplore-0.1.0-*.whl
```

Add development dependencies only when running scripts, tests, notebooks, or parity comparisons:

```bash
python -m pip install -r requirements.txt
```

### Local development build

From this repository:

```bash
python -m pip install maturin
maturin build --release --out dist --interpreter python
python -m pip install --force-reinstall --no-deps dist/nuxplore-*.whl
```

If you import from the source tree with `PYTHONPATH=python`, rebuild the extension after Rust changes so `python/nuxplore/_core.abi3.so` is current. A stale local extension can cause parity regressions unrelated to source code.

Project workflow note: use the `nuxplore_test` micromamba environment for local build and validation when available.

```bash
eval "$(micromamba shell hook -s bash)"
micromamba activate nuxplore_test
```

## Quick Start

### File API

```python
import nuxplore as nf

features = nf.extract_features_from_files(
    "/path/to/tile.png",
    "/path/to/tile.mat",
    mat_key=None,      # auto-detects a suitable 2D instance map when omitted
    use_gpu=False,
)

print(len(features))
print(features[0].keys())
```

The `.mat` input should contain a 2D instance map where `0` is background and positive integer IDs identify nuclei.

### Array API

```python
import numpy as np
import nuxplore as nf

image = np.zeros((64, 64, 3), dtype=np.uint8)
instance_map = np.zeros((64, 64), dtype=np.uint32)
instance_map[16:28, 20:32] = 1
instance_map[36:48, 34:46] = 2

features = nf.extract_features(image, instance_map, use_gpu=False)
```

`extract_features(...)` also accepts a sequence of `(H, W)` boolean masks.

## Python API

- `check_gpu() -> bool`: returns whether a compatible WGPU adapter is available.
- `get_gpu_device_count() -> int`: returns `1` when a compatible adapter is available, else `0`.
- `extract_features(image, masks, use_gpu=None) -> list[dict[str, float]]`:
  - `image`: `np.ndarray[uint8]` with shape `(H, W, 3)`.
  - `masks`: either a `np.ndarray[uint32]` instance map with shape `(H, W)` or a sequence of `np.ndarray[bool]` masks.
  - returns one feature dictionary per nucleus.
  - optional crop saving: set `save_crops=True` and `crop_output_dir=...` to write `pre_normalized_nuclei/` and `post_normalized_nuclei/` PNGs.
- `extract_features_from_files(image_path, mat_path, mat_key=None, use_gpu=None) -> list[dict[str, float]]`:
  - loads RGB image and instance map in Rust.
  - expands `~/` paths.
  - auto-detects a suitable MAT variable when `mat_key` is omitted.
- `save_cropped_nuclei_from_files(image_path, mat_path, output_dir, mat_key=None, padding=10, save_pre_normalized=True, save_post_normalized=True) -> list[dict]`:
  - saves one PNG per nucleus using masked, padded crops.
  - writes under `pre_normalized_nuclei/` and `post_normalized_nuclei/`.
  - returns per-nucleus records with `nucleus_id`, `bbox`, and output paths.
- `batch_extract_features(image_root, mat_root, output_csv_root, output_nuclei_root, ..., save_crops=False, ...) -> BatchResult`:
  - reusable batch entry point for paired image/MAT extraction.
  - set `save_crops=True` to write cropped nuclei alongside the CSV outputs.
- `BatchExtractor.extract_features(..., save_crops=False, ...) -> BatchResult`:
  - performs batch extraction through the reusable class API.
  - set `save_crops=True` to write crop PNGs alongside the CSV outputs.
- `BatchExtractor.extract_and_crop(..., save_pre_normalized_crops=True, save_post_normalized_crops=True) -> BatchResult`:
  - convenience wrapper for batch extraction with crop saving enabled.
- `batch_extract_and_crop(image_root, mat_root, output_csv_root, output_nuclei_root, ...) -> BatchResult`:
  - scans paired image/MAT files and writes one CSV per image.
  - writes cropped nucleus PNGs for each processed image by default.
  - supports metadata enrichment through `metadata_csv`, `metadata_key_column`, `metadata_cols`, and `metadata_id_source`.
  - surfaces non-fatal warnings when `inst_type` is unavailable/missing and uses `nucleus_type="Unknown"` fallback.

The built extension also exposes `normalize_staining(image)` from `nuxplore._core`; the top-level Python wrapper currently exports the extraction and GPU helpers listed above.

Legacy script path:

- `scripts/batch_extract_and_crop.py` is now a thin CLI wrapper that delegates to `nuxplore.batch.main()`.

## Feature Coverage

Each nucleus feature dictionary includes:

- `nucleus_id`
- morphology and centroid features
- Hu moments
- advanced shape features
- NEIS features
- nearest-neighbor spatial distance
- `pre_norm_` intensity, GLCM, LBP, H&E color, HOG, and CCSM features
- `post_norm_` intensity, GLCM, LBP, H&E color, HOG, and CCSM features

By default, extraction keeps the normalized image equal to the input image. Set `NUQR_ENABLE_STAIN_NORMALIZATION=1` to enable Vahadane stain normalization for `post_norm_` feature groups.

## GPU Behavior

- GPU acceleration uses WGPU, not CUDA-specific runtime bindings.
- `use_gpu=False` keeps execution on CPU.
- `use_gpu=True` requires a compatible WGPU adapter and raises an error if none is available.
- `check_gpu()` and `get_gpu_device_count()` report adapter availability.
- Standard wheels are GPU-capable through WGPU on supported systems; there is no separate CUDA wheel variant.

## Validation

The package is exercised against the Python reference pipeline and includes crop-saving and batch orchestration coverage in the test suite.

For contributor validation commands, parity checks, and batch/crop behavior notes, see [`docs/developer-guide.md`](docs/developer-guide.md).

## Type Hints

The wheel includes `py.typed` and `.pyi` stubs:

- `nuxplore/__init__.pyi`
- `nuxplore/_core.pyi`

## Developer Documentation

For architecture, local quality commands, benchmarks, and contributor workflow, see [`docs/developer-guide.md`](docs/developer-guide.md).
