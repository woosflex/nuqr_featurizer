# nuqr-featurizer

`nuqr-featurizer` is a Rust + PyO3 package for high-throughput nucleus feature extraction with optional GPU acceleration through WGPU.

## Installation

### From wheel (recommended)

```bash
pip install nuqr-featurizer
```

### Local development build

```bash
maturin build --release --out dist --interpreter python
python -m pip install --force-reinstall --no-deps dist/nuqr_featurizer-*.whl
```

## Quick start

```python
import numpy as np
import nuqr_featurizer as nf

# H x W x 3 uint8 RGB image
image = np.zeros((64, 64, 3), dtype=np.uint8)

# Preferred mask format: instance map (0=background, 1..N=nuclei)
instance_map = np.zeros((64, 64), dtype=np.uint32)
instance_map[16:28, 20:32] = 1
instance_map[36:48, 34:46] = 2

features = nf.extract_features(image, instance_map, use_gpu=False)
print(len(features))       # number of nuclei
print(features[0].keys())  # feature names
```

## Python API

- `check_gpu() -> bool`: returns whether a compatible WGPU adapter is available.
- `get_gpu_device_count() -> int`: returns `1` when a compatible adapter is available, else `0`.
- `extract_features(image, masks, use_gpu=None) -> list[dict[str, float]]`:
  - `image`: `np.ndarray[uint8]` with shape `(H, W, 3)`
  - `masks`: either:
    - `np.ndarray[uint32]` instance map with shape `(H, W)`, or
    - sequence of `np.ndarray[bool]` masks each with shape `(H, W)`

Current Python-level extraction path returns morphology-oriented features per nucleus (for example: `area`, `perimeter`, `centroid_row`, `centroid_col`).

## GPU behavior

- The package uses **WGPU** for acceleration (not CUDA-specific runtime bindings).
- If `use_gpu=True` is requested but no compatible adapter is available, the call raises an error.
- Internal feature kernels use CPU fallback paths where implemented.

## CUDA wheel status

- No separate CUDA wheel variant is currently produced.
- Cross-platform wheels include the same WGPU-backed runtime and can use compatible GPU adapters at runtime.

## Example notebook

- `examples/quickstart.ipynb` demonstrates an end-to-end Python usage flow.

## Type hints

The wheel includes `py.typed` and `.pyi` stubs:

- `nuqr_featurizer/__init__.pyi`
- `nuqr_featurizer/_core.pyi`

For deeper architecture and contributor workflows, see `docs/developer-guide.md`.
