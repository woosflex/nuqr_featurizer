from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np

DEFAULT_MAT_KEYS: tuple[str, ...] = (
    "instance_map",
    "inst_map",
    "instances",
    "labels",
    "segmentation",
    "mask",
)

PathInput = Union[str, PathLike[str], Path]


def _as_path(path: PathInput, argument_name: str) -> Path:
    try:
        return Path(path).expanduser()
    except TypeError as exc:
        raise TypeError(f"{argument_name} must be a path-like value") from exc


def _get_pillow_image():
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            'Pillow is required for source-tree image loading helpers.'
        ) from exc
    return Image


def _get_scipy_loadmat():
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError(
            'scipy is required for source-tree .mat loading helpers.'
        ) from exc
    return loadmat


def load_rgb_image(path: PathInput) -> np.ndarray:
    image_path = _as_path(path, "image_path")
    image = _get_pillow_image()
    with image.open(image_path) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image at {image_path}, got shape {arr.shape}")
    return arr


def _coerce_instance_map(arr: np.ndarray) -> Optional[np.ndarray]:
    if not isinstance(arr, np.ndarray) or arr.ndim != 2:
        return None
    if not np.issubdtype(arr.dtype, np.number) and arr.dtype != np.bool_:
        return None

    out = np.asarray(arr)
    if out.dtype == np.bool_:
        out = out.astype(np.uint32)
    elif np.issubdtype(out.dtype, np.floating):
        if not np.all(np.isfinite(out)):
            return None
        rounded = np.rint(out)
        if not np.allclose(out, rounded, atol=1e-6, rtol=0.0):
            return None
        out = rounded.astype(np.int64)
    else:
        out = out.astype(np.int64)

    if out.size == 0:
        return None
    if int(out.min()) < 0:
        return None

    return out.astype(np.uint32)


def load_instance_map(
    mat_path: PathInput,
    preferred_key: Optional[str] = None,
    *,
    default_keys: Sequence[str] = DEFAULT_MAT_KEYS,
) -> Tuple[np.ndarray, str]:
    mat_file = _as_path(mat_path, "mat_path")
    data = _get_scipy_loadmat()(mat_file)

    if preferred_key:
        if preferred_key not in data:
            raise KeyError(f"Key '{preferred_key}' not found in {mat_file}")
        coerced = _coerce_instance_map(data[preferred_key])
        if coerced is None:
            raise ValueError(f"Key '{preferred_key}' is not a valid 2D instance map.")
        return coerced, preferred_key

    for key in default_keys:
        if key in data:
            coerced = _coerce_instance_map(data[key])
            if coerced is not None:
                return coerced, key

    best_key: Optional[str] = None
    best_map: Optional[np.ndarray] = None
    best_score = -1
    for key, value in data.items():
        if key.startswith("__"):
            continue
        coerced = _coerce_instance_map(value)
        if coerced is None:
            continue
        unique_count = int(np.unique(coerced).size)
        if unique_count > best_score:
            best_score = unique_count
            best_key = key
            best_map = coerced

    if best_map is None or best_key is None:
        raise ValueError(f"No valid 2D instance map found in {mat_file}")
    return best_map, best_key


__all__ = ["DEFAULT_MAT_KEYS", "load_rgb_image", "load_instance_map"]
