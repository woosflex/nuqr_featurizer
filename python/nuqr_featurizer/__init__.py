from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Optional, Union

from ._core import __version__, check_gpu, extract_features, get_gpu_device_count

PathInput = Union[str, PathLike[str], Path]


def extract_features_from_files(
    image_path: PathInput,
    mat_path: PathInput,
    *,
    mat_key: Optional[str] = None,
    use_gpu: Optional[bool] = None,
) -> list[dict[str, float]]:
    from .io import load_instance_map, load_rgb_image

    image = load_rgb_image(image_path)
    instance_map, _ = load_instance_map(mat_path, preferred_key=mat_key)
    if instance_map.shape != image.shape[:2]:
        raise ValueError(
            f"image/mat shape mismatch: image={image.shape[:2]} mat={instance_map.shape}"
        )
    return extract_features(image, instance_map, use_gpu=use_gpu)


__all__ = [
    "__version__",
    "extract_features",
    "extract_features_from_files",
    "check_gpu",
    "get_gpu_device_count",
]
