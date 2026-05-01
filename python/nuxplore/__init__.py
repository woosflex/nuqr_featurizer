from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ._core import (
    __version__,
    check_gpu,
    extract_features as _extract_features,
    extract_features_from_files,
    get_gpu_device_count,
    normalize_staining as _normalize_staining,
    save_cropped_nuclei_from_files,
)
from .batch import BatchExtractor, batch_extract_and_crop, batch_extract_features


def _iter_crop_masks(masks: Any) -> list[tuple[int, Any]]:
    import numpy as np

    mask_array = np.asarray(masks)
    if mask_array.ndim == 2 and mask_array.dtype == np.bool_:
        return [(1, mask_array)]
    if mask_array.ndim == 2:
        return [
            (int(label), mask_array == label)
            for label in sorted(np.unique(mask_array))
            if int(label) != 0
        ]
    if mask_array.ndim == 3:
        return [
            (index, mask_array[index - 1].astype(bool))
            for index in range(1, mask_array.shape[0] + 1)
        ]
    return [(index, np.asarray(mask, dtype=bool)) for index, mask in enumerate(masks, start=1)]


def extract_features(
    image: Any,
    masks: Any,
    use_gpu: Optional[bool] = None,
    *,
    save_crops: bool = False,
    crop_output_dir: Optional[str | Path] = None,
    padding: int = 10,
    save_pre_normalized_crops: bool = True,
    save_post_normalized_crops: bool = True,
) -> list[dict[str, float]]:
    features = _extract_features(image, masks, use_gpu=use_gpu)
    if not save_crops:
        return features
    if crop_output_dir is None:
        raise ValueError("crop_output_dir is required when save_crops=True")

    import numpy as np

    from .batch import crop_masked_patch, save_rgb_patch

    image_array = np.asarray(image, dtype=np.uint8)
    output_dir = Path(crop_output_dir)
    normalized_image = image_array
    if save_post_normalized_crops:
        try:
            normalized_image = np.asarray(_normalize_staining(image_array), dtype=np.uint8)
            if normalized_image.shape != image_array.shape:
                normalized_image = image_array
        except Exception:
            normalized_image = image_array

    for label, nucleus_mask in _iter_crop_masks(masks):
        if save_pre_normalized_crops:
            pre_patch = crop_masked_patch(image_array, nucleus_mask, padding)
            if pre_patch is not None:
                save_rgb_patch(
                    output_dir / "pre_normalized_nuclei" / f"nucleus_{label:04d}.png",
                    pre_patch,
                )
        if save_post_normalized_crops:
            post_patch = crop_masked_patch(normalized_image, nucleus_mask, padding)
            if post_patch is not None:
                save_rgb_patch(
                    output_dir / "post_normalized_nuclei" / f"nucleus_{label:04d}.png",
                    post_patch,
                )
    return features


__all__ = [
    "__version__",
    "extract_features",
    "extract_features_from_files",
    "save_cropped_nuclei_from_files",
    "BatchExtractor",
    "batch_extract_features",
    "batch_extract_and_crop",
    "check_gpu",
    "get_gpu_device_count",
]
