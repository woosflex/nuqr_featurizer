from os import PathLike as OsPathLike
from typing import Dict, List, Literal, Optional, Sequence, Tuple, TypedDict, Union

import numpy as np
import numpy.typing as npt

__version__: str

RGBImage = npt.NDArray[np.uint8]
InstanceMap = npt.NDArray[np.uint32]
MaskArray = npt.NDArray[np.bool_]
FeatureMap = Dict[str, float]
PathInput = Union[str, OsPathLike[str]]

class CropSaveRecord(TypedDict):
    nucleus_id: int
    bbox: Tuple[int, int, int, int]
    pre_path: Optional[str]
    post_path: Optional[str]

class ImageResult:
    ok: bool
    image_path: str
    mat_path: Optional[str]
    mat_key: Optional[str]
    nuclei: int
    warnings: tuple[str, ...]
    error: Optional[str]
    traceback: Optional[str]

class BatchResult:
    tasks_discovered: int
    completed_images: int
    failed_images: int
    total_nuclei: int
    warnings: List[str]
    results: List[ImageResult]

class BatchExtractor:
    def __init__(
        self,
        image_root: PathInput,
        mat_root: PathInput,
        output_csv_root: PathInput,
        output_nuclei_root: PathInput,
        *,
        image_exts: Sequence[str] = ...,
        recursive: bool = ...,
        workers: Optional[int] = ...,
        max_images: Optional[int] = ...,
        skip_existing: bool = ...,
        mat_key: Optional[str] = ...,
        inst_type_key: str = ...,
        padding: int = ...,
        use_gpu: bool = ...,
        stain_normalization_features: bool = ...,
        metadata_csv: Optional[PathInput] = ...,
        metadata_key_column: str = ...,
        metadata_cols: Sequence[str] = ...,
        metadata_id_source: Literal["first_dir", "parent_dir", "stem"] = ...,
    ) -> None: ...
    def extract(
        self,
        *,
        save_crops: bool = ...,
        save_pre_normalized_crops: bool = ...,
        save_post_normalized_crops: bool = ...,
    ) -> BatchResult: ...
    def extract_features(
        self,
        *,
        save_crops: bool = ...,
        save_pre_normalized_crops: bool = ...,
        save_post_normalized_crops: bool = ...,
    ) -> BatchResult: ...
    def extract_and_crop(
        self,
        *,
        save_pre_normalized_crops: bool = ...,
        save_post_normalized_crops: bool = ...,
    ) -> BatchResult: ...

def check_gpu() -> bool: ...
def get_gpu_device_count() -> int: ...
def extract_features(
    image: RGBImage,
    masks: Union[InstanceMap, Sequence[MaskArray]],
    use_gpu: Optional[bool] = ...,
    *,
    save_crops: bool = ...,
    crop_output_dir: Optional[PathInput] = ...,
    padding: int = ...,
    save_pre_normalized_crops: bool = ...,
    save_post_normalized_crops: bool = ...,
) -> List[FeatureMap]: ...
def extract_features_from_files(
    image_path: PathInput,
    mat_path: PathInput,
    *,
    mat_key: Optional[str] = ...,
    use_gpu: Optional[bool] = ...,
    save_crops: bool = ...,
    crop_output_dir: Optional[PathInput] = ...,
    padding: int = ...,
    save_pre_normalized_crops: bool = ...,
    save_post_normalized_crops: bool = ...,
) -> List[FeatureMap]: ...
def save_cropped_nuclei_from_files(
    image_path: PathInput,
    mat_path: PathInput,
    output_dir: PathInput,
    *,
    mat_key: Optional[str] = ...,
    padding: int = ...,
    save_pre_normalized: bool = ...,
    save_post_normalized: bool = ...,
) -> List[CropSaveRecord]: ...
def batch_extract_features(
    image_root: PathInput,
    mat_root: PathInput,
    output_csv_root: PathInput,
    output_nuclei_root: PathInput,
    *,
    save_crops: bool = ...,
    save_pre_normalized_crops: bool = ...,
    save_post_normalized_crops: bool = ...,
    **kwargs: object,
) -> BatchResult: ...
def batch_extract_and_crop(
    image_root: PathInput,
    mat_root: PathInput,
    output_csv_root: PathInput,
    output_nuclei_root: PathInput,
    *,
    image_exts: Sequence[str] = ...,
    recursive: bool = ...,
    workers: Optional[int] = ...,
    max_images: Optional[int] = ...,
    skip_existing: bool = ...,
    mat_key: Optional[str] = ...,
    inst_type_key: str = ...,
    padding: int = ...,
    use_gpu: bool = ...,
    stain_normalization_features: bool = ...,
    metadata_csv: Optional[PathInput] = ...,
    metadata_key_column: str = ...,
    metadata_cols: Sequence[str] = ...,
    metadata_id_source: Literal["first_dir", "parent_dir", "stem"] = ...,
    save_crops: bool = ...,
    save_pre_normalized_crops: bool = ...,
    save_post_normalized_crops: bool = ...,
) -> BatchResult: ...
