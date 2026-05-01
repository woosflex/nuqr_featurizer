from os import PathLike as OsPathLike
from typing import Dict, List, Optional, Sequence, Tuple, TypedDict, Union

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

def check_gpu() -> bool: ...
def get_gpu_device_count() -> int: ...
def normalize_staining(image: RGBImage) -> RGBImage: ...
def extract_features(
    image: RGBImage,
    masks: Union[InstanceMap, Sequence[MaskArray]],
    use_gpu: Optional[bool] = ...,
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
