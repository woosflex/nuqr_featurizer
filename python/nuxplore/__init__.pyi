from os import PathLike as OsPathLike
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

__version__: str

RGBImage = npt.NDArray[np.uint8]
InstanceMap = npt.NDArray[np.uint32]
MaskArray = npt.NDArray[np.bool_]
FeatureMap = Dict[str, float]
PathInput = Union[str, OsPathLike[str]]

def check_gpu() -> bool: ...
def get_gpu_device_count() -> int: ...
def extract_features(
    image: RGBImage,
    masks: Union[InstanceMap, Sequence[MaskArray]],
    use_gpu: Optional[bool] = ...,
) -> List[FeatureMap]: ...
def extract_features_from_files(
    image_path: PathInput,
    mat_path: PathInput,
    *,
    mat_key: Optional[str] = ...,
    use_gpu: Optional[bool] = ...,
) -> List[FeatureMap]: ...
