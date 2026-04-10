from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

__version__: str

RGBImage = npt.NDArray[np.uint8]
InstanceMap = npt.NDArray[np.uint32]
MaskArray = npt.NDArray[np.bool_]
FeatureMap = Dict[str, float]

def check_gpu() -> bool: ...
def get_gpu_device_count() -> int: ...
def extract_features(
    image: RGBImage,
    masks: Union[InstanceMap, Sequence[MaskArray]],
    use_gpu: Optional[bool] = ...,
) -> List[FeatureMap]: ...
