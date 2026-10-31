"""Map ESA WorldCover codes to the project's five classes."""
import numpy as np
from . import config

_LUT = np.full(256, config.IGNORE_INDEX, dtype=np.uint8)
for _code, _cls in config.WORLDCOVER_TO_CLASS.items():
    _LUT[_code] = _cls


def remap_worldcover(arr):
    """Unknown/nodata codes become IGNORE_INDEX (255)."""
    return _LUT[np.asarray(arr).astype(np.uint8)]
