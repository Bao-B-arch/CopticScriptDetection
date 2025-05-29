from typing import Union
import numpy as np
from numpy.typing import NDArray

NDArrayBool = NDArray[np.bool_]
NDArrayInt = NDArray[np.int64]
NDArrayUInt = NDArray[np.uint8]
NDArrayFloat = NDArray[np.float64]
NDArrayNum = Union[NDArrayInt, NDArrayFloat]
NDArrayStr = NDArray[np.str_]
