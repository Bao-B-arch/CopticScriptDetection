from typing import Union
import numpy as np
from numpy.typing import NDArray

NDArrayBool = NDArray[np.bool_]
NDArrayInt = NDArray[np.int_]
NDArrayFloat = NDArray[np.floating]
NDArrayNum = Union[NDArrayInt, NDArrayFloat]
NDArrayStr = NDArray[np.str_]