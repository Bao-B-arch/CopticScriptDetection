from typing import Union
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectorMixin

NDArrayBool = NDArray[np.bool_]
NDArrayInt = NDArray[np.int_]
NDArrayFloat = NDArray[np.floating]
NDArrayNum = Union[NDArrayInt, NDArrayFloat]
NDArrayStr = NDArray[np.str_]


class Selector(SelectorMixin, BaseEstimator):
    pass


class Transformer(TransformerMixin, BaseEstimator):
    pass
