
from typing import Dict, Optional, Tuple

from attr import define
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

from common.types import NDArrayNum, NDArrayStr


@define(auto_attribs=True, kw_only=True)
class LoadingData:
    classes: NDArrayStr
    current_X: NDArrayNum
    current_y: NDArrayStr
    data_size: int = 0
    transformers: Dict[str, BaseEstimator] = {}


@define(auto_attribs=True, kw_only=True)
class SplitData:
    X_train: NDArrayNum
    y_train: NDArrayStr
    X_test: NDArrayNum
    y_test: NDArrayStr
    models: Dict[str, Tuple[Optional[GridSearchCV], BaseEstimator]] = {}
