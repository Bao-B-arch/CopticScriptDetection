

from typing import List, Tuple, TypeVar, Union
import numpy as np
import pytest

from common.compute_features import patches_mask
from common.types import NDArrayBool, NDArrayNum

image = np.array([[1, 1, 3], [1, 1, 3], [-1, -1, 1]])

T = TypeVar("T", bound=Union[NDArrayNum, NDArrayBool])
def np_put(arr: T, i: Tuple[int, int], v: Union[int, bool, float]) -> T:
    arr[i] = v
    return arr


@pytest.mark.parametrize(
        ("nb_patch", "image_size_sqrt", "results"), 
        [
            (9, 3, [
                np_put(np.zeros((3, 3)).astype(np.bool), (i//3, i%3), True) for i in range(9)
            ]),
            (4, 3, [
                np.array([[True, True, False], [True, True, False], [False, False, False]]),
                np.array([[False, True, True], [False, True, True], [False, False, False]]),
                np.array([[False, False, False], [True, True, False], [True, True, False]]),
                np.array([[False, False, False], [False, True, True], [False, True, True]]),
            ]),
            (1, 3, [
                np.array([[True, True, True], [True, True, True], [True, True, True]]),
            ]),
            (1, 28, [
                np.ones((28, 28), dtype=np.bool),
            ]),
            (784, 28, [
                np_put(np.zeros((28, 28)).astype(np.bool), (i//28, i%28), True) for i in range(784)
            ]),
            (80000, 28, [
                np_put(np.zeros((28, 28)).astype(np.bool), (i//28, i%28), True) for i in range(784)
            ])
        ])
def test_patches_mask(
    nb_patch: int,
    image_size_sqrt: int,
    results: List[NDArrayBool]
) -> None:

    masks = patches_mask(nb_patch_sqrt=int(np.sqrt(nb_patch)), image_size_sqrt=image_size_sqrt)

    assert len(masks) == len(results)
    for m, r in zip(masks, results):
        np.testing.assert_array_equal(m, r)
