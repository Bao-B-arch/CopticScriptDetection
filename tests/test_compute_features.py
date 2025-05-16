

from typing import Dict, Tuple
import numpy as np
import pytest

from common.compute_features import patches_slices_broadcast, from_slices_to_masks, patches_features
from common.types import NDArrayBool, NDArrayFloat, NDArrayInt, NDArrayStr


def create_single_pixel_masks(shape: int) -> NDArrayBool:
    """
    Crée un array 3D de masques booléens où chaque masque 
    a exactement un pixel True à une position unique
    
    Args:
        shape: Dimensions (shape, shape) des masques
    
    Returns:
        Array 3D de forme (N, shape, shape) où N = shape * shape
    """
    # Création des indices linéaires
    k = np.arange(shape * shape)

    # Génération des masques
    masks = np.zeros((shape*shape, shape, shape), dtype=bool)
    masks[k, k // shape, k % shape] = True
    
    return masks


@pytest.mark.parametrize(
        ("nb_patch", "image_size_sqrt", "results"), 
        [
            (9, 3, np.array([[0,1,0,1], [0,1,1,2], [0,1,2,3], [1,2,0,1], [1,2,1,2], [1,2,2,3], [2,3,0,1], [2,3,1,2], [2,3,2,3]])),
            (4, 3, np.array([[0,2,0,2], [0,2,1,3], [1,3,0,2], [1,3,1,3]])),
            (1, 3, np.array([[0,3,0,3]])),
            (1, 28, np.array([[0,28,0,28]])),
            (784, 28, np.array([[k//28, k//28+1, k%28, k%28+1] for k in np.arange(784)], dtype=np.int64)),
            (8000, 28, np.array([[k//28, k//28+1, k%28, k%28+1] for k in np.arange(784)], dtype=np.int64))
        ])
def test_patches_slices(
    nb_patch: int,
    image_size_sqrt: int,
    results: NDArrayBool,
) -> None:

    slices = patches_slices_broadcast(nb_patch_sqrt=int(np.sqrt(nb_patch)), image_size_sqrt=image_size_sqrt)
    assert slices.shape == results.shape
    np.testing.assert_array_equal(slices, results)


@pytest.mark.parametrize(
        ("image_size_sqrt", "slices", "results"), 
        [
            (3, np.array([[0,1,0,1], [1,2,0,1], [1,2,2,3]]), np.array([
                    [[True, False, False], [False, False, False], [False, False, False]],
                    [[False, False, False], [True, False, False], [False, False, False]],
                    [[False, False, False], [False, False, True], [False, False, False]],
                ])),
            (3, np.array([[0,2,0,2], [0,2,1,3], [1,3,0,2], [1,3,1,3]]), np.array([
                    [[True, True, False], [True, True, False], [False, False, False]],
                    [[False, True, True], [False, True, True], [False, False, False]],
                    [[False, False, False], [True, True, False], [True, True, False]],
                    [[False, False, False], [False, True, True], [False, True, True]]
                ])
            )
        ])
def test_patches_masks(
    image_size_sqrt: int,
    slices: NDArrayInt,
    results: NDArrayBool,
) -> None:

    masks = from_slices_to_masks(slices=slices, image_size_sqrt=image_size_sqrt)
    assert masks.shape == results.shape
    np.testing.assert_array_equal(masks, results)


@pytest.mark.parametrize(
        ("database", "data_size", "shape", "image_size_sqrt", "results"), 
        [
            (
                {"alpha": np.array([[[1, 1, 3], [1, 1, 3], [-1, -1, 1]], [[0, 0, 3], [0, 0, 1], [-3, -1, 2]]])}, 
                2, 4, 3,
                (
                    np.array([f"mean_{i}" for i in range(4)] + [f"var_{i}" for i in range(4)] + ["Letter"]),
                    np.array(["alpha", "alpha"], dtype="<U10"),
                    np.array([[1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 1.0, 2.0], [0.0, 1.0, -1.0, 0.5, 0.0, 1.5, 1.5, 1.25]], dtype=np.float64),

                ),
            )
        ])
def test_patches_features(
    database: Dict[str, NDArrayInt],
    data_size: int,
    shape: int,
    image_size_sqrt: int,
    results: Tuple[NDArrayStr, NDArrayStr, NDArrayFloat]
) -> None:

    cols, letters, data = patches_features(database, data_size=data_size, shape=shape, target_name="Letter", image_size_sqrt=image_size_sqrt)
    cols_expected, letters_expected, data_expected = results
    np.testing.assert_array_equal(cols, cols_expected)
    np.testing.assert_array_equal(letters, letters_expected)
    np.testing.assert_array_equal(data, data_expected)
