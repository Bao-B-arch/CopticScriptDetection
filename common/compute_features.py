import os
from pathlib import Path
from typing import Dict, Tuple
import cv2
import numpy as np

from common.config import IMAGE_SIZE
from common.types import NDArrayBool, NDArrayFloat, NDArrayInt, NDArrayNum, NDArrayStr, NDArrayUInt


def patches_slices_broadcast(nb_patch_sqrt: int, image_size_sqrt: int) -> NDArrayInt:
    ## no overlap
    if nb_patch_sqrt >= image_size_sqrt:
        start_end = np.arange(start=0, stop=image_size_sqrt + 1, dtype=np.int8)
        inc = 1
        nb_patch_sqrt = image_size_sqrt
    else: ## overlap
        start_end = np.linspace(start=0, stop=image_size_sqrt, num = 2 + nb_patch_sqrt, dtype=np.int8)
        inc = 2

    patch_slices: NDArrayInt = np.zeros((nb_patch_sqrt*nb_patch_sqrt, 4), dtype=np.int64)

    for i in range(nb_patch_sqrt):
        row_start: int = start_end[i]
        row_end: int = start_end[i + inc]
        
        for j in range(nb_patch_sqrt):
            col_start: int = start_end[j]
            col_end: int = start_end[j + inc]

            # Créer un masque pour ce patch
            patch_slices[i*nb_patch_sqrt + j, :] = [row_start, row_end, col_start, col_end]
    return patch_slices


def from_slices_to_masks(slices: NDArrayInt, image_size_sqrt: int) -> NDArrayBool:
    n_slice = len(slices)
    patch_masks: NDArrayBool = np.zeros((n_slice, image_size_sqrt, image_size_sqrt), dtype=np.bool)
    for n, (rs, re, cs, ce) in enumerate(slices):
        patch_masks[n, rs:re, cs:ce] = True
    return patch_masks


def patches_features(
        database: Dict[str, NDArrayUInt],
        data_size: int,
        shape: int,
        image_size_sqrt: int=IMAGE_SIZE
    ) -> Tuple[NDArrayStr, NDArrayStr, NDArrayFloat]:

    shape_sqrt = int(np.sqrt(shape))
    columns: NDArrayStr = np.array([f"mean_{i}" for i in range(shape)] + [f"var_{i}" for i in range(shape)])

    slices_broadcast: NDArrayInt = patches_slices_broadcast(shape_sqrt, image_size_sqrt)
    data_array: NDArrayFloat = np.zeros((data_size, 2*shape), dtype=np.float64)
    letter_array: NDArrayStr = np.empty(data_size, dtype="<U10")

    idx: int = 0
    n_imgs: int = 0
    for folder, imgs in database.items():
        n_imgs = len(imgs)

        masked_patches = [imgs[:, rs:re, cs:ce] for rs, re, cs, ce in slices_broadcast]
        data_array[idx:idx+n_imgs, :shape] = np.array([np.mean(patches, axis=(1,2)) for patches in masked_patches]).T
        data_array[idx:idx+n_imgs, shape:] = np.array([np.var(patches, axis=(1,2)) for patches in masked_patches]).T
        letter_array[idx:idx+n_imgs] = folder
        idx += n_imgs

    return columns, letter_array, data_array


def export_visual_features(path: Path, images: NDArrayNum, names: NDArrayStr, shape: int, factor: int = 10) -> None:
    shape_sqrt = int(np.sqrt(shape))
    if not os.path.exists(path):
        os.makedirs(path)

    for idx, img in zip(names, images):
        curr_path = path / str(idx)
        if not os.path.exists(curr_path):
            os.makedirs(curr_path)
        curr_path = curr_path / f"example_{idx}.png"

        arr: NDArrayFloat = np.reshape(img.astype(np.uint8), (-1, shape_sqrt))
        zoom_arr = np.kron(arr, np.ones((factor, factor))).astype(np.uint8)
        cv2.imwrite(str(curr_path), zoom_arr)
