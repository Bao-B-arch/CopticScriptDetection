import os
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np
import pandas as pd

from common.config import IMAGE_SIZE
from common.types import NDArrayBool, NDArrayFloat, NDArrayInt, NDArrayNum, NDArrayStr


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


def patches_features(database: Dict[str, NDArrayInt], data_size: int, shape: int, target_name: str, image_size_sqrt: int=IMAGE_SIZE) -> pd.DataFrame:
    shape_sqrt = int(np.sqrt(shape))
    columns: List[str] = [str(i) for i in range(shape)] + [target_name]

    slices_broadcast: NDArrayInt = patches_slices_broadcast(shape_sqrt, image_size_sqrt)
    data_array: NDArrayFloat = np.zeros((data_size, shape), dtype=np.float64)
    letter_array: NDArrayStr = np.empty(data_size, dtype="<U10")

    idx: int = 0
    n_imgs: int = 0
    for folder, imgs in database.items():
        n_imgs = len(imgs)
        masked_patches = np.stack([imgs[:, rs:re, cs:ce] for rs, re, cs, ce in slices_broadcast], axis=1)
        data_array[idx:idx+n_imgs] = np.mean(masked_patches, axis=(2, 3))
        letter_array[idx:idx+n_imgs] = folder
        idx += n_imgs

    return pd.DataFrame(
        data=np.hstack([data_array, letter_array.reshape(-1, 1)]),
        columns=columns
    )


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
