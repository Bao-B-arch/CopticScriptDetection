import os
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np
import pandas as pd

from common.config import IMAGE_SIZE
from common.types import NDArrayBool, NDArrayFloat, NDArrayInt, NDArrayNum, NDArrayStr


def patches_mask(nb_patch_sqrt: int, image_size_sqrt: int = IMAGE_SIZE) -> List[NDArrayBool]:
    ## no overlap
    if nb_patch_sqrt >= image_size_sqrt:
        start_end = np.arange(start=0, stop=image_size_sqrt + 1, dtype=np.int8)
        inc = 1
        nb_patch_sqrt = image_size_sqrt
    else: ## overlap
        start_end = np.linspace(start=0, stop=image_size_sqrt, num = 2 + nb_patch_sqrt, dtype=np.int8)
        inc = 2

    patch_masks: List[NDArrayBool] = [
        np.zeros((image_size_sqrt, image_size_sqrt), dtype=bool) for _ in range(nb_patch_sqrt*nb_patch_sqrt)
    ]

    for i in range(nb_patch_sqrt):
        row_start: int = start_end[i]
        row_end: int = start_end[i + inc]
        
        for j in range(nb_patch_sqrt):
            col_start: int = start_end[j]
            col_end: int = start_end[j + inc]

            # Créer un masque pour ce patch
            mask_idx = i*nb_patch_sqrt + j
            patch_masks[mask_idx][row_start:row_end, col_start:col_end] = True
    return patch_masks


def patches_features(database: Dict[str, List[NDArrayInt]], data_size: int, shape: int) -> pd.DataFrame:
    shape_sqrt = int(np.sqrt(shape))
    patch_masks: List[NDArrayBool] = patches_mask(shape_sqrt)

    columns = [str(i) for i in range(shape)] + ["Letter"]
    df = pd.DataFrame(columns=columns, index=pd.RangeIndex(0, data_size))

    all_data: List[NDArrayFloat] = []
    all_letters: List[str] = []
    for folder, imgs in database.items():
        for img in imgs:
            means = np.array([np.mean(img[mask]) for mask in patch_masks])
            all_data.append(means)
            all_letters.append(folder)

    data_array = np.array(all_data)
    letter_array = np.array(all_letters)

    df.iloc[:, :-1] = data_array
    df.iloc[:, -1] = letter_array
    return df


def export_visual_features(path: Path, images: NDArrayNum, names: NDArrayStr, shape: int, factor: int = 10) -> None:
    shape_sqrt = int(np.sqrt(shape))
    if not os.path.exists(path):
        os.makedirs(path)

    for idx, img in zip(names, images):
        curr_path = path / str(idx)
        if not os.path.exists(curr_path):
            os.makedirs(curr_path)
        curr_path = curr_path / f"example_{idx}.png"

        arr = np.reshape(img.astype(np.uint8), (-1, shape_sqrt))
        zoom_arr = np.kron(arr, np.ones((factor, factor))).astype(np.uint8)
        cv2.imwrite(str(curr_path), zoom_arr)
