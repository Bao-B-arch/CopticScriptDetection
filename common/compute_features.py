import os
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np
import pandas as pd

from common.config import IMAGE_SIZE
from common.types import NDArrayBool, NDArrayFloat, NDArrayInt, NDArrayNum, NDArrayStr


def mean_grayscale(database: Dict[str, List[NDArrayInt]], data_size: int, shape: int) -> pd.DataFrame:
    shape_sqrt = int(np.sqrt(shape))
    patch_masks: List[NDArrayBool] = []

    for i in range(shape_sqrt):
        row_start = int(i * IMAGE_SIZE / shape_sqrt)
        row_end = int((i + 1) * IMAGE_SIZE / shape_sqrt) if i < shape_sqrt - 1 else IMAGE_SIZE
        
        for j in range(shape_sqrt):
            col_start = int(j * IMAGE_SIZE / shape_sqrt)
            col_end = int((j + 1) * IMAGE_SIZE / shape_sqrt) if j < shape_sqrt - 1 else IMAGE_SIZE
            
            # Créer un masque pour ce patch
            mask: NDArrayBool = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
            mask[row_start:row_end, col_start:col_end] = True
            patch_masks.append(mask)

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
