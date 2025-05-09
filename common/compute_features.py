import os
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np
import pandas as pd

from common.config import BACKGROUND_COLOR, IMAGE_SIZE
from common.types import NDArrayBool, NDArrayFloat, NDArrayInt
from common.utils import population_std

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


def export_visual_features(path: Path, database: pd.DataFrame, shape: int, factor: int = 10) -> None:
    shape = np.sqrt(shape).astype(np.uint)
    if not os.path.exists(path):
        os.makedirs(path)

    for idx, row in database.groupby("Letter").first().iterrows():
        curr_path = path / str(idx)
        if not os.path.exists(curr_path):
            os.makedirs(curr_path)
        curr_path = curr_path / f"example_{idx}.png"

        arr = np.reshape(row.to_numpy().astype(np.uint8), (-1, shape))
        zoom_arr = np.kron(arr, np.ones((factor, factor))).astype(np.uint8)
        cv2.imwrite(str(curr_path), zoom_arr)

    distributions = database.groupby("Letter").agg(["mean", population_std])
    letter_stats_max = distributions.T.groupby(level=1).max().T
    max_std = letter_stats_max.loc[:, "population_std"].max()

    for idx, row in distributions.iterrows():
        background = np.zeros(shape=(shape*factor, shape*factor), dtype=np.uint8)
        background += BACKGROUND_COLOR
        curr_path = path / str(idx)
        if not os.path.exists(curr_path):
            os.makedirs(curr_path)
        curr_path = curr_path / f"avg_{idx}.png"

        for patch_idx, value in row.groupby(level=0):
            x_center = (int(patch_idx) // shape) * factor + factor // 2
            y_center = (int(patch_idx) % shape) * factor + factor // 2

            mean = value.loc[(patch_idx, "mean")]
            std = value.loc[(patch_idx, "population_std")]
            std_factor = int(factor * np.sqrt(std / max_std))
            x_start = x_center - std_factor // 2
            x_stop = x_start + std_factor
            y_start = y_center - std_factor // 2
            y_stop = y_start + std_factor

            square = np.ones(shape=(std_factor, std_factor)) * mean
            background[x_start:x_stop, y_start:y_stop] = square.astype(np.uint8)

        cv2.imwrite(str(curr_path), background)
