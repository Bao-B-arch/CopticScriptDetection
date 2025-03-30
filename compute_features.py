import os
import cv2
import numpy as np
import pandas as pd

IMAGE_SIZE = 28
BACKGROUND_COLOR = int(0.3*255)

def mean_grayscale(database: dict, data_size: int, patch_size: int) -> pd.DataFrame:
    shape = IMAGE_SIZE // (patch_size//2) - 1
    patch_pairs = np.linspace(0, IMAGE_SIZE, shape + 2).astype(np.uint8)

    col = pd.RangeIndex(0, shape)
    col.append(pd.Index(["Letter"]))
    df = pd.DataFrame(columns=col, index=pd.RangeIndex(0, data_size))

    idx = 0
    for folder, imgs in database.items():
        for img in imgs:
            patch_idx = 0
            df.loc[idx, "Letter"] = folder

            for i in range(shape):
                row_l = patch_pairs[i]
                row_h = patch_pairs[i + 2]

                for j in range(shape):
                    col_l = patch_pairs[j]
                    col_h = patch_pairs[j + 2]

                    patch = img[row_l:row_h, col_l:col_h]
                    mean = np.mean(patch)

                    df.loc[idx, patch_idx] = mean
                    patch_idx += 1
            idx += 1

    return df

def population_std(x: pd.Series) -> float:
    return x.std(ddof=0)

def export_visual_features(path: str, database: pd.DataFrame, patch_size: int, factor: int = 10) -> None:

    shape = IMAGE_SIZE // (patch_size//2) - 1
    if not os.path.exists(path):
        os.makedirs(path)

    for idx, row in database.groupby("Letter").first().iterrows():
        curr_path = os.path.join(path, idx)
        if not os.path.exists(curr_path):
            os.makedirs(curr_path)
        curr_path = os.path.join(curr_path, f"example_{idx}.png")

        arr = np.reshape(row.to_numpy().astype(np.uint8), (-1, shape))
        zoom_arr = np.kron(arr, np.ones((factor, factor))).astype(np.uint8)
        cv2.imwrite(curr_path, zoom_arr)

    distributions = database.groupby("Letter").agg(["mean", population_std])
    letter_stats_max = distributions.T.groupby(level=1).max().T
    max_std = letter_stats_max.loc[:, "population_std"].max()

    for idx, row in distributions.iterrows():
        background = np.zeros(shape=(shape*factor, shape*factor), dtype=np.uint8)
        background += BACKGROUND_COLOR
        curr_path = os.path.join(path, idx)
        if not os.path.exists(curr_path):
            os.makedirs(curr_path)
        curr_path = os.path.join(curr_path, f"avg_{idx}.png")

        for patch_idx, value in row.groupby(level=0):
            x_center = (patch_idx // shape) * factor + factor // 2
            y_center = (patch_idx % shape) * factor + factor // 2

            mean = value.loc[(patch_idx, "mean")]
            std = value.loc[(patch_idx, "population_std")]
            std_factor = int(factor * np.sqrt(std / max_std))
            x_start = x_center - std_factor // 2
            x_stop = x_start + std_factor
            y_start = y_center - std_factor // 2
            y_stop = y_start + std_factor

            square = np.ones(shape=(std_factor, std_factor)) * mean
            background[x_start:x_stop, y_start:y_stop] = square.astype(np.uint8)

        cv2.imwrite(curr_path, background)
