import os
import cv2
import numpy as np
import pandas as pd

def mean_grayscale(database: dict, size: int) -> pd.DataFrame:
    df_means = pd.DataFrame( columns=["Letter"])

    nb_pairs = 28//(size//2)
    print(f"Computing {(nb_pairs - 1)*(nb_pairs - 1)} patches for each image")
    arr = np.linspace(0, 28, nb_pairs + 1)

    for folder, imgs in database.items():
        for img in imgs:
            patch_idx = 0
            row = pd.DataFrame({"Letter": [folder]})

            for i in range(nb_pairs - 1):
                row_l = int(arr[i])
                row_h = int(arr[i + 2])

                for j in range(nb_pairs - 1):
                    col_l = int(arr[j])
                    col_h = int(arr[j + 2])

                    patch = img[row_l:row_h, col_l:col_h]
                    mean = np.mean(patch)

                    row.loc[:, f"patch_{patch_idx}"] = mean
                    patch_idx += 1

            df_means = pd.concat([df_means, row], ignore_index=True)

    return df_means

def export_features(path: str, database: pd.DataFrame, size: int, factor: int = 10) -> None:

    shape = 28//(size//2) - 1
    if not os.path.exists(path):
            os.makedirs(path)

    for idx, row in database.groupby("Letter").first().iterrows():
        curr_path = os.path.join(path, idx)
        if not os.path.exists(curr_path):
            os.makedirs(curr_path)
        curr_path = os.path.join(curr_path, f"example_{idx}.png")

        arr = np.reshape(row.to_numpy(), (-1, shape))
        zoom_arr = np.kron(arr, np.ones((factor, factor)))
        cv2.imwrite(curr_path, zoom_arr)
