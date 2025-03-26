import numpy as np
import pandas as pd

def mean_grayscale(database, size):
    df_means = pd.DataFrame(
        columns=["Letter"]
    )
    nb_pairs = 28//(size//2)
    print(f"Computing {(nb_pairs - 1)*(nb_pairs - 1)} patches for each image")
    arr = np.linspace(0, 28, nb_pairs + 1)

    for folder, imgs in database.items():
        for img in imgs:
            patch_idx = 0
            row = pd.DataFrame({"Letter": [folder]})

            for i in range(nb_pairs - 1):
                col_l = int(arr[i])
                col_h = int(arr[i + 2])

                for j in range(nb_pairs - 1):
                    row_l = int(arr[j])
                    row_h = int(arr[j + 2])

                    patch = img[row_l:row_h, col_l:col_h]
                    mean = np.mean(patch)
                    row.loc[:, f"patch_{patch_idx}"] = mean
                    patch_idx += 1

            df_means = pd.concat([df_means, row], ignore_index=True)

    return df_means
