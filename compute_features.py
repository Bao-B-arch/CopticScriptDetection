import numpy as np
import pandas as pd
from sklearn.feature_extraction import image

def mean_grayscale(database, size):
    df_means = pd.DataFrame(
        columns=["Letter"]
    )

    for folder, imgs in database.items():
        for img in imgs:
            patches = image.extract_patches_2d(img, (size, size))
            patches = patches[::size//2]

            row = pd.DataFrame({"Letter": [folder]})

            for idx, patch in enumerate(patches):
                mean = np.mean(patch)
                row.loc[:, f"patch_{idx}"] = mean

            df_means = pd.concat([df_means, row], ignore_index=True)

    return df_means
