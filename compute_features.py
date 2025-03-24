import numpy as np
import pandas as pd
import cv2
import os
   
def mean_grayscale(database):
    df_means = pd.DataFrame(columns=["Mean", "Letter"])
    for folder, imgs in database.items():
        for img in imgs:
            mean = np.mean(img)
            row = pd.DataFrame({"Mean": [mean], "Letter": [folder]})
            df_means = pd.concat([df_means, row], ignore_index=True)

    return df_means
