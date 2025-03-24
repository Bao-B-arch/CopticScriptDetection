import numpy as np
import pandas as pd
import cv2
import os
   
def mean_grayscale(database):
    df_means = pd.DataFrame(columns=["Mean", "TopMean", "BottomMean", "Letter"])
    for folder, imgs in database.items():
        for img in imgs:
            mean = np.mean(img)
            lines = img.shape[0]
            top_mean = np.mean(img[:lines//2])
            bottom_mean = np.mean(img[lines//2:])
            row = pd.DataFrame({
                "Mean": [mean], 
                "TopMean": [top_mean], 
                "BottomMean": [bottom_mean], 
                "Letter": [folder]
            })
            df_means = pd.concat([df_means, row], ignore_index=True)

    return df_means
