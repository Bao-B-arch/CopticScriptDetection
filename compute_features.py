import numpy as np
import pandas as pd
import cv2
import os
   
def mean_grayscale(database):
    df_means = pd.DataFrame(
        columns=["Mean", "TopMean", "BottomMean", "LeftMean", "RightMean", "Letter"]
    )

    for folder, imgs in database.items():
        for img in imgs:
            lines = img.shape[0]
            cols = img.shape[1]

            mean = np.mean(img)
            top_mean = np.mean(img[:lines//2])
            bottom_mean = np.mean(img[lines//2:])
            left_mean = np.mean(img[:, :cols//2])
            right_mean = np.mean(img[:, cols//2:])
            
            row = pd.DataFrame({
                "Mean": [mean], 
                "TopMean": [top_mean], 
                "BottomMean": [bottom_mean], 
                "LeftMean": [left_mean], 
                "RightMean": [right_mean], 
                "Letter": [folder]
            })
            df_means = pd.concat([df_means, row], ignore_index=True)

    return df_means
