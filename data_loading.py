import numpy as np
import pandas as pd
import cv2
import os

def load_image(image_path):
    # reads image as grayscale
    img = cv2.imread(image_path, 0) 
    return img

def load_folder(folder_path):
    # charger toutes les images dans un dossier
    # retourner une liste d'images
    imgs = []
    for file in os.listdir(folder_path):
        img = load_image(os.path.join(folder_path, file))
        imgs.append(img)
    
    return imgs

def load_database(path):
    # charger tous les dossier de la base des données 
    # retourner un dictionnaire d'load_image
    db = dict()
    for folder in os.listdir(path):
        imgs = load_folder(os.path.join(path, folder))
        db[folder] = imgs
    return db
   
def mean_grayscale(database):
    df_means = pd.DataFrame(columns=["Mean", "Letter"])
    for folder, imgs in database.items():
        for img in imgs:
            mean = np.mean(img)
            row = pd.DataFrame({"Mean": [mean], "Letter": [folder]})
            df_means = pd.concat([df_means, row], ignore_index=True)

    return df_means


if __name__ == "__main__":
    print("Data_Loading")
