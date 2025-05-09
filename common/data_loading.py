import os
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import pandas as pd

from common.config import IMAGE_SIZE
from common.types import NDArrayInt

class DimensionException(Exception):
    pass


def load_image(image_path: str) -> NDArrayInt:
    # reads image as grayscale
    img: NDArrayInt = cv2.imread(image_path, 0).astype(np.uint8)
    height, width = img.shape

    if height != IMAGE_SIZE and width != IMAGE_SIZE:
        raise DimensionException(
            "Image %s has dimension %dx%d which is different from the expected %dx%d.",
            image_path, height, width, IMAGE_SIZE, IMAGE_SIZE)
    return img


def load_folder(folder_path: Path) -> List[NDArrayInt]:
    # charger toutes les images dans un dossier
    # retourner une liste d'images
    imgs: List[NDArrayInt] = []
    for file in os.listdir(folder_path):
        try:
            img = load_image(str(folder_path / file))
        except DimensionException as e:
            print(e)
        imgs.append(img)

    return imgs


def load_database(path: Path) -> Tuple[Dict[str, List[NDArrayInt]], int]:
    # charger tous les dossier de la base des données
    # retourner un dictionnaire d'load_image
    db: Dict[str, List[NDArrayInt]] = {}
    db_size: int = 0
    for folder in os.listdir(path):
        imgs = load_folder(path / folder)
        db_size += len(imgs)
        db[folder] = imgs
    return db, db_size


def load_database_from_save(path: Path) -> Tuple[pd.DataFrame, int]:
    data = pd.read_feather(path)
    data_size = data.index.size
    return data, data_size
