import os
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np

from config import IMAGE_SIZE
from common.types import NDArrayFloat, NDArrayStr, NDArrayUInt

class DimensionException(Exception):
    pass


def load_image(image_path: str) -> NDArrayUInt:
    # lit une image en nuances de gris
    img: NDArrayUInt = cv2.imread(image_path, 0).astype(np.uint8)
    height, width = img.shape

    # si l'image ne fait pas 28x28 comme attendu, on lève une exception
    if height != IMAGE_SIZE and width != IMAGE_SIZE:
        raise DimensionException(
            "Image %s has dimension %dx%d which is different from the expected %dx%d.",
            image_path, height, width, IMAGE_SIZE, IMAGE_SIZE)
    return img


def load_folder(folder_path: Path) -> NDArrayUInt:
    # à partir du chemin vers un dossier, on lit toutes les images présentes en nuance de gris
    # et on les stockes dans un numpy array de taille (N, 28, 28)
    # avec N le nombre d'image dans le dossier
    imgs: List[NDArrayUInt] = []
    for file in os.listdir(folder_path):
        try:
            img = load_image(str(folder_path / file))
            imgs.append(img)
        except DimensionException as e:
            print(e)

    return np.array(imgs)


def load_database(path: Path) -> Tuple[Dict[str, NDArrayUInt], int]:
    # à partir du chemin vers la base de données
    # on ouvre chaque dossier et on récupère la liste des images en nuances de gris (taille (N, 28, 28))
    # on retourne un dictionnaire chaque clef étant le nom d'une lettre et sa valeur les images en nuances de gris associées
    db: Dict[str, NDArrayUInt] = {}
    db_size: int = 0
    for folder in os.listdir(path):
        imgs = load_folder(path / folder)
        db_size += len(imgs)
        db[folder] = imgs
    return db, db_size


def load_database_from_save(path: Path) -> Tuple[NDArrayStr, NDArrayStr, NDArrayFloat, int]:
    # on charge la base de données à partir d'un fichier npz, utile pour relancer l'apprentissage plusieurs fois
    # sans recalculer les features
    data = np.load(path)
    data_size = data["letters"].shape[0]
    return data["cols"], data["letters"], data["data"], data_size
