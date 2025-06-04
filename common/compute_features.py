import os
from pathlib import Path
from typing import Dict, Tuple
import cv2
import numpy as np

from config import IMAGE_SIZE
from common.types import NDArrayBool, NDArrayFloat, NDArrayInt, NDArrayNum, NDArrayStr, NDArrayUInt


# on calcule les features à explorer sur chaque patch (morceau d'image)
# il nous faut donc d'abord un moyen de calculer les patch demandés
def patches_slices_broadcast(nb_patch_sqrt: int, image_size_sqrt: int) -> NDArrayInt:
    if nb_patch_sqrt >= image_size_sqrt:
        raise ValueError("Computing more overlapping patches than the size of the image does not make any sense.")
    else:
        # on calcule des patches qui se chevauchent de moitié. Par exemple pour l'image suivante et 4 patches
        # [ [100, 220, 200],
        #   [150, 220, 200],
        #   [100, 200, 125] ]
        # on exporte les patches suivants
        # patch 1
        # [ [100, 200],
        #   [150, 220] ]
        # patch 2
        # [ [220, 200],
        #   [220, 200] ]
        # patch 3
        # [ [150, 220],
        #   [100, 200] ]
        # patch 4
        # [ [220, 200],
        #   [200, 125] ]
        start_end = np.linspace(start=0, stop=image_size_sqrt, num = 2 + nb_patch_sqrt, dtype=np.int8)
        inc = 2

    patch_slices: NDArrayInt = np.zeros((nb_patch_sqrt*nb_patch_sqrt, 4), dtype=np.int64)

    for i in range(nb_patch_sqrt):
        row_start: int = start_end[i]
        row_end: int = start_end[i + inc]
        
        for j in range(nb_patch_sqrt):
            col_start: int = start_end[j]
            col_end: int = start_end[j + inc]

            # Par simplicité, on retourne les slices liées au patches. 
            # En effet, on peut retrouver les patches à partir des slices et on a juste besoin de calculer 4 indices. En reprenant l'exemple ci-dessus
            # patch 1
            # row_start = 0
            # row_end = 2
            # col_start = 0
            # col_end = 2
            # plus d'informations peuvent être trouvée au niveau des test unitaires
            patch_slices[i*nb_patch_sqrt + j, :] = [row_start, row_end, col_start, col_end]
    return patch_slices


# fonction utilitaire pour vérifier le calcul des patches dans les test unitaires
def from_slices_to_masks(slices: NDArrayInt, image_size_sqrt: int) -> NDArrayBool:
    n_slice = len(slices)
    patch_masks: NDArrayBool = np.zeros((n_slice, image_size_sqrt, image_size_sqrt), dtype=np.bool)
    for n, (rs, re, cs, ce) in enumerate(slices):
        patch_masks[n, rs:re, cs:ce] = True
    return patch_masks


# pour chaque patch, on calcule la moyenne et la variance du niveau de gris et les 3 harmoniques non-principales du patch
# ainsi on obtient 2*nb_patch features
# en général on a 16 patches, ce qui donne donc 80 features
def patches_features(
        database: Dict[str, NDArrayUInt],
        data_size: int,
        shape: int,
        image_size_sqrt: int=IMAGE_SIZE
    ) -> Tuple[NDArrayStr, NDArrayStr, NDArrayFloat]:

    shape_sqrt = int(np.sqrt(shape))
    columns: NDArrayStr = np.array(
        [f"mean_{i}" for i in range(shape)] + 
        [f"var_{i}" for i in range(shape)]
    )

    letter_array: NDArrayStr = np.empty(data_size, dtype="<U10")
    idx: int = 0
    n_imgs: int = 0

    if shape_sqrt >= image_size_sqrt:
        # si on veut calculer plus de patches que la taille de l'image, on retourne à la place l'image entière en tant que features
        # bien évidemment, on ne calcule pas de moyenne ni écart type dans ce cas
        data_array: NDArrayFloat = np.zeros((data_size, image_size_sqrt*image_size_sqrt), dtype=np.float64)
        for folder, imgs in database.items():
            n_imgs = len(imgs)
            data_array[idx:idx+n_imgs, :] = imgs.reshape(-1, image_size_sqrt*image_size_sqrt)
            letter_array[idx:idx+n_imgs] = folder
            idx += n_imgs

        return columns, letter_array, data_array
    
    # sinon, on rentre dans le cas classique
    # on calcule les slices pour retrouver les patches
    # et on calcule pour chaque patch, la moyenne et la variance
    data_array: NDArrayFloat = np.zeros((data_size, 2*shape), dtype=np.float64)
    slices_broadcast: NDArrayInt = patches_slices_broadcast(shape_sqrt, image_size_sqrt)

    idx: int = 0
    n_imgs: int = 0
    for folder, imgs in database.items():
        n_imgs = len(imgs)

        masked_patches = [imgs[:, rs:re, cs:ce] for rs, re, cs, ce in slices_broadcast]
        data_array[idx:idx+n_imgs, :shape] = np.array([np.mean(patches, axis=(1,2)) for patches in masked_patches]).T
        data_array[idx:idx+n_imgs, shape:2*shape] = np.array([np.var(patches, axis=(1,2)) for patches in masked_patches]).T

        letter_array[idx:idx+n_imgs] = folder
        idx += n_imgs

    return columns, letter_array, data_array


# fonction servant à exporter les features moyennes
# pour chaque lettre, on affiche en tant qu'image les features moyennes de la première image
# ainsi pour une lettre de taille 28x28, la représentation de ses features moyennes devient une image de 4x4 (car on a classiquement 16 patches)
def export_visual_features(path: Path, images: NDArrayNum, names: NDArrayStr, shape: int, factor: int = 10) -> None:
    shape_sqrt = int(np.sqrt(shape))
    if not os.path.exists(path):
        os.makedirs(path)

    for idx, img in zip(names, images):
        curr_path = path / str(idx)
        if not os.path.exists(curr_path):
            os.makedirs(curr_path)
        curr_path_mean = curr_path / f"example_mean_{idx}.png"
        curr_path_std = curr_path / f"example_std_{idx}.png"

        img_mean = img[:shape]
        arr_mean: NDArrayFloat = np.reshape(img_mean.astype(np.uint8), (-1, shape_sqrt))
        zoom_arr_mean = np.kron(arr_mean, np.ones((factor, factor))).astype(np.uint8)
        cv2.imwrite(str(curr_path_mean), zoom_arr_mean)

        if len(img) > shape:
            img_std = np.sqrt(img[shape:])
            arr_std: NDArrayFloat = np.reshape(img_std.astype(np.uint8), (-1, shape_sqrt))
            zoom_arr_std = np.kron(arr_std, np.ones((factor, factor))).astype(np.uint8)
            cv2.imwrite(str(curr_path_std), zoom_arr_std)
