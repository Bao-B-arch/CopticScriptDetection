from typing import Dict, Optional, Tuple, TypeVar
import numpy as np

from common.types import NDArrayNum

class NoneException(Exception):
    pass


T = TypeVar("T")
def unwrap(obj: Optional[T], msg: str = "Cannot unwrap %s") -> T:
    if obj is None:
        raise NoneException(msg, obj)
    return obj


def population_std(x: NDArrayNum) -> float:
    return float(np.std(x, ddof=0))


def jaccard_index(a: NDArrayNum, b: NDArrayNum) -> float:
    union = len(set(a) | set(b))
    inter = len(set(a) & set(b))

    return inter / float(union) if union != 0 else 0.0


def subspace_similarity(components1: NDArrayNum, components2: NDArrayNum) -> float:
    """Calcule la similarité entre 2 sous-espaces via les valeurs singulières"""
    U1, _ = np.linalg.qr(components1.T)
    U2, _ = np.linalg.qr(components2.T)
    shape = min(components1.shape[0], components2.shape[0])
    return float(np.linalg.norm(U1.T @ U2, ord="nuc")) / float(shape) if shape != 0 else 0.0


def mad(data: NDArrayNum, axis: int) -> Tuple[NDArrayNum, NDArrayNum]:
    """
    Calcul de la Median Absolute Deviation (MAD)
    
    Paramètres:
    data (array-like): Données à analyser
    
    Retourne:
    float: La médiane absolue des écarts
    """
    median: NDArrayNum = np.median(data, axis=axis)
    mad_value = np.median(np.abs(data - median), axis=axis)
    return median, mad_value


def outlier_analysis(data: NDArrayNum) -> Dict[str, NDArrayNum]:
# Analyse statistique des outliers avec MAD

    median_val, mad_val = mad(data, axis=0)
    # Seuil standard : points à plus de 3 MAD de la médiane
    lower_bound = median_val - 3 * mad_val
    upper_bound = median_val + 3 * mad_val
    masks_outliers = (data < lower_bound) | (data > upper_bound)

    outliers_stats: Dict[str, NDArrayNum] = {}
    outliers_stats["nb_outliers"] = np.sum(masks_outliers, axis=0)
    outliers_stats["percent"] = np.sum(masks_outliers, axis=0) / len(data) * 100

    outliers_stats["median"] = median_val
    outliers_stats["mad"] = mad_val
    outliers_stats["min_original"] = np.min(data, axis=0)
    outliers_stats["max_original"] = np.max(data, axis=0)
    outliers_stats["mean_original"] = np.mean(data, axis=0)
    outliers_stats["std_original"] = np.std(data, axis=0)
    outliers_stats["lower_bound"] = lower_bound
    outliers_stats["upper_bound"] = upper_bound

    return outliers_stats
