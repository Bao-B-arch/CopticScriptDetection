from typing import Optional, Tuple, TypeVar, cast
import numpy as np
import pandas as pd
from pandas import RangeIndex

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


def mad(data: NDArrayNum) -> Tuple[float, float]:
    """
    Calcul de la Median Absolute Deviation (MAD)
    
    Paramètres:
    data (array-like): Données à analyser
    
    Retourne:
    float: La médiane absolue des écarts
    """
    median = np.median(data)
    mad_value = np.median(np.abs(data - median))
    return float(median), float(mad_value)


def outlier_analysis(data: NDArrayNum) -> pd.DataFrame:
# Analyse statistique des outliers avec MAD
    outliers_stats = pd.DataFrame(
        index=RangeIndex(data.shape[1]),
        columns=['nb_outliers', 'percent', 
                 'median', 'mad', 
                 'min_original', 'max_original', 
                 'mean_original', 'std_original', 
                 'lower_bound', 'upper_bound']
    )
    for ncol, col in enumerate(data.T):
        # Calcul des outliers avec la méthode MAD
        median_val, mad_val = mad(col)

        # Seuil standard : points à plus de 3 MAD de la médiane
        lower_bound = median_val - 3 * mad_val
        upper_bound = median_val + 3 * mad_val

        outliers = data[(col < lower_bound) | (col > upper_bound)]
        outliers_stats.loc[ncol, 'nb_outliers'] = len(outliers)
        outliers_stats.loc[ncol, 'percent'] = len(outliers) / len(data) * 100
        outliers_stats.loc[ncol, 'median'] = float(median_val)
        outliers_stats.loc[ncol, 'mad'] = float(mad_val)
        outliers_stats.loc[ncol, 'min_original'] = float(np.min(col))
        outliers_stats.loc[ncol, 'max_original'] = float(np.min(col))
        outliers_stats.loc[ncol, 'mean_original'] = float(np.mean(col))
        outliers_stats.loc[ncol, 'std_original'] = float(np.std(col))
        outliers_stats.loc[ncol, 'lower_bound'] = float(lower_bound)
        outliers_stats.loc[ncol, 'upper_bound'] = float(upper_bound)

    return outliers_stats
