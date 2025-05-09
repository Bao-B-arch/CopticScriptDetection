from typing import Tuple
import numpy as np
import pandas as pd

from common.types import NDArrayNum


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


def outlier_analysis(data: pd.DataFrame) -> pd.DataFrame:
# Analyse statistique des outliers avec MAD
    outliers_stats = pd.DataFrame(
        index=data.columns,
        columns=['nb_outliers', 'percent', 
                 'median', 'mad', 
                 'min_original', 'max_original', 
                 'mean_original', 'std_original', 
                 'lower_bound', 'upper_bound']
    )
    for col in data.columns:
        # Calcul des outliers avec la méthode MAD
        median_val, mad_val = mad(data[col].to_numpy())

        # Seuil standard : points à plus de 3 MAD de la médiane
        lower_bound = median_val - 3 * mad_val
        upper_bound = median_val + 3 * mad_val

        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
        outliers_stats.loc[col, 'nb_outliers'] = len(outliers)
        outliers_stats.loc[col, 'percent'] = len(outliers) / len(data) * 100
        outliers_stats.loc[col, 'median'] = float(median_val)
        outliers_stats.loc[col, 'mad'] = float(mad_val)
        outliers_stats.loc[col, 'min_original'] = float(data[col].min())
        outliers_stats.loc[col, 'max_original'] = float(data[col].max())
        outliers_stats.loc[col, 'mean_original'] = float(data[col].mean())
        outliers_stats.loc[col, 'std_original'] = float(data[col].std())
        outliers_stats.loc[col, 'lower_bound'] = float(lower_bound)
        outliers_stats.loc[col, 'upper_bound'] = float(upper_bound)

    return outliers_stats
