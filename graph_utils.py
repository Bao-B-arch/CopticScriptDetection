import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def mad(data):
    """
    Calcul de la Median Absolute Deviation (MAD)
    
    Paramètres:
    data (array-like): Données à analyser
    
    Retourne:
    float: La médiane absolue des écarts
    """
    median = np.median(data)
    mad_value = np.median(np.abs(data - median))
    return mad_value

def visualize_scaling(X: pd.DataFrame) -> dict:
    """
    Visualise l'effet du standard scaling avec détection des outliers via MAD
    
    Paramètres:
    X (pandas.DataFrame): DataFrame avec les features à visualiser
    
    Retourne:
    Graphiques de boxplots et statistiques des outliers
    """
    # Créer une figure avec des sous-graphiques
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Analyse du Standard Scaling et Détection des Outliers (Méthode MAD)", fontsize=16)

    # Boxplot avant scaling
    sns.boxplot(data=X, ax=axes[0])
    axes[0].set_title("Distribution Originale")

    # Calcul du scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Boxplot après scaling
    sns.boxplot(data=X_scaled, ax=axes[1])
    axes[1].set_title('Distribution Après Standard Scaling')

    # Analyse statistique des outliers avec MAD
    outliers_stats = {}
    for col in X.columns:
        # Calcul des outliers avec la méthode MAD
        median_val = np.median(X[col])
        mad_val = mad(X[col])

        # Seuil standard : points à plus de 3 MAD de la médiane
        lower_bound = median_val - 3 * mad_val
        upper_bound = median_val + 3 * mad_val

        outliers = X[(X[col] < lower_bound) | (X[col] > upper_bound)]
        outliers_stats[col] = {
            'nb_outliers': len(outliers),
            'percent': len(outliers) / len(X) * 100,
            'median': median_val,
            'mad': mad_val,
            'min_original': X[col].min(),
            'max_original': X[col].max(),
            'mean_original': X[col].mean(),
            'std_original': X[col].std(),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }

    # Affichage des statistiques des outliers
    print("OUTLIERS (USING MAD):")
    start = True
    for col, stats in outliers_stats.items():
        if start:
            print(" "*15, end='\0')
            for stat_name in stats:
                print(f"|{stat_name:15}", end='\0')
            start = False
        print(f"\n{col:15}", end='\0')
        for _, stat_value in stats.items():
            print(f"|{stat_value:15.2f}", end='\0')
    print("\n", end='\0')
    return outliers_stats
