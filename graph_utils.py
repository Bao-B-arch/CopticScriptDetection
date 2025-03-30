import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def visualize_scaling(data: pd.DataFrame) -> dict:
    """
    Visualise l'effet du standard scaling avec détection des outliers via MAD
    
    Paramètres:
    data (pandas.DataFrame): DataFrame avec les features à visualiser
    
    Retourne:
    Graphiques de boxplots et statistiques des outliers
    """
    # Créer une figure avec des sous-graphiques
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Analyse du Standard Scaling et Détection des Outliers (Méthode MAD)", fontsize=16)

    # Boxplot avant scaling
    sns.boxplot(data=data, ax=axes[0])
    axes[0].set_title("Distribution Originale")

    # Calcul du scaling
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Boxplot après scaling
    sns.boxplot(data=data_scaled, ax=axes[1])
    axes[1].set_title('Distribution Après Standard Scaling')

def visualize_correlation(data: pd.DataFrame) -> None:

    f = plt.figure(figsize=(19, 15))
    plt.matshow(data.corr(), fignum=f.number)
    plt.xticks(
        range(data.select_dtypes(['number']).shape[1]),
        data.select_dtypes(['number']).columns,
        fontsize=14,
        rotation=45)
    plt.yticks(
        range(data.select_dtypes(['number']).shape[1]),
        data.select_dtypes(['number']).columns,
        fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)

def visualize_train_test_split(train: pd.Series, test: pd.Series) -> None:

    df_set = pd.concat([
        pd.DataFrame({"Letter": train}).value_counts(),
        pd.DataFrame({"Letter": test}).value_counts(),
    ], axis=1, keys=["train", "test"])

    df_set.plot(kind="bar", stacked=True, color=["steelblue", "red"])
    plt.title("Number of letters in each dataset")

def visualize_correlation_matrix(cm: np.ndarray, labels: np.ndarray, model_name: str) -> None:

    plt.figure()
    sns.heatmap(cm, xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix for {model_name}")
