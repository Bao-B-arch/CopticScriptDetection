import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import Normalize

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

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
    plt.savefig("graphs/scaling.svg")

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
    plt.savefig("graphs/correlation.svg")

def visualize_train_test_split(train: pd.Series, test: pd.Series) -> None:

    df_set = pd.concat([
        pd.DataFrame({"Letter": train}).value_counts(),
        pd.DataFrame({"Letter": test}).value_counts(),
    ], axis=1, keys=["train", "test"])

    df_set.plot(kind="bar", stacked=True, color=["steelblue", "red"])
    plt.title("Number of letters in each dataset")
    plt.subplots_adjust(bottom=0.25)
    plt.savefig("graphs/split.svg")

def visualize_confusion_matrix(cm: np.ndarray, labels: np.ndarray, model_name: str) -> None:

    plt.figure()
    sns.heatmap(cm, xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.savefig(f"graphs/cm_{model_name}.svg")

def visualize_grid_search(grid_search: pd.DataFrame, model_name: str) -> None:

    plt.figure()
    sns.heatmap(grid_search, cmap=plt.cm.hot, norm=MidpointNormalize(vmin=0.2, midpoint=0.92),)
    plt.xlabel(grid_search.columns.name)
    plt.ylabel(grid_search.index.name)
    plt.title(f"Grid Search for {model_name}")
    plt.savefig(f"graphs/gs_{model_name}.svg")
