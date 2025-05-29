from pathlib import Path
from typing import Any, Optional
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from common.types import NDArrayNum, NDArrayStr
from common.utils import unwrap


class MidpointNormalize(Normalize):
    def __init__(self, vmin: Optional[float]=None, vmax: Optional[float]=None, midpoint: Optional[float]=None, clip: bool=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value: Any, clip: Optional[bool]=None) -> Any:
        x, y = [unwrap(self.vmin), unwrap(self.midpoint), unwrap(self.vmax)], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def visualize_scaling(graph_folder: Path, graph_folder_for_quarto: Path, data_before: NDArrayNum, data_after: NDArrayNum) -> None:
    """
    Visualise l"effet du standard scaling avec détection des outliers via MAD
    
    Paramètres:
    
    Retourne:
    Graphiques de boxplots et statistiques des outliers
    """
    # Créer une figure avec des sous-graphiques
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Analyse du Standard Scaling et Détection des Outliers (Méthode MAD)", fontsize=16)

    # Boxplot avant scaling
    sns.boxplot(data=data_before, ax=axes[0])
    axes[0].set_title("Distribution Originale")

    # Boxplot après scaling
    sns.boxplot(data=data_after, ax=axes[1])
    axes[1].set_title("Distribution Après Standard Scaling")
    plt.savefig(graph_folder / "scaling.svg")
    plt.savefig(graph_folder_for_quarto / "scaling.svg")


def visualize_correlation(graph_folder: Path, graph_folder_for_quarto: Path, data: NDArrayNum, labels: NDArrayStr) -> None:

    plt.figure(figsize=(15, 8))
    sns.heatmap(
        np.corrcoef(data, rowvar=False), 
        xticklabels=labels.tolist(), 
        yticklabels=labels.tolist(),
        annot=True, fmt=".2f",
    )
    plt.title("Correlation Matrix", fontsize=16)
    plt.savefig(graph_folder / "correlation.svg")
    plt.savefig(graph_folder_for_quarto / "correlation.svg")


def visualize_train_test_split(graph_folder: Path, graph_folder_for_quarto: Path, train: NDArrayStr, test: NDArrayStr) -> None:

    unique_train, counts_train = np.unique(train, return_counts=True)
    unique_test, counts_test = np.unique(test, return_counts=True)
    count_sort_ind = np.argsort(-counts_train-counts_test)

    plt.figure(figsize=(15, 8))
    plt.bar(unique_train[count_sort_ind], counts_train[count_sort_ind], color=["steelblue"], label="train")
    plt.bar(unique_test[count_sort_ind], counts_test[count_sort_ind], color=["red"], label="test", bottom=counts_train[count_sort_ind])
    plt.xticks(rotation=45)
    plt.legend()
    plt.title("Number of letters in each dataset")
    plt.savefig(graph_folder / "split.svg")
    plt.savefig(graph_folder_for_quarto / "split.svg")


def visualize_confusion_matrix(graph_folder: Path, graph_folder_for_quarto: Path, cm: NDArrayNum, labels: NDArrayStr, model_name: str) -> None:

    plt.figure()
    sns.heatmap(cm, xticklabels=labels.tolist(), yticklabels=labels.tolist())
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.subplots_adjust(bottom=0.20, left=0.20)

    plt.savefig(graph_folder / f"cm_{model_name}.svg")
    plt.savefig(graph_folder_for_quarto / f"cm_{model_name}.svg")


def visualize_grid_search(
    graph_folder: Path,
    graph_folder_for_quarto: Path,
    search: NDArrayNum,
    x_name: NDArrayNum,
    x_name_label: str,
    y_name: NDArrayNum,
    y_name_label: str,
    model_name: str
) -> None:

    plt.figure()
    sns.heatmap(
        search, 
        cmap=sns.color_palette("flare", as_cmap=True), 
        norm=MidpointNormalize(vmin=0.2, midpoint=0.92), 
        xticklabels=x_name.tolist(), 
        yticklabels=y_name.tolist(),
        annot=True, fmt=".2f", 
    )
    plt.xlabel(x_name_label)
    plt.ylabel(y_name_label)
    plt.title(f"Search for {model_name}")
    plt.yticks(rotation=0)
    plt.subplots_adjust(bottom=0.20, left=0.20)

    plt.savefig(graph_folder / f"search_{model_name}.svg")
    plt.savefig(graph_folder_for_quarto / f"search_{model_name}.svg")
