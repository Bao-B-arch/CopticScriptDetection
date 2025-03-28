from timeit import default_timer as timer
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit learn setting to export data as pandas
from sklearn import set_config

# Importation des outils de préprocessing
from sklearn.preprocessing import StandardScaler

# Importation des modèles de Scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Importation des outils de validation
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import train_test_split

# Importation du module de chargement des données
import data_loading
import compute_features
import graph_utils

set_config(transform_output = "pandas")

# Définition de constantes
NUMBER_SECTION_DEL = 50  # Nombre de séparateurs pour l'affichage
DATABASE_PATH = "Images_Traitees"  # Chemin de la base de données
EXPORT_PATH = "export"  # Chemin de la base de données
RANDOM_STATE = 0
PATCH_SIZE = 10
FACTOR_SIZE_EXPORT = 100

def pretty_format(table):
    return np.array2string(table, formatter={'float_kind': lambda x: f'{x:.1f}'})

if __name__ == "__main__":
    # Chargement des données depuis la base
    raw_data, data_size = data_loading.load_database(DATABASE_PATH)

    # Calcul de la moyenne des niveaux de gris des images
    print("FEATURES COMPUTATION:")
    start = timer()
    means_data = compute_features.mean_grayscale(raw_data, data_size, PATCH_SIZE)
    compute_features.export_features(EXPORT_PATH, means_data, PATCH_SIZE, FACTOR_SIZE_EXPORT)
    end = timer()
    print(end - start)
    print("-"*NUMBER_SECTION_DEL)

    # Nettoyage des données
    # Suppression des valeurs manquantes
    means_data = means_data.dropna(axis=0)

    # Définition de la variable cible (lettre correspondante)
    y = means_data.loc[:, "Letter"]
    classes = np.unique(y)

    print("TARGET DESCRIPTION:")
    for index, value in y.value_counts().items():
        print(f"{index}:\t\t{value}")
    print("-"*NUMBER_SECTION_DEL)

    # Définition des caractéristiques (features) utilisées pour la classification
    X = means_data.loc[:, means_data.columns != "Letter"]
    means_features = X.columns

    print(
        f"FEATURE NAME:\n\
{means_features}\n\
{'-'*NUMBER_SECTION_DEL}"
    )

    graph_utils.visualize_scaling(X)
    print("-"*NUMBER_SECTION_DEL)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Séparation des données en ensembles d'entraînement et de test (80% - 20%)
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(
        f"DATA SIZE:\n\
TOTAL DATA SIZE:\t{len(X)}\n\
TRAIN SIZE:\t\t{len(train_X)} | POURCENTAGE:{(len(train_X) / len(X) * 100):.2f}%\n\
TEST SIZE:\t\t{len(test_X)} | POURCENTAGE:{(len(test_X) / len(X) * 100):.2f}%\n\
{'-'*NUMBER_SECTION_DEL}"
    )

    f = plt.figure(figsize=(19, 15))
    plt.matshow(X.corr(), fignum=f.number)
    plt.xticks(
        range(X.select_dtypes(['number']).shape[1]),
        X.select_dtypes(['number']).columns,
        fontsize=14,
        rotation=45)
    plt.yticks(
        range(X.select_dtypes(['number']).shape[1]),
        X.select_dtypes(['number']).columns,
        fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)

    print(
        f"CORRELATION MATRIX:\n\
{X.corr()}\n\
{'-'*NUMBER_SECTION_DEL}"
    )

    # Visualisation de la répartition des données entre Train et Test
    df_set = pd.concat([
        pd.DataFrame({"Letter": train_y}).value_counts(),
        pd.DataFrame({"Letter": test_y}).value_counts(),
    ], axis=1, keys=["train", "test"])
    df_set.plot(kind="bar", stacked=True, color=["steelblue", "red"])
    plt.title("Number of letters in each dataset")

    # Définition et initialisation du modèle de classification Random Forest
    rfc = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")
    svm = SVC(random_state=RANDOM_STATE, class_weight="balanced")

    # Entraînement du modèle sur l'ensemble d'entraînement
    rfc.fit(train_X, train_y)
    svm.fit(train_X, train_y)

    # Prédiction sur un échantillon de 5 données aléatoires
    X_sample = test_X.sample(n=10, random_state=RANDOM_STATE)
    y_sample = test_y.loc[X_sample.index]
    prediction_rfc = rfc.predict(X_sample)
    prediction_svm = svm.predict(X_sample)

    # Visualisation des résultats de prédiction
    print("EXAMPLE:")
    print(f"REAL:\t{pretty_format(y_sample.to_numpy())}")
    print(f"RFC:\t{pretty_format(prediction_rfc)}")
    print(f"SVM:\t{pretty_format(prediction_svm)}")
    print("-"*NUMBER_SECTION_DEL)

    # Vérifier à partir de score la qualité du modèle (avec l'ensemble de test)
    rfc_pred_X = rfc.predict(test_X)
    svm_pred_X = svm.predict(test_X)

    ## accuracy
    acc_rfc = accuracy_score(test_y, rfc_pred_X)
    acc_svm = accuracy_score(test_y, svm_pred_X)

    ## matthews corrcoeff
    mcc_rfc = matthews_corrcoef(test_y, rfc_pred_X)
    mcc_svm = matthews_corrcoef(test_y, svm_pred_X)

    ## confusion matrix
    cm_rfc = confusion_matrix(test_y, rfc_pred_X, labels=classes, normalize="true")
    cm_svm = confusion_matrix(test_y, svm_pred_X, labels=classes, normalize="true")

    # Affichage des performances du modèle
    print("METRICS:")
    print(f"ACC RFC:\t{acc_rfc:.2f}")
    print(f"ACC SVM:\t{acc_svm:.2f}")
    print(f"MCC RFC:\t{mcc_rfc:.2f}")
    print(f"MCC SVM:\t{mcc_svm:.2f}")
    print('-'*NUMBER_SECTION_DEL)

    plt.figure()
    sns.heatmap(cm_rfc, xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix for RFC")

    plt.figure()
    sns.heatmap(cm_svm, xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix for SVM")

    plt.tight_layout()

    # TODO : Améliorations futures
    ## Améliorer les classifier
    ## Implémenter une approche par Bootstrap pour la validation
    ## Exporter les features pour gain de temps lors du calcul
