import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importation des modèles de Scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Importation des outils de validation
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import train_test_split 

# Importation du module de chargement des données
import data_loading
import compute_features

# Définition de constantes
NUMBER_SECTION_DEL = 50  # Nombre de séparateurs pour l'affichage
DATABASE_PATH = "Images_Traitees"  # Chemin de la base de données
RANDOM_STATE = 0

def pretty_format(table):
    return np.array2string(table, formatter={'float_kind': lambda x: f'{x:.1f}'})

if __name__ == "__main__":
    # Chargement des données depuis la base
    raw_data = data_loading.load_database(DATABASE_PATH)

    # Calcul de la moyenne des niveaux de gris des images
    means_data = compute_features.mean_grayscale(raw_data)

    # Nettoyage des données
    # Suppression des valeurs manquantes
    means_data = means_data.dropna(axis=0)

    # Définition de la variable cible (lettre correspondante)
    y = means_data.loc[:, "Letter"]
    classes = np.unique(y)

    # Définition des caractéristiques (features) utilisées pour la classification
    means_features = ["Mean", "TopMean", "BottomMean", "LeftMean", "RightMean"]
    X = means_data.loc[:, means_features]
    print(
        f"FEATURE NAME:\n\
{means_features}\n\
{'-'*NUMBER_SECTION_DEL}"
    )

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

    # Visualisation de la répartition des données entre Train et Test
    df_set = pd.concat([
        pd.DataFrame({"Letter": train_y}).value_counts(),
        pd.DataFrame({"Letter": test_y}).value_counts(),
    ], axis=1, keys=["train", "test"])
    df_set.plot(kind="bar", stacked=True, color=["steelblue", "red"])
    plt.title("Number of letters in each dataset")

    # Définition et initialisation du modèle de classification Random Forest
    rfc = RandomForestClassifier()
    svm = SVC()

    # Entraînement du modèle sur l'ensemble d'entraînement
    rfc.fit(train_X, train_y)
    svm.fit(train_X, train_y)

    # Prédiction sur un échantillon de 5 données aléatoires
    X_sample = test_X.sample(n=5, random_state=RANDOM_STATE)
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
    plt.show()

    # TODO : Améliorations futures
    ## Améliorer les classifier
    ## Extraire de nouvelles caractéristiques (features)(ex: Histogram of Oriented Gradients - HOG)
    ## Implémenter une approche par Bootstrap pour la validation
