import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importation du module de chargement des données
import data_loading

# Importation des modèles de Scikit-learn
from sklearn.ensemble import RandomForestClassifier

# Importation des outils de validation
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.model_selection import train_test_split 

# Définition de constantes
NUMBER_SECTION_DEL = 50  # Nombre de séparateurs pour l'affichage
DATABASE_PATH = "..//..//Images_Traitees"  # Chemin de la base de données

if __name__ == "__main__":
    # Chargement des données depuis la base
    raw_data = data_loading.load_database(DATABASE_PATH)
    
    # Calcul de la moyenne des niveaux de gris des images
    means_data = data_loading.mean_grayscale(raw_data)

    # Nettoyage des données
    # Suppression des valeurs manquantes
    means_data = means_data.dropna(axis=0)

    # Définition de la variable cible (lettre correspondante)
    y = means_data.loc[:, "Letter"]

    # Définition des caractéristiques (features) utilisées pour la classification
    means_features = ["Mean"]
    X = means_data.loc[:, means_features]
    print(
        f"FEATURE NAME:\n\
{means_features}\n\
{'-'*NUMBER_SECTION_DEL}"
    )

    # Séparation des données en ensembles d'entraînement et de test (80% - 20%)
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
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
        pd.DataFrame({"Letter": train_y, "Set": "Train"}), 
        pd.DataFrame({"Letter": test_y, "Set": "Test"})
    ])
    df_set.value_counts().plot(kind="bar", stacked=True)
    plt.show()

    # Définition et initialisation du modèle de classification Random Forest
    rfc = RandomForestClassifier()
  
    # Entraînement du modèle sur l'ensemble d'entraînement
    rfc.fit(train_X, train_y)
        
    # Prédiction sur un échantillon de 5 données aléatoires
    sample_data = means_data.sample(n=5, random_state=0)
    X_sample = sample_data.loc[:, means_features]
    y_sample = sample_data.loc[:, "Letter"]
    prediction_rfc = rfc.predict(X_sample)
       
    # Visualisation des résultats de prédiction
    pretty_format = lambda table: np.array2string(table, formatter={'float_kind': lambda x: f'{x:.1f}'})
    print("EXAMPLE:")
    print(f"REAL:\t{pretty_format(y_sample.to_numpy())}")
    print(f"RFC:\t{pretty_format(prediction_rfc)}")
    print("-"*NUMBER_SECTION_DEL)
    
    # Validation du modèle avec l'ensemble de test
    acc_rfc = accuracy_score(
        test_y,
        rfc.predict(test_X)
    )

    # Affichage des performances du modèle
    print("METRICS:")
    print(f"ACC RFC:\t{acc_rfc:.2f}")
    print('-'*NUMBER_SECTION_DEL)
     
    # TODO : Améliorations futures
    ## Implémenter un classificateur SVM (Support Vector Machine)
    ## Ajouter de nouvelles caractéristiques plus discriminantes (ex: Histogram of Oriented Gradients - HOG)
    ## Améliorer l'analyse des métriques de performance
    ## Implémenter une approche par Bootstrap pour la validation
