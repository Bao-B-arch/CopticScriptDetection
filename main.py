import os
from timeit import default_timer as timer

import numpy as np
import pandas as pd

# Scikit learn setting to export data as pandas
from sklearn import set_config

# Importation des outils de préprocessing
from sklearn.preprocessing import StandardScaler

# Importation des modèles de Scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Importation des outils de validation
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV

# Importation du module de chargement des données
import data_loading
import compute_features
from graph_utils import visualize_grid_search
import reports
from utils import Sets

set_config(transform_output = "pandas")

# Définition de constantes
NUMBER_SECTION_DEL = 50  # Nombre de séparateurs pour l'affichage
DATABASE_PATH = "Images_Traitees"  # Chemin de la base de données
EXPORT_PATH = "export"  # Chemin de la base de données
SAVED_DATABASE_PATH = "database.feather"
RANDOM_STATE = 0
TEST_TRAIN_RATIO = 0.2
PATCH_SIZE = 10
FACTOR_SIZE_EXPORT = 100
LETTER_TO_REMOVE = ["Sampi", "Eta", "Psi", "Ksi", "Zeta"]

# Option pour changer le comportement du scripts
FORCE_COMPUTATION = True
FORCE_PLOT = False
FORCE_REPORT = True

if __name__ == "__main__":
    main_start_timer = timer()

    if (not FORCE_COMPUTATION) & os.path.isfile(SAVED_DATABASE_PATH):
        print("LOADING DATABASE: ", end = '\0')
        start_timer = timer()
        means_data = pd.read_feather(SAVED_DATABASE_PATH)
        data_size = means_data.index.size
        end_timer = timer()
        print(f"Lasted {end_timer - start_timer:.2f} seconds.")
        print("-"*NUMBER_SECTION_DEL)
    else:
        print("LOADING DATABASE: ", end = '\0')
        start_timer = timer()
        raw_data, data_size = data_loading.load_database(DATABASE_PATH)
        end_timer = timer()
        print(f"Lasted {end_timer - start_timer:.2f} seconds.")
        print("-"*NUMBER_SECTION_DEL)

        # Calcul de la moyenne des niveaux de gris des images
        print("FEATURES COMPUTATION: ", end = '\0')
        start_timer = timer()
        means_data = compute_features.mean_grayscale(raw_data, data_size, PATCH_SIZE)
        end_timer = timer()
        print(f"Lasted {end_timer - start_timer:.2f} seconds.")
        print("-"*NUMBER_SECTION_DEL)

        # Nettoyage des données
        # Suppression des valeurs manquantes
        print("FEATURES CLEANING: ", end = '\0')
        start_timer = timer()
        means_data = means_data.dropna(axis=0)
        end_timer = timer()
        print(f"Lasted {end_timer - start_timer:.2f} seconds.")
        print("-"*NUMBER_SECTION_DEL)

    # Retirer les lettres qu'on ne veut pas
    print("REMOVING LETTERS: ", end = '\0')
    start_timer = timer()
    index_to_remove = means_data.loc[:,"Letter"].isin(LETTER_TO_REMOVE)
    removed_means_data = means_data[~index_to_remove]
    data_size = removed_means_data.index.size
    end_timer = timer()
    print(f"Lasted {end_timer - start_timer:.2f} seconds.")
    print("-"*NUMBER_SECTION_DEL)

    # Définition des caractéristiques (features) utilisées pour la classification
    # Définition de la variable cible (lettre correspondante)
    db_noscaling = Sets(
        X = removed_means_data.loc[:, removed_means_data.columns != "Letter"],
        y = removed_means_data.loc[:, "Letter"]
    )

    # Application d'un scaling standard sur les données
    print("FEATURES SCALING: ", end = '\0')
    start_timer = timer()
    scaler = StandardScaler()
    db_scaled = Sets(
        X = scaler.fit_transform(db_noscaling.X),
        y = db_noscaling.y
    )
    end_timer = timer()
    print(f"Lasted {end_timer - start_timer:.2f} seconds.")
    print("-"*NUMBER_SECTION_DEL)

    # Séparation des données en ensembles d'entraînement et de test (80% - 20%)
    print("FEATURES SPLITTING: ", end = '\0')
    start_timer = timer()
    train_X, test_X, train_y, test_y = train_test_split(
        db_scaled.X,
        db_scaled.y,
        test_size=TEST_TRAIN_RATIO,
        random_state=RANDOM_STATE,
        stratify=db_scaled.y
    )
    train = Sets(train_X, train_y)
    test = Sets(test_X, test_y)
    end_timer = timer()
    print(f"Lasted {end_timer - start_timer:.2f} seconds.")
    print("-"*NUMBER_SECTION_DEL)

    # Définition et initialisation du modèle de classification Random Forest
    rfc = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")
    svm = SVC(random_state=RANDOM_STATE, class_weight="balanced", cache_size=1000)

    print("HYPER PARAMETERS TUNNING: ", end = '\0')
    start_timer = timer()
    param_grid = {
        "C": np.logspace(4, 10, 7).tolist(), 
        "gamma": np.logspace(-7, 0, 8).tolist()
    }

    cv = StratifiedShuffleSplit(n_splits=5, test_size=TEST_TRAIN_RATIO, random_state=RANDOM_STATE)
    grid_svm = GridSearchCV(svm,
                            return_train_score=True,
                            param_grid=param_grid,
                            scoring="matthews_corrcoef",
                            cv=cv,
                            n_jobs = -1)

    grid_fit = grid_svm.fit(train.X, train.y)
    cv_result = pd.DataFrame(grid_fit.cv_results_)
    data_gs = cv_result.loc[:, ["param_C", "param_gamma", "mean_test_score"]]
    end_timer = timer()
    print(f"Lasted {end_timer - start_timer:.2f} seconds.")
    print("-"*NUMBER_SECTION_DEL)

    data_gs = data_gs.set_index(["param_C", "param_gamma"]).unstack()
    data_gs.columns = data_gs.columns.droplevel(level=0)
    visualize_grid_search(data_gs, "SVC")

    # Entraînement du modèle sur l'ensemble d'entraînement
    print("MODEL FITTING: ", end = '\0')
    start_timer = timer()
    svm = SVC(random_state=RANDOM_STATE, class_weight="balanced", cache_size=1000, **grid_fit.best_params_)
    svm.fit(train.X, train.y)
    rfc.fit(train.X, train.y)
    end_timer = timer()
    print(f"Lasted {end_timer - start_timer:.2f} seconds.")
    print("-"*NUMBER_SECTION_DEL)

    # Generation d'un report synthétisant les données, les modèles et leur performances
    print("GENERATING REPORT: ", end = '\0')
    start_timer = timer()
    
    removed_letter_counts = means_data.loc[means_data.loc[:, "Letter"].isin(LETTER_TO_REMOVE), "Letter"].value_counts()
    reports.generate_report(
        data_size,
        db_noscaling,
        db_scaled,
        train,
        test,
        {"RFC": rfc, "SVM": svm},
        {l: int(removed_letter_counts[l]) for l in LETTER_TO_REMOVE},
        RANDOM_STATE,
        FORCE_PLOT,
        FORCE_REPORT
    )
    end_timer = timer()
    print(f"Lasted {end_timer - start_timer:.2f} seconds.")
    print("-"*NUMBER_SECTION_DEL)

    # Exportation des données pour visualization
    print("EXPORTING DATA: ", end = '\0')
    start_timer = timer()
    compute_features.export_visual_features(EXPORT_PATH, means_data, PATCH_SIZE, FACTOR_SIZE_EXPORT)
    if FORCE_COMPUTATION | (not os.path.isfile(SAVED_DATABASE_PATH)):
        compute_features.export_database(SAVED_DATABASE_PATH, means_data)
    end_timer = timer()
    print(f"Lasted {end_timer - start_timer:.2f} seconds.")
    print("-"*NUMBER_SECTION_DEL)

    main_end_timer = timer()
    print(f"Script lasted {main_end_timer - main_start_timer:.2f} seconds.")

    # TODO : Améliorations futures
    ## Améliorer les classifier
    ## Implémenter une approche par Bootstrap pour la validation
