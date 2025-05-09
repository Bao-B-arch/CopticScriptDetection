import os
from pathlib import Path
import subprocess
from timeit import default_timer as timer

from matplotlib import pyplot as plt
import numpy as np

# Importation des outils de préprocessing
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler

# Importation des modèles de Scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Importation du module de chargement des données
from common.config import FORCE_COMPUTATION, FORCE_PLOT, FORCE_REPORT, RANDOM_STATE, SAVED_DATABASE_PATH
from common.graph_utils import visualize_confusion_matrix, visualize_correlation, visualize_grid_search, visualize_scaling, visualize_train_test_split
from common.pipeline import TrackedPipeline
from common.transformer import LetterRemover
from common.types import NDArrayStr

def run_ocr(nb_shapes: int, selection: bool) -> None:
    main_start_timer = timer()

    # Créer et configurer le pipeline
    pipeline = TrackedPipeline(name=f"OCR_{nb_shapes}")
    pipeline.load_data(
        target_name="Letter",
        name="initial_data",
        from_save=(not FORCE_COMPUTATION) & os.path.isfile(SAVED_DATABASE_PATH),
        nb_shape=nb_shapes
    )

    # Retirer les lettres du .env
    remover = LetterRemover()
    pipeline.transform(
        transformer=remover,
        name="letter_to_remove",
        transform_y=True,
        apply_y=True,
        LETTER_REMOVED=remover.get_removed_count,
    )
    
    # Normalisaton
    pipeline.transform(
        transformer=StandardScaler(), 
        name="normalisation",
    )

    if selection:
        selector = SelectKBest(k=4)
        # Features Selection
        pipeline.transform(
            transformer=selector,
            name="selection_features",
            transform_y=True,
            SELECTED_FEATURES=selector.get_support
        )

    # Division des données
    pipeline.split_data()

    # Grid search pour trouver le meilleur SVM
    svm = SVC(random_state=RANDOM_STATE, class_weight="balanced", cache_size=1000)
    param_grid = {
        "C": np.logspace(-3, 5, 7).tolist(),
        "gamma": np.logspace(-7, 1, 7).tolist()
    }
    pipeline.grid_search(
        estimator=svm,
        param_grid=param_grid, 
        name="grid_search_svm"
    )
    grid_svm = pipeline.get_last_estimator_grid_search()

    best_svm = SVC(random_state=RANDOM_STATE, class_weight="balanced", cache_size=1000, **grid_svm.best_params_)
    pipeline.train_model(
        model=best_svm, 
        name="svm"
    )

    rfc = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")
    pipeline.train_model(
        model=rfc, 
        name="rfc"
    )

    # Évaluer les modèles
    pipeline.evaluate_model()
    pipeline.example()
    pipeline.export_report(selection=selection)

    if FORCE_COMPUTATION | (not os.path.isfile(SAVED_DATABASE_PATH)):
        pipeline.export_database(path=SAVED_DATABASE_PATH)

    # Génération des graphes
    graph_folder = Path(f"graphs_{nb_shapes}_{selection}")
    if not os.path.exists(graph_folder):
        os.makedirs(graph_folder)
    if not os.path.exists("graphs"):
        os.makedirs("graphs")

    db_before_scaling = next((s for s in pipeline.states if "letter_to_remove" in s.name))
    db_after_scaling = next((s for s in pipeline.states if "normalisation" in s.name))

    labels: NDArrayStr = np.unique(pipeline.current_y)

    visualize_scaling(graph_folder, db_before_scaling.X, db_after_scaling.X)
    visualize_correlation(graph_folder, db_after_scaling.X, pipeline.classes)

    visualize_train_test_split(graph_folder, pipeline.y_train, pipeline.y_test)
    for model_name, metadata in ((s.name, s.metadata) for s in pipeline.states if "eval" in s.name):
        visualize_confusion_matrix(graph_folder, metadata["confusion_matrix"], labels, model_name)

    db_grid = next((s for s in pipeline.states if "grid_search_svm" in s.name))
    data_gs = pd.DataFrame(db_grid.metadata["cv_results"]).loc[:, ["param_C", "param_gamma", "mean_test_score"]]
    data_gs = data_gs.set_index(["param_C", "param_gamma"]).unstack()
    data_gs.columns = data_gs.columns.droplevel(level=0)
    visualize_grid_search(graph_folder, data_gs, "SVC")

    plt.tight_layout()
    if FORCE_PLOT:
        plt.show()

    test_quarto = subprocess.run(["quarto", "--version"], stdout = subprocess.DEVNULL)
    if FORCE_REPORT & (test_quarto.returncode == 0):
        a = subprocess.run(["quarto", "render", ".\coptic_report.qmd", 
                        "--to", "pdf,revealjs",
                        "--output", f"OCR_{nb_shapes}_{selection}",
                        "--output-dir", "report\quarto"], 
                        stdout = subprocess.DEVNULL,
                        stderr = subprocess.DEVNULL)
        print(a)

    main_end_timer = timer()
    print(f"Script lasted {main_end_timer - main_start_timer:.2f} seconds.")

""" 
    compute_features.export_visual_features(EXPORT_PATH, means_data, NB_SHAPE, FACTOR_SIZE_EXPORT)
 """

if __name__ == "__main__":
    run_ocr(4, False)
    run_ocr(16, False)
    run_ocr(16, True)
    run_ocr(784, False)
