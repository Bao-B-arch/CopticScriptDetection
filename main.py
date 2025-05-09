import os
import subprocess
from timeit import default_timer as timer

from matplotlib import pyplot as plt
import numpy as np

# Importation des outils de préprocessing
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

if __name__ == "__main__":
    main_start_timer = timer()

    # Créer et configurer le pipeline
    pipeline = TrackedPipeline(name="OCR_Pipeline")
    pipeline.load_data(
        target_name="Letter",
        name="initial_data",
        from_save=(not FORCE_COMPUTATION) & os.path.isfile(SAVED_DATABASE_PATH)
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

    # Features Selection
    pipeline.transform(
        transformer=SelectKBest(k=4),
        name="selection_features",
        transform_y=True
    )

    # Division des données
    pipeline.split_data()

    # Grid search pour trouver le meilleur SVM
    svm = SVC(random_state=RANDOM_STATE, class_weight="balanced", cache_size=1000)
    param_grid = {
        "C": np.logspace(4, 10, 2).tolist(),
        "gamma": np.logspace(-7, 0, 2).tolist()
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
    pipeline.export_report()

    if FORCE_COMPUTATION | (not os.path.isfile(SAVED_DATABASE_PATH)):
        pipeline.export_database(path=SAVED_DATABASE_PATH)

    # Génération des graphes
    if not os.path.exists("graphs"):
        os.makedirs("graphs")

    db_before_scaling = next((s for s in pipeline.states if "letter_to_remove" in s.name))
    db_after_scaling = next((s for s in pipeline.states if "normalisation" in s.name))

    labels: NDArrayStr = np.unique(pipeline.current_y)

    visualize_scaling(db_before_scaling.X, db_after_scaling.X)
    visualize_correlation(db_after_scaling.X, pipeline.classes)

    visualize_train_test_split(pipeline.y_train, pipeline.y_test)
    for model_name, metadata in ((s.name, s.metadata) for s in pipeline.states if "eval" in s.name):
        visualize_confusion_matrix(metadata["confusion_matrix"], labels, model_name)

    db_grid = next((s for s in pipeline.states if "grid_search_svm" in s.name))
    data_gs = db_grid.metadata["cv_result"].loc[:, ["param_C", "param_gamma", "mean_test_score"]]
    data_gs = data_gs.set_index(["param_C", "param_gamma"]).unstack()
    data_gs.columns = data_gs.columns.droplevel(level=0)
    print(data_gs)
    visualize_grid_search(data_gs, "SVC")

    plt.tight_layout()
    if FORCE_PLOT:
        plt.show()

    test_quarto = subprocess.run(["quarto", "--version"], stdout = subprocess.DEVNULL)
    if FORCE_REPORT & (test_quarto.returncode == 0):
        subprocess.run(["quarto", "render", ".\coptic_report.qmd", 
                        "--to", "pdf,revealjs", 
                        "--output-dir", "report\quarto"], 
                        stdout = subprocess.DEVNULL,
                        stderr = subprocess.DEVNULL)

    main_end_timer = timer()
    print(f"Script lasted {main_end_timer - main_start_timer:.2f} seconds.")

""" 
    compute_features.export_visual_features(EXPORT_PATH, means_data, NB_SHAPE, FACTOR_SIZE_EXPORT)
 """