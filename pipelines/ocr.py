import os
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common.config import FORCE_COMPUTATION, FORCE_PLOT, RANDOM_STATE, SAVED_DATABASE_PATH
from common.transformer import LetterRemover
from pipelines.pipeline import TrackedPipeline


def run_ocr(**config: Any) -> None:

    # Retirer les lettres du .env
    remover = LetterRemover()
    # pour features selection
    selector = SelectKBest(k=4)
    # model svm
    param_grid = {
        "C": np.logspace(-3, 5, 7).tolist(),
        "gamma": np.logspace(-7, 1, 7).tolist()
    }

    TrackedPipeline.from_config(name="OCR_coptic", **config)\
        .load_data(
            target_name="Letter",
            name="initial_data",
            from_save=(not FORCE_COMPUTATION) & os.path.isfile(SAVED_DATABASE_PATH)
    )\
        .transform(
            transformer=remover,
            name="letter_to_remove",
            transform_y=True,
            apply_y=True,
            LETTER_REMOVED=remover.get_removed_count,
    )\
        .transform(
            transformer=StandardScaler(), 
            name="normalisation",
     )\
        .transform(
            transformer=selector,
            name="selection_features",
            transform_y=True,
            SELECTED_FEATURES=lambda : [int(i) for i in selector.get_support(indices=True)]
    )\
        .split()\
        .train_model_searched(
            model=SVC(random_state=RANDOM_STATE, class_weight="balanced", cache_size=1000),
            param_grid=param_grid, 
            name="search_svm"
    )\
        .train_model(
            model=RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced"), 
            name="rfc"
    )\
        .evaluate_models()\
        .example()\
        .export_report()\
        .export_database(
            export=FORCE_COMPUTATION | (not os.path.isfile(SAVED_DATABASE_PATH)),
            path=SAVED_DATABASE_PATH,
    )\
        .export_graphes(force_plot=FORCE_PLOT)\
        .export_visual_features()\
        .build_quarto()
