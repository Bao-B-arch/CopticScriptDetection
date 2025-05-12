import os
from typing import Any

from sklearn.calibration import LinearSVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common.config import FORCE_COMPUTATION, RANDOM_STATE, SAVED_DATABASE_PATH
from common.transformer import LetterRemover
from pipelines.pipeline import TrackedPipeline


def run_selection(**config: Any) -> None:

    # Retirer les lettres du .env
    remover = LetterRemover()

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
        .test_transforms(
            transformers={
                "anova": SelectKBest(score_func=f_classif, k="all"),
                "mi": SelectKBest(score_func=mutual_info_classif, k="all"),
                "pca": PCA(random_state=RANDOM_STATE),
                "lda": LinearDiscriminantAnalysis(),
                "l1": SelectFromModel(LinearSVC(penalty="l1", dual=False, random_state=RANDOM_STATE)),
                "rfe_1": RFE(LinearSVC(penalty="l1", dual=False, random_state=RANDOM_STATE, C=10.0, max_iter=5000), n_features_to_select=1.0),
                "rfe_0": RFE(LinearSVC(penalty="l1", dual=False, random_state=RANDOM_STATE, C=1.0, max_iter=5000), n_features_to_select=1.0),
                "rfe_-1": RFE(LinearSVC(penalty="l1", dual=False, random_state=RANDOM_STATE, C=.1, max_iter=5000), n_features_to_select=1.0),
            },
            model=SVC(random_state=RANDOM_STATE, class_weight="balanced", cache_size=1000),
            name="test_selection_features",
            cv=5
    )\
        .export_report(
            file_path="test_selection_report.yaml"
    )\
        .export_selection_graphes()
