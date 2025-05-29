import os
from typing import Any

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common.config import FORCE_COMPUTATION, RANDOM_STATE, SAVED_DATABASE_PATH
from common.transformer import LetterRemover
from pipelines.config import parse_config_selection
from pipelines.pipeline import TrackedPipeline


def run_selection(**config: Any) -> None:

    # Retirer les lettres du .env
    remover = LetterRemover()
    selectors = parse_config_selection(config)

    TrackedPipeline.from_config(name="OCR_coptic", **config)\
        .load_data(
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
            transformers=selectors,
            model=SVC(random_state=RANDOM_STATE, class_weight="balanced", cache_size=1000),
            name="test_selection_features",
            n_fold=5
    )\
        .export_report(
            file_path="test_selection_report.yaml"
    )\
        .export_selection_graphes()
