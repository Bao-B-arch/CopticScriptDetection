import os
from typing import Any

from config import FORCE_COMPUTATION, SAVED_DATABASE_PATH
from pipelines.config import parse_config_run
from pipelines.pipeline import TrackedPipeline


# Pipeline spécifique permettant de vérifier quelle technique de sélection de features est la plus pertinente.
def run_features(**config: Any) -> None:

    _ = parse_config_run(config)

    TrackedPipeline.from_config(name="OCR_coptic", **config)\
        .load_data(
            name="initial_data",
            from_save=(not FORCE_COMPUTATION) & os.path.isfile(SAVED_DATABASE_PATH)
    )\
        .export_visual_features()
