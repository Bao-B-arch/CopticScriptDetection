import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

class InvalidEnvException(Exception):
    """Exception raised when an environment variable does not exist."""

def load_env(name: str, default: Optional[str]=None) -> str:
    """Loads an environment variable either from the environment or from a default generating function."""
    value = os.getenv(name)

    if value is None:
        if default is None:
            raise InvalidEnvException
        else:
            return default
    return value

# Définition de constantes
NUMBER_SECTION_DEL = int(load_env("NUMBER_SECTION_DEL", "50"))

DATABASE_PATH = Path(load_env("DATABASE_PATH", "Images_Traitees"))
EXPORT_PATH = Path(load_env("EXPORT_PATH", "export"))
REPORT_PATH = Path(load_env("REPORT_PATH", "report"))
GRAPH_PATH = Path(load_env("GRAPH_PATH", "graph"))
SAVED_DATABASE_PATH = Path(load_env("SAVED_DATABASE_PATH", "database.npz"))

RANDOM_STATE = int(load_env("RANDOM_STATE", "0"))
TEST_TRAIN_RATIO = float(load_env("TEST_TRAIN_RATIO", "0.2"))
NB_SHAPE = int(load_env("NB_SHAPE", "25"))
FACTOR_SIZE_EXPORT = int(load_env("FACTOR_SIZE_EXPORT", "100"))
LETTER_TO_REMOVE = load_env("LETTER_TO_REMOVE", "Sampi,Eta,Psi,Ksi,Zeta").split(",")

# Option pour changer le comportement du scripts
FORCE_COMPUTATION = load_env("FORCE_COMPUTATION", "TRUE").upper() in ("TRUE", "1", "T")
FORCE_PLOT = load_env("FORCE_PLOT", "FALSE").upper() in ("TRUE", "1", "T")
FORCE_REPORT = load_env("FORCE_REPORT", "TRUE").upper() in ("TRUE", "1", "T")

IMAGE_SIZE = int(load_env("IMAGE_SIZE", "28"))
BACKGROUND_COLOR = int(load_env("BACKGROUND_COLOR", "77"))
