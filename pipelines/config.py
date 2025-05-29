
from typing import Any, Dict
from sklearn.base import BaseEstimator
from sklearn.calibration import LinearSVC
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, f_classif, mutual_info_classif

from common.config import RANDOM_STATE
from common.transformer import DoNothingSelector
from common.types import Transformer

SELECTORS: Dict[str, Transformer] = {
    "anova": SelectKBest(score_func=f_classif, k="all"),
    "mi": SelectKBest(score_func=mutual_info_classif, k="all"),
    "l1": SelectFromModel(LinearSVC(penalty="l1", dual=False, random_state=RANDOM_STATE, C=.1, max_iter=5000)),
    "rfe": RFE(LinearSVC(penalty="l1", dual=False, random_state=RANDOM_STATE, C=.1, max_iter=5000), n_features_to_select=1.0)
}


def parse_config_run(config: Dict[str, Any]) -> BaseEstimator:
    if "selection" not in config or config["selection"] is None:
        selector = DoNothingSelector()
        config["selection"] = "None"
    else:
        if config["selection"] in SELECTORS:
            selector = SELECTORS[config["selection"]]

            if hasattr(selector, "k"):
                selector.set_params(k=4)
            elif hasattr(selector, "n_components"):
                selector.set_params(n_components=4)
            elif hasattr(selector, "max_features"):
                selector.set_params(max_features=4)
            elif hasattr(selector, "n_features_to_select"):
                selector.set_params(n_features_to_select=4)
            else:
                ValueError("Cannot modify selector %s.", selector)
        else:
            raise ValueError("Invalid selector %s in config.", config)
    return selector


def parse_config_selection(config: Dict[str, Any]) -> Dict[str, BaseEstimator]:
    if "selection" not in config or config["selection"] is None or not isinstance(config["selection"], list):
        raise ValueError("Invalid selector %s in config.", config)

    selectors: Dict[str, BaseEstimator] = {}
    for selection in config["selection"]:
        if selection in SELECTORS:
            selectors[selection] = SELECTORS[selection]
        else:
            raise ValueError("Invalid selectors %s in config.", config["selection"])
    config["selection"] = "None"
    return selectors