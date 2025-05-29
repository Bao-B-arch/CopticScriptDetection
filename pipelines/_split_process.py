
from typing import Any, Dict, List
from attrs import define
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

from common.data_stages import SplitData
from common.types import NDArrayStr
from config import RANDOM_STATE, TEST_TRAIN_RATIO


# classe contenant les fonctionalités permettant de:
# - entrainer un modèle
# - faire une recherche d'hyperparamètres
# - évaluer accuracy, MCC et matrice de confusion d'un modèle
# - calculer un example de prédiction
@define(auto_attribs=True)
class SplitProcess:
    """Pipeline qui garde une trace de toutes les transformations"""
    state: SplitData

    def __init__(self, /, *, state: SplitData):
        self.state = state


    @classmethod
    def train_model(
        cls, /, *,
        data: SplitData,
        model: BaseEstimator,
        name: str,
    ) -> SplitData:
        """Entraîne un modèle"""
        
        # Entraîner le modèle
        model.fit(data.X_train, data.y_train)

        return SplitData(
            X_train=data.X_train,
            y_train=data.y_train,
            X_test=data.X_test,
            y_test=data.y_test,
            models={**data.models, name:(None, model)}
        )
    

    @classmethod
    def _validate_param_grid(cls, /, *, param_grid: Dict[str, List[Any]]) -> None:
        for key, values in param_grid.items():
            if not isinstance(values, list) or len(values) == 0:
                raise ValueError(f"Paramètre invalide pour {key}: {values}")
        

    @classmethod
    def train_model_searched(
        cls, /, *,
        data: SplitData,
        model: BaseEstimator,
        name: str,
        param_grid: Dict[str, List[Any]],
        cv: int = 5,
        scoring: str = "matthews_corrcoef"
    ) -> SplitData:
        """Effectue une recherche d'hyperparamètres et entraîne un modèle"""
        
        cls._validate_param_grid(param_grid=param_grid)
        # Configurer la validation croisée
        stratified_cv = StratifiedShuffleSplit(
            n_splits=cv,
            test_size=TEST_TRAIN_RATIO,
            random_state=RANDOM_STATE
        )

        # Effectuer la recherche par grille
        search = GridSearchCV(
            model,
            param_grid,
            scoring=scoring,
            cv=stratified_cv,
            n_jobs=-1,
            return_train_score=True
        )
        search_fit = search.fit(data.X_train, data.y_train)

        searched_model = model.set_params(**search_fit.best_params_)
        searched_model.fit(data.X_train, data.y_train)

        return SplitData(
            X_train=data.X_train,
            y_train=data.y_train,
            X_test=data.X_test,
            y_test=data.y_test,
            models={**data.models, name:(search_fit, searched_model)}
        )


    @classmethod
    def evaluate(
        cls, /, *,
        data: SplitData,
        model: BaseEstimator
    ) -> Dict[str, np.typing.ArrayLike]:

        y_pred = model.predict(data.X_test)
        # Calculer les métriques
        mcc = matthews_corrcoef(data.y_test, y_pred)
        acc = accuracy_score(data.y_test, y_pred)
        cm = confusion_matrix(data.y_test, y_pred, normalize="true")

        return {
            "mcc": mcc,
            "acc": acc,
            "cm": cm,
        }
    

    @classmethod
    def example(
        cls, /, *,
        data: SplitData,
        n: int = 10,
        random_state: int = RANDOM_STATE,
    ) -> Dict[str, NDArrayStr]:

        # Prédiction sur un échantillon de 5 données aléatoires
        mask = np.random.default_rng(random_state).choice(data.X_test.shape[0], replace=False, size=n) 
        sample_x = data.X_test[mask]
        sample_y = data.y_test[mask]

        pred_dict: Dict[str, NDArrayStr] = {}
        pred_dict["REAL"] = sample_y

        for model_name, (_, model) in data.models.items():
            pred = model.predict(sample_x)
            pred_dict[model_name] = pred

        return pred_dict
