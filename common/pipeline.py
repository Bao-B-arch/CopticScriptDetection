from __future__ import annotations

import functools
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Callable, Concatenate, Dict, List, Optional, Self, Tuple, TypeVar, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit, train_test_split
import yaml

from common import compute_features, data_loading
from common.config import DATABASE_PATH, EXPORT_PATH, NB_SHAPE, NUMBER_SECTION_DEL, \
    RANDOM_STATE, REPORT_PATH, SAVED_DATABASE_PATH, TEST_TRAIN_RATIO
from common.datastate import DataState
from common.types import NDArrayInt, NDArrayNum, NDArrayStr
from common.utils import outlier_analysis

if TYPE_CHECKING:
    _TP = TrackedPipeline
else:
    _TP = TypeVar("_TP", bound="TrackedPipeline")

TrackedPipelineMethod = Callable[Concatenate[_TP, ...], _TP, ]

class PipelineException(Exception):
    pass


def timer(name: str) -> Callable[[TrackedPipelineMethod], TrackedPipelineMethod]:
    """Print the runtime of the decorated function"""
    def decorator_timer(func: TrackedPipelineMethod) -> TrackedPipelineMethod:
        @functools.wraps(func)
        def wrapper_timer(pipeline: TrackedPipeline, **kwargs: Any) -> TrackedPipeline:

            _name = name
            if "name" in kwargs:
                _name += f"_{kwargs['name']}"

            print(f"{_name}: ", end = '\0')
            start_time = time.perf_counter()
            res: TrackedPipeline = func(pipeline, **kwargs)
            end_time = time.perf_counter()
            run_time = end_time - start_time
            print(f"Lasted {run_time:.2f} seconds.")
            print("-"*NUMBER_SECTION_DEL)

            pipeline.add_time(_name, run_time)
            return res
        return cast(TrackedPipelineMethod, wrapper_timer)
    return decorator_timer


class TrackedPipeline:
    """Pipeline qui garde une trace de toutes les transformations"""
    
    def __init__(self, name: str = "OCR_Pipeline"):
        self.name = name
        self.classes: Optional[NDArrayStr] = None
        self.states: List[DataState] = []
        self.transformers: List[Tuple[str, BaseEstimator]] = []
        self.current_X: Optional[NDArrayNum] = None
        self.current_y: Optional[NDArrayStr] = None
        self.raw: Optional[Dict[str, List[NDArrayInt]]] = None
        self.X_train: Optional[NDArrayNum] = None
        self.y_train: Optional[NDArrayStr] = None
        self.X_test: Optional[NDArrayNum] = None
        self.y_test: Optional[NDArrayStr] = None
        self.data_size: int = 0
        self.models: Dict[str, BaseEstimator] = {}
        self.last_estimator_grid_search: Optional[BaseEstimator] = None
    
    def add_state(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Ajoute l'état actuel des données à l'historique"""
        if metadata is None:
            metadata = {}

        if self.current_X is not None and self.current_y is not None:
            self.states.append(DataState(
                X=self.current_X,
                y=self.current_y,
                name=name,
                metadata=metadata
            ))


    def add_time(self, name: str, run_time: float) -> Self:
        {
            "execution_time": run_time,
            "step": name,
        }
        return self


    @timer("LOADING RAW DATABASE")
    def __load_data_from_files(self, /) -> Self:
        raw_data, data_size = data_loading.load_database(DATABASE_PATH)
        self.data_size = data_size
        self.raw = raw_data
        return self
    
    def __to_data(self, /, *, data: pd.DataFrame, target_name: str) -> Self:
        self.classes = data.columns.drop([target_name]).to_numpy()
        self.current_X = data.loc[:, self.classes].to_numpy()
        self.current_y = data.loc[:, target_name].to_numpy()
        return self


    @timer("FEATURES COMPUTATION")
    def __compute_features(self, /, *, nb_shape: int, target_name: str) -> Self:
        if self.raw is None:
            raise PipelineException("No raw data for features computation.")
        data = compute_features.mean_grayscale(self.raw, self.data_size, nb_shape)
        data = data.dropna(axis=0)
        self.__to_data(data=data, target_name=target_name)
        return self


    @timer("LOADING FEATHER DATABASE")
    def __load_data_from_save(self, /, *, target_name: str) -> Self:
        data, data_size = data_loading.load_database_from_save(SAVED_DATABASE_PATH)
        self.data_size = data_size
        self.__to_data(data=data, target_name=target_name)
        return self
    

    def load_data(self, /, *, target_name: str, name: str = "initial_data", from_save: bool = False, nb_shape: int = NB_SHAPE) -> Self:
        """Charge les données initiales dans la pipeline"""
        if from_save:
            self.__load_data_from_save(target_name=target_name)
        else:
            self.__load_data_from_files().__compute_features(nb_shape=nb_shape, target_name=target_name)

        if self.classes is None:
            raise PipelineException("Unset classes.")
        metadata = {
            "SIZE": self.data_size,
            "TARGET": pd.DataFrame(self.current_y, columns=[target_name]).loc[:, target_name].value_counts().to_dict(),
            "NB_PATCHES": nb_shape,
            "FEATURES": list(self.classes),
            "OUTLIERS_ANALYSIS": outlier_analysis(pd.DataFrame(self.current_X, columns=self.classes)).to_dict(),
        }
        self.add_state(name, metadata)
        return self


    @timer("TRANSFORMER")
    def transform(
        self, /, *,
        transformer: BaseEstimator,
        name: str,
        fit: bool = True,
        transform_y: bool = False,
        apply_y: bool = False,
        **metadata: Any
    ) -> Self:
        """Applique une transformation aux données avec traçabilité"""

        if self.current_X is None:
            raise ValueError("No load data.")

        # Appliquer la transformation
        if fit:
            if transform_y:
                transformer.fit(self.current_X, self.current_y)
            else:
                transformer.fit(self.current_X)

        if apply_y:
            self.current_X, self.current_y = transformer.transform(self.current_X, self.current_y)
        else:
            self.current_X = transformer.transform(self.current_X)

        # Stocker le transformateur
        self.transformers.append((name, transformer))

        metadata.update({
            k: v() for k, v in metadata.items() if callable(v)
        })
        metadata.update({
            "transformer_type": type(transformer).__name__,
            "transformer_params": {k: str(v) for k,v in transformer.get_params().items()}
        })

        # Enregistrer le nouvel état
        self.add_state(f"after_{name}", metadata)
        return self


    @timer("TRAIN_TEST_SPLIT")
    def split_data(
        self, /, *,
        test_size: float = TEST_TRAIN_RATIO,
        random_state: int = RANDOM_STATE,
        stratify: bool = True,
        **metadata: Any
    ) -> Self:
        """Divise les données en ensembles d'entraînement et de test"""

        # Effectuer la séparation
        stratify_param = self.current_y if stratify else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.current_X, self.current_y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=stratify_param
        )

        metadata.update({
            k: v() for k, v in metadata.items() if callable(v)
        })
        metadata.update({
            "TRAIN_SIZE": {"value": self.X_train.shape[0], "percent": self.X_train.shape[0] / self.current_X.shape[0] * 100},
            "TEST_SIZE": {"value": self.X_test.shape[0], "percent": self.X_test.shape[0] / self.current_X.shape[0] * 100},
            "TEST_RATIO": test_size,
            "STRATIFIED": stratify
        })

        # Enregistrer l'état actuel
        self.add_state("train_test_split", metadata)
        return self


    @timer("TRAIN")
    def train_model(
            self, /, *,
            model: BaseEstimator,
            name: str,
            **metadata:Any
    ) -> Self:
        """Entraîne un modèle et enregistre les métadonnées"""
        
        # Entraîner le modèle
        if self.X_train is None or self.y_train is None:
            raise PipelineException("Cannot train the model since no train sets are defined.")
        model.fit(self.X_train, self.y_train)
        self.models[name] = model

        metadata.update({
            k: v() for k, v in metadata.items() if callable(v)
        })
        metadata.update({
            "model_type": type(model).__name__,
            "model_params": model.get_params(),
            "training_data_shape": list(self.X_train.shape)
        })

        self.states.append(DataState(
            X=np.array([]),  # Placeholder vide
            y=np.array([]),  # Placeholder vide
            name=f"model_{name}",
            metadata=metadata
        ))

        return self


    @timer("TUNING HYPER PARAMETERS")
    def grid_search(
        self, /, *, 
        estimator: BaseEstimator, 
        param_grid: Dict[str, List[Any]],
        cv: int = 5,
        scoring: str = "matthews_corrcoef",
        name: str = "grid_search",
        **metadata: Any
    ) -> Self:
        """Effectue une recherche par grille et enregistre les résultats"""
        
        # Configurer la validation croisée
        if isinstance(cv, int):
            cv = StratifiedShuffleSplit(
                n_splits=cv, 
                test_size=TEST_TRAIN_RATIO, 
                random_state=RANDOM_STATE
            )
        
        # Effectuer la recherche par grille
        grid = RandomizedSearchCV(
            estimator, 
            param_grid, 
            scoring=scoring, 
            cv=cv, 
            n_jobs=-1,
            return_train_score=True
        )

        if self.X_train is None or self.y_train is None:
            raise PipelineException("Cannot train the model since no train sets are defined.")
        grid_fit = grid.fit(self.X_train, self.y_train)

        metadata.update({
            k: v() for k, v in metadata.items() if callable(v)
        })
        metadata.update({
            "best_params": {k: float(v) for k,v in grid_fit.best_params_.items()},
            "best_score": float(grid_fit.best_score_),
            "cv": str(cv),
            "scoring": scoring,
            "cv_results": pd.DataFrame(grid.cv_results_).to_dict()
        })

        self.states.append(DataState(
            X=np.array([]),  # Placeholder vide
            y=np.array([]),  # Placeholder vide
            name=f"{name}",
            metadata=metadata
        ))

        self.last_estimator_grid_search = grid_fit
        return self


    @timer("EVALUATE")
    def __evaluate(
        self, /, *,
        model: BaseEstimator,
        name: str,
        **metadata: Any
    ) -> Self:

        y_pred = model.predict(self.X_test)
        # Calculer les métriques
        mcc = matthews_corrcoef(self.y_test, y_pred)
        acc = accuracy_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred, normalize="true")

        metadata.update({
            k: v() for k, v in metadata.items() if callable(v)
        })
        metadata.update({
            "matthews_corrcoef": float(mcc),
            "accuracy_score": float(acc),
            "confusion_matrix": cm.tolist()
        })

        self.states.append(DataState(
            X=np.array([]),  # Placeholder vide
            y=np.array([]),  # Placeholder vide
            name=f"eval_{name}",
            metadata=metadata
        ))

        return self


    def evaluate_model(
        self, /, *,
        name_model: Optional[str]=None,
        **metadata: Any
    ) -> Self:
        """Évalue un modèle et enregistre les métriques"""
        
        # Prédire les classes
        if name_model is None:
            for n, m in self.models.items():
                self.__evaluate(model=m, name=n, **metadata)
        else:
            if name_model not in self.models:
                raise PipelineException("Unkown model %s", name_model)

            model = next((m for n, m in self.models.items() if n == name_model))
            self.__evaluate(model=model, name=name_model, **metadata)
        
        return self


    def export_database(self, /, *, path: Path) -> Self:
        datastate = next((ds for ds in self.states if ds.name == "initial_data"))
        database = pd.concat([
            pd.DataFrame(datastate.X, columns=self.classes),
            pd.DataFrame(datastate.y, columns=["Letter"])
        ])
        database.to_feather(path)
        return self


    def get_last_estimator_grid_search(self) -> BaseEstimator:
        if self.last_estimator_grid_search is None:
            raise PipelineException("No model was fitted in grid search previously.")
        return self.last_estimator_grid_search
    

    @timer("EXAMPLE")
    def example(
        self, /, *,
        n: int = 10,
        random_state: int = RANDOM_STATE,
        name: str = "example",
        **metadata: Any
    ) -> Self:

        # Prédiction sur un échantillon de 5 données aléatoires
        mask = np.random.default_rng(random_state).choice(self.X_test.shape[0], replace=False, size=n) 
        sample_x = self.X_test[mask]
        sample_y = self.y_test[mask]

        pred_dict: Dict[str, List[str]] = {}
        for model_name, model in self.models.items():
            pred = model.predict(sample_x)
            pred_dict[model_name] = pred.tolist()

        metadata.update({
            k: v() for k, v in metadata.items() if callable(v)
        })
        metadata.update({
            "REAL": sample_y.tolist(),
            **pred_dict
        })
        self.states.append(DataState(
            X=np.array([]),  # Placeholder vide
            y=np.array([]),  # Placeholder vide
            name=f"{name}",
            metadata=metadata
        ))

        return self


    @timer("EXPORT")
    def export_report(
        self, /, *,
        file_path: Optional[Path] = None
    ) -> Self:
        """Exporte un rapport complet sur le pipeline et ses transformations"""
        if file_path is None:
            file_path = REPORT_PATH / f"{self.name}_report.yaml"
        file_path_for_quarto = REPORT_PATH / "report.yaml"

        # Construire le rapport
        report = {
            "pipeline_name": self.name,
            "states": {state.name: state.to_dict() for state in self.states}
        }

        if not os.path.exists(EXPORT_PATH):
            os.makedirs(EXPORT_PATH)
        with open(file_path, "w", encoding="utf-8") as file:
            yaml.dump(report, file)
        with open(file_path_for_quarto, "w", encoding="utf-8") as file:
            yaml.dump(report, file)
            
        return self
