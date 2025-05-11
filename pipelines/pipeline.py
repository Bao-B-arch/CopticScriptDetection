from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional, Self, Union

from attr import define, field
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit, train_test_split
import yaml

from common import compute_features, data_loading
from common.config import DATABASE_PATH, EXPORT_PATH, FORCE_REPORT, GRAPH_PATH, NB_SHAPE, \
    RANDOM_STATE, REPORT_PATH, SAVED_DATABASE_PATH, TEST_TRAIN_RATIO
from common.datastate import DataState
from common.graph_utils import visualize_confusion_matrix, visualize_correlation, visualize_grid_search, visualize_scaling, visualize_train_test_split
from common.types import NDArrayNum, NDArrayStr
from common.utils import outlier_analysis, unwrap
from pipelines.data_stages import LoadingData, SplitData
from pipelines.decorator import timer_pipeline


class PipelineException(Exception):
    pass

class MissingConfigValue(ValueError):
    pass


class LoadingProcess:
    

    @classmethod
    def __to_data(cls, /, *, data: pd.DataFrame, target_name: str, data_size: int) -> LoadingData:
        classes = data.columns.drop([target_name]).to_numpy()
        return LoadingData(
            classes=classes,
            current_X=data.loc[:, classes].to_numpy(),
            current_y=data.loc[:, target_name].to_numpy(),
            data_size=data_size
        )


    @classmethod
    def __load_data_from_files(cls, /, *, nb_shape: int, target_name: str) -> LoadingData:
        raw_data, data_size = data_loading.load_database(DATABASE_PATH)
        data = compute_features.mean_grayscale(raw_data, data_size, nb_shape)
        data = data.dropna(axis=0)
        
        return LoadingProcess.__to_data(data=data, target_name=target_name, data_size=data_size)


    @classmethod
    def __load_data_from_save(cls, /, *, target_name: str) -> LoadingData:
        data, data_size = data_loading.load_database_from_save(SAVED_DATABASE_PATH)
        
        return LoadingProcess.__to_data(data=data, target_name=target_name, data_size=data_size)


    @classmethod
    def load_data(cls, /, *, target_name: str, from_save: bool = False, nb_shape: int = NB_SHAPE) -> LoadingData:
        """Charge les données initiales dans la pipeline"""
        if from_save:
            loading_data = LoadingProcess.__load_data_from_save(target_name=target_name)
        else:
            loading_data = LoadingProcess.__load_data_from_files(nb_shape=nb_shape, target_name=target_name)

        return loading_data

    @classmethod
    def transform(
        cls, /, *,
        data: LoadingData,
        transformer: BaseEstimator,
        name: str,
        fit: bool = True,
        transform_y: bool = False,
        apply_y: bool = False
    ) -> LoadingData:
        """Applique une transformation aux données avec traçabilité"""

        # Appliquer la transformation
        if fit:
            if transform_y:
                transformer.fit(data.current_X, data.current_y)
            else:
                transformer.fit(data.current_X)

        if apply_y:
            data.current_X, data.current_y = transformer.transform(data.current_X, data.current_y)
        else:
            data.current_X = transformer.transform(data.current_X)

        # Stocker le transformateur
        data.transformers[name] = transformer
        return data
    

    @classmethod
    def split(
        cls, /, *,
        data: LoadingData,
        test_size: float = TEST_TRAIN_RATIO,
        random_state: int = RANDOM_STATE,
        stratify: bool = True
    ) -> SplitData:
        """Divise les données en ensembles d'entraînement et de test"""

        # Effectuer la séparation
        stratify_param = data.current_y if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            data.current_X, data.current_y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=stratify_param
        )

        return SplitData(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )


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
        data.models[name] = (None, model)

        return data
    

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
        
        # Configurer la validation croisée
        if isinstance(cv, int):
            cv = StratifiedShuffleSplit(
                n_splits=cv, 
                test_size=TEST_TRAIN_RATIO, 
                random_state=RANDOM_STATE
            )

        # Effectuer la recherche par grille
        search = RandomizedSearchCV(
            model, 
            param_grid, 
            scoring=scoring, 
            cv=cv, 
            n_jobs=-1,
            return_train_score=True
        )
        search_fit = search.fit(data.X_train, data.y_train)

        searched_model = model.set_params(**search_fit.best_params_)
        searched_model.fit(data.X_train, data.y_train)
        data.models[name] = (search_fit, searched_model)
        return data


    @classmethod
    def evaluate(
        cls, /, *,
        data: SplitData,
        model: BaseEstimator
    ) -> Dict[str, Union[float, List[float]]]:

        y_pred = model.predict(data.X_test)
        # Calculer les métriques
        mcc = matthews_corrcoef(data.y_test, y_pred)
        acc = accuracy_score(data.y_test, y_pred)
        cm = confusion_matrix(data.y_test, y_pred, normalize="true")

        return {
            "mcc": float(mcc),
            "acc": float(acc),
            "cm": cm.tolist(),
        }
    

    @classmethod
    def example(
        cls, /, *,
        data: SplitData,
        n: int = 10,
        random_state: int = RANDOM_STATE,
    ) -> Dict[str, List[str]]:

        # Prédiction sur un échantillon de 5 données aléatoires
        mask = np.random.default_rng(random_state).choice(data.X_test.shape[0], replace=False, size=n) 
        sample_x = data.X_test[mask]
        sample_y = data.y_test[mask]

        pred_dict: Dict[str, List[str]] = {}
        pred_dict["REAL"] = sample_y.tolist()

        for model_name, (_, model) in data.models.items():
            pred = model.predict(sample_x)
            pred_dict[model_name] = pred.tolist()

        return pred_dict

@define(kw_only=True)
class TrackedPipeline:
    name: str
    nb_shapes: int
    selection: bool
    loading_data: Optional[LoadingData] = field(init=False, default=None)
    split_data: Optional[SplitData] = field(init=False, default=None)
    states: List[DataState] = field(init=False, factory=list)


    @classmethod
    def from_config(cls, name: str, **config: Any) -> Self:

        if "nb_shapes" not in config or not isinstance(config["nb_shapes"], int):
            raise MissingConfigValue("Given %s but nb_shapes as int expected.", config)
        if "selection" not in config or not isinstance(config["selection"], bool):
            raise MissingConfigValue("Given %s but selection as bool expected.", config)

        nb_shapes: int = config["nb_shapes"]
        selection: bool = config["selection"]
        _name = f"{name}_{nb_shapes}_{selection}"
        print(f"Starting {_name}...")
        return cls(name=_name, nb_shapes=nb_shapes, selection=selection)


    @timer_pipeline("LOADING DATABASE")
    def load_data(self, /, *, target_name: str, name: str = "initial_data", from_save: bool = False) -> Self:
    
        self.loading_data = LoadingProcess.load_data(
            target_name=target_name,
            from_save=from_save,
            nb_shape=self.nb_shapes
        )

        metadata = {
            "SIZE": self.loading_data.data_size,
            "TARGET": pd.DataFrame(self.loading_data.current_y, columns=[target_name]).loc[:, target_name].value_counts().to_dict(),
            "NB_PATCHES": self.nb_shapes,
            "FEATURES": list(self.loading_data.classes),
            "OUTLIERS_ANALYSIS": outlier_analysis(pd.DataFrame(self.loading_data.current_X, columns=self.loading_data.classes)).to_dict(),
        }

        self.add_state(
            X=self.loading_data.current_X,
            y=self.loading_data.current_y,
            name=name, 
            metadata=metadata
        )
        return self

    
    @timer_pipeline("TRANSFORMER")
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

        self.loading_data = LoadingProcess.transform(
            data=unwrap(self.loading_data),
            transformer=transformer,
            name=name,
            fit=fit,
            transform_y=transform_y,
            apply_y=apply_y,
        )

        metadata.update({
            k: v() for k, v in metadata.items() if callable(v)
        })
        metadata.update({
            "transformer_type": type(transformer).__name__,
            "transformer_params": {k: str(v) for k,v in transformer.get_params().items()}
        })

        # Enregistrer le nouvel état
        self.add_state(
            X=self.loading_data.current_X,
            y=self.loading_data.current_y,
            name=f"after_{name}", 
            metadata=metadata
        )
        return self


    @timer_pipeline("TRAIN_TEST_SPLIT")
    def split(
        self, /, *,
        test_size: float = TEST_TRAIN_RATIO,
        random_state: int = RANDOM_STATE,
        stratify: bool = True,
        **metadata: Any
    ) -> Self:
        """Divise les données en ensembles d'entraînement et de test"""

        loading_data = unwrap(self.loading_data)
        # Effectuer la séparation
        self.split_data = LoadingProcess.split(
            data=loading_data,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

        metadata.update({
            k: v() for k, v in metadata.items() if callable(v)
        })
        metadata.update({
            "TRAIN_SIZE": {
                "value": self.split_data.X_train.shape[0], 
                "percent": self.split_data.X_train.shape[0] / loading_data.current_X.shape[0] * 100
            },
            "TEST_SIZE": {
                "value": self.split_data.X_test.shape[0], 
                "percent": self.split_data.X_test.shape[0] / loading_data.current_X.shape[0] * 100
            },
            "TEST_RATIO": test_size,
            "STRATIFIED": stratify
        })

        # Enregistrer l'état actuel
        self.add_state(
            X=loading_data.current_X,
            y=loading_data.current_y,
            name="train_test_split", 
            metadata=metadata
        )
        return self
    

    @timer_pipeline("TRAIN")
    def train_model(
        self, /, *,
        model: BaseEstimator,
        name: str,
        **metadata:Any
    ) -> Self:
        """Entraîne un modèle et enregistre les métadonnées"""
        
        # Entraîner le modèle
        self.split_data = SplitProcess.train_model(
            data=unwrap(self.split_data),
            model=model,
            name=name,
        )

        metadata.update({
            k: v() for k, v in metadata.items() if callable(v)
        })
        metadata.update({
            "model_type": type(model).__name__,
            "model_params": model.get_params(),
            "training_data_shape": list(self.split_data.X_train.shape)
        })

        self.add_state(
            X=np.array([]),  # Placeholder vide
            y=np.array([]),  # Placeholder vide
            name=f"model_{name}",
            metadata=metadata
        )

        return self


    @timer_pipeline("TUNING HYPER PARAMETERS AND TRAIN")
    def train_model_searched(
        self, /, *, 
        model: BaseEstimator,
        name: str,
        param_grid: Dict[str, List[Any]],
        cv: int = 5,
        scoring: str = "matthews_corrcoef",
        **metadata: Any
    ) -> Self:
        """Effectue une recherche d'hyparamètres, entraîne un modèle et enregistre les métadonnées"""

        self.split_data = SplitProcess.train_model_searched(
            data=unwrap(self.split_data),
            model=model,
            name=name,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring
        )

        metadata.update({
            k: v() for k, v in metadata.items() if callable(v)
        })
        __search, _ = self.split_data.models[name]
        search: RandomizedSearchCV = unwrap(__search)
        metadata.update({
            "model_type": type(model).__name__,
            "cv": str(cv),
            "cv_results": search.cv_results_,
            "scoring": scoring,
            "best_score": search.best_score_,
            "model_params": model.get_params(),
            "training_data_shape": list(self.split_data.X_train.shape)
        })

        self.add_state(
            X=np.array([]),  # Placeholder vide
            y=np.array([]),  # Placeholder vide
            name=f"model_{name}",
            metadata=metadata
        )

        return self
    

    @timer_pipeline("EVALUATE")
    def evaluate(
        self, /, *,
        model: BaseEstimator,
        name: str,
        **metadata: Any
    ) -> Self:

        scores = SplitProcess.evaluate(
            data=unwrap(self.split_data),
            model=model
        )
        metadata.update({
            k: v() for k, v in metadata.items() if callable(v)
        })
        metadata.update({
            "matthews_corrcoef": scores["mcc"],
            "accuracy_score": scores["acc"],
            "confusion_matrix": scores["cm"]
        })

        self.add_state(
            X=np.array([]),  # Placeholder vide
            y=np.array([]),  # Placeholder vide
            name=f"eval_{name}",
            metadata=metadata
        )

        return self


    def evaluate_models(
        self, /, *,
        name_model: Optional[str]=None,
        **metadata: Any
    ) -> Self:
        """Évalue un modèle et enregistre les métriques"""
        
        split_data = unwrap(self.split_data)
        # Prédire les classes
        if name_model is None:
            for n, (_, m) in split_data.models.items():
                self.evaluate(model=m, name=n, **metadata)
        else:
            if name_model not in split_data.models:
                raise PipelineException("Unkown model %s", name_model)

            model = next((m for n, m in split_data.models.items() if n == name_model))
            self.evaluate(model=model, name=name_model, **metadata)

        return self
    
    
    @timer_pipeline("EXAMPLE")
    def example(
        self, /, *,
        n: int = 10,
        random_state: int = RANDOM_STATE,
        name: str = "example",
        **metadata: Any
    ) -> Self:

        pred_dict: Dict[str, List[str]] = SplitProcess.example(
            data=unwrap(self.split_data),
            n=n,
            random_state=random_state
        )

        metadata.update({
            k: v() for k, v in metadata.items() if callable(v)
        })
        metadata.update({
            **pred_dict
        })

        self.add_state(
            X=np.array([]),  # Placeholder vide
            y=np.array([]),  # Placeholder vide
            name=f"{name}",
            metadata=metadata
        )

        return self


    def add_state(self, name: str, X: NDArrayNum, y: NDArrayStr, metadata: Optional[Dict[str, Any]] = None) -> Self:
        """Ajoute l'état actuel des données à l'historique"""
        if metadata is None:
            metadata = {}

        if X is not None and y is not None:
            self.states.append(DataState(
                X=X,
                y=y,
                name=name,
                metadata=metadata
            ))

        return self
    

    def get_state(self, /, *, name: str) -> DataState:
        return next((s for s in self.states if name in s.name))


    def add_time(self, name: str, run_time: float) -> Self:
        {
            "execution_time": run_time,
            "step": name,
        }
        return self


    @timer_pipeline("EXPORT DATABASE")
    def export_database(self, /, *, export: bool, path: Path) -> Self:
        if export:
            datastate = next((ds for ds in self.states if ds.name == "initial_data"))
            loading_data = unwrap(self.loading_data)
            database = pd.concat([
                pd.DataFrame(datastate.X, columns=loading_data.classes),
                pd.DataFrame(datastate.y, columns=["Letter"])
            ])
            database.to_feather(path)
        return self

    
    @timer_pipeline("EXPORT REPORT")
    def export_report(
        self, /, *,
        file_path: Optional[Path] = None
    ) -> Self:
        """Exporte un rapport complet sur le pipeline et ses transformations"""
        if file_path is None:
            file_path = REPORT_PATH / f"{self.name}_report.yaml"
        file_path_for_quarto = REPORT_PATH / "report.yaml"
        loading_data = unwrap(self.loading_data)

        # Construire le rapport
        report = {
            "pipeline_name": self.name,
            "transformers": loading_data.transformers,
            "states": {state.name: state.to_dict() for state in self.states}
        }

        if not os.path.exists(EXPORT_PATH):
            os.makedirs(EXPORT_PATH)
        with open(file_path, "w", encoding="utf-8") as file:
            yaml.dump(report, file)
        with open(file_path_for_quarto, "w", encoding="utf-8") as file:
            yaml.dump(report, file)

        return self
    

    @timer_pipeline("EXPORT GRAPH")
    def export_graphes(self, /, *, graph_folder: Optional[Path]=None, force_plot: bool=False) -> Self:

        if graph_folder is None:
            graph_folder = GRAPH_PATH / f"graphs_{self.nb_shapes}_{self.selection}"
        graph_folder_for_quarto = Path("graphs")
        if not os.path.exists(graph_folder):
            os.makedirs(graph_folder)
        if not os.path.exists(graph_folder_for_quarto):
            os.makedirs(graph_folder_for_quarto)

        db_before_scaling = self.get_state(name="letter_to_remove")
        db_after_scaling = self.get_state(name="normalisation")
        db_grid = self.get_state(name="search_svm")

        loading_data = unwrap(self.loading_data)
        split_data = unwrap(self.split_data)
        labels: NDArrayStr = np.unique(loading_data.current_y)

        if self.nb_shapes < 10:
            visualize_scaling(graph_folder, graph_folder_for_quarto, db_before_scaling.X, db_after_scaling.X)
            visualize_correlation(graph_folder, graph_folder_for_quarto, db_after_scaling.X, loading_data.classes)

        visualize_train_test_split(graph_folder, graph_folder_for_quarto, split_data.y_train, split_data.y_test)
        for model_name, metadata in ((s.name, s.metadata) for s in self.states if "eval" in s.name):
            visualize_confusion_matrix(graph_folder, graph_folder_for_quarto, metadata["confusion_matrix"], labels, model_name)

        unique_C, inv_C = np.unique(db_grid.metadata["cv_results"]["param_C"], return_inverse=True)
        unique_gamma, inv_gamma = np.unique(db_grid.metadata["cv_results"]["param_gamma"], return_inverse=True)
        grid = np.zeros((len(unique_C), len(unique_gamma)))
        grid[inv_C, inv_gamma] = db_grid.metadata["cv_results"]["mean_test_score"]

        visualize_grid_search(graph_folder, graph_folder_for_quarto, grid, unique_C, unique_gamma, "SVC")

        plt.tight_layout()
        if force_plot:
            plt.show()

        return self


    @timer_pipeline("BUILD REPORT")
    def build_quarto(self, /) -> Self:
        test_quarto = subprocess.run(["quarto", "--version"], stdout = subprocess.DEVNULL)
        if FORCE_REPORT & (test_quarto.returncode == 0):
            subprocess.run(["quarto", "render", ".\coptic_report.qmd", 
                            "--to", "pdf,revealjs",
                            "--output", f"OCR_{self.nb_shapes}_{self.selection}",
                            "--output-dir", "report\quarto"], 
                            stdout = subprocess.DEVNULL,
                            stderr = subprocess.DEVNULL)
        return self