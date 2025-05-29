from __future__ import annotations

from itertools import combinations
import os
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional, Self

from attr import define, field
from matplotlib import pyplot as plt
import numpy as np
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from common.compute_features import export_visual_features
from config import EXPORT_PATH, FACTOR_SIZE_EXPORT, FORCE_PLOT, FORCE_REPORT, GRAPH_PATH, \
    RANDOM_STATE, REPORT_PATH, TEST_TRAIN_RATIO
from common.datastate import DataState
from common.graph_utils import visualize_confusion_matrix, visualize_correlation, visualize_grid_search, visualize_scaling, visualize_train_test_split
from common.transformer import get_sorted_idx
from common.types import NDArrayBool, NDArrayNum, NDArrayStr, NDArrayUInt
from common.utils import jaccard_index, outlier_analysis, unwrap
from common import dump_yaml
from common.data_stages import LoadingData, SplitData
from pipelines._loading_process import LoadingProcess
from pipelines._split_process import SplitProcess
from pipelines.decorator import timer_pipeline


class PipelineException(Exception):
    pass


class MissingConfigValue(ValueError):
    pass


# classe gérant les différentes étapes de la construction d'un modèle de ML
# Chaque étape enregistre des informations qui seront exportées pour analyser la comportement du modèle
@define(kw_only=True)
class TrackedPipeline:
    name: str
    nb_shapes: int
    selection: str
    features_selected: Optional[NDArrayUInt] = field(init=False, default=None)
    loading_data: Optional[LoadingData] = field(init=False, default=None)
    split_data: Optional[SplitData] = field(init=False, default=None)
    states: List[DataState] = field(init=False, factory=list)


    @classmethod
    def from_config(cls, name: str, **config: Any) -> Self:

        if "nb_shapes" not in config or not isinstance(config["nb_shapes"], int):
            raise MissingConfigValue("Given %s but nb_shapes as int expected.", config)
        if "selection" not in config or not isinstance(config["selection"], str):
            raise MissingConfigValue("Given %s but selection as str expected.", config)

        nb_shapes: int = config["nb_shapes"]
        selection: str = config["selection"]
        _name = f"{name}_{nb_shapes}_{selection}"
        print(f"Starting {_name}...")
        return cls(name=_name, nb_shapes=nb_shapes, selection=selection)


    @timer_pipeline("LOADING DATABASE")
    def load_data(self, /, *, name: str = "initial_data", from_save: bool = False) -> Self:
    
        self.loading_data = LoadingProcess.load_data(
            from_save=from_save,
            nb_shape=self.nb_shapes
        )

        metadata = {
            "SIZE": self.loading_data.data_size,
            "TARGET": {str(k): int(v) for (k, v) in zip(*np.unique(self.loading_data.current_y, return_counts=True))},
            "NB_PATCHES": self.nb_shapes,
            "FEATURES": self.loading_data.classes,
            "OUTLIERS_ANALYSIS": outlier_analysis(self.loading_data.current_X),
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
            "transformer_params": transformer.get_params()
        })

        # Enregistrer le nouvel état
        self.add_state(
            X=self.loading_data.current_X,
            y=self.loading_data.current_y,
            name=f"after_{name}", 
            metadata=metadata
        )

        if "SELECTED_FEATURES" in metadata:
            self.features_selected = metadata["SELECTED_FEATURES"]
        return self


    @timer_pipeline("TEST TRANSFORMER")
    def test_transform(
        self, /, *,
        transformer: BaseEstimator,
        name: str,
        model: BaseEstimator,
        cv: StratifiedKFold
    ) -> Dict[str, Any]:

        loading_data = unwrap(self.loading_data)
        X = loading_data.current_X
        y = loading_data.current_y

        n_features = X.shape[1]
        n_splits = cv.get_n_splits()
    
        mcc_sum_progression: NDArrayNum = np.zeros(n_features)
        mcc_sumsq_progression: NDArrayNum = np.zeros(n_features)
        all_selected: NDArrayNum = np.zeros((n_splits, n_features))
        all_convergence: NDArrayBool = np.zeros(n_splits, dtype=np.bool)

        for nfold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train = X[train_idx]
            X_test = X[test_idx]
            ## cloning the transformer to ensure previous loops does not affect the transformer state
            transformer_clone = clone(transformer)
            transformer_clone.fit(X_train, y[train_idx])
            ## computing scores for the whole fold

            sorted_idx = get_sorted_idx(transformer_clone)
            all_selected[nfold] = sorted_idx

            X_train_trans = transformer_clone.transform(X_train)
            X_test_trans = transformer_clone.transform(X_test)

            for k in range(1, n_features + 1):
                model_clone = clone(model)
                X_train_k = X_train_trans[:, sorted_idx[:k]]
                X_test_k = X_test_trans[:, sorted_idx[:k]]

                model_clone.fit(X_train_k, y[train_idx])
                y_pred = model_clone.predict(X_test_k)
                mcc = matthews_corrcoef(y[test_idx], y_pred)
                mcc_sum_progression[k-1] += mcc
                mcc_sumsq_progression[k-1] += mcc*mcc

            if hasattr(transformer_clone, "estimator_") and hasattr(transformer_clone.estimator_, "n_iter_"):
                all_convergence[nfold] = transformer_clone.estimator_.n_iter_ < transformer_clone.estimator_.max_iter
            else:
                all_convergence[nfold] = True

        stability = [
            np.mean([jaccard_index(c1, c2) for c1, c2 in combinations(all_selected[:, :k], 2)])
            for k in range(n_features)
        ]

        result = {
            "name": name,
            "transformer_params": transformer.get_params(),
            "mcc_mean_progression": (mcc_sum_progression / n_splits).tolist(),
            "mcc_var_progression": (mcc_sumsq_progression / n_splits - (mcc_sum_progression / n_splits)**2).tolist(),
            "stability_scores_progression": stability,
            "convergence": sum(all_convergence) / n_splits,
        }
        return result


    def test_transforms(
        self, /, *,
        transformers: Dict[str, BaseEstimator],
        model: BaseEstimator,
        name: str,
        n_fold: int,
        **metadata: Any
    ) -> Self:
        """
        Entraine un modèle en ajoutant progressivement tous les features transformées par les transformateurs
        afin de tester les performances de tous les transformateurs
        """

        result: Dict[str, Any] = {}
        cv = StratifiedKFold(n_splits=n_fold)

        for n, transformer in transformers.items():
            result[n] = self.test_transform(
                transformer=transformer,
                name=n,
                model=model,
                cv=cv
            )

        metadata.update({
            k: v() for k, v in metadata.items() if callable(v)
        })
        metadata.update(result)

        # Enregistrer le nouvel état
        loading_data = unwrap(self.loading_data)
        self.add_state(
            X=loading_data.current_X,
            y=loading_data.current_y,
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
            "training_data_shape": self.split_data.X_train.shape
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
        search: GridSearchCV = unwrap(__search)
        metadata.update({
            "model_type": type(model).__name__,
            "cv": str(cv),
            "cv_results": search.cv_results_,
            "scoring": scoring,
            "best_score": search.best_score_,
            "model_params": model.get_params(),
            "training_data_shape": self.split_data.X_train.shape
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

        pred_dict: Dict[str, NDArrayStr] = SplitProcess.example(
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

        self.states.append(DataState(
            X=X,
            y=y,
            name=name,
            metadata=metadata
        ))

        return self
    

    def get_state(self, /, *, name: str) -> DataState:
        return next((s for s in self.states if name in s.name))


    @timer_pipeline("EXPORT DATABASE")
    def export_database(self, /, *, export: bool, path: Path) -> Self:
        if export:
            datastate = self.get_state(name="initial")
            loading_data = unwrap(self.loading_data)
            np.savez(
                path,
                cols=loading_data.classes,
                letters=datastate.y,
                data=datastate.X,
                allow_pickle=False
            )
        return self

    
    @timer_pipeline("EXPORT REPORT")
    def export_report(
        self, /, *,
        file_path: Optional[Path] = None
    ) -> Self:
        """Exporte un rapport complet sur le pipeline et ses transformations"""
        if file_path is None:
            file_path = REPORT_PATH / f"{self.name}_report.yaml"
        else:
            file_path = REPORT_PATH / file_path
        loading_data = unwrap(self.loading_data)

        # Construire le rapport
        report: Dict[str, Any] = {
            "pipeline_name": self.name,
            "transformers": loading_data.transformers,
            "states": {state.name: state.to_dict() for state in self.states}
        }

        dump_yaml.dump(file_path, report)

        return self
    

    @timer_pipeline("EXPORT GRAPH")
    def export_graphes(self, /, *, graph_folder: Optional[Path]=None, force_plot: bool=FORCE_PLOT) -> Self:

        if graph_folder is None:
            graph_folder = GRAPH_PATH / f"graphs_{self.nb_shapes}_{self.selection}"
        if not os.path.exists(graph_folder):
            os.makedirs(graph_folder)

        db_before_scaling = self.get_state(name="letter_to_remove")
        db_after_scaling = self.get_state(name="normalisation")
        db_split = self.get_state(name="train_test_split")
        db_grid = self.get_state(name="search_svm")

        loading_data = unwrap(self.loading_data)
        split_data = unwrap(self.split_data)
        labels: NDArrayStr = np.unique(loading_data.current_y)

        if self.nb_shapes < 20:
            visualize_scaling(graph_folder, db_before_scaling.X, db_after_scaling.X)
        if db_split.X.shape[1] < 10:
            classes = loading_data.classes
            if self.features_selected is not None:
                classes = classes[self.features_selected]
            visualize_correlation(graph_folder, db_split.X, classes)

        visualize_train_test_split(graph_folder, split_data.y_train, split_data.y_test)
        for model_name, metadata in ((s.name, s.metadata) for s in self.states if "eval" in s.name):
            visualize_confusion_matrix(graph_folder, metadata["confusion_matrix"], labels, model_name)

        unique_C, inv_C = np.unique(db_grid.metadata["cv_results"]["param_C"], return_inverse=True)
        unique_gamma, inv_gamma = np.unique(db_grid.metadata["cv_results"]["param_gamma"], return_inverse=True)
        grid = np.zeros((len(unique_C), len(unique_gamma)))
        grid[inv_C, inv_gamma] = db_grid.metadata["cv_results"]["mean_test_score"]

        visualize_grid_search(graph_folder, grid, unique_C, "C", unique_gamma, "gamma", "SVC")

        plt.tight_layout()
        if force_plot:
            plt.show()
        else:
            plt.close()

        return self
    

    @timer_pipeline("EXPORT GRAPH")
    def export_selection_graphes(self, /, *, graph_folder: Optional[Path]=None, force_plot: bool=FORCE_PLOT) -> Self:

        if graph_folder is None:
            graph_folder = GRAPH_PATH / f"graphs_{self.nb_shapes}_{self.selection}"

        test_selection_state = self.get_state(name="test_selection_features").metadata
        metadata = test_selection_state.metadata
    
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        for n, res in metadata.items():
            if res["convergence"] < 1.0:
                continue
            plt.errorbar(range(test_selection_state.X.shape[1]), res["mcc_mean_progression"], yerr=np.sqrt(res["mcc_var_progression"]), label=n)
        plt.title("Performance MCC")
        plt.legend()

        plt.subplot(122)
        for n, res in metadata.items():
            if res["convergence"] < 1.0:
                continue
            plt.plot(res["stability_scores_progression"], label=f"Jaccard {n}")
        plt.title("Stabilité des Sélections")
        plt.legend()

        plt.tight_layout()

        if not os.path.exists(graph_folder):
            os.makedirs(graph_folder)
        plt.savefig(graph_folder / "selection.svg")
        if force_plot:
            plt.show()
        else:
            plt.close()

        return self


    @timer_pipeline("EXPORT VISUAL FEATURES")
    def export_visual_features(self, /, *, export_path: Optional[Path] = None) -> Self:
        if export_path is None:
            export_path = EXPORT_PATH / f"{self.nb_shapes}patches"
        state = self.get_state(name="initial")
        unique, idx = np.unique(state.y, return_index=True)
        export_visual_features(export_path, state.X[idx], unique, self.nb_shapes, FACTOR_SIZE_EXPORT)
        return self


    @timer_pipeline("BUILD REPORT")
    def build_quarto(self, /) -> Self:
        if FORCE_REPORT:
            try:
                subprocess.run(["quarto", "render", "coptic_report.qmd", 
                                "--to", "pdf,revealjs,docx",
                                "--output", f"OCR_{self.nb_shapes}_{self.selection}",
                                "--output-dir", "report\\quarto",
                                "-P", f"name:{self.nb_shapes}_{self.selection}"], 
                                stdout = subprocess.DEVNULL,
                                stderr = subprocess.DEVNULL)
            except subprocess.CalledProcessError as e:
                print(f"Échec de génération du rapport Quarto: {e}")
        return self
