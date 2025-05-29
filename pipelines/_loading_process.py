
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from common.compute_features import patches_features
from common.data_loading import load_database, load_database_from_save
from common.data_stages import LoadingData, SplitData
from common.types import NDArrayFloat, NDArrayStr
from config import DATABASE_PATH, NB_SHAPE, RANDOM_STATE, SAVED_DATABASE_PATH, TEST_TRAIN_RATIO


# classe contenant les fonctionalités permettant de:
# - charges les données
# - calculer les features
# - appliquer des transformations
# - séparer les données en jeu de données train et test
class LoadingProcess:
    @classmethod
    def __to_data(
        cls, /, *,
        cols: NDArrayStr,
        letters: NDArrayStr,
        data: NDArrayFloat,
        data_size: int
    ) -> LoadingData:

        return LoadingData(
            classes=cols,
            current_X=data,
            current_y=letters,
            data_size=data_size
        )


    @classmethod
    def __load_data_from_files(cls, /, *, nb_shape: int) -> LoadingData:
        raw_data, data_size = load_database(DATABASE_PATH)

        cols, letters, data = patches_features(raw_data, data_size, nb_shape)
        mask_na = np.any(~np.isnan(data), axis=1)
    
        return LoadingProcess.__to_data(
            cols=cols,
            letters=letters[mask_na],
            data=data[mask_na],
            data_size=data_size
        )


    @classmethod
    def __load_data_from_save(cls, /) -> LoadingData:
        cols, letters, data, data_size = load_database_from_save(SAVED_DATABASE_PATH)
        
        return LoadingProcess.__to_data(
            cols=cols,
            letters=letters,
            data=data,
            data_size=data_size
        )


    @classmethod
    def load_data(cls, /, *, from_save: bool = False, nb_shape: int = NB_SHAPE) -> LoadingData:
        """Charge les données initiales dans la pipeline"""
        if from_save:
            loading_data = LoadingProcess.__load_data_from_save()
        else:
            loading_data = LoadingProcess.__load_data_from_files(nb_shape=nb_shape)

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
            new_X, new_y = transformer.transform(data.current_X, data.current_y)
        else:
            new_X = transformer.transform(data.current_X)
            new_y = data.current_y

        return LoadingData(
            data_size=data.data_size,
            classes=data.classes,
            current_X=new_X,
            current_y=new_y,
            transformers={**data.transformers, name: transformer}
        )
    

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