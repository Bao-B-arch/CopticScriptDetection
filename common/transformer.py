
from typing import Dict, List, Optional, Self, Tuple
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted
from sklearn.feature_selection import RFE, SelectorMixin

from config import LETTER_TO_REMOVE
from common.types import NDArrayBool, NDArrayNum


# sélecteur personnalisé scikit learn qui ne fait rien
# sert à comparer les différents sélecteur de features au fait de prendre toutes les features
class DoNothingSelector(SelectorMixin, BaseEstimator):
    """Transformateur Nothing"""

    def fit(self, X: NDArrayNum, y: Optional[NDArrayNum]=None) -> Self:
        self.scores_ = np.ones(X.shape[1], dtype=bool)
        return self


    def _get_support_mask(self) -> NDArrayBool:
        check_is_fitted(self)
        return self.scores_


# transformer personnalisé qui masque certaines lettres
class LetterRemover(TransformerMixin, BaseEstimator):
    """Transformateur pour supprimer des lettres spécifiques du jeu de données"""
    
    def __init__(self, letters_to_remove: Optional[List[str]]=None):
        self.letters_to_remove = letters_to_remove or LETTER_TO_REMOVE
        self.removed_counts: Dict[str, int] = {}
        self.mask: Optional[NDArrayBool] = None

    def fit(self, X: NDArrayNum, y: Optional[NDArrayNum]=None) -> Self:
        if y is not None:
            # Compter les occurrences à supprimer
            for letter in self.letters_to_remove:
                self.removed_counts[letter] = int(np.sum(y == letter))
            self.mask = np.isin(y, self.letters_to_remove, invert=True)
        return self
    
    def transform(self, X: NDArrayNum, y: Optional[NDArrayNum]=None) -> Tuple[NDArrayNum, ...]:
        if self.mask is not None:
            if y is not None:
                return X[self.mask, :], y[self.mask]
            return (X[self.mask, :], )
        
        if y is not None:
            return X, y
        return (X, )
    
    def fit_transform(self, X: NDArrayNum, y: Optional[NDArrayNum]=None) -> Tuple[NDArrayNum, ...]:
        return self.fit(X, y).transform(X)
    
    def get_removed_count(self) -> Dict[str, int]:
        return self.removed_counts


def get_sorted_idx(transformer: BaseEstimator) -> NDArrayNum:
    if isinstance(transformer, RFE):
        scores = (1 / transformer.ranking_)
    elif hasattr(transformer, "scores_"):
        scores = transformer.scores_
    elif hasattr(transformer, "explained_variance_ratio_"):
        scores = transformer.explained_variance_ratio_
    elif hasattr(transformer, "estimator_") & hasattr(transformer.estimator_, "coef_"):
        scores = np.mean(np.abs(transformer.estimator_.coef_), axis=0)
    else:
        raise ValueError("Cannot find any scores related to this transformer %s", transformer)
    
    return np.argsort(scores)[::-1]
