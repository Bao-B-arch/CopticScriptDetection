
from typing import Dict, List, Optional, Self, Tuple
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from common.config import LETTER_TO_REMOVE
from common.types import NDArrayBool, NDArrayNum


class LetterRemover(BaseEstimator, TransformerMixin):
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
