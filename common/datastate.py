from typing import Any, Dict
from attrs import define, field

from common.types import NDArrayNum, NDArrayStr

@define
class DataState:
    """Classe pour stocker un état de données et ses métadonnées"""
    X: NDArrayNum
    y: NDArrayStr
    name: str
    metadata: Dict[str, Any] = field(factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'état en dictionnaire pour export"""
        result = {
            "name": self.name,
            "shape": list(self.X.shape),
            "metadata": self.metadata
        }
        if hasattr(self.y, "value_counts"):
            result["class_distribution"] = self.y.value_counts().to_dict()
        return result
