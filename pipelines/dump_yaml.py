
import os
from pathlib import Path
from typing import Any, Dict, Union
import numpy as np
from sklearn.base import BaseEstimator
import yaml

from common.config import EXPORT_PATH

def setup_yaml_representers() -> None:
    """Setup custom YAML representers for NumPy and scikit-learn objects."""
    
    # Handle NumPy arrays
    def ndarray_representer(dumper: yaml.Dumper, array: np.ndarray[Any, Any]) -> yaml.SequenceNode:
        return dumper.represent_list(array.tolist())
    
    # Handle NumPy scalars (like numpy.float64, numpy.int32, etc.)
    def numpy_scalar_representer(dumper: yaml.Dumper, scalar: np.generic) -> yaml.Node:
        return dumper.represent_data(scalar.item())
    
    # Handle NumPy data types
    def numpy_dtype_representer(dumper: yaml.Dumper, dtype: np.typing.DTypeLike) -> yaml.ScalarNode:
        return dumper.represent_str(str(dtype))
    
    # Handle scikit-learn estimators and other complex objects
    def sklearn_representer(dumper: yaml.Dumper, obj: BaseEstimator) -> Union[yaml.MappingNode, yaml.ScalarNode]:
        # For scikit-learn objects, represent key parameters
        if hasattr(obj, "get_params"):
            params = obj.get_params()
            # Filter out complex nested objects for readability
            simple_params = {}
            for key, value in params.items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    simple_params[key] = value
                elif isinstance(value, (list, tuple)) and len(value) < 10:
                    # Include short lists/tuples
                    simple_params[key] = value
            
            return dumper.represent_dict({
                '__class__': f"{obj.__class__.__module__}.{obj.__class__.__name__}",
                'params': simple_params
            })
        else:
            # Fallback: represent as string
            return dumper.represent_str(str(obj))
    
    # Handle other complex objects with a generic approach
    def generic_object_representer(dumper: yaml.Dumper, obj: Any) -> Union[yaml.MappingNode, yaml.ScalarNode]:
        """Fallback representer for complex objects."""
        try:
            # Try to convert to a simple representation
            if hasattr(obj, '__dict__'):
                simple_dict = {}
                for key, value in obj.__dict__.items():
                    if isinstance(value, (str, int, float, bool, type(None))):
                        simple_dict[key] = value
                return dumper.represent_dict({
                    '__class__': f"{obj.__class__.__module__}.{obj.__class__.__name__}",
                    'attributes': simple_dict
                })
            else:
                return dumper.represent_str(str(obj))
        except Exception:
            return dumper.represent_str(f"<{type(obj).__name__} object>")
    
    # Register representers for NumPy types
    yaml.add_representer(np.ndarray, ndarray_representer)
    yaml.add_representer(np.dtype, numpy_dtype_representer)

    for numpy_type in [np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64,
                        np.float16, np.float32, np.float64,
                        np.complex64, np.complex128,
                        np.bool_]:
        yaml.add_representer(numpy_type, numpy_scalar_representer)

    # Handle numpy.generic (parent class for all numpy scalars)
    yaml.add_representer(np.generic, numpy_scalar_representer)
    
    # Register for scikit-learn estimators
    yaml.add_representer(BaseEstimator, sklearn_representer)


def dump(file_path: Path, file_path_for_quarto: Path, report: Dict[str, Any]):
    setup_yaml_representers()

    if not os.path.exists(EXPORT_PATH):
        os.makedirs(EXPORT_PATH)
    with open(file_path, "w", encoding="utf-8") as file:
        yaml.dump(report, file)
    with open(file_path_for_quarto, "w", encoding="utf-8") as file:
        yaml.dump(report, file)
