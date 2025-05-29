
import os
from pathlib import Path
from typing import Any, Dict, Tuple, Union
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.calibration import LinearSVC
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import SelectFromModel
import yaml

from config import EXPORT_PATH
from common.transformer import DoNothingSelector, LetterRemover

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
        if not hasattr(obj, '__module__') or not obj.__module__ or 'sklearn' not in obj.__module__:
            raise ValueError("Cant Identify the current object {obj}")

        if hasattr(obj, "get_params"):
            params = obj.get_params()
            # Filter out complex nested objects for readability
            simple_params = {}
            for key, value in params.items():
                if key == "estimator":
                    continue
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
            return dumper.represent_str(f"{obj.__class__.__module__}.{obj.__class__.__name__}")
        
    def letter_remover_representer(dumper: yaml.Dumper, obj: LetterRemover) -> yaml.MappingNode:
        return dumper.represent_dict({
            '__class__': f"{obj.__class__.__module__}.{obj.__class__.__name__}",
            'letters_to_remove': obj.letters_to_remove
        })
    
    def do_nothing_representer(dumper: yaml.Dumper, obj: DoNothingSelector) -> yaml.MappingNode:
        return dumper.represent_dict({
            '__class__': f"{obj.__class__.__module__}.{obj.__class__.__name__}"
        })

    def numpy_ma_representer(dumper: yaml.Dumper, ma_array: np.ma.MaskedArray[Any, Any]) -> Union[yaml.ScalarNode, yaml.SequenceNode]:
        """Handle numpy masked arrays - represent as simple arrays."""
        try:
            # Convert masked array to regular array/list
            if hasattr(ma_array, 'filled'):
                # Use filled() to get array with masked values filled
                filled_array = ma_array.filled()
                return dumper.represent_list(filled_array.tolist())
            elif hasattr(ma_array, 'data'):
                # Fallback to just the data portion
                return dumper.represent_list(ma_array.data.tolist())
            else:
                # Last resort - convert to string
                return dumper.represent_str(str(ma_array))
        except Exception as e:
            return dumper.represent_str(f"<numpy.ma object - serialization error: {str(e)}>")

    def tuple_representer(dumper: yaml.Dumper, tup: Tuple[Any, ...]) -> Union[yaml.ScalarNode, yaml.SequenceNode]:
        """Handle tuples (including numpy shapes)."""
        try:
            # Convert to list for cleaner YAML representation
            return dumper.represent_list(list(tup))
        except Exception:
            return dumper.represent_str(str(tup))
    
    # Register representers for NumPy types
    yaml.add_representer(np.ndarray, ndarray_representer)
    yaml.add_representer(np.dtype, numpy_dtype_representer)
    yaml.add_representer(np.generic, numpy_scalar_representer)
    yaml.add_representer(tuple, tuple_representer)
    
    # Register for scikit-learn estimators
    yaml.add_representer(SelectFromModel, sklearn_representer)
    yaml.add_representer(StandardScaler, sklearn_representer)
    yaml.add_representer(LinearSVC, sklearn_representer)
    yaml.add_representer(LetterRemover, letter_remover_representer)
    yaml.add_representer(DoNothingSelector, do_nothing_representer)
    yaml.add_representer(np.ma.MaskedArray, numpy_ma_representer)

    # Add multi-representers for better coverage
    yaml.add_multi_representer(np.generic, numpy_scalar_representer)
    yaml.add_multi_representer(np.ndarray, ndarray_representer)
    yaml.add_multi_representer(np.generic, numpy_scalar_representer)
    yaml.add_multi_representer(tuple, tuple_representer)


def dump(file_path: Path, report: Dict[str, Any]):
    setup_yaml_representers()

    if not os.path.exists(EXPORT_PATH):
        os.makedirs(EXPORT_PATH)
    with open(file_path, "w", encoding="utf-8") as file:
        yaml.dump(report, file, default_flow_style=False, allow_unicode=True, width=float('inf'), default_style=None)
