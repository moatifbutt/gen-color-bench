"""
I/O utilities for GenColorBench.
"""

import json
import numpy as np
from pathlib import Path
from typing import Any, Union


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        return super().default(obj)


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file with numpy support.
    
    Args:
        data: Data to save
        path: Output file path
        indent: JSON indentation level
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent, cls=NumpyEncoder)


def load_json(path: Union[str, Path]) -> Any:
    """
    Load data from JSON file.
    
    Args:
        path: Input file path
    
    Returns:
        Loaded data
    """
    with open(path, 'r') as f:
        return json.load(f)
