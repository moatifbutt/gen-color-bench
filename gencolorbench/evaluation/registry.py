"""
Evaluator registry for GenColorBench.

Maps task names to evaluator classes and provides task/color system detection.
Includes category parsing from obj_neg.csv.
"""

from typing import Dict, Type, List, Optional, Any
import pandas as pd

# Import evaluators
from .cna import CNAEvaluator
from .ncu import NCUEvaluator
from .coa import COAEvaluator
from .moc import MOCEvaluator
from .ica import ICAEvaluator

# Task display names
TASK_DISPLAY_NAMES = {
    'cna': 'Color Name Accuracy',
    'coa': 'Color-Object Association',
    'moc': 'Multi-Object Color Generation',
    'ica': 'Implicit Color Association',
    'ncu': 'Numeric Color Understanding',
}

# Task patterns for detection from filename
TASK_PATTERNS = {
    'cna': ['cna', 'color_name'],
    'coa': ['coa', 'color_object'],
    'moc': ['moc', 'multi_object'],
    'ica': ['ica', 'implicit'],
    'ncu': ['ncu', 'numeric'],
}

# Registry mapping task names to evaluator classes
TASK_EVALUATORS: Dict[str, Type] = {
    'cna': CNAEvaluator,
    'coa': COAEvaluator,
    'moc': MOCEvaluator,
    'ica': ICAEvaluator,
    'ncu': NCUEvaluator,
}

# Backward compatibility aliases
TASK_ALIASES = {
    'task1': 'cna',
    'task2': 'ncu',
    'task3': 'coa',
    'task4': 'moc',
    'task5': 'ica',
}


def get_evaluator(task: str) -> Type:
    """
    Get the evaluator class for a given task.
    
    Args:
        task: Task identifier (cna, ncu, coa, moc, ica or task1-5)
    
    Returns:
        Evaluator class
    """
    # Handle backward compatibility
    task_lower = task.lower().strip()
    if task_lower in TASK_ALIASES:
        task_lower = TASK_ALIASES[task_lower]
    
    if task_lower not in TASK_EVALUATORS:
        print(f"Warning: Unknown task '{task}', defaulting to CNAEvaluator")
        return CNAEvaluator
    
    return TASK_EVALUATORS[task_lower]


def detect_task(filename: str) -> str:
    """
    Detect task type from filename.
    
    Args:
        filename: CSV filename
    
    Returns:
        Task identifier (cna, ncu, coa, moc, ica)
    """
    filename_lower = filename.lower()
    
    for task, patterns in TASK_PATTERNS.items():
        if any(p in filename_lower for p in patterns):
            return task
    
    return 'cna'  # Default


def detect_color_system(filename: str) -> str:
    """
    Detect color system from filename.
    
    Args:
        filename: CSV filename
    
    Returns:
        Color system identifier (css, l1, l2, l3)
    """
    filename_lower = filename.lower()
    
    if '_css' in filename_lower or 'css_' in filename_lower:
        return 'css'
    elif '_l3' in filename_lower or 'iscc_l3' in filename_lower or 'rgb_l3' in filename_lower:
        return 'l3'
    elif '_l2' in filename_lower or 'iscc_l2' in filename_lower or 'rgb_l2' in filename_lower:
        return 'l2'
    elif '_l1' in filename_lower or 'iscc_l1' in filename_lower or 'rgb_l1' in filename_lower:
        return 'l1'
    
    return 'l2'  # Default


def detect_color_format(filename: str) -> str:
    """
    Detect color format from filename (for NCU task).
    
    Args:
        filename: CSV filename
    
    Returns:
        Color format ('hex' or 'rgb')
    """
    filename_lower = filename.lower()
    
    if 'hex' in filename_lower:
        return 'hex'
    elif 'rgb' in filename_lower:
        return 'rgb'
    
    return 'rgb'  # Default


def parse_category(dataset_category: str) -> str:
    """
    Parse category from Dataset_Category column in obj_neg.csv.
    
    Format: {Dataset}_{Category}
    Examples:
        - "COCO_Vehicle" → "Vehicle"
        - "ImageNet_furniture_Household" → "furniture_Household"
    
    Args:
        dataset_category: Value from Dataset_Category column
    
    Returns:
        Category name
    """
    if not dataset_category or not isinstance(dataset_category, str):
        return "Unknown"
    
    parts = dataset_category.split('_', 1)
    if len(parts) > 1:
        return parts[1]
    return dataset_category


def build_object_category_map(neg_labels_df: pd.DataFrame) -> Dict[str, str]:
    """
    Build a mapping from object names to categories.
    
    Args:
        neg_labels_df: DataFrame from obj_neg.csv with object and category columns
    
    Returns:
        Dictionary mapping object name (lowercase) to category
    """
    obj_to_category = {}
    
    # Handle different column naming conventions
    obj_col = None
    for col in ['Object', 'Class_Name', 'object', 'class_name']:
        if col in neg_labels_df.columns:
            obj_col = col
            break
    
    cat_col = None
    for col in ['Dataset_Category', 'Category', 'dataset_category', 'category']:
        if col in neg_labels_df.columns:
            cat_col = col
            break
    
    if obj_col is None or cat_col is None:
        print(f"Warning: Could not find object/category columns. Found: {list(neg_labels_df.columns)}")
        return obj_to_category
    
    for _, row in neg_labels_df.iterrows():
        obj_name = str(row.get(obj_col, '')).strip().lower()
        dataset_category = str(row.get(cat_col, ''))
        
        if obj_name:
            category = parse_category(dataset_category)
            obj_to_category[obj_name] = category
    
    return obj_to_category


def compute_category_accuracy(
    results: List[Dict],
    obj_to_category: Dict[str, str],
    object_key: str = 'object'
) -> Dict[str, Dict[str, Any]]:
    """
    Compute accuracy per category.
    
    Args:
        results: List of evaluation result dictionaries
        obj_to_category: Mapping from object name to category
        object_key: Key in result dict for object name ('object', 'main_obj', etc.)
    
    Returns:
        Dictionary with per-category stats:
        {
            "Vehicle": {"correct": 50, "total": 100, "accuracy": 50.0},
            ...
        }
    """
    category_stats = {}
    
    for result in results:
        if not result.get('success', False):
            continue
        
        # Get object name from result
        obj_name = result.get(object_key, '')
        if not obj_name:
            # Try alternative keys
            obj_name = result.get('object', result.get('main_obj', ''))
        
        obj_name_lower = str(obj_name).strip().lower()
        category = obj_to_category.get(obj_name_lower, 'Unknown')
        
        if category not in category_stats:
            category_stats[category] = {'correct': 0, 'total': 0}
        
        category_stats[category]['total'] += 1
        if result.get('correct', False):
            category_stats[category]['correct'] += 1
    
    # Compute accuracy for each category
    for category in category_stats:
        total = category_stats[category]['total']
        correct = category_stats[category]['correct']
        category_stats[category]['accuracy'] = (correct / total * 100) if total > 0 else 0.0
    
    return category_stats
