"""
Evaluation modules for GenColorBench.

Task evaluators:
- CNA: Color Name Accuracy
- COA: Color-Object Association  
- MOC: Multi-Object Color Generation
- ICA: Implicit Color Association
- NCU: Numeric Color Understanding
"""

from .base import BaseTaskEvaluator
from .cna import CNAEvaluator, Task1Evaluator
from .ncu import NCUEvaluator, Task2Evaluator
from .coa import COAEvaluator, Task3Evaluator
from .moc import MOCEvaluator, Task4Evaluator
from .ica import ICAEvaluator, Task5Evaluator
from .registry import (
    get_evaluator,
    detect_task,
    detect_color_system,
    detect_color_format,
    parse_category,
    build_object_category_map,
    compute_category_accuracy,
    TASK_EVALUATORS,
    TASK_DISPLAY_NAMES,
    TASK_PATTERNS,
    TASK_ALIASES,
)

__all__ = [
    # Base
    "BaseTaskEvaluator",
    
    # Evaluators (new names)
    "CNAEvaluator",
    "NCUEvaluator", 
    "COAEvaluator",
    "MOCEvaluator",
    "ICAEvaluator",
    
    # Evaluators (backward compatibility)
    "Task1Evaluator",
    "Task2Evaluator",
    "Task3Evaluator",
    "Task4Evaluator",
    "Task5Evaluator",
    
    # Registry functions
    "get_evaluator",
    "detect_task",
    "detect_color_system",
    "detect_color_format",
    "parse_category",
    "build_object_category_map",
    "compute_category_accuracy",
    
    # Registry data
    "TASK_EVALUATORS",
    "TASK_DISPLAY_NAMES",
    "TASK_PATTERNS",
    "TASK_ALIASES",
]
