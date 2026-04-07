"""
GenColorBench: Comprehensive Color Understanding Benchmark for T2I Models (v3).

A diagnostic benchmark for evaluating color generation capabilities in
text-to-image (T2I) models across five tasks:

Tasks (v3 naming):
    CNA  - Color Name Accuracy: "A red apple"
    NCU  - Numeric Color Understanding: "A ball in rgb(255,0,0)"
    COA  - Color-Object Association: "A red apple on a white plate"
    MOC  - Multi-Object Color Composition: "A red apple and a blue car"
    ICA  - Implicit Color Association: "A red apple next to a fire truck"

Color Systems:
    - ISCC-NBS (L1: 13 basic, L2: 29 intermediate, L3: 267 fine-grained)
    - CSS3/X11 (148 web colors)

Usage:
    # Generate mini benchmark
    python -m gencolorbench mini --output-dir ./prompts --seed 42
    
    # Generate images
    python -m gencolorbench images --model flux-dev --prompts-dir ./prompts
    
    # Run evaluation
    python -m gencolorbench evaluate --prompts-dir ./prompts --images-dir ./images

API:
    from gencolorbench import generate_mini_benchmark, generate_full_benchmark
    from gencolorbench import EvalConfig, run_evaluation
"""

__version__ = "3.0.0"

# Core configuration
from .config import EvalConfig, parse_args

# Evaluation pipeline
from .pipeline import run_evaluation, process_csv, compute_category_accuracy

# Benchmark generation
from .mini_benchmark import (
    generate_mini_benchmark,
    MiniConfig,
    GenerationResult,
    TASK_DISPLAY_NAMES,
)
from .full_benchmark import (
    generate_full_benchmark,
    FullBenchmarkConfig,
)

# Data loaders
from .loaders import (
    load_color_system,
    load_objects,
    load_task5_templates,
    Color,
    ObjectItem,
    Task5Template,
)

# Generation utilities
from .generate import (
    get_task_from_csv,
    get_color_system_from_csv,
    NEW_TO_OLD_CSV,
    OLD_TO_NEW_CSV,
    SUPPORTED_MODELS,
)

# Task name aliases (v3 <-> legacy)
TASK_ALIASES = {
    # v3 -> legacy
    'cna': 'task1',
    'ncu': 'task2',
    'coa': 'task3',
    'moc': 'task4',
    'ica': 'task5',
    # legacy -> v3
    'task1': 'cna',
    'task2': 'ncu',
    'task3': 'coa',
    'task4': 'moc',
    'task5': 'ica',
}


__all__ = [
    # Version
    '__version__',
    # Config
    'EvalConfig',
    'parse_args',
    # Pipeline
    'run_evaluation',
    'process_csv',
    'compute_category_accuracy',
    # Benchmark generation
    'generate_mini_benchmark',
    'generate_full_benchmark',
    'MiniConfig',
    'FullBenchmarkConfig',
    'GenerationResult',
    'TASK_DISPLAY_NAMES',
    # Data loaders
    'load_color_system',
    'load_objects',
    'load_task5_templates',
    'Color',
    'ObjectItem',
    'Task5Template',
    # Generation
    'get_task_from_csv',
    'get_color_system_from_csv',
    'NEW_TO_OLD_CSV',
    'OLD_TO_NEW_CSV',
    'SUPPORTED_MODELS',
    # Aliases
    'TASK_ALIASES',
]
