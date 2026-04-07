"""
Prompt generators for GenColorBench tasks.

Tasks:
    1. Color Name Accuracy (color_name.py)
    2. Numerical Color Understanding (numeric.py)
    3. Color-Object Association (color_object.py)
    4. Multi-Object Color Composition (multi_object.py)
    5. Implicit Color Association (implicit.py)
"""

from .base import (
    StratifiedSampler,
    PromptGenerator,
    TemplateSelector,
    set_seed,
)

from .color_name import (
    ColorNameAccuracyGenerator,
    generate_color_name_accuracy,
)

from .numeric import (
    NumericRGBGenerator,
    NumericHEXGenerator,
    generate_numeric_rgb,
    generate_numeric_hex,
)

from .color_object import (
    ColorObjectAssociationGenerator,
    generate_color_object_association,
)

from .multi_object import (
    MultiObjectCompositionGenerator,
    generate_multi_object_composition,
)

from .implicit import (
    ImplicitColorAssociationGenerator,
    generate_implicit_color_association,
)

__all__ = [
    # Base utilities
    "StratifiedSampler",
    "PromptGenerator",
    "TemplateSelector",
    "set_seed",
    # Task 1: Color Name Accuracy
    "ColorNameAccuracyGenerator",
    "generate_color_name_accuracy",
    # Task 2: Numerical Color Understanding
    "NumericRGBGenerator",
    "NumericHEXGenerator",
    "generate_numeric_rgb",
    "generate_numeric_hex",
    # Task 3: Color-Object Association
    "ColorObjectAssociationGenerator",
    "generate_color_object_association",
    # Task 4: Multi-Object Composition
    "MultiObjectCompositionGenerator",
    "generate_multi_object_composition",
    # Task 5: Implicit Color Association
    "ImplicitColorAssociationGenerator",
    "generate_implicit_color_association",
]
