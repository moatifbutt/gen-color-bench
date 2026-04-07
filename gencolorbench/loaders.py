"""
Data loaders for GenColorBench.

Provides functions to load:
- Color systems (ISCC-NBS L1/L2/L3, CSS3/X11)
- Objects (COCO + ImageNet categories)
- Task 5 templates (implicit color association)
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

# Package data directory - check multiple locations
def _find_data_dir() -> Path:
    """Find the data directory, checking multiple locations."""
    # Check relative to this file
    pkg_data = Path(__file__).parent / "data"
    if pkg_data.exists():
        return pkg_data
    
    # Check parent directory (if running from gsam2/)
    parent_data = Path(__file__).parent.parent / "data"
    if parent_data.exists():
        return parent_data
    
    # Check environment variable
    env_data = os.environ.get("GENCOLORBENCH_DATA_DIR")
    if env_data and Path(env_data).exists():
        return Path(env_data)
    
    # Default to package-relative
    return pkg_data


DATA_DIR = _find_data_dir()

# Valid color systems
VALID_COLOR_SYSTEMS = ["iscc_l1", "iscc_l2", "iscc_l3", "css"]

# Color system file mapping
COLOR_SYSTEM_FILES = {
    "iscc_l1": DATA_DIR / "color_systems" / "iscc_nbs_l1.csv",
    "iscc_l2": DATA_DIR / "color_systems" / "iscc_nbs_l2.csv",
    "iscc_l3": DATA_DIR / "color_systems" / "iscc_nbs_l3.csv",
    "css": DATA_DIR / "color_systems" / "css3_x11_colors.csv",
}

# Objects file
OBJECTS_FILE = DATA_DIR / "objects" / "objects.csv"

# Task 5 templates file
TASK5_TEMPLATES_FILE = DATA_DIR / "templates" / "task5_relative_prompt_templates.csv"


@dataclass
class Color:
    """Represents a color with its properties."""
    name: str
    r: int
    g: int
    b: int
    hex: Optional[str] = None
    css_name: Optional[str] = None  # CamelCase name for CSS colors


@dataclass
class ObjectItem:
    """Represents an object with its category."""
    name: str
    category: str


@dataclass
class Task5Template:
    """Represents a Task 5 implicit color association template."""
    object: str
    ref_object: str
    prompt: str


def load_color_system(system: str) -> List[Color]:
    """
    Load a color system.
    
    Args:
        system: One of 'iscc_l1', 'iscc_l2', 'iscc_l3', 'css'
        
    Returns:
        List of Color objects
        
    Raises:
        ValueError: If system is not recognized
        FileNotFoundError: If color system file not found
    """
    if system not in VALID_COLOR_SYSTEMS:
        raise ValueError(
            f"Unknown color system: {system}. "
            f"Available: {VALID_COLOR_SYSTEMS}"
        )
    
    filepath = COLOR_SYSTEM_FILES[system]
    if not filepath.exists():
        raise FileNotFoundError(f"Color system file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    colors = []
    
    if system == "css":
        # CSS has additional columns: css_name, color_name, hex
        for _, row in df.iterrows():
            colors.append(Color(
                name=row["color_name"],
                r=int(row["r"]),
                g=int(row["g"]),
                b=int(row["b"]),
                hex=row["hex"],
                css_name=row["css_name"],
            ))
    else:
        # ISCC-NBS has: id, color_name, r, g, b
        for _, row in df.iterrows():
            colors.append(Color(
                name=row["color_name"],
                r=int(row["r"]),
                g=int(row["g"]),
                b=int(row["b"]),
            ))
    
    return colors


def load_color_system_df(system: str) -> pd.DataFrame:
    """
    Load a color system as a DataFrame.
    
    Args:
        system: One of 'iscc_l1', 'iscc_l2', 'iscc_l3', 'css'
        
    Returns:
        DataFrame with color data
    """
    if system not in VALID_COLOR_SYSTEMS:
        raise ValueError(
            f"Unknown color system: {system}. "
            f"Available: {VALID_COLOR_SYSTEMS}"
        )
    
    filepath = COLOR_SYSTEM_FILES[system]
    if not filepath.exists():
        raise FileNotFoundError(f"Color system file not found: {filepath}")
    
    return pd.read_csv(filepath)


def load_objects() -> List[ObjectItem]:
    """
    Load object list.
    
    Returns:
        List of ObjectItem objects
    """
    if not OBJECTS_FILE.exists():
        raise FileNotFoundError(f"Objects file not found: {OBJECTS_FILE}")
    
    df = pd.read_csv(OBJECTS_FILE)
    objects = []
    
    for _, row in df.iterrows():
        objects.append(ObjectItem(
            name=row["Class_Name"],
            category=row["Dataset_Category"],
        ))
    
    return objects


def load_objects_df() -> pd.DataFrame:
    """
    Load object list as DataFrame.
    
    Returns:
        DataFrame with object data
    """
    if not OBJECTS_FILE.exists():
        raise FileNotFoundError(f"Objects file not found: {OBJECTS_FILE}")
    
    return pd.read_csv(OBJECTS_FILE)


def load_task5_templates() -> List[Task5Template]:
    """
    Load Task 5 implicit color association templates.
    
    Returns:
        List of Task5Template objects
    """
    if not TASK5_TEMPLATES_FILE.exists():
        raise FileNotFoundError(
            f"Task 5 templates file not found: {TASK5_TEMPLATES_FILE}"
        )
    
    df = pd.read_csv(TASK5_TEMPLATES_FILE)
    templates = []
    
    for _, row in df.iterrows():
        templates.append(Task5Template(
            object=row["object"],
            ref_object=row["ref_object"],
            prompt=row["prompt"],
        ))
    
    return templates


def get_objects_by_category() -> Dict[str, List[ObjectItem]]:
    """
    Get objects grouped by their category.
    
    Returns:
        Dictionary mapping category name to list of ObjectItem
    """
    objects = load_objects()
    by_category: Dict[str, List[ObjectItem]] = {}
    
    for obj in objects:
        if obj.category not in by_category:
            by_category[obj.category] = []
        by_category[obj.category].append(obj)
    
    return by_category


def get_object_categories() -> List[str]:
    """
    Get list of all object categories.
    
    Returns:
        List of category names
    """
    objects = load_objects()
    categories = list(set(obj.category for obj in objects))
    return sorted(categories)


def get_color_system_info(system: str) -> Dict:
    """
    Get metadata about a color system.
    
    Args:
        system: Color system identifier
        
    Returns:
        Dictionary with system info
    """
    colors = load_color_system(system)
    
    names = {
        "iscc_l1": "ISCC-NBS Level 1",
        "iscc_l2": "ISCC-NBS Level 2",
        "iscc_l3": "ISCC-NBS Level 3",
        "css": "CSS3/X11",
    }
    
    return {
        "system": system,
        "display_name": names.get(system, system),
        "n_colors": len(colors),
        "has_hex": system == "css",
    }


def load_all_color_systems() -> Dict[str, List[Color]]:
    """
    Load all available color systems.
    
    Returns:
        Dictionary mapping system name to list of Color objects
    """
    return {system: load_color_system(system) for system in VALID_COLOR_SYSTEMS}
