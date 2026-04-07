"""
Task 2: Numerical Color Understanding

Evaluates whether T2I models can interpret numerical color specifications (RGB/HEX).

Two subtasks:
    - RGB: "A {object} in rgb(R, G, B)."
    - HEX: "A {object} with hex color #RRGGBB."

Output columns (RGB):
    id, color_name, r, g, b, object, dataset_category, prompt

Output columns (HEX):
    id, css_name, color_name, hex, r, g, b, object, dataset_category, prompt
"""

import random
from typing import List, Optional

import pandas as pd

from .base import PromptGenerator, TemplateSelector, set_seed


# Templates for RGB prompts
RGB_TEMPLATES = [
    "A {object} in rgb({r}, {g}, {b}).",
    "A {object} with the color rgb({r}, {g}, {b}).",
    "A {object} rendered in RGB color rgb({r}, {g}, {b}).",
    "A photo of a {object} in color rgb({r}, {g}, {b}).",
    "A {object} with color rgb({r}, {g}, {b}).",
    "An image of a {object} in rgb({r}, {g}, {b}).",
    "A {object} colored rgb({r}, {g}, {b}).",
    "A realistic {object} in rgb({r}, {g}, {b}).",
]

# Templates for HEX prompts
HEX_TEMPLATES = [
    "An image of a {object} in hex color {hex}.",
    "A {object} in color {hex}.",
    "A {object} with hex color {hex}.",
    "A close-up of a {object} in the color {hex}.",
    "A {object} rendered in {hex} color.",
    "A photo of a {object} in the color {hex}.",
    "A {object} rendered entirely in {hex}.",
    "A {object} designed in {hex} color.",
    "A realistic {hex}-colored {object}.",
    "A highly detailed {object} in hex {hex}.",
]


class NumericRGBGenerator(PromptGenerator):
    """
    Generator for Task 2: Numerical Color Understanding (RGB format).
    
    Tests whether models can interpret RGB values in prompts.
    """
    
    task_name = "numeric_rgb"
    
    def __init__(
        self,
        color_system: str,
        seed: Optional[int] = None,
        templates: Optional[List[str]] = None,
    ):
        """
        Initialize generator.
        
        Args:
            color_system: One of 'iscc_l1', 'iscc_l2', 'iscc_l3', 'css'
            seed: Random seed for reproducibility
            templates: Custom templates (optional)
        """
        super().__init__(color_system, seed)
        
        self.templates = templates or RGB_TEMPLATES
        self.template_selector = TemplateSelector(self.templates, seed)
    
    def get_output_columns(self) -> List[str]:
        """Get output CSV columns."""
        return ["id", "color_name", "r", "g", "b", "object", "dataset_category", "prompt"]
    
    def generate(self, n_prompts: int) -> pd.DataFrame:
        """
        Generate RGB numeric color prompts.
        
        Args:
            n_prompts: Number of prompts to generate
            
        Returns:
            DataFrame with RGB prompt data
        """
        if self.seed is not None:
            set_seed(self.seed)
        
        # Sample color-object pairs
        pairs = self.sampler.sample_color_object_pairs(n_prompts)
        
        # Select templates
        templates = self.template_selector.select_stratified(n_prompts)
        
        # Generate prompts
        records = []
        for idx, ((color, obj), template) in enumerate(zip(pairs, templates)):
            prompt = template.format(
                object=obj.name,
                r=color.r,
                g=color.g,
                b=color.b,
            )
            
            records.append({
                "id": idx + 1,
                "color_name": color.name,
                "r": color.r,
                "g": color.g,
                "b": color.b,
                "object": obj.name,
                "dataset_category": obj.category,
                "prompt": prompt,
            })
        
        return pd.DataFrame(records, columns=self.get_output_columns())


class NumericHEXGenerator(PromptGenerator):
    """
    Generator for Task 2: Numerical Color Understanding (HEX format).
    
    Tests whether models can interpret HEX color codes.
    Only works with CSS color system (which has hex values).
    """
    
    task_name = "numeric_hex"
    
    def __init__(
        self,
        seed: Optional[int] = None,
        templates: Optional[List[str]] = None,
    ):
        """
        Initialize generator.
        
        Note: HEX format only available for CSS colors.
        
        Args:
            seed: Random seed for reproducibility
            templates: Custom templates (optional)
        """
        # HEX is only available for CSS color system
        super().__init__("css", seed)
        
        self.templates = templates or HEX_TEMPLATES
        self.template_selector = TemplateSelector(self.templates, seed)
    
    def get_output_columns(self) -> List[str]:
        """Get output CSV columns."""
        return [
            "id", "css_name", "color_name", "hex", "r", "g", "b",
            "object", "dataset_category", "prompt"
        ]
    
    def get_output_filename(self) -> str:
        """Override to use 'css' in filename."""
        return f"{self.task_name}_css.csv"
    
    def generate(self, n_prompts: int) -> pd.DataFrame:
        """
        Generate HEX numeric color prompts.
        
        Args:
            n_prompts: Number of prompts to generate
            
        Returns:
            DataFrame with HEX prompt data
        """
        if self.seed is not None:
            set_seed(self.seed)
        
        # Sample color-object pairs
        pairs = self.sampler.sample_color_object_pairs(n_prompts)
        
        # Select templates
        templates = self.template_selector.select_stratified(n_prompts)
        
        # Generate prompts
        records = []
        for idx, ((color, obj), template) in enumerate(zip(pairs, templates)):
            prompt = template.format(
                object=obj.name,
                hex=color.hex,
            )
            
            records.append({
                "id": idx + 1,
                "css_name": color.css_name,
                "color_name": color.name,
                "hex": color.hex,
                "r": color.r,
                "g": color.g,
                "b": color.b,
                "object": obj.name,
                "dataset_category": obj.category,
                "prompt": prompt,
            })
        
        return pd.DataFrame(records, columns=self.get_output_columns())


def generate_numeric_rgb(
    color_system: str,
    n_prompts: int,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience function to generate Task 2 RGB prompts.
    
    Args:
        color_system: One of 'iscc_l1', 'iscc_l2', 'iscc_l3', 'css'
        n_prompts: Number of prompts to generate
        seed: Random seed for reproducibility
        output_dir: If provided, save CSV to this directory
        
    Returns:
        DataFrame with generated prompts
    """
    generator = NumericRGBGenerator(color_system, seed)
    df = generator.generate(n_prompts)
    
    if output_dir:
        generator.save(df, output_dir)
    
    return df


def generate_numeric_hex(
    n_prompts: int,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience function to generate Task 2 HEX prompts.
    
    Note: HEX is only available for CSS color system.
    
    Args:
        n_prompts: Number of prompts to generate
        seed: Random seed for reproducibility
        output_dir: If provided, save CSV to this directory
        
    Returns:
        DataFrame with generated prompts
    """
    generator = NumericHEXGenerator(seed)
    df = generator.generate(n_prompts)
    
    if output_dir:
        generator.save(df, output_dir)
    
    return df
