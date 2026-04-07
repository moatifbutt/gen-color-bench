"""
Task 1: Color Name Accuracy

Evaluates whether T2I models correctly render objects in colors specified by name.

Prompt structure:
    "A {color} {object}."
    "A photo of a {color} {object}."
    etc.

Output columns:
    id, color, object, prompt
"""

import random
from typing import List, Optional

import pandas as pd

from .base import PromptGenerator, TemplateSelector, set_seed


# Templates for Task 1
TASK1_TEMPLATES = [
    "A {color} {object}.",
    "The {object} is {color}.",
    "A photo of a {color} {object}.",
    "A {object} that is entirely {color}.",
    "An image of a {color} {object}.",
    "A {color} colored {object}.",
    "A single {color} {object}.",
    "A {object}, and it's {color}.",
    "A {object} in a {color} color.",
    "A {object} rendered in {color} color.",
    "A {object} with a {color} color.",
    "A realistic {object} in {color}.",
]


class ColorNameAccuracyGenerator(PromptGenerator):
    """
    Generator for Task 1: Color Name Accuracy.
    
    Tests whether models can generate objects in specified named colors.
    """
    
    task_name = "color_name_accuracy"
    
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
            templates: Custom templates (optional, uses defaults if None)
        """
        super().__init__(color_system, seed)
        
        self.templates = templates or TASK1_TEMPLATES
        self.template_selector = TemplateSelector(self.templates, seed)
    
    def get_output_columns(self) -> List[str]:
        """Get output CSV columns."""
        return ["id", "color", "object", "prompt"]
    
    def generate(self, n_prompts: int) -> pd.DataFrame:
        """
        Generate color name accuracy prompts.
        
        Args:
            n_prompts: Number of prompts to generate
            
        Returns:
            DataFrame with columns: id, color, object, prompt
        """
        # Reset seed for reproducibility
        if self.seed is not None:
            set_seed(self.seed)
        
        # Sample color-object pairs with stratified distribution
        pairs = self.sampler.sample_color_object_pairs(n_prompts)
        
        # Select templates (stratified)
        templates = self.template_selector.select_stratified(n_prompts)
        
        # Generate prompts
        records = []
        for idx, ((color, obj), template) in enumerate(zip(pairs, templates)):
            prompt = template.format(color=color.name, object=obj.name)
            
            records.append({
                "id": idx + 1,
                "color": color.name,
                "object": obj.name,
                "prompt": prompt,
            })
        
        return pd.DataFrame(records, columns=self.get_output_columns())


def generate_color_name_accuracy(
    color_system: str,
    n_prompts: int,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience function to generate Task 1 prompts.
    
    Args:
        color_system: One of 'iscc_l1', 'iscc_l2', 'iscc_l3', 'css'
        n_prompts: Number of prompts to generate
        seed: Random seed for reproducibility
        output_dir: If provided, save CSV to this directory
        
    Returns:
        DataFrame with generated prompts
    """
    generator = ColorNameAccuracyGenerator(color_system, seed)
    df = generator.generate(n_prompts)
    
    if output_dir:
        generator.save(df, output_dir)
    
    return df
