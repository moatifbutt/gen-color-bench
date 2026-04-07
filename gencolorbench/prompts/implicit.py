"""
Task 5: Implicit Color Association

Evaluates whether T2I models can infer color relationships from context,
where a second object's color is defined relative to a first object.

Prompt structure:
    "A {color} backpack is placed next to a suitcase of the same color."

The model must understand that both objects should share the same color.

Output columns:
    id, color, object, ref_object, prompt
"""

import random
from typing import List, Optional

import pandas as pd

from .base import PromptGenerator, TemplateSelector, set_seed
from ..loaders import load_task5_templates, Task5Template


class ImplicitColorAssociationGenerator(PromptGenerator):
    """
    Generator for Task 5: Implicit Color Association.
    
    Tests whether models can infer that two objects should share
    the same color based on contextual description.
    """
    
    task_name = "implicit_color_association"
    
    def __init__(
        self,
        color_system: str,
        seed: Optional[int] = None,
        templates: Optional[List[Task5Template]] = None,
    ):
        """
        Initialize generator.
        
        Args:
            color_system: One of 'iscc_l1', 'iscc_l2', 'iscc_l3', 'css'
            seed: Random seed for reproducibility
            templates: Custom templates (optional, loads from file if None)
        """
        super().__init__(color_system, seed)
        
        # Load templates from file or use provided
        if templates is None:
            self.templates = load_task5_templates()
        else:
            self.templates = templates
    
    def get_output_columns(self) -> List[str]:
        """Get output CSV columns."""
        return ["id", "color", "object", "ref_object", "prompt"]
    
    def generate(self, n_prompts: int) -> pd.DataFrame:
        """
        Generate implicit color association prompts.
        
        Args:
            n_prompts: Number of prompts to generate
            
        Returns:
            DataFrame with columns: id, color, object, ref_object, prompt
        """
        if self.seed is not None:
            set_seed(self.seed)
        
        # Sample colors with stratified distribution
        colors = self.sampler.sample_colors(n_prompts)
        
        # Select templates (stratified across available templates)
        selected_templates = self._select_templates_stratified(n_prompts)
        
        # Generate prompts
        records = []
        for idx, (color, template) in enumerate(zip(colors, selected_templates)):
            # Replace {color} placeholder in prompt
            prompt = template.prompt.replace("{color}", color.name)
            
            records.append({
                "id": idx + 1,
                "color": color.name,
                "object": template.object,
                "ref_object": template.ref_object,
                "prompt": prompt,
            })
        
        return pd.DataFrame(records, columns=self.get_output_columns())
    
    def _select_templates_stratified(self, n: int) -> List[Task5Template]:
        """
        Select n templates with stratified coverage.
        
        Args:
            n: Number of templates to select
            
        Returns:
            List of Task5Template objects
        """
        n_templates = len(self.templates)
        
        if n <= n_templates:
            return random.sample(self.templates, n)
        
        # Need to repeat templates
        result = []
        
        # Full cycles
        full_cycles = n // n_templates
        for _ in range(full_cycles):
            cycle = self.templates.copy()
            random.shuffle(cycle)
            result.extend(cycle)
        
        # Remainder
        remainder = n % n_templates
        if remainder > 0:
            result.extend(random.sample(self.templates, remainder))
        
        return result


def generate_implicit_color_association(
    color_system: str,
    n_prompts: int,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience function to generate Task 5 prompts.
    
    Args:
        color_system: One of 'iscc_l1', 'iscc_l2', 'iscc_l3', 'css'
        n_prompts: Number of prompts to generate
        seed: Random seed for reproducibility
        output_dir: If provided, save CSV to this directory
        
    Returns:
        DataFrame with generated prompts
    """
    generator = ImplicitColorAssociationGenerator(color_system, seed)
    df = generator.generate(n_prompts)
    
    if output_dir:
        generator.save(df, output_dir)
    
    return df
