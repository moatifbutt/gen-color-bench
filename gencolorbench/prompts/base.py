"""
Base utilities for prompt generation.

Provides:
- StratifiedSampler: Distribute prompts evenly across colors and object categories
- TemplateSelector: Random template selection with optional cycling
- Seed management for reproducibility
"""

import random
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict

from ..loaders import (
    load_color_system,
    load_objects,
    get_objects_by_category,
    Color,
    ObjectItem,
    VALID_COLOR_SYSTEMS,
)


def set_seed(seed: Optional[int] = None) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed. If None, use system randomness.
    """
    if seed is not None:
        random.seed(seed)


class TemplateSelector:
    """
    Select templates randomly from a pool.
    
    Supports:
    - Random selection with replacement
    - Cycling through all templates before repeating
    """
    
    def __init__(
        self,
        templates: List[str],
        seed: Optional[int] = None,
        cycle_through: bool = False,
    ):
        """
        Initialize template selector.
        
        Args:
            templates: List of template strings
            seed: Random seed for reproducibility
            cycle_through: If True, use each template once before repeating
        """
        self.templates = templates
        self.cycle_through = cycle_through
        
        if seed is not None:
            set_seed(seed)
        
        self._reset_cycle()
    
    def _reset_cycle(self) -> None:
        """Reset the cycling state."""
        self._remaining = list(self.templates)
        random.shuffle(self._remaining)
    
    def select(self) -> str:
        """
        Select a template.
        
        Returns:
            Selected template string
        """
        if self.cycle_through:
            if not self._remaining:
                self._reset_cycle()
            return self._remaining.pop()
        else:
            return random.choice(self.templates)
    
    def select_n(self, n: int) -> List[str]:
        """
        Select n templates.
        
        Args:
            n: Number of templates to select
            
        Returns:
            List of selected templates
        """
        return [self.select() for _ in range(n)]
    
    def select_stratified(self, n: int) -> List[str]:
        """
        Select n templates with stratified coverage.
        
        Ensures each template is used roughly equally.
        
        Args:
            n: Number of templates to select
            
        Returns:
            List of selected templates
        """
        n_templates = len(self.templates)
        
        if n <= n_templates:
            return random.sample(self.templates, n)
        
        # Need to repeat templates - ensure even distribution
        result = []
        
        # Full cycles through all templates
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


class StratifiedSampler:
    """
    Stratified sampling for prompt generation.
    
    Ensures:
    - Each color appears at least once (minimum coverage)
    - Even distribution across colors
    - Even distribution across object categories within each color
    """
    
    def __init__(
        self,
        color_system: str,
        seed: Optional[int] = None,
    ):
        """
        Initialize stratified sampler.
        
        Args:
            color_system: One of 'iscc_l1', 'iscc_l2', 'iscc_l3', 'css'
            seed: Random seed for reproducibility
        """
        if color_system not in VALID_COLOR_SYSTEMS:
            raise ValueError(
                f"Invalid color system: {color_system}. "
                f"Valid options: {VALID_COLOR_SYSTEMS}"
            )
        
        self.color_system = color_system
        self.colors = load_color_system(color_system)
        self.objects_by_category = get_objects_by_category()
        self.categories = list(self.objects_by_category.keys())
        self.all_objects = load_objects()
        
        if seed is not None:
            set_seed(seed)
    
    @staticmethod
    def _distribute_count(total: int, n_buckets: int) -> List[int]:
        """
        Distribute total count evenly across buckets.
        
        Args:
            total: Total count to distribute
            n_buckets: Number of buckets
            
        Returns:
            List of counts per bucket (sum equals total)
        """
        if n_buckets == 0:
            return []
        
        base = total // n_buckets
        remainder = total % n_buckets
        
        # First 'remainder' buckets get one extra
        counts = [base + 1 if i < remainder else base for i in range(n_buckets)]
        return counts
    
    def sample_color_object_pairs(
        self,
        n_prompts: int,
        min_per_color: int = 1,
    ) -> List[Tuple[Color, ObjectItem]]:
        """
        Sample (color, object) pairs with stratified distribution.
        
        Strategy:
        1. If n_prompts >= n_colors: ensure each color appears at least min_per_color times
        2. If n_prompts < n_colors: sample n_prompts distinct colors
        3. Distribute across object categories
        
        Args:
            n_prompts: Total number of prompts to generate
            min_per_color: Minimum prompts per color (only if n_prompts >= n_colors)
            
        Returns:
            List of (Color, ObjectItem) tuples
        """
        n_colors = len(self.colors)
        
        # Handle case where we don't have enough prompts for all colors
        if n_prompts < n_colors:
            # Sample a subset of colors
            sampled_colors = random.sample(self.colors, n_prompts)
            pairs = []
            for color in sampled_colors:
                # Sample object with category stratification
                category = random.choice(self.categories)
                obj = random.choice(self.objects_by_category[category])
                pairs.append((color, obj))
            random.shuffle(pairs)
            return pairs
        
        # Normal case: distribute prompts across colors
        prompts_per_color = self._distribute_count(n_prompts, n_colors)
        
        pairs = []
        
        for color, n_for_color in zip(self.colors, prompts_per_color):
            # Distribute this color's prompts across categories
            prompts_per_category = self._distribute_count(
                n_for_color, len(self.categories)
            )
            
            for category, n_for_category in zip(self.categories, prompts_per_category):
                if n_for_category == 0:
                    continue
                
                # Get objects from this category
                category_objects = self.objects_by_category[category]
                
                # Sample with replacement if needed
                if n_for_category <= len(category_objects):
                    sampled_objects = random.sample(category_objects, n_for_category)
                else:
                    # Need to repeat some objects
                    sampled_objects = random.choices(category_objects, k=n_for_category)
                
                for obj in sampled_objects:
                    pairs.append((color, obj))
        
        # Shuffle to avoid ordering by color
        random.shuffle(pairs)
        
        return pairs
    
    def sample_colors(
        self,
        n_prompts: int,
        min_per_color: int = 1,
    ) -> List[Color]:
        """
        Sample colors with stratified distribution (for tasks without objects).
        
        Args:
            n_prompts: Number of colors to sample
            min_per_color: Minimum appearances per color
            
        Returns:
            List of Color objects
        """
        n_colors = len(self.colors)
        
        if n_prompts < n_colors:
            return random.sample(self.colors, n_prompts)
        
        # Distribute prompts across colors
        counts = self._distribute_count(n_prompts, n_colors)
        
        result = []
        for color, count in zip(self.colors, counts):
            result.extend([color] * count)
        
        random.shuffle(result)
        return result
    
    def sample_color_sets(
        self,
        n_prompts: int,
        colors_per_set: int,
        min_per_color: int = 1,
        allow_duplicates_in_set: bool = False,
    ) -> List[List[Color]]:
        """
        Sample sets of colors (for multi-object tasks).
        
        Args:
            n_prompts: Number of color sets to generate
            colors_per_set: Number of colors in each set
            min_per_color: Minimum appearances per color across all sets
            allow_duplicates_in_set: If False, colors in a set must be distinct
            
        Returns:
            List of color lists, each containing colors_per_set colors
        """
        n_colors = len(self.colors)
        
        if not allow_duplicates_in_set and colors_per_set > n_colors:
            raise ValueError(
                f"colors_per_set ({colors_per_set}) exceeds available colors "
                f"({n_colors}) and duplicates are not allowed"
            )
        
        # Track how many times each color has been used
        color_usage = {color.name: 0 for color in self.colors}
        
        color_sets = []
        
        for _ in range(n_prompts):
            # Prioritize colors that haven't met minimum usage
            underused = [
                c for c in self.colors 
                if color_usage[c.name] < min_per_color
            ]
            
            if allow_duplicates_in_set:
                # Can pick any colors, prioritize underused
                if len(underused) >= colors_per_set:
                    selected = random.sample(underused, colors_per_set)
                else:
                    selected = underused + random.choices(
                        self.colors, k=colors_per_set - len(underused)
                    )
            else:
                # Must pick distinct colors
                if len(underused) >= colors_per_set:
                    selected = random.sample(underused, colors_per_set)
                else:
                    # Start with underused, fill rest from remaining
                    selected = list(underused)
                    remaining = [c for c in self.colors if c not in selected]
                    selected.extend(
                        random.sample(remaining, colors_per_set - len(selected))
                    )
            
            # Update usage counts
            for color in selected:
                color_usage[color.name] += 1
            
            color_sets.append(selected)
        
        return color_sets
    
    def get_all_colors(self) -> List[Color]:
        """Get all colors in the color system."""
        return self.colors.copy()
    
    def get_all_objects(self) -> List[ObjectItem]:
        """Get all objects."""
        return self.all_objects.copy()
    
    def sample_distinct_colors(self, n: int) -> List[Color]:
        """
        Sample n distinct colors.
        
        Args:
            n: Number of distinct colors to sample
            
        Returns:
            List of n distinct Color objects
            
        Raises:
            ValueError: If n > number of available colors
        """
        if n > len(self.colors):
            raise ValueError(
                f"Cannot sample {n} distinct colors from {len(self.colors)} available"
            )
        return random.sample(self.colors, n)


class PromptGenerator:
    """
    Abstract base class for task-specific prompt generators.
    
    Subclasses should:
    - Set task_name class attribute
    - Implement generate() method
    - Implement get_output_columns() method
    """
    
    # Task name for file naming (override in subclasses)
    task_name: str = "base"
    
    def __init__(
        self,
        color_system: str,
        seed: Optional[int] = None,
    ):
        """
        Initialize generator.
        
        Args:
            color_system: One of 'iscc_l1', 'iscc_l2', 'iscc_l3', 'css'
            seed: Random seed for reproducibility
        """
        self.color_system = color_system
        self.seed = seed
        self.sampler = StratifiedSampler(color_system, seed)
        
        if seed is not None:
            set_seed(seed)
    
    def generate(self, n_prompts: int):
        """
        Generate prompts. Override in subclass.
        
        Args:
            n_prompts: Number of prompts to generate
            
        Returns:
            DataFrame with generated prompts
        """
        raise NotImplementedError("Subclasses must implement generate()")
    
    def get_output_columns(self) -> List[str]:
        """
        Get column names for output CSV. Override in subclass.
        
        Returns:
            List of column names
        """
        raise NotImplementedError("Subclasses must implement get_output_columns()")
    
    def get_output_filename(self) -> str:
        """
        Get output filename based on task name and color system.
        
        Returns:
            Filename string (without directory)
        """
        return f"{self.task_name}_{self.color_system}.csv"
    
    def save(self, df, output_dir: str) -> str:
        """
        Save generated prompts to CSV.
        
        Args:
            df: DataFrame with prompts
            output_dir: Output directory
            
        Returns:
            Full path to saved file
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, self.get_output_filename())
        df.to_csv(filepath, index=False)
        
        return filepath


def compute_distribution_stats(
    items: List[Any],
    key_fn: callable = None,
) -> Dict[str, int]:
    """
    Compute distribution statistics for debugging/verification.
    
    Args:
        items: List of items (or tuples)
        key_fn: Function to extract grouping key. If None, use item directly.
        
    Returns:
        Dictionary mapping key to count
    """
    counts = defaultdict(int)
    for item in items:
        if key_fn:
            key = key_fn(item)
        else:
            key = item
        
        # Extract name if it's a dataclass with name attribute
        if hasattr(key, 'name'):
            key = key.name
        
        counts[key] += 1
    
    return dict(counts)
