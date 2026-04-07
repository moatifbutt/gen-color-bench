"""
Task 3: Color-Object Association

Evaluates whether T2I models can apply a specified color to a main object
while keeping a secondary (contextual) object in its natural/neutral color.

Prompt structure:
    "A {color} apple on a white plate."
    
The main_obj (apple) should have the specified color.
The sec_obj (plate) should remain in a natural/neutral color.

Output columns:
    id, color, main_obj, sec_obj, prompt
"""

import random
import re
from typing import List, Optional, Tuple

import pandas as pd

from .base import PromptGenerator, TemplateSelector, set_seed
from ..loaders import load_objects


# =============================================================================
# Templates with Ground Truth
# Each entry: (main_obj, sec_obj, template)
# main_obj: The colored object (immediately after {color})
# sec_obj: The contextual/background object (should remain natural color)
# =============================================================================

TEMPLATES_WITH_GT = [
    # ----- Fruits/Vegetables -----
    ("apple", "plate", "A {color} apple on a white plate."),
    ("banana", "orange", "A {color} banana next to a sliced orange."),
    ("carrot", "kitchen counter", "A {color} carrot placed on a kitchen counter."),
    ("mango", "lemon", "A {color} mango in a fruit bowl with a lemon."),
    ("strawberry", "dessert plate", "A {color} strawberry on top of a dessert plate."),
    ("broccoli", "cutting board", "A {color} broccoli beside a cutting board."),
    ("guava", "wire fruit basket", "A {color} guava resting in a wire fruit basket."),
    ("papaya", "wooden table", "A {color} papaya cut in half on a wooden table."),
    ("lemon", "breakfast tray", "A {color} lemon on a breakfast tray."),
    ("orange", "ceramic bowl", "A {color} orange in a ceramic bowl."),
    
    # ----- Vehicles -----
    ("car", "sidewalk", "A {color} car parked near a sidewalk."),
    ("truck", "loading dock", "A {color} truck beside a loading dock."),
    ("bus", "bus stop", "A {color} bus at a bus stop."),
    ("motorcycle", "street corner", "A {color} motorcycle on a street corner."),
    ("taxi", "building", "A {color} taxi in front of a building."),
    ("jeep", "dirt road", "A {color} jeep driving along a dirt road."),
    ("sports car", "highway", "A {color} sports car on a highway."),
    ("airplane", "runway", "A {color} airplane at the runway gate."),
    ("bicycle", "fence", "A {color} bicycle leaning against a fence."),
    ("boat", "pier", "A {color} boat docked at a pier."),
    ("minivan", "driveway", "A {color} minivan parked in a driveway."),
    
    # ----- Furniture/Household -----
    ("chair", "wooden table", "A {color} chair next to a wooden table."),
    ("couch", "window", "A {color} couch in front of a window."),
    ("potted plant", "bookshelf", "A {color} potted plant on a bookshelf."),
    ("teapot", "breakfast tray", "A {color} teapot on a breakfast tray."),
    ("clock", "white wall", "A {color} clock on a white wall."),
    ("vase", "dining table", "A {color} vase placed on a dining table."),
    ("mug", "books", "A {color} mug on a desk with books."),
    ("candle", "mirror", "A {color} candle beside a mirror."),
    ("wardrobe", "small chair", "A {color} wardrobe beside a small chair."),
    ("sink", "marble countertop", "A {color} sink installed in a marble countertop."),
    ("desk", "home office", "A {color} desk in a home office."),
    ("bookcase", "living room wall", "A {color} bookcase against a living room wall."),
    ("book", "nightstand", "A {color} book on a nightstand."),
    ("table", "rug", "A {color} table on a patterned rug."),
    ("lipstick", "vanity mirror", "A {color} lipstick on a vanity mirror."),
    ("trash can", "kitchen corner", "A {color} trash can in a kitchen corner."),
    ("curtain", "window frame", "A {color} curtain hanging by a window."),
    ("pillow", "bed", "A {color} pillow on a bed."),
    ("thermos", "picnic table", "A {color} thermos on a picnic table."),
    ("tissue box", "coffee table", "A {color} tissue box on a coffee table."),
    
    # ----- Animals -----
    ("cat", "couch", "A {color} cat sleeping on a couch."),
    ("dog", "ball", "A {color} dog playing with a ball."),
    ("horse", "stable", "A {color} horse standing in a stable."),
    ("sheep", "green field", "A {color} sheep grazing on a green field."),
    ("cow", "wooden fence", "A {color} cow near a wooden fence."),
    ("parrot", "tree branch", "A {color} parrot on a tree branch."),
    ("duck", "pond", "A {color} duck floating on a pond."),
    ("owl", "wooden stump", "A {color} owl perched on a wooden stump."),
    ("elephant", "watering hole", "A {color} elephant standing near a watering hole."),
    ("bear", "forest", "A {color} bear walking through a forest."),
    ("crocodile", "riverbank", "A {color} crocodile resting on a riverbank."),
    ("turtle", "sandy beach", "A {color} turtle on a sandy beach."),
    
    # ----- Clothing/Accessories -----
    ("T-shirt", "table", "A {color} T-shirt folded on a table."),
    ("jacket", "coat rack", "A {color} jacket hanging on a coat rack."),
    ("jeans", "bed", "A {color} jeans on a bed."),
    ("hat", "chair", "A {color} hat resting on a chair."),
    ("tie", "hanger", "A {color} tie draped over a hanger."),
    ("coat", "door", "A {color} coat hanging near the door."),
    ("backpack", "wall", "A {color} backpack leaning against the wall."),
    ("handbag", "desk", "A {color} handbag on a desk."),
    ("sweatshirt", "chair", "A {color} sweatshirt draped over a chair."),
    ("suit", "closet", "A {color} suit hanging in a closet."),
    ("pants", "shelf", "A {color} pants folded on a shelf."),
    ("towel", "bathroom rack", "A {color} towel hanging on a bathroom rack."),
    ("socks", "dresser", "A {color} socks placed on a dresser."),
    ("suitcase", "airport terminal", "A {color} suitcase in an airport terminal."),
    ("umbrella", "bench", "A {color} umbrella resting against a bench."),
    ("wallet", "wooden table", "A {color} wallet on a wooden table."),
    
    # ----- Sports/Toys -----
    ("sports ball", "gym floor", "A {color} sports ball on a gym floor."),
    ("kite", "clear sky", "A {color} kite flying in a clear sky."),
    ("baseball glove", "bench", "A {color} baseball glove on a bench."),
    ("frisbee", "grass", "A {color} frisbee lying on the grass."),
    ("snowboard", "wall", "A {color} snowboard resting against a wall."),
    ("teddybear", "child's bed", "A {color} teddybear on a child's bed."),
    ("boxing glove", "shelf", "A {color} boxing glove placed on a shelf."),
    ("baseball bat", "dugout wall", "A {color} baseball bat leaning against a dugout wall."),
    ("surfboard", "sandy beach", "A {color} surfboard standing on a sandy beach."),
    ("skis", "cabin wall", "A {color} skis propped against a cabin wall."),
    ("skateboard", "concrete ramp", "A {color} skateboard on a concrete ramp."),
    ("football helmet", "locker room bench", "A {color} football helmet on a locker room bench."),
    ("golf ball", "putting green", "A {color} golf ball on a putting green."),
    ("balloon", "ceiling", "A {color} balloon floating near a ceiling."),
    
    # ----- Tools/Appliances -----
    ("microwave", "kitchen shelf", "A {color} microwave on a kitchen shelf."),
    ("hair dryer", "bathroom counter", "A {color} hair dryer on a bathroom counter."),
    ("toaster", "coffee machine", "A {color} toaster beside a coffee machine."),
    ("refrigerator", "kitchen", "A {color} refrigerator in the corner of the kitchen."),
    ("cutting board", "kitchen island", "A {color} cutting board on a kitchen island."),
    ("sponge", "faucet", "A {color} sponge near a faucet."),
    ("ruler", "notebook", "A {color} ruler beside an open notebook."),
    ("fan", "window", "A {color} fan placed near a window."),
    ("remote", "coffee table", "A {color} remote on a coffee table."),
    ("oven", "modern kitchen", "A {color} oven in a modern kitchen."),
    ("knife", "cutting board", "A {color} knife on a cutting board."),
    ("computer mouse", "desk", "A {color} computer mouse on a desk."),
    ("iron", "ironing board", "A {color} iron on an ironing board."),
    ("hammer", "workbench", "A {color} hammer on a workbench."),
    ("wrench", "toolbox", "A {color} wrench beside a toolbox."),
    ("saw", "garage wall", "A {color} saw hanging on a garage wall."),
]


# TASK3_TEMPLATES for compatibility
TASK3_TEMPLATES = [t[2] for t in TEMPLATES_WITH_GT]


def extract_objects_from_template(template: str) -> Tuple[str, str]:
    """
    Extract main_obj and sec_obj from a Task 3 template.
    
    Args:
        template: Template string with {color} placeholder
        
    Returns:
        Tuple of (main_obj, sec_obj)
    """
    # Look up in TEMPLATES_WITH_GT
    for main_obj, sec_obj, tmpl in TEMPLATES_WITH_GT:
        if tmpl == template:
            return (main_obj, sec_obj)
    
    # Fallback: try to parse using regex
    match = re.search(
        r'\{color\}\s+([a-zA-Z][a-zA-Z\s]*?)(?:\s+(?:on|in|beside|next|near|at|by|against|'
        r'hanging|resting|placed|leaning|flying|parked|sitting|standing|floating|draped|'
        r'folded|perched|swimming|grazing|holding|facing|docked|driving|walking|sleeping|'
        r'playing|installed|cut))',
        template
    )
    
    if match:
        main_obj = match.group(1).strip()
        return (main_obj, "")
    
    return ("", "")


def _get_valid_objects() -> set:
    """Get set of valid object names from objects.csv."""
    objects = load_objects()
    return {obj.name for obj in objects}


class ColorObjectAssociationGenerator(PromptGenerator):
    """
    Generator for Task 3: Color-Object Association.
    
    Tests whether models can apply a specified color to the main object
    while keeping a secondary object in a natural color.
    """
    
    task_name = "color_object_association"
    
    def __init__(
        self,
        color_system: str,
        seed: Optional[int] = None,
        templates: Optional[List[tuple]] = None,
    ):
        """
        Initialize generator.
        
        Args:
            color_system: One of 'iscc_l1', 'iscc_l2', 'iscc_l3', 'css'
            seed: Random seed for reproducibility
            templates: Custom templates (optional, uses TEMPLATES_WITH_GT if None)
        """
        super().__init__(color_system, seed)
        
        self.templates = templates if templates is not None else TEMPLATES_WITH_GT
        self.template_selector = TemplateSelector(
            [t[2] for t in self.templates],
            seed=seed,
            cycle_through=True,
        )
        
        # Build lookup for ground truth
        self._template_to_gt = {
            t[2]: (t[0], t[1]) for t in self.templates
        }
    
    def get_output_columns(self) -> List[str]:
        """Get output CSV columns."""
        return ["id", "color", "main_obj", "sec_obj", "prompt"]
    
    def generate(self, n_prompts: int) -> pd.DataFrame:
        """
        Generate color-object association prompts.
        
        Args:
            n_prompts: Number of prompts to generate
            
        Returns:
            DataFrame with columns: id, color, main_obj, sec_obj, prompt
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
            # Get ground truth for this template
            main_obj, sec_obj = self._template_to_gt[template]
            
            # Replace {color} placeholder
            prompt = template.replace("{color}", color.name)
            
            records.append({
                "id": idx + 1,
                "color": color.name,
                "main_obj": main_obj,
                "sec_obj": sec_obj,
                "prompt": prompt,
            })
        
        return pd.DataFrame(records, columns=self.get_output_columns())
    
    def _select_templates_stratified(self, n: int) -> List[str]:
        """
        Select n templates with stratified coverage.
        
        Args:
            n: Number of templates to select
            
        Returns:
            List of template strings
        """
        template_strings = [t[2] for t in self.templates]
        n_templates = len(template_strings)
        
        if n <= n_templates:
            return random.sample(template_strings, n)
        
        # Need to repeat templates - ensure even distribution
        result = []
        
        # Full cycles through all templates
        full_cycles = n // n_templates
        for _ in range(full_cycles):
            cycle = template_strings.copy()
            random.shuffle(cycle)
            result.extend(cycle)
        
        # Remainder
        remainder = n % n_templates
        if remainder > 0:
            result.extend(random.sample(template_strings, remainder))
        
        return result


def generate_color_object_association(
    color_system: str,
    n_prompts: int,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience function to generate Task 3 prompts.
    
    Args:
        color_system: One of 'iscc_l1', 'iscc_l2', 'iscc_l3', 'css'
        n_prompts: Number of prompts to generate
        seed: Random seed for reproducibility
        output_dir: If provided, save CSV to this directory
        
    Returns:
        DataFrame with generated prompts
    """
    generator = ColorObjectAssociationGenerator(color_system, seed)
    df = generator.generate(n_prompts)
    
    if output_dir:
        generator.save(df, output_dir)
    
    return df
