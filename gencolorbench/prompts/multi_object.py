"""
Task 4: Multi-Object Color Composition

Evaluates whether T2I models can assign distinct colors to multiple objects
in a single scene without color leakage or confusion.

Prompt structure:
    "A {color} banana and a {color} apple on a wooden table."
    "A {color} dog next to a {color} cat and a {color} couch in a living room."

Output columns:
    id, color_count, object1, color1, object2, color2, object3, color3, prompt
"""

import random
import re
from typing import List, Optional, Tuple

import pandas as pd

from .base import PromptGenerator, TemplateSelector, set_seed
from ..loaders import load_objects


# =============================================================================
# Semantic Exclusion Groups
# =============================================================================

EXCLUSION_GROUPS = [
    {"pants", "jeans"},
    {"jacket", "coat", "sweatshirt", "suit"},
    {"car", "sports car", "taxi", "jeep", "minivan"},
    {"owl", "parrot", "duck"},
    {"chair", "couch"},
]


# =============================================================================
# Templates with Ground Truth
# Format: (object1, object2, object3_or_None, template)
# =============================================================================

TEMPLATES_WITH_GT = [
    # =========================================================================
    # 2-OBJECT TEMPLATES
    # =========================================================================
    
    # Fruits/Vegetables
    ("banana", "apple", None, "A {color} banana and a {color} apple on a wooden table."),
    ("carrot", "broccoli", None, "A {color} carrot and a {color} broccoli on a cutting board."),
    ("mango", "papaya", None, "A {color} mango and a {color} papaya in a fruit bowl."),
    ("strawberry", "lemon", None, "A {color} strawberry next to a {color} lemon on a plate."),
    ("orange", "guava", None, "A {color} orange beside a {color} guava on the kitchen counter."),
    
    # Vehicles
    ("car", "bus", None, "A {color} car parked near a {color} bus on the street."),
    ("bicycle", "motorcycle", None, "A {color} bicycle next to a {color} motorcycle in the garage."),
    ("airplane", "truck", None, "A {color} airplane flying above a {color} truck on the runway."),
    ("boat", "jeep", None, "A {color} boat docked near a {color} jeep at the marina."),
    ("taxi", "bus", None, "A {color} taxi and a {color} bus at the station."),
    ("minivan", "bicycle", None, "A {color} minivan parked beside a {color} bicycle in the driveway."),
    ("sports car", "truck", None, "A {color} sports car next to a {color} truck in the parking lot."),
    
    # Animals
    ("dog", "cat", None, "A {color} dog sitting beside a {color} cat on the floor."),
    ("horse", "cow", None, "A {color} horse standing with a {color} cow in the field."),
    ("elephant", "bear", None, "A {color} elephant near a {color} bear in the wild."),
    ("sheep", "horse", None, "A {color} sheep grazing next to a {color} horse in the meadow."),
    ("turtle", "crocodile", None, "A {color} turtle resting near a {color} crocodile by the river."),
    ("parrot", "dog", None, "A {color} parrot perched near a {color} dog in the yard."),
    ("owl", "cat", None, "A {color} owl perched above a {color} cat at night."),
    ("duck", "cow", None, "A {color} duck swimming near a {color} cow by the pond."),
    
    # Furniture/Household
    ("chair", "desk", None, "A {color} chair and a {color} desk in a sunlit room."),
    ("couch", "bookcase", None, "A {color} couch facing a {color} bookcase in the living room."),
    ("table", "wardrobe", None, "A {color} table beside a {color} wardrobe in the bedroom."),
    ("vase", "clock", None, "A {color} vase next to a {color} clock on the shelf."),
    ("mug", "teapot", None, "A {color} mug beside a {color} teapot on the table."),
    ("candle", "book", None, "A {color} candle next to a {color} book on the nightstand."),
    ("pillow", "curtain", None, "A {color} pillow on the bed near a {color} curtain by the window."),
    ("sink", "trash can", None, "A {color} sink beside a {color} trash can in the kitchen."),
    ("thermos", "tissue box", None, "A {color} thermos next to a {color} tissue box on the desk."),
    ("potted plant", "vase", None, "A {color} potted plant beside a {color} vase on the windowsill."),
    
    # Clothing/Accessories
    ("T-shirt", "jeans", None, "A {color} T-shirt and {color} jeans folded on the bed."),
    ("jacket", "hat", None, "A {color} jacket hanging beside a {color} hat on the rack."),
    ("tie", "suit", None, "A {color} tie draped over a {color} suit in the closet."),
    ("backpack", "umbrella", None, "A {color} backpack leaning against a {color} umbrella by the door."),
    ("handbag", "suitcase", None, "A {color} handbag placed on a {color} suitcase at the airport."),
    ("coat", "hat", None, "A {color} coat hanging next to a {color} hat on the hook."),
    ("sweatshirt", "pants", None, "A {color} sweatshirt folded beside {color} pants on the shelf."),
    ("socks", "towel", None, "A {color} socks next to a {color} towel on the dresser."),
    ("wallet", "tie", None, "A {color} wallet beside a {color} tie on the table."),
    
    # Sports/Toys
    ("sports ball", "baseball glove", None, "A {color} sports ball next to a {color} baseball glove on the bench."),
    ("skateboard", "frisbee", None, "A {color} skateboard beside a {color} frisbee on the grass."),
    ("surfboard", "kite", None, "A {color} surfboard leaning near a {color} kite on the beach."),
    ("snowboard", "skis", None, "A {color} snowboard resting against {color} skis by the cabin."),
    ("teddybear", "balloon", None, "A {color} teddybear holding a {color} balloon on the bed."),
    ("football helmet", "boxing glove", None, "A {color} football helmet beside a {color} boxing glove on the shelf."),
    ("baseball bat", "golf ball", None, "A {color} baseball bat next to a {color} golf ball on the floor."),
    
    # Tools/Appliances
    ("microwave", "toaster", None, "A {color} microwave next to a {color} toaster on the counter."),
    ("refrigerator", "oven", None, "A {color} refrigerator beside a {color} oven in the kitchen."),
    ("hair dryer", "iron", None, "A {color} hair dryer next to an {color} iron on the bathroom counter."),
    ("remote", "computer mouse", None, "A {color} remote beside a {color} computer mouse on the desk."),
    ("hammer", "wrench", None, "A {color} hammer next to a {color} wrench on the workbench."),
    ("knife", "cutting board", None, "A {color} knife on a {color} cutting board in the kitchen."),
    ("fan", "sponge", None, "A {color} fan beside a {color} sponge on the shelf."),
    ("ruler", "saw", None, "A {color} ruler next to a {color} saw on the workshop table."),
    ("lipstick", "handbag", None, "A {color} lipstick inside a {color} handbag on the vanity."),
    
    # Cross-Category
    ("backpack", "bicycle", None, "A {color} backpack hanging on a {color} bicycle."),
    ("umbrella", "car", None, "A {color} umbrella leaning against a {color} car."),
    ("book", "mug", None, "A {color} book next to a {color} mug on the coffee table."),
    ("hat", "suitcase", None, "A {color} hat resting on a {color} suitcase."),
    ("clock", "candle", None, "A {color} clock beside a {color} candle on the mantle."),
    
    # =========================================================================
    # 3-OBJECT TEMPLATES
    # =========================================================================
    
    # Fruits/Vegetables
    ("banana", "apple", "orange", "A {color} banana, a {color} apple, and a {color} orange in a fruit bowl."),
    ("strawberry", "mango", "lemon", "A {color} strawberry, a {color} mango, and a {color} lemon on a plate."),
    ("papaya", "guava", "carrot", "A {color} papaya, a {color} guava, and a {color} carrot on the table."),
    ("broccoli", "orange", "apple", "A {color} broccoli beside a {color} orange and a {color} apple on the counter."),
    
    # Vehicles
    ("car", "bus", "truck", "A {color} car parked near a {color} bus and a {color} truck on the street."),
    ("bicycle", "motorcycle", "airplane", "A {color} bicycle, a {color} motorcycle, and a {color} airplane at the airport."),
    ("boat", "jeep", "airplane", "A {color} boat, a {color} jeep, and a {color} airplane at the lakeside."),
    
    # Animals
    ("dog", "cat", "horse", "A {color} dog next to a {color} cat and a {color} horse in the yard."),
    ("cow", "sheep", "elephant", "A {color} cow, a {color} sheep, and a {color} elephant at the zoo."),
    ("bear", "turtle", "crocodile", "A {color} bear near a {color} turtle and a {color} crocodile by the river."),
    
    # Furniture/Household
    ("chair", "desk", "bookcase", "A {color} chair, a {color} desk, and a {color} bookcase in the office."),
    ("couch", "table", "vase", "A {color} couch with a {color} table and a {color} vase in the living room."),
    ("mug", "teapot", "candle", "A {color} mug, a {color} teapot, and a {color} candle on the dining table."),
    ("pillow", "curtain", "clock", "A {color} pillow on the bed, a {color} curtain by the window, and a {color} clock on the wall."),
    ("wardrobe", "bookcase", "desk", "A {color} wardrobe beside a {color} bookcase and a {color} desk in the room."),
    
    # Clothing/Accessories
    ("tie", "hat", "jacket", "A {color} tie, a {color} hat, and a {color} jacket on the bed."),
    ("backpack", "umbrella", "suitcase", "A {color} backpack, a {color} umbrella, and a {color} suitcase by the door."),
    ("T-shirt", "jeans", "socks", "A {color} T-shirt, {color} jeans, and {color} socks on the bed."),
    ("handbag", "wallet", "hat", "A {color} handbag with a {color} wallet and a {color} hat on the table."),
    
    # Sports/Toys
    ("skateboard", "sports ball", "baseball bat", "A {color} skateboard beside a {color} sports ball and a {color} baseball bat."),
    ("surfboard", "kite", "frisbee", "A {color} surfboard, a {color} kite, and a {color} frisbee on the beach."),
    ("teddybear", "balloon", "football helmet", "A {color} teddybear with a {color} balloon next to a {color} football helmet."),
    
    # Tools/Appliances
    ("microwave", "toaster", "refrigerator", "A {color} microwave, a {color} toaster, and a {color} refrigerator in the kitchen."),
    ("hammer", "wrench", "saw", "A {color} hammer, a {color} wrench, and a {color} saw on the workbench."),
    ("remote", "computer mouse", "fan", "A {color} remote, a {color} computer mouse, and a {color} fan on the desk."),
    
    # Cross-Category
    ("backpack", "bicycle", "umbrella", "A {color} backpack on a {color} bicycle next to a {color} umbrella."),
    ("book", "mug", "candle", "A {color} book, a {color} mug, and a {color} candle on the coffee table."),
    ("hat", "suitcase", "umbrella", "A {color} hat, a {color} suitcase, and a {color} umbrella at the entrance."),
    ("clock", "vase", "book", "A {color} clock, a {color} vase, and a {color} book on the shelf."),
]


# TASK4_TEMPLATES for compatibility
TASK4_TEMPLATES = [t[3] for t in TEMPLATES_WITH_GT]


def extract_objects_from_multi_template(template: str) -> List[str]:
    """
    Extract object list from a Task 4 template.
    
    Args:
        template: Template string with {color} placeholders
        
    Returns:
        List of object names in order
    """
    # Look up in TEMPLATES_WITH_GT
    for obj1, obj2, obj3, tmpl in TEMPLATES_WITH_GT:
        if tmpl == template:
            return [obj1, obj2] + ([obj3] if obj3 else [])
    
    # Fallback: regex extraction
    pattern = r'\{color\}\s+([a-zA-Z][a-zA-Z\s]*?)(?=\s+(?:and|,|on|in|beside|next|near|at|with|flying|parked|sitting|resting|hanging|leaning|holding|floating|\.|\,))'
    matches = re.findall(pattern, template)
    
    if matches:
        return [m.strip() for m in matches]
    
    return []


def _get_valid_objects() -> set:
    """Get set of valid object names from objects.csv."""
    objects = load_objects()
    return {obj.name for obj in objects}


def _check_exclusion_violation(objects: List[str]) -> Optional[set]:
    """Check if objects violate exclusion rules."""
    obj_set = set(objects)
    for group in EXCLUSION_GROUPS:
        overlap = obj_set & group
        if len(overlap) > 1:
            return overlap
    return None


class MultiObjectCompositionGenerator(PromptGenerator):
    """Generator for Task 4: Multi-Object Color Composition."""
    
    task_name = "multi_object_composition"
    
    def __init__(
        self,
        color_system: str,
        seed: Optional[int] = None,
        templates: Optional[List[tuple]] = None,
    ):
        super().__init__(color_system, seed)
        self.templates = templates if templates is not None else TEMPLATES_WITH_GT
    
    def get_output_columns(self) -> List[str]:
        return ["id", "color_count", "object1", "color1", "object2", "color2", "object3", "color3", "prompt"]
    
    def generate(self, n_prompts: int) -> pd.DataFrame:
        if self.seed is not None:
            set_seed(self.seed)
        
        all_colors = self.sampler.get_all_colors()
        color_usage = {c.name: 0 for c in all_colors}
        templates = self._select_templates_stratified(n_prompts)
        
        records = []
        for idx, (obj1, obj2, obj3, template) in enumerate(templates):
            n_colors_needed = 3 if obj3 else 2
            colors = self._sample_colors_for_coverage(n_colors_needed, color_usage)
            
            for c in colors:
                color_usage[c.name] += 1
            
            prompt = template
            for c in colors:
                prompt = prompt.replace("{color}", c.name, 1)
            
            records.append({
                "id": idx + 1,
                "color_count": n_colors_needed,
                "object1": obj1,
                "color1": colors[0].name,
                "object2": obj2,
                "color2": colors[1].name,
                "object3": obj3 if obj3 else "",
                "color3": colors[2].name if n_colors_needed == 3 else "",
                "prompt": prompt,
            })
        
        return pd.DataFrame(records, columns=self.get_output_columns())
    
    def _select_templates_stratified(self, n: int) -> List[tuple]:
        n_templates = len(self.templates)
        if n <= n_templates:
            return random.sample(self.templates, n)
        
        result = []
        full_cycles = n // n_templates
        for _ in range(full_cycles):
            cycle = list(self.templates)
            random.shuffle(cycle)
            result.extend(cycle)
        
        remainder = n % n_templates
        if remainder > 0:
            result.extend(random.sample(self.templates, remainder))
        return result
    
    def _sample_colors_for_coverage(self, n: int, usage: dict) -> List:
        all_colors = self.sampler.get_all_colors()
        sorted_colors = sorted(all_colors, key=lambda c: usage[c.name])
        min_usage = min(usage.values())
        underused = [c for c in sorted_colors if usage[c.name] == min_usage]
        
        if len(underused) >= n:
            return random.sample(underused, n)
        else:
            result = list(underused)
            remaining = [c for c in all_colors if c not in result]
            random.shuffle(remaining)
            result.extend(remaining[:n - len(result)])
            return result[:n]


def generate_multi_object_composition(
    color_system: str,
    n_prompts: int,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Convenience function to generate Task 4 prompts."""
    generator = MultiObjectCompositionGenerator(color_system, seed)
    df = generator.generate(n_prompts)
    if output_dir:
        generator.save(df, output_dir)
    return df
