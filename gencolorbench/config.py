"""
Configuration and argument parsing for GenColorBench evaluation.
"""

import argparse
import os
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any


# =============================================================================
# Constants
# =============================================================================

# Base directory for GenColorBench data on cudahpc44
DEFAULT_BASE_DIR = "/data/144-1/users/mabutt/gencolorbench_v4"
DEFAULT_CACHE_DIR = f"{DEFAULT_BASE_DIR}/cache"
DEFAULT_JND = 5.0
DEFAULT_IMAGES_PER_PROMPT = 4
DEFAULT_GROUNDING_MODEL = "IDEA-Research/grounding-dino-base"

# Default paths (relative to gsam2/ or absolute)
DEFAULT_NEG_CSV = f"{DEFAULT_BASE_DIR}/data/objects/obj_neg.csv"
DEFAULT_COLORS_DIR = f"{DEFAULT_BASE_DIR}/data/neighborhoods"
DEFAULT_COLOR_TABLES_DIR = f"{DEFAULT_BASE_DIR}/data/color_systems"
DEFAULT_SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

VLM_VARIANTS = {
    "1.3B": "deepseek-ai/Janus-1.3B",
    "7B": "deepseek-ai/Janus-Pro-7B",
}

TASK_PATTERNS = {
    'task1': ['task1', 'color_name'],
    'task2': ['task2', 'numeric'],
    'task3': ['task3', 'color_object'],
    'task4': ['task4', 'multi_object'],
    'task5': ['task5', 'implicit'],
}


# =============================================================================
# Configuration Dataclass
# =============================================================================

@dataclass
class EvalConfig:
    """Configuration for GenColorBench evaluation pipeline."""
    
    # Paths
    prompts_dir: Path
    images_dir: Path
    output_dir: Path
    neg_csv: Path
    colors_dir: Optional[Path] = None
    color_tables_dir: Optional[Path] = None
    
    # Model paths
    sam2_checkpoint: Path = None
    sam2_config: str = None
    gsam2_dir: Path = Path("./gsam2")
    grounding_model: str = DEFAULT_GROUNDING_MODEL
    vlm_model: Optional[str] = None
    vlm_variant: str = "1.3B"
    
    # Evaluation settings
    images_per_prompt: int = DEFAULT_IMAGES_PER_PROMPT
    jnd: float = DEFAULT_JND
    gt_selection_percentile: float = 0.1
    device: str = "cuda"
    
    # Feature flags
    use_vlm: bool = False
    save_viz: bool = False
    parallel: bool = False
    num_workers: int = 4
    
    # Filtering
    task: str = "all"
    csv_filter: Optional[str] = None
    
    # Cache
    cache_dir: str = DEFAULT_CACHE_DIR
    
    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.prompts_dir, str):
            self.prompts_dir = Path(self.prompts_dir)
        if isinstance(self.images_dir, str):
            self.images_dir = Path(self.images_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.neg_csv, str):
            self.neg_csv = Path(self.neg_csv)
        if self.colors_dir and isinstance(self.colors_dir, str):
            self.colors_dir = Path(self.colors_dir)
        if self.color_tables_dir and isinstance(self.color_tables_dir, str):
            self.color_tables_dir = Path(self.color_tables_dir)
        if isinstance(self.sam2_checkpoint, str):
            self.sam2_checkpoint = Path(self.sam2_checkpoint)
        if isinstance(self.gsam2_dir, str):
            self.gsam2_dir = Path(self.gsam2_dir)
    
    @property
    def vlm_model_path(self) -> str:
        """Get the VLM model path based on variant or custom path."""
        if self.vlm_model:
            return self.vlm_model
        return VLM_VARIANTS.get(self.vlm_variant, VLM_VARIANTS["1.3B"])
    
    @property
    def color_tables_path(self) -> Optional[Path]:
        """Get color tables directory (fallback to colors_dir)."""
        return self.color_tables_dir or self.colors_dir


# =============================================================================
# Environment Setup
# =============================================================================

def setup_environment(device: str) -> str:
    """
    Setup torch environment with appropriate optimizations.
    
    Args:
        device: Device string (e.g., "cuda:0", "cpu")
    
    Returns:
        Validated device string
    """
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        return device
    return "cpu"


def setup_cache_environment(cache_dir: str):
    """
    Setup HuggingFace cache environment variables.
    
    Args:
        cache_dir: Base cache directory path
    """
    os.environ["HF_HOME"] = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = f"{cache_dir}/hub"
    os.environ["TRANSFORMERS_CACHE"] = f"{cache_dir}/transformers"
    os.environ["HF_DATASETS_CACHE"] = f"{cache_dir}/datasets"
    os.environ["TOKENIZERS_CACHE"] = f"{cache_dir}/tokenizers"
    
    # Ensure directories exist
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(f"{cache_dir}/hub", exist_ok=True)
    os.makedirs(f"{cache_dir}/transformers", exist_ok=True)


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args() -> EvalConfig:
    """Parse command line arguments and return EvalConfig."""
    parser = argparse.ArgumentParser(
        description="GenColorBench Evaluation Pipeline v2 - All Tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required paths
    parser.add_argument("--prompts-dir", type=str, required=True,
                        help="Directory containing prompt CSV files")
    parser.add_argument("--images-dir", type=str, required=True,
                        help="Directory containing generated images")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for results")
    
    # Paths with defaults
    parser.add_argument("--neg-csv", type=str, default=DEFAULT_NEG_CSV,
                        help="Path to negative labels CSV")
    parser.add_argument("--colors-dir", type=str, default=DEFAULT_COLORS_DIR,
                        help="Directory containing color neighborhood files")
    parser.add_argument("--color-tables-dir", type=str, default=DEFAULT_COLOR_TABLES_DIR,
                        help="Directory containing color lookup tables")
    
    # Model paths with defaults
    parser.add_argument("--sam2-checkpoint", type=str, default=DEFAULT_SAM2_CHECKPOINT,
                        help="Path to SAM2 checkpoint")
    parser.add_argument("--sam2-config", type=str, default=DEFAULT_SAM2_CONFIG,
                        help="SAM2 config name")
    parser.add_argument("--gsam2-dir", type=str, default=".",
                        help="Path to gsam2 directory (for Hydra config resolution)")
    parser.add_argument("--grounding-model", type=str, default=DEFAULT_GROUNDING_MODEL,
                        help="GroundingDINO model ID")
    parser.add_argument("--vlm-model", type=str, default=None,
                        help="Custom VLM model path (overrides --vlm-variant)")
    parser.add_argument("--vlm-variant", type=str, default="1.3B",
                        choices=["1.3B", "7B"],
                        help="Janus VLM variant: 1.3B (~3GB VRAM) or 7B (~14GB VRAM)")
    
    # Evaluation settings
    parser.add_argument("--images-per-prompt", type=int, default=DEFAULT_IMAGES_PER_PROMPT,
                        help="Number of images per prompt")
    parser.add_argument("--jnd", type=float, default=DEFAULT_JND,
                        help="Just Noticeable Difference threshold")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda:N or cpu)")
    
    # Feature flags
    parser.add_argument("--use-vlm", action="store_true",
                        help="Enable VLM prior check (only for Tasks 3,4,5)")
    parser.add_argument("--save-viz", action="store_true",
                        help="Save visualizations")
    parser.add_argument("--parallel", action="store_true",
                        help="Enable parallel color extraction (CPU-bound, experimental)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of worker threads for parallel processing")
    
    # Filtering
    parser.add_argument("--task", type=str, default="all",
                        choices=["all", "cna", "ncu", "coa", "moc", "ica", "task1", "task2", "task3", "task4", "task5"],
                        help="Task to evaluate (default: all)")
    parser.add_argument("--csv-filter", type=str, default=None,
                        help="Filter CSV files by substring")
    
    # Cache
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR,
                        help="Cache directory for models")
    
    args = parser.parse_args()
    
    return EvalConfig(
        prompts_dir=args.prompts_dir,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        neg_csv=args.neg_csv,
        colors_dir=args.colors_dir,
        color_tables_dir=args.color_tables_dir,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        gsam2_dir=args.gsam2_dir,
        grounding_model=args.grounding_model,
        vlm_model=args.vlm_model,
        vlm_variant=args.vlm_variant,
        images_per_prompt=args.images_per_prompt,
        jnd=args.jnd,
        device=args.device,
        use_vlm=args.use_vlm,
        save_viz=args.save_viz,
        parallel=args.parallel,
        num_workers=args.num_workers,
        task=args.task,
        csv_filter=args.csv_filter,
        cache_dir=args.cache_dir,
    )


# =============================================================================
# Task Detection Utilities
# =============================================================================

def detect_task(filename: str) -> str:
    """
    Detect task type from filename.
    
    Args:
        filename: CSV filename
    
    Returns:
        Task identifier (task1-task5)
    """
    filename_lower = filename.lower()
    
    for task, patterns in TASK_PATTERNS.items():
        if any(p in filename_lower for p in patterns):
            return task
    
    return 'task1'  # Default


def detect_color_system(filename: str) -> str:
    """
    Detect color system from filename.
    
    Args:
        filename: CSV filename
    
    Returns:
        Color system identifier (css, l1, l2, l3)
    """
    filename_lower = filename.lower()
    
    if '_css' in filename_lower or 'css_' in filename_lower:
        return 'css'
    elif '_l3' in filename_lower or 'iscc_l3' in filename_lower:
        return 'l3'
    elif '_l2' in filename_lower or 'iscc_l2' in filename_lower:
        return 'l2'
    elif '_l1' in filename_lower or 'iscc_l1' in filename_lower:
        return 'l1'
    
    return 'l2'  # Default


def filter_csv_files(csv_files: List[Path], task: str, csv_filter: Optional[str]) -> List[Path]:
    """
    Filter CSV files by task and optional substring filter.
    
    Args:
        csv_files: List of CSV file paths
        task: Task identifier or "all"
        csv_filter: Optional substring to filter by
    
    Returns:
        Filtered list of CSV paths
    """
    filtered = csv_files
    
    if task != "all":
        patterns = TASK_PATTERNS.get(task, [])
        filtered = [f for f in filtered if any(p in f.name.lower() for p in patterns)]
    
    if csv_filter:
        filtered = [f for f in filtered if csv_filter in f.name]
    
    return filtered
