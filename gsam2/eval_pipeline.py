#!/usr/bin/env python3
"""
GenColorBench Evaluation Pipeline v2 - Entry Point

This script must be run from within the gsam2/ directory to properly
resolve SAM2 and GroundingDINO dependencies.

The actual evaluation logic is in the gencolorbench package.

Usage:
    cd gsam2/
    python eval_pipeline.py \
        --prompts-dir ../mini_bench_prompt/ \
        --images-dir ../flux-dev/ \
        --output-dir ../eval_results/ \
        --neg-csv ../data/objects/obj_neg.csv \
        --colors-dir ../data/neighborhoods/ \
        --color-tables-dir ../data/color_systems/ \
        --sam2-checkpoint ./checkpoints/sam2.1_hiera_large.pt \
        --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
        --device cuda:0 \
        --use-vlm \
        --vlm-variant 1.3B \
        --save-viz \
        --task all

Reference: GenColorBench (arXiv:2510.20586)
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for gencolorbench package
SCRIPT_DIR = Path(__file__).parent.absolute()
PARENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PARENT_DIR))

# Now import from gencolorbench
from gencolorbench import parse_args, run_evaluation


def main():
    """Main entry point."""
    # Parse arguments
    config = parse_args()
    
    # Run evaluation
    summary = run_evaluation(config)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
