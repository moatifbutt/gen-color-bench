"""
Image generation runner with checkpointing.

Processes task CSV files and generates images with resume support.
"""

import os
os.environ["HF_HOME"] = "/data/144-1/users/mabutt/gencolorbench_v4/cache"
os.environ["TRANSFORMERS_CACHE"] = "/data/144-1/users/mabutt/gencolorbench_v4/cache"
os.environ["HF_DATASETS_CACHE"] = "/data/144-1/users/mabutt/gencolorbench_v4/cache"

import argparse
import gc
import json
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import torch

from .models import MODEL_CONFIGS, load_pipeline, generate_single_image


# =============================================================================
# Checkpointing
# =============================================================================

def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Load checkpoint if exists."""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return {
        "completed_csvs": [],
        "current_csv": None,
        "current_row": 0,
        "total_images": 0,
    }


def save_checkpoint(checkpoint_path: Path, state: Dict[str, Any]) -> None:
    """Save checkpoint."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, 'w') as f:
        json.dump(state, f, indent=2)


# =============================================================================
# CSV Processing
# =============================================================================

def process_csv(
    csv_path: Path,
    output_dir: Path,
    pipe,
    config: Dict[str, Any],
    images_per_prompt: int,
    checkpoint_state: Dict[str, Any],
    checkpoint_path: Path,
    start_row: int = 0,
    device: str = "cuda",
    checkpoint_interval: int = 50,
) -> None:
    """
    Process a single CSV file.
    
    Args:
        csv_path: Path to CSV file
        output_dir: Base output directory
        pipe: Loaded T2I pipeline
        config: Model configuration
        images_per_prompt: Number of images per prompt
        checkpoint_state: Current checkpoint state
        checkpoint_path: Path to checkpoint file
        start_row: Row to start from (for resume)
        device: Device for generation
        checkpoint_interval: Save checkpoint every N rows
    """
    df = pd.read_csv(csv_path)
    csv_name = csv_path.stem
    
    # Create output directory for this CSV
    csv_output_dir = output_dir / csv_name
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Processing: {csv_path.name}")
    print(f"Total rows: {len(df)}, Starting from: {start_row}")
    print(f"Output: {csv_output_dir}")
    print(f"{'='*60}")
    
    for idx in range(start_row, len(df)):
        row = df.iloc[idx]
        
        # Get prompt and ID
        prompt = str(row.get('prompt', row.get('text', '')))
        prompt_id = str(row.get('id', row.get('prompt_id', idx + 1)))
        base_seed = int(row.get('seed', 42))
        
        if not prompt or prompt == 'nan':
            print(f"  [{idx+1}/{len(df)}] Skipping: no prompt")
            continue
        
        # Check if already completed (using 1-indexed naming)
        existing = list(csv_output_dir.glob(f"{prompt_id}_[1-9]*.png"))
        if len(existing) >= images_per_prompt:
            print(f"  [{idx+1}/{len(df)}] {prompt_id}: exists, skipping")
            continue
        
        print(f"  [{idx+1}/{len(df)}] Generating: {prompt_id}")
        print(f"      Prompt: {prompt[:80]}...")
        
        try:
            for img_idx in range(images_per_prompt):
                # Use 1-indexed naming to match evaluation convention
                img_num = img_idx + 1
                seed = base_seed + img_idx
                img_path = csv_output_dir / f"{prompt_id}_{img_num}.png"
                
                # Skip if this specific image exists
                if img_path.exists():
                    continue
                
                image = generate_single_image(
                    pipe=pipe,
                    config=config,
                    prompt=prompt,
                    seed=seed,
                    # device=device,
                )
                
                image.save(img_path)
            
            checkpoint_state["total_images"] += images_per_prompt
            print(f"      Saved {images_per_prompt} images")
            
        except Exception as e:
            print(f"      ERROR: {e}")
            traceback.print_exc()
            continue
        
        # Update checkpoint
        checkpoint_state["current_csv"] = str(csv_path)
        checkpoint_state["current_row"] = idx + 1
        save_checkpoint(checkpoint_path, checkpoint_state)
        
        # Periodic memory cleanup
        if (idx + 1) % checkpoint_interval == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    # Mark CSV as completed
    checkpoint_state["completed_csvs"].append(str(csv_path))
    checkpoint_state["current_csv"] = None
    checkpoint_state["current_row"] = 0
    save_checkpoint(checkpoint_path, checkpoint_state)
    
    print(f"\nCompleted: {csv_path.name}")


def run_generation(
    model_name: str,
    prompts_dir: Path,
    output_dir: Path,
    model_path: Optional[str] = None,
    device: str = "cuda",
    images_per_prompt: int = 4,
    csv_filter: Optional[str] = None,
    resume: bool = True,
    token: Optional[str] = None,
) -> None:
    """
    Run image generation for all CSVs.
    
    Args:
        model_name: Model to use
        prompts_dir: Directory containing prompt CSV files
        output_dir: Output directory for images
        model_path: Local path to model (optional)
        device: Device for generation
        images_per_prompt: Number of images per prompt
        csv_filter: Only process CSVs containing this string
        resume: Whether to resume from checkpoint
        token: HuggingFace token
    """
    prompts_dir = Path(prompts_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = output_dir / f"checkpoint_{model_name}.json"
    
    # Load or initialize checkpoint
    if resume:
        checkpoint_state = load_checkpoint(checkpoint_path)
        if checkpoint_state["completed_csvs"]:
            print(f"Resuming: {len(checkpoint_state['completed_csvs'])} CSVs completed")
        if checkpoint_state["current_csv"]:
            print(f"Resuming from: {checkpoint_state['current_csv']} row {checkpoint_state['current_row']}")
    else:
        checkpoint_state = {
            "completed_csvs": [],
            "current_csv": None,
            "current_row": 0,
            "total_images": 0,
        }
    
    # Find CSV files
    csv_files = sorted(prompts_dir.glob("*.csv"))
    if csv_filter:
        csv_files = [f for f in csv_files if csv_filter in f.name]
    
    if not csv_files:
        print(f"No CSV files found in {prompts_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"GenColorBench Image Generation")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Model path: {model_path or 'HuggingFace'}")
    print(f"Device: {device}")
    print(f"Images per prompt: {images_per_prompt}")
    print(f"CSV files: {len(csv_files)}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")
    
    # Load model
    start_time = time.time()
    pipe, config = load_pipeline(
        model_name=model_name,
        model_path=model_path,
        device=device,
        token=token,
    )
    
    # Process each CSV
    for csv_path in csv_files:
        csv_str = str(csv_path)
        
        # Skip completed CSVs
        if csv_str in checkpoint_state["completed_csvs"]:
            print(f"\nSkipping {csv_path.name}: already completed")
            continue
        
        # Determine start row
        start_row = 0
        if checkpoint_state["current_csv"] == csv_str:
            start_row = checkpoint_state["current_row"]
        
        # Process CSV
        process_csv(
            csv_path=csv_path,
            output_dir=output_dir,
            pipe=pipe,
            config=config,
            images_per_prompt=images_per_prompt,
            checkpoint_state=checkpoint_state,
            checkpoint_path=checkpoint_path,
            start_row=start_row,
            device=device,
        )
    
    # Cleanup
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Generation Complete!")
    print(f"{'='*60}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Total images: {checkpoint_state['total_images']}")
    print(f"Output: {output_dir}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GenColorBench Image Generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to use"
    )
    parser.add_argument(
        "--prompts-dir", type=str, required=True,
        help="Directory containing prompt CSV files"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for generated images"
    )
    
    # Optional arguments
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Local path to model (downloads from HF if not set)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (cuda, cuda:0, etc.)"
    )
    parser.add_argument(
        "--images-per-prompt", type=int, default=4,
        help="Number of images per prompt"
    )
    parser.add_argument(
        "--csv-filter", type=str, default=None,
        help="Only process CSVs containing this string"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Don't resume from checkpoint"
    )
    parser.add_argument(
        "--token", type=str, default=None,
        help="HuggingFace token for gated models"
    )
    
    args = parser.parse_args()
    
    run_generation(
        model_name=args.model,
        prompts_dir=Path(args.prompts_dir),
        output_dir=Path(args.output_dir),
        model_path=args.model_path,
        device=args.device,
        images_per_prompt=args.images_per_prompt,
        csv_filter=args.csv_filter,
        resume=not args.no_resume,
        token=args.token,
    )


if __name__ == "__main__":
    main()
