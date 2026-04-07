"""
GenColorBench CLI - Unified Command-Line Interface (v3 Naming).

Commands:
    mini        Generate mini benchmark (~10K prompts)
    full        Generate full benchmark (~50K prompts)
    images      Generate images from prompts using T2I models
    evaluate    Run evaluation pipeline on generated images

Usage:
    python -m gencolorbench mini --output-dir ./mini_bench_prompt --seed 42
    python -m gencolorbench full --output-dir ./full_bench_prompt --seed 42
    python -m gencolorbench images --model flux-dev --prompts-dir ./gencolorbench/mini_bench_prompt
    python -m gencolorbench evaluate --prompts-dir ./mini_bench_prompt --images-dir ./generated_images/flux-dev

Naming Convention (v3):
    CNA  = Color Name Accuracy (was Task 1)
    NCU  = Numeric Color Understanding (was Task 2)
    COA  = Color-Object Association (was Task 3)
    MOC  = Multi-Object Color Composition (was Task 4)
    ICA  = Implicit Color Association (was Task 5)
"""

import argparse
import sys
from pathlib import Path


def cmd_mini(args):
    """Generate mini benchmark."""
    from .mini_benchmark import generate_mini_benchmark, MiniConfig
    
    config = MiniConfig(
        l3_shades_per_l1=args.l3_shades,
        css_sample_size=args.css_sample,
        object_sample_ratio=args.object_ratio,
        seed=args.seed,
    )
    
    results = generate_mini_benchmark(
        output_dir=args.output_dir,
        config=config,
        seed=args.seed,
        verbose=not args.quiet,
    )
    
    return results


def cmd_full(args):
    """Generate full benchmark."""
    from .full_benchmark import generate_full_benchmark, FullBenchmarkConfig
    
    config = FullBenchmarkConfig(
        l3_shades_per_l2=args.l3_shades,
        seed=args.seed,
    )
    
    results = generate_full_benchmark(
        output_dir=args.output_dir,
        config=config,
        seed=args.seed,
        verbose=not args.quiet,
    )
    
    return results


def cmd_images(args):
    """Generate images from prompts."""
    from .generation import run_generation
    
    print("GenColorBench Image Generation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"CSV: {args.csv}")
    print(f"Output: {args.output_dir}")
    print(f"Images per prompt: {args.images_per_prompt}")
    print("=" * 60)
    print()
    
    # Run generation
    summary = run_generation(
        csv_path=args.csv,
        output_dir=args.output_dir,
        model=args.model,
        device=args.device,
        images_per_prompt=args.images_per_prompt,
        id_column=args.id_column,
        prompt_column=args.prompt_column,
        filter_csv=args.filter_csv,
        resume=not args.no_resume,
    )
    
    print(f"\n✓ Image generation complete!")
    print(f"  Processed: {summary['processed']}")
    print(f"  Skipped: {summary['skipped']}")
    print(f"  Failed: {summary['failed']}")


def cmd_evaluate(args):
    """Run evaluation pipeline."""
    from .config import EvalConfig, parse_args as parse_eval_args
    from .pipeline import run_evaluation
    
    # Build config from CLI args
    config = EvalConfig(
        prompts_dir=args.prompts_dir,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        neg_csv=args.neg_csv,
        colors_dir=args.colors_dir,
        color_tables_dir=args.color_tables_dir,
        sam2_checkpoint=args.sam2_checkpoint,
        images_per_prompt=args.images_per_prompt,
        csv_filter=args.csv_filter,
        task=args.task,
        use_vlm=args.use_vlm,
        vlm_model=args.vlm_model,
        save_viz=args.save_viz,
        device=args.device,
        jnd=args.jnd,
        gt_selection_percentile=args.gt_percentile,
    )
    
    run_evaluation(config)


def main():
    parser = argparse.ArgumentParser(
        prog='gencolorbench',
        description='GenColorBench: Comprehensive Color Understanding Benchmark for T2I Models (v3)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tasks (v3 naming):
  CNA   Color Name Accuracy        "A red apple"
  NCU   Numeric Color Understanding "A ball in rgb(255,0,0)"
  COA   Color-Object Association   "A red apple on a white plate"
  MOC   Multi-Object Composition   "A red apple and a blue car"
  ICA   Implicit Color Association "A red apple next to a fire truck"

Examples:
  gencolorbench mini --output-dir ./prompts
  gencolorbench images --model flux-dev --prompts-dir ./prompts
  gencolorbench evaluate --prompts-dir ./prompts --images-dir ./images --task cna
""",
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # =========================================================================
    # Mini benchmark
    # =========================================================================
    mini_parser = subparsers.add_parser('mini', help='Generate mini benchmark (~10K prompts)')
    mini_parser.add_argument('--output-dir', '-o', default='./mini_bench_prompt',
                             help='Output directory for CSV files')
    mini_parser.add_argument('--seed', type=int, default=42,
                             help='Random seed')
    mini_parser.add_argument('--l3-shades', type=int, default=4,
                             help='L3 shades to sample per L1 parent')
    mini_parser.add_argument('--css-sample', type=int, default=45,
                             help='Number of CSS colors to sample')
    mini_parser.add_argument('--object-ratio', type=float, default=0.45,
                             help='Object sampling ratio per category')
    mini_parser.add_argument('--quiet', '-q', action='store_true',
                             help='Suppress output')
    mini_parser.set_defaults(func=cmd_mini)
    
    # =========================================================================
    # Full benchmark
    # =========================================================================
    full_parser = subparsers.add_parser('full', help='Generate full benchmark (~50K prompts)')
    full_parser.add_argument('--output-dir', '-o', default='./full_bench_prompt',
                             help='Output directory for CSV files')
    full_parser.add_argument('--seed', type=int, default=42,
                             help='Random seed')
    full_parser.add_argument('--l3-shades', type=int, default=5,
                             help='L3 shades to sample per L2 category')
    full_parser.add_argument('--quiet', '-q', action='store_true',
                             help='Suppress output')
    full_parser.set_defaults(func=cmd_full)
    
    # =========================================================================
    # Image generation
    # =========================================================================
    images_parser = subparsers.add_parser('images', help='Generate images from prompts')
    images_parser.add_argument('--csv', '-i', required=True,
                               help='Path to prompts CSV file')
    images_parser.add_argument('--output-dir', '-o', required=True,
                               help='Output directory for generated images')
    images_parser.add_argument('--model', '-m', default='flux-schnell',
                               help='Model to use (flux-schnell, flux-dev, sd3, sana). Default: flux-schnell')
    images_parser.add_argument('--device', '-d', default='cuda',
                               help='CUDA device (e.g., cuda, cuda:0). Default: cuda')
    images_parser.add_argument('--images-per-prompt', '-n', type=int, default=4,
                               help='Images to generate per prompt. Default: 4')
    images_parser.add_argument('--id-column', default='id',
                               help='CSV column for prompt IDs. Default: id')
    images_parser.add_argument('--prompt-column', default='prompt',
                               help='CSV column for prompts. Default: prompt')
    images_parser.add_argument('--filter-csv', default=None,
                               help='CSV with IDs to process (for resuming specific prompts)')
    images_parser.add_argument('--no-resume', action='store_true',
                               help="Don't resume from checkpoint, start fresh")
    images_parser.add_argument('--limit', type=int, default=None,
                               help='Limit number of prompts to process')
    images_parser.set_defaults(func=cmd_images)
    
    # =========================================================================
    # Evaluation
    # =========================================================================
    eval_parser = subparsers.add_parser('evaluate', help='Run evaluation pipeline')
    eval_parser.add_argument('--prompts-dir', '-p', required=True,
                             help='Directory containing prompt CSVs')
    eval_parser.add_argument('--images-dir', '-i', required=True,
                             help='Directory containing generated images')
    eval_parser.add_argument('--output-dir', '-o', default='./eval_results',
                             help='Output directory for results')
    eval_parser.add_argument('--neg-csv', default=None,
                             help='Path to negative labels CSV')
    eval_parser.add_argument('--colors-dir', default=None,
                             help='Directory containing color neighborhood CSVs')
    eval_parser.add_argument('--color-tables-dir', default=None,
                             help='Directory containing color table CSVs')
    eval_parser.add_argument('--sam2-checkpoint', default='./checkpoints/sam2.1_hiera_large.pt',
                             help='Path to SAM2 checkpoint')
    eval_parser.add_argument('--images-per-prompt', '-n', type=int, default=4,
                             help='Number of images per prompt')
    eval_parser.add_argument('--csv-filter', default=None,
                             help='Filter CSVs by prefix (e.g., cna, coa_l2)')
    eval_parser.add_argument('--task', default='all',
                             choices=['all', 'cna', 'ncu', 'coa', 'moc', 'ica',
                                      'task1', 'task2', 'task3', 'task4', 'task5'],
                             help='Task to evaluate (v3 names or legacy task1-5)')
    eval_parser.add_argument('--use-vlm', action='store_true',
                             help='Use VLM for verification (COA/MOC/ICA)')
    eval_parser.add_argument('--vlm-model', default='janus-1.3b',
                             choices=['janus-1.3b', 'janus-pro-7b'],
                             help='VLM model to use')
    eval_parser.add_argument('--save-viz', action='store_true',
                             help='Save visualization images')
    eval_parser.add_argument('--device', default='cuda:0',
                             help='Device to use')
    eval_parser.add_argument('--jnd', type=float, default=5.0,
                             help='Just Noticeable Difference threshold')
    eval_parser.add_argument('--gt-percentile', type=float, default=10.0,
                             help='GT selection percentile for color extraction')
    eval_parser.set_defaults(func=cmd_evaluate)
    
    # =========================================================================
    # Parse and execute
    # =========================================================================
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == '__main__':
    main()
