"""
Main evaluation pipeline for GenColorBench.

Handles:
- CSV processing loop
- Image iteration
- Result aggregation
- Checkpointing
- Category-wise accuracy reporting
"""

import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Any

from .config import EvalConfig, setup_environment, setup_cache_environment
from .data import load_negative_labels, load_color_neighborhoods, load_color_tables
from .models import load_segmentation_models, load_vlm_model, SegmentationModels, VLMModels
from .evaluation import get_evaluator, TASK_EVALUATORS, TASK_DISPLAY_NAMES
from .evaluation.registry import (
    detect_task, detect_color_system,
    build_object_category_map, compute_category_accuracy
)
from .utils import NumpyEncoder, save_json


def process_csv(
    csv_path: Path,
    images_dir: Path,
    output_dir: Path,
    seg_models: SegmentationModels,
    vlm_models: Optional[VLMModels],
    neg_labels_dict: Dict[str, List[str]],
    color_tables: Dict[str, pd.DataFrame],
    neighborhoods: Dict[str, pd.DataFrame],
    obj_to_category: Dict[str, str],
    config: EvalConfig,
) -> Dict[str, Any]:
    """
    Process a single CSV file.
    
    Args:
        csv_path: Path to CSV file
        images_dir: Base images directory
        output_dir: Output directory for results
        seg_models: Segmentation models
        vlm_models: VLM models (optional)
        neg_labels_dict: Negative labels lookup
        color_tables: Color lookup tables
        neighborhoods: Color neighborhoods
        obj_to_category: Object to category mapping
        config: Evaluation configuration
    
    Returns:
        Summary dictionary with results
    """
    task = detect_task(csv_path.name)
    color_system = detect_color_system(csv_path.name)
    csv_stem = csv_path.stem
    
    # Get task display name
    task_display = TASK_DISPLAY_NAMES.get(task, task.upper())
    
    # Determine if VLM should be used for this task
    evaluator_class = get_evaluator(task)
    task_use_vlm = config.use_vlm and evaluator_class.use_vlm_for_task
    
    vlm_status = "enabled" if task_use_vlm else "skipped (single object task)"
    print(f"  Task: {task_display} ({task}), Color System: {color_system}, VLM: {vlm_status}")
    
    df = pd.read_csv(csv_path)
    
    csv_output_dir = output_dir / csv_stem
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Mapping from new CSV names to old image directory names
    NEW_TO_OLD_DIR = {
        'cna_iscc_l1': 'task1_color_name_iscc_l1',
        'cna_iscc_l2': 'task1_color_name_iscc_l2',
        'cna_iscc_l3': 'task1_color_name_iscc_l3',
        'cna_css': 'task1_color_name_css',
        'ncu_hex_css': 'task2_numeric_hex_css',
        'ncu_rgb_l1': 'task2_numeric_rgb_iscc_l1',
        'ncu_rgb_l2': 'task2_numeric_rgb_iscc_l2',
        'ncu_rgb_l3': 'task2_numeric_rgb_iscc_l3',
        'coa_iscc_l1': 'task3_color_object_iscc_l1',
        'coa_iscc_l2': 'task3_color_object_iscc_l2',
        'coa_iscc_l3': 'task3_color_object_iscc_l3',
        'coa_css': 'task3_color_object_css',
        'moc_iscc_l1': 'task4_multi_object_iscc_l1',
        'moc_iscc_l2': 'task4_multi_object_iscc_l2',
        'moc_iscc_l3': 'task4_multi_object_iscc_l3',
        'moc_css': 'task4_multi_object_css',
        'ica_iscc_l1': 'task5_implicit_iscc_l1',
        'ica_iscc_l2': 'task5_implicit_iscc_l2',
        'ica_iscc_l3': 'task5_implicit_iscc_l3',
        'ica_css': 'task5_implicit_css',
    }
    
    # Find images directory
    images_subdir = images_dir / csv_stem
    if not images_subdir.exists():
        # Try without _mini/_full suffix
        alt_name = csv_stem.replace('_mini', '').replace('_full', '')
        images_subdir = images_dir / alt_name
    
    if not images_subdir.exists():
        # Try mapping new name to old directory name
        base_name = csv_stem.replace('_mini', '').replace('_full', '')
        suffix = '_mini' if '_mini' in csv_stem else ('_full' if '_full' in csv_stem else '')
        if base_name in NEW_TO_OLD_DIR:
            old_dir_name = NEW_TO_OLD_DIR[base_name] + suffix
            images_subdir = images_dir / old_dir_name
    
    if not images_subdir.exists():
        print(f"  ⚠️ Images directory not found: {images_subdir}")
        return {
            'csv': csv_stem,
            'csv_name': csv_stem,
            'task': task,
            'task_display': task_display,
            'color_system': color_system,
            'error': 'Images not found',
            'total': 0,
            'correct': 0,
            'accuracy': 0.0,
            'category_accuracy': {},
            'results': [],
        }
    
    # Create evaluator
    evaluator = evaluator_class(
        seg_models=seg_models,
        vlm_models=vlm_models if task_use_vlm else None,
        neg_labels_dict=neg_labels_dict,
        color_tables=color_tables,
        neighborhoods=neighborhoods,
        color_system=color_system,
        jnd=config.jnd,
        use_vlm=task_use_vlm,
        save_viz=config.save_viz,
    )
    
    results = []
    correct_count = 0
    total_count = 0
    
    # Collect all image paths and rows
    work_items = []
    for _, row in df.iterrows():
        prompt_id = str(row['id'])
        
        for img_idx in range(1, config.images_per_prompt + 1):
            # New flat structure: {prompt_id}_{image_id}.png
            image_path = images_subdir / f"{prompt_id}_{img_idx}.png"
            
            # Fallback to old structure: {prompt_id}/image_{image_id}.png
            if not image_path.exists():
                prompt_dir = images_subdir / prompt_id
                image_path = prompt_dir / f"image_{img_idx}.png"
            
            if image_path.exists():
                work_items.append((image_path, row, prompt_id, img_idx))
    
    # Process images
    for idx, (image_path, row, prompt_id, img_idx) in enumerate(tqdm(work_items, desc=f"  {csv_stem}")):
        result = evaluator.evaluate(
            image_path=image_path,
            row=row,
            output_dir=csv_output_dir,
            prompt_id=prompt_id,
            image_idx=img_idx,
        )
        
        result['prompt_id'] = prompt_id
        result['image_idx'] = img_idx
        results.append(result)
        
        total_count += 1
        if result.get('correct', False):
            correct_count += 1
        
        # Checkpoint every 50 iterations
        if (idx + 1) % 50 == 0:
            checkpoint = {
                'csv': csv_stem,
                'task': task,
                'color_system': color_system,
                'processed': idx + 1,
                'total': len(work_items),
                'correct': correct_count,
                'accuracy': correct_count / total_count if total_count > 0 else 0,
                'results': results
            }
            save_json(checkpoint, csv_output_dir / 'checkpoint.json')
    
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    print(f"  Accuracy: {accuracy*100:.2f}% ({correct_count}/{total_count})")
    
    # Compute category-wise accuracy
    # Determine object key based on task
    if task == 'coa':
        object_key = 'main_obj'
    elif task == 'ica':
        object_key = 'object'
    elif task == 'moc':
        object_key = 'object'  # Will need special handling
    else:
        object_key = 'object'
    
    category_accuracy = compute_category_accuracy(results, obj_to_category, object_key)
    
    # Print category accuracy
    if category_accuracy:
        print(f"  Category-wise accuracy:")
        for cat, stats in sorted(category_accuracy.items()):
            print(f"    {cat}: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})")
    
    # Build summary with all required keys
    summary = {
        'csv': csv_stem,
        'csv_name': csv_stem,
        'task': task,
        'task_display': task_display,
        'color_system': color_system,
        'total': total_count,
        'correct': correct_count,
        'accuracy': accuracy,
        'category_accuracy': category_accuracy,
        'results': results,
    }
    return summary


def run_evaluation(config: EvalConfig) -> Dict[str, Any]:
    """
    Run full evaluation pipeline.
    
    Args:
        config: Evaluation configuration
    
    Returns:
        Overall summary dictionary
    """
    # Setup environment
    device = setup_environment(config.device)
    setup_cache_environment(config.cache_dir)
    
    # Load models
    seg_models = load_segmentation_models(
        sam2_checkpoint=str(config.sam2_checkpoint),
        sam2_config=config.sam2_config,
        grounding_model_id=config.grounding_model,
        device=device,
        cache_dir=config.cache_dir,
    )
    
    vlm_models = None
    if config.use_vlm:
        vlm_models = load_vlm_model(
            model_path=config.vlm_model_path,
            device=device,
            cache_dir=config.cache_dir,
        )
    
    # Load data
    neg_labels_dict = load_negative_labels(config.neg_csv)
    
    # Load neg_labels DataFrame separately for category mapping
    neg_labels_df = pd.read_csv(config.neg_csv) if config.neg_csv.exists() else pd.DataFrame()
    
    # Build object to category mapping
    obj_to_category = build_object_category_map(neg_labels_df)
    print(f"  Built category map for {len(obj_to_category)} objects")
    
    neighborhoods = {}
    if config.colors_dir:
        neighborhoods = load_color_neighborhoods(config.colors_dir)
    
    color_tables = {}
    color_tables_dir = config.color_tables_path
    if color_tables_dir:
        color_tables = load_color_tables(color_tables_dir)
    
    # Find CSVs
    csv_files = sorted(config.prompts_dir.glob("*.csv"))
    
    # Filter by task
    if config.task != "all":
        task_patterns = {
            'cna': ['cna', 'color_name'],
            'ncu': ['ncu', 'numeric'],
            'coa': ['coa', 'color_object'],
            'moc': ['moc', 'multi_object'],
            'ica': ['ica', 'implicit'],
            # Backward compatibility
            'task1': ['cna', 'color_name', 'task1'],
            'task2': ['ncu', 'numeric', 'task2'],
            'task3': ['coa', 'color_object', 'task3'],
            'task4': ['moc', 'multi_object', 'task4'],
            'task5': ['ica', 'implicit', 'task5'],
        }
        patterns = task_patterns.get(config.task, [config.task])
        csv_files = [f for f in csv_files if any(p in f.name.lower() for p in patterns)]
    
    if config.csv_filter:
        csv_files = [f for f in csv_files if config.csv_filter in f.name]
    
    print(f"\nFound {len(csv_files)} CSV files to process")
    
    # Print optimization settings
    print("\n" + "=" * 60)
    print("OPTIMIZATION SETTINGS")
    print("=" * 60)
    print(f"  VLM enabled: {config.use_vlm}")
    if config.use_vlm:
        print(f"  VLM variant: {config.vlm_variant}")
        print(f"  VLM for Tasks 1/2: SKIPPED (single object)")
        print(f"  VLM for Tasks 3/4/5: ENABLED (single-call optimized)")
    print(f"  Save visualizations: {config.save_viz}")
    print("=" * 60)
    
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    task_results = {}
    
    for csv_path in csv_files:
        print(f"\nProcessing: {csv_path.name}")
        
        summary = process_csv(
            csv_path=csv_path,
            images_dir=config.images_dir,
            output_dir=config.output_dir,
            seg_models=seg_models,
            vlm_models=vlm_models,
            neg_labels_dict=neg_labels_dict,
            color_tables=color_tables,
            neighborhoods=neighborhoods,
            obj_to_category=obj_to_category,
            config=config,
        )
        
        result_entry = {
            'csv': summary['csv'],
            'task': summary['task'],
            'task_display': summary.get('task_display', summary['task'].upper()),
            'color_system': summary.get('color_system', 'unknown'),
            'accuracy': summary.get('accuracy', 0),
            'total': summary.get('total', 0),
            'correct': summary.get('correct', 0),
            'category_accuracy': summary.get('category_accuracy', {}),
        }
        all_results.append(result_entry)
        
        # Group by task
        task = summary['task']
        if task not in task_results:
            task_results[task] = []
        task_results[task].append(result_entry)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    for task in sorted(task_results.keys()):
        task_display = TASK_DISPLAY_NAMES.get(task, task.upper())
        print(f"\n{task_display} ({task}):")
        print("-" * 40)
        task_total = 0
        task_correct = 0
        for r in task_results[task]:
            print(f"  {r['csv']}: {r['accuracy']*100:.2f}% ({r['correct']}/{r['total']})")
            task_total += r['total']
            task_correct += r['correct']
            
            # Print category breakdown if available
            if r.get('category_accuracy'):
                for cat, stats in sorted(r['category_accuracy'].items()):
                    print(f"    └─ {cat}: {stats['accuracy']:.2f}%")
        
        if task_total > 0:
            print(f"  --- Overall: {task_correct/task_total*100:.2f}% ({task_correct}/{task_total})")
    
    # Overall summary
    total_all = sum(r['total'] for r in all_results)
    correct_all = sum(r['correct'] for r in all_results)
    if total_all > 0:
        print(f"\n{'='*60}")
        print(f"OVERALL: {correct_all/total_all*100:.2f}% ({correct_all}/{total_all})")
        print(f"{'='*60}")
    
    # Save summaries
    overall_summary = {
        'all_results': all_results,
        'by_task': task_results,
        'overall': {
            'total': total_all,
            'correct': correct_all,
            'accuracy': correct_all / total_all if total_all > 0 else 0
        }
    }
    
    summary_path = config.output_dir / 'summary_all.json'
    save_json(overall_summary, summary_path)
    
    print(f"\nResults saved to: {config.output_dir}")
    
    return overall_summary
