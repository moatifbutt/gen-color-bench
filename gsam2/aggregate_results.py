#!/usr/bin/env python3
"""
Aggregate GenColorBench evaluation results from checkpoint files.

Usage:
    python aggregate_results.py --results-dir ../eval_results/
    python aggregate_results.py --results-dir ../eval_results/ --output summary.json
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd


def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load a checkpoint JSON file."""
    with open(checkpoint_path, 'r') as f:
        return json.load(f)


def compute_task_accuracy(checkpoint: dict, neg_labels: dict = None) -> dict:
    """Compute accuracy metrics from checkpoint results."""
    results = checkpoint.get("results", [])
    
    if not results:
        return {
            "n_samples": 0,
            "accuracy": 0.0,
            "mean_delta_e": None,
            "by_category": {},
        }
    
    # Count matches and compute mean delta_e
    n_samples = len(results)
    n_correct = 0
    delta_e_values = []
    
    # Category-wise tracking
    category_stats = defaultdict(lambda: {"n_samples": 0, "n_correct": 0, "delta_e_values": []})
    
    for r in results:
        # Check for color match - actual field is "correct"
        matched = r.get("correct", r.get("color_match", r.get("matched", r.get("is_match", False))))
        if matched:
            n_correct += 1
        
        # Collect delta_e values - actual field is in metrics.ciede2000
        metrics = r.get("metrics", {})
        delta_e = metrics.get("ciede2000", r.get("delta_e", r.get("min_delta_e", None)))
        if delta_e is not None:
            delta_e_values.append(delta_e)
        
        # Get object and its category
        obj = r.get("object", r.get("main_obj", r.get("object1", "")))
        category = "Unknown"
        if neg_labels and obj in neg_labels:
            category = neg_labels[obj]
        elif "category" in r:
            category = r["category"]
        
        category_stats[category]["n_samples"] += 1
        if matched:
            category_stats[category]["n_correct"] += 1
        if delta_e is not None:
            category_stats[category]["delta_e_values"].append(delta_e)
    
    accuracy = n_correct / n_samples if n_samples > 0 else 0.0
    mean_delta_e = sum(delta_e_values) / len(delta_e_values) if delta_e_values else None
    
    # Compute category accuracies
    by_category = {}
    for cat, stats in sorted(category_stats.items()):
        cat_acc = stats["n_correct"] / stats["n_samples"] if stats["n_samples"] > 0 else 0.0
        cat_de = sum(stats["delta_e_values"]) / len(stats["delta_e_values"]) if stats["delta_e_values"] else None
        by_category[cat] = {
            "accuracy": cat_acc,
            "n_samples": stats["n_samples"],
            "n_correct": stats["n_correct"],
            "mean_delta_e": cat_de,
        }
    
    return {
        "n_samples": n_samples,
        "n_correct": n_correct,
        "accuracy": accuracy,
        "mean_delta_e": mean_delta_e,
        "by_category": by_category,
    }


def load_neg_labels(neg_csv_path: Path) -> dict:
    """Load object to category mapping from neg_csv."""
    if not neg_csv_path or not neg_csv_path.exists():
        return {}
    
    try:
        df = pd.read_csv(neg_csv_path)
        
        # Actual structure: Dataset_Category, Class_Name, Negative_Labels
        # We want: Class_Name -> Dataset_Category (cleaned)
        if "Class_Name" in df.columns and "Dataset_Category" in df.columns:
            mapping = {}
            for _, row in df.iterrows():
                obj = row["Class_Name"]
                # Clean category: "COCO_Vehicle" -> "Vehicle"
                cat = row["Dataset_Category"]
                if "_" in cat:
                    cat = cat.split("_", 1)[1]  # Take part after first underscore
                mapping[obj] = cat
            return mapping
        elif "object" in df.columns and "category" in df.columns:
            return dict(zip(df["object"], df["category"]))
        elif "name" in df.columns and "category" in df.columns:
            return dict(zip(df["name"], df["category"]))
        else:
            # Try first two columns
            return dict(zip(df.iloc[:, 1], df.iloc[:, 0]))
    except Exception as e:
        print(f"Warning: Could not load neg_csv: {e}")
        return {}


def parse_csv_name(csv_name: str) -> dict:
    """Parse task and color system from CSV name."""
    # Remove .csv extension if present
    name = csv_name.replace(".csv", "").replace("_mini", "").replace("_full", "")
    
    # v3 naming: cna_l1, cna_l2, cna_l3, cna_css, ncu_rgb_l1, ncu_hex_css, etc.
    # Old naming: task1_color_name_iscc_l1, task2_numeric_rgb_iscc_l2, etc.
    
    # Task mapping
    task_map = {
        "cna": "CNA",
        "ncu": "NCU", 
        "coa": "COA",
        "moc": "MOC",
        "ica": "ICA",
        "task1": "CNA",
        "task2": "NCU",
        "task3": "COA",
        "task4": "MOC",
        "task5": "ICA",
    }
    
    # Detect task
    task = None
    color_system = None
    
    for prefix, task_name in task_map.items():
        if name.startswith(prefix):
            task = task_name
            remainder = name[len(prefix):].lstrip("_")
            
            # Parse color system
            if "css" in remainder.lower():
                color_system = "CSS"
            elif "l1" in remainder.lower() or "iscc_l1" in remainder.lower():
                color_system = "L1"
            elif "l2" in remainder.lower() or "iscc_l2" in remainder.lower():
                color_system = "L2"
            elif "l3" in remainder.lower() or "iscc_l3" in remainder.lower():
                color_system = "L3"
            
            # Detect RGB vs HEX for NCU
            if task == "NCU":
                if "hex" in remainder.lower():
                    color_system = f"HEX-{color_system}" if color_system else "HEX"
                elif "rgb" in remainder.lower():
                    color_system = f"RGB-{color_system}" if color_system else "RGB"
            
            break
    
    return {
        "task": task or "Unknown",
        "color_system": color_system or "Unknown",
        "csv_name": csv_name,
    }


def aggregate_results(results_dir: Path, neg_csv: Path = None) -> dict:
    """Aggregate all checkpoint results."""
    results_dir = Path(results_dir)
    
    # Load object->category mapping
    neg_labels = load_neg_labels(neg_csv) if neg_csv else {}
    if neg_labels:
        print(f"Loaded {len(neg_labels)} object->category mappings")
    
    # Find all checkpoint files
    checkpoint_files = list(results_dir.glob("**/checkpoint.json"))
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {results_dir}")
        return {}
    
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    # Aggregate results
    all_results = []
    task_summaries = defaultdict(lambda: {"n_samples": 0, "n_correct": 0, "delta_e_sum": 0, "delta_e_count": 0})
    color_system_summaries = defaultdict(lambda: {"n_samples": 0, "n_correct": 0})
    
    for cp_path in sorted(checkpoint_files):
        # Get CSV name from parent directory
        csv_name = cp_path.parent.name
        
        # Load checkpoint
        checkpoint = load_checkpoint(cp_path)
        
        # Compute accuracy with category breakdown
        metrics = compute_task_accuracy(checkpoint, neg_labels)
        
        # Parse task and color system
        parsed = parse_csv_name(csv_name)
        
        result = {
            **parsed,
            **metrics,
        }
        all_results.append(result)
        
        # Aggregate by task
        task = parsed["task"]
        task_summaries[task]["n_samples"] += metrics["n_samples"]
        task_summaries[task]["n_correct"] += metrics.get("n_correct", 0)
        if metrics["mean_delta_e"] is not None:
            task_summaries[task]["delta_e_sum"] += metrics["mean_delta_e"] * metrics["n_samples"]
            task_summaries[task]["delta_e_count"] += metrics["n_samples"]
        
        # Aggregate by color system
        cs = parsed["color_system"]
        color_system_summaries[cs]["n_samples"] += metrics["n_samples"]
        color_system_summaries[cs]["n_correct"] += metrics.get("n_correct", 0)
        
        print(f"  {csv_name}: {metrics['accuracy']:.1%} ({metrics['n_correct']}/{metrics['n_samples']})")
    
    # Compute task-level accuracies
    task_accuracies = {}
    for task, summary in task_summaries.items():
        acc = summary["n_correct"] / summary["n_samples"] if summary["n_samples"] > 0 else 0
        mean_de = summary["delta_e_sum"] / summary["delta_e_count"] if summary["delta_e_count"] > 0 else None
        task_accuracies[task] = {
            "accuracy": acc,
            "n_samples": summary["n_samples"],
            "n_correct": summary["n_correct"],
            "mean_delta_e": mean_de,
        }
    
    # Compute color system accuracies
    cs_accuracies = {}
    for cs, summary in color_system_summaries.items():
        acc = summary["n_correct"] / summary["n_samples"] if summary["n_samples"] > 0 else 0
        cs_accuracies[cs] = {
            "accuracy": acc,
            "n_samples": summary["n_samples"],
            "n_correct": summary["n_correct"],
        }
    
    # Overall accuracy
    total_samples = sum(r["n_samples"] for r in all_results)
    total_correct = sum(r.get("n_correct", 0) for r in all_results)
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    return {
        "overall": {
            "accuracy": overall_accuracy,
            "n_samples": total_samples,
            "n_correct": total_correct,
        },
        "by_task": task_accuracies,
        "by_color_system": cs_accuracies,
        "detailed": all_results,
    }


def print_summary(summary: dict, show_categories: bool = True):
    """Print formatted summary."""
    print("\n" + "=" * 60)
    print("GENCOLORBENCH EVALUATION SUMMARY")
    print("=" * 60)
    
    # Overall
    overall = summary.get("overall", {})
    print(f"\nOverall Accuracy: {overall.get('accuracy', 0):.1%}")
    print(f"Total Samples: {overall.get('n_samples', 0):,}")
    print(f"Total Correct: {overall.get('n_correct', 0):,}")
    
    # By task
    print("\n" + "-" * 40)
    print("BY TASK")
    print("-" * 40)
    task_order = ["CNA", "NCU", "COA", "MOC", "ICA"]
    by_task = summary.get("by_task", {})
    for task in task_order:
        if task in by_task:
            t = by_task[task]
            de_str = f", ΔE={t['mean_delta_e']:.2f}" if t.get('mean_delta_e') else ""
            print(f"  {task}: {t['accuracy']:.1%} ({t['n_correct']}/{t['n_samples']}){de_str}")
    
    # By color system
    print("\n" + "-" * 40)
    print("BY COLOR SYSTEM")
    print("-" * 40)
    cs_order = ["L1", "L2", "L3", "CSS", "RGB-L1", "RGB-L2", "RGB-L3", "HEX-CSS"]
    by_cs = summary.get("by_color_system", {})
    for cs in cs_order:
        if cs in by_cs:
            c = by_cs[cs]
            print(f"  {cs}: {c['accuracy']:.1%} ({c['n_correct']}/{c['n_samples']})")
    
    # Any remaining
    for cs, c in by_cs.items():
        if cs not in cs_order:
            print(f"  {cs}: {c['accuracy']:.1%} ({c['n_correct']}/{c['n_samples']})")
    
    # Detailed breakdown with categories
    if show_categories:
        print("\n" + "=" * 60)
        print("DETAILED BREAKDOWN BY TASK/COLOR SYSTEM")
        print("=" * 60)
        
        detailed = summary.get("detailed", [])
        for entry in sorted(detailed, key=lambda x: (x.get("task", ""), x.get("color_system", ""))):
            task = entry.get("task", "Unknown")
            cs = entry.get("color_system", "Unknown")
            csv_name = entry.get("csv_name", "")
            acc = entry.get("accuracy", 0)
            n_correct = entry.get("n_correct", 0)
            n_samples = entry.get("n_samples", 0)
            
            print(f"\n{task} ({cs}): {acc:.2%} ({n_correct}/{n_samples})")
            print("-" * 40)
            
            by_category = entry.get("by_category", {})
            if by_category:
                for cat, cat_stats in sorted(by_category.items()):
                    cat_acc = cat_stats.get("accuracy", 0)
                    cat_n = cat_stats.get("n_samples", 0)
                    cat_correct = cat_stats.get("n_correct", 0)
                    print(f"  └─ {cat}: {cat_acc:.2%} ({cat_correct}/{cat_n})")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Aggregate GenColorBench evaluation results")
    parser.add_argument("--results-dir", "-r", type=str, required=True,
                        help="Directory containing evaluation results")
    parser.add_argument("--neg-csv", type=str, default=None,
                        help="Path to obj_neg.csv for object->category mapping")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output JSON file (default: prints to console)")
    parser.add_argument("--csv", "-c", type=str, default=None,
                        help="Output CSV file with detailed results")
    parser.add_argument("--no-categories", action="store_true",
                        help="Don't show category breakdown")
    
    args = parser.parse_args()
    
    # Aggregate results
    neg_csv = Path(args.neg_csv) if args.neg_csv else None
    summary = aggregate_results(Path(args.results_dir), neg_csv)
    
    if not summary:
        print("No results found.")
        return 1
    
    # Print summary
    print_summary(summary, show_categories=not args.no_categories)
    
    # Save JSON
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved JSON summary to: {args.output}")
    
    # Save CSV
    if args.csv:
        df = pd.DataFrame(summary["detailed"])
        df.to_csv(args.csv, index=False)
        print(f"Saved detailed CSV to: {args.csv}")
    
    return 0


if __name__ == "__main__":
    exit(main())