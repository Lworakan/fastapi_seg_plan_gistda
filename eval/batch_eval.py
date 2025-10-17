"""
Batch evaluation script for UNet models
Evaluates multiple model checkpoints and compares performance.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from eval_unet import UNetEvaluator

def evaluate_multiple_models(model_dir, test_dir, save_dir, ground_truth_dir=None):
    """
    Evaluate multiple model checkpoints
    
    Args:
        model_dir: Directory containing .h5 model files
        test_dir: Directory with test images
        save_dir: Directory to save results
        ground_truth_dir: Optional ground truth directory
    
    Returns:
        Comparison results
    """
    model_dir = Path(model_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .h5 files
    model_files = list(model_dir.glob("*.h5"))
    
    if not model_files:
        print(f"❌ No .h5 files found in {model_dir}")
        return None
    
    print(f"📁 Found {len(model_files)} model files")
    
    all_results = []
    
    for model_path in model_files:
        print(f"\n{'='*70}")
        print(f"🔬 Evaluating: {model_path.name}")
        print("="*70)
        
        model_save_dir = save_dir / model_path.stem
        
        try:
            # Create evaluator
            evaluator = UNetEvaluator(str(model_path))
            
            # Evaluate
            summary, results, metrics = evaluator.evaluate_directory(
                test_dir=test_dir,
                save_dir=str(model_save_dir),
                ground_truth_dir=ground_truth_dir
            )
            
            model_result = {
                'model_name': model_path.name,
                'model_path': str(model_path),
                'summary': summary
            }
            
            all_results.append(model_result)
            
        except Exception as e:
            print(f"❌ Error evaluating {model_path.name}: {e}")
            continue
    
    # Create comparison
    if all_results:
        create_comparison_report(all_results, save_dir)
    
    return all_results

def create_comparison_report(results, save_dir):
    """Create comparison report and visualizations"""
    save_dir = Path(save_dir)
    
    # Extract metrics for comparison
    comparison_data = []
    for result in results:
        if 'metrics' in result['summary']:
            metrics = result['summary']['metrics']
            comparison_data.append({
                'Model': result['model_name'],
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'IoU': metrics['iou']
            })
    
    if not comparison_data:
        print("⚠️  No metrics available for comparison")
        return
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    csv_path = save_dir / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"📊 Comparison saved to: {csv_path}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'IoU']
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]
        ax.bar(range(len(df)), df[metric], color='steelblue', alpha=0.7)
        ax.set_xlabel('Model')
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels([m[:20] for m in df['Model']], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(df[metric]):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Hide last subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plot_path = save_dir / "model_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Comparison plot saved to: {plot_path}")
    
    # Print summary table
    print(f"\n{'='*70}")
    print("📊 MODEL COMPARISON SUMMARY")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluation of multiple UNet models"
    )
    
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing .h5 model files')
    parser.add_argument('--test_dir', type=str, default='test_images',
                       help='Directory containing test images')
    parser.add_argument('--save_dir', type=str, default='batch_results',
                       help='Directory to save results')
    parser.add_argument('--ground_truth_dir', type=str, default=None,
                       help='Directory with ground truth masks')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🔬 BATCH MODEL EVALUATION")
    print("="*70)
    
    results = evaluate_multiple_models(
        model_dir=args.model_dir,
        test_dir=args.test_dir,
        save_dir=args.save_dir,
        ground_truth_dir=args.ground_truth_dir
    )
    
    print("\n✅ Batch evaluation complete!")

if __name__ == "__main__":
    main()
