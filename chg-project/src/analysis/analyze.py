"""
Analysis and visualization tools for CHG results.

Generates:
- Ablation curves
- Per-head heatmaps
- Distribution plots
- CSV exports
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from src.model.gate_wrapper import HeadGates
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_ablation_results(results_path: str):
    """Load ablation results from JSON or NPZ file."""
    if results_path.endswith('.json'):
        with open(results_path) as f:
            data = json.load(f)
        
        if 'rankings' in data:
            # Individual ablation results
            rankings = data['rankings']
            effects = np.array([r['effect'] for r in rankings])
            layers = np.array([r['layer'] for r in rankings])
            heads = np.array([r['head'] for r in rankings])
            return effects, layers, heads, data.get('summary', {})
        elif 'effects' in data:
            # Sequential ablation results
            return np.array(data['effects']), np.array(data['sequence']), data
    
    elif results_path.endswith('.npz'):
        data = np.load(results_path)
        return data['effects'], None, None, {}
    
    else:
        raise ValueError(f"Unsupported file format: {results_path}")


def plot_ablation_curve(effects: np.ndarray, output_path: str):
    """
    Plot ablation curve showing performance vs number of heads ablated.
    
    Args:
        effects: Array of performance metrics at each ablation step
        output_path: Where to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(effects))
    plt.plot(x, effects, linewidth=2, color='#2E86AB')
    
    # Mark baseline
    plt.axhline(y=effects[0], color='red', linestyle='--', 
                label=f'Baseline: {effects[0]:.3f}')
    
    plt.xlabel('Number of Heads Ablated', fontsize=12)
    plt.ylabel('Average Log-Probability', fontsize=12)
    plt.title('Sequential Head Ablation Curve', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved ablation curve to {output_path}")
    plt.close()


def plot_head_importance_heatmap(
    effects: np.ndarray,
    output_path: str,
    title: str = "Head Importance Heatmap"
):
    """
    Plot heatmap of per-head importance scores.
    
    Args:
        effects: (num_layers, num_heads) array of importance scores
        output_path: Where to save the plot
        title: Plot title
    """
    num_layers, num_heads = effects.shape
    
    fig, ax = plt.subplots(figsize=(max(12, num_heads), max(8, num_layers * 0.5)))
    
    # Create heatmap
    sns.heatmap(
        effects,
        cmap='RdYlGn_r',  # Red = important (negative), Green = unimportant (positive)
        center=0,
        annot=num_layers * num_heads <= 200,  # Annotate if not too many cells
        fmt='.3f',
        cbar_kws={'label': 'Log-Prob Change (negative = important)'},
        ax=ax
    )
    
    ax.set_xlabel('Head Index', fontsize=12)
    ax.set_ylabel('Layer Index', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved heatmap to {output_path}")
    plt.close()


def plot_gate_distribution(gates_path: str, output_path: str):
    """
    Plot distribution of gate values.
    
    Args:
        gates_path: Path to saved gates
        output_path: Where to save the plot
    """
    # Load gates
    checkpoint = torch.load(gates_path, map_location='cpu')
    logits = checkpoint['logits']
    gates = torch.sigmoid(logits).numpy().flatten()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(gates, bins=50, color='#A23B72', alpha=0.7, edgecolor='black')
    ax1.axvline(x=0.5, color='red', linestyle='--', label='Threshold = 0.5')
    ax1.set_xlabel('Gate Value', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Gate Values', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # CDF
    sorted_gates = np.sort(gates)
    cdf = np.arange(1, len(sorted_gates) + 1) / len(sorted_gates)
    ax2.plot(sorted_gates, cdf, linewidth=2, color='#2E86AB')
    ax2.axvline(x=0.5, color='red', linestyle='--', label='Threshold = 0.5')
    ax2.set_xlabel('Gate Value', fontsize=12)
    ax2.set_ylabel('Cumulative Probability', fontsize=12)
    ax2.set_title('Cumulative Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved gate distribution to {output_path}")
    plt.close()


def plot_layer_importance(effects: np.ndarray, output_path: str):
    """
    Plot average importance per layer.
    
    Args:
        effects: (num_layers, num_heads) array of importance scores
        output_path: Where to save the plot
    """
    layer_avg = effects.mean(axis=1)
    layer_std = effects.std(axis=1)
    
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(layer_avg))
    plt.bar(x, layer_avg, color='#F18F01', alpha=0.7, edgecolor='black')
    plt.errorbar(x, layer_avg, yerr=layer_std, fmt='none', 
                 ecolor='black', capsize=5, alpha=0.5)
    
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Average Importance (Log-Prob Change)', fontsize=12)
    plt.title('Average Head Importance by Layer', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved layer importance to {output_path}")
    plt.close()


def export_rankings_csv(
    effects: np.ndarray,
    output_path: str,
    top_k: int = None
):
    """
    Export head rankings to CSV.
    
    Args:
        effects: (num_layers, num_heads) array of importance scores
        output_path: Where to save the CSV
        top_k: If specified, only export top k heads
    """
    num_layers, num_heads = effects.shape
    
    # Create records
    records = []
    for l in range(num_layers):
        for h in range(num_heads):
            records.append({
                'layer': l,
                'head': h,
                'importance': effects[l, h],
                'abs_importance': abs(effects[l, h])
            })
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Sort by importance (most negative = most important)
    df = df.sort_values('importance')
    df['rank'] = np.arange(1, len(df) + 1)
    
    # Reorder columns
    df = df[['rank', 'layer', 'head', 'importance', 'abs_importance']]
    
    # Optionally filter
    if top_k:
        df = df.head(top_k)
    
    # Save
    df.to_csv(output_path, index=False)
    logger.info(f"Saved rankings to {output_path}")
    
    return df


def generate_summary_report(
    effects: np.ndarray,
    gates_path: str,
    output_path: str
):
    """
    Generate a text summary report.
    
    Args:
        effects: (num_layers, num_heads) array of importance scores
        gates_path: Path to gates file
        output_path: Where to save the report
    """
    num_layers, num_heads = effects.shape
    total_heads = num_layers * num_heads
    
    # Load gate statistics
    checkpoint = torch.load(gates_path, map_location='cpu')
    gates = torch.sigmoid(checkpoint['logits']).numpy()
    
    # Compute statistics
    importance_stats = {
        'mean': np.mean(effects),
        'std': np.std(effects),
        'min': np.min(effects),
        'max': np.max(effects),
        'median': np.median(effects)
    }
    
    gate_stats = {
        'mean': np.mean(gates),
        'std': np.std(gates),
        'min': np.min(gates),
        'max': np.max(gates),
        'median': np.median(gates)
    }
    
    # Count heads by threshold
    low_gates = (gates < 0.3).sum()
    mid_gates = ((gates >= 0.3) & (gates < 0.7)).sum()
    high_gates = (gates >= 0.7).sum()
    
    # Find most/least important
    flat_effects = effects.flatten()
    flat_idx = np.argsort(flat_effects)
    
    top_5_idx = flat_idx[:5]
    bottom_5_idx = flat_idx[-5:]
    
    # Write report
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("CHG ANALYSIS SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Model Configuration:\n")
        f.write(f"  Layers: {num_layers}\n")
        f.write(f"  Heads per layer: {num_heads}\n")
        f.write(f"  Total heads: {total_heads}\n\n")
        
        f.write("Importance Statistics:\n")
        for key, value in importance_stats.items():
            f.write(f"  {key.capitalize()}: {value:.4f}\n")
        f.write("\n")
        
        f.write("Gate Statistics:\n")
        for key, value in gate_stats.items():
            f.write(f"  {key.capitalize()}: {value:.4f}\n")
        f.write(f"  Heads with gates < 0.3: {low_gates} ({100*low_gates/total_heads:.1f}%)\n")
        f.write(f"  Heads with gates 0.3-0.7: {mid_gates} ({100*mid_gates/total_heads:.1f}%)\n")
        f.write(f"  Heads with gates > 0.7: {high_gates} ({100*high_gates/total_heads:.1f}%)\n\n")
        
        f.write("Top 5 Most Important Heads:\n")
        for i, idx in enumerate(top_5_idx):
            layer = idx // num_heads
            head = idx % num_heads
            f.write(f"  {i+1}. Layer {layer}, Head {head}: {flat_effects[idx]:.4f}\n")
        f.write("\n")
        
        f.write("Top 5 Least Important Heads:\n")
        for i, idx in enumerate(bottom_5_idx):
            layer = idx // num_heads
            head = idx % num_heads
            f.write(f"  {i+1}. Layer {layer}, Head {head}: {flat_effects[idx]:.4f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
    
    logger.info(f"Saved summary report to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize CHG results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--results", required=True,
                       help="Path to ablation results (JSON or NPZ)")
    parser.add_argument("--gates", required=True,
                       help="Path to fitted gates")
    parser.add_argument("--output_dir", default="analysis_output",
                       help="Output directory for plots and reports")
    parser.add_argument("--top_k", type=int, default=None,
                       help="Number of top heads to export to CSV")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    logger.info(f"Loading results from {args.results}")
    results_data = load_ablation_results(args.results)
    
    if len(results_data) == 4:
        # Individual ablation
        effects_flat, layers, heads, summary = results_data
        
        # Reconstruct 2D array
        num_layers = int(layers.max()) + 1
        num_heads = int(heads.max()) + 1
        effects = np.zeros((num_layers, num_heads))
        
        for i, (l, h) in enumerate(zip(layers, heads)):
            effects[int(l), int(h)] = effects_flat[i]
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        
        plot_head_importance_heatmap(
            effects,
            os.path.join(args.output_dir, "importance_heatmap.png")
        )
        
        plot_layer_importance(
            effects,
            os.path.join(args.output_dir, "layer_importance.png")
        )
        
        export_rankings_csv(
            effects,
            os.path.join(args.output_dir, "head_rankings.csv"),
            top_k=args.top_k
        )
        
        generate_summary_report(
            effects,
            args.gates,
            os.path.join(args.output_dir, "summary_report.txt")
        )
    
    else:
        # Sequential ablation
        effects, sequence, metadata = results_data
        
        plot_ablation_curve(
            effects,
            os.path.join(args.output_dir, "ablation_curve.png")
        )
    
    # Always plot gate distribution
    plot_gate_distribution(
        args.gates,
        os.path.join(args.output_dir, "gate_distribution.png")
    )
    
    logger.info(f"Analysis complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()