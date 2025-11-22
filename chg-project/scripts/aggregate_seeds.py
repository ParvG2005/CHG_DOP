"""Aggregate results from multiple seed runs."""
import json
import sys
import numpy as np
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python aggregate_seeds.py result1.json result2.json ...")
        sys.exit(1)
    
    all_results = []
    for path in sys.argv[1:]:
        with open(path) as f:
            all_results.append(json.load(f))
    
    # Extract effects
    all_effects = []
    for result in all_results:
        if 'rankings' in result:
            effects = [r['effect'] for r in result['rankings']]
            all_effects.append(effects)
    
    if not all_effects:
        print("No valid results found")
        return
    
    # Compute statistics
    all_effects = np.array(all_effects)
    mean_effects = all_effects.mean(axis=0)
    std_effects = all_effects.std(axis=0)
    
    print(f"Aggregated over {len(all_results)} seeds:")
    print(f"  Mean importance: {mean_effects.mean():.4f} ± {mean_effects.std():.4f}")
    print(f"  Min importance: {mean_effects.min():.4f}")
    print(f"  Max importance: {mean_effects.max():.4f}")
    print(f"  Average std across heads: {std_effects.mean():.4f}")

if __name__ == "__main__":
    main()