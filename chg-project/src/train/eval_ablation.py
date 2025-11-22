"""
Evaluate head importance through systematic ablation experiments.

Computes per-head effects by measuring log-probability changes when
heads are ablated (set to zero).
"""
import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model.gate_wrapper import HeadGates, install_attention_hooks, remove_hooks
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_sequence_logprob(
    model,
    tokenizer,
    text: str,
    device: str = 'cuda'
) -> float:
    """
    Compute the log-probability of a text sequence.
    
    Returns average log-prob per token (higher is better).
    """
    encoding = tokenizer(text, return_tensors='pt').to(device)
    input_ids = encoding['input_ids']
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        # Negative loss is average log-prob
        return -outputs.loss.item()


def ablate_heads_sequential(
    model,
    tokenizer,
    gates: HeadGates,
    prompts: List[str],
    device: str = 'cuda',
    ablation_order: str = 'forward'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform sequential ablation: remove heads one at a time in order.
    
    Args:
        model: The transformer model
        tokenizer: Tokenizer
        gates: HeadGates module
        prompts: List of evaluation prompts
        device: Device
        ablation_order: 'forward', 'backward', or 'random'
    
    Returns:
        effects: (num_steps,) array of average log-probs after each ablation
        ablation_sequence: (num_steps, 2) array of (layer, head) indices
    """
    num_layers, num_heads = gates.logits.shape
    total_heads = num_layers * num_heads
    
    # Create ablation order
    head_indices = [(l, h) for l in range(num_layers) for h in range(num_heads)]
    
    if ablation_order == 'backward':
        head_indices = head_indices[::-1]
    elif ablation_order == 'random':
        import random
        random.shuffle(head_indices)
    
    # Store results
    effects = []
    ablation_sequence = []
    
    # Install hooks
    handles = install_attention_hooks(model, gates)
    
    # Compute baseline (no ablation)
    baseline_logprobs = [
        compute_sequence_logprob(model, tokenizer, p, device)
        for p in prompts
    ]
    baseline = np.mean(baseline_logprobs)
    effects.append(baseline)
    ablation_sequence.append((-1, -1))  # Sentinel for no ablation
    
    logger.info(f"Baseline log-prob: {baseline:.4f}")
    
    # Sequentially ablate heads
    saved_gates = gates.logits.data.clone()
    
    pbar = tqdm(head_indices, desc="Ablating heads")
    for layer_idx, head_idx in pbar:
        # Ablate this head
        gates.ablate_head(layer_idx, head_idx, value=0.0)
        
        # Evaluate
        logprobs = [
            compute_sequence_logprob(model, tokenizer, p, device)
            for p in prompts
        ]
        avg_logprob = np.mean(logprobs)
        
        effects.append(avg_logprob)
        ablation_sequence.append((layer_idx, head_idx))
        
        pbar.set_postfix({
            'layer': layer_idx,
            'head': head_idx,
            'logprob': f"{avg_logprob:.4f}",
            'delta': f"{avg_logprob - baseline:.4f}"
        })
    
    # Restore gates
    gates.logits.data = saved_gates
    
    # Cleanup
    remove_hooks(handles)
    
    return np.array(effects), np.array(ablation_sequence)


def compute_individual_head_effects(
    model,
    tokenizer,
    gates: HeadGates,
    prompts: List[str],
    device: str = 'cuda'
) -> np.ndarray:
    """
    Compute the effect of ablating each head individually.
    
    Returns:
        effects: (num_layers, num_heads) array where effects[l, h] is the
                 log-prob change when head (l, h) is ablated
    """
    num_layers, num_heads = gates.logits.shape
    
    # Install hooks
    handles = install_attention_hooks(model, gates)
    
    # Compute baseline
    baseline_logprobs = [
        compute_sequence_logprob(model, tokenizer, p, device)
        for p in prompts
    ]
    baseline = np.mean(baseline_logprobs)
    
    logger.info(f"Baseline log-prob: {baseline:.4f}")
    
    # Store effects
    effects = np.zeros((num_layers, num_heads))
    
    # Save original gates
    saved_gates = gates.logits.data.clone()
    
    # Test each head
    for layer_idx in tqdm(range(num_layers), desc="Evaluating layers"):
        for head_idx in range(num_heads):
            # Ablate this head only
            gates.logits.data = saved_gates.clone()
            gates.ablate_head(layer_idx, head_idx, value=0.0)
            
            # Evaluate
            logprobs = [
                compute_sequence_logprob(model, tokenizer, p, device)
                for p in prompts
            ]
            avg_logprob = np.mean(logprobs)
            
            # Store effect (negative means performance degraded)
            effects[layer_idx, head_idx] = avg_logprob - baseline
    
    # Restore gates
    gates.logits.data = saved_gates
    
    # Cleanup
    remove_hooks(handles)
    
    return effects


def rank_heads_by_importance(effects: np.ndarray) -> List[Tuple[int, int, float]]:
    """
    Rank heads by their importance (most negative effect = most important).
    
    Returns:
        List of (layer, head, effect) tuples sorted by importance
    """
    num_layers, num_heads = effects.shape
    
    rankings = []
    for l in range(num_layers):
        for h in range(num_heads):
            rankings.append((l, h, effects[l, h]))
    
    # Sort by effect (most negative first = most important)
    rankings.sort(key=lambda x: x[2])
    
    return rankings


def save_results(
    effects: np.ndarray,
    rankings: List[Tuple[int, int, float]],
    output_path: str
):
    """Save ablation results to disk."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Save as npz
    np.savez(
        output_path.replace('.json', '.npz'),
        effects=effects
    )
    
    # Save rankings as JSON
    results = {
        'rankings': [
            {'layer': int(l), 'head': int(h), 'effect': float(e)}
            for l, h, e in rankings
        ],
        'summary': {
            'mean_effect': float(np.mean(effects)),
            'std_effect': float(np.std(effects)),
            'min_effect': float(np.min(effects)),
            'max_effect': float(np.max(effects)),
            'num_layers': int(effects.shape[0]),
            'num_heads': int(effects.shape[1])
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate head importance through ablation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument("--model", default="gpt2",
                       help="HuggingFace model identifier")
    parser.add_argument("--gates", required=True,
                       help="Path to fitted gates")
    
    # Evaluation arguments
    parser.add_argument("--prompts", nargs='+',
                       default=["The capital of France is",
                               "When Alice met Bob, she gave him a"],
                       help="Evaluation prompts")
    parser.add_argument("--prompts_file", default=None,
                       help="File containing evaluation prompts (one per line)")
    
    # Ablation arguments
    parser.add_argument("--mode", choices=['individual', 'sequential'],
                       default='individual',
                       help="Ablation mode")
    parser.add_argument("--ablation_order", choices=['forward', 'backward', 'random'],
                       default='forward',
                       help="Order for sequential ablation")
    
    # Output arguments
    parser.add_argument("--output", default="ablation_results.json",
                       help="Output path for results")
    
    # Device argument
    parser.add_argument("--device", default="cuda",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Load prompts
    if args.prompts_file and os.path.exists(args.prompts_file):
        with open(args.prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
    else:
        prompts = args.prompts
        logger.info(f"Using {len(prompts)} command-line prompts")
    
    # Load model and tokenizer
    logger.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    ).to(args.device)
    model.eval()
    
    # Load gates
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    gates = HeadGates(num_layers, num_heads)
    gates.load(args.gates, map_location=args.device)
    gates.to(args.device)
    
    logger.info(f"Loaded gates from {args.gates}")
    
    # Run ablation
    if args.mode == 'individual':
        logger.info("Computing individual head effects...")
        effects = compute_individual_head_effects(
            model, tokenizer, gates, prompts, args.device
        )
        rankings = rank_heads_by_importance(effects)
        
        # Print top 10
        logger.info("\nTop 10 most important heads:")
        for i, (l, h, e) in enumerate(rankings[:10]):
            logger.info(f"{i+1}. Layer {l}, Head {h}: {e:.4f}")
        
        logger.info("\nTop 10 least important heads:")
        for i, (l, h, e) in enumerate(rankings[-10:]):
            logger.info(f"{i+1}. Layer {l}, Head {h}: {e:.4f}")
        
        save_results(effects, rankings, args.output)
        
    else:  # sequential
        logger.info(f"Running sequential ablation (order: {args.ablation_order})...")
        effects, sequence = ablate_heads_sequential(
            model, tokenizer, gates, prompts, args.device, args.ablation_order
        )
        
        # Save sequential results
        results = {
            'ablation_order': args.ablation_order,
            'effects': effects.tolist(),
            'sequence': sequence.tolist(),
            'baseline': float(effects[0]),
            'final': float(effects[-1]),
            'total_degradation': float(effects[-1] - effects[0])
        }
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Baseline: {effects[0]:.4f}")
        logger.info(f"Final: {effects[-1]:.4f}")
        logger.info(f"Total degradation: {effects[-1] - effects[0]:.4f}")


if __name__ == "__main__":
    main()
