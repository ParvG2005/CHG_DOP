"""
Fit gating parameters by optimizing gates while keeping model weights frozen.

Three types of masks:
1. Initialization (λ=0): Baseline gates that preserve model behavior
2. Retention (λ>0): Gates that identify critical heads (G+)
3. Removal (λ<0): Gates that identify removable heads (G-)
"""
import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import os
import sys
from pathlib import Path

# Add src to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model.gate_wrapper import HeadGates, install_attention_hooks, remove_hooks
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_nll_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100):
    """
    Compute next-token negative log-likelihood loss.
    
    Args:
        logits: (batch, seq_len, vocab_size)
        labels: (batch, seq_len)
        ignore_index: Token ID to ignore (padding)
    
    Returns:
        Scalar loss value
    """
    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Compute cross-entropy
    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    return loss


def inverse_sigmoid_regularizer(gates: HeadGates, lambda_reg: float, epsilon: float = 1e-7):
    """
    Inverse sigmoid regularizer from the paper.
    
    For λ > 0: encourages gates toward 0 (removal)
    For λ < 0: encourages gates toward 1 (retention)
    
    R(g) = -log(g) for λ < 0 (retention)
    R(g) = -log(1-g) for λ > 0 (removal)
    """
    g = torch.sigmoid(gates.logits)
    g = g.clamp(epsilon, 1 - epsilon)
    
    if lambda_reg > 0:
        # Removal: penalize high gates
        reg = -torch.log(1 - g).sum()
    elif lambda_reg < 0:
        # Retention: penalize low gates
        reg = -torch.log(g).sum()
    else:
        # No regularization
        reg = torch.tensor(0.0, device=gates.logits.device)
    
    return abs(lambda_reg) * reg


def collate_fn(batch, tokenizer, max_length: int = 256):
    """Collate function for DataLoader."""
    # Handle different dataset formats
    if isinstance(batch[0], dict):
        if 'text' in batch[0]:
            texts = [b['text'] for b in batch]
        elif 'content' in batch[0]:
            texts = [b['content'] for b in batch]
        else:
            # Try to find any string field
            texts = [str(list(b.values())[0]) for b in batch]
    else:
        texts = [str(b) for b in batch]
    
    # Filter empty texts
    texts = [t for t in texts if t and len(t.strip()) > 0]
    
    if not texts:
        # Return dummy batch if all empty
        return tokenizer(["dummy text"], return_tensors='pt', padding=True, 
                        truncation=True, max_length=max_length)
    
    encoding = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_length
    )
    
    return encoding


def fit_gates(
    model_name: str,
    dataset_name: str,
    dataset_config: str,
    out_path: str,
    init_path: str = None,
    device: str = 'cuda',
    epochs: int = 3,
    lr: float = 1e-2,
    lambda_reg: float = 0.0,
    batch_size: int = 2,
    max_length: int = 256,
    grad_accum_steps: int = 1,
    dataset_split: str = 'train[:1%]',
    seed: int = 42
):
    """
    Fit gate parameters for a model on a dataset.
    
    Args:
        model_name: HuggingFace model identifier
        dataset_name: HuggingFace dataset identifier
        dataset_config: Dataset configuration/subset
        out_path: Path to save fitted gates
        init_path: Path to initialization gates (for G+/G- fitting)
        device: Device to use ('cuda' or 'cpu')
        epochs: Number of training epochs
        lr: Learning rate
        lambda_reg: Regularization strength (>0 for removal, <0 for retention, =0 for init)
        batch_size: Batch size
        max_length: Maximum sequence length
        grad_accum_steps: Gradient accumulation steps
        dataset_split: Dataset split specification
        seed: Random seed
    """
    # Set seed
    torch.manual_seed(seed)
    
    # Load tokenizer and model
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use fp32 for stability
        low_cpu_mem_usage=True
    ).to(device)
    model.eval()  # Freeze model weights
    
    # Get model dimensions
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    logger.info(f"Model has {num_layers} layers and {num_heads} heads per layer")
    
    # Initialize or load gates
    gates = HeadGates(num_layers, num_heads)
    
    if init_path and os.path.exists(init_path):
        logger.info(f"Loading initialization from {init_path}")
        gates.load(init_path, map_location=device)
    else:
        logger.info("Using default initialization")
    
    gates.to(device)
    
    # Load dataset
    logger.info(f"Loading dataset: {dataset_name}")
    try:
        if dataset_config:
            ds = load_dataset(dataset_name, dataset_config, split=dataset_split)
        else:
            ds = load_dataset(dataset_name, split=dataset_split)
    except Exception as e:
        logger.warning(f"Failed to load dataset {dataset_name}: {e}")
        logger.info("Falling back to wikitext")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split='train[:1%]')
    
    # Create dataloader
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_length)
    )
    
    # Install hooks
    handles = install_attention_hooks(model, gates)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW([gates.logits], lr=lr)
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(epochs):
        total_loss = 0.0
        total_nll = 0.0
        total_reg = 0.0
        num_batches = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass with gating
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            
            # Compute NLL loss
            loss_nll = compute_nll_loss(logits, input_ids)
            
            # Compute regularization
            loss_reg = inverse_sigmoid_regularizer(gates, lambda_reg)
            
            # Total loss
            loss = loss_nll + loss_reg
            
            # Backward pass
            loss = loss / grad_accum_steps
            loss.backward()
            
            # Update every grad_accum_steps
            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Track metrics
            total_loss += loss.item() * grad_accum_steps
            total_nll += loss_nll.item()
            total_reg += loss_reg.item() if isinstance(loss_reg, torch.Tensor) else loss_reg
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss/num_batches:.4f}",
                'nll': f"{total_nll/num_batches:.4f}",
                'reg': f"{total_reg/num_batches:.4f}"
            })
        
        # Epoch summary
        avg_loss = total_loss / num_batches
        avg_nll = total_nll / num_batches
        avg_reg = total_reg / num_batches
        
        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Loss: {avg_loss:.4f}, NLL: {avg_nll:.4f}, Reg: {avg_reg:.4f}"
        )
        
        # Print gate statistics
        with torch.no_grad():
            g = gates.get_gates()
            logger.info(
                f"Gate stats - Mean: {g.mean():.4f}, "
                f"Std: {g.std():.4f}, "
                f"Min: {g.min():.4f}, "
                f"Max: {g.max():.4f}"
            )
    
    # Cleanup
    remove_hooks(handles)
    
    # Save gates
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    gates.save(out_path)
    
    logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Fit CHG gate parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument("--model", default="gpt2", 
                       help="HuggingFace model identifier")
    
    # Dataset arguments
    parser.add_argument("--dataset", default="wikitext",
                       help="HuggingFace dataset identifier")
    parser.add_argument("--dataset_config", default="wikitext-2-raw-v1",
                       help="Dataset configuration/subset")
    parser.add_argument("--dataset_split", default="train[:1%]",
                       help="Dataset split specification")
    
    # Output arguments
    parser.add_argument("--out", default="gates.pt",
                       help="Output path for fitted gates")
    parser.add_argument("--init", default=None,
                       help="Path to initialization gates (for G+/G-)")
    
    # Training arguments
    parser.add_argument("--device", default="cuda",
                       help="Device to use")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-2,
                       help="Learning rate")
    parser.add_argument("--lambda_reg", type=float, default=0.0,
                       help="Regularization strength (>0 removal, <0 retention)")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size")
    parser.add_argument("--max_length", type=int, default=256,
                       help="Maximum sequence length")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    fit_gates(
        model_name=args.model,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        out_path=args.out,
        init_path=args.init,
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
        lambda_reg=args.lambda_reg,
        batch_size=args.batch_size,
        max_length=args.max_length,
        grad_accum_steps=args.grad_accum_steps,
        dataset_split=args.dataset_split,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
