"""
Contrastive CHG: Learn gates that retain one dataset variant while forgetting another.

This is useful for identifying heads responsible for specific behaviors or biases.
"""
import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm
import os
import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model.gate_wrapper import HeadGates, install_attention_hooks, remove_hooks
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContrastiveDataset(Dataset):
    """Dataset that pairs retain and forget examples."""
    
    def __init__(self, retain_data: List[str], forget_data: List[str]):
        """
        Args:
            retain_data: Examples to retain
            forget_data: Examples to forget
        """
        # Ensure same length or truncate
        min_len = min(len(retain_data), len(forget_data))
        self.retain_data = retain_data[:min_len]
        self.forget_data = forget_data[:min_len]
    
    def __len__(self):
        return len(self.retain_data)
    
    def __getitem__(self, idx):
        return {
            'retain': self.retain_data[idx],
            'forget': self.forget_data[idx]
        }


def compute_nll_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100):
    """Compute next-token NLL loss."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    return loss


def contrastive_loss(
    model,
    tokenizer,
    gates: HeadGates,
    retain_texts: List[str],
    forget_texts: List[str],
    device: str = 'cuda',
    alpha: float = 1.0
) -> Dict[str, float]:
    """
    Compute contrastive loss: minimize retain NLL, maximize forget NLL.
    
    Loss = L_retain - α * L_forget
    
    Args:
        model: The transformer model
        tokenizer: Tokenizer
        gates: HeadGates module
        retain_texts: Texts to retain
        forget_texts: Texts to forget
        device: Device
        alpha: Weight for forget loss
    
    Returns:
        Dict with loss components
    """
    # Encode retain batch
    retain_enc = tokenizer(
        retain_texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=256
    ).to(device)
    
    # Forward pass on retain data
    retain_outputs = model(
        input_ids=retain_enc['input_ids'],
        attention_mask=retain_enc['attention_mask']
    )
    loss_retain = compute_nll_loss(retain_outputs.logits, retain_enc['input_ids'])
    
    # Encode forget batch
    forget_enc = tokenizer(
        forget_texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=256
    ).to(device)
    
    # Forward pass on forget data
    forget_outputs = model(
        input_ids=forget_enc['input_ids'],
        attention_mask=forget_enc['attention_mask']
    )
    loss_forget = compute_nll_loss(forget_outputs.logits, forget_enc['input_ids'])
    
    # Contrastive objective: minimize retain loss, maximize forget loss
    # Maximizing forget loss = minimizing negative forget loss
    loss = loss_retain - alpha * loss_forget
    
    return {
        'total': loss,
        'retain': loss_retain.item(),
        'forget': loss_forget.item()
    }


def fit_contrastive_gates(
    model_name: str,
    retain_dataset: str,
    forget_dataset: str,
    out_path: str,
    device: str = 'cuda',
    epochs: int = 3,
    lr: float = 1e-2,
    alpha: float = 1.0,
    batch_size: int = 2,
    dataset_split: str = 'train[:1%]',
    seed: int = 42
):
    """
    Fit gates using contrastive objective.
    
    Args:
        model_name: HuggingFace model identifier
        retain_dataset: Dataset to retain (HF identifier or 'wikitext')
        forget_dataset: Dataset to forget (HF identifier or 'wikitext')
        out_path: Output path for fitted gates
        device: Device
        epochs: Training epochs
        lr: Learning rate
        alpha: Weight for forget loss
        batch_size: Batch size
        dataset_split: Dataset split specification
        seed: Random seed
    """
    torch.manual_seed(seed)
    
    # Load model
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    
    # Initialize gates
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    gates = HeadGates(num_layers, num_heads)
    gates.to(device)
    
    # Load datasets
    logger.info(f"Loading retain dataset: {retain_dataset}")
    if retain_dataset == 'wikitext':
        retain_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=dataset_split)
    else:
        retain_ds = load_dataset(retain_dataset, split=dataset_split)
    
    logger.info(f"Loading forget dataset: {forget_dataset}")
    if forget_dataset == 'wikitext':
        forget_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=dataset_split)
    else:
        forget_ds = load_dataset(forget_dataset, split=dataset_split)
    
    # Extract texts
    def extract_text(example):
        if 'text' in example:
            return example['text']
        elif 'content' in example:
            return example['content']
        else:
            return str(list(example.values())[0])
    
    retain_texts = [extract_text(ex) for ex in retain_ds]
    forget_texts = [extract_text(ex) for ex in forget_ds]
    
    # Filter empty
    retain_texts = [t for t in retain_texts if t and len(t.strip()) > 0]
    forget_texts = [t for t in forget_texts if t and len(t.strip()) > 0]
    
    logger.info(f"Retain: {len(retain_texts)} examples")
    logger.info(f"Forget: {len(forget_texts)} examples")
    
    # Create contrastive dataset
    dataset = ContrastiveDataset(retain_texts, forget_texts)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Install hooks
    handles = install_attention_hooks(model, gates)
    
    # Optimizer
    optimizer = torch.optim.AdamW([gates.logits], lr=lr)
    
    # Training loop
    logger.info("Starting contrastive training...")
    for epoch in range(epochs):
        total_loss = 0.0
        total_retain = 0.0
        total_forget = 0.0
        num_batches = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            retain_batch = batch['retain']
            forget_batch = batch['forget']
            
            # Compute contrastive loss
            losses = contrastive_loss(
                model, tokenizer, gates,
                retain_batch, forget_batch,
                device, alpha
            )
            
            # Backward
            optimizer.zero_grad()
            losses['total'].backward()
            optimizer.step()
            
            # Track
            total_loss += losses['total'].item()
            total_retain += losses['retain']
            total_forget += losses['forget']
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{total_loss/num_batches:.4f}",
                'retain': f"{total_retain/num_batches:.4f}",
                'forget': f"{total_forget/num_batches:.4f}"
            })
        
        # Epoch summary
        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Loss: {total_loss/num_batches:.4f}, "
            f"Retain NLL: {total_retain/num_batches:.4f}, "
            f"Forget NLL: {total_forget/num_batches:.4f}"
        )
        
        # Gate statistics
        with torch.no_grad():
            g = gates.get_gates()
            logger.info(
                f"Gate stats - Mean: {g.mean():.4f}, Std: {g.std():.4f}, "
                f"Min: {g.min():.4f}, Max: {g.max():.4f}"
            )
    
    # Cleanup
    remove_hooks(handles)
    
    # Save
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    gates.save(out_path)
    
    logger.info("Contrastive training complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Fit gates with contrastive objective",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--model", default="gpt2",
                       help="HuggingFace model identifier")
    parser.add_argument("--retain_dataset", required=True,
                       help="Dataset to retain")
    parser.add_argument("--forget_dataset", required=True,
                       help="Dataset to forget")
    parser.add_argument("--out", default="gates_contrastive.pt",
                       help="Output path")
    parser.add_argument("--device", default="cuda",
                       help="Device")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-2,
                       help="Learning rate")
    parser.add_argument("--alpha", type=float, default=1.0,
                       help="Weight for forget loss")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size")
    parser.add_argument("--dataset_split", default="train[:1%]",
                       help="Dataset split")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    fit_contrastive_gates(
        model_name=args.model,
        retain_dataset=args.retain_dataset,
        forget_dataset=args.forget_dataset,
        out_path=args.out,
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
        alpha=args.alpha,
        batch_size=args.batch_size,
        dataset_split=args.dataset_split,
        seed=args.seed
    )


if __name__ == "__main__":
    main()