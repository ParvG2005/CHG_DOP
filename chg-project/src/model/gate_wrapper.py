"""
Contextual Head Gating (CHG) implementation.
Provides gating parameters and hook mechanisms for attention heads.
"""
import torch
import torch.nn as nn
from typing import List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HeadGates(nn.Module):
    """
    Stores gating parameters for attention heads across all layers.
    
    Gates are stored as logits and converted to [0,1] via sigmoid.
    Shape: (num_layers, num_heads)
    """
    def __init__(self, num_layers: int, num_heads: int, init_value: float = 2.0):
        super().__init__()
        # Initialize logits such that sigmoid(init_value) ≈ 0.88
        self.logits = nn.Parameter(
            torch.full((num_layers, num_heads), fill_value=init_value)
        )
        self.num_layers = num_layers
        self.num_heads = num_heads
        
    def forward(self) -> torch.Tensor:
        """Returns gates in [0,1] via sigmoid transformation."""
        return torch.sigmoid(self.logits)
    
    def get_gates(self) -> torch.Tensor:
        """Alias for forward() for clarity."""
        return self.forward()
    
    def set_gate(self, layer_idx: int, head_idx: int, value: float):
        """Set a specific gate to a target value (converts to logit)."""
        # Convert value in [0,1] to logit space
        logit = torch.logit(torch.tensor(value).clamp(1e-7, 1-1e-7))
        with torch.no_grad():
            self.logits[layer_idx, head_idx] = logit
    
    def ablate_head(self, layer_idx: int, head_idx: int, value: float = 0.0):
        """Ablate a head by setting its gate logit to produce ~value."""
        if value == 0.0:
            logit = -50.0  # sigmoid(-50) ≈ 0
        elif value == 1.0:
            logit = 50.0   # sigmoid(50) ≈ 1
        else:
            logit = torch.logit(torch.tensor(value).clamp(1e-7, 1-1e-7))
        
        with torch.no_grad():
            self.logits[layer_idx, head_idx] = logit
    
    def save(self, path: str):
        """Save gate parameters to disk."""
        torch.save({
            'logits': self.logits.data,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads
        }, path)
        logger.info(f"Saved gates to {path}")
    
    def load(self, path: str, map_location: str = 'cpu'):
        """Load gate parameters from disk."""
        checkpoint = torch.load(path, map_location=map_location)
        self.logits.data = checkpoint['logits']
        assert checkpoint['num_layers'] == self.num_layers
        assert checkpoint['num_heads'] == self.num_heads
        logger.info(f"Loaded gates from {path}")


def install_attention_hooks(
    model: nn.Module,
    gates: HeadGates,
    model_type: str = "auto"
) -> List:
    """
    Install forward hooks to multiply attention head outputs by gate values.
    
    Args:
        model: The transformer model (HuggingFace format)
        gates: HeadGates module containing gate parameters
        model_type: Type of model architecture ("llama", "gpt2", "opt", "auto")
    
    Returns:
        List of hook handles for cleanup
    """
    handles = []
    num_layers, num_heads = gates.logits.shape
    
    # Detect model architecture
    if model_type == "auto":
        model_type = detect_model_type(model)
    
    logger.info(f"Detected model type: {model_type}")
    
    # Get layers container
    layers = get_model_layers(model, model_type)
    
    if len(layers) != num_layers:
        logger.warning(
            f"Mismatch: gates have {num_layers} layers but model has {len(layers)}"
        )
    
    for layer_idx, layer in enumerate(layers):
        if layer_idx >= num_layers:
            break
            
        # Get attention module
        attn_module = get_attention_module(layer, model_type)
        
        if attn_module is None:
            raise RuntimeError(
                f"Could not find attention module in layer {layer_idx}. "
                f"Model type: {model_type}"
            )
        
        # Create and register hook
        hook = create_gating_hook(layer_idx, gates, num_heads)
        handle = attn_module.register_forward_hook(hook)
        handles.append(handle)
    
    logger.info(f"Installed {len(handles)} attention hooks")
    return handles


def detect_model_type(model: nn.Module) -> str:
    """Detect the model architecture type."""
    model_class = model.__class__.__name__.lower()
    
    if "llama" in model_class:
        return "llama"
    elif "gpt2" in model_class or "gpt-2" in model_class:
        return "gpt2"
    elif "opt" in model_class:
        return "opt"
    elif "mistral" in model_class:
        return "llama"  # Mistral uses similar architecture
    else:
        logger.warning(f"Unknown model type: {model_class}, assuming llama-style")
        return "llama"


def get_model_layers(model: nn.Module, model_type: str):
    """Get the layers container from the model."""
    # Try common paths
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h  # GPT-2 style
    elif hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers  # OPT style
    else:
        raise RuntimeError(
            "Could not find transformer layers. "
            "Please check model structure and update get_model_layers()"
        )


def get_attention_module(layer: nn.Module, model_type: str):
    """Extract the attention module from a layer."""
    if hasattr(layer, "self_attn"):
        return layer.self_attn  # Llama, Mistral, OPT
    elif hasattr(layer, "attn"):
        return layer.attn  # GPT-2
    elif hasattr(layer, "attention"):
        return layer.attention
    else:
        # Search submodules
        for name, module in layer.named_modules():
            if "attn" in name.lower() and name.count('.') == 0:
                return module
        return None


def create_gating_hook(layer_idx: int, gates: HeadGates, num_heads: int):
    """
    Create a forward hook that applies gating to attention outputs.
    
    The hook multiplies each head's output by its corresponding gate value.
    """
    def hook(module, input_tuple, output):
        """
        Hook function applied to attention module output.
        
        Output formats vary by model:
        - Tuple: (attn_output, attn_weights, ...) 
        - Tensor: just attn_output
        
        attn_output shape: (batch, seq_len, hidden_dim)
        We reshape to (batch, seq_len, num_heads, head_dim), apply gates, reshape back.
        """
        # Handle tuple outputs (most common)
        if isinstance(output, tuple):
            attn_out = output[0]
            other_outputs = output[1:]
        else:
            attn_out = output
            other_outputs = None
        
        # Get dimensions
        batch_size, seq_len, hidden_dim = attn_out.shape
        
        # Check if we can split into heads
        if hidden_dim % num_heads != 0:
            logger.warning(
                f"Cannot split hidden_dim {hidden_dim} into {num_heads} heads. "
                "Skipping gating for this layer."
            )
            return output
        
        head_dim = hidden_dim // num_heads
        
        # Reshape: (B, S, H*D) -> (B, S, H, D)
        x = attn_out.view(batch_size, seq_len, num_heads, head_dim)
        
        # Get gates for this layer: (num_heads,) -> (1, 1, num_heads, 1)
        g = torch.sigmoid(gates.logits[layer_idx]).view(1, 1, num_heads, 1)
        g = g.to(x.device)
        
        # Apply gates
        x = x * g
        
        # Reshape back: (B, S, H, D) -> (B, S, H*D)
        gated_out = x.view(batch_size, seq_len, hidden_dim)
        
        # Return in same format as input
        if other_outputs is not None:
            return (gated_out,) + other_outputs
        else:
            return gated_out
    
    return hook


def remove_hooks(handles: List):
    """Remove all hooks from their handles."""
    for handle in handles:
        handle.remove()
    logger.info(f"Removed {len(handles)} hooks")