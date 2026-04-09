import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
import requests
from huggingface_hub.utils import build_hf_headers
import gc

# Critical: Enable memory fragmentation fix
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def setup_directories():
    dirs = [
        "goodfire_70b_out/goodfire_70b_heatmaps",
        "goodfire_70b_out/goodfire_70b_sae_plots",
        "goodfire_70b_out/goodfire_70b_sae_json",
        "goodfire_70b_out/goodfire_70b_layer_dynamics"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID  = "meta-llama/Llama-3.3-70B-Instruct"
SAE_REPO  = "Goodfire/Llama-3.3-70B-Instruct-SAE-l50"
SAE_FILE  = "Llama-3.3-70B-Instruct-SAE-l50.pt"   # exact filename in repo
SAE_LAYER = 50                                      # SAE is trained on layer 50 only
HF_CACHE  = "/scratch/hrishikesh/shared_models/huggingface/hub"

PROMPT = "what is explainability in AI , answer in 1-2 lines"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

setup_directories()

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load in bfloat16 (half precision) - still full model, no quantization
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",  # Use SDPA (faster, less memory)
    low_cpu_mem_usage=True,
)
model.eval()

# Enable gradient checkpointing - trades compute for memory without affecting results
model.gradient_checkpointing_enable()

print(f"Model loaded in bfloat16 (full precision, no quantization)")
print(f"Gradient checkpointing enabled to reduce memory during backward pass")

# ── Locate transformer layers ─────────────────────────────────────────────────
LAYER_PATH = None
if hasattr(model.model, 'layers'):
    LAYER_PATH = lambda m: m.model.layers
elif hasattr(model.model, 'text_model') and hasattr(model.model.text_model, 'layers'):
    LAYER_PATH = lambda m: m.model.text_model.layers
elif hasattr(model.model, 'language_model'):
    lang_model = model.model.language_model
    if hasattr(lang_model, 'model') and hasattr(lang_model.model, 'layers'):
        LAYER_PATH = lambda m: m.model.language_model.model.layers
    elif hasattr(lang_model, 'layers'):
        LAYER_PATH = lambda m: m.model.language_model.layers

if LAYER_PATH is None:
    raise RuntimeError("Could not find layers in model structure")

num_layers = len(LAYER_PATH(model))
LAYERS_TO_ANALYZE = list(range(num_layers))
print(f"Detected {num_layers} layers. SAE analysis will run only for layer {SAE_LAYER}.")

# ── Tokenize & generate full response ─────────────────────────────────────────
print(f"\nPrompt: '{PROMPT}'")
inputs = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)

print("Generating full response ...")
with torch.no_grad():
    full_output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )
full_response = tokenizer.decode(
    full_output[0][inputs.input_ids.shape[1]:],
    skip_special_tokens=True
).strip()
print(f"\n--- MODEL FULL RESPONSE ---\n{full_response}\n---------------------------\n")

# ── Forward pass with grad tracking ──────────────────────────────────────────
# Disable attention outputs to save memory (SDPA doesn't support them anyway)
print("Running forward pass...")
outputs = model(
    **inputs,
    output_hidden_states=True,
    output_attentions=False  # Critical: saves ~15-20GB on 70B model
)

# CRITICAL: Move ALL hidden states to CPU immediately to free GPU memory
print("Moving hidden states to CPU to free GPU memory...")
hidden_states_cpu = [h.detach().cpu() for h in outputs.hidden_states]
hidden_states = hidden_states_cpu
attentions = None  # Not needed

# Get logits for prediction
logits = outputs.logits
final_token_logits = logits[0, -1, :]
predicted_token_id = torch.argmax(final_token_logits).item()
predicted_word = tokenizer.decode(predicted_token_id)
print(f"Model predicted first token: '{predicted_word}'")

# Clear the original outputs to free memory
del outputs, logits, final_token_logits
torch.cuda.empty_cache()
gc.collect()

print(f"Memory optimization: All hidden states moved to CPU, no backward pass")
print(f"Note: SAE analysis will use activation magnitudes only (no gradients)")

# ── Skip backward pass to avoid OOM ──────────────────────────────────────────
print("\nSkipping backward pass to avoid OOM on 70B model")
print("SAE analysis will use activation-based feature importance instead")
sae_layer_grad = None  # No gradients available

# ── Pass 1: collect logit-lens data for every layer (no generate calls here) ──
print("\nCollecting logit-lens data for all layers ...")
all_layer_data   = {}
layer_entropy    = []
layer_similarity = []

for layer_idx in LAYERS_TO_ANALYZE:
    target_block = LAYER_PATH(model)[layer_idx]
    # Hidden states are on CPU, move to GPU temporarily for processing
    h_raw = hidden_states[layer_idx][0, -1, :].to(DEVICE).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        normed_h = target_block.input_layernorm(h_raw).squeeze()

        # Cosine similarity with previous layer
        if layer_idx > 0:
            cos_sim = F.cosine_similarity(
                hidden_states[layer_idx - 1][0, -1, :].to(DEVICE).unsqueeze(0),
                hidden_states[layer_idx    ][0, -1, :].to(DEVICE).unsqueeze(0)
            ).item()
        else:
            cos_sim = 1.0
        layer_similarity.append(cos_sim)

        # Logit lens
        lens_logits = model.lm_head(normed_h)
        probs       = F.softmax(lens_logits, dim=-1)
        entropy     = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        layer_entropy.append(entropy)

        # Single topk call (not two)
        top_k_result = torch.topk(probs, 10)
        logit_lens_results = [
            {"word": tokenizer.decode(idx.item()).strip(), "prob": float(p.item())}
            for idx, p in zip(top_k_result.indices, top_k_result.values)
        ]

    all_layer_data[layer_idx] = {
        "normed_h":           normed_h,
        "logit_lens_results": logit_lens_results,
        "entropy":            entropy,
        "cos_sim":            cos_sim,
    }
    print(f"  Layer {layer_idx:02d} | H: {entropy:.4f} | CosSim: {cos_sim:.4f} "
          f"| Top: {logit_lens_results[0]['word']}")

# ── Pass 2: AI self-explanation — sampled every 10 layers only ────────────────
print("\nRunning AI self-explanation (every 10th layer) ...")
explanation_layers = list(range(0, num_layers, 10))
layer_explanations = {}

for layer_idx in explanation_layers:
    top_5_words    = [r['word'] for r in all_layer_data[layer_idx]['logit_lens_results'][:5]]
    explainer_text = (
        f"The following 5 words represent a concept: {', '.join(top_5_words)}. "
        f"Identify the common concept concisely: "
    )
    exp_in = tokenizer(explainer_text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        exp_out = model.generate(
            **exp_in,
            max_new_tokens=15,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    explanation = tokenizer.decode(
        exp_out[0][exp_in.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()
    layer_explanations[layer_idx] = explanation
    print(f"  Layer {layer_idx:02d}: {explanation}")

# ── Load SAE weights (layer 50 only, from shared HF cache) ───────────────────
print(f"\nLoading SAE for layer {SAE_LAYER} from cache ...")
sae_W_enc = None
sae_b_enc = None

try:
    pt_path = hf_hub_download(
        repo_id=SAE_REPO,
        filename=SAE_FILE,
        cache_dir=HF_CACHE
    )
    print(f"SAE file resolved at: {pt_path}")

    ckpt = torch.load(pt_path, map_location=DEVICE, weights_only=False)

    # Confirmed key names from checkpoint inspection:
    #   encoder_linear.weight : torch.Size([65536, 8192])
    #   encoder_linear.bias   : torch.Size([65536])
    #   decoder_linear.weight : torch.Size([8192, 65536])
    #   decoder_linear.bias   : torch.Size([8192])
    sae_W_enc = ckpt['encoder_linear.weight'].to(DEVICE).to(torch.bfloat16)
    sae_b_enc = ckpt['encoder_linear.bias'  ].to(DEVICE).to(torch.bfloat16)
    print(f"SAE encoder loaded — W: {sae_W_enc.shape}, b: {sae_b_enc.shape}")

except Exception as e:
    print(f"Failed to load SAE: {e}")
    print("SAE plots will be skipped.")

# ── Per-layer export: JSON + heatmaps + SAE plot ──────────────────────────────
print("\nExporting per-layer analysis ...")

for layer_idx in LAYERS_TO_ANALYZE:
    data     = all_layer_data[layer_idx]
    normed_h = data["normed_h"]

    # Find nearest sampled explanation at or below this layer
    nearest_exp_layer = max(
        (k for k in layer_explanations if k <= layer_idx),
        default=0
    )

    export_data = {
        "layer":                      layer_idx,
        "prompt":                     PROMPT,
        "predicted_target":           predicted_word,
        "residual_cosine_similarity": data["cos_sim"],
        "entropy":                    data["entropy"],
        "logit_lens":                 data["logit_lens_results"],
        "ai_explanation":             layer_explanations.get(nearest_exp_layer, ""),
        "sae_top_features":           []
    }

    # Skip attention heatmaps to save memory (attentions disabled with SDPA)
    # For 70B models, attention analysis is too memory-intensive

    # SAE activation analysis — layer 50 only (no gradients, just activations)
    if layer_idx == SAE_LAYER and sae_W_enc is not None:
        with torch.no_grad():
            # Ensure SAE weights are on same device as normed_h
            W_enc_device = sae_W_enc.to(normed_h.device)
            b_enc_device = sae_b_enc.to(normed_h.device)
            
            # Compute SAE activations (feature magnitudes)
            acts_sae = torch.matmul(normed_h, W_enc_device.t()) + b_enc_device
            
            # Use activation magnitude as importance (no gradients available)
            # This shows which features are most active for this token
            top_val, top_idx = torch.topk(acts_sae.abs(), 10)
            sae_results = [
                {"id": i.item(), "activation": float(acts_sae[i].item())}
                for i in top_idx
            ]

        labels = [f"Feature-{x['id']}" for x in sae_results]
        values = [x['activation'] for x in sae_results]
        colors = ['coral' if v >= 0 else 'steelblue' for v in values]

        plt.figure(figsize=(12, 5))
        plt.bar(range(10), values, color=colors)
        plt.xticks(range(10), labels, rotation=45, ha='right')
        plt.axhline(0, color='black', linewidth=0.8)
        plt.title(f"Top 10 Goodfire SAE Feature Activations — Layer {SAE_LAYER}")
        plt.ylabel("Feature Activation Magnitude")
        plt.tight_layout()
        plt.savefig(
            f"goodfire_70b_out/goodfire_70b_sae_plots/layer_{layer_idx:02d}_sae.png"
        )
        plt.close()
        export_data["sae_top_features"] = sae_results
        print(f"  Layer {layer_idx:02d}: SAE activation plot saved ({len(sae_results)} features).")

    with open(
        f"goodfire_70b_out/goodfire_70b_sae_json/layer_{layer_idx:02d}_analysis.json", "w"
    ) as f:
        json.dump(export_data, f, indent=2)

# ── Overall dynamics plot ─────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(LAYERS_TO_ANALYZE, layer_entropy, marker='o', color='purple', markersize=3)
ax1.axvline(SAE_LAYER, color='red', linestyle='--', linewidth=1, label=f'SAE layer ({SAE_LAYER})')
ax1.set_xlabel("Layer")
ax1.set_ylabel("Entropy")
ax1.set_title("Logit Lens Entropy per Layer")
ax1.legend()

ax2.plot(LAYERS_TO_ANALYZE, layer_similarity, marker='s', color='teal', markersize=3)
ax2.axvline(SAE_LAYER, color='red', linestyle='--', linewidth=1, label=f'SAE layer ({SAE_LAYER})')
ax2.set_xlabel("Layer")
ax2.set_ylabel("Cosine Similarity")
ax2.set_title("Consecutive Layer Cosine Similarity")
ax2.legend()

plt.tight_layout()
plt.savefig("goodfire_70b_out/goodfire_70b_layer_dynamics/overall_dynamics.png")
plt.close()

print("\nAnalysis complete. All outputs written to goodfire_70b_out/")
