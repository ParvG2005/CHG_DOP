import os
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

# ── Memory optimization ───────────────────────────────────────────────────────
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# ==========================================
# 1. Setup & Initialization
# ==========================================
def setup_directories():
    dirs = [
        "llama_out/llama_heatmaps",
        "llama_out/llama_sae_plots",
        "llama_out/llama_sae_json",
        "llama_out/llama_layer_dynamics"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

# FIX 1: Correct model ID — Meta-Llama-3.1-8B (not Meta-Llama-3.1-8B-Instruct,
# must match what the SAE was trained on)
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"

# FIX 2: Correct SAE repo — EleutherAI/sae-llama-3.1-8b does not exist as a flat
# repo. The real repos are suffixed with expansion factor.
# 32x = standard quality, covers all 32 layers via hookpoint="layers.N"
SAE_REPO = "EleutherAI/sae-llama-3.1-8b-32x"

PROMPT = "what is explainability in AI , answer in 1-2 lines"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

setup_directories()

print(f"Loading {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# FIX 3: torch_dtype → dtype (deprecation fix)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    dtype=torch.bfloat16,
    attn_implementation="eager",
)
model.eval()

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
print(f"Detected {num_layers} layers. Will load SAE per-layer from {SAE_REPO}.")

# ==========================================
# 2. Load all SAEs upfront using EleutherAI's Sae library
# ==========================================
# FIX 4: EleutherAI SAEs must be loaded via their Sae library, not raw safetensors.
# The Sae object exposes .encoder.weight and .encoder.bias directly.
# pip install sparsify  (EleutherAI's SAE library, formerly called 'sae')
print(f"\nLoading all SAEs from {SAE_REPO} ...")
saes = {}
try:
    from sparsify import Sae
    saes = Sae.load_many(SAE_REPO)
    # saes is a dict keyed by hookpoint e.g. {"layers.0.mlp": Sae, "layers.1.mlp": Sae, ...}
    print(f"Loaded {len(saes)} SAEs from {SAE_REPO}")
    print(f"Available hookpoints: {sorted(saes.keys())}")
except ImportError:
    print("WARNING: 'sparsify' not installed. Run: pip install sparsify")
    print("SAE plots will be skipped. Install sparsify and re-run for full analysis.")
except Exception as e:
    print(f"WARNING: Could not load SAEs: {e}")
    print("SAE plots will be skipped.")

# ==========================================
# 3. Full generation for context
# ==========================================
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

# ==========================================
# 4. Forward pass + backward for gradients
# ==========================================
outputs = model(
    **inputs,
    output_hidden_states=True,
    output_attentions=True
)

hidden_states = outputs.hidden_states
attentions    = outputs.attentions

# FIX 5: retain_grad must be called before backward, which it is here.
# Also retain ALL hidden state grads since this is 8B (fits in VRAM)
# and we want SAE attribution for every layer.
for h in hidden_states:
    if h.requires_grad:
        h.retain_grad()

if attentions:
    for a in attentions:
        a.requires_grad_(True)
        a.retain_grad()

logits             = outputs.logits
final_token_logits = logits[0, -1, :]
predicted_token_id = torch.argmax(final_token_logits).item()
predicted_word     = tokenizer.decode(predicted_token_id)
print(f"Model predicted first token: '{predicted_word}'")

# FIX 6: retain_graph=False — only calling backward once, no need to keep graph
torch.cuda.empty_cache()
model.zero_grad()
final_token_logits[predicted_token_id].backward(retain_graph=False)

# ==========================================
# 5. Pass 1 — collect logit-lens for all layers (no generate calls)
# ==========================================
# FIX 7: model.generate() moved OUT of the per-layer loop.
# Original called it 32 times (once per layer). Now called ~4 times (every 8th layer).
print("\nCollecting logit-lens data for all layers ...")
all_layer_data   = {}
layer_entropy    = []
layer_similarity = []

for layer_idx in LAYERS_TO_ANALYZE:
    target_block = LAYER_PATH(model)[layer_idx]
    h_raw = hidden_states[layer_idx][0, -1, :].unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        normed_h = target_block.input_layernorm(h_raw).squeeze()

        if layer_idx > 0:
            cos_sim = F.cosine_similarity(
                hidden_states[layer_idx - 1][0, -1, :].unsqueeze(0),
                hidden_states[layer_idx    ][0, -1, :].unsqueeze(0)
            ).item()
        else:
            cos_sim = 1.0
        layer_similarity.append(cos_sim)

        lens_logits = model.lm_head(normed_h)
        probs       = F.softmax(lens_logits, dim=-1)
        entropy     = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        layer_entropy.append(entropy)

        # FIX 8: single topk call (original called it twice)
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

# ==========================================
# 6. Pass 2 — AI self-explanation (every 8th layer only)
# ==========================================
print("\nRunning AI self-explanation (every 8th layer) ...")
explanation_layers = list(range(0, num_layers, 8))
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

# ==========================================
# 7. Per-layer export: JSON + heatmaps + SAE plots
# ==========================================
print("\nExporting per-layer analysis ...")

for layer_idx in LAYERS_TO_ANALYZE:
    data     = all_layer_data[layer_idx]
    normed_h = data["normed_h"]

    nearest_exp_layer = max(
        (k for k in layer_explanations if k <= layer_idx),
        default=0
    )

    # FIX 9: initialise sae_results before try/except so JSON export never crashes
    # on NameError if the try block fails partway through
    sae_results = []

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

    # ── Attention heatmap ─────────────────────────────────────────────────────
    if attentions and layer_idx < len(attentions):
        attn = attentions[layer_idx]
        if attn.grad is not None:
            relevance = (attn * attn.grad)[0].sum(dim=0).cpu().detach().float().numpy()
            plt.figure(figsize=(10, 8))
            sns.heatmap(relevance, cmap="viridis", annot=False)
            plt.title(f"Attention Attribution Map — Layer {layer_idx} — Llama 3.1 8B")
            plt.xlabel("Key Tokens (Context)")
            plt.ylabel("Query Tokens (Current state)")
            plt.tight_layout()
            plt.savefig(f"llama_out/llama_heatmaps/layer_{layer_idx:02d}_attn.png")
            plt.close()
            export_data["attention_attribution_saved"] = True
        else:
            print(f"  Layer {layer_idx:02d}: attention grad is None — heatmap skipped.")

    # ── SAE feature attribution ───────────────────────────────────────────────
    # FIX 10: use Sae library object directly instead of manual safetensors loading.
    # Hookpoint format is "layers.N.mlp" for EleutherAI SAEs (trained on MLP output)
    hookpoint = f"layers.{layer_idx}.mlp"
    sae = saes.get(hookpoint, None)

    if sae is not None:
        h_grad = hidden_states[layer_idx].grad
        if h_grad is not None:
            h_grad_vec = h_grad[0, -1, :]
            try:
                # FIX 11: EleutherAI Sae object exposes .encoder.weight and .encoder.bias
                W_enc = sae.encoder.weight.to(DEVICE).to(torch.bfloat16)  # [d_sae, d_model]
                b_enc = sae.encoder.bias.to(DEVICE).to(torch.bfloat16)    # [d_sae]

                with torch.no_grad():
                    acts_sae    = torch.matmul(normed_h, W_enc.t()) + b_enc
                    grad_sae    = torch.matmul(h_grad_vec, W_enc.t())
                    attribution = acts_sae * grad_sae

                    top_val, top_idx = torch.topk(attribution.abs(), 10)
                    sae_results = [
                        {"id": i.item(), "val": float(attribution[i].item())}
                        for i in top_idx
                    ]

                labels = [f"Feature-{x['id']}" for x in sae_results]
                values = [x['val'] for x in sae_results]
                colors = ['coral' if v >= 0 else 'steelblue' for v in values]

                plt.figure(figsize=(10, 5))
                plt.bar(range(10), values, color=colors)
                plt.xticks(range(10), labels, rotation=45, ha='right')
                plt.axhline(0, color='black', linewidth=0.8)
                plt.title(f"Top 10 SAE Feature Attributions — Layer {layer_idx}")
                plt.ylabel("Activation × Gradient")
                plt.tight_layout()
                plt.savefig(f"llama_out/llama_sae_plots/layer_{layer_idx:02d}_sae.png")
                plt.close()
                print(f"  Layer {layer_idx:02d}: SAE plot saved.")

            except Exception as e:
                # FIX 12: print the actual error instead of silently swallowing it
                print(f"  Layer {layer_idx:02d}: SAE attribution failed — {e}")
        else:
            print(f"  Layer {layer_idx:02d}: hidden state grad is None — SAE skipped.")
    # Don't print warning for every missing SAE - will summarize at the end

    export_data["sae_top_features"] = sae_results

    with open(f"llama_out/llama_sae_json/layer_{layer_idx:02d}_analysis.json", "w") as f:
        json.dump(export_data, f, indent=2)

# ==========================================
# 8. Global dynamics plot
# ==========================================
print("\nSaving Layer Dynamics Plots ...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(LAYERS_TO_ANALYZE, layer_entropy, marker='o', color='purple', markersize=4)
ax1.set_title("Logit Lens Entropy per Layer")
ax1.set_xlabel("Layer")
ax1.set_ylabel("Entropy (nats)")
ax1.grid(True, linestyle="--", alpha=0.6)

ax2.plot(LAYERS_TO_ANALYZE, layer_similarity, marker='s', color='teal', markersize=4)
ax2.set_title("Consecutive Layer Cosine Similarity")
ax2.set_xlabel("Layer")
ax2.set_ylabel("Cosine Sim")
ax2.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("llama_out/llama_layer_dynamics/overall_dynamics.png")
plt.close()

# Print SAE coverage summary
if saes:
    available_layers = [int(hp.split('.')[1]) for hp in saes.keys() if hp.startswith('layers.')]
    print(f"\n{'='*60}")
    print(f"SAE Coverage: {len(available_layers)}/{num_layers} layers")
    print(f"Available SAE layers: {sorted(available_layers)}")
    print(f"{'='*60}")

print("\nAnalysis complete. All outputs written to llama_out/")
