import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import Gemma3ForConditionalGeneration, AutoTokenizer
from huggingface_hub import hf_hub_download

# ==========================================
# 1. Setup & Initialization
# ==========================================
def setup_directories():
    dirs = ["heatmaps", "sae_plots", "sae_json"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

# Configuration
MODEL_ID = "google/gemma-3-27b-it"
SAE_REPO = "uzaymacar/gemma-3-27b-saes"
PROMPT = "The capital of France is"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAE_CONFIGS = {
    45: {"resid": "layer_45/dict_16k_k80"},
    47: {"resid": "layer_47/dict_16k_k80"},
}
LAYERS_TO_ANALYZE = [45, 47]

setup_directories()

print(f"Loading Gemma-3-27b-it...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.eval()

# ==========================================
# 2. Forward & Backward Pass
# ==========================================
print(f"Analyzing prompt: '{PROMPT}'")
inputs = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)

# Correct Path: Gemma3ForConditionalGeneration -> model (Gemma3Model)
outputs = model(
    **inputs,
    output_hidden_states=True, 
    output_attentions=True
)

hidden_states = outputs.hidden_states
attentions = outputs.attentions

# Enable gradient tracking
for h in hidden_states:
    h.requires_grad_(True)
    h.retain_grad()

if attentions:
    for a in attentions:
        a.requires_grad_(True)
        a.retain_grad()

# Get predicted token
logits = outputs.logits
final_token_logits = logits[0, -1, :]
predicted_token_id = torch.argmax(final_token_logits).item()
predicted_word = tokenizer.decode(predicted_token_id)
print(f"Model predicted: '{predicted_word}'")

# Backward pass
model.zero_grad()
target_logit = final_token_logits[predicted_token_id]
target_logit.backward(retain_graph=True)

# ==========================================
# 3. Analysis Loop
# ==========================================
with torch.no_grad():
    for layer_idx in LAYERS_TO_ANALYZE:
        print(f"\n--- Layer {layer_idx} Analysis ---")

        # A. Logit Lens using model.model.layers
        try:
            # Gemma-3 layers are at model.model.layers
            target_block = model.model.layers[layer_idx]
            h_raw = hidden_states[layer_idx][0, -1, :].unsqueeze(0).unsqueeze(0)
            
            # Apply layer-specific RMSNorm
            normed_h = target_block.input_layernorm(h_raw).squeeze()
            lens_logits = model.lm_head(normed_h)
            
            top_5_indices = torch.topk(lens_logits, 5).indices
            top_5_words = [tokenizer.decode(idx.item()).strip() for idx in top_5_indices]
            print(f"Logit Lens: {top_5_words}")
        except Exception as e:
            print(f"Logit Lens Error: {e}")
            normed_h = hidden_states[layer_idx][0, -1, :]
            top_5_words = ["Error"]

        # B. AI Self-Explanation
        explainer_text = f"Layer {layer_idx} focus: {', '.join(top_5_words)}. Concept:"
        exp_in = tokenizer(explainer_text, return_tensors="pt").to(DEVICE)
        exp_out = model.generate(**exp_in, max_new_tokens=25, do_sample=False)
        explanation = tokenizer.decode(exp_out[0][exp_in.input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"Explanation: {explanation}")

        # C. Attention LRP
        if attentions and layer_idx < len(attentions):
            attn = attentions[layer_idx]
            if attn.grad is not None:
                relevance = (attn * attn.grad)[0].sum(dim=0).cpu().float().numpy()
                plt.figure(figsize=(6, 5))
                sns.heatmap(relevance, cmap="RdBu_r", center=0)
                plt.title(f"Attention Attribution (Layer {layer_idx})")
                plt.savefig(f"heatmaps/layer_{layer_idx}_attn.png")
                plt.close()

        # D. SAE Feature Attribution
        sae_path = SAE_CONFIGS[layer_idx].get("resid")
        if sae_path:
            try:
                ae_file = hf_hub_download(repo_id=SAE_REPO, filename=f"{sae_path}/ae.pt")
                ckpt = torch.load(ae_file, map_location=DEVICE)
                
                # Unwrap nested 'ae' key for checkpoints like Layer 47
                ae_state = ckpt['ae'] if 'ae' in ckpt else ckpt
                w_key = 'encoder.weight' if 'encoder.weight' in ae_state else 'W_enc'
                W_enc = ae_state[w_key].to(DEVICE).to(torch.bfloat16)

                h_grad = hidden_states[layer_idx].grad[0, -1, :]
                
                acts_sae = torch.matmul(normed_h, W_enc.t())
                grad_sae = torch.matmul(h_grad, W_enc.t())
                attribution = acts_sae * grad_sae

                top_val, top_idx = torch.topk(attribution.abs(), 10)
                sae_results = [{"id": i.item(), "val": attribution[i].item()} for i in top_idx]
                
                plt.figure(figsize=(8, 4))
                plt.bar(range(10), [x['val'] for x in sae_results])
                plt.xticks(range(10), [f"F-{x['id']}" for x in sae_results], rotation=45)
                plt.title(f"Top 10 SAE Attributions (Layer {layer_idx})")
                plt.tight_layout()
                plt.savefig(f"sae_plots/layer_{layer_idx}_sae.png")
                plt.close()

            except Exception as e:
                print(f"SAE Error: {e}")

print("\nAnalysis Finished.")