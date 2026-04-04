import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE


# ==========================================
# 1. Setup & Initialization
# ==========================================
def setup_directories():
    """Creates the output directory structure."""
    dirs = ["heatmaps", "sae_plots", "sae_json"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"Directory structure initialized: {dirs}")


# Configuration
MODEL_ID = "google/gemma-3-27b-it"
SAE_RELEASE = "uzaymacar/gemma-3-27b-saes"
PROMPT = "The capital of France is"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

setup_directories()

print(f"Loading tokenizer and model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,  # Recommended for 27B models to save VRAM
)
model.eval()

# ==========================================
# 2. The Core Hooking & Forward/Backward Pass
# ==========================================
print(f"Tokenizing prompt: '{PROMPT}'")
inputs = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)

# Run forward pass, explicitly requesting hidden states and attention matrices
print("Running forward pass...")
outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

hidden_states = outputs.hidden_states
attentions = outputs.attentions
logits = outputs.logits

# Retain gradients for LRP calculations
for h in hidden_states:
    h.retain_grad()
for a in attentions:
    a.retain_grad()

# Get the final predicted next-token
final_token_logits = logits[0, -1, :]
predicted_token_id = torch.argmax(final_token_logits).item()
predicted_word = tokenizer.decode(predicted_token_id)
print(f"Predicted next token: '{predicted_word}'")

# Backward pass strictly from the predicted token's logit
print("Running backward pass from predicted logit...")
target_logit = final_token_logits[predicted_token_id]
model.zero_grad()
target_logit.backward()

# Number of hidden layers (ignoring the embedding layer at index 0)
num_layers = len(hidden_states) - 1

# ==========================================
# 3. Layer-by-Layer Analysis Loop
# ==========================================
# We wrap the loop in no_grad to prevent OOM when doing generation inside
with torch.no_grad():
    for layer_idx in range(1, num_layers + 1):
        print(f"\n--- Processing Layer {layer_idx}/{num_layers} ---")

        # Extract activations and gradients for the final token position
        h = hidden_states[layer_idx]
        h_grad = h.grad

        # We only care about the last token's representation for next-token prediction
        h_last = h[0, -1, :]
        h_grad_last = h_grad[0, -1, :]

        # ------------------------------------------
        # A. Logit Lens & AI Self-Explanation
        # ------------------------------------------
        # Apply LayerNorm and LM Head
        normed_h = model.model.norm(h_last)
        lens_logits = model.lm_head(normed_h)

        # Get top 5 tokens
        top_5_probs, top_5_indices = torch.topk(F.softmax(lens_logits, dim=-1), 5)
        top_5_words = [tokenizer.decode(idx.item()).strip() for idx in top_5_indices]

        # Generate Self-Explanation
        explanation_prompt = (
            f"I am analyzing an AI's internal brain. At layer {layer_idx}, "
            f"it is heavily focusing on the following words: {', '.join(top_5_words)}. "
            f"Explain in exactly 1 concise sentence what semantic concept or grammatical syntax this layer is currently processing."
        )

        exp_inputs = tokenizer(explanation_prompt, return_tensors="pt").to(DEVICE)
        exp_outputs = model.generate(
            **exp_inputs, max_new_tokens=40, temperature=0.3, do_sample=True
        )
        # Decode and strip out the prompt
        explanation = tokenizer.decode(
            exp_outputs[0][exp_inputs.input_ids.shape[1] :], skip_special_tokens=True
        ).strip()
        print(f"AI Self-Explanation: {explanation}")

        # ------------------------------------------
        # B. Attention Attribution (LRP)
        # ------------------------------------------
        # Attention shape: (batch, heads, seq_len, seq_len)
        attn = attentions[layer_idx - 1]  # 0-indexed in the tuple
        attn_grad = attn.grad

        # Relevance = Attention * Gradient (element-wise)
        attn_relevance = attn * attn_grad

        # Sum over batch and head dimensions to get an N x N matrix
        attn_heatmap = attn_relevance[0].sum(dim=0).cpu().numpy()

        # Plot Heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            attn_heatmap,
            cmap="coolwarm",
            center=0,
            xticklabels=tokenizer.convert_ids_to_tokens(inputs.input_ids[0]),
            yticklabels=tokenizer.convert_ids_to_tokens(inputs.input_ids[0]),
        )
        plt.title(f"Attention LRP Heatmap - Layer {layer_idx}")
        plt.xlabel("Key Tokens")
        plt.ylabel("Query Tokens")
        plt.tight_layout()
        plt.savefig(f"heatmaps/layer_{layer_idx}_attention_lrp.png")
        plt.close()

        # ------------------------------------------
        # C. Sparse Autoencoder (SAE) Projection
        # ------------------------------------------
        # Load layer-specific SAE
        sae_id = f"layer_{layer_idx}_resid_post"
        try:
            # sae_lens uses a specific loading mechanism
            sae, _, _ = SAE.from_pretrained(
                release=SAE_RELEASE, sae_id=sae_id, device=DEVICE
            )

            # Element-wise relevance vector
            relevance_vector = h_last * h_grad_last

            # Project onto the SAE Encoder Weights to get feature relevance
            # sae.W_enc shape is typically (d_model, d_sae)
            feature_relevance = torch.matmul(relevance_vector, sae.W_enc)

            # Extract top 15 features
            top_15_scores, top_15_indices = torch.topk(feature_relevance.abs(), 15)
            # Use original signed scores for the top indices
            top_15_actual_scores = feature_relevance[top_15_indices]

            features_data = []
            for score, feat_id in zip(top_15_actual_scores, top_15_indices):
                feat_id_val = feat_id.item()
                features_data.append(
                    {
                        "feature_id": feat_id_val,
                        "score": float(score.item()),
                        "neuronpedia_url": f"https://www.neuronpedia.org/gemma-3-27b/{layer_idx}/{feat_id_val}",
                    }
                )

            # Plot Bar Chart
            plt.figure(figsize=(10, 8))
            y_pos = np.arange(len(features_data))
            scores = [f["score"] for f in features_data]
            labels = [f"Feature {f['feature_id']}" for f in features_data]

            plt.barh(
                y_pos,
                scores,
                align="center",
                color=["red" if s < 0 else "blue" for s in scores],
            )
            plt.yticks(y_pos, labels)
            plt.gca().invert_yaxis()  # Highest scores at the top
            plt.xlabel("Relevance Score ($R_l \cdot W_{enc}$)")
            plt.title(f"Top 15 SAE Features - Layer {layer_idx}")
            plt.tight_layout()
            plt.savefig(f"sae_plots/layer_{layer_idx}_sae_features.png")
            plt.close()

        except Exception as e:
            print(
                f"Warning: Could not load or process SAE for layer {layer_idx}. Error: {e}"
            )
            features_data = [{"error": str(e)}]

        # ------------------------------------------
        # D. Data Export
        # ------------------------------------------
        export_data = {
            "layer": layer_idx,
            "prompt": PROMPT,
            "final_predicted_word": predicted_word,
            "top_5_logit_lens_words": top_5_words,
            "ai_self_explanation": explanation,
            "top_15_sae_features": features_data,
        }

        with open(f"sae_json/layer_{layer_idx}_data.json", "w") as f:
            json.dump(export_data, f, indent=4)

print("\nAnalysis complete! Check the heatmaps, sae_plots, and sae_json directories.")
