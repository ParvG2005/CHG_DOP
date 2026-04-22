import os
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparsify import Sae

# Memory optimization
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

def initialize_engine(model_id="meta-llama/Meta-Llama-3.1-8B", sae_repo="EleutherAI/sae-llama-3.1-8b-32x"):
    """
    Initializes the model, tokenizer, and SAEs once and keeps them in VRAM.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {model_id} ...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.eval()

    print(f"Loading SAEs from {sae_repo} ...")
    saes = {}
    try:
        saes = Sae.load_many(sae_repo)
        print(f"Loaded {len(saes)} SAEs.")
    except Exception as e:
        print(f"WARNING: Could not load SAEs: {e}")

    # Determine layer path
    layer_path_fn = None
    if hasattr(model.model, 'layers'):
        layer_path_fn = lambda m: m.model.layers
    elif hasattr(model.model, 'text_model') and hasattr(model.model.text_model, 'layers'):
        layer_path_fn = lambda m: m.model.text_model.layers
    
    if layer_path_fn is None:
        raise RuntimeError("Could not detect layer structure for this model.")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "saes": saes,
        "device": device,
        "layer_path_fn": layer_path_fn,
        "num_layers": len(layer_path_fn(model))
    }

def run_analysis(engine, prompt, output_dir):
    """
    Performs full interpretability analysis for a single prompt.
    """
    model = engine["model"]
    tokenizer = engine["tokenizer"]
    saes = engine["saes"]
    device = engine["device"]
    layer_path_fn = engine["layer_path_fn"]
    num_layers = engine["num_layers"]

    # Setup directories
    dirs = [
        f"{output_dir}/heatmaps",
        f"{output_dir}/sae_plots",
        f"{output_dir}/layer_json"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # 1. Forward Pass
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model(
        **inputs,
        output_hidden_states=True,
        output_attentions=True
    )

    hidden_states = outputs.hidden_states
    attentions = outputs.attentions
    
    # Retain grads
    for h in hidden_states:
        if h.requires_grad:
            h.retain_grad()
    if attentions:
        for a in attentions:
            a.requires_grad_(True)
            a.retain_grad()

    logits = outputs.logits
    final_token_logits = logits[0, -1, :]
    predicted_token_id = torch.argmax(final_token_logits).item()
    predicted_word = tokenizer.decode(predicted_token_id).strip()

    # 2. Backward Pass
    model.zero_grad()
    final_token_logits[predicted_token_id].backward(retain_graph=False)

    # 3. Logit Lens & Entropy
    all_layer_data = {}
    layer_entropy = []
    layer_similarity = []

    for layer_idx in range(num_layers):
        target_block = layer_path_fn(model)[layer_idx]
        h_raw = hidden_states[layer_idx][0, -1, :].unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            normed_h = target_block.input_layernorm(h_raw).squeeze()
            
            # Cosine similarity
            if layer_idx > 0:
                cos_sim = F.cosine_similarity(
                    hidden_states[layer_idx - 1][0, -1, :].unsqueeze(0),
                    hidden_states[layer_idx][0, -1, :].unsqueeze(0)
                ).item()
            else:
                cos_sim = 1.0
            layer_similarity.append(cos_sim)

            # Logit Lens
            lens_logits = model.lm_head(normed_h)
            probs = F.softmax(lens_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            layer_entropy.append(entropy)

            top_k_result = torch.topk(probs, 10)
            logit_lens_results = [
                {"word": tokenizer.decode(idx.item()).strip(), "prob": float(p.item())}
                for idx, p in zip(top_k_result.indices, top_k_result.values)
            ]

        all_layer_data[layer_idx] = {
            "normed_h": normed_h,
            "logit_lens_results": logit_lens_results,
            "entropy": entropy,
            "cos_sim": cos_sim,
        }

    # 4. Layer Explanations (Every 8th layer)
    layer_explanations = {}
    for layer_idx in range(0, num_layers, 8):
        top_5_words = [r['word'] for r in all_layer_data[layer_idx]['logit_lens_results'][:5]]
        explainer_text = f"The following 5 words represent a concept: {', '.join(top_5_words)}. Identify the common concept concisely: "
        exp_in = tokenizer(explainer_text, return_tensors="pt").to(device)
        with torch.no_grad():
            exp_out = model.generate(**exp_in, max_new_tokens=15, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        explanation = tokenizer.decode(exp_out[0][exp_in.input_ids.shape[1]:], skip_special_tokens=True).strip()
        layer_explanations[layer_idx] = explanation

    # 5. Export Plots & JSON
    for layer_idx in range(num_layers):
        data = all_layer_data[layer_idx]
        normed_h = data["normed_h"]
        nearest_exp_layer = max((k for k in layer_explanations if k <= layer_idx), default=0)

        export_data = {
            "layer": layer_idx,
            "prompt": prompt,
            "predicted_target": predicted_word,
            "residual_cosine_similarity": data["cos_sim"],
            "entropy": data["entropy"],
            "logit_lens": data["logit_lens_results"],
            "ai_explanation": layer_explanations.get(nearest_exp_layer, ""),
            "sae_top_features": []
        }

        # Attention Heatmap
        if attentions and layer_idx < len(attentions):
            attn = attentions[layer_idx]
            if attn.grad is not None:
                relevance = (attn * attn.grad)[0].sum(dim=0).cpu().detach().float().numpy()
                plt.figure(figsize=(10, 8))
                sns.heatmap(relevance, cmap="viridis", annot=False)
                plt.title(f"Attention Attribution — Layer {layer_idx}")
                plt.savefig(f"{output_dir}/heatmaps/layer_{layer_idx:02d}_attn.png")
                plt.close()

        # SAE Attribution
        hookpoint = f"layers.{layer_idx}.mlp"
        sae = saes.get(hookpoint)
        if sae is not None:
            h_grad = hidden_states[layer_idx].grad
            if h_grad is not None:
                h_grad_vec = h_grad[0, -1, :]
                try:
                    W_enc = sae.encoder.weight.to(device).to(torch.bfloat16)
                    b_enc = sae.encoder.bias.to(device).to(torch.bfloat16)
                    with torch.no_grad():
                        acts_sae = torch.matmul(normed_h, W_enc.t()) + b_enc
                        grad_sae = torch.matmul(h_grad_vec, W_enc.t())
                        attribution = acts_sae * grad_sae
                        top_val, top_idx = torch.topk(attribution.abs(), 10)
                        sae_results = [{"id": i.item(), "val": float(attribution[i].item())} for i in top_idx]
                        export_data["sae_top_features"] = sae_results

                    # Plot SAE
                    labels = [f"F-{x['id']}" for x in sae_results]
                    values = [x['val'] for x in sae_results]
                    plt.figure(figsize=(10, 5))
                    plt.bar(range(10), values, color=['coral' if v >= 0 else 'steelblue' for v in values])
                    plt.xticks(range(10), labels, rotation=45)
                    plt.title(f"Top 10 SAE Features — Layer {layer_idx}")
                    plt.savefig(f"{output_dir}/sae_plots/layer_{layer_idx:02d}_sae.png")
                    plt.close()
                except:
                    pass

        with open(f"{output_dir}/layer_json/layer_{layer_idx:02d}_analysis.json", "w") as f:
            json.dump(export_data, f, indent=2)

    # Save summary dynamics plot for this prompt
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_layers), layer_entropy, label="Entropy", color="purple")
    plt.plot(range(num_layers), layer_similarity, label="Cos Sim", color="teal")
    plt.title(f"Dynamics: {prompt[:30]}...")
    plt.legend()
    plt.savefig(f"{output_dir}/summary_dynamics.png")
    plt.close()

    return {
        "predicted_word": predicted_word,
        "avg_entropy": sum(layer_entropy) / len(layer_entropy),
        "avg_sim": sum(layer_similarity) / len(layer_similarity)
    }
