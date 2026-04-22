# llama_8b_analysis.py - Technical Explanation

This script is a sophisticated interpretability tool designed to analyze the inner workings of the **Llama 3.1 8B** model during a single inference pass. It uses several advanced techniques to "peer inside" the transformer layers and understand how the model reaches its conclusions.

## Key Features

### 1. Model & SAE Integration
- **Model**: Uses `meta-llama/Meta-Llama-3.1-8B` (requires Hugging Face authentication).
- **Sparse Autoencoders (SAEs)**: Loads pre-trained SAEs from `EleutherAI/sae-llama-3.1-8b-32x`. These are used to decompose dense activations into thousands of sparse, interpretable "features."

### 2. Logit Lens Analysis
The script performs "Logit Lens" decoding at every layer. This involves:
- Taking the hidden state of a layer and passing it through the model's final normalization and output head.
- This allows us to see what the model "thinks" the next token should be at intermediate stages of computation.
- **Entropy Tracking**: Calculates how focused or uncertain the model's prediction is at each layer.
- **Cosine Similarity**: Measures how much the internal representation changes between consecutive layers.

### 3. Gradient-Based Attribution
Instead of just looking at activations, the script uses **Integrated Gradients (or raw Gradient $\times$ Activation)**:
- It calculates the gradient of the predicted token's logit with respect to internal hidden states and attention maps.
- **Attention Heatmaps**: Generates maps showing which tokens the model was "paying attention to" when making a specific prediction, weighted by their actual influence on the output.
- **SAE Feature Attribution**: Identifies which specific SAE features (conceptual units) contributed most to the final predicted token.

### 4. AI Self-Explanation
The script implements a "recursive" interpretability trick:
- It takes the top words found by the Logit Lens at various depths (every 8th layer).
- It prompts the model itself to describe the "common concept" shared by those words.
- This provides human-readable labels for abstract internal states.

## Output Structure
The script generates a `llama_out/` directory with the following subdirectories:

- `llama_heatmaps/`: PNG files showing attention relevance per layer.
- `llama_sae_plots/`: Bar charts showing the top 10 SAE features that influenced the prediction at each layer.
- `llama_sae_json/`: Detailed JSON files for every layer, containing logit lens results, entropy, and feature IDs.
- `llama_layer_dynamics/`: An overview plot showing the evolution of entropy and representation similarity across the entire model.

## Requirements
- `torch`
- `transformers`
- `matplotlib`
- `seaborn`
- `sparsify` (EleutherAI's SAE library)
- `huggingface_hub` (with access to Llama 3.1)

## How to Run
```bash
python llama_8b_analysis.py
```
*Note: This requires a GPU with enough VRAM to load Llama 3.1 8B (approx 16GB-20GB in bfloat16).*
