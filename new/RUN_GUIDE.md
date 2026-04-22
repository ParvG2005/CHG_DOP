# User Guide: Llama 3.1 Interpretability Framework

This framework allows you to perform large-scale interpretability analysis on Llama 3.1 8B using Logit Lens and Sparse Autoencoders (SAEs) across diverse domains.

## 🚀 Quick Start

### 1. Prerequisites
- **Hardware**: A GPU with at least **48GB VRAM** is recommended to load the model and all SAEs simultaneously (e.g., A100, H100, or multiple RTX 3090/4090s).
- **Authentication**: You must be logged into Hugging Face and have access to the `meta-llama/Meta-Llama-3.1-8B` model.
  ```bash
  huggingface-cli login
  ```

### 2. Installation
Install the required interpretability and machine learning libraries:
```bash
pip install torch transformers matplotlib seaborn pandas sparsify huggingface_hub
```

### 3. Execution
The workflow is divided into two simple steps:

#### Step A: Run Batch Analysis
This will iterate through all 50 prompts and generate detailed internal data (heatmaps, SAE plots, and JSON metrics).
```bash
python run_batch_analysis.py
```
*Wait for completion (approx. 30-60 minutes).*

#### Step B: Generate Comparative Report
This will aggregate the data and generate the final cross-domain study.
```bash
python generate_comparative_report.py
```

---

## 📂 Folder Structure & Outputs

After running the workflow, your `batch_results/` directory will look like this:

- **`[Domain_Name]/`**: Folders for each of the 10 domains (e.g., Medicine, Law).
  - **`[Prompt_ID]/`**: Individual results for each prompt.
    - `heatmaps/`: 32 PNGs showing which tokens were most influential at each layer.
    - `sae_plots/`: 32 PNGs showing the top 10 conceptual "features" active in each layer.
    - `layer_json/`: 32 JSON files containing raw metrics for deep data analysis.
    - `summary_dynamics.png`: A high-level view of how entropy and stability evolved through the layers.
- **`Comparative_Analysis/`**:
  - `COMPARATIVE_REPORT.md`: Your final human-readable report with cross-domain charts.
  - `entropy_comparison.png`: Chart showing which domains had the most predictive uncertainty.
  - `similarity_comparison.png`: Chart showing where the most "conceptual shifts" occurred.

---

## 🛠️ Customization

### Adding New Prompts
To add your own research questions, open `diverse_prompts.py` and modify the `PROMPTS_BY_DOMAIN` dictionary. You can add new categories or change existing prompts.

### Changing the Model
If you wish to analyze a different model (e.g., Llama 3.1 8B Instruct), edit the `initialize_engine` function in `llama_interpret_lib.py`. Ensure you also update the `SAE_REPO` to match the target model.

---

## 🧠 Key Concepts
- **Logit Lens**: Peering into intermediate layers to see what the model "thinks" the answer is before it reaches the end.
- **SAE (Sparse Autoencoders)**: Decomposing abstract neural activations into human-interpretable concepts (features).
- **Entropy**: Measuring how "confused" or "certain" the model is at various depths.
