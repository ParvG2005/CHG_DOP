# Implementation Plan: Large-Scale Comparative Interpretability Analysis

This plan outlines the creation of a system to run interpretability analysis on 50 diverse prompts across 10 domains using Llama 3.1 8B and Sparse Autoencoders (SAEs), followed by a comparative analysis of how the model's internal representations vary across domains.

## User Review Required

> [!IMPORTANT]
> **Scale and Performance**: Running 50 full analyses (including 32 layers of plots per prompt) will generate **~3,200 images** and 1,600 JSON files.
> - Full plots (Attention Heatmaps and SAE Attributions) will be generated for **all 50 prompts**.
> - This is a manageable scale that should complete in ~30-60 minutes.

> [!TIP]
> **GPU Memory Optimization (100GB VRAM)**: 
> - Since 100GB VRAM is available, the library will keep the **entire model (~16GB)** and **all 32 SAE layers (~32GB)** permanently resident in VRAM.
> - **Performance**: This eliminates the latency of SAE swapping, allowing for significantly faster batch processing across the 50 prompts.

## Proposed Changes

### 1. Library Refactoring
#### [NEW] [llama_interpret_lib.py](file:///c:/3-2/project/tasks/llama_interpret_lib.py)
- Refactor `llama_8b_analysis.py` into a reusable library.
- Functions:
    - `initialize_engine()`: Loads Model, Tokenizer, and SAEs once.
    - `run_analysis(prompt, output_dir, generate_plots=True)`: Performs the logit lens, SAE attribution, and attribution mapping.

### 2. Data Preparation
#### [NEW] [diverse_prompts.py](file:///c:/3-2/project/tasks/diverse_prompts.py)
- A script to define/generate 50 prompts (5 per domain) across 10 categories:
    1.  **Software Engineering**: Coding, debugging, patterns.
    2.  **Medicine**: Diagnosis, pharmacology, anatomy.
    3.  **Law**: Contracts, constitutional, IP.
    4.  **Philosophy**: Ethics, epistemology, logic.
    5.  **Mathematics**: Calculus, statistics, algebra.
    6.  **Physics**: Quantum, relativity, energy.
    7.  **Economics**: Macro, game theory, behavioral.
    8.  **History**: Ancient, modern, revolutionary events.
    9.  **Psychology**: Cognitive biases, therapy models.
    10. **Literature**: Symbolism, narrative, poetry.

### 3. Execution Engine
#### [NEW] [run_batch_analysis.py](file:///c:/3-2/project/tasks/run_batch_analysis.py)
- Driver script that:
    - Calls `initialize_engine()`.
    - Iterates through the prompt list.
    - Saves JSON data for every prompt.
    - Saves plots only for a "Gold Set" of prompts.
    - Implements a progress bar and basic error handling.

### 4. Analysis & Comparison
#### [NEW] [generate_comparative_report.py](file:///c:/3-2/project/tasks/generate_comparative_report.py)
- Aggregates the JSON results from `batch_out/`.
- Computes:
    - Average Entropy per domain (which domains make the model "think" harder?).
    - Average Layer Similarity per domain (which domains have the most "conceptual shifts"?).
    - Top 10 most active SAE features per domain.
- Generates a final `Comparative_Analysis.md` with embedded summary charts.

## Output Directory Structure

The system will generate a structured `batch_results/` directory as follows:

```text
batch_results/
├── [Domain_Name]/                  # e.g., Software_Engineering/
│   ├── [Prompt_ID]/                # e.g., P01_python_debug/
│   │   ├── heatmaps/               # 32 Attention Attribution PNGs
│   │   ├── sae_plots/              # 32 SAE Attribution PNGs
│   │   ├── layer_json/             # 32 JSON files (logit lens, entropy, features)
│   │   └── summary_dynamics.png    # Entropy/Similarity overview for this prompt
│   └── ... (5 prompts per domain)
├── Comparative_Analysis/
│   ├── aggregate_stats.json        # Compiled data for all 50 prompts
│   ├── domain_comparison_plots/    # Cross-domain charts (Entropy, Feature overlap)
│   └── COMPARATIVE_REPORT.md       # Final human-readable analysis
└── logs/                           # Execution logs and errors
```

## Verification Plan

### Automated Tests
- Run `run_batch_analysis.py` with a small subset (e.g., 2 prompts per domain) to verify JSON and plot output.
- Run `generate_comparative_report.py` on the small subset to verify aggregation logic.

### Manual Verification
- Inspect the generated `Comparative_Analysis.md` to ensure the insights are meaningful.
- Check the `batch_out/` directory structure.
