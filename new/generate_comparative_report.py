import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def generate_report():
    summary_path = "batch_results/master_summary.json"
    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found. Run the batch analysis first.")
        return

    with open(summary_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    
    report_dir = "batch_results/Comparative_Analysis"
    os.makedirs(report_dir, exist_ok=True)

    # 1. Domain Entropy Comparison
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="domain", y="avg_entropy", data=df, palette="viridis")
    plt.xticks(rotation=45)
    plt.title("Llama 3.1 8B: Prediction Uncertainty (Entropy) by Domain")
    plt.tight_layout()
    plt.savefig(f"{report_dir}/entropy_comparison.png")
    plt.close()

    # 2. Domain Similarity Comparison
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="domain", y="avg_sim", data=df, palette="magma")
    plt.xticks(rotation=45)
    plt.title("Llama 3.1 8B: Representation Stability (Cosine Sim) by Domain")
    plt.tight_layout()
    plt.savefig(f"{report_dir}/similarity_comparison.png")
    plt.close()

    # 3. Generate Markdown Report
    report_md = f"""# Comparative Interpretability Report: Llama 3.1 8B

## Executive Summary
This report analyzes how Llama 3.1 8B internally processes 50 diverse prompts across 10 domains. We measure **Entropy** (predictive uncertainty at intermediate layers) and **Cosine Similarity** (stability of representations between layers).

## Domain Insights

### 1. Predictive Uncertainty (Entropy)
![Entropy Comparison](entropy_comparison.png)
*High entropy indicates the model is exploring many possible tokens before settling. Low entropy suggests immediate conceptual clarity.*

### 2. Representation Stability (Cosine Similarity)
![Similarity Comparison](similarity_comparison.png)
*High similarity suggests smooth, incremental updates to the hidden state. Low similarity indicates sharp 'conceptual shifts' occurring in the layers.*

## Detailed Statistics

| Domain | Avg Entropy | Avg Cosine Sim |
|--------|-------------|----------------|
"""
    
    stats = df.groupby("domain")[["avg_entropy", "avg_sim"]].mean().reset_index()
    for _, row in stats.iterrows():
        report_md += f"| {row['domain']} | {row['avg_entropy']:.4f} | {row['avg_sim']:.4f} |\n"

    report_md += "\n\n## Conclusion\nBy analyzing the cross-domain variance, we can identify which subjects push the model's 'reasoning' layers to work harder vs. which subjects are handled by shallow, memorized patterns."

    with open(f"{report_dir}/COMPARATIVE_REPORT.md", "w") as f:
        f.write(report_md)

    print(f"Report generated at {report_dir}/COMPARATIVE_REPORT.md")

if __name__ == "__main__":
    generate_report()
