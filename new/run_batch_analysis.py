import os
import time
import torch
import json
from llama_interpret_lib import initialize_engine, run_analysis
from diverse_prompts import PROMPTS_BY_DOMAIN

def main():
    # 1. Initialize
    print("=== Initializing Batch Analysis Framework ===")
    try:
        engine = initialize_engine()
    except Exception as e:
        print(f"FAILED TO INITIALIZE: {e}")
        return

    output_root = "batch_results"
    os.makedirs(output_root, exist_ok=True)

    summary_log = []
    
    start_time = time.time()
    
    # 2. Iterate Domains
    total_prompts = sum(len(p) for p in PROMPTS_BY_DOMAIN.values())
    processed_count = 0

    for domain, prompts in PROMPTS_BY_DOMAIN.items():
        print(f"\n>>> Processing Domain: {domain}")
        domain_dir = f"{output_root}/{domain}"
        os.makedirs(domain_dir, exist_ok=True)

        for i, prompt in enumerate(prompts):
            processed_count += 1
            prompt_id = f"P{i+1:02d}_{prompt[:15].replace(' ', '_').lower()}"
            prompt_dir = f"{domain_dir}/{prompt_id}"
            
            print(f"  [{processed_count}/{total_prompts}] Analyzing: '{prompt[:50]}...'")
            
            try:
                result = run_analysis(engine, prompt, prompt_dir)
                
                summary_log.append({
                    "domain": domain,
                    "prompt": prompt,
                    "predicted": result["predicted_word"],
                    "avg_entropy": result["avg_entropy"],
                    "avg_sim": result["avg_sim"],
                    "path": prompt_dir
                })
            except Exception as e:
                print(f"    ERROR analyzing prompt: {e}")
                torch.cuda.empty_cache() # Try to recover
            
            # Memory housekeeping
            torch.cuda.empty_cache()

    # 3. Finalize
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n=== Batch Analysis Complete ===")
    print(f"Total Prompts: {processed_count}")
    print(f"Total Time: {duration/60:.2f} minutes")
    
    # Save a master summary file for the reporting script
    with open(f"{output_root}/master_summary.json", "w") as f:
        json.dump(summary_log, f, indent=2)
    
    print(f"Master summary saved to {output_root}/master_summary.json")

if __name__ == "__main__":
    main()
