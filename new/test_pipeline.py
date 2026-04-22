from llama_interpret_lib import initialize_engine, run_analysis
import os
import torch

def test():
    print("Testing pipeline with a single prompt...")
    try:
        engine = initialize_engine()
        prompt = "What is the capital of France?"
        output_dir = "test_run"
        os.makedirs(output_dir, exist_ok=True)
        
        result = run_analysis(engine, prompt, output_dir)
        print(f"SUCCESS! Predicted: {result['predicted_word']}")
        print(f"Avg Entropy: {result['avg_entropy']:.4f}")
    except Exception as e:
        print(f"TEST FAILED: {e}")
    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    test()
