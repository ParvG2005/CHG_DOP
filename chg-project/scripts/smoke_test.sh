set -e

echo "Running CHG smoke test..."

# Test 1: Gate fitting
echo "Test 1: Fitting gates on tiny dataset..."
python src/train/fit_gates.py \
  --model gpt2 \
  --dataset wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --dataset_split "train[:10]" \
  --out /tmp/test_gates.pt \
  --epochs 1 \
  --batch_size 1

# Test 2: Ablation
echo "Test 2: Running ablation..."
python src/train/eval_ablation.py \
  --model gpt2 \
  --gates /tmp/test_gates.pt \
  --mode individual \
  --output /tmp/test_results.json \
  --prompts "Hello world"

# Test 3: Analysis
echo "Test 3: Generating analysis..."
python src/analysis/analyze.py \
  --results /tmp/test_results.json \
  --gates /tmp/test_gates.pt \
  --output_dir /tmp/test_analysis

echo ""
echo "✅ All tests passed!"
echo "Installation is working correctly."

# Cleanup
rm -rf /tmp/test_gates.pt /tmp/test_results.json /tmp/test_analysis