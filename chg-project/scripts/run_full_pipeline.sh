set -e

MODEL="gpt2"
DATASET="wikitext"
CONFIG="wikitext-2-raw-v1"
EPOCHS=3
LR=0.01
DEVICE="cuda"

echo "===== CHG Full Pipeline ====="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo ""

# Create directories
mkdir -p checkpoints
mkdir -p results
mkdir -p plots

# Step 1: Fit initialization (G0)
echo "Step 1/5: Fitting initialization mask (λ=0)..."
python src/train/fit_gates.py \
  --model $MODEL \
  --dataset $DATASET \
  --dataset_config $CONFIG \
  --out checkpoints/gates_init.pt \
  --epochs $EPOCHS \
  --lr $LR \
  --lambda_reg 0.0 \
  --device $DEVICE

# Step 2: Fit retention mask (G+)
echo "Step 2/5: Fitting retention mask (λ<0)..."
python src/train/fit_gates.py \
  --model $MODEL \
  --dataset $DATASET \
  --dataset_config $CONFIG \
  --init checkpoints/gates_init.pt \
  --out checkpoints/gates_retention.pt \
  --epochs $EPOCHS \
  --lr $LR \
  --lambda_reg -0.1 \
  --device $DEVICE

# Step 3: Fit removal mask (G-)
echo "Step 3/5: Fitting removal mask (λ>0)..."
python src/train/fit_gates.py \
  --model $MODEL \
  --dataset $DATASET \
  --dataset_config $CONFIG \
  --init checkpoints/gates_init.pt \
  --out checkpoints/gates_removal.pt \
  --epochs $EPOCHS \
  --lr $LR \
  --lambda_reg 0.1 \
  --device $DEVICE

# Step 4: Evaluate ablations
echo "Step 4/5: Evaluating head importance..."
python src/train/eval_ablation.py \
  --model $MODEL \
  --gates checkpoints/gates_init.pt \
  --mode individual \
  --output results/ablation_results.json \
  --device $DEVICE

# Step 5: Generate analysis
echo "Step 5/5: Generating visualizations..."
python src/analysis/analyze.py \
  --results results/ablation_results.json \
  --gates checkpoints/gates_init.pt \
  --output_dir plots

echo ""
echo "===== Pipeline Complete! ====="
echo "Checkpoints saved in: checkpoints/"
echo "Results saved in: results/"
echo "Plots saved in: plots/"