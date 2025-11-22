MODEL="gpt2"
SEEDS=(42 123 456 789 1024)

for seed in "${SEEDS[@]}"; do
  echo "Running with seed $seed..."
  
  # Fit gates
  python src/train/fit_gates.py \
    --model $MODEL \
    --out "checkpoints/gates_init_seed${seed}.pt" \
    --seed $seed \
    --epochs 3
  
  # Evaluate
  python src/train/eval_ablation.py \
    --model $MODEL \
    --gates "checkpoints/gates_init_seed${seed}.pt" \
    --output "results/ablation_seed${seed}.json"
done

echo "Multi-seed experiment complete!"
echo "Average results across seeds:"
python scripts/aggregate_seeds.py results/ablation_seed*.json
