#!/bin/bash

# BigCodeBench Evaluation Script

# Check if model path is provided
if [ -z "$1" ]; then
  echo "Usage: bash run_bcb_eval.sh <model_path> [tensor_parallel_size]"
  echo "Example: bash run_bcb_eval.sh /path/to/model 1"
  exit 1
fi

MODEL_PATH=$1
TP_SIZE=${2:-1}  # Default TP=1

echo "========================================================"
echo "BigCodeBench Evaluation Pipeline"
echo "Model: $MODEL_PATH"
echo "Tensor Parallel Size: $TP_SIZE"
echo "========================================================"

# Step 1: Generate samples
echo ""
echo "[1/4] Generating samples..."
python bigCodeBench.py \
    --model_path "$MODEL_PATH" \
    --dataset "bigcode/bigcodebench" \
    --output "bcb_samples.jsonl" \
    --tensor_parallel_size "$TP_SIZE"

if [ $? -ne 0 ]; then
  echo "Error: Sample generation failed."
  exit 1
fi

# Step 2: Sanitize samples
echo ""
echo "[2/4] Sanitizing samples..."
bigcodebench.sanitize --samples bcb_samples.jsonl --calibrate

if [ $? -ne 0 ]; then
  echo "Error: Sanitization failed."
  exit 1
fi

# Step 3: Syntax check
echo ""
echo "[3/4] Running syntax check..."
bigcodebench.syncheck --samples bcb_samples-sanitized-calibrated.jsonl

if [ $? -ne 0 ]; then
  echo "Error: Syntax check failed."
  exit 1
fi

# Step 4: Evaluate (Full and Hard subsets)
echo ""
echo "[4/4] Running evaluation..."

# Evaluate Full subset
echo "Evaluating Full subset..."
bigcodebench.evaluate \
  --split complete \
  --subset full \
  --samples bcb_samples-sanitized-calibrated.jsonl \
  --execution local \
  --bs 1

if [ $? -ne 0 ]; then
  echo "Error: Full subset evaluation failed."
  exit 1
fi

# Evaluate Hard subset
echo "Evaluating Hard subset..."
bigcodebench.evaluate \
  --split complete \
  --subset hard \
  --samples bcb_samples-sanitized-calibrated.jsonl \
  --execution local \
  --bs 1

if [ $? -ne 0 ]; then
  echo "Error: Hard subset evaluation failed."
  exit 1
fi

echo ""
echo "========================================================"
echo "BigCodeBench evaluation completed successfully!"
echo "Results:"
echo "  - Full subset: bcb_samples-sanitized-calibrated_eval_results.json"
echo "  - Hard subset: bcb_samples-sanitized-calibrated_eval_results.json"
echo "========================================================"

