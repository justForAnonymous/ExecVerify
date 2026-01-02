#!/bin/bash

# 1. Parameter check
if [ -z "$1" ]; then
  echo "Usage: bash run_eval.sh <model_path> [tp_size]"
  echo "Example: bash run_eval.sh /path/to/model 4"
  exit 1
fi

MODEL_PATH=$1
TP_SIZE=${2:-1} # Default TP=1

# 2. Define task configurations
# Format: "Python parameter name | Output filename | EvalPlus parameter name"
CONFIGS=(
  "human_eval|he_samples.jsonl|humaneval"
  "mbpp|mbpp_samples.jsonl|mbpp"
)

echo "========================================================"
echo "Starting Evaluation Pipeline"
echo "Model: $MODEL_PATH"
echo "TP Size: $TP_SIZE"
echo "========================================================"

for CONFIG in "${CONFIGS[@]}"; do
  # Parse configuration
  IFS='|' read -r PY_DATASET OUTPUT_FILE EVAL_DATASET <<< "$CONFIG"
  
  # Derive related filenames
  SANITIZED_FILE="${OUTPUT_FILE%.jsonl}-sanitized.jsonl"
  RESULTS_FILE="${SANITIZED_FILE%.jsonl}_eval_results.json"

  echo ""
  echo "--------------------------------------------------------"
  echo "Processing Dataset: $EVAL_DATASET"
  echo "--------------------------------------------------------"

  # [0/4] Check and clean old files
  echo "[0/4] Checking for old files..."
  
  # Use [[ ]] for conditional check, see if any file exists
  if [[ -f "$OUTPUT_FILE" || -f "$SANITIZED_FILE" || -f "$RESULTS_FILE" ]]; then
      echo "Found old files. Cleaning up:"
      # Only delete if file exists, -v shows deletion details
      rm -vf "$OUTPUT_FILE" "$SANITIZED_FILE" "$RESULTS_FILE"
  else
      echo "No old files found. Clean start."
  fi

  # [1/4] Generate code
  echo "[1/4] Generating samples..."
  python eval_plus.py \
    --model "$MODEL_PATH" \
    --dataset "$PY_DATASET" \
    --tp "$TP_SIZE" \
    --output "$OUTPUT_FILE"

  # Check if Python script executed successfully
  if [ $? -ne 0 ]; then
    echo "Error: Generation failed for $EVAL_DATASET. Skipping evaluation."
    continue
  fi

  # [2/4] Sanitize data
  echo "[2/4] Sanitizing samples..."
  evalplus.sanitize --samples "$OUTPUT_FILE"

  # [3/4] Syntax check
  echo "[3/4] Running syntax check..."
  evalplus.syncheck \
    --samples "$SANITIZED_FILE" \
    --dataset "$EVAL_DATASET"

  # [4/4] Final evaluation
  echo "[4/4] Running evaluation..."
  evalplus.evaluate \
    --dataset "$EVAL_DATASET" \
    --samples "$SANITIZED_FILE" \
    --parallel 8

  echo "Finished pipeline for $EVAL_DATASET"
done

echo ""
echo "========================================================"
echo "All evaluations completed."
echo "========================================================"