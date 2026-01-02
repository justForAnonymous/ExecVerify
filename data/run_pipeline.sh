#!/bin/bash
###############################################################################
# ExecVerify Data Pipeline - Complete Execution Script
#
# This script runs the entire data generation pipeline from start to finish.
# It generates datasets for training code execution verification models.
#
# Usage:
#   ./run_pipeline.sh
#
# Prerequisites:
#   1. Update config.yaml with your model paths
#   2. Install dependencies: pip install -r requirements.txt
#   3. Set environment variables (optional):
#      - CUDA_VISIBLE_DEVICES: GPU devices to use
#      - NCCL_NVLS_ENABLE: Set to 0 if needed
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if a file exists
check_file() {
    if [ ! -f "$1" ]; then
        log_error "Required file not found: $1"
        return 1
    fi
    return 0
}

# Run a Python script and check for success
run_step() {
    local script=$1
    local description=$2
    local start_time=$(date +%s)
    
    log_info "=========================================="
    log_info "Step: $description"
    log_info "Script: $script"
    log_info "=========================================="
    
    if python "$script"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_info "✓ Completed in ${duration}s: $description"
        return 0
    else
        log_error "✗ Failed: $description"
        return 1
    fi
}

# Main pipeline execution
main() {
    log_info "Starting ExecVerify Data Pipeline"
    log_info "Working directory: $(pwd)"
    
    # Check for config file
    if ! check_file "config.yaml"; then
        log_error "config.yaml not found. Please create it from the template."
        exit 1
    fi
    
    # Check for methods directory
    if [ ! -d "methods" ]; then
        log_error "methods/ directory not found"
        exit 1
    fi
    
    log_info "Configuration and prerequisites check passed"
    echo ""
    
    # Pipeline start time
    pipeline_start=$(date +%s)
    
    # Step 1: Code Synthesis
    run_step "code_synthesis.py" "Generate raw dataset" || exit 1
    echo ""
    
    # Step 2: Input Synthesis
    run_step "input_synthesis.py" "Mutate input arguments" || exit 1
    echo ""
    
    # Step 3: Filter by Execution
    run_step "filter_by_execution.py" "Filter executable code" || exit 1
    echo ""
    
    # Step 4a: Filter by Difficulty (Raw)
    run_step "filter_by_difficulty.py --dataset raw" "Evaluate difficulty (raw)" || exit 1
    echo ""
    
    # Step 4b: Filter by Difficulty (Mutated)
    run_step "filter_by_difficulty.py --dataset mutated" "Evaluate difficulty (mutated)" || exit 1
    echo ""
    
    # Step 5: Combine Data
    run_step "combine_data_with_difficulties.py" "Combine and filter datasets" || exit 1
    echo ""
    
    # Step 6a: Extract IO Chain-of-Thought
    run_step "extract_io_cot.py" "Extract IO chain-of-thought" || exit 1
    echo ""
    
    # Step 6b: Extract OI Chain-of-Thought
    run_step "extract_oi_cot.py" "Extract OI chain-of-thought" || exit 1
    echo ""
    
    # Step 7: Combine SFT Dataset
    run_step "combine_sft_dataset.py" "Combine SFT datasets" || exit 1
    echo ""
    
    log_info "SFT dataset generation complete!"
    log_info "Continuing with RL dataset generation..."
    echo ""
    
    # Step 8a: Extract IO Candidates for Multi-Task
    run_step "extract_candidates_for_multi_task.py --mode io --start-idx 40000 --end-idx 80000" \
        "Extract IO candidates for multi-task" || exit 1
    echo ""
    
    # Step 8b: Extract OI Candidates for Multi-Task
    run_step "extract_candidates_for_multi_task.py --mode oi --start-idx 40000 --end-idx 62000" \
        "Extract OI candidates for multi-task" || exit 1
    echo ""
    
    # Step 9: Extract Execution Traces
    run_step "extract_trace.py" "Generate execution traces" || exit 1
    echo ""
    
    # Step 10: Extract Final RL Dataset
    run_step "extract_trace_dataset.py" "Generate final RL dataset" || exit 1
    echo ""
    
    # Calculate total time
    pipeline_end=$(date +%s)
    total_duration=$((pipeline_end - pipeline_start))
    hours=$((total_duration / 3600))
    minutes=$(((total_duration % 3600) / 60))
    seconds=$((total_duration % 60))
    
    # Final summary
    log_info "=========================================="
    log_info "Pipeline Complete!"
    log_info "=========================================="
    log_info "Total time: ${hours}h ${minutes}m ${seconds}s"
    log_info ""
    log_info "Generated datasets:"
    log_info "  - sft_dataset.json (for supervised fine-tuning)"
    log_info "  - rl_dataset_multi_task.json (for reinforcement learning)"
    log_info ""
    log_info "All intermediate files are also saved for inspection."
    log_info "=========================================="
}

# Run main function
main "$@"

