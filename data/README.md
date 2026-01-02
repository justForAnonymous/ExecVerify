# ExecVerify Data Pipeline

This directory contains the complete data generation pipeline for ExecVerify, a system for training models to verify code execution through trace-based reasoning.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
  - [1. Python Environment](#1-python-environment)
  - [2. Model Setup](#2-model-setup)
  - [3. Configuration](#3-configuration)
- [Quick Start](#quick-start)
- [Pipeline Steps](#pipeline-steps)
  - [Step 1: Code Synthesis](#step-1-code-synthesis)
  - [Step 2: Input Synthesis](#step-2-input-synthesis)
  - [Step 3: Filter by Execution](#step-3-filter-by-execution)
  - [Step 4: Filter by Difficulty](#step-4-filter-by-difficulty)
  - [Step 5: Combine Data with Difficulties](#step-5-combine-data-with-difficulties)
  - [Step 6a: Extract IO Chain-of-Thought](#step-6a-extract-io-chain-of-thought)
  - [Step 6b: Extract OI Chain-of-Thought](#step-6b-extract-oi-chain-of-thought)
  - [Step 7: Combine SFT Dataset](#step-7-combine-sft-dataset)
  - [Step 8: Extract Candidates for RL](#step-8-extract-candidates-for-rl)
  - [Step 9: Extract Execution Traces](#step-9-extract-execution-traces)
  - [Step 10: Extract Trace Dataset for White-box RL](#step-10-extract-trace-dataset)
- [Output Files](#output-files)
  - [Final Datasets](#final-datasets)
  - [Intermediate Files](#intermediate-files)
- [Configuration Options](#configuration-options)
  - [Model Paths](#model-paths)
  - [Generation Parameters](#generation-parameters)
  - [Dataset Sizes](#dataset-sizes)
  - [Filtering Parameters](#filtering-parameters)

## Overview

The pipeline generates two types of datasets:
1. **SFT Dataset**: For supervised fine-tuning on input-output and output-input prediction tasks
2. **RL Dataset**: For white-box reinforcement learning with control-flow and data-flow questions

## Prerequisites

### 1. Python Environment

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Setup

Download or prepare the following models:
- **Code Generator**: A code generation model (e.g.,  QwQ-32B)
- **Reasoning Model**: A reasoning model (e.g., QwQ-32B)
- **Difficulty Evaluator**: A smaller model for evaluation (e.g., Qwen2.5-Coder-7B)

### 3. Configuration

Update `config.yaml` with your model paths:

```yaml
models:
  code_generator: /path/to/your/code-generation-model
  reasoning_model: /path/to/your/reasoning-model
  difficulty_evaluator: /path/to/your/evaluation-model
```

You can also adjust hyperparameters, dataset sizes, and other settings in `config.yaml`.



## Quick Start

Run the entire pipeline with a single command:

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

This will execute all 10 steps sequentially and generate the final datasets.

## Pipeline Steps

The pipeline consists of 10 steps:

### Step 1: Code Synthesis
```bash
python code_synthesis.py
```
**Output**: `raw_dataset.json`

Generates initial code samples by testing various Python built-in methods with different complexity configurations (nested calls, control flow, etc.).

### Step 2: Input Synthesis
```bash
python input_synthesis.py
```
**Output**: `mutated_raw_dataset.json`

Mutates input arguments in the raw dataset to create diverse test cases with varied data  values.

### Step 3: Filter by Execution
```bash
python filter_by_execution.py
```
**Output**: `processed_raw_dataset.json`, `processed_mutated_dataset.json`

Filters code samples by executing them and keeping only those that run successfully and produce valid outputs.

### Step 4: Filter by Difficulty
```bash
python filter_by_difficulty.py --dataset raw
python filter_by_difficulty.py --dataset mutated
```
**Output**: `processed_raw_dataset_with_difficulties.json`, `processed_mutated_dataset_with_difficulties.json`

Evaluates the difficulty of each sample by testing how often a model can correctly predict outputs (pass_cnt).

### Step 5: Combine Data with Difficulties
```bash
python combine_data_with_difficulties.py
```
**Output**: `filtered_all_dataset_with_difficulties.json`

Combines raw and mutated datasets, filtering out samples that are too easy (pass_cnt > threshold).

### Step 6a: Extract IO Chain-of-Thought
```bash
python extract_io_cot.py
```
**Output**: `filtered_all_dataset_with_difficulties_subset_with_io_cot.json`, `io_sft_dataset.json`

Extracts chain-of-thought reasoning for input-to-output prediction tasks.

### Step 6b: Extract OI Chain-of-Thought
```bash
python extract_oi_cot.py
```
**Output**: `filtered_all_dataset_with_difficulties_subset_with_oi_cot.json`, `oi_sft_dataset.json`

Extracts chain-of-thought reasoning for output-to-input prediction tasks.

### Step 7: Combine SFT Dataset
```bash
python combine_sft_dataset.py
```
**Output**: `sft_dataset.json`

Combines IO and OI SFT datasets into a single training dataset.

### Step 8: Extract Candidates for rl
```bash
python extract_candidates_for_multi_task.py --mode io --start-idx 40000 --end-idx 80000
python extract_candidates_for_multi_task.py --mode oi --start-idx 40000 --end-idx 62000
```
**Output**: `candidates_io_for_multi_task_with_cot.json`, `candidates_oi_for_multi_task_with_cot.json`

Extracts high-quality candidates for white-box reinforcement learning.

### Step 9: Extract Execution Traces
```bash
python extract_trace.py
```
**Output**: `io_dataset_for_mutiple_tasks_with_traces.json`

Generates execution traces by running code and recording line-by-line execution with variable states.

### Step 10: Extract Trace Dataset for White-box RL
```bash
python extract_trace_dataset.py
```
**Output**: `rl_dataset.json`

Creates the final white-box RL dataset by extracting various question types from execution traces (CF questions, DF questions) and combining with OI candidates.

## Output Files

### Final Datasets

- **`sft_dataset.json`**: Supervised fine-tuning dataset (~30,000 samples)
  - Format: LLaMA-Factory compatible conversation format
  - Tasks: Input-output prediction, output-input prediction

- **`rl_dataset.json`**: Reinforcement learning dataset (~30,000 samples)
  - Format: Json format with prompts and ground truth
  - Tasks: Assertion completion, white-box questions like path prediction and variable state prediction

### Intermediate Files

All intermediate files are preserved for inspection and debugging:
- `raw_dataset.json`: Initial generated code
- `mutated_raw_dataset.json`: Code with mutated inputs
- `processed_*.json`: Executable code samples
- `*_with_difficulties.json`: Samples with difficulty scores
- `*_with_cot.json`: Samples with chain-of-thought reasoning
- `*_with_traces.json`: Samples with execution traces

## Configuration Options

### Model Paths
Set in `config.yaml` under `models:` section

### Generation Parameters
Adjust temperature, top_p, max_tokens, etc. in `config.yaml` under `generation:` section

### Dataset Sizes
Control the number of samples at each stage in `config.yaml` under `dataset_limits:` section

### Filtering Parameters
Adjust difficulty threshold, worker count, timeouts in `config.yaml` under `filtering:` section

