# ExecVerify Training

This directory contains the complete training pipeline for ExecVerify, including supervised fine-tuning (SFT) and two-stage reinforcement learning (RL) post-training.

## Table of Contents

- [Overview](#overview)
- [Part 1: Supervised Fine-Tuning (SFT)](#part-1-supervised-fine-tuning-sft)
  - [1.1 Install LLaMA-Factory](#11-install-llama-factory)
  - [1.2 Prepare SFT Dataset](#12-prepare-sft-dataset)
  - [1.3 Configure Dataset](#13-configure-dataset)
  - [1.4 Configure Training](#14-configure-training)
  - [1.5 Run SFT Training](#15-run-sft-training)
- [Part 2: Two-Stage Reinforcement Learning](#part-2-two-stage-reinforcement-learning)
  - [2.1 Install verl Framework](#21-install-verl-framework)
  - [2.2 Prepare RL Dataset](#22-prepare-rl-dataset)
  - [2.3 Stage One: White-Box Reinforcement Learning](#23-stage-one-white-box-reinforcement-learning)
  - [2.4 Stage Two: Code Generation Reinforcement Learning](#24-stage-two-code-generation-reinforcement-learning)
- [Quick Start Guide](#quick-start-guide)
- [File Structure](#file-structure)

## Overview

The training pipeline consists of three stages:
1. **Supervised Fine-Tuning (SFT)**: SFT warm-up
2. **RL Stage One**: White-box reinforcement learning
3. **RL Stage Two**: Code generation reinforcement learning

---

## Part 1: Supervised Fine-Tuning (SFT)

### 1.1 Install LLaMA-Factory

Clone and install LLaMA-Factory:

```bash
cd train
git clone https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e ".[torch,metrics]"
```

For more details, visit the [LLaMA-Factory GitHub repository](https://github.com/hiyouga/LlamaFactory).

### 1.2 Prepare SFT Dataset

Ensure the SFT dataset is ready:
- **Source**: Generated from the data pipeline (see `../data/README.md`)
- **Expected file**: `sft_dataset.json`

### 1.3 Configure Dataset

Edit `LlamaFactory/data/dataset_info.json` to specify your dataset location:

```json
"sft_new_dataset": {
  "file_name": "/path/to/your/sft_dataset.json",
  "formatting": "sharegpt",
  "columns": { "messages": "conversations" }
}
```

### 1.4 Configure Training

Edit `LlamaFactory/examples/train_full/ExecVerify_sft.yaml`:

```yaml
model_name_or_path: /path/to/qwen2.5-coder-7b-instruct
output_dir: /path/to/output/sft-model
```

### 1.5 Run SFT Training

```bash
cd LlamaFactory
llamafactory-cli train examples/train_full/ExecVerify_sft.yaml
```

---

## Part 2: Two-Stage Reinforcement Learning

### 2.1 Install verl Framework

Clone and install verl (Volcano Engine Reinforcement Learning):

```bash
cd train
git clone https://github.com/volcengine/verl.git
cd verl
pip install -e .
```

For more details, visit the [verl GitHub repository](https://github.com/volcengine/verl).

### 2.2 Prepare RL Dataset

Ensure the RL dataset is ready:

- **Source**: Generated from the complete data pipeline (see `../data/README.md`)


### 2.3 Stage One: White-Box Reinforcement Learning

#### 2.3.1 Configure Dataset Path

Edit `verl/examples/data_preprocess/rl_dataset.py` to specify your RL dataset location:

```python
data_source = "/path/to/your/rl_dataset.json"
```

#### 2.3.2 Configure Model Path

Edit `verl/examples/grpo_trainer/stage_one_rl.sh`:

Set the model path to your SFT model from Part 1 (the model after SFT warm-up):

```bash
actor_rollout_ref.model.path=/path/to/your/sft-model
```

#### 2.3.3 Preprocess Data

```bash
cd verl
python examples/data_preprocess/rl_dataset.py
```

#### 2.3.4 Run Training

```bash
cd verl
bash examples/grpo_trainer/stage_one_rl.sh
```

### 2.4 Stage Two: Code Generation Reinforcement Learning

#### 2.4.1 Preprocess Data

```bash
cd verl
python examples/data_preprocess/code_gen.py
```

#### 2.4.2 Configure Model Path

Edit `verl/examples/grpo_trainer/stage_two_rl.sh`:

Update the model path to use the checkpoint from Stage One:

```bash
actor_rollout_ref.model.path=/path/to/stage-one-checkpoint
```

#### 2.4.3 Run Training

```bash
cd verl
bash examples/grpo_trainer/stage_two_rl.sh
```

---

## Quick Start Guide

For a complete training run:

```bash
# 1. SFT Training
cd train/LlamaFactory
llamafactory-cli train examples/train_full/ExecVerify_sft.yaml

# 2. RL Stage One
cd ../verl
python examples/data_preprocess/rl_dataset.py
bash examples/grpo_trainer/stage_one_rl.sh

# 3. RL Stage Two
python examples/data_preprocess/code_gen.py
bash examples/grpo_trainer/stage_two_rl.sh
```

---

## File Structure

```
train/
├── README.md                          # This file
├── LlamaFactory/                      # SFT training framework
│   ├── data/dataset_info.json        # Dataset configuration
│   └── examples/train_full/
│       └── ExecVerify_sft.yaml       # SFT training config
└── verl/                              # RL training framework
    ├── examples/
    │   ├── data_preprocess/
    │   │   ├── rl_dataset.py         # Stage one data preprocessing
    │   │   └── code_gen.py           # Stage two data preprocessing
    │   └── grpo_trainer/
    │       ├── stage_one_rl.sh       # Stage one training script
    │       └── stage_two_rl.sh       # Stage two training script
    └── verl/                          # Core verl library
```
