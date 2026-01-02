# ExecVerify Evaluation

This directory contains evaluation scripts and pipelines for testing **ExecVerify** models across various code execution and reasoning benchmarks.

## Table of Contents

- [CRUXEval & LiveCodeBench-Exec](#cruxeval--livecodebench-exec)
- [REval](#reval)
- [EvalPlus](#evalplus)
- [LiveCodeBench](#livecodebench)
- [BigCodeBench](#bigcodebench)
- [CRUXEval-X (Multi-lingual)](#cruxeval-x-multi-lingual)

> [!IMPORTANT]
> **Global Configuration Note:**
> Before running any scripts, please ensure you replace `/path/to/ExecVerify` in the commands or config files with the **actual absolute path** to your model checkpoint.

---

## CRUXEval & LiveCodeBench-Exec

Lightweight evaluation for input prediction and execution reasoning.

### 1. Install Dependencies

```bash
cd evaluation/crux_lcb
pip install vllm transformers datasets torch tree-sitter tree-sitter-python
```

### 2. Run Evaluation

**CRUXEval**
```bash
python eval.py /path/to/ExecVerify cruxeval-org/cruxeval
```

**LiveCodeBench-Exec**
```bash
python eval.py /path/to/ExecVerify livecodebench/execution-v2
```

---

## REval

**Official Repository:** [r-eval/REval](https://github.com/r-eval/REval)

### 1. Setup

Follow the installation instructions in the official repository, then install Python requirements:

```bash
cd evaluation/REval
pip install -r requirements.txt
```

### 2. Configure Model

Modify the `.eval_config` file to point to your model path:

```json
{
    "prompt_type": "direct",
    "model_id": "ExecVerify",
    "model_path": "/path/to/ExecVerify",
    "num_gpus": 1,
    "gpu_ordinals": ["0"],
    "temp": 0.0
}
```

### 3. Run Evaluation

```bash
python evaluation_exec_verify.py
```

---

## EvalPlus

**Official Repository:** [evalplus/evalplus](https://github.com/evalplus/evalplus)

### 1. Setup

Follow the installation instructions in the official repository, then navigate to the directory:

```bash
cd evaluation/evalplus
```

### 2. Run Pipeline

Use the automatic pipeline script. You can specify the tensor parallel size (number of GPUs) as the second argument.

**Syntax:** `bash run_eval.sh <MODEL_PATH> <TP_SIZE>`

**Example (Single GPU):**
```bash
bash run_eval.sh /path/to/ExecVerify 1
```

**Example (4 GPUs with Tensor Parallelism):**
```bash
bash run_eval.sh /path/to/ExecVerify 4
```

---

## LiveCodeBench

**Official Repository:** [livecodebench/livecodebench](https://github.com/livecodebench/livecodebench)

### 1. Setup

Follow the official installation instructions:

```bash
cd evaluation/LiveCodeBench
```

### 2. Configure Model Path

> [!WARNING]
> You must manually set the model path in the source code before running.

Edit `lcb_runner/prompts/code_generation.py` (Line 405):

```python
model_path = "/path/to/ExecVerify"  # Set your model path here
```

### 3. Run Evaluation

```bash
python -m lcb_runner.runner.main \
  --model ExecVerify \
  --local_model_path /path/to/ExecVerify \
  --trust_remote_code \
  --scenario codegeneration \
  --release_version release_v6 \
  --n 1 \
  --temperature 0.0 \
  --max_tokens 16384 \
  --tensor_parallel_size 1 \
  --dtype bfloat16 \
  --start_date 2024-08-01 \
  --end_date 2025-05-01 \
  --evaluate
```

---

## BigCodeBench

**Official Repository:** [bigcode-project/bigcodebench](https://github.com/bigcode-project/bigcodebench)

### 1. Setup

Follow the official installation instructions:

```bash
cd evaluation/bigCodeBench
```

### 2. Run Pipeline

**Syntax:** `bash run_bcb_eval.sh <MODEL_PATH> <TP_SIZE>`

**Example (Single GPU):**
```bash
bash run_bcb_eval.sh /path/to/ExecVerify 1
```

**Example (Multiple GPUs):**
```bash
bash run_bcb_eval.sh /path/to/ExecVerify 4
```

---

## CRUXEval-X (Multi-lingual)

**Official Repository:** [CRUXEVAL-X/cruxeval-x](https://github.com/CRUXEVAL-X/cruxeval-x)

Evaluates on 6 languages: **Java, C++, C#, Go, JavaScript, and PHP**.

### 1. Setup

Follow the official installation instructions:

```bash
cd evaluation/cruxeval-x
```

### 2. Configuration

Edit the execution script `script/infer_execverify.bash` and update the `--model_name` argument:

```bash
--model_name /path/to/ExecVerify
```

### 3. Run Evaluation

```bash
bash script/infer_execverify.bash
```