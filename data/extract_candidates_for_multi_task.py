"""
Extract Candidates for Multi-Task - Step 8 of ExecVerify Data Pipeline

Extracts candidate samples for multi-task reinforcement learning by
generating predictions and validating them. Supports both IO and OI modes.

Input: filtered_all_dataset_with_difficulties.json
Output: candidates_io_for_multi_task_with_cot.json (IO mode)
        candidates_oi_for_multi_task_with_cot.json (OI mode)
"""

import sys
import re
import ast
import subprocess
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from utils import load_config, setup_logging, load_json, save_json, build_chat_prompt


def get_prompt_str(func_str: str, func_name: str, exec_result: str, args: str = "????") -> str:
    """Build prompt for candidate extraction."""
    return f"""
Try to execute the program step by step and fill in the missing assertion. 
Try to find out the ???? in the following code. 
Here is the provided code: 
```
{func_str}

assert {func_name}({args}) == {exec_result}

```

Output the complete test case in the following format:
```python
<test case content, including the function content and the assertion statement>
```
"""


def generate_prompt(tokenizer, func_name: str, func_str: str,
                    exec_result: str, args_str: str = "????") -> str:
    """Generate chat prompt."""
    return build_chat_prompt(
        tokenizer,
        "You are a programming expert",
        get_prompt_str(func_str, func_name, exec_result, args_str)
    )


def run_code(code_str: str, timeout: float = 0.5) -> bool:
    """Execute code to validate it."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code_str],
            capture_output=True,
            text=True,
            check=False,
            encoding="utf-8",
            timeout=timeout,
        )
        return result.returncode == 0
    except Exception:
        return False


def extract_code_from_output(output_text: str) -> str:
    """Extract code from model output."""
    pattern = r"```python\n(.*?)\n```"
    matches = re.findall(pattern, output_text, re.DOTALL)
    if len(matches) <= 0:
        return None
    return matches[-1]


def validate_io_output(output_code: str, gt_exec_output: str) -> bool:
    """Validate IO (input-output) prediction."""
    try:
        predict_output = output_code.split('==')[-1].strip()
        return ast.literal_eval(gt_exec_output) == ast.literal_eval(predict_output)
    except Exception:
        return False


def validate_oi_output(output_code: str) -> bool:
    """Validate OI (output-input) prediction by execution."""
    return run_code(output_code)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Extract candidates for multi-task RL"
    )
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['io', 'oi'],
        help="Extraction mode: 'io' for input-output, 'oi' for output-input"
    )
    parser.add_argument(
        '--start-idx',
        type=int,
        default=40000,
        help="Starting index in the dataset (default: 40000)"
    )
    parser.add_argument(
        '--end-idx',
        type=int,
        default=80000,
        help="Ending index in the dataset (default: 80000)"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    logger = setup_logging(config)
    
    logger.info(f"Starting {args.mode.upper()} candidate extraction...")
    logger.info(f"Processing samples from index {args.start_idx} to {args.end_idx}")
    
    # Load dataset
    input_file = config['output_files']['filtered_all_with_difficulties']
    filtered_all_dataset = load_json(input_file, logger)
    
    # Filter subset
    filtered_subset = filtered_all_dataset[args.start_idx:args.end_idx]
    
    # For OI mode, filter by result length
    if args.mode == 'oi':
        max_result_length = config['dataset_limits']['oi_result_max_length']
        logger.info(f"Filtering OI samples with result length <= {max_result_length}")
        filtered_subset = [
            s for s in filtered_subset
            if len(s['result']) <= max_result_length
        ]
    
    logger.info(f"Processing {len(filtered_subset)} samples")
    
    # Load model
    model_path = config['models']['reasoning_model']
    logger.info(f"Loading model from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    gen_config = config['generation']['cot_extraction']
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=gen_config['max_model_len'],
        gpu_memory_utilization=gen_config['gpu_memory_utilization'],
        tensor_parallel_size=gen_config['tensor_parallel_size'],
    )
    
    # Generate prompts
    logger.info("Generating prompts...")
    chat_prompts = []
    
    for sample in filtered_subset:
        if args.mode == 'io':
            prompt = generate_prompt(
                tokenizer,
                sample['func_name'],
                sample['func_str'],
                "????",
                sample['func_args']
            )
        else:  # oi mode
            prompt = generate_prompt(
                tokenizer,
                sample['func_name'],
                sample['func_str'],
                sample["result"],
                "????"
            )
        chat_prompts.append(prompt)
    
    # Generate outputs
    logger.info("Generating predictions...")
    sampling_params = SamplingParams(
        temperature=gen_config['temperature'],
        top_p=gen_config['top_p'],
        max_tokens=gen_config['max_tokens'],
        n=1,
        repetition_penalty=gen_config['repetition_penalty'],
    )
    
    generated_outputs = llm.generate(chat_prompts, sampling_params)
    
    # Extract and validate
    logger.info("Validating outputs...")
    candidates_with_cot = []
    
    for sample, sample_outputs in zip(filtered_subset, generated_outputs):
        for output in sample_outputs.outputs:
            output_text = output.text
            output_code = extract_code_from_output(output_text)
            
            if output_code is None:
                continue
            
            # Validate based on mode
            is_valid = False
            if args.mode == 'io':
                is_valid = validate_io_output(output_code, sample['result'])
                cot_key = 'io_cot'
            else:
                is_valid = validate_oi_output(output_code)
                cot_key = 'oi_cot'
            
            if is_valid:
                sample[cot_key] = output_text
                candidates_with_cot.append(sample)
                break
    
    logger.info(f"Successfully extracted {len(candidates_with_cot)} candidates")
    
    # Save results
    if args.mode == 'io':
        output_file = config['output_files']['candidates_io_multi_task']
    else:
        output_file = config['output_files']['candidates_oi_multi_task']
    
    save_json(candidates_with_cot, output_file, logger)
    logger.info(f"{args.mode.upper()} candidate extraction complete!")


if __name__ == "__main__":
    main()

