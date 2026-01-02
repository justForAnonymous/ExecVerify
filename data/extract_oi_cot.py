"""
Extract OI Chain-of-Thought - Step 6b of ExecVerify Data Pipeline

Extracts output-input prediction chain-of-thought reasoning by having
a model predict inputs given outputs and function code.

Input: filtered_all_dataset_with_difficulties.json
Output: filtered_all_dataset_with_difficulties_subset_with_oi_cot.json,
        oi_sft_dataset.json
"""

import re
import sys
import subprocess
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from utils import load_config, setup_logging, load_json, save_json, build_chat_prompt


def get_oi_prompt(func_str: str, func_name: str, exec_result: str, args: str = "????") -> str:
    """
    Build prompt for output-input prediction.
    
    Args:
        func_str: Function definition code
        func_name: Function name
        exec_result: Actual execution result
        args: Arguments placeholder
        
    Returns:
        Formatted prompt string
    """
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
    """Generate chat prompt for OI prediction."""
    return build_chat_prompt(
        tokenizer,
        "You are a programming expert",
        get_oi_prompt(func_str, func_name, exec_result, args_str)
    )


def run_code(code_str: str, timeout: float = 0.5) -> bool:
    """
    Execute Python code to validate it.
    
    Args:
        code_str: Python code to execute
        timeout: Maximum execution time
        
    Returns:
        True if code executes successfully, False otherwise
    """
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


def format_for_sft(sample: dict, cot_key: str, prompt_func) -> dict:
    """
    Format sample for supervised fine-tuning.
    
    Args:
        sample: Sample with chain-of-thought
        cot_key: Key containing chain-of-thought ('io_cot' or 'oi_cot')
        prompt_func: Function to generate the question prompt
        
    Returns:
        Formatted conversation dictionary
    """
    question = prompt_func(
        sample['func_str'],
        sample['func_name'],
        sample['result'],
        "????"
    )
    
    cot_text = sample[cot_key]
    
    # Split by </think> tag if present
    if '</think>' in cot_text:
        reasoning = cot_text.split('</think>')[0].strip()
        answer = cot_text.split('</think>')[-1].strip()
    else:
        reasoning = ""
        answer = cot_text.strip()
    
    formatted_answer = f"<reasoning>\n{reasoning}\n</reasoning>\n<answer>\n{answer}\n</answer>"
    
    return {
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": formatted_answer},
        ]
    }


def main():
    """Main execution function."""
    # Load configuration
    config = load_config()
    logger = setup_logging(config)
    
    logger.info("Starting OI chain-of-thought extraction...")
    
    # Load dataset
    input_file = config['output_files']['filtered_all_with_difficulties']
    filtered_all_dataset = load_json(input_file, logger)
    
    # Take subset and filter by result length
    subset_size = config['dataset_limits']['oi_cot_subset']
    max_result_length = config['dataset_limits']['oi_result_max_length']
    
    logger.info(f"Filtering samples with result length <= {max_result_length}")
    filtered_subset = []
    for sample in filtered_all_dataset[:subset_size]:
        if len(sample['result']) <= max_result_length:
            filtered_subset.append(sample)
    
    logger.info(f"Using {len(filtered_subset)} samples after filtering")
    
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
    logger.info("Generating OI prediction prompts...")
    chat_prompts = []
    for sample in filtered_subset:
        prompt = generate_prompt(
            tokenizer,
            sample['func_name'],
            sample['func_str'],
            sample["result"],
            "????"
        )
        chat_prompts.append(prompt)
    
    logger.info(f"Generated {len(chat_prompts)} prompts")
    
    # Generate outputs
    logger.info("Generating chain-of-thought reasoning...")
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
    dataset_with_oi_cot = []
    
    for sample, sample_outputs in zip(filtered_subset, generated_outputs):
        for output in sample_outputs.outputs:
            output_text = output.text
            output_code = extract_code_from_output(output_text)
            
            if output_code is None:
                continue
            
            if run_code(output_code):
                sample['oi_cot'] = output_text
                dataset_with_oi_cot.append(sample)
                break
    
    logger.info(f"Successfully extracted {len(dataset_with_oi_cot)} samples with OI CoT")
    
    # Save dataset with CoT
    output_file = config['output_files']['oi_cot_dataset']
    save_json(dataset_with_oi_cot, output_file, logger)
    
    # Create SFT dataset
    logger.info("Creating SFT dataset...")
    sft_size = config['dataset_limits']['sft_dataset_size']
    oi_sft_dataset = []
    
    for sample in dataset_with_oi_cot[:sft_size]:
        sft_sample = format_for_sft(sample, 'oi_cot', get_oi_prompt)
        oi_sft_dataset.append(sft_sample)
    
    # Save SFT dataset
    sft_output_file = config['output_files']['oi_sft_dataset']
    save_json(oi_sft_dataset, sft_output_file, logger)
    
    logger.info("OI chain-of-thought extraction complete!")


if __name__ == "__main__":
    main()
