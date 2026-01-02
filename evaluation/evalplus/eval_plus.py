import os
import re
import argparse
from typing import List, Dict

# Note: It's recommended to set CUDA_VISIBLE_DEVICES in the command line before running the script
# Example: CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_plus.py ...

from evalplus.data import get_human_eval_plus, write_jsonl, get_mbpp_plus
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# def get_prompt_str(prompt_str):
#     return f"""
# Your task is to complete the provided function.
# {prompt_str}

# Verify your generated function using the provided inputs.
# Output format:
# <reasoning>
# Your step-by-step reasoning.
# </reasoning>
# <answer>
# ```python
# The generated function.
# ```
# </answer>
# """

# def get_prompt_str(prompt_str):
#     return f"""
# Your task is to complete the provided function.
# {prompt_str}

# Verify your generated function using the provided inputs.
# Output format:
# <reasoning>
# Your step-by-step reasoning.
# </reasoning>
# <answer>
# ```python
# The generated function.
# ```
# </answer>
# """

def get_prompt_str(prompt_str):
    return f"""
Your task is to complete the provided function.
{prompt_str}

Verify your generated function using the provided inputs.
Output format:
<reasoning>
Your step-by-step reasoning.
</reasoning>
<answer>
```python
The generated function.
```
</answer>
"""


def extract_code_from_answer(text: str) -> str:
    """
    Extract code from model output.
    First try to match content within <answer> tags, then remove any markdown markers.
    """
    # 1. Try to extract content within <answer> tags
    if "<answer>" in text and "</answer>" in text:
        text = text.split("<answer>")[-1].split("</answer>")[0]
    
    # 2. Clean up markdown code block markers and python identifiers
    # Match ```python code ``` or python code
    pattern = r"(?:```python\n|python\n)(.*?)(?:```|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        return matches[-1].strip()
    
    # If no explicit python marker found, return cleaned text directly
    return text.strip()

def main():
    parser = argparse.ArgumentParser(description="Generate samples for HumanEval or MBPP using vLLM.")
    
    # Required arguments
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--dataset", type=str, required=True, choices=["human_eval", "mbpp"], help="Dataset to use")
    
    # Optional arguments
    parser.add_argument("--output", type=str, default=None, help="Output jsonl file path. Defaults to he_samples.jsonl or mbpp_samples.jsonl")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size (number of GPUs)")
    
    args = parser.parse_args()

    # 1. Set dataset and output file
    if args.dataset == "human_eval":
        problems = get_human_eval_plus()
        default_output_file = "he_samples.jsonl"
    else:
        problems = get_mbpp_plus()
        default_output_file = "mbpp_samples.jsonl"

    output_file = args.output if args.output else default_output_file
    print(f"Dataset: {args.dataset}")
    print(f"Model Path: {args.model}")
    print(f"Output File: {output_file}")
    print(f"Tensor Parallel Size: {args.tp}")

    # 2. Initialize Tokenizer and vLLM
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    RESP_LEN = 4096*3
    PROMPT_LEN = 2048  # Slightly increase prompt length for safety
    
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        max_model_len=RESP_LEN + PROMPT_LEN,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=args.tp,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=RESP_LEN,
        repetition_penalty=1.0,

    )

    # 3. Prepare Prompts
    samples = []
    formatted_prompts = []
    task_ids = []

    print("Building prompts...")
    for task_id, prob in problems.items():
        prompt = prob["prompt"]
        task_ids.append(task_id)
        
        # Build chat format
        chat_message = [
            {"role": "user", "content": get_prompt_str(prompt)},
        ]
        
        formatted_prompts.append(
            tokenizer.apply_chat_template(
                chat_message,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    # 4. Batch generation
    print(f"Generating {len(formatted_prompts)} samples...")
    generated_outputs = llm.generate(formatted_prompts, sampling_params)

    # 5. Extract and save results
    for task_id, generated_output in zip(task_ids, generated_outputs):
        raw_output = generated_output.outputs[0].text
        
        # Use new extraction function
        code = extract_code_from_answer(raw_output)
        code = 'from typing import Any, List, Tuple\n' + code
        
        # Simple debug print, if extraction is empty then print raw output
        if not code.strip():
            print(f"Warning: Empty code extracted for {task_id}")
            # print(raw_output) 

        samples.append({"task_id": task_id, "solution": code,"raw_output":raw_output})

    write_jsonl(output_file, samples)
    print(f"Successfully saved to {output_file}")

if __name__ == "__main__":
    main()