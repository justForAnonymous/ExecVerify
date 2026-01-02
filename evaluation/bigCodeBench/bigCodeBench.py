import os
import re
import json
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset

# ==========================================
# Configuration Constants
# ==========================================
RESP_LEN = 4096*3
PROMPT_LEN = 2048
BATCH_SIZE = 1024



def get_prompt_str(prompt_str):
    """
    Build the complete prompt template
    """
    return f"""
Your task is to complete the provided function.
{prompt_str}


Exception Handling Guard:
Do not add try/except blocks or custom raise statements unless the task explicitly tells you to.
Call the required library functions directly and let them raise their standard exceptions.
Only when the spec demands a particular error/message may you return or raise that exact text.
Before finalizing, confirm that no extra exception handling slipped in.


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

def extract_xml_answer(text: str) -> str:
    """
    Helper function: Extract answer from XML tags (kept for future extension)
    """
    if "<answer>" in text and "</answer>" in text:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    return text

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Run code generation tasks using vLLM")
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Local model path or HuggingFace model ID"
    )
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        help="Dataset name (e.g., bigcode/bigcodebench)"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="bcb_samples.jsonl", 
        help="Output JSONL file path (default: bcb_samples.jsonl)"
    )
    
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)"
    )

    return parser.parse_args()

def main():
    # 1. Parse arguments
    args = parse_args()
    
    print(f"Model Path: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Output File: {args.output}")

    # Note: It's recommended to set CUDA_VISIBLE_DEVICES in command line before running
    # Example: CUDA_VISIBLE_DEVICES=3 python run_generation.py ...

    # 2. Initialize Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    except Exception as e:
        print(f"Failed to load Tokenizer: {e}")
        return

    # 3. Initialize vLLM engine
    print("Initializing vLLM...")
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        max_model_len=RESP_LEN + PROMPT_LEN,
        gpu_memory_utilization=0.8,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    # 4. Set sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=RESP_LEN, 
        repetition_penalty=1.0
    )

    # 5. Load dataset
    print(f"Loading dataset: {args.dataset}")
    try:
        # Try to load all splits to determine the latest version
        all_splits = load_dataset(args.dataset, split=None)
        # Use the last sorted version as split name
        ver = sorted(all_splits.keys())[-1]
        print(f"Auto-selected dataset split version: {ver}")
        ds = load_dataset(args.dataset, split=ver)
    except Exception as e:
        print(f"Cannot auto-determine split version, trying default 'train' split: {e}")
        ds = load_dataset(args.dataset, split='train')

    # 6. Prepare Prompts
    formatted_prompts = []
    task_ids = []
    prompts = []
    
    print("Building Prompts...")
    for row in ds:
        task_id = row['task_id']
        prompt = row['complete_prompt']
        
        task_ids.append(task_id)
        
        # Apply Chat Template
        formatted_prompts.append(
            tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": f"{get_prompt_str(prompt)}"},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        prompts.append(get_prompt_str(prompt))

    # 7. Execute generation
    print(f"Starting generation for {len(formatted_prompts)} samples...")
    generated_outputs = llm.generate(
        formatted_prompts, sampling_params
    )

    # 8. Extract code and save
    samples = []
    # Regex: match markdown python code blocks
    pattern = r"```python\n(.*?)\n```"

    print("Processing generated results...")
    for prompt, task_id, generated_output in zip(prompts, task_ids, generated_outputs):
        # Get generated text
        full_output_text = generated_output.outputs[0].text
        
        # Extract code using regex
        matches = re.findall(pattern, full_output_text, re.DOTALL)
        code = ''
        if len(matches) > 0:
            # Take the last matched code block
            code = matches[-1]
        
        # For debugging, you can print cases where no code was extracted
        # if not code:
        #     print(f"Warning: No code block found for {task_id}")

        samples.append({
            "prompt":prompt,
            "task_id": task_id, 
            "full_output_text": full_output_text,
            "solution": code, 
            "_identifier": "raw_rl"
        })

    # 9. Write to file
    print(f"Writing results to {args.output}...")
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            
    print("Task completed.")

if __name__ == "__main__":
    main()
