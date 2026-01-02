"""
Input Synthesis - Step 2 of ExecVerify Data Pipeline

Mutates the input arguments of raw dataset code samples by generating
diverse test inputs (integers, strings, collections) to increase test coverage.

Input: raw_dataset.json
Output: mutated_raw_dataset.json
"""

import json
import random
import string
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from utils import load_config, setup_logging, load_json, save_json, build_chat_prompt


def generate_random_int(min_val: int = 5, max_val: int = 20) -> int:
    """Generate random integer within specified range."""
    return random.randint(min_val, max_val)


def generate_random_int_small(min_val: int = 5, max_val: int = 15) -> int:
    """Generate smaller random integer."""
    return random.randint(min_val, max_val)


def generate_random_character() -> str:
    """Generate random character from all printable characters."""
    all_chars = string.ascii_letters + string.digits + string.punctuation + string.whitespace
    return random.choice(all_chars)


def generate_random_string(max_length: int) -> str:
    """Generate random alphanumeric string."""
    length = random.randint(5, max_length)
    characters = string.ascii_letters + string.digits
    return "".join(random.choice(characters) for _ in range(length))


def generate_random_string_with_punctuation(max_length: int) -> str:
    """Generate random string including punctuation."""
    length = random.randint(5, max_length)
    characters = string.ascii_letters + string.digits + string.punctuation
    return "".join(random.choice(characters) for _ in range(length))


def get_ref_values(config: dict) -> str:
    """
    Generate reference values for input mutation.
    
    Creates a collection of random integers and strings that can be used
    to replace arguments in the function calls.
    
    Args:
        config: Configuration dictionary with generation parameters
        
    Returns:
        Formatted string containing reference values
    """
    gen_config = config['code_generation']['reference_values']
    length = generate_random_int_small(
        gen_config['collection_length_min'],
        gen_config['collection_length_max']
    )
    
    int_strs = ''
    str_strs = ''
    
    # Generate integer values
    for _ in range(length):
        int_strs = int_strs + '  ' + str(generate_random_int_small(
            gen_config['int_range_min'],
            gen_config['int_range_max']
        ))
    
    # Generate string values
    for _ in range(length):
        gen_str = ''
        rand_choice = random.randint(1, 10)
        
        if rand_choice <= 5:
            gen_str = generate_random_string(10)
        elif rand_choice <= 9:
            char_length = generate_random_int_small(
                gen_config['collection_length_min'],
                gen_config['collection_length_max']
            )
            gen_str = generate_random_character() * char_length
        else:
            gen_str = generate_random_string_with_punctuation(10)
        
        str_strs = str_strs + '  ' + f"'{gen_str}'"
    
    return f"integer values:{int_strs}\n\nstring values:{str_strs}\n\n"


def get_prompt(code_str: str, ref_values: str) -> str:
    """
    Build prompt for input mutation.
    
    Args:
        code_str: Original code to mutate
        ref_values: Reference values to use in mutation
        
    Returns:
        Formatted prompt string
    """
    parts = []
    
    parts.append(f"""
Task
----
Modify the arguments in the provided code.
""")
    
    parts.append(f"""
Here is the provided code:
-----------------
{code_str}

Reconstruct the arguments in the function call by directly writing the arguments inside the function call 
(i.e., you must not assign them to variables before calling the function) and 
try to use one or more values below:
{ref_values}

Notes

For types such as str, dict, list, or set, their length must exceed five.

You may modify the code if needed, as long as it aligns with the reference values.

In the print statement, call the entry function only at one time. Do not use nested function calls in the print statement.

Output Format
-------------

call the entry point function with the new arguments and use the print statement to print the result.

Return only a single Markdown code block:
```python
# complete code snippet
```
""".strip())
    
    return "\n\n".join(parts)


def extract_think_and_code(text: str):
    """
    Extract code content from model output.
    
    Args:
        text: Model output text
        
    Returns:
        Tuple of (think_content, code_content)
    """
    think_match = re.search(r"</think>\s*(.*?)(?=```|$)", text, re.DOTALL)
    think_content = think_match.group(1).strip() if think_match else None
    
    code_match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    code_content = code_match.group(1).strip() if code_match else None
    
    return think_content, code_content


def main():
    """Main execution function."""
    # Load configuration
    config = load_config()
    logger = setup_logging(config)
    
    logger.info("Starting input synthesis...")
    
    # Load raw dataset
    raw_dataset_file = config['output_files']['raw_dataset']
    raw_dataset = load_json(raw_dataset_file, logger)
    
    # Load model
    model_path = config['models']['code_generator']
    logger.info(f"Loading model from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    gen_config = config['generation']['input_synthesis']
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=gen_config['max_model_len'],
        gpu_memory_utilization=gen_config['gpu_memory_utilization'],
        tensor_parallel_size=gen_config['tensor_parallel_size'],
    )
    
    # Generate prompts
    logger.info("Generating mutation prompts...")
    prompts = []
    settings = []
    
    for sample in raw_dataset:
        code = sample['code']
        setting = sample['setting']
        
        prompt = get_prompt(code, get_ref_values(config))
        prompts.append(prompt)
        settings.append(setting)
    
    logger.info(f"Generated {len(prompts)} prompts")
    
    # Build chat prompts
    logger.info("Building chat prompts...")
    chat_prompts = []
    for prompt in prompts:
        chat_prompt = build_chat_prompt(
            tokenizer,
            "You are a code generator.",
            prompt
        )
        chat_prompts.append(chat_prompt)
    
    # Generate mutated code
    logger.info("Generating mutated code samples...")
    sampling_params = SamplingParams(
        temperature=gen_config['temperature'],
        top_p=gen_config['top_p'],
        max_tokens=gen_config['max_tokens'],
        repetition_penalty=gen_config['repetition_penalty'],
    )
    
    outputs = llm.generate(chat_prompts, sampling_params)
    
    # Extract mutated code
    logger.info("Extracting mutated code from outputs...")
    mutated_raw_dataset = []
    for output, setting in zip(outputs, settings):
        output_text = output.outputs[0].text
        _, code = extract_think_and_code(output_text)
        if code is None:
            continue
        
        mutated_raw_dataset.append({
            'code': code,
            'setting': setting
        })
    
    # Save dataset
    output_file = config['output_files']['mutated_raw_dataset']
    save_json(mutated_raw_dataset, output_file, logger)
    logger.info(f"Input synthesis complete! Generated {len(mutated_raw_dataset)} mutated samples")


if __name__ == "__main__":
    main()
