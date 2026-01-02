"""
Code Synthesis - Step 1 of ExecVerify Data Pipeline

Generates raw dataset by creating Python code samples that test various
built-in methods with different complexity levels (nested calls, control flow, etc.).

Output: raw_dataset.json
"""

import os
import json
import random
import string
import textwrap
import re
from typing import List
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from utils import load_config, setup_logging, save_json, build_chat_prompt


def get_control_stmts(max_depth: int = 3) -> str:
    """
    Randomly describe a control-flow blueprint in natural language.
    
    Args:
        max_depth: Maximum depth of nested control structures
        
    Returns:
        Natural language description of control flow requirements
    """
    pool = ["if", "for", "while", "if-elif"]
    depth = max_depth

    if depth == 0:
        return "Do not include any control structures (if, for, while, elif)."

    seq = [random.choice(pool) for _ in range(depth)]

    label = {
        "if": "if statement",
        "for": "for statement",
        "while": "while statement",
        "if-elif": "if-elif chain",
    }

    def article(phrase: str) -> str:
        return "an" if phrase[0].lower() in "aeiou" else "a"

    if depth == 1:
        p = label[seq[0]]
        return f"Include {article(p)} {p}."

    parts: List[str] = []
    first = label[seq[0]]
    parts.append(f"Include {article(first)} {first}")

    for i in range(1, depth):
        current = label[seq[i]]
        parent = label[seq[i - 1]]
        if i == 1:
            parts.append(f"nest {article(current)} {current} inside it")
        else:
            parts.append(f"then nest {article(current)} {current} inside the {parent}")

    return ", ".join(parts) + "."


def use_nested_calls(flag: bool) -> str:
    """Generate requirement string for nested function calls."""
    return (
        "Use nested calls to the test method with at least two levels of depth and "
        "invoke the test method multiple times in the generated code."
        if flag else
        "Avoid nested function calls."
    )


def use_other_methods(flag: bool, prefer_types: List[str] = None) -> str:
    """Generate requirement string for using additional built-in methods."""
    prefer_types = prefer_types or ["str", "list", "set", "tuple"]
    if flag:
        return (
            "Use at least one additional built-in method (preferably from "
            + ", ".join(prefer_types)
            + ") besides the method under test."
        )
    return "Do not call any additional methods beyond the method under test."


def use_call_chains(flag: bool) -> str:
    """Generate requirement string for function call chains."""
    return (
        "Define exactly two functions: helper A and entry-point B; B must call A at least once."
        if flag else
        "Define exactly one function; do not create helper functions."
    )


def get_prompt(
    type_str: str,
    method_str: str,
    use_nested_calls_flag: bool,
    use_other_methods_flag: bool,
    use_call_chains_flag: bool,
    max_depth: int
) -> str:
    """
    Build prompt for code generation with specific constraints.
    
    Args:
        type_str: Python type to test (e.g., 'dict', 'list')
        method_str: Method to test (e.g., 'append', 'pop')
        use_nested_calls_flag: Whether to use nested function calls
        use_other_methods_flag: Whether to use additional methods
        use_call_chains_flag: Whether to use helper functions
        max_depth: Maximum depth of control flow structures
        
    Returns:
        Formatted prompt string
    """
    parts = []

    parts.append(textwrap.dedent(f"""
        Task
        ----
        Write Python code that tests the `{method_str}` method of the `{type_str}` type.
    """).strip())

    parts.append(textwrap.dedent(f"""
        Hard Requirements
        -----------------
        1) Do not include any comments.
        2) Use exactly one print(...) statement to display the result.
        3) Place that print(...) statement outside the function.
        4) Call the function exactly once — inside that single print(...) call.
        5) Do not use default parameter values in the function definition.
        6) Each function must have at least one argument.
    """).strip())

    parts.append("Guidelines\n----------")
    parts.append(f"- {use_nested_calls(use_nested_calls_flag)}")
    parts.append(f"- {use_other_methods(use_other_methods_flag)}")
    parts.append(f"- {use_call_chains(use_call_chains_flag)}")
    parts.append(f"- {get_control_stmts(max_depth)}")

    parts.append(textwrap.dedent("""
        Output Format
        -------------
        Return only a single Markdown code block:
        ```python
        # complete code snippet
        ```
    """).strip())

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


def generate_prompts_for_configuration(
    methods_dir: str,
    use_nested_calls_flag: bool,
    use_other_methods_flag: bool,
    use_call_chains_flag: bool,
    max_depth: int,
    max_limit: int
):
    """
    Generate prompts for a specific configuration setting.
    
    Returns:
        Tuple of (prompts_list, settings_list)
    """
    prompts = []
    settings = []
    
    while len(prompts) < max_limit:
        for file_name in os.listdir(methods_dir):
            type_name = file_name.split('.')[0]
            file_path = os.path.join(methods_dir, file_name)
            
            with open(file_path) as f:
                file_content = f.read()
            
            for method in file_content.split('\n'):
                method = method.strip()
                if not method:
                    continue
                    
                prompt = get_prompt(
                    type_name, method,
                    use_nested_calls_flag,
                    use_other_methods_flag,
                    use_call_chains_flag,
                    max_depth
                )
                
                settings.append({
                    'use_nested_calls_flag': use_nested_calls_flag,
                    'use_other_methods': use_other_methods_flag,
                    'use_call_chains_flag': use_call_chains_flag,
                    'max_depth': max_depth,
                    'type': type_name,
                    'method': method
                })
                
                prompts.append(prompt)
                
                if len(prompts) >= max_limit:
                    return prompts, settings
    
    return prompts, settings


def main():
    """Main execution function."""
    # Load configuration
    config = load_config()
    logger = setup_logging(config)
    
    logger.info("Starting code synthesis...")
    
    # Load model
    model_path = config['models']['code_generator']
    logger.info(f"Loading model from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    gen_config = config['generation']['code_synthesis']
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=gen_config['max_model_len'],
        gpu_memory_utilization=gen_config['gpu_memory_utilization'],
        tensor_parallel_size=gen_config['tensor_parallel_size'],
    )
    
    # Generate prompts for different configurations
    methods_dir = os.path.join(
        config['paths']['methods_dir']
    )
    max_limit = config['dataset_limits']['raw_dataset_per_config']
    
    logger.info("Generating prompts for 6 different configurations...")
    
    all_prompts = []
    all_settings = []
    
    # Configuration 1: Nested calls, no other methods, no call chains, no control flow
    logger.info("Config 1/6: Nested calls only")
    prompts, settings = generate_prompts_for_configuration(
        methods_dir, True, False, False, 0, max_limit
    )
    all_prompts.extend(prompts)
    all_settings.extend(settings)
    
    # Configuration 2: Nested calls + other methods
    logger.info("Config 2/6: Nested calls + other methods")
    prompts, settings = generate_prompts_for_configuration(
        methods_dir, True, True, False, 0, max_limit
    )
    all_prompts.extend(prompts)
    all_settings.extend(settings)
    
    # Configuration 3: Nested calls + other methods + control flow (depth 1)
    logger.info("Config 3/6: + control flow depth 1")
    prompts, settings = generate_prompts_for_configuration(
        methods_dir, True, True, False, 1, max_limit
    )
    all_prompts.extend(prompts)
    all_settings.extend(settings)
    
    # Configuration 4: Nested calls + other methods + control flow (depth 2)
    logger.info("Config 4/6: + control flow depth 2")
    prompts, settings = generate_prompts_for_configuration(
        methods_dir, True, True, False, 2, max_limit
    )
    all_prompts.extend(prompts)
    all_settings.extend(settings)
    
    # Configuration 5: Nested calls + other methods + control flow (depth 3)
    logger.info("Config 5/6: + control flow depth 3")
    prompts, settings = generate_prompts_for_configuration(
        methods_dir, True, True, False, 3, max_limit
    )
    all_prompts.extend(prompts)
    all_settings.extend(settings)
    
    # Configuration 6: All features including call chains
    logger.info("Config 6/6: All features + call chains")
    prompts, settings = generate_prompts_for_configuration(
        methods_dir, True, True, True, 3, max_limit
    )
    all_prompts.extend(prompts)
    all_settings.extend(settings)
    
    logger.info(f"Generated {len(all_prompts)} prompts total")
    
    # Build chat prompts
    logger.info("Building chat prompts...")
    chat_prompts = []
    for prompt in all_prompts:
        chat_prompt = build_chat_prompt(
            tokenizer,
            "You are a code generator.",
            prompt
        )
        chat_prompts.append(chat_prompt)
    
    # Generate code
    logger.info("Generating code samples...")
    sampling_params = SamplingParams(
        temperature=gen_config['temperature'],
        top_p=gen_config['top_p'],
        max_tokens=gen_config['max_tokens'],
        repetition_penalty=gen_config['repetition_penalty'],
    )
    
    outputs = llm.generate(chat_prompts, sampling_params)
    
    # Extract code from outputs
    logger.info("Extracting code from outputs...")
    raw_dataset = []
    for output, setting in zip(outputs, all_settings):
        output_text = output.outputs[0].text
        _, code = extract_think_and_code(output_text)
        if code is None:
            continue
        
        raw_dataset.append({
            'code': code,
            'setting': setting
        })
    
    # Save dataset
    output_file = config['output_files']['raw_dataset']
    save_json(raw_dataset, output_file, logger)
    logger.info(f"Code synthesis complete! Generated {len(raw_dataset)} code samples")


if __name__ == "__main__":
    main()
