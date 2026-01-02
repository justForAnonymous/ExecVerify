"""
Filter by Difficulty - Step 4 of ExecVerify Data Pipeline

Evaluates difficulty of code samples by testing how many times a model
can correctly predict the output. Samples with pass_cnt > threshold are
considered too easy.

Input: processed_raw_dataset.json, processed_mutated_dataset.json
Output: processed_raw_dataset_with_difficulties.json, 
        processed_mutated_dataset_with_difficulties.json
"""

import sys
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from utils import load_config, setup_logging, load_json, save_json, build_chat_prompt


def get_prompt_str(func_str: str, func_name: str, exec_result: str, args: str = "????") -> str:
    """
    Build prompt for difficulty evaluation.
    
    Args:
        func_str: Function definition code
        func_name: Name of the function
        exec_result: Expected execution result
        args: Function arguments (or ???? as placeholder)
        
    Returns:
        Formatted prompt string
    """
    return f"""Fill in the missing assertion. Try to find out the ???? in the following code. 
Here is the provided code: 
```
{func_str}

assert {func_name}({args}) == {exec_result}

```

Output the complete test case in the following format:
```python
<test case content, including the function content and the assertion statement>
```

Format your response strictly as follows:
<reasoning>
your step-by-step reasoning here
</reasoning>
<answer>
You answer to the question
</answer>
"""


def generate_r1_prompt(tokenizer, func_name: str, func_str: str, 
                       exec_result: str, args_str: str = "????") -> str:
    """
    Generate chat prompt for difficulty evaluation.
    
    Args:
        tokenizer: HuggingFace tokenizer
        func_name: Function name
        func_str: Function code
        exec_result: Expected result
        args_str: Function arguments
        
    Returns:
        Formatted chat prompt
    """
    return build_chat_prompt(
        tokenizer,
        "You are a programming expert",
        get_prompt_str(func_str, func_name, exec_result, args_str)
    )


def evaluate_difficulty(dataset: list, config: dict, logger) -> list:
    """
    Evaluate difficulty of dataset samples.
    
    Args:
        dataset: List of processed samples
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Dataset with difficulty scores (pass_cnt) added
    """
    # Load model
    model_path = config['models']['difficulty_evaluator']
    logger.info(f"Loading difficulty evaluation model from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    gen_config = config['generation']['difficulty_evaluation']
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=gen_config['max_model_len'],
        gpu_memory_utilization=gen_config['gpu_memory_utilization'],
        tensor_parallel_size=gen_config['tensor_parallel_size'],
    )
    
    # Generate prompts
    logger.info("Generating evaluation prompts...")
    formatted_prompts = []
    
    for sample in dataset:
        prompt = generate_r1_prompt(
            tokenizer,
            sample['func_name'],
            sample['func_str'],
            "????",
            sample['func_args']
        )
        formatted_prompts.append(prompt)
    
    # Generate outputs
    logger.info(f"Evaluating difficulty for {len(dataset)} samples...")
    sampling_params = SamplingParams(
        temperature=gen_config['temperature'],
        top_p=gen_config['top_p'],
        max_tokens=gen_config['max_tokens'],
        n=gen_config['n_samples'],
    )
    
    generated_outputs = llm.generate(formatted_prompts, sampling_params)
    
    # Count pass rate
    logger.info("Calculating pass rates...")
    dataset_with_difficulties = []
    
    for sample, sample_outputs in zip(dataset, generated_outputs):
        gt_exec_output = sample['result']
        pass_cnt = 0
        
        for output in sample_outputs.outputs:
            if gt_exec_output in output.text:
                pass_cnt += 1
        
        sample['pass_cnt'] = pass_cnt
        dataset_with_difficulties.append(sample)
    
    return dataset_with_difficulties


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Evaluate difficulty of processed datasets"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['raw', 'mutated'],
        help="Which dataset to process: 'raw' or 'mutated'"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    logger = setup_logging(config)
    
    logger.info(f"Starting difficulty evaluation for {args.dataset} dataset...")
    
    # Load appropriate dataset
    if args.dataset == 'raw':
        input_file = config['output_files']['processed_raw_dataset']
        output_file = config['output_files']['processed_raw_with_difficulties']
    else:
        input_file = config['output_files']['processed_mutated_dataset']
        output_file = config['output_files']['processed_mutated_with_difficulties']
    
    dataset = load_json(input_file, logger)
    
    # Evaluate difficulty
    dataset_with_difficulties = evaluate_difficulty(dataset, config, logger)
    
    # Save results
    save_json(dataset_with_difficulties, output_file, logger)
    
    # Log statistics
    pass_counts = [s['pass_cnt'] for s in dataset_with_difficulties]
    avg_pass = sum(pass_counts) / len(pass_counts) if pass_counts else 0
    logger.info(f"Average pass count: {avg_pass:.2f}")
    logger.info(f"Difficulty evaluation complete for {args.dataset} dataset!")


if __name__ == "__main__":
    main()

