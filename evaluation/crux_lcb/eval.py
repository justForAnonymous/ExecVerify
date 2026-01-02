"""
Evaluation script for CRUXEval and LiveCodeBench-Exec benchmarks.

This script evaluates models on input-output (IO) and output-input (OI) 
prediction tasks using assertion completion.
"""

import argparse
import re

from datasets import load_dataset
from vllm import LLM, SamplingParams
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from util import (
    extract_xml_answer,
    extract_func_name,
    extract_args,
    extract_result,
    generate_r1_prompt,
    test_python_code
)


RESP_LEN = 4096 * 3
PROMPT_LEN = 512
BATCH_SIZE = 1024


def custom_collate_fn(batch):
    """
    Custom collate function for DataLoader.
    
    Args:
        batch: Batch of data samples
        
    Returns:
        dict: Collated batch with code, input, and output lists
    """
    codes = [item['code'] for item in batch]
    inputs = [item['input'] for item in batch]
    outputs = [item['output'] for item in batch]
    
    return {
        "code": codes,
        "input": inputs,
        "output": outputs
    }


err_infos = []


def eval_oi(tokenizer, dataset, llm, sampling_params):
    """
    Evaluate output-to-input (OI) prediction task.
    
    Given function code and expected output, predict the input arguments.
    
    Args:
        tokenizer: Model tokenizer
        dataset: Evaluation dataset
        llm: vLLM model instance
        sampling_params: Sampling parameters for generation
        
    Returns:
        int: Number of correct predictions
    """
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)
    true_cnt = 0
    
    for problems_batch in data_loader:
        func_strs = []
        inputs = []
        outputs = []
        formatted_prompts = []
        
        for func_str, input_val, output_val in zip(
            problems_batch["code"], problems_batch["input"], problems_batch["output"]
        ):
            func_strs.append(func_str)
            func_name = extract_func_name(func_str)
            if input_val.startswith(func_name):
                input_val = input_val[len(func_name):]
            inputs.append(input_val)
            outputs.append(output_val)
            formatted_prompts.append(generate_r1_prompt(tokenizer, func_str, output_val))

        generated_outputs = llm.generate(formatted_prompts, sampling_params)
        
        for func_str, input_val, output_val, generated_output in zip(
            func_strs, inputs, outputs, generated_outputs
        ):
            answer = extract_xml_answer(generated_output.outputs[0].text)
            pattern = r"```python\n(.*?)\n```"
            matches = re.findall(pattern, answer, re.DOTALL)
            
            if len(matches) <= 0:
                output_code = answer
            else:
                output_code = matches[0]

            output_args = extract_args(output_code)
            code_to_run = f"""
from typing import *
from itertools import *
from collections import *
import bisect

{func_str} 

assert {extract_func_name(func_str)}{output_args} == {output_val}
"""
            
            if test_python_code(code_to_run):
                true_cnt += 1
            else:
                err_infos.append(code_to_run)

    return true_cnt


def eval_io(tokenizer, dataset, llm, sampling_params):
    """
    Evaluate input-to-output (IO) prediction task.
    
    Given function code and input arguments, predict the output.
    
    Args:
        tokenizer: Model tokenizer
        dataset: Evaluation dataset
        llm: vLLM model instance
        sampling_params: Sampling parameters for generation
        
    Returns:
        int: Number of correct predictions
    """
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)
    true_cnt = 0
    
    for problems_batch in data_loader:
        func_strs = []
        inputs = []
        outputs = []
        formatted_prompts = []
        
        for func_str, input_val, output_val in zip(
            problems_batch["code"], problems_batch["input"], problems_batch["output"]
        ):
            func_strs.append(func_str)
            func_name = extract_func_name(func_str)
            if input_val.startswith(func_name):
                input_val = input_val[len(func_name):]
                input_val = input_val[1:-1]
        
            inputs.append(input_val)
            outputs.append(output_val)
            formatted_prompts.append(generate_r1_prompt(tokenizer, func_str, "????", input_val))

        generated_outputs = llm.generate(formatted_prompts, sampling_params)
        
        for func_str, input_val, output_val, generated_output in zip(
            func_strs, inputs, outputs, generated_outputs
        ):
            answer = extract_xml_answer(generated_output.outputs[0].text)
            pattern = r"```python\n(.*?)\n```"
            matches = re.findall(pattern, answer, re.DOTALL)
            
            if len(matches) <= 0:
                output_code = answer
            else:
                output_code = matches[0]
            
            output_output = extract_result(output_code)

            code_to_run = f"""
from typing import *
from itertools import *
from collections import *
import bisect

{func_str} 

assert {extract_func_name(func_str)}({input_val}) == {output_output}
"""

            if test_python_code(code_to_run):
                true_cnt += 1
            else:
                err_infos.append({
                    'code_to_run': code_to_run,
                    'generated_output': generated_output.outputs[0].text,
                    'expected_output': output_val
                })
    
    return true_cnt




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate models on CRUXEval and LiveCodeBench-Exec benchmarks'
    )
    parser.add_argument("model_path", help="Path to the model")
    parser.add_argument(
        "dataset_name",
        help="Dataset name (e.g., cruxeval-org/cruxeval or livecodebench/execution-v2)"
    )
    
    args = parser.parse_args()
    model_path = args.model_path
    dataset_name = args.dataset_name

    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_name}")

    dataset = load_dataset(dataset_name)
    dataset = dataset['test']

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=RESP_LEN + PROMPT_LEN,
        gpu_memory_utilization=0.8,
        tensor_parallel_size=1
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=RESP_LEN,
        repetition_penalty=1.0
    )

    io_true_cnt = eval_io(tokenizer, dataset, llm, sampling_params)
    oi_true_cnt = eval_oi(tokenizer, dataset, llm, sampling_params)
           
    print(f"\n{'='*50}")
    print(f"{dataset_name} - IO pass@1: {io_true_cnt/len(dataset):.4f}")
    print(f"{dataset_name} - OI pass@1: {oi_true_cnt/len(dataset):.4f}")
    print(f"{'='*50}")