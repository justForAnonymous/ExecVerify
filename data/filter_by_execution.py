"""
Filter by Execution - Step 3 of ExecVerify Data Pipeline

Filters code samples by checking if they can be successfully executed.
Extracts function definitions, arguments, and validates execution.

Input: raw_dataset.json, mutated_raw_dataset.json
Output: processed_raw_dataset.json, processed_mutated_dataset.json
"""

import sys
import subprocess
import concurrent.futures
from tqdm import tqdm
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
from utils import load_config, setup_logging, load_json, save_json


def extract_func_and_args(test_str: str):
    """
    Extract function definitions and arguments from code using tree-sitter.
    
    Args:
        test_str: Python code string
        
    Returns:
        Tuple of (function_definitions, entry_function_name, arguments, success_flag)
    """
    language = Language(tspython.language())
    parser = Parser(language)
    tree = parser.parse(bytes(test_str, "utf8"))
    root_node = tree.root_node
    
    # Query for function definitions
    def_query_string = """
    (   
        function_definition
            name:(identifier)@func_name 
    ) @func_def
    """
    def_query = language.query(def_query_string)
    def_captures = def_query.captures(root_node)
    func_defs = []
    if "func_def" in def_captures:
        func_defs = def_captures["func_def"]
    if len(func_defs) == 0:
        return "", "", "", False
    
    # Query for function call arguments in print statement
    args_query_string = """
    ((call
    function: (identifier) @func_name
    arguments: (
        argument_list(
            call
                function: (identifier) @entry_func_name
                arguments: (argument_list) @extracted_args
        )
    )
    ) 
    (#eq? @func_name "print"))
    """
    args_query = language.query(args_query_string)
    args_captures = args_query.captures(root_node)
    args = []
    if "extracted_args" in args_captures:
        args = args_captures["extracted_args"]
    entry_func_names = []
    if "entry_func_name" in args_captures:
        entry_func_names = args_captures["entry_func_name"]
    
    if len(args) != 1:
        return "", "", "", False
    
    if len(entry_func_names) != 1:
        return "", "", "", False
    
    return func_defs, entry_func_names[0], args[0], True


def run_code(code_str: str, timeout: float = 0.5) -> str:
    """
    Execute Python code and return stdout.
    
    Args:
        code_str: Python code to execute
        timeout: Maximum execution time in seconds
        
    Returns:
        Standard output from execution, or -1 on failure
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
        if result.returncode == 0:
            return result.stdout
    except Exception:
        return -1
    
    return -1


def process_single_sample(sample: dict, timeout: float) -> dict:
    """
    Process a single code sample to extract and validate execution.
    
    Args:
        sample: Dictionary containing 'code' and 'setting' keys
        timeout: Execution timeout in seconds
        
    Returns:
        Processed sample dictionary or None if invalid
    """
    try:
        func_defs_nodes, func_name_node, args_node, flag = extract_func_and_args(sample['code'])
        if not flag:
            return None
        
        func_str = ''
        for func_def_node in func_defs_nodes:
            func_str = func_str + func_def_node.text.decode('utf8') + '\n\n'
        
        func_args = args_node.text.decode('utf8')[1:-1]
        func_name = func_name_node.text.decode('utf8')
        
        code_to_run = f"""
{func_str}    

print(repr({func_name}({func_args})))
"""
        
        result_str = run_code(code_to_run, timeout)
        if result_str == -1:
            return None
        result_str = result_str.strip()
        
        code_to_run_check = f"""
{func_str}

assert {func_name}({func_args}) == {result_str}
"""
        if run_code(code_to_run_check, timeout) == -1:
            return None
        
        return {
            'func_str': func_str,
            'func_name': func_name,
            'func_args': func_args,
            'result': result_str,
            'setting': sample['setting']
        }
    except Exception:
        return None


def process_dataset(dataset: list, max_workers: int, timeout: float, logger) -> list:
    """
    Process dataset using parallel workers.
    
    Args:
        dataset: List of code samples
        max_workers: Number of parallel workers
        timeout: Execution timeout
        logger: Logger instance
        
    Returns:
        List of successfully processed samples
    """
    processed_dataset = []
    
    logger.info(f"Processing {len(dataset)} samples with {max_workers} workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_sample, sample, timeout) 
            for sample in dataset
        ]
        
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(dataset),
            desc="Processing"
        ):
            result = future.result()
            if result is not None:
                processed_dataset.append(result)
    
    logger.info(f"Successfully processed: {len(processed_dataset)}/{len(dataset)}")
    return processed_dataset


def main():
    """Main execution function."""
    # Load configuration
    config = load_config()
    logger = setup_logging(config)
    
    logger.info("Starting execution filtering...")
    
    # Get parameters from config
    max_workers = config['filtering']['max_workers']
    timeout = config['filtering']['execution_timeout']
    
    # Process raw dataset
    logger.info("Processing raw dataset...")
    raw_dataset_file = config['output_files']['raw_dataset']
    raw_dataset = load_json(raw_dataset_file, logger)
    
    processed_raw_dataset = process_dataset(raw_dataset, max_workers, timeout, logger)
    
    output_file = config['output_files']['processed_raw_dataset']
    save_json(processed_raw_dataset, output_file, logger)
    
    # Process mutated dataset
    logger.info("Processing mutated dataset...")
    mutated_dataset_file = config['output_files']['mutated_raw_dataset']
    mutated_raw_dataset = load_json(mutated_dataset_file, logger)
    
    processed_mutated_dataset = process_dataset(
        mutated_raw_dataset, max_workers, timeout, logger
    )
    
    output_file = config['output_files']['processed_mutated_dataset']
    save_json(processed_mutated_dataset, output_file, logger)
    
    logger.info("Execution filtering complete!")


if __name__ == "__main__":
    main()
