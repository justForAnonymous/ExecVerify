"""
Utility functions for the ExecVerify data pipeline.
Provides configuration loading, logging setup, and shared helper functions.
"""

import os
import sys
import json
import logging
import subprocess
from typing import Any, Dict, Optional
import yaml


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file '{config_path}' not found. "
            f"Please create it based on config.yaml.example"
        )
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate that required configuration parameters are set.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If required parameters are missing or invalid
    """
    # Check that model paths are set
    models = config.get('models', {})
    for model_key in ['code_generator', 'reasoning_model', 'difficulty_evaluator']:
        if not models.get(model_key):
            raise ValueError(
                f"Model path '{model_key}' is not set in config.yaml. "
                f"Please update the configuration file with valid model paths."
            )
        # Check that model path exists
        model_path = models[model_key]
        if not os.path.exists(model_path):
            raise ValueError(
                f"Model path '{model_path}' for '{model_key}' does not exist. "
                f"Please verify the path in config.yaml"
            )


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Setup logging configuration for the pipeline.
    
    Args:
        config: Configuration dictionary containing logging settings
        
    Returns:
        Configured logger instance
    """
    logging_config = config.get('logging', {})
    level = logging_config.get('level', 'INFO')
    format_str = logging_config.get(
        'format', 
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logging.basicConfig(
        level=getattr(logging, level),
        format=format_str,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger('ExecVerify')


def run_code(code_str: str, timeout: float = 0.5) -> str:
    """
    Execute Python code string and return stdout.
    
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


def save_json(data: Any, filepath: str, logger: Optional[logging.Logger] = None) -> None:
    """
    Save data to JSON file with logging.
    
    Args:
        data: Data to save
        filepath: Output file path
        logger: Optional logger instance
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    if logger:
        logger.info(f"Saved {len(data) if hasattr(data, '__len__') else 'data'} items to {filepath}")


def load_json(filepath: str, logger: Optional[logging.Logger] = None) -> Any:
    """
    Load data from JSON file with logging.
    
    Args:
        filepath: Input file path
        logger: Optional logger instance
        
    Returns:
        Loaded data
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file '{filepath}' not found")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if logger:
        logger.info(f"Loaded {len(data) if hasattr(data, '__len__') else 'data'} items from {filepath}")
    
    return data


def build_chat_prompt(tokenizer, system_content: str, user_content: str) -> str:
    """
    Build chat prompt using tokenizer's chat template.
    
    Args:
        tokenizer: HuggingFace tokenizer with chat template
        system_content: System message content
        user_content: User message content
        
    Returns:
        Formatted chat prompt string
    """
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def check_file_exists(filepath: str, description: str = "File") -> None:
    """
    Check if a file exists, raise informative error if not.
    
    Args:
        filepath: Path to check
        description: Description of the file for error message
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"{description} '{filepath}' not found. "
            f"Please run the previous pipeline steps first."
        )

