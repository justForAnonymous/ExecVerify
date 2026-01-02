"""
Combine Data with Difficulties - Step 5 of ExecVerify Data Pipeline

Combines raw and mutated datasets with difficulty scores, filtering out
samples that are too easy (pass_cnt > max_pass_count threshold).

Input: processed_raw_dataset_with_difficulties.json,
       processed_mutated_dataset_with_difficulties.json
Output: filtered_all_dataset_with_difficulties.json
"""

import json
from utils import load_config, setup_logging, load_json, save_json


def main():
    """Main execution function."""
    # Load configuration
    config = load_config()
    logger = setup_logging(config)
    
    logger.info("Starting data combination and filtering...")
    
    # Load datasets with difficulties
    raw_file = config['output_files']['processed_raw_with_difficulties']
    mutated_file = config['output_files']['processed_mutated_with_difficulties']
    
    processed_raw_dataset_with_difficulties = load_json(raw_file, logger)
    processed_mutated_dataset_with_difficulties = load_json(mutated_file, logger)
    
    # Get filtering threshold
    max_pass_count = config['filtering']['max_pass_count']
    logger.info(f"Filtering samples with pass_cnt <= {max_pass_count}")
    
    # Filter and combine datasets
    filtered_all_dataset_with_difficulties = []
    
    for sample in processed_raw_dataset_with_difficulties:
        if sample['pass_cnt'] <= max_pass_count:
            filtered_all_dataset_with_difficulties.append(sample)
    
    for sample in processed_mutated_dataset_with_difficulties:
        if sample['pass_cnt'] <= max_pass_count:
            filtered_all_dataset_with_difficulties.append(sample)
    
    # Log statistics
    total_samples = (
        len(processed_raw_dataset_with_difficulties) +
        len(processed_mutated_dataset_with_difficulties)
    )
    filtered_count = len(filtered_all_dataset_with_difficulties)
    logger.info(
        f"Filtered {filtered_count}/{total_samples} samples "
        f"({100*filtered_count/total_samples:.1f}%)"
    )
    
    # Save combined dataset
    output_file = config['output_files']['filtered_all_with_difficulties']
    save_json(filtered_all_dataset_with_difficulties, output_file, logger)
    
    logger.info("Data combination complete!")


if __name__ == "__main__":
    main()
