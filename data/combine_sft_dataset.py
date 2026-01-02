"""
Combine SFT Dataset - Step 7 of ExecVerify Data Pipeline

Combines input-output and output-input SFT datasets into a single
training dataset for supervised fine-tuning.

Input: io_sft_dataset.json, oi_sft_dataset.json
Output: sft_dataset.json
"""

from utils import load_config, setup_logging, load_json, save_json


def main():
    """Main execution function."""
    # Load configuration
    config = load_config()
    logger = setup_logging(config)
    
    logger.info("Starting SFT dataset combination...")
    
    # Load IO and OI SFT datasets
    io_file = config['output_files']['io_sft_dataset']
    oi_file = config['output_files']['oi_sft_dataset']
    
    io_sft_dataset = load_json(io_file, logger)
    oi_sft_dataset = load_json(oi_file, logger)
    
    # Combine datasets
    sft_dataset = []
    sft_dataset.extend(io_sft_dataset)
    sft_dataset.extend(oi_sft_dataset)
    
    logger.info(
        f"Combined {len(io_sft_dataset)} IO samples and "
        f"{len(oi_sft_dataset)} OI samples"
    )
    
    # Save combined dataset
    output_file = config['output_files']['sft_dataset']
    save_json(sft_dataset, output_file, logger)
    
    logger.info(f"SFT dataset combination complete! Total: {len(sft_dataset)} samples")


if __name__ == "__main__":
    main()

