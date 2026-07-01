#!/usr/bin/env python3
"""
Update YOLO label files for filtered fracture datasets.

This script automatically handles 4 different categories:
- alignment: uses alignment_category column
- classification: uses classification_category column  
- healing: uses healing_category column
- anatomical_regions: uses anatomical_region column

This script:
1. Assumes all files are in images/ and labels_raw/ directories (no train/val/test splits yet)
2. Creates labels/ directory
3. Processes each .txt file from labels_raw/ to labels/ with correct class IDs
4. Removes confidence scores (keeps only first 5 values: class_id + 4 coords)

USAGE:
    # Process all 4 categories with default paths
    python s20250815_train_filtered_fractures_new_dataset_update_labels.py --test
    
    # Process all 4 categories with default paths (actual execution)
    python s20250815_train_filtered_fractures_new_dataset_update_labels.py
    
    # Process specific category
    python s20250815_train_filtered_fractures_new_dataset_update_labels.py \
        --category alignment --test
    
    # Custom paths
    python s20250815_train_filtered_fractures_new_dataset_update_labels.py \
        --dataset_dir /path/to/dataset \
        --csv_path /path/to/mapping.csv \
        --yaml_path /path/to/classes.yaml \
        --test
"""

import os
import shutil
import argparse
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_BASE1 = "/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/pngs_aug2025/"
DEFAULT_BASE2 = "/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/experiments/labeller_running_for_andy/s20250711_generate_structured_reports/"
DEFAULT_BASE3 = "/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/yolov7/data"

# Category configurations
CATEGORY_CONFIGS = {
    'alignment': {
        'dataset_suffix': 'alignment_01fracture',
        'csv_suffix': 'alignment_01fracture',
        'yaml_suffix': 'alignment_filter',
        'column_name': 'alignment_category'
    },
    'classification': {
        'dataset_suffix': 'classification_01fracture', 
        'csv_suffix': 'classification_01fracture',
        'yaml_suffix': 'classification_filter',
        'column_name': 'classification_category'
    },
    'healing': {
        'dataset_suffix': 'healing_01fracture',
        'csv_suffix': 'healing_01fracture', 
        'yaml_suffix': 'healing_filter',
        'column_name': 'healing_category'
    },
    'anatomical_regions': {
        'dataset_suffix': 'anatomical_regions_01fracture',
        'csv_suffix': 'anatomical_regions_01fracture',
        'yaml_suffix': 'anatomical_regions_filter', 
        'column_name': 'anatomical_region'
    }
}

def load_class_names_from_yaml(yaml_path: Path) -> List[str]:
    """
    Load class names from YAML file.
    
    Args:
        yaml_path: Path to YAML file
        
    Returns:
        List of class names in order
    """
    if not yaml_path.exists():
        logger.error(f"YAML file does not exist: {yaml_path}")
        return []
    
    try:
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # Handle both list and string formats for names
        names_data = yaml_data.get('names', [])
        
        if isinstance(names_data, list):
            class_names = names_data
        elif isinstance(names_data, str):
            # If names is a string, try to parse it as a list
            # Remove brackets and quotes, then split by comma
            names_str = names_data.strip()
            if names_str.startswith('[') and names_str.endswith(']'):
                names_str = names_str[1:-1]  # Remove brackets
            class_names = [name.strip().strip("'\"") for name in names_str.split(',')]
        else:
            logger.error(f"Unexpected format for 'names' in YAML file: {type(names_data)}")
            return []
        
        if not class_names:
            logger.error(f"No class names found in YAML file: {yaml_path}")
            return []
        
        logger.info(f"Loaded class names from YAML: {class_names}")
        return class_names
        
    except Exception as e:
        logger.error(f"Failed to load YAML file {yaml_path}: {e}")
        return []

def create_class_mapping(csv_path: Path, class_names: List[str], category_column: str) -> Dict[str, int]:
    """
    Create mapping from report IDs to class IDs based on CSV data.
    
    Args:
        csv_path: Path to CSV file
        class_names: List of class names in order (index = class_id)
        category_column: Name of the category column to use
        
    Returns:
        Dictionary mapping report IDs to class IDs
    """
    if not csv_path.exists():
        logger.error(f"CSV file does not exist: {csv_path}")
        return {}
    
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded CSV file: {csv_path}")
        logger.info(f"CSV shape: {df.shape}")
        logger.info(f"CSV columns: {list(df.columns)}")
    except Exception as e:
        logger.error(f"Failed to load CSV file: {e}")
        return {}
    
    # Create class_name to class_id mapping
    class_mappings = {class_name: i for i, class_name in enumerate(class_names)}
    logger.info(f"Class mappings: {class_mappings}")
    
    # Check required columns
    required_columns = ['report', 'num_fractures', category_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        logger.error(f"Available columns: {list(df.columns)}")
        return {}
    
    # Create filename to class_id mapping
    filename_to_class_id = {}
    
    for _, row in df.iterrows():
        # Get report ID
        report_id = row['report']
        num_fractures = row['num_fractures']
        category_value = row[category_column]
        
        # Skip if report_id is missing
        if pd.isna(report_id):
            continue
        
        # Convert report_id to string for matching
        report_id_str = str(int(report_id))
        
        # Handle no_fracture case
        if pd.isna(num_fractures) or num_fractures == 0:
            if 'no_fracture' in class_mappings:
                filename_to_class_id[report_id_str] = class_mappings['no_fracture']
                logger.debug(f"Report {report_id_str} mapped to no_fracture (class_id: {class_mappings['no_fracture']})")
            continue
        
        # Handle fracture cases
        if pd.isna(category_value):
            logger.warning(f"Missing {category_column} for report {report_id_str} with num_fractures={num_fractures}")
            continue
        
        # Map category to class_id
        if category_value in class_mappings:
            class_id = class_mappings[category_value]
            filename_to_class_id[report_id_str] = class_id
            logger.debug(f"Report {report_id_str} mapped to {category_value} (class_id: {class_id})")
        else:
            logger.warning(f"Category '{category_value}' not found in class mappings for report {report_id_str}")
    
    logger.info(f"Created mapping for {len(filename_to_class_id)} files")
    logger.info(f"Sample mappings: {dict(list(filename_to_class_id.items())[:5])}")
    
    return filename_to_class_id

def process_label_line(line: str, filename: str, filename_to_class_id: Dict[str, int], class_names: List[str], test_mode: bool = False) -> str:
    """
    Process a single label line to replace class_id '3' with correct class based on filename.
    
    Args:
        line: Original label line (space-separated values)
        filename: Name of the label file (without .txt extension)
        filename_to_class_id: Dictionary mapping report IDs to class IDs
        class_names: List of class names for mapping display
        test_mode: If True, show detailed mapping information
        
    Returns:
        Processed label line with correct class_id (replacing '3')
        or empty string if line is invalid
    """
    parts = line.strip().split()
    
    # Handle empty lines (no fractures)
    if len(parts) == 0:
        return ""
    
    # Validate line format: must have exactly 6 values (class_id, 4 coords, confidence)
    if len(parts) != 6:
        logger.warning(f"Invalid label line format. Expected 6 values, got {len(parts)}: {line.strip()}")
        return ""
    
    # Extract values
    original_class_id = int(parts[0])
    coords = parts[1:5]  # x_center, y_center, width, height
    confidence = parts[5]  # confidence (will be discarded)
    
    # Check if original class_id is '3' as expected
    if original_class_id != 3:
        logger.warning(f"Expected class_id '3', but got '{original_class_id}' for filename {filename}")
    
    # Extract report ID from filename (first part before dash)
    report_id = filename.split('-')[0] if '-' in filename else filename
    
    # Get the correct class_id based on report_id
    if report_id in filename_to_class_id:
        new_class_id = filename_to_class_id[report_id]
        if test_mode:
            category_name = class_names[new_class_id] if new_class_id < len(class_names) else f"unknown_{new_class_id}"
            logger.info(f"  {filename}.txt: '3' -> '{new_class_id}' (corresponding to '{category_name}')")
    else:
        logger.warning(f"No class mapping found for report_id {report_id} (from filename {filename}), keeping original class_id {original_class_id}")
        new_class_id = original_class_id
        if test_mode:
            logger.info(f"  {filename}.txt: '3' -> '{new_class_id}' (no mapping found, keeping original)")
    
    # Return formatted line with 5 values (new_class_id + 4 coords)
    return f"{new_class_id} {' '.join(coords)}"

def process_label_file(input_file: Path, output_file: Path, filename_to_class_id: Dict[str, int], class_names: List[str], test_mode: bool = False) -> bool:
    """
    Process a single label file.
    
    Args:
        input_file: Path to input label file
        output_file: Path to output label file
        filename_to_class_id: Dictionary mapping filenames to class IDs
        class_names: List of class names for mapping display
        test_mode: If True, only show what would be done without writing
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(input_file, 'r') as f_in:
            lines = f_in.readlines()
        
        processed_lines = []
        for line_num, line in enumerate(lines, 1):
            processed_line = process_label_line(line, input_file.stem, filename_to_class_id, class_names, test_mode)
            if processed_line:  # Only add non-empty lines
                processed_lines.append(processed_line)
        
        if test_mode:
            logger.info(f"TEST: Would write {len(processed_lines)} lines to {output_file}")
            if processed_lines:
                logger.info(f"TEST: Sample processed line: {processed_lines[0]}")
        else:
            with open(output_file, 'w') as f_out:
                f_out.write('\n'.join(processed_lines))
        
        return True
    except Exception as e:
        logger.error(f"Failed to process {input_file}: {e}")
        return False

def process_dataset(dataset_dir: str, csv_path: str, yaml_path: str, category_column: str, test_mode: bool = False) -> bool:
    """
    Process all label files in the dataset.
    
    Args:
        dataset_dir: Path to dataset directory
        csv_path: Path to CSV file with class mappings
        yaml_path: Path to YAML file with class names
        category_column: Name of the category column to use
        test_mode: If True, only show what would be done without making changes
        
    Returns:
        True if successful, False otherwise
    """
    dataset_path = Path(dataset_dir)
    
    # Check if dataset directory exists
    if not dataset_path.exists():
        logger.error(f"Dataset directory does not exist: {dataset_path}")
        return False
    
    # Check for required directories
    images_dir = dataset_path / "images"
    labels_raw_dir = dataset_path / "labels_raw"
    
    if not images_dir.exists():
        logger.error(f"Images directory does not exist: {images_dir}")
        return False
    
    if not labels_raw_dir.exists():
        logger.error(f"Labels_raw directory does not exist: {labels_raw_dir}")
        return False
    
    logger.info(f"Processing dataset: {dataset_path}")
    logger.info(f"Images directory: {images_dir}")
    logger.info(f"Labels_raw directory: {labels_raw_dir}")
    
    # Load class names from YAML
    class_names = load_class_names_from_yaml(Path(yaml_path))
    if not class_names:
        logger.error("Failed to load class names from YAML")
        return False
    
    # Create class mapping
    filename_to_class_id = create_class_mapping(Path(csv_path), class_names, category_column)
    if not filename_to_class_id:
        logger.error("Failed to create class mapping")
        return False
    
    # Show detailed mapping information in test mode
    if test_mode:
        logger.info(f"\n{'='*50}")
        logger.info(f"CLASS MAPPINGS FOR {category_column.upper()}:")
        logger.info(f"{'='*50}")
        for class_id, class_name in enumerate(class_names):
            logger.info(f"  Class ID {class_id}: '{class_name}'")
        logger.info(f"\nSAMPLE REPORT ID MAPPINGS:")
        sample_mappings = dict(list(filename_to_class_id.items())[:10])
        for report_id, class_id in sample_mappings.items():
            category_name = class_names[class_id] if class_id < len(class_names) else f"unknown_{class_id}"
            logger.info(f"  Report {report_id} -> Class ID {class_id} ('{category_name}')")
        if len(filename_to_class_id) > 10:
            logger.info(f"  ... and {len(filename_to_class_id) - 10} more mappings")
        logger.info(f"\n{'='*50}")
        logger.info(f"PROCESSING FILES:")
        logger.info(f"{'='*50}")
    
    # Create labels directory
    labels_dir = dataset_path / "labels"
    if not test_mode:
        labels_dir.mkdir(exist_ok=True)
        logger.info(f"Created labels directory: {labels_dir}")
    else:
        logger.info(f"TEST: Would create labels directory: {labels_dir}")
    
    # Process all label files
    total_files = 0
    processed_files = 0
    skipped_files = 0
    
    # Get list of all txt files
    txt_files = list(labels_raw_dir.glob("*.txt"))
    
    # Limit to 20 files in test mode
    if test_mode:
        txt_files = txt_files[:20]
        logger.info(f"TEST MODE: Processing only first 20 files out of {len(list(labels_raw_dir.glob('*.txt')))} total files")
    
    for txt_file in txt_files:
        total_files += 1
        output_file = labels_dir / txt_file.name
        
        if process_label_file(txt_file, output_file, filename_to_class_id, class_names, test_mode):
            processed_files += 1
        else:
            skipped_files += 1
        
        # Show progress every 100 files (or every 5 files in test mode)
        progress_interval = 5 if test_mode else 100
        if total_files % progress_interval == 0:
            logger.info(f"Processed {total_files} files...")
    
    logger.info(f"Processing complete: {processed_files}/{total_files} files processed (skipped: {skipped_files})")
    return processed_files == total_files

def process_all_categories(test_mode: bool = False):
    """
    Process all 4 categories using default paths.
    
    Args:
        test_mode: If True, only show what would be done without making changes
    """
    logger.info("Processing all 4 categories...")
    
    for category_name, config in CATEGORY_CONFIGS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing category: {category_name}")
        logger.info(f"{'='*60}")
        
        # Build paths using defaults
        dataset_dir = f"{DEFAULT_BASE1}/{config['dataset_suffix']}"
        csv_path = f"{DEFAULT_BASE2}/ready_for_symlinks_{config['csv_suffix']}.csv"
        yaml_path = f"{DEFAULT_BASE3}/s20250815_{config['yaml_suffix']}.yaml"
        category_column = config['column_name']
        
        logger.info(f"Dataset dir: {dataset_dir}")
        logger.info(f"CSV path: {csv_path}")
        logger.info(f"YAML path: {yaml_path}")
        logger.info(f"Category column: {category_column}")
        
        # Process this category
        success = process_dataset(
            dataset_dir=dataset_dir,
            csv_path=csv_path,
            yaml_path=yaml_path,
            category_column=category_column,
            test_mode=test_mode
        )
        
        if success:
            logger.info(f"✓ Successfully processed {category_name}")
        else:
            logger.error(f"✗ Failed to process {category_name}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Update YOLO label files for filtered fracture datasets")
    parser.add_argument("--dataset_dir", type=str,
                       help="Path to dataset directory containing images/ and labels_raw/")
    parser.add_argument("--csv_path", type=str,
                       help="Path to CSV file with class mappings")
    parser.add_argument("--yaml_path", type=str,
                       help="Path to YAML file with class names")
    parser.add_argument("--category", type=str, choices=list(CATEGORY_CONFIGS.keys()),
                       help="Specific category to process (if not specified, processes all)")
    parser.add_argument("--test", action="store_true",
                       help="Test mode: show what would be done without making changes")
    
    args = parser.parse_args()
    
    if args.test:
        logger.info("TEST MODE - No changes will be made")
    
    # If no specific paths provided, use defaults and process all categories
    if not args.dataset_dir and not args.csv_path and not args.yaml_path:
        if args.category:
            # Process specific category with defaults
            config = CATEGORY_CONFIGS[args.category]
            dataset_dir = f"{DEFAULT_BASE1}/{config['dataset_suffix']}"
            csv_path = f"{DEFAULT_BASE2}/ready_for_symlinks_{config['csv_suffix']}.csv"
            yaml_path = f"{DEFAULT_BASE3}/s20250815_{config['yaml_suffix']}.yaml"
            category_column = config['column_name']
            
            logger.info(f"Processing category: {args.category}")
            logger.info(f"Dataset dir: {dataset_dir}")
            logger.info(f"CSV path: {csv_path}")
            logger.info(f"YAML path: {yaml_path}")
            logger.info(f"Category column: {category_column}")
            
            success = process_dataset(
                dataset_dir=dataset_dir,
                csv_path=csv_path,
                yaml_path=yaml_path,
                category_column=category_column,
                test_mode=args.test
            )
        else:
            # Process all categories
            process_all_categories(test_mode=args.test)
            success = True  # Assume success for all categories
    else:
        # Custom paths provided - validate that all required args are provided
        if not all([args.dataset_dir, args.csv_path, args.yaml_path]):
            parser.error("If custom paths are provided, all of --dataset_dir, --csv_path, and --yaml_path must be provided")
        
        if not args.category:
            parser.error("Category must be specified when using custom paths")
        
        config = CATEGORY_CONFIGS[args.category]
        category_column = config['column_name']
        
        success = process_dataset(
            dataset_dir=args.dataset_dir,
            csv_path=args.csv_path,
            yaml_path=args.yaml_path,
            category_column=category_column,
            test_mode=args.test
        )
    
    if success:
        logger.info("Processing completed successfully!")
    else:
        logger.error("Processing failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 