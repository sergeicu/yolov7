#!/usr/bin/env python3
"""
Update YOLO label files for filtered fracture datasets.

This script:
1. Assumes all files are in images/ and labels_raw/ directories (no train/val/test splits yet)
2. Creates labels/ directory
3. Processes each .txt file from labels_raw/ to labels/ with correct class IDs
4. Removes confidence scores (keeps only first 5 values: class_id + 4 coords)

USAGE:
    python s20250815_train_filtered_fractures_new_dataset_update_labels.py --dataset_dir /path/to/dataset
    python s20250815_train_filtered_fractures_new_dataset_update_labels.py --dataset_dir /path/to/dataset --test
"""

import os
import shutil
import argparse
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_class_mapping(csv_path: Path, class_names: List[str]) -> Dict[str, int]:
    """
    Create mapping from report IDs to class IDs based on CSV data.
    
    Args:
        csv_path: Path to CSV file
        class_names: List of class names in order (index = class_id)
        
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
    required_columns = ['report', 'num_fractures', 'category']
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
        category = row['category']
        
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
        if pd.isna(category):
            logger.warning(f"Missing category for report {report_id_str} with num_fractures={num_fractures}")
            continue
        
        # Map category to class_id
        if category in class_mappings:
            class_id = class_mappings[category]
            filename_to_class_id[report_id_str] = class_id
            logger.debug(f"Report {report_id_str} mapped to {category} (class_id: {class_id})")
        else:
            logger.warning(f"Category '{category}' not found in class mappings for report {report_id_str}")
    
    logger.info(f"Created mapping for {len(filename_to_class_id)} files")
    logger.info(f"Sample mappings: {dict(list(filename_to_class_id.items())[:5])}")
    
    return filename_to_class_id

def process_label_line(line: str, filename: str, filename_to_class_id: Dict[str, int]) -> str:
    """
    Process a single label line to replace class_id '3' with correct class based on filename.
    
    Args:
        line: Original label line (space-separated values)
        filename: Name of the label file (without .txt extension)
        filename_to_class_id: Dictionary mapping report IDs to class IDs
        
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
    else:
        logger.warning(f"No class mapping found for report_id {report_id} (from filename {filename}), keeping original class_id {original_class_id}")
        new_class_id = original_class_id
    
    # Return formatted line with 5 values (new_class_id + 4 coords)
    return f"{new_class_id} {' '.join(coords)}"

def process_label_file(input_file: Path, output_file: Path, filename_to_class_id: Dict[str, int], test_mode: bool = False) -> bool:
    """
    Process a single label file.
    
    Args:
        input_file: Path to input label file
        output_file: Path to output label file
        filename_to_class_id: Dictionary mapping filenames to class IDs
        test_mode: If True, only show what would be done without writing
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(input_file, 'r') as f_in:
            lines = f_in.readlines()
        
        processed_lines = []
        for line_num, line in enumerate(lines, 1):
            processed_line = process_label_line(line, input_file.stem, filename_to_class_id)
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

def process_dataset(dataset_dir: str, csv_path: str, class_names: List[str], test_mode: bool = False) -> bool:
    """
    Process all label files in the dataset.
    
    Args:
        dataset_dir: Path to dataset directory
        csv_path: Path to CSV file with class mappings
        class_names: List of class names in order
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
    
    # Create class mapping
    filename_to_class_id = create_class_mapping(Path(csv_path), class_names)
    if not filename_to_class_id:
        logger.error("Failed to create class mapping")
        return False
    
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
    
    for txt_file in labels_raw_dir.glob("*.txt"):
        total_files += 1
        output_file = labels_dir / txt_file.name
        
        if process_label_file(txt_file, output_file, filename_to_class_id, test_mode):
            processed_files += 1
        else:
            skipped_files += 1
        
        # Show progress every 100 files
        if total_files % 100 == 0:
            logger.info(f"Processed {total_files} files...")
    
    logger.info(f"Processing complete: {processed_files}/{total_files} files processed (skipped: {skipped_files})")
    return processed_files == total_files

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Update YOLO label files for filtered fracture datasets")
    parser.add_argument("--dataset_dir", type=str, required=True,
                       help="Path to dataset directory containing images/ and labels_raw/")
    parser.add_argument("--csv_path", type=str, required=True,
                       help="Path to CSV file with class mappings")
    parser.add_argument("--class_names", nargs="+", required=True,
                       help="List of class names in order (index = class_id)")
    parser.add_argument("--test", action="store_true",
                       help="Test mode: show what would be done without making changes")
    
    args = parser.parse_args()
    
    if args.test:
        logger.info("TEST MODE - No changes will be made")
    
    # Process the dataset
    success = process_dataset(
        dataset_dir=args.dataset_dir,
        csv_path=args.csv_path,
        class_names=args.class_names,
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