#!/usr/bin/env python3
"""
Split dataset into train/val/test sets by moving files efficiently.

This script takes a dataset directory containing images and labels
and splits them into train/val/test sets using a more efficient approach.

USAGE:
    python s20250815_train_filtered_fractures_new_dataset_split_test_val.py \
        --dataset_dir /path/to/dataset \
        --val_size 0.1 --test_size 0.1
    
    OR
    
    python s20250815_train_filtered_fractures_new_dataset_split_test_val.py \
        --dataset_dir /path/to/dataset \
        --val_size 100 --test_size 100 --use_absolute_numbers

INPUT:
    - Dataset directory with images/ and labels/ subdirectories
    - Each subdirectory contains image and label files

OUTPUT:
    - train/, val/, test/ subdirectories under images/ and labels/
    - Files moved from train/ to val/ and test/ according to split ratios
"""

import os
import shutil
import random
import argparse
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def move_file_safely(source_path, target_path):
    """
    Move a file safely, handling existing files.
    """
    try:
        # Remove existing file if it exists
        if target_path.exists():
            target_path.unlink()
        
        # Move the file
        shutil.move(str(source_path), str(target_path))
        return True
    except Exception as e:
        logger.error(f"Failed to move {source_path} to {target_path}: {e}")
        return False

def split_dataset(dataset_dir, val_size, test_size, use_absolute_numbers=False, seed=42):
    """
    Split dataset into train/val/test sets using efficient directory operations.
    
    Args:
        dataset_dir: Path to dataset directory
        val_size: Size of validation set (percentage or absolute number)
        test_size: Size of test set (percentage or absolute number)
        use_absolute_numbers: If True, val_size and test_size are absolute numbers
        seed: Random seed for reproducibility
    """
    logger.info(f"Starting dataset split for: {dataset_dir}")
    logger.info(f"Parameters: val_size={val_size}, test_size={test_size}, use_absolute_numbers={use_absolute_numbers}, seed={seed}")
    
    dataset_path = Path(dataset_dir)
    
    # Check if dataset directory exists
    logger.info(f"Checking if dataset directory exists: {dataset_dir}")
    if not dataset_path.exists():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")
    logger.info("✓ Dataset directory found")
    
    # Set random seed
    logger.info(f"Setting random seed to: {seed}")
    random.seed(seed)
    
    # Define subdirectories
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    
    logger.info(f"Checking for images directory: {images_dir}")
    logger.info(f"Checking for labels directory: {labels_dir}")
    
    if not images_dir.exists() or not labels_dir.exists():
        raise ValueError(f"Dataset must contain 'images' and 'labels' subdirectories")
    logger.info("✓ Images and labels directories found")
    
    # Step 1: Move entire directories to temp locations
    logger.info("Step 1: Moving directories to temp locations...")
    
    # Create temp directories
    images_temp = dataset_path / "images_temp"
    labels_temp = dataset_path / "labels_temp"
    
    # Check if temp directories already exist (from previous run)
    if images_temp.exists():
        logger.info("✓ images_temp already exists, skipping move")
    else:
        logger.info("Moving images directory to images_temp...")
        shutil.move(str(images_dir), str(images_temp))
        logger.info("✓ Moved images to images_temp")
    
    if labels_temp.exists():
        logger.info("✓ labels_temp already exists, skipping move")
    else:
        logger.info("Moving labels directory to labels_temp...")
        shutil.move(str(labels_dir), str(labels_temp))
        logger.info("✓ Moved labels to labels_temp")
    
    # Step 2: Create new directory structure
    logger.info("Step 2: Creating new directory structure...")
    
    # Recreate images and labels directories
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    
    # Create split subdirectories
    for subdir in ["train", "val", "test"]:
        (images_dir / subdir).mkdir(exist_ok=True)
        (labels_dir / subdir).mkdir(exist_ok=True)
        logger.info(f"  ✓ Created {subdir}/ subdirectories")
    
    # Step 3: Move entire temp directories to train (fast bulk operation)
    logger.info("Step 3: Moving temp directories to train...")
    
    train_images_dir = images_dir / "train"
    train_labels_dir = labels_dir / "train"
    
    # Move entire directory contents at once (much faster)
    logger.info("Moving entire images_temp directory to images/train...")
    shutil.move(str(images_temp), str(train_images_dir))
    logger.info("✓ Moved images_temp to train")
    
    logger.info("Moving entire labels_temp directory to labels/train...")
    shutil.move(str(labels_temp), str(train_labels_dir))
    logger.info("✓ Moved labels_temp to train")
    
    # Step 4: Get sorted list of image files (fast)
    logger.info("Step 4: Getting sorted list of image files...")
    
    # Get all PNG files and sort them for consistent indexing
    image_files = sorted([f for f in train_images_dir.iterdir() if f.is_file() and f.suffix.lower() == '.png'])
    total_files = len(image_files)
    logger.info(f"✓ Found {total_files} image files")
    
    if total_files == 0:
        raise ValueError("No image files found in train directory")
    
    # Step 4.5: Validate image-label pairs exist (strict validation)
    logger.info("Step 4.5: Validating image-label pairs...")
    
    missing_labels = []
    
    for i, img_file in enumerate(image_files):
        # Show progress every 1000 files
        if i % 1000 == 0:
            logger.info(f"  Validating pairs {i+1}/{total_files} ({(i+1)/total_files*100:.1f}%)")
        
        # Check if corresponding label exists (simple extension change)
        expected_label_file = train_labels_dir / f"{img_file.stem}.txt"
        if not expected_label_file.exists():
            missing_labels.append(img_file.name)
    
    # If any missing labels found, halt the operation
    if missing_labels:
        logger.error("❌ VALIDATION FAILED: Missing label files detected!")
        logger.error(f"Found {len(missing_labels)} images without corresponding label files:")
        
        # Show first 20 missing labels
        for i, missing_img in enumerate(missing_labels[:20]):
            expected_label = missing_img.replace('.png', '.txt')
            logger.error(f"  {i+1:3d}. {missing_img} → missing {expected_label}")
        
        if len(missing_labels) > 20:
            logger.error(f"  ... and {len(missing_labels) - 20} more missing labels")
        
        logger.error("")
        logger.error("OPERATION HALTED: Please ensure all image files have corresponding label files")
        logger.error("before running this script again.")
        raise ValueError(f"Validation failed: {len(missing_labels)} images missing corresponding label files")
    
    logger.info(f"✓ VALIDATION PASSED: All {total_files} images have corresponding label files")
    logger.info("✓ Proceeding with dataset split...")
    
    # Step 5: Calculate split sizes
    logger.info(f"Step 5: Calculating split sizes for {total_files} total files...")
    
    if use_absolute_numbers:
        logger.info(f"Using absolute numbers: val_size={val_size}, test_size={test_size}")
        val_count = min(int(val_size), total_files)
        test_count = min(int(test_size), total_files - val_count)
        train_count = total_files - val_count - test_count
    else:
        logger.info(f"Using percentages: val_size={val_size*100}%, test_size={test_size*100}%")
        val_count = int(total_files * val_size)
        test_count = int(total_files * test_size)
        train_count = total_files - val_count - test_count
    
    # Ensure all counts are integers
    train_count = int(train_count)
    val_count = int(val_count)
    test_count = int(test_count)
    
    logger.info(f"✓ Split sizes calculated: train={train_count}, val={val_count}, test={test_count}")
    
    # Step 6: Select random indices for splits
    logger.info("Step 6: Selecting random indices for splits...")
    
    # Create list of indices and shuffle them
    indices = list(range(len(image_files)))
    random.shuffle(indices)
    logger.info("✓ Indices shuffled")
    
    # Split indices into sets
    val_indices = indices[:val_count]
    test_indices = indices[val_count:val_count + test_count]
    train_indices = indices[val_count + test_count:]
    
    logger.info(f"✓ Selected {len(val_indices)} indices for val, {len(test_indices)} for test, {len(train_indices)} for train")
    
    # Step 7: Move files to val and test directories
    logger.info("Step 7: Moving files to val and test directories...")
    
    splits = [
        ("val", val_indices, images_dir / "val", labels_dir / "val"),
        ("test", test_indices, images_dir / "test", labels_dir / "test")
    ]
    
    for split_name, indices_list, target_images_dir, target_labels_dir in splits:
        if len(indices_list) == 0:
            logger.info(f"No files to move to {split_name} set")
            continue
            
        logger.info(f"Moving {len(indices_list)} files to {split_name} set...")
        
        moved_count = 0
        missing_labels = 0
        
        for i, idx in enumerate(indices_list):
            # Show progress every 100 files
            if i % 100 == 0:
                logger.info(f"  Moving files {i+1}/{len(indices_list)} to {split_name} ({(i+1)/len(indices_list)*100:.1f}%)")
            
            # Get image file by index (fast access)
            img_file = image_files[idx]
            
            # Find corresponding label file by changing extension
            label_file = train_labels_dir / f"{img_file.stem}.txt"
            
            # Move image file
            img_success = move_file_safely(img_file, target_images_dir / img_file.name)
            # Move label file
            label_success = move_file_safely(label_file, target_labels_dir / label_file.name)
            
            if img_success and label_success:
                moved_count += 1
            else:
                logger.error(f"Failed to move pair: {img_file.name}")
        
        logger.info(f"✓ Moved {moved_count}/{len(indices_list)} file pairs to {split_name}")
    
    # Print summary
    logger.info("Dataset split complete!")
    logger.info(f"Train set: {len(train_indices)} files")
    logger.info(f"Validation set: {len(val_indices)} files")
    logger.info(f"Test set: {len(test_indices)} files")
    
    # Verify the split
    for split_name in ["train", "val", "test"]:
        split_images = len(list((images_dir / split_name).glob("*.png")))
        split_labels = len(list((labels_dir / split_name).glob("*.txt")))
        logger.info(f"{split_name.capitalize()}: {split_images} images, {split_labels} labels")

def main():
    parser = argparse.ArgumentParser(
        description='Split dataset into train/val/test sets by moving files efficiently'
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        required=True,
        help='Path to dataset directory containing images/ and labels/ subdirectories'
    )
    parser.add_argument(
        '--val_size',
        type=float,
        default=0.1,
        help='Size of validation set (percentage 0.0-1.0 or absolute number)'
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.1,
        help='Size of test set (percentage 0.0-1.0 or absolute number)'
    )
    parser.add_argument(
        '--use_absolute_numbers',
        action='store_true',
        help='If True, val_size and test_size are treated as absolute numbers instead of percentages'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()
    
    # Validate arguments
    if not args.use_absolute_numbers:
        if args.val_size < 0 or args.val_size > 1:
            raise ValueError("val_size must be between 0.0 and 1.0 when using percentages")
        if args.test_size < 0 or args.test_size > 1:
            raise ValueError("test_size must be between 0.0 and 1.0 when using percentages")
        if args.val_size + args.test_size >= 1.0:
            raise ValueError("val_size + test_size must be less than 1.0")
    else:
        if args.val_size < 0 or args.test_size < 0:
            raise ValueError("val_size and test_size must be non-negative when using absolute numbers")
    
    # Split the dataset
    split_dataset(
        dataset_dir=args.dataset_dir,
        val_size=args.val_size,
        test_size=args.test_size,
        use_absolute_numbers=args.use_absolute_numbers,
        seed=args.seed
    )

if __name__ == "__main__":
    main() 