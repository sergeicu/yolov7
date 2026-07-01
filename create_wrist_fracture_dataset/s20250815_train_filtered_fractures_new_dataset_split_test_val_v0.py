#!/usr/bin/env python3
"""
Split dataset into train/val/test sets by creating symlinks to original files.

This script takes a dataset directory containing images and labels (as symlinks)
and splits them into train/val/test sets while preserving the original symlinks.

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
    - Each subdirectory contains symlinks to original files

OUTPUT:
    - train/, val/, test/ subdirectories under images/ and labels/
    - all/ subdirectory containing original symlinks
    - New symlinks in train/val/test pointing to original files
"""

import os
import shutil
import random
import argparse
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def resolve_symlink_target(file_path):
    """
    Resolve symlink to get the actual target file path.
    Handles cases where the target itself might be a symlink.
    """
    try:
        # Get the real path (resolves all symlinks in the chain)
        real_path = os.path.realpath(file_path)
        return real_path
    except (OSError, FileNotFoundError):
        logger.warning(f"Could not resolve symlink: {file_path}")
        return None

def create_symlink_safely(source_path, target_path):
    """
    Create a symlink safely, removing existing one if it exists.
    """
    try:
        # Remove existing symlink if it exists
        if os.path.islink(target_path):
            os.unlink(target_path)
        elif os.path.exists(target_path):
            os.remove(target_path)
        
        # Create new symlink
        os.symlink(source_path, target_path)
        return True
    except Exception as e:
        logger.error(f"Failed to create symlink {target_path} -> {source_path}: {e}")
        return False

def create_symlink_batch(args):
    """
    Create a single symlink - used for parallel processing.
    """
    source_path, target_path = args
    return create_symlink_safely(source_path, target_path)

def split_dataset(dataset_dir, val_size, test_size, use_absolute_numbers=False, seed=42, max_workers=8):
    """
    Split dataset into train/val/test sets.
    
    Args:
        dataset_dir: Path to dataset directory
        val_size: Size of validation set (percentage or absolute number)
        test_size: Size of test set (percentage or absolute number)
        use_absolute_numbers: If True, val_size and test_size are absolute numbers
        seed: Random seed for reproducibility
        max_workers: Number of parallel workers for symlink creation
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
    
    # Create new directory structure
    logger.info("Creating directory structure...")
    for subdir in ["all", "train", "val", "test"]:
        (images_dir / subdir).mkdir(exist_ok=True)
        (labels_dir / subdir).mkdir(exist_ok=True)
        logger.info(f"  ✓ Created {subdir}/ subdirectories")
    
    # Check if files are already in 'all' directory
    all_images_dir = images_dir / "all"
    all_labels_dir = labels_dir / "all"
    
    logger.info(f"Checking 'all' directories for existing files...")
    all_images_count = len(list(all_images_dir.glob("*.png")))
    all_labels_count = len(list(all_labels_dir.glob("*.txt")))
    logger.info(f"  Found {all_images_count} images and {all_labels_count} labels in 'all' directories")
    
    # If 'all' directories are empty, move files from parent directories
    if all_images_count == 0 and all_labels_count == 0:
        logger.info("Moving files from parent directories to 'all' directories...")
        
        # Count files before moving
        parent_images = [f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() == '.png']
        parent_labels = [f for f in labels_dir.iterdir() if f.is_file() and f.suffix.lower() == '.txt']
        logger.info(f"  Found {len(parent_images)} images and {len(parent_labels)} labels in parent directories")
        
        # Create temp directories and move all files at once using os.rename
        temp_images_dir = images_dir / "temp_images"
        temp_labels_dir = labels_dir / "temp_labels"
        
        # Move all image files to temp directory
        temp_images_dir.mkdir(exist_ok=True)
        for file_path in parent_images:
            os.rename(str(file_path), str(temp_images_dir / file_path.name))
        logger.info(f"  ✓ Moved {len(parent_images)} image files to temp directory")
        
        # Move all label files to temp directory
        temp_labels_dir.mkdir(exist_ok=True)
        for file_path in parent_labels:
            os.rename(str(file_path), str(temp_labels_dir / file_path.name))
        logger.info(f"  ✓ Moved {len(parent_labels)} label files to temp directory")
        
        # Rename temp directories to 'all'
        os.rename(str(temp_images_dir), str(all_images_dir))
        os.rename(str(temp_labels_dir), str(all_labels_dir))
        logger.info("  ✓ Renamed temp directories to 'all'")
    else:
        logger.info("✓ Files already in 'all' directories, skipping move operation")
    
    # Get all image files from 'all' directory
    logger.info("Scanning 'all' directories for files...")
    image_files = list(all_images_dir.glob("*.png"))
    label_files = list(all_labels_dir.glob("*.txt"))
    
    logger.info(f"✓ Found {len(image_files)} image files and {len(label_files)} label files")
    
    # Create a mapping of image files to label files
    logger.info(f"Creating image-label pairs for {len(image_files)} images...")
    file_pairs = []
    missing_labels = 0
    
    # Create a set of available label files for faster lookup
    available_labels = {f.stem for f in label_files}
    logger.info(f"Found {len(available_labels)} unique label files")
    
    # Process images in batches for progress reporting
    batch_size = 1000
    for i, img_file in enumerate(image_files):
        # Show progress every batch_size files
        if i % batch_size == 0:
            logger.info(f"  Processing image {i+1}/{len(image_files)} ({(i+1)/len(image_files)*100:.1f}%)")
        
        # Check if corresponding label exists
        if img_file.stem in available_labels:
            label_file = all_labels_dir / f"{img_file.stem}.txt"
            file_pairs.append((img_file, label_file))
        else:
            missing_labels += 1
            if missing_labels <= 10:  # Only log first 10 missing labels to avoid spam
                logger.warning(f"No label file found for {img_file.name}")
            elif missing_labels == 11:
                logger.warning("... (additional missing labels will not be logged)")
    
    logger.info(f"✓ Found {len(file_pairs)} valid image-label pairs")
    if missing_labels > 0:
        logger.warning(f"⚠ {missing_labels} images have missing label files")
    
    if len(file_pairs) == 0:
        raise ValueError("No valid image-label pairs found")
    
    # Calculate split sizes
    total_files = len(file_pairs)
    logger.info(f"Calculating split sizes for {total_files} total files...")
    
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
    
    # Shuffle file pairs
    logger.info("Shuffling file pairs for random split...")
    random.shuffle(file_pairs)
    logger.info("✓ File pairs shuffled")
    
    # Split into sets
    train_pairs = file_pairs[:train_count]
    val_pairs = file_pairs[train_count:train_count + val_count]
    test_pairs = file_pairs[train_count + val_count:]
    
    # Pre-resolve all symlink targets to avoid repeated filesystem calls
    logger.info(f"Resolving symlink targets for {len(file_pairs)} file pairs...")
    resolved_pairs = []
    unresolved_count = 0
    
    for i, (img_file, label_file) in enumerate(file_pairs):
        # Show progress every 1000 files
        if i % 1000 == 0:
            logger.info(f"  Resolving symlinks {i+1}/{len(file_pairs)} ({(i+1)/len(file_pairs)*100:.1f}%)")
        
        img_target = resolve_symlink_target(img_file)
        label_target = resolve_symlink_target(label_file)
        if img_target and label_target:
            resolved_pairs.append((img_file, label_file, img_target, label_target))
        else:
            unresolved_count += 1
            if unresolved_count <= 5:  # Only log first 5 unresolved symlinks
                logger.warning(f"Skipping {img_file.name} due to unresolved symlinks")
            elif unresolved_count == 6:
                logger.warning("... (additional unresolved symlinks will not be logged)")
    
    logger.info(f"✓ Resolved {len(resolved_pairs)} symlink pairs")
    if unresolved_count > 0:
        logger.warning(f"⚠ {unresolved_count} file pairs have unresolved symlinks")
    
    # Create symlinks for each split using parallel processing
    splits = [
        ("train", train_pairs),
        ("val", val_pairs),
        ("test", test_pairs)
    ]
    
    for split_name, pairs in splits:
        logger.info(f"Creating symlinks for {split_name} set ({len(pairs)} files)...")
        
        # Prepare symlink creation tasks
        logger.info(f"  Preparing symlink tasks for {split_name}...")
        symlink_tasks = []
        tasks_prepared = 0
        
        for img_file, label_file in pairs:
            # Show progress every 500 files
            if tasks_prepared % 500 == 0 and tasks_prepared > 0:
                logger.info(f"    Prepared {tasks_prepared}/{len(pairs)} tasks for {split_name}")
            
            # Find the resolved targets for this pair
            resolved_targets = None
            for r_img, r_label, r_img_target, r_label_target in resolved_pairs:
                if r_img == img_file and r_label == label_file:
                    resolved_targets = (r_img_target, r_label_target)
                    break
            
            if resolved_targets:
                img_target, label_target = resolved_targets
                # Create symlinks in the split directory
                symlink_tasks.append((
                    img_target, 
                    images_dir / split_name / img_file.name
                ))
                symlink_tasks.append((
                    label_target, 
                    labels_dir / split_name / label_file.name
                ))
                tasks_prepared += 1
        
        logger.info(f"  ✓ Prepared {len(symlink_tasks)} symlink tasks for {split_name}")
        
        # Create symlinks in parallel
        if symlink_tasks:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(create_symlink_batch, task): task 
                    for task in symlink_tasks
                }
                
                # Process completed tasks
                completed = 0
                for future in as_completed(future_to_task):
                    completed += 1
                    if completed % 100 == 0:  # Log progress every 100 files
                        logger.info(f"Created {completed}/{len(symlink_tasks)} symlinks for {split_name}")
                    
                    try:
                        success = future.result()
                        if not success:
                            task = future_to_task[future]
                            logger.error(f"Failed to create symlink: {task[1]}")
                    except Exception as e:
                        task = future_to_task[future]
                        logger.error(f"Exception creating symlink {task[1]}: {e}")
    
    # Print summary
    logger.info("Dataset split complete!")
    logger.info(f"Train set: {len(train_pairs)} files")
    logger.info(f"Validation set: {len(val_pairs)} files")
    logger.info(f"Test set: {len(test_pairs)} files")
    
    # Verify the split
    for split_name in ["train", "val", "test"]:
        train_images = len(list((images_dir / split_name).glob("*.png")))
        train_labels = len(list((labels_dir / split_name).glob("*.txt")))
        logger.info(f"{split_name.capitalize()}: {train_images} images, {train_labels} labels")

def main():
    parser = argparse.ArgumentParser(
        description='Split dataset into train/val/test sets by creating symlinks'
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
    parser.add_argument(
        '--max_workers',
        type=int,
        default=8,
        help='Number of parallel workers for symlink creation'
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
        seed=args.seed,
        max_workers=args.max_workers
    )

if __name__ == "__main__":
    main() 