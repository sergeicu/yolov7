import os
import shutil
import argparse
import logging
import multiprocessing as mp
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
from pathlib import Path


def setup_logging(log_file):
    """Setup logging configuration with both file and console handlers."""
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def get_all_png_files(source_dir):
    """Get all PNG files from the source directory tree."""
    png_files = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.png'):
                source_file = os.path.join(root, file)
                png_files.append(source_file)
    return png_files


def copy_single_file(args):
    """Copy a single PNG file. Designed to be used with multiprocessing."""
    source_file, dest_dir, file_lock = args
    
    try:
        # Extract filename from source path
        filename = os.path.basename(source_file)
        dest_file = os.path.join(dest_dir, filename)
        
        # Check if file already exists
        if os.path.exists(dest_file):
            return {'status': 'skipped', 'file': source_file, 'error': None}
        
        # Copy the file
        shutil.copy2(source_file, dest_file)
        return {'status': 'copied', 'file': source_file, 'error': None}
        
    except Exception as e:
        return {'status': 'error', 'file': source_file, 'error': str(e)}


def copy_png_files_parallel(source_dir, dest_dir, logger, num_workers=None):
    """Copy PNG files to the destination directory using parallel processing."""
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        logger.info(f"Created destination directory: {dest_dir}")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid overwhelming the system
    
    logger.info(f"Starting parallel PNG file extraction from: {source_dir}")
    logger.info(f"Destination directory: {dest_dir}")
    logger.info(f"Using {num_workers} worker processes")
    
    # Get all PNG files first
    logger.info("Scanning for PNG files...")
    png_files = get_all_png_files(source_dir)
    logger.info(f"Found {len(png_files)} PNG files to process")
    
    if not png_files:
        logger.info("No PNG files found in source directory")
        return
    
    # Statistics tracking
    total_files_found = len(png_files)
    total_files_copied = 0
    total_files_skipped = 0
    total_files_error = 0
    
    # Create a lock for thread-safe operations
    file_lock = threading.Lock()
    
    # Prepare arguments for parallel processing
    copy_args = [(file_path, dest_dir, file_lock) for file_path in png_files]
    
    # Process files in parallel with progress bar
    logger.info("Starting parallel file copying...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(copy_single_file, args): args[0] for args in copy_args}
        
        # Process completed tasks with progress bar
        with tqdm(total=len(png_files), desc="Copying files") as pbar:
            for future in as_completed(future_to_file):
                result = future.result()
                
                with file_lock:
                    if result['status'] == 'copied':
                        total_files_copied += 1
                    elif result['status'] == 'skipped':
                        total_files_skipped += 1
                    elif result['status'] == 'error':
                        total_files_error += 1
                        logger.error(f"Failed to copy {result['file']}: {result['error']}")
                
                pbar.update(1)
                
                # Log progress every 1000 files
                if (total_files_copied + total_files_skipped + total_files_error) % 1000 == 0:
                    logger.info(f"Progress: {total_files_copied} copied, "
                               f"{total_files_skipped} skipped, "
                               f"{total_files_error} errors")
    
    # Final summary
    logger.info("=" * 60)
    logger.info("PARALLEL EXTRACTION COMPLETE - FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total PNG files found: {total_files_found}")
    logger.info(f"Files successfully copied: {total_files_copied}")
    logger.info(f"Files skipped (already exist): {total_files_skipped}")
    logger.info(f"Files with errors: {total_files_error}")
    if total_files_copied + total_files_skipped > 0:
        logger.info(f"Success rate: {total_files_copied/(total_files_copied + total_files_skipped)*100:.1f}%")
    logger.info(f"Used {num_workers} worker processes")
    logger.info("=" * 60)


def copy_png_files_chunked(source_dir, dest_dir, logger, num_workers=None, chunk_size=1000):
    """Alternative implementation using chunked processing for better memory management."""
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        logger.info(f"Created destination directory: {dest_dir}")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)
    
    logger.info(f"Starting chunked parallel PNG file extraction from: {source_dir}")
    logger.info(f"Destination directory: {dest_dir}")
    logger.info(f"Using {num_workers} worker processes with chunk size {chunk_size}")
    
    # Get all PNG files first
    logger.info("Scanning for PNG files...")
    png_files = get_all_png_files(source_dir)
    logger.info(f"Found {len(png_files)} PNG files to process")
    
    if not png_files:
        logger.info("No PNG files found in source directory")
        return
    
    # Statistics tracking
    total_files_found = len(png_files)
    total_files_copied = 0
    total_files_skipped = 0
    total_files_error = 0
    
    # Process files in chunks
    for i in range(0, len(png_files), chunk_size):
        chunk = png_files[i:i + chunk_size]
        logger.info(f"Processing chunk {i//chunk_size + 1}/{(len(png_files) + chunk_size - 1)//chunk_size} "
                   f"({len(chunk)} files)")
        
        # Create a lock for thread-safe operations
        file_lock = threading.Lock()
        
        # Prepare arguments for parallel processing
        copy_args = [(file_path, dest_dir, file_lock) for file_path in chunk]
        
        # Process chunk in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_file = {executor.submit(copy_single_file, args): args[0] for args in copy_args}
            
            for future in as_completed(future_to_file):
                result = future.result()
                
                with file_lock:
                    if result['status'] == 'copied':
                        total_files_copied += 1
                    elif result['status'] == 'skipped':
                        total_files_skipped += 1
                    elif result['status'] == 'error':
                        total_files_error += 1
                        logger.error(f"Failed to copy {result['file']}: {result['error']}")
        
        # Log chunk progress
        logger.info(f"Chunk complete: {total_files_copied} copied, "
                   f"{total_files_skipped} skipped, "
                   f"{total_files_error} errors so far")
    
    # Final summary
    logger.info("=" * 60)
    logger.info("CHUNKED PARALLEL EXTRACTION COMPLETE - FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total PNG files found: {total_files_found}")
    logger.info(f"Files successfully copied: {total_files_copied}")
    logger.info(f"Files skipped (already exist): {total_files_skipped}")
    logger.info(f"Files with errors: {total_files_error}")
    if total_files_copied + total_files_skipped > 0:
        logger.info(f"Success rate: {total_files_copied/(total_files_copied + total_files_skipped)*100:.1f}%")
    logger.info(f"Used {num_workers} worker processes with chunk size {chunk_size}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy all PNG files to a common directory using parallel processing.")
    parser.add_argument("--source_dir", 
                       default='/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/extraction_aug_2025/dcm/',
                       help="Directory containing subdirectories with PNG files.")
    parser.add_argument("--dest_dir", 
                       default='/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/pngs_aug2025_fixed/',
                       help="Directory to copy all PNG files to.")
    parser.add_argument("--log_file", 
                       default=None,
                       help="Path to log file (default: logs/png_extraction_parallel_YYYYMMDD_HHMMSS.log)")
    parser.add_argument("--num_workers", 
                       type=int, 
                       default=None,
                       help="Number of worker processes (default: min(CPU_count, 8))")
    parser.add_argument("--chunk_size", 
                       type=int, 
                       default=1000,
                       help="Chunk size for processing (default: 1000)")
    parser.add_argument("--use_chunked", 
                       action="store_true",
                       help="Use chunked processing for better memory management")
    
    args = parser.parse_args()
    
    # Generate log file name if not provided
    if args.log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = f"logs/png_extraction_parallel_{timestamp}.log"
    
    # Setup logging
    logger = setup_logging(args.log_file)
    
    # Start the extraction process
    if args.use_chunked:
        copy_png_files_chunked(args.source_dir, args.dest_dir, logger, args.num_workers, args.chunk_size)
    else:
        copy_png_files_parallel(args.source_dir, args.dest_dir, logger, args.num_workers)