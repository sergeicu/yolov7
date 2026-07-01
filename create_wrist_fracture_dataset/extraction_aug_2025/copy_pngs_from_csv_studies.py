import os
import shutil
import argparse
import logging
import multiprocessing as mp
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import pandas as pd
from pathlib import Path
import json
import pickle


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


def read_study_ids_from_csv(csv_file, logger, quicktest=False, start_idx=None, end_idx=None):
    """Read study IDs from the 'Accession Number' column in the CSV file."""
    try:
        logger.info(f"Reading study IDs from CSV file: {csv_file}")
        df = pd.read_csv(csv_file)
        
        if 'Accession Number' not in df.columns:
            logger.error("'Accession Number' column not found in CSV file")
            logger.info(f"Available columns: {list(df.columns)}")
            return []
        
        study_ids = df['Accession Number'].dropna().unique().tolist()
        logger.info(f"Found {len(study_ids)} unique study IDs in CSV file")
        
        # Convert to strings and remove any whitespace
        study_ids = [str(sid).strip() for sid in study_ids if str(sid).strip()]
        logger.info(f"After cleaning: {len(study_ids)} valid study IDs")
        
        # Apply range slicing if start_idx and/or end_idx are specified
        if start_idx is not None or end_idx is not None:
            original_count = len(study_ids)
            start = start_idx if start_idx is not None else 0
            end = end_idx if end_idx is not None else len(study_ids)
            
            # Validate indices
            if start < 0:
                start = 0
            if end > len(study_ids):
                end = len(study_ids)
            if start >= end:
                logger.error(f"Invalid range: start ({start}) >= end ({end})")
                return []
            
            study_ids = study_ids[start:end]
            logger.info(f"RANGE MODE: Limited to study IDs {start} to {end-1} (from {original_count} total)")
        
        # Apply quicktest limit if requested (after range slicing)
        elif quicktest and len(study_ids) > 1000:
            original_count = len(study_ids)
            study_ids = study_ids[:1000]
            logger.info(f"QUICKTEST MODE: Limited to first 1000 study IDs (from {original_count} total)")
        
        return study_ids
        
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return []


def get_png_files_from_study_fixed_folder(root_dir, study_id, logger):
    """Get all PNG files from the 'fixed' subdirectory of a specific study folder."""
    fixed_folder_path = os.path.join(root_dir, study_id, 'fixed')
    png_files = []
    
    if not os.path.exists(fixed_folder_path):
        logger.warning(f"Fixed folder not found for study {study_id}: {fixed_folder_path}")
        return png_files
    
    try:
        for file in os.listdir(fixed_folder_path):
            if file.lower().endswith('.png'):
                png_file_path = os.path.join(fixed_folder_path, file)
                png_files.append(png_file_path)
        
        if png_files:
            logger.debug(f"Found {len(png_files)} PNG files in study {study_id}")
        else:
            logger.warning(f"No PNG files found in study {study_id}")
            
    except Exception as e:
        logger.error(f"Error accessing fixed folder for study {study_id}: {e}")
    
    return png_files


def test_file_creation(file_path, logger):
    """Test if we can create a file at the specified path."""
    try:
        if not file_path or not file_path.strip():
            logger.error("Invalid file path: empty or None")
            return False
        
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Try to create an empty file
        with open(file_path, 'w') as f:
            pass  # Just create an empty file
        
        # Remove the test file
        os.remove(file_path)
        
        logger.info(f"Successfully tested file creation at: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create test file at {file_path}: {e}")
        return False


def save_file_list_to_json(file_list, output_file, logger):
    """Save the list of PNG files to a JSON file."""
    try:
        # Test file creation first
        if not test_file_creation(output_file, logger):
            return False
        
        # Save as JSON
        with open(output_file, 'w') as f:
            json.dump(file_list, f, indent=2)
        
        logger.info(f"Saved {len(file_list)} file paths to: {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving file list to JSON: {e}")
        return False


def save_file_list_to_txt(file_list, output_file, logger):
    """Save the list of PNG files to a text file (one path per line)."""
    try:
        # Test file creation first
        if not test_file_creation(output_file, logger):
            return False
        
        # Save as text file
        with open(output_file, 'w') as f:
            for file_path in file_list:
                f.write(f"{file_path}\n")
        
        logger.info(f"Saved {len(file_list)} file paths to: {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving file list to text file: {e}")
        return False


def read_file_list_from_json(input_file, logger):
    """Read the list of PNG files from a JSON file."""
    try:
        with open(input_file, 'r') as f:
            file_list = json.load(f)
        logger.info(f"Loaded {len(file_list)} file paths from: {input_file}")
        return file_list
    except Exception as e:
        logger.error(f"Error reading file list from JSON: {e}")
        return []


def read_file_list_from_txt(input_file, logger):
    """Read the list of PNG files from a text file (one path per line)."""
    try:
        file_list = []
        with open(input_file, 'r') as f:
            for line in f:
                file_path = line.strip()
                if file_path:  # Skip empty lines
                    file_list.append(file_path)
        logger.info(f"Loaded {len(file_list)} file paths from: {input_file}")
        return file_list
    except Exception as e:
        logger.error(f"Error reading file list from text file: {e}")
        return []


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


def copy_files_inline(all_png_files, dest_dir, logger):
    """Copy files sequentially (inline) instead of using parallel processing."""
    total_files_found = len(all_png_files)
    total_files_copied = 0
    total_files_skipped = 0
    total_files_error = 0
    
    logger.info("Starting inline file copying...")
    
    with tqdm(total=len(all_png_files), desc="Copying files") as pbar:
        for file_path in all_png_files:
            try:
                # Extract filename from source path
                filename = os.path.basename(file_path)
                dest_file = os.path.join(dest_dir, filename)
                
                # Check if file already exists
                if os.path.exists(dest_file):
                    total_files_skipped += 1
                    logger.debug(f"Skipped (already exists): {file_path}")
                else:
                    # Copy the file
                    shutil.copy2(file_path, dest_file)
                    total_files_copied += 1
                    logger.debug(f"Copied: {file_path}")
                
            except Exception as e:
                total_files_error += 1
                logger.error(f"Failed to copy {file_path}: {e}")
            
            pbar.update(1)
            
            # Log progress every 1000 files
            if (total_files_copied + total_files_skipped + total_files_error) % 1000 == 0:
                logger.info(f"Progress: {total_files_copied} copied, "
                           f"{total_files_skipped} skipped, "
                           f"{total_files_error} errors")
    
    return total_files_copied, total_files_skipped, total_files_error


def copy_pngs_from_csv_studies(csv_file, root_dir, dest_dir, logger, num_workers=None, 
                              save_file_list=None, read_file_list=None, quicktest=False, 
                              inline=False, start_idx=None, end_idx=None):
    """Copy PNG files from study folders based on CSV file using parallel processing."""
    
    # Test file creation at the start if save_file_list is specified
    if save_file_list and save_file_list.strip():
        logger.info(f"Testing file creation for save_file_list: {save_file_list}")
        if not test_file_creation(save_file_list, logger):
            logger.error("Failed to test file creation. Exiting.")
            return
    
    # Create destination directory
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        logger.info(f"Created destination directory: {dest_dir}")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)
    
    # Get the list of PNG files
    if read_file_list:
        # Read from existing file list
        logger.info(f"Reading file list from: {read_file_list}")
        if read_file_list.endswith('.json'):
            all_png_files = read_file_list_from_json(read_file_list, logger)
        elif read_file_list.endswith('.txt'):
            all_png_files = read_file_list_from_txt(read_file_list, logger)
        else:
            logger.error("Unsupported file format. Use .json or .txt files.")
            return
        
        if not all_png_files:
            logger.error("No files found in the provided file list.")
            return
            
        studies_with_pngs = "N/A (read from file list)"
        studies_without_pngs = "N/A (read from file list)"
        
    else:
        # Read study IDs from CSV and scan for PNG files
        study_ids = read_study_ids_from_csv(csv_file, logger, quicktest, start_idx, end_idx)
        if not study_ids:
            logger.error("No study IDs found in CSV file. Exiting.")
            return
        
        logger.info(f"Starting PNG extraction from {len(study_ids)} study folders")
        logger.info(f"Root directory: {root_dir}")
        logger.info(f"Destination directory: {dest_dir}")
        if inline:
            logger.info("Using inline (sequential) file copying")
        else:
            logger.info(f"Using {num_workers} worker processes for parallel copying")
        if quicktest:
            logger.info("QUICKTEST MODE: Processing only first 1000 study IDs")
        if start_idx is not None or end_idx is not None:
            start = start_idx if start_idx is not None else 0
            end = end_idx if end_idx is not None else "end"
            logger.info(f"RANGE MODE: Processing study IDs {start} to {end}")
        
        # Collect all PNG files from all study folders
        all_png_files = []
        studies_with_pngs = 0
        studies_without_pngs = 0
        
        logger.info("Scanning study folders for PNG files...")
        for study_id in tqdm(study_ids, desc="Scanning studies"):
            png_files = get_png_files_from_study_fixed_folder(root_dir, study_id, logger)
            if png_files:
                all_png_files.extend(png_files)
                studies_with_pngs += 1
            else:
                studies_without_pngs += 1
    
    logger.info(f"Found {len(all_png_files)} PNG files from {studies_with_pngs} studies")
    logger.info(f"Studies without PNG files: {studies_without_pngs}")
    
    if not all_png_files:
        logger.info("No PNG files found in any study folders")
        return
    
    # Save file list if requested
    if save_file_list and save_file_list.strip():
        if save_file_list.endswith('.json'):
            save_file_list_to_json(all_png_files, save_file_list, logger)
        elif save_file_list.endswith('.txt'):
            save_file_list_to_txt(all_png_files, save_file_list, logger)
        else:
            logger.warning("Unsupported file format for saving. Use .json or .txt extension.")
    
    # Copy files using either inline or parallel processing
    if inline:
        # Use inline (sequential) copying
        total_files_copied, total_files_skipped, total_files_error = copy_files_inline(
            all_png_files, dest_dir, logger)
    else:
        # Use parallel processing
        # Statistics tracking
        total_files_found = len(all_png_files)
        total_files_copied = 0
        total_files_skipped = 0
        total_files_error = 0
        
        # Create a lock for thread-safe operations
        file_lock = threading.Lock()
        
        # Prepare arguments for parallel processing
        copy_args = [(file_path, dest_dir, file_lock) for file_path in all_png_files]
        
        # Process files in parallel with progress bar
        logger.info("Starting parallel file copying...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(copy_single_file, args): args[0] for args in copy_args}
            
            # Process completed tasks with progress bar
            with tqdm(total=len(all_png_files), desc="Copying files") as pbar:
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
    logger.info("CSV-BASED PNG EXTRACTION COMPLETE - FINAL SUMMARY")
    logger.info("=" * 60)
    if not read_file_list:
        logger.info(f"Total study IDs from CSV: {len(study_ids)}")
    if quicktest:
        logger.info("QUICKTEST MODE: Processed only first 1000 study IDs")
    if start_idx is not None or end_idx is not None:
        start = start_idx if start_idx is not None else 0
        end = end_idx if end_idx is not None else "end"
        logger.info(f"RANGE MODE: Processed study IDs {start} to {end}")
    logger.info(f"Studies with PNG files: {studies_with_pngs}")
    logger.info(f"Studies without PNG files: {studies_without_pngs}")
    logger.info(f"Total PNG files found: {len(all_png_files)}")
    logger.info(f"Files successfully copied: {total_files_copied}")
    logger.info(f"Files skipped (already exist): {total_files_skipped}")
    logger.info(f"Files with errors: {total_files_error}")
    if total_files_copied + total_files_skipped > 0:
        logger.info(f"Success rate: {total_files_copied/(total_files_copied + total_files_skipped)*100:.1f}%")
    if inline:
        logger.info("Used inline (sequential) file copying")
    else:
        logger.info(f"Used {num_workers} worker processes")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy PNG files from study folders based on CSV file using parallel processing.")
    parser.add_argument("--csv_file", 
                       default='XRWRIST.csv',
                       help="CSV file containing study IDs in 'Accession Number' column.")
    parser.add_argument("--root_dir", 
                       default='/home/ch215616/ww/code/llm/experiments/yolov7/create_wrist_fracture_dataset/extraction_aug_2025/dcm/',
                       help="Root directory containing study folders.")
    parser.add_argument("--dest_dir", 
                       default='/home/ch215616/ww/code/llm/experiments/yolov7/create_wrist_fracture_dataset/pngs_aug2025_fixed/',
                       help="Directory to copy all PNG files to.")
    parser.add_argument("--log_file", 
                       default=None,
                       help="Path to log file (default: logs/png_extraction_csv_YYYYMMDD_HHMMSS.log)")
    parser.add_argument("--num_workers", 
                       type=int, 
                       default=None,
                       help="Number of worker processes (default: min(CPU_count, 8))")
    parser.add_argument("--save_file_list", 
                       default=None,
                       help="Save the list of PNG files to a file (.json or .txt format)")
    parser.add_argument("--read_file_list", 
                       default=None,
                       help="Read the list of PNG files from a file (.json or .txt format) instead of scanning CSV")
    parser.add_argument("--quicktest", 
                       action="store_true",
                       help="Process only the first 1000 study IDs for quick testing")
    parser.add_argument("--inline", 
                       action="store_true",
                       help="Use inline (sequential) file copying instead of parallel processing")
    parser.add_argument("--start", 
                       type=int, 
                       default=None,
                       help="Starting index for study IDs (0-based)")
    parser.add_argument("--end", 
                       type=int, 
                       default=None,
                       help="Ending index for study IDs (exclusive, 0-based)")
    
    args = parser.parse_args()
    
    # Generate log file name if not provided
    if args.log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = f"logs/png_extraction_csv_{timestamp}.log"
    
    # Setup logging
    logger = setup_logging(args.log_file)
    
    # Start the extraction process
    copy_pngs_from_csv_studies(args.csv_file, args.root_dir, args.dest_dir, logger, 
                              args.num_workers, args.save_file_list, args.read_file_list, 
                              args.quicktest, args.inline, args.start, args.end) 