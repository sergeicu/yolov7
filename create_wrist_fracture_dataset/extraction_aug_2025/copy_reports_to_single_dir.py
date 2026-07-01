import os
import shutil
import argparse
import logging
from datetime import datetime
from tqdm import tqdm


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


def copy_report_files(source_dir, dest_dir, logger):
    """Copy report files to the destination directory using os.walk, skipping existing ones."""
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        logger.info(f"Created destination directory: {dest_dir}")
    
    # Statistics tracking
    total_folders_processed = 0
    total_folders_with_files = 0
    total_files_found = 0
    total_files_copied = 0
    total_files_skipped = 0
    empty_folders = 0
    
    logger.info(f"Starting report file extraction from: {source_dir}")
    logger.info(f"Destination directory: {dest_dir}")
    
    # Use os.walk to traverse the directory tree
    for root, dirs, files in os.walk(source_dir):
        total_folders_processed += 1
        
        # Filter for report files in current directory
        report_files = [f for f in files if f.lower().endswith('_report.txt')]
        
        if not report_files:
            empty_folders += 1
            if total_folders_processed % 1000 == 0:  # Log every 1000 folders
                logger.info(f"Processed {total_folders_processed} folders, {empty_folders} empty")
            continue
        
        total_folders_with_files += 1
        total_files_found += len(report_files)
        
        # Process report files in current directory
        for report_file in report_files:
            source_file = os.path.join(root, report_file)
            dest_file = os.path.join(dest_dir, report_file)
            
            if os.path.exists(dest_file):
                total_files_skipped += 1
            else:
                try:
                    shutil.copy2(source_file, dest_file)
                    total_files_copied += 1
                except Exception as e:
                    logger.error(f"Failed to copy {source_file}: {str(e)}")
        
        # Log progress every 1000 folders with files
        if total_folders_with_files % 1000 == 0:
            logger.info(f"Progress: {total_folders_processed} folders processed, "
                       f"{total_folders_with_files} with files, "
                       f"{total_files_copied} files copied, "
                       f"{total_files_skipped} files skipped")
    
    # Final summary
    logger.info("=" * 60)
    logger.info("REPORT EXTRACTION COMPLETE - FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total folders processed: {total_folders_processed}")
    logger.info(f"Folders with report files: {total_folders_with_files}")
    logger.info(f"Empty folders: {empty_folders}")
    logger.info(f"Total report files found: {total_files_found}")
    logger.info(f"Files successfully copied: {total_files_copied}")
    logger.info(f"Files skipped (already exist): {total_files_skipped}")
    if total_files_copied + total_files_skipped > 0:
        logger.info(f"Success rate: {total_files_copied/(total_files_copied + total_files_skipped)*100:.1f}%")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy all report files to a common directory using os.walk, skipping existing ones.")
    parser.add_argument("--source_dir", 
                       default='/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/extraction_aug_2025/dcm/',
                       help="Directory containing subdirectories with PNG and report files.")
    parser.add_argument("--dest_dir", 
                       default='/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/reports_aug2025/',
                       help="Directory to copy all reports files to.")
    parser.add_argument("--log_file", 
                       default=None,
                       help="Path to log file (default: logs/report_extraction_YYYYMMDD_HHMMSS.log)")
    
    args = parser.parse_args()
    
    # Generate log file name if not provided
    if args.log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = f"logs/report_extraction_{timestamp}.log"
    
    # Setup logging
    logger = setup_logging(args.log_file)
    
    # Start the extraction process
    copy_report_files(args.source_dir, args.dest_dir, logger)

