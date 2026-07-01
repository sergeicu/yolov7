import os
import logging
from datetime import datetime


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


def remove_empty_pngs(root_dir, logger):
    """Remove empty PNG files from the directory using efficient file scanning."""
    
    if not os.path.exists(root_dir):
        logger.error(f"Directory does not exist: {root_dir}")
        return
    
    logger.info(f"Starting empty PNG removal from: {root_dir}")
    
    # Statistics tracking
    total_files_scanned = 0
    total_png_files = 0
    empty_files_found = 0
    empty_files_removed = 0
    errors_encountered = 0
    
    try:
        # Use os.scandir() for more efficient directory scanning
        with os.scandir(root_dir) as entries:
            for entry in entries:
                total_files_scanned += 1
                
                # Check if it's a PNG file
                if entry.name.lower().endswith('.png'):
                    total_png_files += 1
                    
                    try:
                        # Check file size
                        if entry.stat().st_size == 0:
                            empty_files_found += 1
                            logger.info(f"Found empty file: {entry.path}")
                            
                            # Remove the empty file
                            os.remove(entry.path)
                            empty_files_removed += 1
                            logger.info(f"Removed empty file: {entry.path}")
                            
                    except OSError as e:
                        errors_encountered += 1
                        logger.error(f"Error processing {entry.path}: {str(e)}")
                
                # Log progress every 1000 files scanned
                if total_files_scanned % 1000 == 0:
                    logger.info(f"Progress: {total_files_scanned} files scanned, "
                               f"{total_png_files} PNG files found, "
                               f"{empty_files_removed} empty files removed")
    
    except Exception as e:
        logger.error(f"Error scanning directory {root_dir}: {str(e)}")
        return
    
    # Final summary
    logger.info("=" * 60)
    logger.info("EMPTY PNG REMOVAL COMPLETE - FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total files scanned: {total_files_scanned}")
    logger.info(f"PNG files found: {total_png_files}")
    logger.info(f"Empty PNG files found: {empty_files_found}")
    logger.info(f"Empty PNG files removed: {empty_files_removed}")
    logger.info(f"Errors encountered: {errors_encountered}")
    if total_png_files > 0:
        logger.info(f"Empty PNG percentage: {empty_files_found/total_png_files*100:.1f}%")
    logger.info("=" * 60)


if __name__ == "__main__":
    root_dir = '/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/pngs_aug2025/'
    
    # Generate log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/empty_png_removal_{timestamp}.log"
    
    # Setup logging
    logger = setup_logging(log_file)
    
    # Start the removal process
    remove_empty_pngs(root_dir, logger)
