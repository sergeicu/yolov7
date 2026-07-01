import os
import pandas as pd
import subprocess
from tqdm import tqdm
import argparse
import shutil
import random
import multiprocessing as mp
from multiprocessing import Pool, Manager
import threading
import logging
import time
from datetime import datetime


def setup_logger(name, log_file, level=logging.INFO):
    """Set up a logger with file handler"""
    formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
    
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger

def convert_dicom_to_png(dcm_path, png_path, overwrite=False, success_logger=None, error_logger=None):
    cmd = [
        'python', 
        '/home/ch215616/w/code/llm/experiments/yolov7/dicom-to-png/mritopng.py',
        dcm_path,
        png_path,
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if success_logger:
            success_logger.info(f"Successfully converted: {dcm_path} -> {png_path}")
        return True
    except subprocess.CalledProcessError as e:
        if error_logger:
            error_logger.error(f"Failed to convert: {dcm_path} -> {png_path}. Error: {str(e)}")
        if os.path.exists(png_path):
            os.remove(png_path)
        return False
            
            
def extract_medical_report(report_dcm_path, output_path, success_logger=None, error_logger=None):
    try:
        subprocess.run([
            'dcmdump',
            '+L',
            '+P', '0040,a160',
            report_dcm_path,
        ], stdout=open(output_path, 'w'), check=True)
        if success_logger:
            success_logger.info(f"Successfully extracted report: {report_dcm_path} -> {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        if error_logger:
            error_logger.error(f"Failed to extract report: {report_dcm_path} -> {output_path}. Error: {str(e)}")
        return False

def process_scan_id_directory(args):
    scan_id_path, scan_id, overwrite, no_report_log, no_png_log, process_id = args
    
    # Set up process-specific loggers
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    success_logger = setup_logger(f'success_{process_id}', f'logs/success_conversions_{timestamp}.log')
    error_logger = setup_logger(f'error_{process_id}', f'logs/failed_conversions_{timestamp}.log')
    
    results = []
    png_count = 0
    conversion_success_count = 0
    conversion_fail_count = 0

    for root, dirs, files in os.walk(scan_id_path):
        for file in files:
            if file.lower().startswith('dx'):
                if '~' in file:
                    # Remove files with '~' in the name
                    os.remove(os.path.join(root, file))
                    continue

                dcm_path = os.path.join(root, file)
                scan_type = os.path.basename(os.path.dirname(dcm_path))
                png_filename = f"{scan_id}-{scan_type}-{png_count}.png"
                png_path = os.path.join(scan_id_path, png_filename)
                
                if os.path.exists(png_path):
                    if os.path.getsize(png_path) == 0:
                        os.remove(png_path)
                        error_logger.warning(f"Removed empty file: {png_path}. Re-processing")
                    
                if not os.path.exists(png_path):
                    success = convert_dicom_to_png(dcm_path, png_path, overwrite, success_logger, error_logger)
                    if success:
                        conversion_success_count += 1
                    else:
                        conversion_fail_count += 1
                
                results.append({
                    'scan_id': scan_id,
                    'png_file': png_filename,
                    'original_dcm': os.path.relpath(dcm_path, scan_id_path),
                    'scan_type': scan_type
                })
                
                png_count += 1

    if png_count == 0 and no_png_log:
        with open(no_png_log, 'a') as f:
            f.write(f"{scan_id}\n")
        error_logger.warning(f"No PNG files generated for scan_id: {scan_id}")

    # Find and extract medical report
    report_found = False
    for root, dirs, files in os.walk(scan_id_path):
        if any(folder.startswith('999') for folder in root.split(os.sep)):
            for file in files:
                report_dcm_path = os.path.join(root, file)
                if '~' in report_dcm_path:
                    # Remove files with '~' in the name
                    os.remove(report_dcm_path)
                    continue
                
                report_txt_path = os.path.join(scan_id_path, f"{scan_id}_report.txt")
                if extract_medical_report(report_dcm_path, report_txt_path, success_logger, error_logger):
                    report_found = True
                    break
            if report_found:
                break

    if not report_found and no_report_log:
        with open(no_report_log, 'a') as f:
            f.write(f"{scan_id}\n")
        error_logger.warning(f"No medical report found for scan_id: {scan_id}")
    
    # Log summary for this scan_id
    success_logger.info(f"Scan {scan_id} completed - Success: {conversion_success_count}, Failed: {conversion_fail_count}, Total: {png_count}")
    
    return results

def main(main_dir, overwrite=False, num_processes=None):
    if num_processes is None:
        # Use all cores minus 2, but ensure at least 1 process
        num_processes = max(1, mp.cpu_count() - 2)
    
    print(f"Using {num_processes} processes (out of {mp.cpu_count()} available cores)")
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    no_report_log = 'no_report_log.txt'
    no_png_log = 'no_png_log.txt'

    # Clear log files if they exist
    open(no_report_log, 'w').close()
    open(no_png_log, 'w').close()

    # Get all scan IDs more efficiently for large numbers of directories
    print("Scanning directories...")
    items = []
    for item in os.scandir(main_dir):
        if item.is_dir() and item.name[0].isdigit():
            items.append(item.name)
    
    print(f"Found {len(items)} scan directories. Starting processing...")
    
    # Prepare arguments for parallel processing
    process_args = []
    for i, scan_id in enumerate(items):
        scan_id_path = os.path.join(main_dir, scan_id)
        process_args.append((scan_id_path, scan_id, overwrite, no_report_log, no_png_log, i))
    
    # Process in parallel
    all_results = []
    start_time = time.time()
    
    with Pool(processes=num_processes) as pool:
        # Use tqdm to show progress
        print(f"Starting parallel processing")
        for results in tqdm(pool.imap(process_scan_id_directory, process_args), total=len(process_args)):
            all_results.extend(results)
    
    end_time = time.time()
    
    # Create DataFrame from all results
    df = pd.DataFrame(all_results)
    
    # Save the DataFrame to a CSV file
    df.to_csv('dataset_info.csv', index=False)
    
    # Log final summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_logger = setup_logger('summary', f'logs/processing_summary_{timestamp}.log')
    summary_logger.info(f"Processing completed in {end_time - start_time:.2f} seconds")
    summary_logger.info(f"Total scan IDs processed: {len(items)}")
    summary_logger.info(f"Total PNG files generated: {len(all_results)}")
    summary_logger.info(f"Results saved to: dataset_info.csv")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DICOM files to PNG and extract medical reports.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PNG files")
    parser.add_argument("--num-processes", type=int, default=None, 
                       help=f"Number of processes to use (default: {max(1, mp.cpu_count() - 2)} - all cores minus 2)")
    args = parser.parse_args()

    main_directory = '/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/extraction_aug_2025/dcm'
    main(main_directory, args.overwrite, args.num_processes)

print("Processing complete. Results saved in dataset_info.csv")
print("Scan IDs with no report logged in no_report_log.txt")
print("Scan IDs with no PNG files logged in no_png_log.txt")
print("Detailed logs saved in logs/ directory")