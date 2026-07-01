#!/usr/bin/env python3
"""
Create symlinks for images and labels based on CSV input with fracture counts

This script takes a CSV file with report IDs and fracture counts, then:
1. Finds corresponding PNG images using regex pattern <report>-*.png
2. Checks if fracture counts match between CSV and YOLO label files
3. Creates symlinks for matching cases or empty label files for zero fractures
4. Logs missing PNGs and mismatched fracture counts to separate CSV files

USAGE:
    python grounding_report_split_regions_s20250821_create_symlinks.py \
        --input_csv ready_for_symlinks_salter_harris_acute.csv \
        --image_dir /path/to/images \
        --label_dir /path/to/labels \
        --output_dir /path/to/output

INPUT:
    - CSV file with 'report' and 'num_fractures' columns
    - Directory containing PNG images
    - Directory containing YOLO label files (.txt)

OUTPUT:
    - Organized training dataset with symlinks
    - CSV files logging missing PNGs and mismatched fracture counts
"""

import os
import shutil
import glob
import pandas as pd
import argparse
from pathlib import Path
import re

def setup_directories(output_dir):
    """Create the required directory structure"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

def create_symlink(src, dst):
    """Create a symlink with robust error handling"""
    # Convert to absolute paths
    src = Path(src).absolute()
    
    # If source is a symlink, get its ultimate target
    if os.path.islink(str(src)):
        try:
            real_src = os.path.realpath(str(src))
            src = Path(real_src)
        except Exception as e:
            print(f"Error resolving source symlink {src}: {e}")
            return False
    
    # Convert to string and handle lab-share path
    src = str(src).replace('/fileserver/', '/lab-share/')
    dst = str(Path(dst).absolute())
    
    # Check if source file exists
    if not os.path.exists(src):
        print(f"Warning: Source file does not exist: {src}")
        return False
        
    try:
        # Remove existing symlink or file if it exists
        if os.path.exists(dst) or os.path.islink(dst):
            os.unlink(dst)
        
        # Create the symlink
        os.symlink(src, dst)
        return True
    except OSError as e:
        print(f"Error creating symlink from {src} to {dst}: {e}")
        try:
            if os.path.exists(dst) or os.path.islink(dst):
                os.remove(dst)
            os.symlink(src, dst)
            return True
        except Exception as e:
            print(f"Failed second attempt: {e}")
            return False

def count_fractures_in_label(label_file_path):
    """Count number of fractures in a YOLO label file by counting non-empty lines"""
    try:
        with open(label_file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        return len(lines)
    except FileNotFoundError:
        print(f"Label file not found: {label_file_path}")
        return 0
    except Exception as e:
        print(f"Error reading label file {label_file_path}: {e}")
        return 0

def find_png_files_for_report(image_dir, report_id):
    """Find all PNG files matching the pattern <report>-*.png"""
    pattern = os.path.join(image_dir, f"{report_id}-*.png")
    png_files = glob.glob(pattern)
    return sorted(png_files)  # Sort for consistent ordering

def process_reports(input_csv_path, image_dir, label_dir, output_dir):
    """Process all reports in the CSV file"""
    
    # Setup output directories
    setup_directories(output_dir)
    output_images_dir = os.path.join(output_dir, 'images')
    output_labels_dir = os.path.join(output_dir, 'labels')
    
    # Read input CSV
    print(f"Reading input CSV: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    print(f"Found {len(df)} reports to process")
    
    # Initialize tracking lists for logging
    missing_pngs = []
    wrong_yolo_results = []
    
    # Statistics counters
    total_reports = 0
    total_images_processed = 0
    successful_symlinks = 0
    empty_labels_created = 0
    
    # Process each report
    for idx, row in df.iterrows():
        report_id = str(row['report'])
        expected_fractures = int(row['num_fractures'])
        
        print(f"\nProcessing report {report_id} (expected fractures: {expected_fractures})")
        total_reports += 1
        
        # Find all PNG files for this report
        png_files = find_png_files_for_report(image_dir, report_id)
        
        if not png_files:
            print(f"No PNG files found for report {report_id}")
            missing_pngs.append({
                'report': report_id,
                'expected_fractures': expected_fractures,
                'image_dir': image_dir
            })
            continue
        
        print(f"Found {len(png_files)} PNG files for report {report_id}")
        
        # Process each PNG file for this report
        for png_file in png_files:
            png_filename = os.path.basename(png_file)
            label_filename = png_filename.replace('.png', '.txt')
            label_file = os.path.join(label_dir, label_filename)
            
            print(f"  Processing: {png_filename}")
            total_images_processed += 1
            
            # Determine output filenames
            output_png = os.path.join(output_images_dir, png_filename)
            output_label = os.path.join(output_labels_dir, label_filename)
            
            # Handle case where expected fractures is 0
            if expected_fractures == 0:
                print(f"    Expected 0 fractures - creating empty label file")
                # Create symlink for image
                if create_symlink(png_file, output_png):
                    successful_symlinks += 1
                
                # Create empty label file
                with open(output_label, 'w') as f:
                    pass  # Create empty file
                empty_labels_created += 1
                print(f"    Created empty label file: {output_label}")
                
            else:
                # Check if label file exists and count fractures
                actual_fractures = count_fractures_in_label(label_file)
                print(f"    Expected fractures: {expected_fractures}, Actual fractures: {actual_fractures}")
                
                if actual_fractures == expected_fractures:
                    # Fracture counts match - create symlinks
                    print(f"    Fracture counts match - creating symlinks")
                    
                    # Create symlink for image
                    if create_symlink(png_file, output_png):
                        successful_symlinks += 1
                    
                    # Create symlink for label
                    if create_symlink(label_file, output_label):
                        print(f"    Created label symlink: {output_label}")
                        
                else:
                    # Fracture counts don't match - log to wrong_yolo_results
                    print(f"    Fracture counts don't match - logging to wrong_yolo_results")
                    wrong_yolo_results.append({
                        'report': report_id,
                        'png_file': png_file,
                        'label_file': label_file,
                        'expected_fractures': expected_fractures,
                        'actual_fractures': actual_fractures
                    })
    
    # Save logging CSV files
    if missing_pngs:
        missing_pngs_df = pd.DataFrame(missing_pngs)
        missing_pngs_path = os.path.join(output_dir, 'missing_pngs.csv')
        missing_pngs_df.to_csv(missing_pngs_path, index=False)
        print(f"\nSaved missing PNGs log to: {missing_pngs_path}")
    
    if wrong_yolo_results:
        wrong_yolo_df = pd.DataFrame(wrong_yolo_results)
        wrong_yolo_path = os.path.join(output_dir, 'wrong_yolo_results.csv')
        wrong_yolo_df.to_csv(wrong_yolo_path, index=False)
        print(f"Saved wrong YOLO results log to: {wrong_yolo_path}")
    
    # Print final statistics
    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Total reports processed: {total_reports}")
    print(f"Total images processed: {total_images_processed}")
    print(f"Successful symlinks created: {successful_symlinks}")
    print(f"Empty label files created: {empty_labels_created}")
    print(f"Missing PNG files: {len(missing_pngs)}")
    print(f"Wrong YOLO results: {len(wrong_yolo_results)}")
    
    return {
        'total_reports': total_reports,
        'total_images': total_images_processed,
        'successful_symlinks': successful_symlinks,
        'empty_labels': empty_labels_created,
        'missing_pngs': len(missing_pngs),
        'wrong_yolo_results': len(wrong_yolo_results)
    }

def main():
    parser = argparse.ArgumentParser(
        description='Create symlinks for images and labels based on CSV input with fracture counts'
    )
    parser.add_argument(
        '--input_csv',
        type=str,
        required=True,
        help='Path to input CSV file with report and num_fractures columns'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        required=True,
        help='Directory containing PNG image files'
    )
    parser.add_argument(
        '--label_dir',
        type=str,
        required=True,
        help='Directory containing YOLO label files (.txt)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for organized training dataset'
    )
    
    args = parser.parse_args()
    
    # Validate input files and directories
    if not os.path.exists(args.input_csv):
        parser.error(f"Input CSV file not found: {args.input_csv}")
    
    if not os.path.exists(args.image_dir):
        parser.error(f"Image directory not found: {args.image_dir}")
    
    if not os.path.exists(args.label_dir):
        parser.error(f"Label directory not found: {args.label_dir}")
    
    # Process the reports
    stats = process_reports(
        args.input_csv,
        args.image_dir,
        args.label_dir,
        args.output_dir
    )
    
    print(f"\nOrganization complete in {args.output_dir}")

if __name__ == "__main__":
    main() 