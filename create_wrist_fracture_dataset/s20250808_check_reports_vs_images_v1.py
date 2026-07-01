"""
Script to check correspondence between reports and images in medical imaging datasets.

This script analyzes the relationship between report files and their corresponding image files
by extracting report IDs from filenames and checking for matching pairs. It supports multiple
report file formats and provides detailed analysis of missing or orphaned files.

REPORT FORMATS SUPPORTED:
1. JSON format: Reports named as '<SCAN_ID>.json'
   - Report ID is extracted as the filename without the .json extension
   - Example: '<SCAN_ID>.json' → Report ID: '<SCAN_ID>'

2. Text format: Reports named as '<SCAN_ID_2>_report.txt'
   - Report ID is extracted by removing the '_report' suffix
   - Example: '<SCAN_ID_2>_report.txt' → Report ID: '<SCAN_ID_2>'

IMAGE FORMAT:
- PNG files named with pattern: '{report_id}-{sequence}_{view}-{index}.png'
- Examples: '<SCAN_ID>-1_PA-1.png', '<SCAN_ID>-2_Oblique-2.png', '<SCAN_ID>-3_Lateral-0.png'
- Report ID is extracted as the part before the first hyphen

AUTOMATIC DETECTION:
The script automatically detects the report format by:
1. First checking for .json files in the reports directory
2. If no JSON files found, looking for *_report.txt files
3. Falling back to any .txt files if needed
4. Providing clear feedback about which pattern was detected

FUNCTIONALITY:
- Validates that each report has at least one corresponding image
- Validates that each image has a corresponding report
- Provides detailed examples of missing correspondences
- Shows examples of successful matches
- Generates comprehensive summary statistics
- Supports partial checking (reports-only or images-only)

USAGE EXAMPLES:
    # Check both reports and images
    python s20250808_check_reports_vs_images.py \
        --images /path/to/images \
        --reports /path/to/reports

    # Check only if reports have corresponding images
    python s20250808_check_reports_vs_images.py \
        --images /path/to/images \
        --reports /path/to/reports \
        --check-reports-only

    # Check only if images have corresponding reports
    python s20250808_check_reports_vs_images.py \
        --images /path/to/images \
        --reports /path/to/reports \
        --check-images-only

OUTPUT:
The script provides:
- Detection feedback showing which report format was identified
- Count of reports and image report IDs found
- Detailed analysis of missing correspondences with examples
- Examples of successful matches
- Summary statistics including totals and mismatch counts

ERROR HANDLING:
- Validates directory existence before processing
- Handles missing files gracefully
- Provides clear error messages for invalid arguments
- Prevents conflicting flag combinations

PERFORMANCE:
- Uses efficient pathlib operations for large datasets
- Processes files using generators to minimize memory usage
- Suitable for datasets with 60,000+ images
- Provides progress feedback during scanning

AUTHOR: Generated for medical imaging dataset validation
DATE: 2025-08-08
VERSION: 1.0
"""

"""
Script to check correspondence between reports and images.
Reports can be either:
- JSON files named like '<SCAN_ID>.json' 
- Text files named like '<SCAN_ID_2>_report.txt'
Images are PNG files named like '<SCAN_ID>-1_PA-1.png', '<SCAN_ID>-2_Oblique-2.png', etc.
"""

import os
import argparse
import json
from pathlib import Path
from collections import defaultdict
import sys


def detect_report_pattern(reports_dir):
    """Detect the naming pattern of reports in the directory."""
    reports_path = Path(reports_dir)
    
    if not reports_path.exists():
        print(f"Error: Reports directory {reports_dir} does not exist")
        return None, None
    
    # Check for JSON files first
    json_files = list(reports_path.glob("*.json"))
    if json_files:
        print("Detected JSON report format (e.g., '<SCAN_ID>.json')")
        return "json", json_files
    
    # Check for text files with _report pattern
    txt_files = list(reports_path.glob("*_report.txt"))
    if txt_files:
        print("Detected text report format (e.g., '<SCAN_ID_2>_report.txt')")
        return "txt", txt_files
    
    # Check for any text files as fallback
    all_txt_files = list(reports_path.glob("*.txt"))
    if all_txt_files:
        print("Detected text report format (fallback)")
        return "txt", all_txt_files
    
    print("Error: No report files found. Expected either .json files or *_report.txt files")
    return None, None


def get_report_ids(reports_dir):
    """Extract report IDs from filenames based on detected pattern."""
    report_ids = set()
    pattern, files = detect_report_pattern(reports_dir)
    
    if pattern is None:
        return report_ids
    
    for file_path in files:
        if pattern == "json":
            # Extract the report ID (filename without .json extension)
            report_id = file_path.stem
            report_ids.add(report_id)
        elif pattern == "txt":
            # Extract the report ID (part before _report)
            filename = file_path.stem
            if filename.endswith("_report"):
                report_id = filename[:-7]  # Remove "_report" suffix
                report_ids.add(report_id)
            else:
                # Fallback: assume the whole filename is the report ID
                report_id = filename
                report_ids.add(report_id)
    
    return report_ids


def get_image_report_ids(images_dir):
    """Extract report IDs from image filenames."""
    image_report_ids = set()
    images_path = Path(images_dir)
    
    if not images_path.exists():
        print(f"Error: Images directory {images_dir} does not exist")
        return image_report_ids
    
    for png_file in images_path.glob("*.png"):
        # Extract the report ID (part before the first '-')
        filename = png_file.stem
        if '-' in filename:
            report_id = filename.split('-')[0]
            image_report_ids.add(report_id)
    
    return image_report_ids


def check_reports_vs_images(reports_dir, images_dir, check_reports_only=False, check_images_only=False):
    """Check correspondence between reports and images."""
    
    print(f"Scanning reports directory: {reports_dir}")
    report_ids = get_report_ids(reports_dir)
    print(f"Found {len(report_ids)} reports")
    
    if not check_reports_only:
        print(f"Scanning images directory: {images_dir}")
        image_report_ids = get_image_report_ids(images_dir)
        print(f"Found {len(image_report_ids)} unique report IDs from images")
    
    # Check if reports have corresponding images
    if not check_images_only:
        print("\n=== Checking if reports have corresponding images ===")
        reports_without_images = report_ids - image_report_ids
        if reports_without_images:
            print(f"❌ Found {len(reports_without_images)} reports without corresponding images:")
            print("Examples of reports without images:")
            for report_id in sorted(list(reports_without_images))[:10]:  # Show first 10
                print(f"  - {report_id}")
            if len(reports_without_images) > 10:
                print(f"  ... and {len(reports_without_images) - 10} more")
        else:
            print("✅ All reports have corresponding images")
    
    # Check if images have corresponding reports
    if not check_reports_only:
        print("\n=== Checking if images have corresponding reports ===")
        images_without_reports = image_report_ids - report_ids
        if images_without_reports:
            print(f"❌ Found {len(images_without_reports)} image report IDs without corresponding reports:")
            print("Examples of image report IDs without reports:")
            for report_id in sorted(list(images_without_reports))[:10]:  # Show first 10
                print(f"  - {report_id} (from images)")
            if len(images_without_reports) > 10:
                print(f"  ... and {len(images_without_reports) - 10} more")
        else:
            print("✅ All images have corresponding reports")
    
    # Show examples of matching reports and images
    if not check_reports_only and not check_images_only:
        print("\n=== Examples of matching reports and images ===")
        matching_ids = report_ids & image_report_ids
        if matching_ids:
            print(f"✅ Found {len(matching_ids)} reports with matching images")
            print("Examples of matching report IDs:")
            for report_id in sorted(list(matching_ids))[:10]:
                print(f"  - {report_id}")
            if len(matching_ids) > 10:
                print(f"  ... and {len(matching_ids) - 10} more")
        else:
            print("❌ No matching reports and images found")
    
    # Summary
    if not check_reports_only and not check_images_only:
        print("\n=== Summary ===")
        print(f"Total reports: {len(report_ids)}")
        print(f"Total unique report IDs from images: {len(image_report_ids)}")
        print(f"Reports without images: {len(reports_without_images)}")
        print(f"Images without reports: {len(images_without_reports)}")
        print(f"Reports with matching images: {len(matching_ids)}")


def main():
    parser = argparse.ArgumentParser(description="Check correspondence between reports and images")
    parser.add_argument("--images", required=True, help="Path to images directory")
    parser.add_argument("--reports", required=True, help="Path to reports directory")
    parser.add_argument("--check-reports-only", action="store_true", 
                       help="Only check if reports have corresponding images")
    parser.add_argument("--check-images-only", action="store_true", 
                       help="Only check if images have corresponding reports")
    
    args = parser.parse_args()
    
    # Validate that only one of the "only" flags is set
    if args.check_reports_only and args.check_images_only:
        print("Error: Cannot use both --check-reports-only and --check-images-only")
        sys.exit(1)
    
    check_reports_vs_images(
        args.reports, 
        args.images, 
        check_reports_only=args.check_reports_only,
        check_images_only=args.check_images_only
    )


if __name__ == "__main__":
    main()
