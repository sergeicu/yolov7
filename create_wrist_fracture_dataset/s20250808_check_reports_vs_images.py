#!/usr/bin/env python3
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


def get_image_info(images_dir):
    """Extract report IDs from image filenames and count total images."""
    image_report_ids = set()
    total_images = 0
    images_path = Path(images_dir)
    
    if not images_path.exists():
        print(f"Error: Images directory {images_dir} does not exist")
        return image_report_ids, total_images
    
    for png_file in images_path.glob("*.png"):
        total_images += 1
        # Extract the report ID (part before the first '-')
        filename = png_file.stem
        if '-' in filename:
            report_id = filename.split('-')[0]
            image_report_ids.add(report_id)
    
    return image_report_ids, total_images


def check_reports_vs_images(reports_dir, images_dir, check_reports_only=False, check_images_only=False):
    """Check correspondence between reports and images."""
    
    print(f"Scanning reports directory: {reports_dir}")
    report_ids = get_report_ids(reports_dir)
    print(f"Found {len(report_ids)} reports")
    
    if not check_reports_only:
        print(f"Scanning images directory: {images_dir}")
        image_report_ids, total_images = get_image_info(images_dir)
        print(f"Found {total_images} total images")
        print(f"Found {len(image_report_ids)} unique report IDs from images")
    
    # Check if reports have corresponding images
    if not check_images_only:
        print("\n=== Checking if reports have corresponding images ===")
        reports_without_images = report_ids - image_report_ids
        reports_with_images = report_ids & image_report_ids
        if reports_without_images:
            print(f"❌ Found {len(reports_without_images)} reports without corresponding images:")
            print("Examples of reports without images:")
            for report_id in sorted(list(reports_without_images))[:10]:  # Show first 10
                print(f"  - {report_id}")
            if len(reports_without_images) > 10:
                print(f"  ... and {len(reports_without_images) - 10} more")
        else:
            print("✅ All reports have corresponding images")
        
        print(f"📊 Reports with at least one image: {len(reports_with_images)}")
    
    # Check if images have corresponding reports
    if not check_reports_only:
        print("\n=== Checking if images have corresponding reports ===")
        images_without_reports = image_report_ids - report_ids
        images_with_reports = image_report_ids & report_ids
        if images_without_reports:
            print(f"❌ Found {len(images_without_reports)} image report IDs without corresponding reports:")
            print("Examples of image report IDs without reports:")
            for report_id in sorted(list(images_without_reports))[:10]:  # Show first 10
                print(f"  - {report_id} (from images)")
            if len(images_without_reports) > 10:
                print(f"  ... and {len(images_without_reports) - 10} more")
        else:
            print("✅ All images have corresponding reports")
        
        print(f"📊 Image report IDs with corresponding reports: {len(images_with_reports)}")
    
    # Show examples of matching reports and images
    if not check_reports_only and not check_images_only:
        print("\n=== Examples of matching reports and images ===")
        matching_ids = report_ids & image_report_ids
        if matching_ids:
            print(f"✅ Found {len(matching_ids)} report IDs that appear in both reports and images")
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
        print(f"Total images: {total_images}")
        print(f"Unique report IDs from images: {len(image_report_ids)}")
        print(f"Reports without images: {len(reports_without_images)}")
        print(f"Images without reports: {len(images_without_reports)}")
        print(f"Report IDs that appear in both: {len(matching_ids)}")
        
        # Additional analysis
        print(f"\n📈 Relationship Analysis:")
        print(f"  - Reports with at least one image: {len(reports_with_images)}")
        print(f"  - Image report IDs with corresponding reports: {len(images_with_reports)}")
        print(f"  - Orphaned image report IDs: {len(images_without_reports)}")
        
        if len(reports_with_images) > 0:
            print(f"  - Average images per report: {total_images / len(reports_with_images):.2f}")
            print(f"  - Average images per report ID (from images): {total_images / len(image_report_ids):.2f}")


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