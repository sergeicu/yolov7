#!/usr/bin/env python3
"""
Script to compare files between two directories and move newer files to ~/ww/ directory.
Only moves files that are different in content, excluding PNG files.
"""

import os
import shutil
import hashlib
import filecmp
import argparse
import json
from pathlib import Path
from datetime import datetime

def get_file_hash(filepath):
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def get_file_mtime(filepath):
    """Get file modification time."""
    try:
        return os.path.getmtime(filepath)
    except Exception as e:
        print(f"Error getting mtime for {filepath}: {e}")
        return 0

def should_skip_directory(file_path, skip_dirs):
    """Check if a directory should be skipped based on skip_dirs list."""
    if not skip_dirs:
        return False
    
    # Get the relative path from the root directory
    try:
        relative_path = file_path.relative_to(file_path.parts[0])
        for skip_dir in skip_dirs:
            if skip_dir in str(relative_path):
                return True
    except ValueError:
        # If we can't get relative path, check if any skip_dir is in the path
        for skip_dir in skip_dirs:
            if skip_dir in str(file_path):
                return True
    return False

def delete_identical_files(log_file_path, w_dir=None):
    """Delete files that were marked as identical in a previous log file."""
    if not log_file_path.exists():
        print(f"Error: Log file {log_file_path} does not exist!")
        return
    
    try:
        with open(log_file_path, 'r') as f:
            log_data = json.load(f)
    except Exception as e:
        print(f"Error reading log file: {e}")
        return
    
    # Use w_dir from log if not provided, or fall back to default
    if w_dir is None:
        w_dir = Path(log_data.get('source_dir', str(Path.home() / "w" / "code" / "llm" / "experiments" / "yolov7" / "create_wrist_fracture_dataset")))
    else:
        w_dir = Path(w_dir)
    
    if not w_dir.exists():
        print(f"Error: Source directory {w_dir} does not exist!")
        return
    
    deleted_count = 0
    error_count = 0
    
    print(f"Deleting identical files from {w_dir}")
    print(f"Based on log file: {log_file_path}")
    
    for entry in log_data.get('identical_files', []):
        filename = entry.get('filename')
        file_path = entry.get('w_file_path')
        
        if not filename or not file_path:
            continue
        
        # Use the stored path or construct it
        if file_path:
            target_file = Path(file_path)
        else:
            target_file = w_dir / filename
        
        if target_file.exists():
            try:
                target_file.unlink()
                print(f"Deleted: {filename}")
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {filename}: {e}")
                error_count += 1
        else:
            print(f"File not found: {filename}")
            error_count += 1
    
    print(f"\nDelete Summary:")
    print(f"Files deleted: {deleted_count}")
    print(f"Errors: {error_count}")

def compare_and_move_files(relative_path=None, root_only=False, skip_dirs=None, log_file=None, delete_mode=False):
    if delete_mode and log_file:
        delete_identical_files(log_file)
        return
    
    # Determine directories
    if relative_path:
        # Use provided relative path to construct w and ww directories
        w_dir = Path.home() / "w" / relative_path
        ww_dir = Path.home() / "ww" / relative_path
    else:
        # Use default directories
        w_dir = Path.home() / "w" / "code" / "llm" / "experiments" / "yolov7" / "create_wrist_fracture_dataset"
        ww_dir = Path.home() / "ww" / "code" / "llm" / "experiments" / "yolov7" / "create_wrist_fracture_dataset"
    
    print(f"Source directory: {w_dir}")
    print(f"Destination directory: {ww_dir}")
    print(f"Root only mode: {root_only}")
    if skip_dirs:
        print(f"Skipping directories: {', '.join(skip_dirs)}")
    if log_file:
        print(f"Log file: {log_file}")
    
    # Check if directories exist
    if not w_dir.exists():
        print(f"Error: Source directory {w_dir} does not exist!")
        return
    
    if not ww_dir.exists():
        print(f"Error: Destination directory {ww_dir} does not exist!")
        return
    
    # Get all files from both directories (excluding PNG files)
    w_files = set()
    ww_files = set()
    
    # Function to collect files based on root_only flag
    def collect_files(directory, files_set):
        if root_only:
            # Only process files in the root directory
            for file_path in directory.iterdir():
                if file_path.is_file() and file_path.suffix.lower() != '.png':
                    files_set.add(file_path.name)
        else:
            # Process all files recursively, respecting skip_dirs
            for file_path in directory.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() != '.png':
                    # Check if this file is in a directory that should be skipped
                    if not should_skip_directory(file_path, skip_dirs):
                        files_set.add(file_path.name)
    
    collect_files(w_dir, w_files)
    collect_files(ww_dir, ww_files)
    
    print(f"Found {len(w_files)} non-PNG files in ~/w/ directory")
    print(f"Found {len(ww_files)} non-PNG files in ~/ww/ directory")
    
    # Find common files
    common_files = w_files.intersection(ww_files)
    print(f"Found {len(common_files)} common files to compare")
    
    moved_count = 0
    skipped_count = 0
    identical_files = []
    
    for filename in common_files:
        # Find the actual file paths (handling root_only vs recursive)
        w_file = None
        ww_file = None
        
        if root_only:
            w_file = w_dir / filename
            ww_file = ww_dir / filename
        else:
            # Find the first occurrence of the file in each directory
            for file_path in w_dir.rglob(filename):
                if file_path.is_file() and not should_skip_directory(file_path, skip_dirs):
                    w_file = file_path
                    break
            
            for file_path in ww_dir.rglob(filename):
                if file_path.is_file() and not should_skip_directory(file_path, skip_dirs):
                    ww_file = file_path
                    break
        
        # Skip if either file doesn't exist
        if w_file is None or ww_file is None or not w_file.exists() or not ww_file.exists():
            print(f"Warning: {filename} not found in one of the directories")
            continue
        
        # Compare file contents using hash
        w_hash = get_file_hash(w_file)
        ww_hash = get_file_hash(ww_file)
        
        if w_hash is None or ww_hash is None:
            print(f"Warning: Could not read {filename}, skipping")
            continue
        
        # If files are identical, skip
        if w_hash == ww_hash:
            print(f"Skipping {filename}: files are identical")
            skipped_count += 1
            identical_files.append({
                'filename': filename,
                'w_file_path': str(w_file),
                'ww_file_path': str(ww_file),
                'hash': w_hash,
                'timestamp': datetime.now().isoformat()
            })
            continue
        
        # Files are different, check which is newer
        w_mtime = get_file_mtime(w_file)
        ww_mtime = get_file_mtime(ww_file)
        
        if w_mtime > ww_mtime:
            # ~/w/ file is newer, move it to ~/ww/
            try:
                shutil.copy2(w_file, ww_file)
                print(f"Moved newer file: {filename} (from ~/w/ to ~/ww/)")
                moved_count += 1
            except Exception as e:
                print(f"Error moving {filename}: {e}")
        else:
            # ~/ww/ file is newer or same age, no action needed
            print(f"Skipping {filename}: ~/ww/ version is newer or same age")
            skipped_count += 1
    
    # Save log file if specified
    if log_file:
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'source_dir': str(w_dir),
            'dest_dir': str(ww_dir),
            'root_only': root_only,
            'skip_dirs': skip_dirs,
            'identical_files': identical_files,
            'moved_count': moved_count,
            'skipped_count': skipped_count,
            'total_processed': moved_count + skipped_count
        }
        
        try:
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
            print(f"Log saved to: {log_file}")
        except Exception as e:
            print(f"Error saving log file: {e}")
    
    print(f"\nSummary:")
    print(f"Files moved: {moved_count}")
    print(f"Files skipped: {skipped_count}")
    print(f"Total processed: {moved_count + skipped_count}")

def main():
    parser = argparse.ArgumentParser(description='Compare and move files between ~/w/ and ~/ww/ directories')
    parser.add_argument('--path', type=str,
                       help='Relative path (e.g., "code/llm/experiments/yolov7") - will compare ~/w/path vs ~/ww/path')
    parser.add_argument('--root-only', action='store_true', 
                       help='Only process files in the root directory (no subdirectories)')
    parser.add_argument('--skip-dirs', nargs='+', 
                       help='List of directory names to skip (relative paths, not absolute)')
    parser.add_argument('--log-file', type=Path,
                       help='Path to save detailed log file (JSON format)')
    parser.add_argument('--delete', action='store_true',
                       help='Delete identical files from ~/w/ directory based on a previous log file')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.delete and not args.log_file:
        print("Error: --delete requires --log-file to be specified")
        return
    
    compare_and_move_files(
        relative_path=args.path,
        root_only=args.root_only, 
        skip_dirs=args.skip_dirs, 
        log_file=args.log_file,
        delete_mode=args.delete
    )

if __name__ == "__main__":
    main() 