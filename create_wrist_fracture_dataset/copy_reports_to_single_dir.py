import os
import shutil
import argparse
import glob
from tqdm import tqdm



def copy_png_files(source_dir, dest_dir):
    """Copy PNG files to the destination directory, skipping existing ones."""
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    folders = glob.glob(os.path.join(source_dir, "[0-9]*/"))
    L = len(folders)
    for i, folder in enumerate(folders):
        files = glob.glob(folder+"/*_report.txt")
        if not files:
            print(f"{i}/{L}. Folder is empty: {folder}") 
            continue 
        for file in files: 
            dest_file = os.path.join(dest_dir, os.path.basename(file))
            if os.path.exists(dest_file):
                pass # print(f"{i}/{L}. Skipping existing file: {dest_file}")
            else:
                shutil.copy2(file, dest_file)
                # print(f"{i}/{L}. Copied: {file} to {dest_file}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy all PNG files to a common directory, skipping existing ones.")
    parser.add_argument("--source_dir", default = '/home/ch215616/w/code/llm/experiments/yolov7/wrist_fracture_dataset/dcm/',help="Directory containing subdirectories with PNG and report files.")
    parser.add_argument("--dest_dir", default = '/home/ch215616/w/code/llm/experiments/yolov7/wrist_fracture_dataset/reports/', help="Directory to copy all reports files to.")
    args = parser.parse_args()

    copy_png_files(args.source_dir, args.dest_dir)

