"""This file identifies which files were inverted and then make symlinks from pngs_aug2025_fixed to pngs_inverted_from_png_csv. Why? """

import os
import csv

PNGS_FIXED_DIR = "/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset//pngs_aug2025_fixed" #29093  - many files were fixed...not just those that were inveerted
OUTPUT_DIR = "/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset//pngs_inverted_from_png_csv" # 7360 
CSV_PATH = "png_inversion.csv"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    failed_symlinks = []

    with open(CSV_PATH, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if str(row['inversion']).lower() == 'true':
                rel_path = row['filename']
                src = os.path.join(PNGS_FIXED_DIR, rel_path)
                dst = os.path.join(OUTPUT_DIR, rel_path)
                dst_dir = os.path.dirname(dst)
                os.makedirs(dst_dir, exist_ok=True)
                # Remove existing file or symlink (even if broken)
                if os.path.lexists(dst):
                    os.unlink(dst)
                try:
                    os.symlink(src, dst)
                    print(f"Symlinked: {dst} -> {src}")
                    # Check if the symlink is valid (points to a real file)
                    if not os.path.exists(dst):
                        print(f"WARNING: Symlink {dst} is broken (target does not exist)")
                        failed_symlinks.append(rel_path)
                except Exception as e:
                    print(f"Failed to symlink {dst} -> {src}: {e}")
                    failed_symlinks.append(rel_path)

    print(f"All symlinks created in {OUTPUT_DIR}")
    if failed_symlinks:
        print("\nThe following files were NOT successfully symlinked:")
        for fname in failed_symlinks:
            print(fname)
    else:
        print("\nAll symlinks were created successfully.")

if __name__ == "__main__":
    main() 