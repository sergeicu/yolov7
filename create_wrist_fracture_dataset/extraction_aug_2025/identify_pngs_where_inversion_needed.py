"""Identifies if there is a file difference between pngs_aug2025 (old) and pngs_aug2025_fixed (invertsion fixed) - if yes - then we say it was inverted"""

import os
import csv

PNGS_ORIG_DIR = "/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/pngs_aug2025" #35412 
PNGS_FIXED_DIR = "/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/pngs_aug2025_fixed" #29093
OUTPUT_CSV = "png_inversion.csv"

def main():
    results = []
    for root, dirs, files in os.walk(PNGS_FIXED_DIR):
        for fname in files:
            fixed_path = os.path.join(root, fname)
            # Get relative path from PNGS_FIXED_DIR
            rel_path = os.path.relpath(fixed_path, PNGS_FIXED_DIR)
            orig_path = os.path.join(PNGS_ORIG_DIR, rel_path)
            if os.path.exists(orig_path):
                fixed_size = os.path.getsize(fixed_path)
                orig_size = os.path.getsize(orig_path)
                inversion = orig_size != fixed_size
                results.append({'filename': rel_path, 'inversion': inversion})
            else:
                print(f"Warning: {rel_path} not found in original directory, adding anyway.")
                results.append({'filename': rel_path, 'inversion': inversion})


                # (Example warnings with accession-numbered filenames redacted for PHI safety)

    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['filename', 'inversion'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Done. Results written to {OUTPUT_CSV}")

if __name__ == "__main__":
    main() 