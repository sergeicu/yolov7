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


                # Warning: 25237513-1_PA-0.png not found in original directory, skipping.
                # Warning: 25237513-2_Lateral-2.png not found in original directory, skipping.
                # Warning: 25237513-3_Lateral-1.png not found in original directory, skipping.
                # Warning: 25356045-1_PA-0.png not found in original directory, skipping.
                # Warning: 25356045-2_Lateral-1.png not found in original directory, skipping.
                # Warning: 25369712-25369712-0.png not found in original directory, skipping.
                # Warning: 25509899-25509899-2.png not found in original directory, skipping.
                # Warning: 25663909-1_PA-0.png not found in original directory, skipping.
                # Warning: 25819909-1_PA-0.png not found in original directory, skipping.
                # Warning: 25819909-2_Lateral-1.png not found in original directory, skipping.
                # Warning: 25870895-2_Oblique-3.png not found in original directory, skipping.
                # Warning: 26051198-1_PA-0.png not found in original directory, skipping.
                # Warning: 26051198-2_Oblique-3.png not found in original directory, skipping.
                # Warning: 26051198-3_Lateral-1.png not found in original directory, skipping.
                # Warning: 26051198-4_Navicular-2.png not found in original directory, skipping.
                # Warning: 26055389-26055389-0.png not found in original directory, skipping.
                # Warning: 26055389-26055389-1.png not found in original directory, skipping.
                # Warning: 26255461-1_PA-0.png not found in original directory, skipping.
                # Warning: 26255461-2_Lateral-3.png not found in original directory, skipping.
                # Warning: 26255461-3_Navicular-2.png not found in original directory, skipping.
                # Warning: 26255461-4_Navicular-1.png not found in original directory, skipping.                

    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['filename', 'inversion'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Done. Results written to {OUTPUT_CSV}")

if __name__ == "__main__":
    main() 