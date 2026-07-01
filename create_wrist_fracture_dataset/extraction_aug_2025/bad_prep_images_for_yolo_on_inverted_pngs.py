import os
import argparse
import csv

def load_studyids(csv_path):
    studyids = set()
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            studyids.add(row['studyid'])
    return studyids

def main(pngs_dir, csv_path):
    pngs_dir = os.path.abspath(pngs_dir)
    parent_dir = os.path.dirname(pngs_dir)
    inverted_dir = os.path.join(parent_dir, 'pngs_inverted')
    os.makedirs(inverted_dir, exist_ok=True)

    studyids = load_studyids(csv_path)

    for fname in os.listdir(pngs_dir):
        if fname.lower().endswith('.png'):
            for studyid in studyids:
                if fname.startswith(studyid + '-'):
                    src = os.path.join(pngs_dir, fname)
                    dst = os.path.join(inverted_dir, fname)
                    if os.path.islink(dst) or os.path.exists(dst):
                        os.remove(dst)
                    os.symlink(src, dst)
                    print(f"Symlinked: {dst} -> {src}")
                    break  # Only need to match one studyid

    print(f"All symlinks created in {inverted_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create symlinks for PNGs in a new 'pngs_inverted' directory, only for studyids in a CSV.")
    parser.add_argument(
        "--pngs_dir",
        type=str,
        default="/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/pngs_aug2025_fixed/",
        help="Directory containing PNG images (default: %(default)s)"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="studyid_inversion.csv",
        help="CSV file with 'studyid' column (default: %(default)s)"
    )
    args = parser.parse_args()
    main(args.pngs_dir, args.csv_path) 