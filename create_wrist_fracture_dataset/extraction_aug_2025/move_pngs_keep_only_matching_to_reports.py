"""Move away any pngs that do not have corresponding report (for whatever reason the report was not extracted)"""

import os
import shutil
import argparse

def get_studyids_from_reports(reports_dir):
    studyids = set()
    for fname in os.listdir(reports_dir):
        if fname.endswith('.json'):
            studyid = fname[:-5]  # remove .json
            studyids.add(studyid)
    return studyids

def main(reports_dir, images_dir, dryrun):
    # Get all studyids from reports
    studyids = get_studyids_from_reports(reports_dir)
    # Prepare output dir
    parent_dir = os.path.dirname(os.path.abspath(images_dir))
    output_dir = os.path.join(parent_dir, os.path.basename(images_dir) + "_no_reports_found")
    if not dryrun and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    found_nonmatching = False
    for fname in os.listdir(images_dir):
        if fname.endswith('.png'):
            # Extract studyid from image filename
            if '-' not in fname:
                continue  # skip files not matching pattern
            studyid = fname.split('-', 1)[0]
            if studyid not in studyids:
                if dryrun:
                    print(f"Non-matching image found: {fname}")
                    return
                # Move file
                src = os.path.join(images_dir, fname)
                dst = os.path.join(output_dir, fname)
                shutil.move(src, dst)
                found_nonmatching = True

    if not dryrun:
        if found_nonmatching:
            print(f"Moved non-matching images to {output_dir}")
        else:
            print("No non-matching images found.")
    else:
        print("No non-matching images found in dry run.")

if __name__ == "__main__":
    input = 'please note it mistakenly puts files in this folder below. please press to proceed \n /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/_no_reports_found/'
    parser = argparse.ArgumentParser(description="Move PNGs without matching reports.")
    reports_dir = "/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/experiments/labeller_running_for_andy/s20250711_generate_structured_reports/outputs/grounded_33k/puyangwang-medgemma-27b-it-q8-reports_aug2025/"
    images_dir = "/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/pngs_aug2025_fixed/"
    parser.add_argument("--reports_dir", default=reports_dir, help="Directory containing report JSON files")
    parser.add_argument("--images_dir", default=images_dir, help="Directory containing PNG images")
    parser.add_argument("--dryrun", action="store_true", help="Only print the first non-matching image and exit")
    args = parser.parse_args()
    main(args.reports_dir, args.images_dir, args.dryrun)



    # Example path: /lab-share/.../create_wrist_fracture_dataset/_no_reports_found/<SCAN_ID>-1_PA-0.png