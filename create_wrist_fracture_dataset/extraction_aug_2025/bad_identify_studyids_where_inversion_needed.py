import os
import csv
import pydicom as dicom
import numpy as np

REPORTS_DIR = "/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/experiments/labeller_running_for_andy/s20250711_generate_structured_reports/outputs/grounded_33k/puyangwang-medgemma-27b-it-q8-reports_aug2025/"
DCM_ROOTDIR = "/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/extraction_aug_2025/dcm"
OUTPUT_CSV = "studyid_inversion.csv"

def inversion_needed(dcm_path):
    try:
        plan = dicom.dcmread(dcm_path, stop_before_pixels=True)
        # Check PhotometricInterpretation
        if hasattr(plan, 'PhotometricInterpretation'):
            if plan.PhotometricInterpretation == 'MONOCHROME1':
                return True
        # Check PresentationLUTShape
        if hasattr(plan, 'PresentationLUTShape'):
            if plan.PresentationLUTShape == 'INVERSE':
                return True
    except Exception as e:
        print(f"Error reading {dcm_path}: {e}")
    return False

def main():
    studyids = []
    for fname in os.listdir(REPORTS_DIR):
        if fname.endswith('.json'):
            studyid = fname[:-5]  # remove .json
            studyids.append(studyid)

    results = []
    for studyid in studyids:
        study_dir = os.path.join(DCM_ROOTDIR, studyid)
        inversion = False
        if os.path.isdir(study_dir):
            for root, dirs, files in os.walk(study_dir):
                for file in files:
                    if file.lower().startswith('dx'):
                        dcm_path = os.path.join(root, file)
                        if inversion_needed(dcm_path):
                            inversion = True
                            print(f"Inversion needed for {studyid}")
                            break
                if inversion:
                    break
        else:
            print(f"Warning: Study directory not found for {studyid}")
        results.append({'studyid': studyid, 'inversion': inversion})

    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['studyid', 'inversion'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Done. Results written to {OUTPUT_CSV}")

if __name__ == "__main__":
    main() 