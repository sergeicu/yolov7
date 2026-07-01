# Define placeholder paths for source folders and destination folder names
json_source_folder = '/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/experiments/labeller_running_for_andy/s20250711_generate_structured_reports/outputs/grounded_33k/puyangwang-medgemma-27b-it-q8-reports_aug2025/' # Replace with the actual path to your JSON files
#png_source_folder = '/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/pngs_aug2025/'
png_source_folder = '/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/pngs_aug2025_fixed/'
# png_source_folder = '/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/yolov7/ALL/run1/'
destination_folder_json = json_source_folder[:-1]+'_NOTWRIST/'
destination_folder_png = png_source_folder[:-1]+'_NOTWRIST/'
bad_study_ids_csv_path = 'bad_study_ids.csv' # Path to the CSV created in a previous step

# we are left with 10566 reports... (+ 8040 previously processed reports)
# 50,000 images or so.

# for some reason we have 13000 bad ids and only 6777 reports that were fitting - well ok 

# In [72]: len(bad_study_ids)
# Out[72]: 13195

# In [74]: ls /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/experiments/labeller_running_for_andy/s20250711_generate_structured_reports/output
#     ...: s/grounded_33k/puyangwang-medgemma-27b-it-q8-reports_aug2025_NOTWRIST/puyangwang-medgemma-27b-it-q8-dataset_NOTWRIST/ | wc -l
# 6777

import os
import shutil
import pandas as pd

def process_study_files(bad_study_ids_path, source_folder_path, destination_folder_name):
    """
    Loads accession numbers from a CSV and moves corresponding JSON and PNG files
    from a source folder to a destination folder.

    Args:
        bad_study_ids_path (str): Path to the CSV file containing bad study IDs.
        source_folder_path (str): Path to the source folder containing JSON and PNG files.
        destination_folder_name (str): Name of the destination folder to move files to.
    """
    if not os.path.exists(bad_study_ids_path):
        print(f"Error: bad_study_ids.csv not found at {bad_study_ids_path}")
        return

    try:
        bad_study_ids_df = pd.read_csv(bad_study_ids_path)
        bad_study_ids = bad_study_ids_df['Accession Number'].astype(str).tolist()
    except Exception as e:
        print(f"Error reading {bad_study_ids_path}: {e}")
        return

    destination_folder_path = os.path.join(source_folder_path, destination_folder_name)
    if not os.path.exists(destination_folder_path):
        os.makedirs(destination_folder_path)

    if not os.path.exists(source_folder_path):
        print(f"Error: Source folder not found at {source_folder_path}")
        return

    for filename in os.listdir(source_folder_path):
        file_path = os.path.join(source_folder_path, filename)
        if os.path.isfile(file_path):
            # Check for JSON files (AccessionNumber.json)
            if filename.endswith('.json'):
                accession_number = filename.replace('.json', '')
                if accession_number in bad_study_ids:
                    shutil.move(file_path, destination_folder_path)
                    print(f"Moved JSON file: {filename}")
            # Check for PNG files (AccessionNumber-<somecharacters>.png)
            elif filename.endswith('.png'):
                # Extract Accession Number from PNG filename
                parts = filename.split('-')
                if len(parts) > 1:
                    accession_number = parts[0]
                    if accession_number in bad_study_ids:
                        shutil.move(file_path, destination_folder_path)
                        print(f"Moved PNG file: {filename}")


# Call the function for JSON files
# print(f"Processing JSON files in {json_source_folder}...")
# process_study_files(bad_study_ids_csv_path, json_source_folder, destination_folder_json)

# Call the function for PNG files
print(f"\nProcessing PNG files in {png_source_folder}...")
process_study_files(bad_study_ids_csv_path, png_source_folder, destination_folder_png)

