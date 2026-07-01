import os
import logging

# Set up logging
logging.basicConfig(
    filename='remove_empty_pngs2.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

root_dir = '/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/extraction_aug_2025/dcm/'

# Use os.walk to traverse through the directory
folders = []
for root, dirs, files in os.walk(root_dir):
    # Only process immediate subdirectories that match the pattern [0-9]*
    if root == root_dir:
        for dir_name in dirs:
            if dir_name.isdigit() or (dir_name[0].isdigit() and any(c.isdigit() for c in dir_name)):
                folders.append(os.path.join(root, dir_name))
    break  # Only process the immediate subdirectories, not deeper

print(f"Found {len(folders)} folders to process")
logging.info(f"Found {len(folders)} folders to process")

removed_count = 0

for counter, folder in enumerate(folders):
    # Use os.listdir instead of glob for PNG files
    try:
        print(f"Processing folder {counter} of {len(folders)}")
        files = [f for f in os.listdir(folder + "/fixed") if f.endswith('.png')]
        for file in files:
            file_path = os.path.join(folder, file)
            if os.path.getsize(file_path) == 0:
                print(f"Removing empty file: {file_path}")
                logging.info(f"Removed empty file: {file_path}")
                os.remove(file_path)
                removed_count += 1
    except PermissionError:
        error_msg = f"Permission denied accessing folder: {folder}"
        print(error_msg)
        logging.error(error_msg)
    except Exception as e:
        error_msg = f"Error processing folder {folder}: {e}"
        print(error_msg)
        logging.error(error_msg)

print(f"Finished processing all folders. Removed {removed_count} empty PNG files.")
logging.info(f"Finished processing all folders. Removed {removed_count} empty PNG files.")
