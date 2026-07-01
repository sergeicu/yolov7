import os
import pandas as pd
import subprocess
from tqdm import tqdm
import argparse
import shutil
import multiprocessing
from multiprocessing import Manager

def convert_dicom_to_png(dcm_path, png_path, overwrite=False):
    # if os.path.exists(png_path): 
    #     if not overwrite:
    #         return
    #     else: 
    #         os.remove(png_path)    

    cmd = [
        'python', 
        '/home/ch215616/w/code/llm/experiments/yolov7/dicom-to-png/mritopng.py',
        dcm_path,
        png_path,
    ]
    
    if overwrite: 
        os.remove('png_path')
        cmd.extend(['--overwrite'])
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def extract_medical_report(report_dcm_path, output_path):
    subprocess.run([
        'dcmdump',
        '+L',
        '+P', '0040,a160',
        report_dcm_path,
    ], stdout=open(output_path, 'w'), check=True)

def process_scan_id_directory(args):
    scan_id_path, scan_id, overwrite, no_report_log, no_png_log = args
    local_df = []
    png_count = 0

    for root, dirs, files in os.walk(scan_id_path):
        for file in files:
            if file.lower().startswith('dx'):
                if '~' in file:
                    # Remove files with '~' in the name
                    os.remove(os.path.join(root, file))
                    continue

                dcm_path = os.path.join(root, file)
                scan_type = os.path.basename(os.path.dirname(dcm_path))
                png_filename = f"{scan_id}-{scan_type}-{png_count}.png"
                png_path = os.path.join(scan_id_path, png_filename)
                
                convert_dicom_to_png(dcm_path, png_path, overwrite)
                
                local_df.append({
                    'scan_id': scan_id,
                    'png_file': png_filename,
                    'original_dcm': os.path.relpath(dcm_path, scan_id_path),
                    'scan_type': scan_type
                })
                
                png_count += 1

    if png_count == 0 and no_png_log:
        with open(no_png_log, 'a') as f:
            f.write(f"{scan_id}\n")

    # Find and extract medical report
    report_found = False
    for root, dirs, files in os.walk(scan_id_path):
        if any(folder.startswith('999') for folder in root.split(os.sep)):
            for file in files:
                report_dcm_path = os.path.join(root, file)
                if '~' in report_dcm_path:
                    # Remove files with '~' in the name
                    os.remove(report_dcm_path)
                    continue
                
                report_txt_path = os.path.join(scan_id_path, f"{scan_id}_report.txt")
                extract_medical_report(report_dcm_path, report_txt_path)
                report_found = True
                break
            if report_found:
                break

    if not report_found and no_report_log:
        with open(no_report_log, 'a') as f:
            f.write(f"{scan_id}\n")

    return local_df

def main(main_dir, overwrite=False):
    no_report_log = 'no_report_log.txt'
    no_png_log = 'no_png_log.txt'

    # Clear log files if they exist
    open(no_report_log, 'w').close()
    open(no_png_log, 'w').close()

    # Prepare arguments for multiprocessing
    args_list = []
    for item in os.listdir(main_dir):
        if item[0].isdigit():
            scan_id = item
            scan_id_path = os.path.join(main_dir, scan_id)
            if os.path.isdir(scan_id_path):
                args_list.append((scan_id_path, scan_id, overwrite, no_report_log, no_png_log))

    # Use all available cores
    num_cores = multiprocessing.cpu_count()
    
    # Create a process pool and process the directories in parallel
    with multiprocessing.Pool(num_cores) as pool:
        results = list(tqdm(pool.imap(process_scan_id_directory, args_list), total=len(args_list)))

    # Combine results from all processes
    df = pd.DataFrame([item for sublist in results for item in sublist])

    # Save the DataFrame to a CSV file
    df.to_csv('dataset_info.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DICOM files to PNG and extract medical reports.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PNG files")
    args = parser.parse_args()

    main_directory = '/home/ch215616/w/code/llm/experiments/yolov7/wrist_fracture_dataset/dcm'
    main(main_directory, args.overwrite)

print("Processing complete. Results saved in dataset_info.csv")
print("Scan IDs with no report logged in no_report_log.txt")
print("Scan IDs with no PNG files logged in no_png_log.txt")