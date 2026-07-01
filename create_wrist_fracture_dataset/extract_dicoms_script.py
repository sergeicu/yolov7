import pandas as pd
import os
from datetime import datetime
import subprocess
from concurrent.futures import ThreadPoolExecutor
import logging

# Set up logging
logging.basicConfig(filename='extract_dicoms.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

f = 'wrist_fracture_dataset/wrist facture Jan1 2022 till Aug13 2024.xlsx'
df = pd.read_excel(f)
df2 = df.drop_duplicates(subset='Accession Number', keep="first")

os.makedirs('wrist_fracture_dataset', exist_ok=True)
savedirr = 'wrist_fracture_dataset/dcm/'
os.makedirs(savedirr, exist_ok=True)
bash_script2 = 'retrieve_dcmtk_by-acc.sh'

def has_dx_files(directory):
    for root, dirs, files in os.walk(directory):
        if any(file.lower().startswith('dx') for file in files):
            return True
    return False

def execute_command(cmd, savedir):
    if not has_dx_files(savedir):
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logging.info(f"Successfully executed command for {savedir}")
            print(f"Completed: {savedir}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error executing command for {savedir}: {str(e)}")
            print(f"Error: {savedir}")
    else:
        logging.info(f"Skipped {savedir} as it contains files starting with 'dx' in subfolders")
        print(f"Skipped: {savedir}")

commands = []
for row in df2.iterrows():
    accession = row[1]['Accession Number']
    savedir = os.path.join(savedirr, str(accession))
    os.makedirs(savedir, exist_ok=True)
    
    cmd = ["bash", bash_script2, str(accession), savedir]
    commands.append((cmd, savedir))

print(f"Total commands to execute: {len(commands)}")

with ThreadPoolExecutor() as executor:
    executor.map(lambda x: execute_command(*x), commands)

print("All commands have been processed.")
