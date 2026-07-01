########################################################


TOTAL NUMBER OF REPORTS IN THE EXCEL FILE


########################################################


cd /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset
ipython 

import pandas as pd

# Read the Excel file
df = pd.read_excel('wrist facture Jan1 2022 till Aug13 2024.xlsx')

# Count total entries
total_entries = len(df)
# 14018 

# Count unique entries (removing duplicates)
unique_entries = len(df.drop_duplicates(subset='Accession Number', keep="first"))
# 14018 

print(f"Total entries: {total_entries}")
print(f"Unique entries (after removing duplicates): {unique_entries}")
print(f"Duplicate entries: {total_entries - unique_entries}")




########################################################


TOTAL NUMBER OF PROCESSED REPORTS


########################################################
iphython 
import os

def count_reports():
    count = 0
    main_dir = '/home/ch215616/w/code/llm/experiments/yolov7/wrist_fracture_dataset/dcm'
    for scan_id in os.listdir(main_dir):
        if scan_id[0].isdigit():
            report_path = os.path.join(main_dir, scan_id, f"{scan_id}_report.txt")
            if os.path.exists(report_path):
                count += 1
                print(f"Found report {count}: {scan_id}")
    return count

print(f"Total reports: {count_reports()}")
# 13879 total number of reports  (which means that 139 are missing for one reason or another) 



########################################################


Find out which exam codes are used in the excel file to get xray of the wrist 


########################################################

import pandas as pd
import re

# Read the Excel file
df = pd.read_excel("wrist facture Jan1 2022 till Aug13 2024.xlsx")

# Get unique values in 'Exam Code' column
unique_exam_codes = df['Exam Description'].unique()

print("All unique Exam Codes:")
for code in unique_exam_codes:
    print(f"  {code}")

# Count codes that match 'XR-WRIST' pattern
xr_wrist_codes = [code for code in unique_exam_codes if 'WRIST' in str(code)]
non_xr_wrist_codes = [code for code in unique_exam_codes if 'WRIST' not in str(code)]

print(f"\nCodes matching 'WRIST' pattern ({len(xr_wrist_codes)}):")
for code in xr_wrist_codes:
    print(f"  {code}")

print(f"\nCodes NOT matching 'WRIST' pattern ({len(non_xr_wrist_codes)}):")
for code in non_xr_wrist_codes:
    print(f"  {code}")

print(f"\nSummary:")
print(f"  Total unique exam codes: {len(unique_exam_codes)}")
print(f"  Matching 'WRIST': {len(xr_wrist_codes)}")
print(f"  NOT matching 'WRIST': {len(non_xr_wrist_codes)}")


# Codes matching 'WRIST' pattern (4):
#   XR WRIST 3+ VIEWS RIGHT
#   XR WRIST 3+ VIEWS LEFT
#   XR WRIST 1-2 VIEWS LEFT
#   XR WRIST 1-2 VIEWS RIGHT

# Codes NOT matching 'WRIST' pattern (7):
#   XR-Wrist 3+ Views Right
#   XR-Wrist 3+ Views Left
#   XR-Wrist 2 Views Right
#   XR-Wrist 2 Views Left
#   XR-Wrist 3+ Views
#   XR-Wrist 2 Views
#   XR Wrist

########################################################


New lets find out latest and earliest dates 


########################################################

# old 
df = pd.read_excel("wrist facture Jan1 2022 till Aug13 2024.xlsx")
print(f"Earliest: {df['Exam Completed Date'].min()}")
print(f"Latest: {df['Exam Completed Date'].max()}")
# Earliest: 2022-01-03 00:54:14
# Latest: 2024-08-13 15:09:03




df = pd.read_excel("wrist facture 14 aug 2024 till29 jul 2025.xlsx")
print(f"Earliest: {df['Exam Completed Date'].min()}")
print(f"Latest: {df['Exam Completed Date'].max()}")
# Earliest: 2024-08-14 00:46:28
# Latest: 2025-07-29 14:53:00
len(df)
# 8455 more reports (and we already have 13879) (for some reason only 8000 of them are processed with yolo right?)


df = pd.read_excel("wrist facture 01 jan 2018 till 31 dec 2021.xlsx")
print(f"Earliest: {df['Exam Completed Date'].min()}")
print(f"Latest: {df['Exam Completed Date'].max()}")
# Earliest: 2018-01-01 18:15:33
# Latest: 2021-12-31 22:33:02
len(df)
# 25000 more reports 


# total number of reports that would be available (before processing) 
25000 + 8455 + 13879 = 47334

# let's process these reports here 
see s20250730_pipeline.sh