##################################################

please edit the file called convert_dataset.py file in the following way - 



here is an example of a structure of one of the subfolders the main_directory (main_directory that is given in the code as '/home/ch215616/w/code/llm/experiments/yolov7/wrist_fracture_dataset/dcm') 

$ tree <SCAN_ID>
<SCAN_ID>
├── <STUDY_ID>
│   └── <SCAN_ID>
│       ├── 1_PA
│       │   ├── DX.<DICOM_UID_1>
│       │   └── DX.<DICOM_UID_1>.~1~
│       ├── 2_Oblique
│       │   ├── DX.<DICOM_UID_2>
│       │   └── DX.<DICOM_UID_2>.~1~
│       ├── 3_Lateral
│       │   ├── DX.<DICOM_UID_3>
│       │   └── DX.<DICOM_UID_3>.~1~
│       └── 999_FUJI Basic Text SR for HL7 Radiological Report
│           ├── <DICOM_SR_UID>
│           └── <DICOM_SR_UID>.~1~
└── STUDY_INFO
    └── rsp0001.dcm

7 directories, 9 files

please note that the depth of folders and subfolders and files may vary in each case. but the general structure of processing that you need to edit is the following - 


edit the code to adhere to the following: 

1. for top level subdirectories that are found within main_directory  (in the example above it is '<SCAN_ID>')  - only consider folders that start with a number. we will assign a variable called 'scan_id' to it. e.g. in example above scan_id = <SCAN_ID>.

2. find all the files for each scan_id subdirectory. please note that files can be at different depths of subdirectories inside the scan_id (the depth may vary). 


a) We are interested only in files that start with DX. 
b) We are NOT interested in files that contain '~'' in the name (in fact these should be removed - as the script is running). 

for each file that starts with 'dx' - we will add the following information to the pandas dataframe - 

- scan_id  
- full path to that file 
- note down the name of the directory in which this file is located in (the directory in which it is immediately placed) and call it 'scan_type'. e.g. in the example above for file called 'DX.1.3.46.670589.30.966169792574.4976.1641822575188' it would be '1_PA' directory which becomes scan_type. 

- then we need to run convert_dicom_to_png function on the dx file. please note below the changes we need to make to convert_dicom_to_png function. 
 

we need to modify the naming convention of the png_filename that is sent to that function. It should be the following - 

- the png file should be saved inside the scan_id directory
- the png file should be named as 'scan_id'-'scan_type'-'png-count'.png 
png-count refers to the number of png inside the scan_id directory - this should be incremented as more and more pngs are being put in these (from different scan_type folders)


in addition to this, for each scan_id we need to fetch the report. the report would always start with 999 - as given in current code. please make sure report is placed in scan_id folder and is named as f"{accession_number}_report.txt" - please note that 'accession_number' is technically the same as 'scan_id' - please rename every variable called 'accession_number' to 'scan_id' in the current code. 



finally - you should keep in tact the no_png_log and no_report_log - these are important. 

