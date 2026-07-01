# pipeline: 


# extract dicoms from pacs 
python extract_dicoms_script.py

# remove pngs that are empty (bad conversion results in empty files)
python remove_empty_pngs2.py 

# either re-run for specific items 
python convert_dataset_process_removed.py # (this will read from 'removed_files.txt' )

# or run the whole pipeline - which will run everything - but will stop 
python convert_dataset_skip_bad_files.py
    # results stored here 
    removed_files_consistently_fails.txt # there are 21 scans that consistently fail

    # the generated fail is the following: 
    ValueError: The length of the pixel data in the dataset (0 bytes) doesnt match the expected length (1719326 bytes). The dataset may be corrupted or there may be an issue with the pixel data handler.
    # test again 
    cd ~/w/code/llm/experiments/yolov7
    python dicom-to-png/mritopng.py wrist_fracture_dataset/dcm/<SCAN_ID>/<STUDY_ID>/<SCAN_ID>/1_PA/<DICOM_FILE> wrist_fracture_dataset/dcm/<SCAN_ID>/remove.png
    # removing the duplicates (~) dcm files. then re-running the dcm fetch pipeline and then re-running conversion helps... 
    # this is manual step - will bring extra 21 subjects into the dataset (out of 14,000....) with possibly 3-4 images in each subject

# find out how many directories are simply empty: 150 scans! 
python copy_files_to_single_dir.py 
python copy_reports_to_single_dir.py 
    # results stored here
    empty_folders.txt

# remove empty files from the final list -> removed around 70 files (thats anythign that is remaining over)
python remove_empty_pngs_final_folder.py
    # results stored here 
    removed_files_final_folder.txt


# extract impressions (only works for basic example) -> this was later substituted by llama script
python extract_core_text.py



# files that are not mentioned in this pipeline 
    # useless 
    check_mission_impressions.py
    convert_dataset_instructions.py
    convert_dataset_parallel.py

    # useful 
    convert_dataset.py  # predcessor to its variants 
    retrieve_dcmtk_by-acc.sh  # original script 
    wrist facture Jan1 2022 till Aug13 2024.xlsx # actual dataset we downloading
    yolov7-grazped-inferrence.sh # how to perform inferrene with grazped! 








END 
######################################################################














######################################################################
MISC - removed 
# next: run inferrence pipeline... with yolo - but 
# instructions are here - but we need to verify how do we mass run it and save in appropriate place 
yolov7-grazped-inferrence.sh


# only pick files where bounding box was drawn... 


# OTHER FILES: 
draw_bounding_boxes.py # draws bounding boxes from coordinates (red ones)