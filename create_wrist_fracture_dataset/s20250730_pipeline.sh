### FOR FUTURE: 
# a better way to handle extraction 
# 0. extract .excel spreadsheet with studyids. Filter them by those that have XRWRIST or XRWRST in the report description 
# 1. extract all the dcm files 
# 2. extract reports 
# 3. check if any reports are empty and exclude them 
# 4. only extract pngs for dcms that have reports (into 'fixed' folder)
# 5. remove pngs that are empty 
# 6. copy reports to one directory 
# 7. copy pngs to one directory 

# 8. run yolo on all pngs (now we have to run them on those that had inverted contrast)
    # 
# 9. 


# copy the scripts to a new folder 
cd /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset
mkdir -p extraction_aug_2025
cp -n extract_dicoms_script.py remove_empty_pngs2.py convert_dataset_process_removed.py convert_dataset_skip_bad_files.py extraction_aug_2025
cp -n copy_files_to_single_dir.py copy_reports_to_single_dir.py remove_empty_pngs_final_folder.py extract_core_text.py extraction_aug_2025
cp retrieve_dcmtk_by-acc.sh extraction_aug_2025/

# copy the excel files into one giant excel file 
ipython 
import pandas as pd
df1 = pd.read_excel("../wrist facture 01 jan 2018 till 31 dec 2021.xlsx")
df2 = pd.read_excel("../wrist facture 14 aug 2024 till29 jul 2025.xlsx")
merged_df = pd.concat([df1, df2], ignore_index=True)
merged_df = merged_df.drop_duplicates()
merged_df.to_excel("wrist_fracture_2018_01_01_to_2025_07_29_excluding_2022_01_01_to_2024_08_13.xlsx", index=False)
print(f"Total rows: {len(merged_df)}")
# 33455

# look at number of available dirs 
f = 'wrist_fracture_2018_01_01_to_2025_07_29_excluding_2022_01_01_to_2024_08_13.xlsx'
df = pd.read_excel(f)
df2 = df.drop_duplicates(subset='Accession Number', keep="first")
    print(f"original merged excel: {len(df)}")
    print(f"after dropping duplicates: {len(df2)}")
    # original merged excel: 33455
    # after dropping duplicates: 33039


# create a space on external body to hold all the dcm extracts ... 
/fileserver/external/body/serge/llm/datasets/wrist_aug2025



# start procesing
cd /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset
cd extraction_aug_2025



# extract dicoms from pacs 
python extract_dicoms_script.py

    # find all entries that did not execute
    grep -v "Successfully executed command" extract_dicoms.log | wc -l 


    # find all entries 
    grep "Successfully executed command" extract_dicoms.log | wc -l
    # 33039 


# or run the whole pipeline - which will run everything - but will stop 
            # python convert_dataset_skip_bad_files_parallel.py --overwrite
python convert_dataset_skip_bad_files_parallel2.py --overwrite # inverts and puts ALL png files into 'fixed' directory
            # find failed
            find logs -name "failed_conversions_20250731_*" -type f ! -empty | wc -l 
            # 379

            # find empty (i.e. success) processed 
            find logs -name "failed_conversions_20250731_*" -type f -empty | wc -l 
            # 1809


# remove pngs that are empty (bad conversion results in empty files)
python remove_empty_pngs2.py 

# either re-run for specific items 
python convert_dataset_process_removed.py # (this will read from 'removed_files.txt' )


# find out how many directories are simply empty: 150 scans! 
python copy_files_to_single_dir.py 
python copy_reports_to_single_dir.py 
# python copy_files_to_single_dir_parallel_wrist_only.py  # this is bad. 
    # results stored here
    empty_folders.txt


# this is us trying to deal separately with inverted files 
    python copy_pngs_from_csv_studies.py --inline # this copies only the files where the correct study ids are in thef csv file 
    python move_pngs_keep_only_matching_to_reports.py # move away pngs that do not have corresponding report (for whatever reason the report was not extracted) # one mistake we do is we look at all dcm files - instead we should have written it to only generate images for directories that have reports... 

    python identify_pngs_where_inversion_needed.py  # identifies which files were actually inverted (looking at file sizes)
    python prep_images_for_yolo_on_inverted_pngs_from_png_csv.py # creates symlink only to files that were inverted 

            # THIS IS SUPER BAD FILE 
            # python identify_studyids_where_inversion_needed.py # this will create a csv file with the studyids that need inversion 
# now we need to re-run yolo using this csv (or its derivatives)
python remove_bad_study_ids.py

# remove empty files from the final list -> removed around 70 files (thats anythign that is remaining over)
python remove_empty_pngs_final_folder.py
    # results stored here 
    removed_files_final_folder.txt

# now to process the images we need 4 gpus 

# and to process the reports we need ollama... 


# request gpus 
srun -A crl -p bch-gpu-xlarge -t 24:00:00 --gres=gpu:1 --cpus-per-task=8 --mem=32G --pty /bin/bash
srun -A crl -p crl-gpu -t 144:00:00 --gres=gpu:4 --cpus-per-task=32 --mem=64G --pty /bin/bash # 6days - 1aug 3am London to 7 aug 3am London 
srun -A crl -p bch-gpu -t 24:00:00 --gres=gpu:1 --cpus-per-task=8 --mem=32G --pty /bin/bash




# Show performance on images that have been reduced by 1/3 of the size 


# Show LLM, VLM and Region hallucinations 

    



#### EXTRACTION 
    # GOOD 
        python s20250801_analyze_hallucinations_another_version.py


    # BAD 
        $ python s20250801_review_hallucinations.py
        Total findings: 12437
        Unique cases: 8040
        Fracture findings: 11241
        No fracture findings: 1196
        Estimated hallucination rate: 31.33%
        Most common bone: radius
        Most common region: distal





# running yolo on allthe images  -> see 
 >>> s20250802_testing_yolo_inf_settings.sh

                    srun -A crl -p bch-gpu -t 24:00:00 --gres=gpu:1 --cpus-per-task=8 --mem=32G --pty /bin/bash
                    conda activate llava-med
                    cd ~/w/code/llm/experiments/yolov7/
                    source venv/bin/activate 
                    cd yolov7/
                    o=/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/pngs_aug2025_yolo_pred/
                    fo=/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/pngs_aug2025/
                    python detect.py --weights yolov7-p6-bonefracture.pt --conf 0.25 --img-size 1280 --source $fo/ --save-txt --save-conf --project $o --name run1 --classes 3 --exist-ok


                    # see 


# create a csv file with the reports and the images 
python create_data_csv.py

    Summary Statistics:
    Total unique reports: 16923
    Total images: 46797
    Average images per report: 2.77
    Reports with fractures: 30755
    Reports without fractures: 16042

    fracture_dataset.csv
    reports_without_images.txt


# remove ids that do not coincide with wrists and other things... 
python remove_bad_study_ids.py 


# now extract the image names again and check them in collab for weird names etc. 
cd ~/ww/code/llm/experiments/LISA/experiments/labeller_running_for_andy/s20250711_generate_structured_reports
create_data_csv.py

# now go to colab and load this file - 
load_dataset_of_extracted_reports_images_final_count_fractures_and_make_sure_correct.ipynb
    # copy here - 

# now check correspondence between reports and images
cd /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/
json_source_folder='/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/experiments/labeller_running_for_andy/s20250711_generate_structured_reports/outputs/grounded_33k/puyangwang-medgemma-27b-it-q8-reports_aug2025/' # Replace with the actual path to your JSON files
png_source_folder='/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/pngs_aug2025/'
python s20250808_check_reports_vs_images.py --reports $json_source_folder --images $png_source_folder

    # THIS FILE IS BROKEN BECAUSE WE HAVE 35413 images (not 12553)

    === Summary ===
    Total reports: 10566
    Total unique report IDs from images: 12553
    Reports without images: 230
    Images without reports: 2217
    Report IDs that appear in both: 10336

    📈 Relationship Analysis:
    - Reports with at least one image: 10336
    - Image report IDs with corresponding reports: 10336
    - Orphaned image report IDs: 2217
    - Average images per report: 1.21

            === Checking if reports have corresponding images ===
            ❌ Found 230 reports without corresponding images:
            Examples of reports without images:
            - 25110721
            - 25116221
            - 25116948
            - 25118621
            - 25119648
            - 25119692
            - 25120326
            - 25123545
            - 25123648
            - 25127051
            ... and 220 more
            📊 Reports with at least one image: 10336

            === Checking if images have corresponding reports ===
            ❌ Found 2217 image report IDs without corresponding reports:
            Examples of image report IDs without reports:
            - 120953 (from images)
            - 25109542 (from images)
            - 25109696 (from images)
            - 25110402 (from images)
            - 25110817 (from images)
            - 25111270 (from images)
            - 25112143 (from images)
            - 25114019 (from images)
            - 25114369 (from images)
            - 25115572 (from images)
            ... and 2207 more
            📊 Image report IDs with corresponding reports: 10336


            ls $png_source_folder/25110721*
            ls $json_source_folder/25110721*


# need to remove reports without images (or )
cd /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/experiments/labeller_running_for_andy


# count the number of files 
cd /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/experiments/labeller_running_for_andy/s20250711_generate_structured_reports/
json_source_folder='/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/experiments/labeller_running_for_andy/s20250711_generate_structured_reports/outputs/grounded_33k/puyangwang-medgemma-27b-it-q8-reports_aug2025/' # Replace with the actual path to your JSON files
python s20250811_grounding_report_split_regions.py --folder-path $json_source_folder --save-folder "_v2"



# this is how we filter before starting to train 
/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/experiments/20250116_report_filtering_v2.sh

# and this is how we train i think 
/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/experiments/20250119_finetuning_v3_submit_training_jobs_set3b.sh