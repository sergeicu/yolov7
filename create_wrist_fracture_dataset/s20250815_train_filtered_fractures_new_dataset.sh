This file follows s20250815_filter_fractures.sh
Goal is to train yolo with the filtered dataset 
We have the following categories:

            ## Dataset Summary

            **Overall Dataset Statistics:**
            - **Total original reports:** 15,307 (from fracture counts: 2750 no fracture + 12,557 with fractures)
            - **No fracture cases:** 2,750 (used as baseline for all categories)

            **Filtered Categories Created:**

            **Anatomical Regions Filter:**
                        - **Included regions:** distal_radius_shaft, distal_ulna_shaft, ulnar_styloid, scaphoid
                        - **1 fracture cases:** 3,290 files
                        - **No fracture cases:** 2,750 files
                        - **Total:** 6,040 files
                        - **Images processed:** 17,556
                        - **Successful symlinks:** 14,325

            **Classification Filter:**
                        - **Included classifications:** transverse, salter_harris_II, buckle, comminuted, avulsion
                        - **1 fracture cases:** 1,982 files
                        - **No fracture cases:** 2,750 files
                        - **Total:** 4,732 files
                        - **Images processed:** 13,924
                        - **Successful symlinks:** 12,340

            **Healing Filter:**
                        - **Excluded categories:** unknown, not_applicable
                        - **1 fracture cases:** 2,118 files
                        - **No fracture cases:** 2,750 files
                        - **Total:** 4,868 files
                        - **Images processed:** 14,263
                        - **Successful symlinks:** 12,233

            **Alignment Filter:**
                        - **Excluded categories:** unknown, not_applicable
                        - **1 fracture cases:** 2,752 files
                        - **No fracture cases:** 2,750 files
                        - **Total:** 5,502 files
                        - **Images processed:** 16,188
                        - **Successful symlinks:** 13,362

            **Data Quality Notes:**
                        - **Empty label files:** 8,721 (consistent across all categories - these are no fracture cases)
                        - **Missing PNG files:** 167-200 per category
                        - **Wrong YOLO results:** 1,584-3,231 per category (varies by filter) 


Next we prepar yolo script: create yaml files: 
            # step 1: create yaml files 
            # step 2: update labels to have correct classes. 
            # step 3: split into train, val, test (via symlinks)
            # step 4: train yolo 


            # step 1: create yaml files 

                        python s20250815_create_yaml_files_fixed.py 

                        # 1. Anatomical Regions Filter YAML
                        names: ["distal_radius_shaft", "distal_ulna_shaft", "ulnar_styloid", "scaphoid", "no_fracture"]

                        # 2. Classification Filter YAML
                        names: ["transverse", "salter_harris_II", "buckle", "comminuted", "avulsion", "no_fracture"]

                        # 3. Healing Filter YAML
                        names: ["healing", "acute", "healed", "nonunion", "no_fracture"]

                        # 4. Alignment Filter YAML
                        names: ["well_aligned", "acceptable_alignment", "poor_alignment", "no_fracture"]

            # step 2: update labels to have correct classes. 
                        # python s20250815_train_filtered_fractures_new_dataset_update_labels.py --test_set


                        # b1=/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/pngs_aug2025/
                        # b2=/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/experiments/labeller_running_for_andy/s20250711_generate_structured_reports/
                        # b3=/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/yolov7/data


                        # python s20250815_train_filtered_fractures_new_dataset_update_labels.py \
                        #     --dataset_dir $b1/alignment_01fracture \
                        #     --csv_path $b2/ready_for_symlinks_alignment_01fracture.csv \
                        #     --yaml_path $b3/s20250815_alignment_filter.yaml \
                        #     --test                       


                        # full
                        python s20250815_train_filtered_fractures_new_dataset_update_labels.py

                        # test 
                        python s20250815_train_filtered_fractures_new_dataset_update_labels.py \
                        --category alignment \
                        --test 

            # step 3: split into train, val, test (via symlinks)

                    # Define paths
                    split_script=/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/s20250815_train_filtered_fractures_new_dataset_split_test_val.py
                    base_dataset_dir=/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/pngs_aug2025/

                    # Split anatomical regions dataset (6040 total files)
                    echo "Splitting anatomical regions dataset..."
                    python $split_script \
                        --dataset_dir $base_dataset_dir/anatomical_regions_01fracture \
                        --val_size 100 --test_size 100 --use_absolute_numbers --seed 42

                        # Train set: 14125 files

                    # Split classification dataset (4732 total files)
                    echo "Splitting classification dataset..."
                    python $split_script \
                        --dataset_dir $base_dataset_dir/classification_01fracture \
                        --val_size 100 --test_size 100 --use_absolute_numbers --seed 42

                        # Train set: 12140

                    # Split healing dataset (4868 total files)
                    echo "Splitting healing dataset..."
                    python $split_script \
                        --dataset_dir $base_dataset_dir/healing_01fracture \
                        --val_size 100 --test_size 100 --use_absolute_numbers --seed 42

                        # Train set: 12033

                    # Split alignment dataset (5502 total files)
                    echo "Splitting alignment dataset..."
                    python $split_script \
                        --dataset_dir $base_dataset_dir/alignment_01fracture \
                        --val_size 100 --test_size 100 --use_absolute_numbers --seed 42

                        # Train set: 13162

                    echo "Dataset splitting complete for all categories!"
                    echo "Each dataset now has train/val/test splits with 100 files each for val and test sets"





also - let's check that our pngs are not inverted...

then start train (davno pora) 

