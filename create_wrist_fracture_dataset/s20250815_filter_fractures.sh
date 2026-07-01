

# Filter new dataset reports by desirable categories for training 


            # turn JSON reports into .csv 
                        # step 1: build csv 
                                    cd /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/experiments/labeller_running_for_andy/s20250711_generate_structured_reports/                        
                                    python grounding_report_split_regions_s20250821_build_csv.py # fetch all jsons into csv (only read the categories that we want)
                        # step 2: analyze csv 
                                    python grounding_report_split_regions_s20250821_analyze.py  # aggregate these csvs into statistics and show results                 

                        # check what missing_pngs.csv and wrong_yolo_results.csv
                                    25110721
                                    ls /lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/pngs_aug2025/missing_pngs.csv
                                    ls /lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/pngs_aug2025/wrong_yolo_results.csv

                        # stats of missing data 
                                    - 11366 - wrong yolo results (how do we calculate this???)
                                    - 17707 - actual data (but multiple fractures per each case)
                                    - 231 - missing pngs 




            # view available categories and numbers (for new dataset of 29000 images and ~10k reports) 
                        $     python grounding_report_split_regions_s20250821_create_filtered_categories_csv.py \
                                --input_csv grounded_s20250728_split_regions.csv \
                                --output_csv ready_for_symlinks_new_dataset_ALL.csv \
                                --show_available_options
                        Reading CSV to show available filtering options...

                        Available anatomical regions:
                        distal_radius_shaft    7299
                        distal_ulna_shaft      2871
                        not_applicable         2750
                        ulnar_styloid          1685
                        scaphoid                418
                        unknown                 137
                        other                    98
                        radial_styloid           21
                        radius_other             21
                        proximal_radius           4
                        ulna_other                2
                        proximal_ulna             1

                        Available healing categories:
                        healing           6008
                        unknown           4840
                        not_applicable    2750
                        acute             1325
                        healed             255
                        nonunion           129

                        Available classification categories:
                        unknown             5976
                        not_applicable      2750
                        transverse          2307
                        salter_harris_II    2263
                        buckle              1244
                        comminuted           244
                        oblique              226
                        avulsion              99
                        other                 98
                        salter_harris_IV      39
                        greenstick            36
                        salter_harris_I       25

                        Available alignment categories:
                        well_aligned            6712
                        unknown                 3347
                        not_applicable          2750
                        acceptable_alignment    2171
                        poor_alignment           327

                        Available immobilization types:
                        unknown           7588
                        cast              4959
                        not_applicable    2750
                        other               10

                        Available has_hardware values:
                        False    14371
                        True       936

                        Available fracture counts:
                        0    2750
                        1    3419
                        2    8180
                        3     813
                        4     140
                        5       5

                        Available has_fracture values:
                        True     12557
                        False     2750

            # Create categories (1 fracture cases only)

                        # Generate the no fracture cases once (same for all scenarios)
                        python grounding_report_split_regions_s20250821_create_filtered_categories_csv.py \
                            --input_csv grounded_s20250728_split_regions.csv \
                            --output_csv ready_for_symlinks_nofracture.csv \
                            --fracture_counts 0

                            - Total unique file IDs: 2750

                        # 1. Anatomical Regions - 1 fracture cases only
                        python grounding_report_split_regions_s20250821_create_filtered_categories_csv.py \
                            --input_csv grounded_s20250728_split_regions.csv \
                            --output_csv ready_for_symlinks_anatomical_regions_1fracture.csv \
                            --anatomical_regions distal_radius_shaft distal_ulna_shaft ulnar_styloid scaphoid \
                            --fracture_counts 1

                            - Total unique file IDs: 3290

                        # 2. Classification - 1 fracture cases only
                        python grounding_report_split_regions_s20250821_create_filtered_categories_csv.py \
                            --input_csv grounded_s20250728_split_regions.csv \
                            --output_csv ready_for_symlinks_classification_1fracture.csv \
                            --classification_categories transverse salter_harris_II buckle comminuted avulsion \
                            --fracture_counts 1

                            - Total unique file IDs: 1982

                        # 3. Healing - 1 fracture cases only
                        python grounding_report_split_regions_s20250821_create_filtered_categories_csv.py \
                            --input_csv grounded_s20250728_split_regions.csv \
                            --output_csv ready_for_symlinks_healing_1fracture.csv \
                            --exclude_healing_categories unknown not_applicable \
                            --fracture_counts 1

                            - Total unique file IDs: 2118

                        # 4. Alignment - 1 fracture cases only
                        python grounding_report_split_regions_s20250821_create_filtered_categories_csv.py \
                            --input_csv grounded_s20250728_split_regions.csv \
                            --output_csv ready_for_symlinks_alignment_1fracture.csv \
                            --exclude_alignment_categories unknown not_applicable \
                            --fracture_counts 1

                            - Total unique file IDs: 2752


            # Merge no fractures with categories

                        # Merge anatomical regions  
                        python -c "
                        import pandas as pd
                        df1 = pd.read_csv('ready_for_symlinks_anatomical_regions_1fracture.csv')
                        df2 = pd.read_csv('ready_for_symlinks_nofracture.csv')
                        merged = pd.concat([df1, df2], ignore_index=True)
                        merged.to_csv('ready_for_symlinks_anatomical_regions_01fracture.csv', index=False)
                        print(f'Anatomical regions: {len(df1)} + {len(df2)} = {len(merged)} cases')
                        "

                        6040 cases 

                        # Merge classification
                        python -c "
                        import pandas as pd
                        df1 = pd.read_csv('ready_for_symlinks_classification_1fracture.csv')
                        df2 = pd.read_csv('ready_for_symlinks_nofracture.csv')
                        merged = pd.concat([df1, df2], ignore_index=True)
                        merged.to_csv('ready_for_symlinks_classification_01fracture.csv', index=False)
                        print(f'Classification: {len(df1)} + {len(df2)} = {len(merged)} cases')
                        "

                        4732 cases 

                        # Merge healing
                        python -c "
                        import pandas as pd
                        df1 = pd.read_csv('ready_for_symlinks_healing_1fracture.csv')
                        df2 = pd.read_csv('ready_for_symlinks_nofracture.csv')
                        merged = pd.concat([df1, df2], ignore_index=True)
                        merged.to_csv('ready_for_symlinks_healing_01fracture.csv', index=False)
                        print(f'Healing: {len(df1)} + {len(df2)} = {len(merged)} cases')
                        "

                        4868 cases 

                        # Merge alignment
                        python -c "
                        import pandas as pd
                        df1 = pd.read_csv('ready_for_symlinks_alignment_1fracture.csv')
                        df2 = pd.read_csv('ready_for_symlinks_nofracture.csv')
                        merged = pd.concat([df1, df2], ignore_index=True)
                        merged.to_csv('ready_for_symlinks_alignment_01fracture.csv', index=False)
                        print(f'Alignment: {len(df1)} + {len(df2)} = {len(merged)} cases')
                        "

                        5502 cases 


            # Create simlinks for training yolo (and corresponding labels)
                        # create symlinks 
                        c=/home/ch215616/ww/code/llm/experiments/yolov7/create_wrist_fracture_dataset/grounding_report_split_regions_s20250821_create_symlinks.py
                        # image_dir=/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/pngs_aug2025/ # bad file - many pngs here not fixed
                        image_dir=/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/pngs_aug2025_fixed
                        label_dir=/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/yolov7/ALL/run1/labels/
                        csvdir=/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/experiments/labeller_running_for_andy/s20250711_generate_structured_reports/
                        savedir=/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/pngs_aug2025/

                        # Create symlinks for anatomical regions dataset
                        name=anatomical_regions_01fracture
                        mkdir -p $savedir/$name/
                        python $c --input_csv $csvdir/ready_for_symlinks_anatomical_regions_01fracture.csv --image_dir $image_dir --label_dir $label_dir \
                        --output_dir $savedir/$name

                                    === PROCESSING COMPLETE ===
                                    Total reports processed: 6040
                                    Total images processed: 17556
                                    Successful symlinks created: 14325
                                    Empty label files created: 8721
                                    Missing PNG files: 199
                                    Wrong YOLO results: 3231


                                    === PROCESSING COMPLETE ===
                                    Total reports processed: 6040
                                    Total images processed: 17568
                                    Successful image symlinks created: 14330
                                    Successful label files copied: 5605
                                    Empty label files created: 8725
                                    Missing PNG files: 196
                                    Wrong YOLO results: 3238                                    

                        # Create symlinks for classification dataset
                        name=classification_01fracture
                        mkdir -p $savedir/$name/
                        python $c --input_csv $csvdir/ready_for_symlinks_classification_01fracture.csv --image_dir $image_dir --label_dir $label_dir \
                        --output_dir $savedir/$name


                                    === PROCESSING COMPLETE ===
                                    Total reports processed: 4732
                                    Total images processed: 13924
                                    Successful symlinks created: 12340
                                    Empty label files created: 8721
                                    Missing PNG files: 167
                                    Wrong YOLO results: 1584    

                                    === PROCESSING COMPLETE ===
                                    Total reports processed: 4732
                                    Total images processed: 13929
                                    Successful image symlinks created: 12344
                                    Successful label files copied: 3619
                                    Empty label files created: 8725
                                    Missing PNG files: 166
                                    Wrong YOLO results: 1585                                                        

                        # Create symlinks for healing dataset
                        name=healing_01fracture
                        mkdir -p $savedir/$name/
                        python $c --input_csv $csvdir/ready_for_symlinks_healing_01fracture.csv --image_dir $image_dir --label_dir $label_dir \
                        --output_dir $savedir/$name

                                    === PROCESSING COMPLETE ===
                                    Total reports processed: 4868
                                    Total images processed: 14263
                                    Successful symlinks created: 12233
                                    Empty label files created: 8721
                                    Missing PNG files: 184
                                    Wrong YOLO results: 2030    


                                    === PROCESSING COMPLETE ===
                                    Total reports processed: 4868
                                    Total images processed: 14272
                                    Successful image symlinks created: 12237
                                    Successful label files copied: 3512
                                    Empty label files created: 8725
                                    Missing PNG files: 182
                                    Wrong YOLO results: 2035                                                        

                        # Create symlinks for alignment dataset
                        name=alignment_01fracture
                        mkdir -p $savedir/$name/
                        python $c --input_csv $csvdir/ready_for_symlinks_alignment_01fracture.csv --image_dir $image_dir --label_dir $label_dir \
                        --output_dir $savedir/$name


                                    === PROCESSING COMPLETE ===
                                    Total reports processed: 5502
                                    Total images processed: 16188
                                    Successful symlinks created: 13362
                                    Empty label files created: 8721
                                    Missing PNG files: 200
                                    Wrong YOLO results: 2826        


                                    # new 
                                    === PROCESSING COMPLETE ===
                                    Total reports processed: 5502
                                    Total images processed: 16196
                                    Successful image symlinks created: 13367
                                    Successful label files copied: 4642
                                    Empty label files created: 8725
                                    Missing PNG files: 198
                                    Wrong YOLO results: 2829                                                    





########################################################################################
# Current todo
########################################################################################


                        # now let's filter only those that have 1 fracture or zero fractures... 
                            # then train just with those... 
                        # then let's setup some sort of fancy script to train those that have 2 fractures 
                            # then train with those + the ones that have 1 fracture 
                        # finally if there is time we should fix the dicoms on the old cohort (the inversions), then merge them with our files and then train altogether... 

                        # ok - let's filter those with one fracture only for now... 

        
