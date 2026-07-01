# all file are here 
/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/testing_andy_labelled_images

###### COPY FILES ######
    
    #### COPY IMAGES WITH CORRECT LABELS (VISUAL ASSESSMENT) ######
    ls /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/experiments/labeller_running_for_andy/outputs_backup_dec31/andys_labels_final_boxes_all_images/*.png | xargs -n1 basename > /tmp/png_filenames.txt
    cd /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/testing_andy_labelled_images/andys_labels_final_boxes_all_images
    while read filename; do
        if [ -f "/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/experiments/labeller_running_for_andy/outputs_backup_dec31/andys_labels_final_boxes_all_images/$filename" ]; then
            ln -sf "/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/experiments/labeller_running_for_andy/outputs_backup_dec31/andys_labels_final_boxes_all_images/$filename" "$filename"
        fi
    done < /tmp/png_filenames.txt
    
    #### COPY CLEAN IMAGES ######
    ls /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/experiments/labeller_running_for_andy/outputs_backup_dec31/inference_clean/*.png | xargs -n1 basename > /tmp/png_filenames.txt
    cd /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/testing_andy_labelled_images/images_clean
    while read filename; do
        if [ -f "/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/pngs/$filename" ]; then
            ln -sf "/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/pngs/$filename" "$filename"
        fi
    done < /tmp/png_filenames.txt    



    #### COPY LABELS ######
    source=/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/experiments/labeller_running_for_andy/outputs_backup_dec31/andys_labels_final_boxes_all
    indir=/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/testing_andy_labelled_images/andys_labels_final_boxes_all/
    mkdir -p $indir
    cp -rn $source/* $indir/
    # now run inference on these images with different settings - name the directories as they should 






# init 
cd ~/w/code/llm/experiments/yolov7/
source venv/bin/activate 
cd yolov7/
source=/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/pngs_aug2025/



###### RUN INFERENCE ######
        #!/bin/bash

        # YOLO Performance Testing Script
        # Tests different combinations of image size, confidence threshold, and IoU threshold

        # Configuration
        WEIGHTS="yolov7-p6-bonefracture.pt"
        SOURCE="/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/testing_andy_labelled_images/images_clean/"
        tt=/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/
        PROJECT=$tt/"testing_andy_labelled_images"
        CLASSES="3"

        # Parameter arrays
        # IMG_SIZES=(640 960 1280 1600)
        # CONF_THRESHOLDS=(0.15 0.25 0.35)
        # IOU_THRESHOLDS=(0.35 0.45 0.55)
        IMG_SIZES=(1280)
        CONF_THRESHOLDS=(0.35 0.45)
        IOU_THRESHOLDS=(0.45 0.55 0.65)

        # Counter for experiment numbering
        exp_counter=18

        echo "Starting YOLO performance testing..."
        echo "Total experiments: $(( ${#IMG_SIZES[@]} * ${#CONF_THRESHOLDS[@]} * ${#IOU_THRESHOLDS[@]} ))"
        echo ""

        # Loop through all parameter combinations
        for img_size in "${IMG_SIZES[@]}"; do
            for conf_thres in "${CONF_THRESHOLDS[@]}"; do
                for iou_thres in "${IOU_THRESHOLDS[@]}"; do
                    # Create descriptive experiment name
                    name="exp${exp_counter}_img${img_size}_conf${conf_thres}_iou${iou_thres}"
                    
                    echo "Running experiment $exp_counter: $name"
                    echo "  Image size: ${img_size}"
                    echo "  Confidence threshold: ${conf_thres}"
                    echo "  IoU threshold: ${iou_thres}"
                    echo ""
                    
                    # Run YOLO detection
                    python detect.py \
                        --weights $WEIGHTS \
                        --conf $conf_thres \
                        --iou-thres $iou_thres \
                        --img-size $img_size \
                        --source $SOURCE \
                        --save-txt \
                        --save-conf \
                        --project $PROJECT \
                        --name $name \
                        --classes $CLASSES \
                        --exist-ok
                    
                    # Check if the command was successful
                    if [ $? -eq 0 ]; then
                        echo "✓ Experiment $exp_counter completed successfully"
                    else
                        echo "✗ Experiment $exp_counter failed"
                    fi
                    
                    echo "----------------------------------------"
                    exp_counter=$((exp_counter + 1))
                done
            done
        done

        echo "All experiments completed!"
        echo "Results saved in: $PROJECT/"
        echo ""
        echo "Experiment summary:"
        echo "Total experiments run: $((exp_counter - 1))"
        echo ""
        echo "Parameter combinations tested:"
        echo "  Image sizes: ${IMG_SIZES[*]}"
        echo "  Confidence thresholds: ${CONF_THRESHOLDS[*]}"
        echo "  IoU thresholds: ${IOU_THRESHOLDS[*]}"



###### RUN EVALUATION ######


# evaluate the results and pick best performing case
====================================================================================================
EXPERIMENT RESULTS SUMMARY
====================================================================================================
                    Experiment  mAP@0.5  mAP@0.95  Num Images
 exp10_img960_conf0.15_iou0.35 0.819573  0.268707         294
 exp11_img960_conf0.15_iou0.45 0.821119  0.268707         294
 exp12_img960_conf0.15_iou0.55 0.821119  0.268707         294
exp13_img1280_conf0.15_iou0.45 0.790146  0.674397         294
exp14_img1280_conf0.25_iou0.45 0.788085  0.674397         294
exp15_img1280_conf0.35_iou0.45 0.811173  0.703154         294
exp16_img1280_conf0.25_iou0.35 0.700577  0.591528         294
exp17_img1280_conf0.25_iou0.55 0.788085  0.674397         294
  exp1_img640_conf0.15_iou0.35 0.822975  0.204082         294
  exp2_img640_conf0.15_iou0.45 0.822975  0.204082         294
  exp3_img640_conf0.15_iou0.55 0.823361  0.204082         294
  exp4_img640_conf0.25_iou0.35 0.827304  0.219233         294
  exp5_img640_conf0.25_iou0.45 0.827304  0.219233         294
  exp6_img640_conf0.25_iou0.55 0.827304  0.219233         294
  exp7_img640_conf0.35_iou0.35 0.817718  0.221088         294
  exp8_img640_conf0.35_iou0.45 0.817718  0.221088         294
  exp9_img640_conf0.35_iou0.55 0.817718  0.221088         294


I think that this is the best model 
exp15_img1280_conf0.35_iou0.45
map 0.5 vs 0.95 
0.811173  0.703154

despite that 
exp4_img640_conf0.25_iou0.35
gets this 
 0.827304  0.219233

 its close enough - and results are too wild with 640. So clearly 1280 with 0.35 and 0.45 is best model. 

 we could test 1600 with the beset one. 

but we are not calculating accuracy here - just map - so lets think about that too. 


# let's just run the model now with the best settings we found - on ALL the images (but parallelize it somehow)
# first lets try to run it inside e3 - since we have the gpus ready now... 


# Create index of all images in your 40k+ directory
find pngs_aug2025 -name "*.png" -type f | head -100 | xargs -I {} cp {} pngs_aug2025_100
cd ~/w/code/llm/experiments/yolov7/yolov7
indir=/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/pngs_aug2025_100/
python create_csv_from_folder.py --input-dir $indir --output-csv pngs_aug2025_100.csv

# Process files 

PROJECT=ALL
NAME=RUN1
WEIGHTS="yolov7-p6-bonefracture.pt"
python detect_with_start_number.py \
    --source pngs_aug2025_100.csv \
    --csv \
    --start 40 \
    --max-cases 20 \
    --batch-size 5 \
    --weights $WEIGHTS \
    --conf 0.35 \
    --iou-thres 0.45 \
    --img-size 1280 \
    --save-txt \
    --save-conf \
    --project $PROJECT \
    --name $NAME \
    --classes 3 \
    --exist-ok \
    --input-dir $indir \
    --output-csv output.csv



srun -A crl -p bch-gpu -t 24:00:00 --gres=gpu:1 --cpus-per-task=8 --mem=32G --pty /bin/bash
conda activate llava-med
cd ~/w/code/llm/experiments/yolov7/
source venv/bin/activate 
cd yolov7/

PROJECT=ALL
indir=/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/pngs_aug2025/
python detect.py --weights yolov7-p6-bonefracture.pt \
    --conf 0.35 \
    --iou-thres 0.45 \
    --img-size 1280 \
    --classes 3 \
    --exist-ok --skip-existing \
    --source $indir/ --save-txt --save-conf \
    --project $PROJECT --name run1 


# check if dataloader works 
indir=/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/pngs_aug2025/
python check_dataloader.py \
    --img-size 1280 \
    --source $indir/


########################################################
# INVERSION 
########################################################

# updated png files - where inversion was fixed... 
PROJECT=ALL
indir=/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/pngs_inverted_from_png_csv/
python detect.py --weights yolov7-p6-bonefracture.pt \
    --conf 0.35 \
    --iou-thres 0.45 \
    --img-size 1280 \
    --classes 3 \
    --exist-ok --skip-existing \
    --source $indir/ --save-txt --save-conf \
    --project $PROJECT --name run1_fixed_pngs_only


# next step - cp the labels in run1 into the other pngs 
cd ~/w/code/llm/experiments/yolov7/yolov7/ALL
cp -r run1/labels run1_labels_BAD_NOTINVERTED # these are all the labels 
cp -r run1_fixed_pngs_only/labels/* run1/labels/

# next we need to separate these things... into 1 fracture reports only (and then train yolo on them)
# let's deal with two fracture things later.... 

# check mismatch in the files 
        ls run1_fixed_pngs_only/labels/ | wc -l  
        # 4822
        ls run1_fixed_pngs_only/*png| wc -l 
        # 7359
        # List base filenames (without extension) for both sets
        ls run1_fixed_pngs_only/labels/*.txt | sed 's|labels/||;s|\.txt$||' | sort > txt_basenames.txt
        ls run1_fixed_pngs_only/*.png | sed 's|\.png$||' | sort > png_basenames.txt
        # Find .txt files with no matching .png
        comm -23 txt_basenames.txt png_basenames.txt > unique_txt.txt
        # Find .png files with no matching .txt
        comm -13 txt_basenames.txt png_basenames.txt > unique_png.txt

# now copy files - is a bit complicated... 
cd /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/
python s20250817_fixing_inverted_pngs.py


        # count the files more than 1 day old 
        find . -maxdepth 1 -name "*.png" -type f -mtime +1 | wc -l

        # count files less than 1 day old 
        find . -maxdepth 1 -name "*.png" -type f -mtime -1 | wc -l


        # there are still a lot of elbow files... wtf... let's look at the script that was supposed to remove them... 
        120953

        ls /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/experiments/labeller_running_for_andy/s20250711_generate_structured_reports/outputs/grounded_33k/puyangwang-medgemma-27b-it-q8-reports_aug2025/ | wc -l 
        # 10566

        # the thing is - there should be around 29k images but we have 35k... 
        # some images dont have reports - so we just need to go by reports is ok 
        $ ls *png | wc -l 
        35412

        # ok let's remove that dont have reports 
        d=/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/
        comm -23 \
  <(ls ${d}/pngs_aug2025/ | sort) \
  <(ls ${d}/pngs_aug2025_fixed/ | sort) \
  > ${d}/pngs_with_no_reports.txt

  mkdir -p ~/w/code/llm/experiments/yolov7/yolov7/ALL/run1_no_reports


    # Move each file listed in the txt file
    while read -r fname; do
    mv ~/w/code/llm/experiments/yolov7/yolov7/ALL/run1/"$fname" ~/w/code/llm/experiments/yolov7/yolov7/ALL/run1_no_reports/
    done < /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/pngs_with_no_reports.txt

    # now confirm the sizes 
    ls ~/w/code/llm/experiments/yolov7/yolov7/ALL/run1 | wc -l 
    # 29093
    ls /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/pngs_aug2025/ | wc -l 
    # 35412
    ls /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/pngs_aug2025_fixed/ | wc -l 
    # 29093

    # perfect... 



############################################################
# OLD FIltr fractures - see s20250815_filter_fractures.sh for cleaner file 
############################################################



# now let's fix the inversion issue on the old yolo results, now? 
# also - where are the reports of the old data? 



# now we need to split these b images apart. 
# we do it here 
    cd /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/experiments/labeller_running_for_andy

    #old reports 
    # cannot do this until i fix inversion issues... 
    # report_dir_old=/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/experiments/labeller_running_for_andy/s20250711_generate_structured_reports/outputs/grounded_33k/puyangwang-medgemma-27b-it-q8-reports_aug2025/
    # yolo_dir_old=/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/yolov7/ALL/run1/
    # output_dir=/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/pngs_aug2025/
    # python split_yolo_results_v2.py $yolo_dir --reports_dir $report_dir --skip_existing --output_dir $output_dir #--test # --test runs it on 10 files 


    # new reports 
    report_dir=/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/experiments/labeller_running_for_andy/s20250711_generate_structured_reports/outputs/grounded_33k/puyangwang-medgemma-27b-it-q8-reports_aug2025/
    yolo_dir=/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/yolov7/ALL/run1/
    output_dir=/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/pngs_aug2025/
    mkdir -p $output_dir/
    python split_yolo_results_v2.py $yolo_dir --reports_dir $report_dir --skip_existing --output_dir $output_dir #--test # --test runs it on 10 files 
    


            Files per directory:
            Original labels directory: 43571
            Directory 0: 11247 images, 11247 labels
            Directory 1: 12974 images, 12974 labels
            Directory 2: 4295 images, 4295 labels
            Directory 3: 498 images, 498 labels
            Directory 4: 47 images, 47 labels

            Ratio of images in directory '1' to total images: 0.446 (12974/29061)



    # now let's match reports to yolo... 
    so these are the reports that coincide only to ONE fracture. 
    now i can take these reports and only filter those that have one fracture - that way we would know the number of false negatives and positives 
    (but limited only to one fracture) 
    how many reports are there? 
    12974 images out of 29,000... 
    we can also take images that do not have any fractures... (from reports) and make sure that these images are added to our database... 
    but these images - split_yolo_results_v2 - are based on yolo results (not the actual fractures). 

    I wonder if it is better first to fix them by actual report (e.g. reports that have only 1 fracture). 
    and then from them find images that correspond to these reports... 
    only then trail yolo? oof.. 




        I say we do the following: 
        1. Extract images with 1 fracture for classes that we want
            1. take these files and split into:
                1. 1 fracture 
                2. not 1 fracture 
            2. Take the matching fracture cases and run through visual check -> get accuracy of them matching 
        2. Extract images with 2 fractures 
            1. take these files and split into:
                1. 2 fractures 
                2. not 2 fractures 
            2. For 2 fractures 
                1. generate an image with 1 fracture and ask VLM if it matches A or B (one of the two fractures) 
                2. label the image accordingly (generate a new json with a new tag) 
            3. 
        3. 

    
    # FOLLOWING instructions in /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/experiments/s20250819_how_to_filter_single_case_fractures_by_type.md
    # to generate single fracture cases 
    cd /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/experiments/labeller_running_for_andy
    python filter_fractures.py --json_dir s20250711_generate_structured_reports/outputs/grounded_33k/puyangwang-medgemma-27b-it-q8-reports_aug2025/ --n_fractures 1 --output_csv /lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/pngs_aug2025/fractures_1_v2.csv

    # Step 2: Analyze and Prepare the DataFrame (Optional, for stats)
    python analyze_reports.py --csv_path /lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/pngs_aug2025/fractures_1_v2.csv --json_dir s20250711_generate_structured_reports/outputs/grounded_33k/puyangwang-medgemma-27b-it-q8-reports_aug2025/ --output_path /lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/pngs_aug2025/fractures_1_df.csv --show_stats


        Detailed Statistics:
        --------------------------------------------------
        Total reports analyzed: 3419

        Total fractures found: 3416

        Distribution of fractures per report:
        0 fracture(s): 3 reports
        1 fracture(s): 3416 reports

        Location:

        Bone:
        capitate         1.0
        hamate           3.0
        humerus          1.0
        metacarpal      22.0
        navicular        1.0
        phalange         1.0
        phalanges        2.0
        phalanx          1.0
        pisiform         4.0
        radius        2879.0
        scaphoid       383.0
        trapezium        3.0
        triquetrum      21.0
        ulna            94.0

        # NB that's very bad distirbution of ulna and scaphoid fractures. 

    # I need to update filter_fractures.py script to split my cases into the groups that i defind earlier. 
    # i will use this script that i built 
    cd /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/experiments/labeller_running_for_andy/s20250711_generate_structured_reports/
    python grounding_report_split_regions_s20250821_build_csv.py # fetch all jsons into csv (only read the categories that we want)
    python grounding_report_split_regions_s20250821_analyze.py  # aggregate these csvs into statistics and show results 

    # simple test 
    python grounding_report_split_regions_s20250821_create_filtered_categories_csv.py \
        --input_csv grounded_s20250728_split_regions.csv \
        --output_csv ready_for_symlinks_new_dataset_ALL.csv \
        # --fractures_only \
        # --classification_categories salter_harris_II \
        # --healing_categories acute    

    c=grounding_report_split_regions_s20250821_create_filtered_categories_csv.py
    i=grounded_s20250728_split_regions.csv
    name=<choose_here_appropriate_name>
    python $c --input_csv $i \
        --output_csv ready_for_symlinks_new_dataset_${name}.csv



    # create symlinks 
    # images_dir=/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/pngs/
    # output_dir=/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/pngs_fractures_1_v2/any_fracture
    # python create_symlinks.py \
    #     --csv_file ready_for_symlinks_salter_harris_acute.csv \
    #     --images_dir $images_dir \
    #     --output_dir $output_dir

     

    # use this script to understand how things work 
    # cd /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/experiments/
    # ls 20250116_report_filtering_v2.sh
    #         # this script is less helpful that i made 
    #         /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/experiments/s20250819_how_to_filter_single_case_fractures_by_type.md


    # create symlinks 
    c=/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/experiments/grounding_report_split_regions_s20250821_create_symlinks.py
    csvdir=/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/experiments/labeller_running_for_andy/s20250711_generate_structured_reports/
    python $c --input_csv $csvdir/ready_for_symlinks_new_dataset_ALL.csv \
    --image_dir /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/pngs_aug2025/ \
    --label_dir /lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/yolov7/ALL/run1/labels/ \
    --output_dir /lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/pngs_aug2025/


    # let's look at missing_pngs.csv and wrong_yolo_results.csv
    # <ACCESSION_NUMBER>
    ls /lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/pngs_aug2025/missing_pngs.csv
    ls /lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/pngs_aug2025/wrong_yolo_results.csv

    # ok right now we have: 
    - 11366 - wrong yolo results 
    - 17707 - actual data (but multiple fractures per each case)
    - 231 - missing pngs 

    # now let's filter only those that have 1 fracture or zero fractures... 
        # then train just with those... 
    # then let's setup some sort of fancy script to train those that have 2 fractures 
        # then train with those + the ones that have 1 fracture 
    # finally if there is time we should fix the dicoms on the old cohort (the inversions), then merge them with our files and then train altogether... 

    # ok - let's filter those with one fracture only for now... 










