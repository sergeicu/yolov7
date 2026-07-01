#!/bin/bash

# YOLO Performance Testing and Evaluation Script
# Runs YOLO experiments with different parameters and evaluates mAP

# Configuration
WEIGHTS="yolov7-p6-bonefracture.pt"
SOURCE="/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/testing_andy_labelled_images/images_clean/"
PROJECT="testing_andy_labelled_images"
CLASSES="3"

# Parameter arrays
IMG_SIZES=(640 960 1280 1600)
CONF_THRESHOLDS=(0.15 0.25 0.35)
IOU_THRESHOLDS=(0.35 0.45 0.55)

# Counter for experiment numbering
exp_counter=1

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

echo "All YOLO experiments completed!"
echo ""

# Now evaluate all experiments
echo "Starting evaluation of all experiments..."
echo ""

# Run the evaluation script
python evaluate_yolo_simple.py \
    --gt_dir "testing_andy_labelled_images/andys_labels_final_boxes_all" \
    --experiments_dir "testing_andy_labelled_images" \
    --output_file "evaluation_results.json"

echo ""
echo "Evaluation completed!"
echo "Results saved in: evaluation_results.json" 