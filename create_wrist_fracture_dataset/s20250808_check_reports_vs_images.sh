# Check both reports and images
python s20250808_check_reports_vs_images.py \
    --images /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/pngs \
    --reports /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/experiments/labeller_running_for_andy/s20250711_generate_structured_reports/outputs/grounded/puyangwang-medgemma-27b-it-q8-dataset




$ python s20250808_check_reports_vs_images.py \
    --images /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/pngs \
    --reports /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/experiments/labeller_running_for_andy/s20250711_generate_structured_reports/outputs/grounded/puyangwang-medgemma-27b-it-q8-dataset 
Scanning reports directory: /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/experiments/labeller_running_for_andy/s20250711_generate_structured_reports/outputs/grounded/puyangwang-medgemma-27b-it-q8-dataset
Found 8040 reports
Scanning images directory: /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/pngs
Found 13858 unique report IDs from images

=== Checking if reports have corresponding images ===
✅ All reports have corresponding images

=== Checking if images have corresponding reports ===
❌ Found 5818 image report IDs without corresponding reports:
  - 100012341 (from images)
  - 100012430 (from images)
  - 100013390 (from images)
  - 100023420 (from images)
  - 100035450 (from images)
  - 100036834 (from images)
  - 100036995 (from images)
  - 100057293 (from images)
  - 100058833 (from images)
  - 100059940 (from images)
  ... and 5808 more

=== Summary ===
Total reports: 8040
Total unique report IDs from images: 13858
Reports without images: 0
Images without reports: 5818
Reports with matching images: 8040

Example matching report IDs:
  - 100005253
  - 100008228
  - 100008230
  - 100008262
  - 100008271


python s20250808_check_reports_vs_images.py \
    --images /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/results/v2/run1 \
    --reports /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/experiments/labeller_running_for_andy/s20250711_generate_structured_reports/outputs/grounded/puyangwang-medgemma-27b-it-q8-dataset 


=== Summary ===
Total reports: 8040
Total unique report IDs from images: 10221
Reports without images: 0
Images without reports: 2181
Reports with matching images: 8040




python s20250808_check_reports_vs_images_v1.py \
    --images /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/results/v2/run1 \
    --reports /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/reports


=== Summary ===
Total reports: 13879
Total unique report IDs from images: 10221
Reports without images: 3658
Images without reports: 0
Reports with matching images: 10221




python s20250808_check_reports_vs_images_v1.py \
    --images /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/results/v2/run1 \
    --reports /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/experiments/labeller_running_for_andy/outputs/grounded/

=== Summary ===
Total reports: 2941
Total unique report IDs from images: 10221
Reports without images: 2
Images without reports: 7282
Reports with matching images: 2939


CONCLUSION: 

old
  we have 2941 reports 
  matching images from these extracted reports - 10221 (derived by yolo)
  total images we have - 13879