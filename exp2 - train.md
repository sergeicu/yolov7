"""
# YOLOv7 Bone Fracture Detection Experiments

## Setup Instructions
```bash
ssh rayan  
conda activate llava-med  
cd ~/w/code/llm/experiments/yolov7/  
source venv/bin/activate   
cd yolov7/  
```


## Key Experiments

### 1. BCH Elbow Dataset Training (Latest Version - v5)
```bash
hyp=data/hyp.scratch.p6_bch_2.yaml
name=yolov7-p6-bonefracture-finetune-bch-elbow-v5
cfg=cfg/training/yolov7-w6_ch9_bonefracture-bch-elbow.yaml
data=data/yolov7-p6-bonefracture-finetune_bch_elbow.yaml
img=640

python train_aux.py --workers 8 --device 0 --batch-size 1 \
    --data $data --img $img $img --cfg $cfg \
    --weights 'yolov7-p6-.pt' --name $name --hyp $hyp
```

- Uses custom hyperparameters
- Image size: 640x640
- Batch size: 1
- No augmentation settings

### 2. Dataset Split Configuration
- Train: 180 images (82%)
- Validation: 25 images (11%)
- Test: 12 images (5%)
- Total: 217 images

### 3. Model Performance Metrics
Key metrics in results.txt:
- Precision
- Recall
- mAP@.5 (IoU threshold 0.5)
- mAP@.5:.95 (IoU thresholds 0.5-0.95)

### 4. Inference Testing
```bash
# Test on BCH elbow dataset
python detect.py --weights runs/train/yolov7-p6-bonefracture-finetune-bch-elbow-v16/weights/best.pt \
    --conf 0.25 --img-size 1280 --source $test_images

# Compare with GRAZPED model
python detect.py --weights yolov7-p6-bonefracture.pt \
    --conf 0.25 --img-size 1280 --source $test_images
```


## Key Findings

1. **Model Configuration**
   - Best results with image size 640x640
   - Small batch size (1) due to limited dataset
   - Custom hyperparameters needed for optimal performance

2. **Augmentation Impact**
   - Heavy augmentation (v2) showed poor results
   - No augmentation (v3) performed better
   - Custom minimal augmentation (v5) showed best results

3. **Dataset Considerations**
   - Small dataset (217 images) requires careful training approach
   - Separate train/val/test split important for reliable evaluation
   - Consider freezing layers due to limited data

## Recommendations

1. Use minimal augmentation with small dataset
2. Keep batch size small (1-4)
3. Consider image size 640x640 for training
4. Monitor validation metrics closely for overfitting
5. Consider transfer learning approaches with frozen layers

## Future Work

1. Experiment with layer freezing
2. Test different image resizing strategies
3. Collect additional training data
4. Evaluate model on diverse test cases
5. Compare performance with other architectures
