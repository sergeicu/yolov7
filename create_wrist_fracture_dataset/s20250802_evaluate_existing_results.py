#!/usr/bin/env python3
"""
Simple script to evaluate existing YOLO experiment results
Compares ground truth annotations with predicted bounding boxes
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

def parse_yolo_txt(file_path):
    """Parse YOLO format txt file (class x_center y_center width height confidence)"""
    boxes = []
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        confidence = float(parts[5]) if len(parts) > 5 else 1.0
                        boxes.append([class_id, x_center, y_center, width, height, confidence])
    return boxes

def convert_yolo_to_xyxy(box):
    """Convert YOLO format to (x1, y1, x2, y2)"""
    # Handle cases where confidence might be missing
    if len(box) == 5:
        # No confidence provided, use dummy value
        class_id, x_center, y_center, width, height = box
        confidence = 1.0  # Dummy confidence value
    else:
        # Confidence is provided
        class_id, x_center, y_center, width, height, confidence = box
    
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    return [class_id, x1, y1, x2, y2, confidence]

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1[1:5]
    x1_2, y1_2, x2_2, y2_2 = box2[1:5]
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def calculate_ap(gt_boxes, pred_boxes, iou_threshold=0.5):
    """Calculate Average Precision for a single image"""
    if not gt_boxes and not pred_boxes:
        return 1.0
    if not gt_boxes:
        return 0.0
    if not pred_boxes:
        return 0.0
    
    # Convert to xyxy format
    gt_xyxy = [convert_yolo_to_xyxy(box) for box in gt_boxes]
    pred_xyxy = [convert_yolo_to_xyxy(box) for box in pred_boxes]
    
    # Sort predictions by confidence (descending)
    pred_xyxy.sort(key=lambda x: x[5], reverse=True)
    
    # Initialize
    tp = np.zeros(len(pred_xyxy))
    fp = np.zeros(len(pred_xyxy))
    gt_matched = [False] * len(gt_xyxy)
    
    # For each prediction
    for pred_idx, pred_box in enumerate(pred_xyxy):
        best_iou = 0
        best_gt_idx = -1
        
        # Find best matching ground truth
        for gt_idx, gt_box in enumerate(gt_xyxy):
            if not gt_matched[gt_idx] and pred_box[0] == gt_box[0]:  # Same class
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
        
        # Determine if prediction is true positive
        if best_iou >= iou_threshold:
            tp[pred_idx] = 1
            gt_matched[best_gt_idx] = True
        else:
            fp[pred_idx] = 1
    
    # Calculate precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / len(gt_xyxy)
    
    # Add sentinel values
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    
    # Calculate AP using 11-point interpolation
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap = ap + p / 11.0
    
    return ap

def evaluate_experiment(gt_dir, pred_dir, iou_thresholds=[0.5, 0.95]):
    """Evaluate a single experiment"""
    results = {}
    
    for iou_thresh in iou_thresholds:
        aps = []
        
        # Get all ground truth files
        gt_files = glob.glob(os.path.join(gt_dir, "*.txt"))
        
        for gt_file in gt_files:
            filename = os.path.basename(gt_file)
            pred_file = os.path.join(pred_dir, "labels", filename)
            
            # Parse ground truth and predictions
            gt_boxes = parse_yolo_txt(gt_file)
            pred_boxes = parse_yolo_txt(pred_file)
            
            # Calculate AP for this image
            ap = calculate_ap(gt_boxes, pred_boxes, iou_thresh)
            aps.append(ap)
        
        # Calculate mAP
        map_score = np.mean(aps) if aps else 0.0
        results[f'mAP@{iou_thresh}'] = map_score
        results[f'AP@{iou_thresh}_std'] = np.std(aps) if aps else 0.0
        results[f'num_images'] = len(aps)
    
    return results

def find_experiment_dirs(base_dir):
    """Find all experiment directories"""
    # Look for common YOLO experiment patterns
    patterns = [
        os.path.join(base_dir, "exp*_img*_conf*_iou*"),
        # os.path.join(base_dir, "*_img*_conf*_iou*"),
        # os.path.join(base_dir, "run*"),
        # os.path.join(base_dir, "test*")
    ]
    
    exp_dirs = []
    for pattern in patterns:
        exp_dirs.extend(glob.glob(pattern))
    
    # Filter to only directories that contain labels subdirectory
    valid_dirs = []
    for exp_dir in exp_dirs:
        if os.path.isdir(exp_dir):
            labels_dir = os.path.join(exp_dir, "labels")
            if os.path.exists(labels_dir) and os.path.isdir(labels_dir):
                valid_dirs.append(exp_dir)
    
    return valid_dirs

def parse_experiment_name(exp_dir):
    """Parse experiment directory name to extract parameters"""
    exp_name = os.path.basename(exp_dir)
    
    # Try to parse different naming patterns
    if exp_name.startswith('exp'):
        parts = exp_name.split('_')
        if len(parts) >= 6:
            try:
                img_size = int(parts[2].replace('img', ''))
                conf_thresh = float(parts[3].replace('conf', ''))
                iou_thresh = float(parts[4].replace('iou', ''))
                return {
                    'img_size': img_size,
                    'conf_thresh': conf_thresh,
                    'iou_thresh': iou_thresh
                }
            except (ValueError, IndexError):
                pass
    
    return None

def main():
    # Configuration
    gt_dir = "testing_andy_labelled_images/andys_labels_final_boxes_all"
    experiments_dir = "testing_andy_labelled_images"
    
    print("Finding experiment directories...")
    exp_dirs = find_experiment_dirs(experiments_dir)
    
    if not exp_dirs:
        print("No experiment directories found!")
        print("Looking for directories with 'labels' subdirectory containing .txt files")
        return
    
    print(f"Found {len(exp_dirs)} experiment directories:")
    for exp_dir in exp_dirs:
        print(f"  - {os.path.basename(exp_dir)}")
    
    # Evaluate each experiment
    all_results = {}
    
    for exp_dir in sorted(exp_dirs):
        exp_name = os.path.basename(exp_dir)
        print(f"\nEvaluating {exp_name}...")
        
        # Parse experiment parameters
        params = parse_experiment_name(exp_dir)
        
        # Evaluate the experiment
        results = evaluate_experiment(gt_dir, exp_dir)
        
        # Combine parameters and results
        if params:
            all_results[exp_name] = {**params, **results}
        else:
            all_results[exp_name] = results
        
        print(f"  mAP@0.5: {results.get('mAP@0.5', 0):.4f}")
        print(f"  mAP@0.95: {results.get('mAP@0.95', 0):.4f}")
    
    # Save results
    with open('evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to evaluation_results.json")
    
    # Create summary table
    print("\n" + "="*100)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*100)
    
    # Create DataFrame for easy analysis
    data = []
    for exp_name, results in all_results.items():
        row = {'Experiment': exp_name}
        if 'img_size' in results:
            row.update({
                'Img Size': results['img_size'],
                'Conf Thresh': results['conf_thresh'],
                'IoU Thresh': results['iou_thresh']
            })
        row.update({
            'mAP@0.5': results.get('mAP@0.5', 0),
            'mAP@0.95': results.get('mAP@0.95', 0),
            'Num Images': results.get('num_images', 0)
        })
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Print table
    if 'Img Size' in df.columns:
        print(f"{'Experiment':<25} {'Img Size':<8} {'Conf':<6} {'IoU':<6} {'mAP@0.5':<8} {'mAP@0.95':<8}")
        print("-"*100)
        for _, row in df.iterrows():
            print(f"{row['Experiment']:<25} {row['Img Size']:<8} {row['Conf Thresh']:<6.2f} "
                  f"{row['IoU Thresh']:<6.2f} {row['mAP@0.5']:<8.4f} {row['mAP@0.95']:<8.4f}")
    else:
        print(df.to_string(index=False))
    
    # Find best performing experiments
    print("\n" + "="*100)
    print("BEST PERFORMING EXPERIMENTS")
    print("="*100)
    
    if len(df) > 0:
        # Best mAP@0.5
        best_map50 = df.loc[df['mAP@0.5'].idxmax()]
        print(f"Best mAP@0.5: {best_map50['Experiment']} = {best_map50['mAP@0.5']:.4f}")
        
        # Best mAP@0.95
        best_map95 = df.loc[df['mAP@0.95'].idxmax()]
        print(f"Best mAP@0.95: {best_map95['Experiment']} = {best_map95['mAP@0.95']:.4f}")
    
    # Create plots
    if len(df) > 0 and 'Img Size' in df.columns:
        print("\nCreating plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: mAP@0.5 vs Image Size
        if len(df['Img Size'].unique()) > 1:
            df.boxplot(column='mAP@0.5', by='Img Size', ax=axes[0,0])
            axes[0,0].set_title('mAP@0.5 vs Image Size')
            axes[0,0].set_xlabel('Image Size')
        
        # Plot 2: mAP@0.95 vs Image Size
        if len(df['Img Size'].unique()) > 1:
            df.boxplot(column='mAP@0.95', by='Img Size', ax=axes[0,1])
            axes[0,1].set_title('mAP@0.95 vs Image Size')
            axes[0,1].set_xlabel('Image Size')
        
        # Plot 3: mAP@0.5 vs Confidence Threshold
        if len(df['Conf Thresh'].unique()) > 1:
            df.boxplot(column='mAP@0.5', by='Conf Thresh', ax=axes[1,0])
            axes[1,0].set_title('mAP@0.5 vs Confidence Threshold')
            axes[1,0].set_xlabel('Confidence Threshold')
        
        # Plot 4: mAP@0.95 vs IoU Threshold
        if len(df['IoU Thresh'].unique()) > 1:
            df.boxplot(column='mAP@0.95', by='IoU Thresh', ax=axes[1,1])
            axes[1,1].set_title('mAP@0.95 vs IoU Threshold')
            axes[1,1].set_xlabel('IoU Threshold')
        
        plt.tight_layout()
        plt.savefig('evaluation_plots.png', dpi=300, bbox_inches='tight')
        print("Plots saved to evaluation_plots.png")
    
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main() 