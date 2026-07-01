#!/usr/bin/env python3
"""
YOLO Experiment Evaluation Script
Calculates mAP (mean Average Precision) for YOLO experiments by comparing
predicted bounding boxes with ground truth annotations.
"""

import os
import glob
import numpy as np
from pathlib import Path
import argparse
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def parse_yolo_txt(file_path):
    """
    Parse YOLO format txt file (class x_center y_center width height confidence)
    Returns: list of [class_id, x_center, y_center, width, height, confidence]
    """
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
    """
    Convert YOLO format (x_center, y_center, width, height) to (x1, y1, x2, y2)
    """
    class_id, x_center, y_center, width, height, confidence = box
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    return [class_id, x1, y1, x2, y2, confidence]

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union between two boxes
    box format: [x1, y1, x2, y2]
    """
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = box1[1:5]
    x1_2, y1_2, x2_2, y2_2 = box2[1:5]
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def calculate_ap(gt_boxes, pred_boxes, iou_threshold=0.5):
    """
    Calculate Average Precision for a single image
    """
    if not gt_boxes and not pred_boxes:
        return 1.0  # Both empty, perfect match
    if not gt_boxes:
        return 0.0  # No ground truth, all predictions are false positives
    if not pred_boxes:
        return 0.0  # No predictions, all ground truths are false negatives
    
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
    """
    Evaluate a single experiment
    """
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
    """
    Find all experiment directories matching the pattern exp*_img*_conf*_iou*
    """
    pattern = os.path.join(base_dir, "exp*_img*_conf*_iou*")
    return glob.glob(pattern)

def parse_experiment_name(exp_dir):
    """
    Parse experiment directory name to extract parameters
    """
    exp_name = os.path.basename(exp_dir)
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
    parser = argparse.ArgumentParser(description='Evaluate YOLO experiments')
    parser.add_argument('--gt_dir', type=str, 
                       default='testing_andy_labelled_images/andys_labels_final_boxes_all',
                       help='Ground truth directory')
    parser.add_argument('--experiments_dir', type=str, 
                       default='testing_andy_labelled_images',
                       help='Directory containing experiment results')
    parser.add_argument('--output_file', type=str, 
                       default='evaluation_results.json',
                       help='Output file for results')
    parser.add_argument('--plot_results', action='store_true',
                       help='Generate plots of results')
    
    args = parser.parse_args()
    
    # Find all experiment directories
    exp_dirs = find_experiment_dirs(args.experiments_dir)
    
    if not exp_dirs:
        print("No experiment directories found!")
        return
    
    print(f"Found {len(exp_dirs)} experiment directories")
    
    # Evaluate each experiment
    all_results = {}
    
    for exp_dir in sorted(exp_dirs):
        exp_name = os.path.basename(exp_dir)
        print(f"Evaluating {exp_name}...")
        
        # Parse experiment parameters
        params = parse_experiment_name(exp_dir)
        
        # Evaluate the experiment
        results = evaluate_experiment(args.gt_dir, exp_dir)
        
        # Combine parameters and results
        if params:
            all_results[exp_name] = {**params, **results}
        else:
            all_results[exp_name] = results
        
        print(f"  mAP@0.5: {results.get('mAP@0.5', 0):.4f}")
        print(f"  mAP@0.95: {results.get('mAP@0.95', 0):.4f}")
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {args.output_file}")
    
    # Generate summary table
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    print(f"{'Experiment':<30} {'Img Size':<8} {'Conf':<6} {'IoU':<6} {'mAP@0.5':<8} {'mAP@0.95':<8}")
    print("-"*80)
    
    for exp_name, results in sorted(all_results.items()):
        if 'img_size' in results:
            print(f"{exp_name:<30} {results['img_size']:<8} {results['conf_thresh']:<6.2f} "
                  f"{results['iou_thresh']:<6.2f} {results.get('mAP@0.5', 0):<8.4f} "
                  f"{results.get('mAP@0.95', 0):<8.4f}")
    
    # Generate plots if requested
    if args.plot_results:
        generate_plots(all_results)

def generate_plots(results):
    """
    Generate visualization plots for the results
    """
    # Prepare data for plotting
    data = []
    for exp_name, exp_results in results.items():
        if 'img_size' in exp_results:
            data.append({
                'Experiment': exp_name,
                'Image Size': exp_results['img_size'],
                'Confidence': exp_results['conf_thresh'],
                'IoU': exp_results['iou_thresh'],
                'mAP@0.5': exp_results.get('mAP@0.5', 0),
                'mAP@0.95': exp_results.get('mAP@0.95', 0)
            })
    
    if not data:
        print("No valid data for plotting")
        return
    
    df = pd.DataFrame(data)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: mAP@0.5 vs Image Size
    sns.boxplot(data=df, x='Image Size', y='mAP@0.5', ax=axes[0,0])
    axes[0,0].set_title('mAP@0.5 vs Image Size')
    
    # Plot 2: mAP@0.95 vs Image Size
    sns.boxplot(data=df, x='Image Size', y='mAP@0.95', ax=axes[0,1])
    axes[0,1].set_title('mAP@0.95 vs Image Size')
    
    # Plot 3: mAP@0.5 vs Confidence Threshold
    sns.boxplot(data=df, x='Confidence', y='mAP@0.5', ax=axes[1,0])
    axes[1,0].set_title('mAP@0.5 vs Confidence Threshold')
    
    # Plot 4: mAP@0.95 vs IoU Threshold
    sns.boxplot(data=df, x='IoU', y='mAP@0.95', ax=axes[1,1])
    axes[1,1].set_title('mAP@0.95 vs IoU Threshold')
    
    plt.tight_layout()
    plt.savefig('evaluation_plots.png', dpi=300, bbox_inches='tight')
    print("Plots saved to evaluation_plots.png")

if __name__ == "__main__":
    main() 